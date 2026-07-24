// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Implementation of the HSA dispatch runtime (see HsaRuntime.h).
//
// Compiled once into libtriton_npu_hsa.so and linked by every generated
// launcher, so the HsaRuntime singleton is process-global: one hsa_init, queue,
// completion signal, kernarg pool, and vmem buffer pool shared across all
// kernel signatures. Dispatches are serialized on a single queue via a dispatch
// mutex.
//
// Memory strategy:
// * PDI + instructions: plain HSA pool allocation from the dev pool
//   (coarse-grained, non-allocatable), loaded once per (pdi, insts) and cached.
// * Tensor I/O: the vmem API (handle_create -> reserve -> map -> set_access),
//   RW-accessible to CPU and AIE agents, pooled and reused across dispatches.
// * Kernel arguments: a fixed-slot pool (one slot per ring slot); slot(i) is
//   pure pointer arithmetic, no HSA call on the hot path.

#include "HsaRuntime/HsaRuntime.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hsa/hsa_ext_amd_aie.h"

namespace {

// Queue depth cap. Also bounds the kernarg slot pool (one slot per ring slot).
constexpr std::uint32_t QUEUE_SIZE = 32;

// Bytes in the AIE dispatch packet after completion_signal up to
// kernarg_address; the ABI requires this to be exactly 24.
constexpr std::uint16_t AIE_PACKET_COUNT = 24;

// Throw std::runtime_error with the HSA status string on a non-success status.
#define HSA_CHECK(expr)                                                        \
  do {                                                                         \
    hsa_status_t _s = (expr);                                                  \
    if (_s != HSA_STATUS_SUCCESS) {                                            \
      const char *_m = nullptr;                                                \
      hsa_status_string(_s, &_m);                                              \
      throw std::runtime_error(std::string(#expr) +                            \
                               " failed: " + (_m ? _m : "unknown HSA error")); \
    }                                                                          \
  } while (0)

// ---- Agent discovery -------------------------------------------------------
// Search state for collect_agents: the device type to match, and the vector
// that matching agents are appended to.
struct AgentSearch {
  hsa_device_type_t want{};
  std::vector<hsa_agent_t> *out{};
};

// hsa_iterate_agents callback: append `agent` to AgentSearch::out when its
// device type equals AgentSearch::want (`data` is the AgentSearch).
hsa_status_t collect_agents(hsa_agent_t agent, void *data) {
  auto *s = static_cast<AgentSearch *>(data);
  hsa_device_type_t t{};
  hsa_status_t st = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &t);
  if (st != HSA_STATUS_SUCCESS)
    return st;
  if (t == s->want)
    s->out->push_back(agent);
  return HSA_STATUS_SUCCESS;
}

// ---- Memory pool discovery -------------------------------------------------
// Search state for find_pool: the global-flag mask and allocatability to match,
// plus the first matching pool (with `found` set true on a hit).
struct PoolSearch {
  hsa_amd_memory_pool_global_flag_t flags{};
  bool allocatable{false};
  hsa_amd_memory_pool_t pool{};
  bool found{false};
};

// hsa_amd_agent_iterate_memory_pools callback: select the first GLOBAL pool
// whose flags include PoolSearch::flags and whose allocatability matches, then
// stop the iteration with HSA_STATUS_INFO_BREAK (`data` is the PoolSearch).
hsa_status_t find_pool(hsa_amd_memory_pool_t pool, void *data) {
  auto *d = static_cast<PoolSearch *>(data);
  hsa_amd_segment_t seg{};
  if (hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                   &seg) != HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;
  if (seg != HSA_AMD_SEGMENT_GLOBAL)
    return HSA_STATUS_SUCCESS;

  hsa_amd_memory_pool_global_flag_t f{};
  if (hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                                   &f) != HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;
  if ((f & d->flags) == 0) return HSA_STATUS_SUCCESS;

  std::size_t granule = 0;
  if (hsa_amd_memory_pool_get_info(
          pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE, &granule) !=
      HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;
  bool allocatable = (granule != 0);
  if (allocatable != d->allocatable) return HSA_STATUS_SUCCESS;

  d->pool = pool;
  d->found = true;
  return HSA_STATUS_INFO_BREAK;
}

// ---- device-memory buffer --------------------------------------------------
// A chunk of device memory (address `va` + `size`). Two flavors share this
// type, distinguished by whether `handle` is set:
//   * vmem buffers (tensor I/O): allocated through the vmem API; `handle` is a
//     valid vmem handle and `va` is the mapped virtual address. Freed with
//     unmap / address_free / handle_release.
//   * plain pool allocations (PDI + instructions): allocated with
//     hsa_amd_memory_pool_allocate from the dev pool; `handle` is zero and `va`
//     is the pool pointer. Freed with hsa_amd_memory_pool_free. (PDI/insts must
//     live in the dev heap, which is incompatible with the vmem reserve+map
//     path, so they cannot use the vmem flavor.)
struct DeviceBuffer {
  hsa_amd_vmem_alloc_handle_t handle{};
  void *va{};
  std::size_t size{};
};

}  // namespace

// A prepared program is just its two device buffers; the opaque handle in the
// C ABI points at one of these, owned by the runtime's program cache.
struct triton_npu_hsa_program {
  DeviceBuffer pdi;
  DeviceBuffer insts;
};

namespace {

// ---- Process-global HSA runtime -------------------------------------------
// Owns every persistent HSA resource: the AIE agent, the dev/data/kernarg
// memory pools, the single command queue and completion signal, the fixed-slot
// kernarg pool, the pooled vmem I/O buffers, and the cache of prepared programs.
// One instance per process (see runtime()).
class HsaRuntime {
 public:
  // Construct by running the full one-time HSA initialization.
  HsaRuntime() { init(); }

  // Best-effort teardown of all owned HSA resources at process exit.
  // No drain wait is needed: dispatch() is synchronous (it waits for each
  // packet to complete before returning), so nothing is ever in flight here.
  // (An unconditional wait-for-zero would also hang if a prior dispatch left
  // the signal at a negative device-error code.)
  ~HsaRuntime() {
    for (auto &kv : vmem_pool_) {
      for (auto &b : kv.second) {
        if (!b.va) continue;
        hsa_amd_vmem_unmap(b.va, b.size);
        hsa_amd_vmem_address_free(b.va, b.size);
        hsa_amd_vmem_handle_release(b.handle);
      }
    }
    for (auto &kv : programs_) {
      if (kv.second->pdi.va) hsa_amd_memory_pool_free(kv.second->pdi.va);
      if (kv.second->insts.va) hsa_amd_memory_pool_free(kv.second->insts.va);
    }
    if (kernarg_buffer_) hsa_amd_memory_pool_free(kernarg_buffer_);
    if (signal_.handle) hsa_signal_destroy(signal_);
    if (queue_) hsa_queue_destroy(queue_);
    hsa_shut_down();
  }

  // Load + cache the PDI/insts for a kernel and return its program handle. Keyed
  // by (pdi_path, insts_path) so repeated prepares (or a PDI shared by two
  // signatures) reuse the same device allocation. Thread-safe.
  triton_npu_hsa_program *prepare(const char *pdi_path, const char *insts_path) {
    std::string key = std::string(pdi_path) + '\0' + insts_path;
    std::lock_guard<std::mutex> lock(programs_mtx_);
    auto it = programs_.find(key);
    if (it != programs_.end()) return it->second.get();
    auto prog = std::make_unique<triton_npu_hsa_program>();
    prog->pdi = load_binary(pdi_path);
    prog->insts = load_binary(insts_path);
    triton_npu_hsa_program *raw = prog.get();
    programs_.emplace(std::move(key), std::move(prog));
    return raw;
  }

  // Run one dispatch of `program` over num_tensors (host_ptr, size) pairs.
  // Serialized against every other dispatch (single shared queue).
  void dispatch(triton_npu_hsa_program *program, std::uint32_t num_tensors,
                void *const *host_ptrs, const std::uint64_t *sizes) {
    if (program == nullptr)
      throw std::runtime_error(
          "dispatch called with a null program handle (was set_paths / prepare "
          "successful?)");
    if (num_tensors > TRITON_NPU_HSA_MAX_KERNARGS)
      throw std::runtime_error(
          "kernel has " + std::to_string(num_tensors) +
          " tensor arguments but the HSA runtime supports at most " +
          std::to_string(TRITON_NPU_HSA_MAX_KERNARGS) +
          "; raise TRITON_NPU_HSA_MAX_KERNARGS in HsaRuntime.h and rebuild");

    std::lock_guard<std::mutex> dlock(dispatch_mtx_);

    std::array<DeviceBuffer, TRITON_NPU_HSA_MAX_KERNARGS> bufs{};
    std::uint32_t acquired = 0;
    try {
      // Acquire an I/O buffer per tensor (from the pool) and copy inputs in.
      for (std::uint32_t i = 0; i < num_tensors; ++i) {
        bufs[i] = acquire((std::size_t)sizes[i]);
        acquired = i + 1;
        std::memcpy(bufs[i].va, host_ptrs[i], (std::size_t)sizes[i]);
      }

      // Reserve a ring slot (queue is single-producer under this lock).
      hsa_queue_t *q = queue_;
      const std::uint64_t wr_idx = hsa_queue_add_write_index_relaxed(q, 1);
      while (wr_idx - hsa_queue_load_read_index_scacquire(q) >= q->size) {
        // Wait for the device to consume a packet. Never reached under the
        // current synchronous model (queue is never full), but yield rather
        // than hot-spin should batched/async submission ever be added.
        std::this_thread::yield();
      }
      const std::uint64_t pkt_idx = wr_idx % q->size;

      // Each ring slot owns a fixed kernarg slot of the same index; no
      // allocation on the hot path. Layout: [addr0..addrN-1, size0..sizeN-1].
      auto *kernargs = static_cast<std::uint64_t *>(kernarg_slot((std::uint32_t)pkt_idx));
      for (std::uint32_t i = 0; i < num_tensors; ++i) {
        kernargs[i] = reinterpret_cast<std::uint64_t>(bufs[i].va);
        kernargs[num_tensors + i] = sizes[i];
      }

      // Build the AIE dispatch packet.
      hsa_amd_aie_kernel_dispatch_packet_t pkt{};
      pkt.header =
          (HSA_AMD_AIE_PACKET_TYPE_READY << HSA_PACKET_HEADER_TYPE) |
          (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
          (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
      pkt.opcode = HSA_AMD_AIE_PACKET_OPCODE_KMQ;
      pkt.count = AIE_PACKET_COUNT;
      pkt.completion_signal = signal_;
      pkt.insts_addr_low =
          reinterpret_cast<std::uintptr_t>(program->insts.va) & 0xFFFFFFFF;
      pkt.insts_addr_high =
          reinterpret_cast<std::uintptr_t>(program->insts.va) >> 32;
      pkt.num_kernargs = num_tensors;
      // The ABI requires kernarg_address to be NULL when num_kernargs is 0.
      pkt.kernarg_address = (num_tensors > 0) ? kernargs : nullptr;
      pkt.insts_size = program->insts.size;
      pkt.pdi_addr = program->pdi.va;

      // Arm the signal to 1 (device decrements to 0 on success, or sets a
      // negative error code), publish the packet, then ring the doorbell.
      hsa_signal_store_screlease(signal_, 1);
      static_cast<hsa_amd_aie_kernel_dispatch_packet_t *>(q->base_address)[pkt_idx] =
          pkt;
      hsa_signal_store_screlease(q->doorbell_signal, wr_idx);

      // Wait for completion. Wait for value < 1 (rather than == 0) so a device
      // error (negative code) is observed instead of hanging; loop to ignore
      // spurious wake-ups that still see the armed value. The caller releases
      // the GIL around this whole dispatch.
      hsa_signal_value_t sig_val = 1;
      do {
        sig_val = hsa_signal_wait_scacquire(signal_, HSA_SIGNAL_CONDITION_LT, 1,
                                            UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
      } while (sig_val == 1);
      if (sig_val != 0)
        throw std::runtime_error("AIE dispatch failed: completion signal = " +
                                 std::to_string((long long)sig_val));

      // Copy every tensor buffer back to its host pointer. We cannot know which
      // argument(s) the kernel writes, so copying all back is correct regardless
      // of output position (unmodified inputs just copy identical bytes).
      for (std::uint32_t i = 0; i < num_tensors; ++i)
        std::memcpy(host_ptrs[i], bufs[i].va, (std::size_t)sizes[i]);

      for (std::uint32_t i = 0; i < num_tensors; ++i) release(bufs[i]);
    } catch (...) {
      // Return any buffers acquired before the failure so they are not leaked.
      for (std::uint32_t i = 0; i < acquired; ++i) release(bufs[i]);
      throw;
    }
  }

 private:
  hsa_agent_t aie_agent_{};
  hsa_amd_memory_pool_t dev_pool_{};
  hsa_amd_memory_pool_t data_pool_{};
  hsa_queue_t *queue_ = nullptr;
  hsa_signal_t signal_{};
  std::vector<hsa_amd_memory_access_desc_t> access_descs_;  // RW, built once
  std::size_t data_granule_ = 0;

  // Fixed-slot kernarg pool: one backing allocation, one slot per ring slot.
  // (The pool it came from and the slot count are only needed during init.)
  void *kernarg_buffer_ = nullptr;
  std::size_t kernarg_slot_size_ = 0;

  // Free-list of vmem data buffers keyed by rounded size, reused across
  // dispatches rather than freed (a vmem region unmapped after an AIE dispatch
  // cannot be reliably re-reserved+mapped, and reuse removes per-launch cost).
  std::unordered_map<std::size_t, std::vector<DeviceBuffer>> vmem_pool_;

  // Prepared programs, keyed by "pdi\0insts"; owns the loaded PDI/insts and
  // makes prepare() idempotent (a repeated (pdi, insts) reuses one dev
  // allocation rather than loading/leaking a second copy).
  std::map<std::string, std::unique_ptr<triton_npu_hsa_program>> programs_;
  std::mutex programs_mtx_;

  // Serializes dispatches (one shared queue, one packet in flight).
  std::mutex dispatch_mtx_;

  // One-time setup: init HSA, discover the AIE + CPU agents and the
  // dev/data/kernarg pools, build the vmem access-descriptor list, create the
  // queue and completion signal, and allocate the kernarg slot pool.
  void init() {
    HSA_CHECK(hsa_init());

    std::vector<hsa_agent_t> aies, cpus;
    AgentSearch as_aie{HSA_DEVICE_TYPE_AIE, &aies};
    HSA_CHECK(hsa_iterate_agents(collect_agents, &as_aie));
    AgentSearch as_cpu{HSA_DEVICE_TYPE_CPU, &cpus};
    HSA_CHECK(hsa_iterate_agents(collect_agents, &as_cpu));
    if (aies.empty())
      throw std::runtime_error(
          "no HSA AIE agent found (is the NPU driver loaded?)");
    aie_agent_ = aies.front();

    // Every vmem I/O buffer must be RW-accessible to the CPU (host memcpy) and
    // the AIE agent (execution). Build that descriptor list once, here.
    std::vector<hsa_agent_t> access_agents;
    for (auto c : cpus) access_agents.push_back(c);
    for (auto a : aies) access_agents.push_back(a);
    access_descs_.reserve(access_agents.size());
    for (auto a : access_agents)
      access_descs_.push_back({HSA_ACCESS_PERMISSION_RW, a});

    // dev pool: coarse-grained, non-allocatable (PDI + instructions).
    if (!discover_pool(HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED, false,
                       &dev_pool_))
      throw std::runtime_error("no dev memory pool on AIE agent");
    // data pool: coarse-grained, allocatable (tensor data via vmem).
    if (!discover_pool(HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED, true,
                       &data_pool_))
      throw std::runtime_error("no data memory pool on AIE agent");
    // kernarg pool: KERNARG_INIT allocatable; fall back to the data pool.
    hsa_amd_memory_pool_t kernarg_pool{};
    if (!discover_pool(HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT, true,
                       &kernarg_pool))
      kernarg_pool = data_pool_;

    HSA_CHECK(hsa_amd_memory_pool_get_info(
        data_pool_, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
        &data_granule_));
    if (data_granule_ == 0) data_granule_ = 4096;  // sane fallback for round_up

    // Queue, capped at QUEUE_SIZE but clamped into the agent's supported range
    // (the AIE agent requires an exact size, e.g. 64, so clamp up to min_q).
    std::uint32_t min_q = 0, max_q = 0;
    HSA_CHECK(
        hsa_agent_get_info(aie_agent_, HSA_AGENT_INFO_QUEUE_MIN_SIZE, &min_q));
    HSA_CHECK(
        hsa_agent_get_info(aie_agent_, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &max_q));
    std::uint32_t qsize = QUEUE_SIZE;
    if (qsize < min_q) qsize = min_q;
    if (max_q > 0 && qsize > max_q) qsize = max_q;
    HSA_CHECK(hsa_queue_create(aie_agent_, qsize, HSA_QUEUE_TYPE_SINGLE, nullptr,
                               nullptr, 0, 0, &queue_));

    // Persistent completion signal, armed per dispatch.
    HSA_CHECK(hsa_signal_create(0, 0, nullptr, &signal_));

    // Kernarg slot pool: one slot per ring slot, sized for the worst case
    // (MAX_KERNARGS addresses + MAX_KERNARGS sizes), aligned to 64 bytes.
    std::uint32_t kernarg_slot_count = static_cast<std::uint32_t>(queue_->size);
    std::size_t raw = static_cast<std::size_t>(TRITON_NPU_HSA_MAX_KERNARGS) * 2 *
                      sizeof(std::uint64_t);
    kernarg_slot_size_ = (raw + 63u) & ~static_cast<std::size_t>(63u);
    HSA_CHECK(hsa_amd_memory_pool_allocate(
        kernarg_pool, kernarg_slot_size_ * kernarg_slot_count, 0,
        &kernarg_buffer_));
  }

  // Find the first memory pool on the AIE agent matching flags/allocatable (via
  // find_pool); store it in *out and return true, or return false if none.
  bool discover_pool(hsa_amd_memory_pool_global_flag_t flags, bool allocatable,
                     hsa_amd_memory_pool_t *out) {
    PoolSearch s{flags, allocatable, {}, false};
    hsa_amd_agent_iterate_memory_pools(aie_agent_, find_pool, &s);
    if (s.found) {
      *out = s.pool;
      return true;
    }
    return false;
  }

  // Slot address for ring index -- pure pointer arithmetic, no HSA call.
  void *kernarg_slot(std::uint32_t index) {
    return static_cast<std::byte *>(kernarg_buffer_) +
           static_cast<std::size_t>(index) * kernarg_slot_size_;
  }

  // Round a byte count up to the data-pool allocation granule (which the vmem
  // API requires; data_granule_ is normalized non-zero in init()).
  std::size_t round_up(std::size_t n) const {
    return ((n + data_granule_ - 1) / data_granule_) * data_granule_;
  }

  // Read a file into a fresh dev-pool allocation (a plain-pool DeviceBuffer:
  // `handle` stays zero, `va` is the pool pointer).
  DeviceBuffer load_binary(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("failed to open '" + path + "'");
    std::streamsize sz = f.tellg();
    if (sz <= 0) throw std::runtime_error("empty or unreadable '" + path + "'");
    f.seekg(0);
    void *buf = nullptr;
    HSA_CHECK(hsa_amd_memory_pool_allocate(
        dev_pool_, static_cast<std::size_t>(sz), 0, &buf));
    if (!f.read(static_cast<char *>(buf), sz)) {
      hsa_amd_memory_pool_free(buf);
      throw std::runtime_error("short read loading '" + path + "'");
    }
    DeviceBuffer b{};
    b.va = buf;
    b.size = static_cast<std::size_t>(sz);
    return b;
  }

  // Allocate a fresh vmem buffer of at least `size` bytes (rounded up to the
  // pool granule), mapped and granted RW access to the CPU and AIE agents.
  DeviceBuffer vmem_alloc(std::size_t size) {
    size = round_up(size);
    DeviceBuffer b{};
    b.size = size;
    HSA_CHECK(hsa_amd_vmem_handle_create(data_pool_, size, MEMORY_TYPE_PINNED, 0,
                                         &b.handle));
    HSA_CHECK(hsa_amd_vmem_address_reserve_align(
        &b.va, size, 0, 0, HSA_AMD_VMEM_ADDRESS_NO_REGISTER));
    HSA_CHECK(hsa_amd_vmem_map(b.va, size, 0, b.handle, 0));
    HSA_CHECK(hsa_amd_vmem_set_access(b.va, size, access_descs_.data(),
                                      access_descs_.size()));
    return b;
  }

  // Get a buffer of at least `size` bytes: reuse a pooled one of the matching
  // rounded size, or allocate. Not internally locked -- callers hold
  // dispatch_mtx_, which serializes all pool access.
  DeviceBuffer acquire(std::size_t size) {
    std::size_t rounded = round_up(size);
    auto it = vmem_pool_.find(rounded);
    if (it != vmem_pool_.end() && !it->second.empty()) {
      DeviceBuffer b = it->second.back();
      it->second.pop_back();
      return b;
    }
    return vmem_alloc(size);  // rounds to the same `rounded` size
  }

  // Return a buffer to the free-list for reuse (does not unmap).
  void release(DeviceBuffer &b) {
    if (!b.va) return;
    vmem_pool_[b.size].push_back(b);
    b.va = nullptr;
  }
};

// Meyers singleton: thread-safe lazy init, one HSA context per process.
HsaRuntime &runtime() {
  static HsaRuntime rt;
  return rt;
}

// Copy `msg` into a caller-provided errbuf, NUL-terminated.
void write_err(char *errbuf, size_t errbuf_len, const std::string &msg) {
  if (!errbuf || errbuf_len == 0) return;
  std::size_t n = msg.size() < errbuf_len - 1 ? msg.size() : errbuf_len - 1;
  std::memcpy(errbuf, msg.data(), n);
  errbuf[n] = '\0';
}

}  // namespace

extern "C" triton_npu_hsa_program_t triton_npu_hsa_prepare(
    const char *pdi_path, const char *insts_path, char *errbuf,
    size_t errbuf_len) {
  try {
    return runtime().prepare(pdi_path, insts_path);
  } catch (const std::exception &e) {
    write_err(errbuf, errbuf_len, std::string("HSA prepare failed: ") + e.what());
    return nullptr;
  }
}

extern "C" int triton_npu_hsa_dispatch(triton_npu_hsa_program_t program,
                                       uint32_t num_tensors,
                                       void *const *host_ptrs,
                                       const uint64_t *sizes, char *errbuf,
                                       size_t errbuf_len) {
  try {
    runtime().dispatch(program, num_tensors, host_ptrs, sizes);
    return 0;
  } catch (const std::exception &e) {
    write_err(errbuf, errbuf_len, std::string("HSA dispatch failed: ") + e.what());
    return -1;
  }
}
