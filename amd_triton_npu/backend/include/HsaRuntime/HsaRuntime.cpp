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
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// Per-dispatch watchdog timeout in seconds (fractional accepted). Unset, empty,
// unparseable, or <= 0 means wait forever, which is the default -- see the
// note on triton_npu_hsa_dispatch() in the header for why.
constexpr const char *TIMEOUT_ENV = "AMD_TRITON_NPU_HSA_TIMEOUT";

// Thrown when a dispatch exceeds the watchdog timeout. A distinct type because
// the recovery differs from an ordinary failure: the device may still be
// executing, so every device-visible resource the dispatch handed it must be
// abandoned rather than reused. Only used where that is actually true -- a
// timeout waiting for a *free* queue slot is a plain runtime_error, since the
// buffers acquired for it were never published to the device.
struct HsaTimeoutError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

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
  if ((f & d->flags) == 0)
    return HSA_STATUS_SUCCESS;

  std::size_t granule = 0;
  if (hsa_amd_memory_pool_get_info(
          pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE, &granule) !=
      HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;
  bool allocatable = (granule != 0);
  if (allocatable != d->allocatable)
    return HSA_STATUS_SUCCESS;

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

} // namespace

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
// kernarg pool, the pooled vmem I/O buffers, and the cache of prepared
// programs. One instance per process (see runtime()).
class HsaRuntime {
public:
  // Construct by running the full one-time HSA initialization.
  HsaRuntime() { init(); }

  ~HsaRuntime() {
    for (auto &kv : vmem_pool_)
      for (auto &b : kv.second)
        vmem_free(b);
    for (auto &kv : programs_) {
      if (kv.second->pdi.va)
        hsa_amd_memory_pool_free(kv.second->pdi.va);
      if (kv.second->insts.va)
        hsa_amd_memory_pool_free(kv.second->insts.va);
    }
    if (kernarg_buffer_)
      hsa_amd_memory_pool_free(kernarg_buffer_);
    if (signal_.handle)
      hsa_signal_destroy(signal_);
    if (queue_)
      hsa_queue_destroy(queue_);
    hsa_shut_down();
  }

  // The AIE agent's device name -- "aie2" (npu1/Phoenix) or "aie2p"
  // (npu2/Strix). Read from HSA_AGENT_INFO_NAME rather than the agent's ISA:
  // ROCR reports no ISA for the AIE agent (the query fails and leaves a null
  // handle), while the name carries the generation.
  std::string agent_name() {
    // HSA_AGENT_INFO_NAME writes a fixed 64-byte NUL-terminated string.
    char buf[64] = {};
    HSA_CHECK(hsa_agent_get_info(aie_agent_, HSA_AGENT_INFO_NAME, buf));
    buf[sizeof(buf) - 1] = '\0';
    return std::string(buf);
  }

  // Load + cache the PDI/insts for a kernel and return its program handle.
  // Keyed by (pdi_path, insts_path) so repeated prepares (or a PDI shared by
  // two signatures) reuse the same device allocation. Thread-safe.
  triton_npu_hsa_program *prepare(const char *pdi_path,
                                  const char *insts_path) {
    std::string key = std::string(pdi_path) + '\0' + insts_path;
    std::lock_guard<std::mutex> lock(programs_mtx_);
    auto it = programs_.find(key);
    if (it != programs_.end())
      return it->second.get();
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

      // Claim a ring slot. Only *peek* at the write index here; the advance is
      // published further down, once the packet is fully written. Reserving up
      // front instead would, on any failure in between, leave the read index
      // behind a slot that never gets written, wedging every later dispatch.
      // Safe to peek because the queue is single-producer under this lock.
      hsa_queue_t *q = queue_;
      const std::uint64_t wr_idx = hsa_queue_load_write_index_relaxed(q);
      wait_for_queue_space(q, wr_idx);
      const std::uint64_t pkt_idx = wr_idx % q->size;

      // Each ring slot owns a fixed kernarg slot of the same index; no
      // allocation on the hot path. Layout: [addr0..addrN-1, size0..sizeN-1].
      auto *kernargs =
          static_cast<std::uint64_t *>(kernarg_slot((std::uint32_t)pkt_idx));
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
      // negative error code) and write the packet into the slot. Nothing from
      // here to the write-index store can fail, so the index never runs ahead
      // of a written packet.
      hsa_signal_store_screlease(signal_, 1);
      static_cast<hsa_amd_aie_kernel_dispatch_packet_t *>(
          q->base_address)[pkt_idx] = pkt;

      // Publish the slot, then ring the doorbell.
      hsa_queue_store_write_index_screlease(q, wr_idx + 1);
      hsa_signal_store_screlease(q->doorbell_signal, wr_idx);

      const hsa_signal_value_t sig_val = wait_for_completion();
      if (sig_val != 0)
        throw std::runtime_error("AIE dispatch failed: completion signal = " +
                                 std::to_string((long long)sig_val));

      // Copy every tensor buffer back to its host pointer. We cannot know which
      // argument(s) the kernel writes, so copying all back is correct
      // regardless of output position (unmodified inputs just copy identical
      // bytes).
      for (std::uint32_t i = 0; i < num_tensors; ++i)
        std::memcpy(host_ptrs[i], bufs[i].va, (std::size_t)sizes[i]);

      for (std::uint32_t i = 0; i < num_tensors; ++i)
        release(bufs[i]);
    } catch (const HsaTimeoutError &) {
      // Deliberately do NOT return the buffers to the pool. The dispatch is
      // still outstanding as far as we know, so the device may write into them
      // at any time; handing them to the next dispatch would be a device-side
      // use-after-free. Abandoning them leaks, which is the lesser evil and is
      // why the watchdog is opt-in.
      //
      // The kernarg slot cannot be abandoned the same way -- it is fixed to
      // ring slot pkt_idx. That is tolerable: the read index does not advance
      // past a dispatch that never completed, so the next dispatch takes the
      // following slot and it would take a full lap of the ring to reuse the
      // one still in the device's hands.
      throw;
    } catch (...) {
      // Ordinary failure: the device is done with (or never saw) these
      // buffers, so return them to the pool rather than leaking them.
      for (std::uint32_t i = 0; i < acquired; ++i)
        release(bufs[i]);
      throw;
    }
  }

private:
  hsa_agent_t aie_agent_{};
  hsa_amd_memory_pool_t dev_pool_{};
  hsa_amd_memory_pool_t data_pool_{};
  hsa_queue_t *queue_ = nullptr;
  hsa_signal_t signal_{};
  std::vector<hsa_amd_memory_access_desc_t> access_descs_; // RW, built once
  std::vector<hsa_amd_memory_access_desc_t> revoke_descs_; // NONE, built once
  std::size_t data_granule_ = 0;

  // Watchdog. Zero duration means "wait forever" (the default) and disables
  // every deadline check below. wait_ticks_ bounds one hsa_signal_wait call so
  // the deadline is actually reached; it stays UINT64_MAX when disabled.
  std::chrono::steady_clock::duration timeout_{};
  std::uint64_t wait_ticks_ = UINT64_MAX;
  double timeout_secs_ = 0.0; // as configured, for the error message

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
    init_timeout(); // queries system info, so it must follow hsa_init

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
    for (auto c : cpus)
      access_agents.push_back(c);
    for (auto a : aies)
      access_agents.push_back(a);
    access_descs_.reserve(access_agents.size());
    revoke_descs_.reserve(access_agents.size());
    for (auto a : access_agents) {
      access_descs_.push_back({HSA_ACCESS_PERMISSION_RW, a});
      // The mirror image, used to drop those grants before unmapping.
      revoke_descs_.push_back({HSA_ACCESS_PERMISSION_NONE, a});
    }

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
    if (data_granule_ == 0)
      data_granule_ = 4096; // sane fallback for round_up

    // Queue, capped at QUEUE_SIZE but clamped into the agent's supported range
    // (the AIE agent requires an exact size, e.g. 64, so clamp up to min_q).
    std::uint32_t min_q = 0, max_q = 0;
    HSA_CHECK(
        hsa_agent_get_info(aie_agent_, HSA_AGENT_INFO_QUEUE_MIN_SIZE, &min_q));
    HSA_CHECK(
        hsa_agent_get_info(aie_agent_, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &max_q));
    std::uint32_t qsize = QUEUE_SIZE;
    if (qsize < min_q)
      qsize = min_q;
    if (max_q > 0 && qsize > max_q)
      qsize = max_q;
    HSA_CHECK(hsa_queue_create(aie_agent_, qsize, HSA_QUEUE_TYPE_SINGLE,
                               nullptr, nullptr, 0, 0, &queue_));

    // Persistent completion signal, armed per dispatch.
    HSA_CHECK(hsa_signal_create(0, 0, nullptr, &signal_));

    // Kernarg slot pool: one slot per ring slot, sized for the worst case
    // (MAX_KERNARGS addresses + MAX_KERNARGS sizes), aligned to 64 bytes.
    std::uint32_t kernarg_slot_count = static_cast<std::uint32_t>(queue_->size);
    std::size_t raw = static_cast<std::size_t>(TRITON_NPU_HSA_MAX_KERNARGS) *
                      2 * sizeof(std::uint64_t);
    kernarg_slot_size_ = (raw + 63u) & ~static_cast<std::size_t>(63u);
    HSA_CHECK(hsa_amd_memory_pool_allocate(
        kernarg_pool, kernarg_slot_size_ * kernarg_slot_count, 0,
        &kernarg_buffer_));
  }

  // Read the watchdog timeout from the environment and convert it into the
  // per-wait tick hint hsa_signal_wait_scacquire takes. That hint is advisory
  // -- the wait may return early or late -- so it is not the timeout itself;
  // callers re-check a wall-clock deadline around it. All it has to do is be
  // short enough that the deadline gets checked, hence the clamp to the
  // agent's maximum supported wait. Leaves the watchdog disabled if the
  // timestamp frequency is unavailable, since without it seconds cannot be
  // converted to ticks and every wait would block past the deadline anyway.
  void init_timeout() {
    const char *v = std::getenv(TIMEOUT_ENV);
    if (!v || !*v)
      return;
    const double secs = std::strtod(v, nullptr);
    if (!(secs > 0.0))
      return;

    std::uint64_t freq = 0;
    if (hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &freq) !=
            HSA_STATUS_SUCCESS ||
        freq == 0) {
      std::fprintf(stderr,
                   "[triton-npu-hsa] %s ignored: the HSA timestamp frequency "
                   "is unavailable, so the dispatch watchdog cannot be armed\n",
                   TIMEOUT_ENV);
      return;
    }

    std::uint64_t max_wait = 0;
    if (hsa_system_get_info(HSA_SYSTEM_INFO_SIGNAL_MAX_WAIT, &max_wait) !=
            HSA_STATUS_SUCCESS ||
        max_wait == 0)
      max_wait = UINT64_MAX;

    const double ticks = secs * static_cast<double>(freq);
    wait_ticks_ = ticks >= static_cast<double>(max_wait)
                      ? max_wait
                      : static_cast<std::uint64_t>(ticks);
    if (wait_ticks_ == 0)
      wait_ticks_ = 1;
    timeout_ = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(secs));
    timeout_secs_ = secs;
  }

  // True once the watchdog is armed (see init_timeout).
  bool timed() const {
    return timeout_ != std::chrono::steady_clock::duration{};
  }

  // Block until ring slot `wr_idx` is free, i.e. the device has consumed the
  // packet that previously occupied it. Unreachable under the current
  // synchronous model -- dispatch() waits for each packet before returning, so
  // at most one is ever outstanding -- but a timed-out dispatch leaves the read
  // index permanently behind, so honor the watchdog here rather than spinning
  // forever. Throws a plain runtime_error, not HsaTimeoutError: the caller's
  // buffers were never handed to the device and are safe to reclaim.
  void wait_for_queue_space(hsa_queue_t *q, std::uint64_t wr_idx) const {
    const auto deadline = std::chrono::steady_clock::now() + timeout_;
    while (wr_idx - hsa_queue_load_read_index_scacquire(q) >= q->size) {
      if (timed() && std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error(
            "timed out waiting for a free slot in the AIE queue; an earlier "
            "dispatch is still outstanding and the device is not draining");
      std::this_thread::yield();
    }
  }

  // Wait for the in-flight dispatch to finish and return the signal value.
  // Waits for < 1 rather than == 1 so a device error (a negative code) is
  // observed instead of hanging, and loops to ignore spurious wake-ups that
  // still see the armed value. The caller releases the GIL around all of this.
  hsa_signal_value_t wait_for_completion() {
    const auto deadline = std::chrono::steady_clock::now() + timeout_;
    for (;;) {
      const hsa_signal_value_t v =
          hsa_signal_wait_scacquire(signal_, HSA_SIGNAL_CONDITION_LT, 1,
                                    wait_ticks_, HSA_WAIT_STATE_BLOCKED);
      if (v < 1)
        return v;
      if (timed() && std::chrono::steady_clock::now() >= deadline) {
        abandon_signal();
        throw HsaTimeoutError("AIE dispatch did not complete within " +
                              std::to_string(timeout_secs_) + "s (" +
                              TIMEOUT_ENV + ")");
      }
    }
  }

  // Install a fresh completion signal after a timeout. The old one is leaked on
  // purpose: the dispatch may still be in flight, and the device will decrement
  // whenever it finishes. Reusing it would let that late decrement corrupt the
  // next dispatch's count; destroying it would leave the device writing to
  // freed memory.
  void abandon_signal() {
    hsa_signal_t fresh{};
    if (hsa_signal_create(0, 0, nullptr, &fresh) == HSA_STATUS_SUCCESS) {
      signal_ = fresh;
      return;
    }
    // Nothing better to do from an error path than keep the old signal and let
    // the report through; say so rather than silently degrading.
    std::fprintf(stderr,
                 "[triton-npu-hsa] could not replace the completion signal "
                 "after a timeout; subsequent dispatches may misreport\n");
  }

  // Report a non-success status from a path that must not throw (teardown).
  // Only fires on an actual HSA error. Dropping these entirely is what kept
  // the unmap-before-revoke bug invisible.
  static void log_status(const char *what, hsa_status_t st) {
    if (st == HSA_STATUS_SUCCESS)
      return;
    const char *m = nullptr;
    hsa_status_string(st, &m);
    std::fprintf(stderr, "[triton-npu-hsa] %s failed: %s\n", what,
                 m ? m : "unknown HSA error");
  }

  // Grant (RW to every CPU and AIE agent) or revoke (no access at all) access
  // to a mapped vmem range.
  hsa_status_t set_vmem_access(void *va, std::size_t size, bool grant) {
    const auto &d = grant ? access_descs_ : revoke_descs_;
    return hsa_amd_vmem_set_access(va, size, d.data(), d.size());
  }

  // Tear down a vmem buffer. Access must be revoked before unmapping: ROCR
  // rejects an unmap while agents still hold access grants, and since the range
  // then stays mapped, the next reservation at that address fails too.
  void vmem_free(DeviceBuffer &b) {
    if (!b.va)
      return;
    log_status("hsa_amd_vmem_set_access(NONE)",
               set_vmem_access(b.va, b.size, false));
    log_status("hsa_amd_vmem_unmap", hsa_amd_vmem_unmap(b.va, b.size));
    log_status("hsa_amd_vmem_address_free",
               hsa_amd_vmem_address_free(b.va, b.size));
    log_status("hsa_amd_vmem_handle_release",
               hsa_amd_vmem_handle_release(b.handle));
    b.va = nullptr;
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
    if (!f)
      throw std::runtime_error("failed to open '" + path + "'");
    std::streamsize sz = f.tellg();
    if (sz <= 0)
      throw std::runtime_error("empty or unreadable '" + path + "'");
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
  // Each step is undone if a later one fails. This is not just leak hygiene:
  // a half-built buffer strands a *mapping*, and a stranded mapping makes the
  // next allocation that reserves the same virtual address fail -- so leaking
  // here breaks unrelated allocations later, not merely this one.
  DeviceBuffer vmem_alloc(std::size_t size) {
    size = round_up(size);
    DeviceBuffer b{};
    b.size = size;
    HSA_CHECK(hsa_amd_vmem_handle_create(data_pool_, size, MEMORY_TYPE_PINNED,
                                         0, &b.handle));
    try {
      HSA_CHECK(hsa_amd_vmem_address_reserve_align(
          &b.va, size, 0, 0, HSA_AMD_VMEM_ADDRESS_NO_REGISTER));
    } catch (...) {
      log_status("hsa_amd_vmem_handle_release",
                 hsa_amd_vmem_handle_release(b.handle));
      throw;
    }
    try {
      HSA_CHECK(hsa_amd_vmem_map(b.va, size, 0, b.handle, 0));
    } catch (...) {
      log_status("hsa_amd_vmem_address_free",
                 hsa_amd_vmem_address_free(b.va, size));
      log_status("hsa_amd_vmem_handle_release",
                 hsa_amd_vmem_handle_release(b.handle));
      throw;
    }
    try {
      HSA_CHECK(set_vmem_access(b.va, size, true));
    } catch (...) {
      // Access was never granted, so unmapping directly is safe here (the
      // revoke in vmem_free exists to drop grants that *were* applied).
      log_status("hsa_amd_vmem_unmap", hsa_amd_vmem_unmap(b.va, size));
      log_status("hsa_amd_vmem_address_free",
                 hsa_amd_vmem_address_free(b.va, size));
      log_status("hsa_amd_vmem_handle_release",
                 hsa_amd_vmem_handle_release(b.handle));
      throw;
    }
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
    return vmem_alloc(size); // rounds to the same `rounded` size
  }

  // Return a buffer to the free-list for reuse (does not unmap).
  void release(DeviceBuffer &b) {
    if (!b.va)
      return;
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
  if (!errbuf || errbuf_len == 0)
    return;
  std::size_t n = msg.size() < errbuf_len - 1 ? msg.size() : errbuf_len - 1;
  std::memcpy(errbuf, msg.data(), n);
  errbuf[n] = '\0';
}

} // namespace

extern "C" int triton_npu_hsa_agent_name(char *buf, size_t buf_len,
                                         char *errbuf, size_t errbuf_len) {
  try {
    if (!buf || buf_len == 0)
      throw std::runtime_error("null or zero-length output buffer");
    const std::string name = runtime().agent_name();
    if (name.size() + 1 > buf_len)
      throw std::runtime_error("agent name '" + name +
                               "' does not fit in the " +
                               std::to_string(buf_len) + "-byte buffer");
    std::memcpy(buf, name.c_str(), name.size() + 1);
    return 0;
  } catch (const std::exception &e) {
    write_err(errbuf, errbuf_len,
              std::string("HSA agent query failed: ") + e.what());
    return -1;
  }
}

extern "C" triton_npu_hsa_program_t
triton_npu_hsa_prepare(const char *pdi_path, const char *insts_path,
                       char *errbuf, size_t errbuf_len) {
  try {
    return runtime().prepare(pdi_path, insts_path);
  } catch (const std::exception &e) {
    write_err(errbuf, errbuf_len,
              std::string("HSA prepare failed: ") + e.what());
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
    write_err(errbuf, errbuf_len,
              std::string("HSA dispatch failed: ") + e.what());
    return -1;
  }
}
