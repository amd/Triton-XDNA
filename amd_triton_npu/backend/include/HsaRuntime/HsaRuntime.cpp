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
// * Shared regions: the same vmem API, but owned by the caller rather than the
//   pool, and either allocated here or imported from a dma-buf another runtime
//   exported. A dispatch naming one runs on it in place -- no staging buffer,
//   neither copy -- which is the whole point of them.
// * Kernel arguments: a fixed-slot pool (one slot per ring slot); slot(i) is
//   pure pointer arithmetic, no HSA call on the hot path.

#include "HsaRuntime/HsaRuntime.h"

#include <array>
#include <atomic>
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

#include <sys/stat.h> // fstat(), to size the object a dma-buf names
#include <unistd.h>   // close(), for the dma-buf an import is reached through

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

// Build (but do not throw) a runtime_error describing a failed HSA call.
// Returning it rather than throwing lets a caller that must release resources
// first run its cleanup and then throw once, instead of unwinding through a
// catch block only to rethrow.
std::runtime_error hsa_error(const char *what, hsa_status_t status) {
  const char *m = nullptr;
  hsa_status_string(status, &m);
  return std::runtime_error(std::string(what) +
                            " failed: " + (m ? m : "unknown HSA error"));
}

// Throw std::runtime_error with the HSA status string on a non-success status.
// For calls with nothing to clean up on failure; where cleanup is needed,
// check the status directly and throw hsa_error() after unwinding.
#define HSA_CHECK(expr)                                                        \
  do {                                                                         \
    hsa_status_t _s = (expr);                                                  \
    if (_s != HSA_STATUS_SUCCESS)                                              \
      throw hsa_error(#expr, _s);                                              \
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

// How each tensor argument reached the device, counted since the process
// started. Outside HsaRuntime on purpose: reading a counter must not be able to
// bring the runtime up, or asking "is my buffer being shared?" from a process
// driving the NPU through XRT would try to open a queue XRT already holds.
// Written under the dispatch lock, read without it.
std::atomic<std::uint64_t> g_in_place{0};
std::atomic<std::uint64_t> g_staged{0};

// Which agents a vmem range is granted access to. Locally allocated ranges take
// CPU_AND_AIE, so the host can stage into them; an imported range can only take
// AIE_ONLY, because ROCR rejects a CPU grant on memory another runtime owns
// (whether the CPU can reach it is then that runtime's business, not ours).
enum class Access { CPU_AND_AIE, AIE_ONLY };

// A caller-owned range both the AIE agent and someone else can address.
//
// `size` is what the caller asked for, not the granule-rounded `buf.size`: it
// is what bounds checking must use, since the tail of the rounding is not the
// caller's memory.
struct SharedRegion {
  DeviceBuffer buf{};   // the vmem allocation or import behind it
  void *aie_va{};       // what the dispatch packet needs; buf.va, or an
                        // offset into it when the mapping covers more than
                        // the caller's range (see vmem_import)
  std::size_t size{};   // requested size
  bool imported{false}; // someone else's memory, mapped here -- which decides
                        // both how it was granted and how it is released
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
    // Regions first, and before hsa_shut_down: a caller that leaked a shared
    // buffer would otherwise leave a mapping to unmap after the runtime it
    // belongs to is gone. Through shared_free, so that the several keys that
    // may name one region are retired together rather than freed once each.
    while (!regions_.empty())
      shared_free(reinterpret_cast<void *>(regions_.begin()->first));
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
    // What the device is given for tensor i: its own address when it is
    // already in a shared region, otherwise the staging buffer's.
    std::array<void *, TRITON_NPU_HSA_MAX_KERNARGS> dev_addr{};
    try {
      // A shared tensor is already memory the AIE agent can reach: dispatch on
      // it in place. Every other one gets a pooled I/O buffer and a copy in.
      for (std::uint32_t i = 0; i < num_tensors; ++i) {
        if (sizes[i] == 0)
          // Otherwise this surfaces further down as an HSA argument error
          // about a zero-byte allocation, which says nothing about which
          // operand the caller got wrong.
          throw std::runtime_error(
              "tensor argument " + std::to_string(i) +
              " is empty; there is nothing to give the device for it");
        if (void *shared =
                resolve_shared(host_ptrs[i], (std::size_t)sizes[i])) {
          dev_addr[i] = shared;
          ++g_in_place;
          continue;
        }
        bufs[i] = acquire((std::size_t)sizes[i]);
        std::memcpy(bufs[i].va, host_ptrs[i], (std::size_t)sizes[i]);
        dev_addr[i] = bufs[i].va;
        ++g_staged;
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
        kernargs[i] = reinterpret_cast<std::uint64_t>(dev_addr[i]);
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

      // Copy every staged tensor back to its host pointer. We cannot know which
      // argument(s) the kernel writes, so copying all back is correct
      // regardless of output position (unmodified inputs just copy identical
      // bytes). A shared tensor has no staging buffer and needs no copy: the
      // device wrote where the caller reads.
      for (std::uint32_t i = 0; i < num_tensors; ++i)
        if (bufs[i].va)
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
      //
      // Shared regions are not ours to abandon either way: they belong to the
      // caller, who must not free one while a dispatch on it may still be
      // outstanding. Nothing enforces that, for the same reason nothing stops
      // a caller freeing a tensor mid-launch.
      throw;
    } catch (...) {
      // Ordinary failure: the device is done with (or never saw) these
      // buffers, so return them to the pool rather than leaking them. Slots
      // that were never filled -- a shared tensor takes none, and a failure
      // partway leaves the rest empty -- are ignored by release().
      for (std::uint32_t i = 0; i < num_tensors; ++i)
        release(bufs[i]);
      throw;
    }
  }

  // ---- Shared regions ------------------------------------------------------
  // Allocate a region the CPU and the AIE agent can both reach, and register
  // it under its own address.
  void *shared_alloc(std::size_t size) {
    if (size == 0)
      throw std::runtime_error("shared region size must be non-zero");
    auto region = std::make_shared<SharedRegion>();
    region->buf = vmem_alloc(size);
    region->aie_va = region->buf.va;
    region->size = size;
    std::lock_guard<std::mutex> lock(regions_mtx_);
    regions_[reinterpret_cast<std::uintptr_t>(region->aie_va)] = region;
    return region->aie_va;
  }

  // Map memory another agent owns for the AIE agent, and register it under
  // both that agent's address for it and our own.
  void *shared_import(void *ptr, std::size_t size) {
    if (size == 0)
      throw std::runtime_error("shared region size must be non-zero");
    if (!ptr)
      throw std::runtime_error("cannot import a null address");
    auto region = std::make_shared<SharedRegion>();
    vmem_import(ptr, size, *region);
    std::lock_guard<std::mutex> lock(regions_mtx_);
    regions_[reinterpret_cast<std::uintptr_t>(region->aie_va)] = region;
    if (ptr != region->aie_va)
      regions_[reinterpret_cast<std::uintptr_t>(ptr)] = region;
    return region->aie_va;
  }

  // Register one more address for an existing region.
  void shared_alias(void *alias, void *va, std::size_t size) {
    if (!alias)
      throw std::runtime_error("alias address must not be null");
    std::lock_guard<std::mutex> lock(regions_mtx_);
    const RegionHit hit = find_region(va);
    if (!hit.region)
      throw std::runtime_error("no shared region at the given address");
    if (hit.offset != 0)
      throw std::runtime_error(
          "an alias must name the start of a region, not an offset into it");
    if (size > hit.region->size)
      throw std::runtime_error("alias covers " + std::to_string(size) +
                               " bytes but the region is " +
                               std::to_string(hit.region->size));
    regions_[reinterpret_cast<std::uintptr_t>(alias)] = hit.region;
  }

  // Forget one address, leaving the region and its other addresses alone.
  void shared_unalias(void *alias) {
    std::lock_guard<std::mutex> lock(regions_mtx_);
    regions_.erase(reinterpret_cast<std::uintptr_t>(alias));
  }

  // Release a region and forget every address that named it. Silent about an
  // address that names no region, so the Python side can release twice.
  void shared_free(void *va) {
    std::shared_ptr<SharedRegion> region;
    {
      std::lock_guard<std::mutex> lock(regions_mtx_);
      region = find_region(va).region;
      if (!region)
        return;
      for (auto it = regions_.begin(); it != regions_.end();)
        it = (it->second == region) ? regions_.erase(it) : std::next(it);
    }
    // Unmapped outside the lock: the teardown is several HSA calls and holding
    // regions_mtx_ across them would block every concurrent dispatch's lookup.
    // Safe because the region is unreachable by then -- no key names it.
    vmem_free(region->buf, region->imported);
  }

private:
  hsa_agent_t aie_agent_{};
  hsa_amd_memory_pool_t dev_pool_{};
  hsa_amd_memory_pool_t data_pool_{};
  hsa_queue_t *queue_ = nullptr;
  hsa_signal_t signal_{};
  // The agents a vmem range can be granted to, discovered once in init().
  std::vector<hsa_agent_t> aie_agents_;
  std::vector<hsa_agent_t> cpu_agents_;
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

  // Shared regions, keyed by every address a caller may name one by. Ordered,
  // because a dispatch resolves an address that may point *into* a region, not
  // just at its base. shared_ptr rather than a value: several keys (a region
  // and its aliases) name one region, and dropping one key must not disturb
  // the others.
  std::map<std::uintptr_t, std::shared_ptr<SharedRegion>> regions_;

  // Guards regions_ only. Deliberately not dispatch_mtx_: buffers are
  // registered and released from Python while a dispatch may be running, so
  // the two are not serialized against each other. A dispatch takes this one
  // while holding dispatch_mtx_ and the shared_* entry points never take
  // dispatch_mtx_, so the pair cannot deadlock.
  std::mutex regions_mtx_;

  // Serializes dispatches (one shared queue, one packet in flight).
  std::mutex dispatch_mtx_;

  // Where `ptr` lands: the region it is inside and how far into it. `region`
  // is null when it is inside none, and `offset` is then meaningless.
  struct RegionHit {
    std::shared_ptr<SharedRegion> region;
    std::size_t offset{};
  };

  // The region `ptr` falls inside, with its offset. Caller holds regions_mtx_.
  //
  // The nearest key at or below `ptr` is the only candidate: no two regions
  // overlap, since each is a separate mapping, and each alias names memory
  // some other runtime allocated separately.
  //
  // An alias shares the region's layout by construction -- it is a second
  // address for the same pages -- so the offset from whichever key matched is
  // the offset into the region.
  RegionHit find_region(void *ptr) const {
    const auto p = reinterpret_cast<std::uintptr_t>(ptr);
    auto it = regions_.upper_bound(p);
    if (it == regions_.begin())
      return {};
    --it;
    const std::size_t offset = p - it->first;
    if (offset >= it->second->size)
      return {};
    return {it->second, offset};
  }

  // The AIE-side address for [ptr, ptr+size), or null when ptr names no shared
  // region -- in which case the caller stages the tensor as usual.
  //
  // A tensor that starts inside a region but runs past its end throws rather
  // than falling back to staging: the fallback would memcpy an address that is
  // only partly the caller's, which on unified memory reads back as plausible
  // data instead of faulting.
  //
  // The span is taken as ptr + size because that is what the dispatch ABI says
  // a tensor occupies. A strided view does not occupy that span, but no more so
  // here than in the staging path, which copies exactly those bytes.
  void *resolve_shared(void *ptr, std::size_t size) {
    std::lock_guard<std::mutex> lock(regions_mtx_);
    const RegionHit hit = find_region(ptr);
    if (!hit.region)
      return nullptr;
    if (hit.offset + size > hit.region->size)
      throw std::runtime_error(
          "tensor at offset " + std::to_string(hit.offset) + " spanning " +
          std::to_string(size) + " bytes runs past the end of the " +
          std::to_string(hit.region->size) +
          "-byte shared region it starts in");
    return static_cast<std::byte *>(hit.region->aie_va) + hit.offset;
  }

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

    // Who a vmem range can be granted to: the AIE agent always (execution), and
    // the CPU when the range is ours to grant (host staging). Kept as agents
    // rather than as ready-made descriptor lists, since the permission varies
    // as much as the audience does -- see set_vmem_access.
    aie_agents_ = aies;
    cpu_agents_ = cpus;

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

  // Give the agents `who` names permission `perm` over a mapped vmem range;
  // HSA_ACCESS_PERMISSION_NONE revokes. Off every hot path -- once per
  // allocation and once per teardown -- so the descriptor list is built here
  // rather than cached per (permission, agent set) combination.
  hsa_status_t set_vmem_access(void *va, std::size_t size,
                               hsa_access_permission_t perm, Access who) {
    std::vector<hsa_amd_memory_access_desc_t> d;
    d.reserve(aie_agents_.size() + cpu_agents_.size());
    for (auto a : aie_agents_)
      d.push_back({perm, a});
    if (who == Access::CPU_AND_AIE)
      for (auto a : cpu_agents_)
        d.push_back({perm, a});
    return hsa_amd_vmem_set_access(va, size, d.data(), d.size());
  }

  // Tear down a vmem buffer. Access must be revoked before unmapping: ROCR
  // rejects an unmap while agents still hold access grants, and since the range
  // then stays mapped, the next reservation at that address fails too.
  //
  // Except on an imported range, where ROCR refuses the revoke -- the grant is
  // over memory it does not own -- and the unmap drops it anyway. Asking would
  // only print a failure for something that then works. So `imported` decides
  // both halves, which is why it is the only thing the caller has to say: how
  // the range was granted follows from where it came from.
  void vmem_free(DeviceBuffer &b, bool imported = false) {
    if (!b.va)
      return;
    if (!imported)
      log_status("hsa_amd_vmem_set_access(NONE)",
                 set_vmem_access(b.va, b.size, HSA_ACCESS_PERMISSION_NONE,
                                 Access::CPU_AND_AIE));
    log_status("hsa_amd_vmem_unmap", hsa_amd_vmem_unmap(b.va, b.size));
    release_address(b);
    release_handle(b);
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

  // Reserve `b.size` bytes of address space for the handle `b` already holds,
  // map it there, and grant `who` RW access. Each step is undone if a later one
  // fails. This is not just leak hygiene: a half-built buffer strands a
  // *mapping*, and a stranded mapping makes the next allocation that reserves
  // the same virtual address fail -- so leaking here breaks unrelated
  // allocations later, not merely this one. Which is also why both ways of
  // obtaining a handle share this one ladder rather than each having its own.
  void vmem_publish(DeviceBuffer &b, Access who) {
    hsa_status_t st = hsa_amd_vmem_address_reserve_align(
        &b.va, b.size, 0, 0, HSA_AMD_VMEM_ADDRESS_NO_REGISTER);
    if (st != HSA_STATUS_SUCCESS) {
      release_handle(b);
      throw hsa_error("hsa_amd_vmem_address_reserve_align", st);
    }

    st = hsa_amd_vmem_map(b.va, b.size, 0, b.handle, 0);
    if (st != HSA_STATUS_SUCCESS) {
      release_address(b);
      release_handle(b);
      throw hsa_error("hsa_amd_vmem_map", st);
    }

    st = set_vmem_access(b.va, b.size, HSA_ACCESS_PERMISSION_RW, who);
    if (st != HSA_STATUS_SUCCESS) {
      // Access was never granted, so unmapping directly is safe here (the
      // revoke in vmem_free exists to drop grants that *were* applied).
      log_status("hsa_amd_vmem_unmap", hsa_amd_vmem_unmap(b.va, b.size));
      release_address(b);
      release_handle(b);
      throw hsa_error("hsa_amd_vmem_set_access", st);
    }
  }

  // Allocate a fresh vmem buffer of at least `size` bytes (rounded up to the
  // pool granule), mapped and granted RW access to the CPU and AIE agents.
  DeviceBuffer vmem_alloc(std::size_t size) {
    DeviceBuffer b{};
    b.size = round_up(size);
    hsa_status_t st = hsa_amd_vmem_handle_create(
        data_pool_, b.size, MEMORY_TYPE_PINNED, 0, &b.handle);
    if (st != HSA_STATUS_SUCCESS)
      throw hsa_error("hsa_amd_vmem_handle_create", st);
    vmem_publish(b, Access::CPU_AND_AIE);
    return b;
  }

  // Reclaim steps of a vmem buffer, in reverse order of acquisition. Used both
  // by vmem_free and to unwind a partially built buffer in vmem_alloc; they
  // report rather than throw, so an allocation failure surfaces its own status
  // instead of one from the cleanup.
  void release_address(const DeviceBuffer &b) {
    log_status("hsa_amd_vmem_address_free",
               hsa_amd_vmem_address_free(b.va, b.size));
  }

  void release_handle(const DeviceBuffer &b) {
    log_status("hsa_amd_vmem_handle_release",
               hsa_amd_vmem_handle_release(b.handle));
  }

  // Map memory another agent owns, granting it to the AIE agents (see Access
  // for why they are the only ones), and fill in `region` -- the whole buffer
  // object becomes the mapping, and the caller's range is a slice of it.
  //
  // The range is reached through its dma-buf, and an iGPU allocation is rarely
  // a whole buffer object -- ROCm packs several into one, and a descriptor
  // names the object, not the allocation. Hence the offset, which the export
  // reports; asking HIP for the descriptor instead would not have reported it,
  // which is why the export is done here.
  //
  // The whole object is mapped and the offset applied to the address handed
  // out, rather than passing it to the map as `in_offset`: the AIE path
  // accepts that argument and ignores it, so a range at a non-zero offset
  // would be mapped from the object's start -- reading and writing a
  // neighbour's memory, with nothing reporting a problem.
  //
  // Two consequences of mapping the whole object, both accepted. The AIE agent
  // can reach whatever else shares it -- those are this process's own
  // allocations, and ROCR offers no finer granularity. And two ranges from one
  // object are mapped twice rather than once: the same pages, at two addresses,
  // costing address space and a second set of page-table entries but no
  // physical memory. Deduplicating would mean identifying the object behind a
  // descriptor and refcounting the mapping, which is more machinery than the
  // handful of buffers a caller shares is worth.
  void vmem_import(void *ptr, std::size_t size, SharedRegion &region) {
    int dmabuf_fd = -1;
    std::uint64_t offset = 0;
    hsa_status_t st =
        hsa_amd_portable_export_dmabuf(ptr, size, &dmabuf_fd, &offset);
    if (st != HSA_STATUS_SUCCESS)
      throw hsa_error("hsa_amd_portable_export_dmabuf", st);

    // The descriptor's size is the object's size, which is what has to be
    // mapped for the tail of it to be reachable.
    struct stat info {};
    const bool sized = ::fstat(dmabuf_fd, &info) == 0 && info.st_size > 0;
    region.buf.size =
        sized ? (std::size_t)info.st_size : round_up(offset + size);

    st = hsa_amd_vmem_import_shareable_handle(dmabuf_fd, &region.buf.handle);
    // The import holds its own reference to the underlying allocation, so the
    // descriptor has done its job either way and closing it here keeps its
    // lifetime inside this function.
    ::close(dmabuf_fd);
    if (st != HSA_STATUS_SUCCESS)
      throw hsa_error("hsa_amd_vmem_import_shareable_handle", st);

    if (offset + size > region.buf.size) {
      release_handle(region.buf);
      throw std::runtime_error(
          "the exported buffer object is " + std::to_string(region.buf.size) +
          " bytes but the range starts at " + std::to_string(offset) +
          " and runs for " + std::to_string(size));
    }

    vmem_publish(region.buf, Access::AIE_ONLY);
    region.aie_va = static_cast<std::byte *>(region.buf.va) + offset;
    region.size = size;
    region.imported = true;
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

extern "C" void triton_npu_hsa_dispatch_counts(uint64_t *in_place,
                                               uint64_t *staged) {
  // No runtime() here, and so nothing that can throw: see g_in_place.
  if (in_place)
    *in_place = g_in_place.load();
  if (staged)
    *staged = g_staged.load();
}

extern "C" void *triton_npu_hsa_shared_alloc(uint64_t size, char *errbuf,
                                             size_t errbuf_len) {
  try {
    return runtime().shared_alloc((size_t)size);
  } catch (const std::exception &e) {
    write_err(errbuf, errbuf_len,
              std::string("HSA shared allocation failed: ") + e.what());
    return nullptr;
  }
}

extern "C" void *triton_npu_hsa_shared_import(void *ptr, uint64_t size,
                                              char *errbuf, size_t errbuf_len) {
  try {
    return runtime().shared_import(ptr, (size_t)size);
  } catch (const std::exception &e) {
    write_err(errbuf, errbuf_len,
              std::string("HSA shared import failed: ") + e.what());
    return nullptr;
  }
}

extern "C" int triton_npu_hsa_shared_alias(void *alias, void *va, uint64_t size,
                                           char *errbuf, size_t errbuf_len) {
  try {
    runtime().shared_alias(alias, va, (size_t)size);
    return 0;
  } catch (const std::exception &e) {
    write_err(errbuf, errbuf_len,
              std::string("HSA shared alias failed: ") + e.what());
    return -1;
  }
}

extern "C" int triton_npu_hsa_shared_unalias(void *alias, char *errbuf,
                                             size_t errbuf_len) {
  try {
    runtime().shared_unalias(alias);
    return 0;
  } catch (const std::exception &e) {
    write_err(errbuf, errbuf_len,
              std::string("HSA shared unalias failed: ") + e.what());
    return -1;
  }
}

extern "C" int triton_npu_hsa_shared_free(void *va, char *errbuf,
                                          size_t errbuf_len) {
  try {
    runtime().shared_free(va);
    return 0;
  } catch (const std::exception &e) {
    write_err(errbuf, errbuf_len,
              std::string("HSA shared release failed: ") + e.what());
    return -1;
  }
}
