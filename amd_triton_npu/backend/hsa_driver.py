# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""HSA launch runtime for the Triton-XDNA NPU backend.

This driver dispatches Triton-generated NPU kernels through ROCR using the
AIE agent path (``hsa_amd_aie_kernel_dispatch_packet_t``) instead of XRT. It
reuses the shared MLIR -> AIR -> PDI compile pipeline (``compile_module`` with
``output_format="pdi"``) and swaps in an HSA-based C++ launcher.

Selection:

* ``AMD_TRITON_NPU_RUNTIME=hsa`` (or ``npu_config.runtime = "hsa"``), then
  ``triton.runtime.driver.set_active(HSADriver())``.

Memory strategy (see docs/hsa-zero-copy-notes.md for the deferred zero-copy
path):

* **PDI + instructions**: plain HSA pool allocation from the AIE agent's dev
  pool (coarse-grained, non-allocatable), loaded once per artifact path and
  cached.
* **Data (I/O tensors)**: the vmem API (handle_create -> address_reserve ->
  map -> set_access), made accessible to both the CPU (host memcpy) and AIE
  (execution) agents. Allocated per launch and freed after.
* **Kernel arguments**: a fixed-slot pool (ggml-hsa pattern) -- one backing
  buffer allocated once from the kernarg pool and carved into ``queue->size``
  aligned slots. Slot ``i`` is owned by ring slot ``i``; ``slot(i)`` is pure
  pointer arithmetic with no HSA call on the hot path.
"""

from .config import npu_config
from ._codegen import (
    _extract_signature_and_constants,
    _extracted_type,
    _format_of,
    _ty_to_cpp,
)
from .driver import NPUDriver, NPULauncher

# Queue depth cap. Also bounds the kernarg slot pool (one slot per ring slot).
HSA_QUEUE_SIZE = 32


def _generate_hsa_launcher(constants, signature, _kernel_name) -> str:
    """Generate the C++ CPython launcher that dispatches via HSA/ROCR.

    The generated module exposes ``set_paths(pdi_path, insts_path)`` and
    ``launch(...)``. A process-global ``HsaRuntime`` singleton owns the HSA
    context (agent, pools, one size-32 queue, one completion signal, the kernarg
    slot pool, and the loaded PDI/insts). It is created when ``set_paths`` is
    first called (which also loads the binaries) and reused for the process
    lifetime.

    ``_kernel_name`` is accepted for signature parity with the XRT launcher
    generators but is unused: the HSA path selects work by PDI/insts address,
    not by a kernel symbol name.
    """
    arg_decls = ", ".join(f"{_ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
    args_format = "".join(
        [_format_of(_extracted_type(ty)) for ty in signature.values()]
    )
    fmt = "iiiOOOO" + args_format
    args_list = (
        ", " + ", ".join(f"&_arg{i}" for i, ty in signature.items())
        if len(signature) > 0
        else ""
    )

    # Pointer (tensor) args excluding constexpr constants. These become the
    # kernel arguments: N addresses followed by N sizes in bytes.
    ptr_args = [
        (i, ty) for i, ty in signature.items() if i not in constants and ty[0] == "*"
    ]
    num_ptr_args = len(ptr_args)

    size_param_decls = ", ".join(f"long size{i}" for i, ty in ptr_args)
    if size_param_decls:
        size_param_decls += ", "

    # Per-launch data buffers: acquire a vmem buffer per tensor (from the pool)
    # and copy inputs in. Buffers are held in a vector so that if any acquire
    # throws mid-way, the catch handler can return the already-acquired buffers
    # to the pool.
    alloc_and_copy_in = "\n    ".join(
        f"bufs.push_back(rt.acquire((size_t)size{i})); "
        f"std::memcpy(bufs.back().va, arg{i}, (size_t)size{i});"
        for i, _ty in ptr_args
    )
    set_kernarg_ptrs = "\n    ".join(
        f"kernargs[{pos}] = reinterpret_cast<std::uint64_t>(bufs[{pos}].va);"
        for pos in range(num_ptr_args)
    )
    set_kernarg_sizes = "\n    ".join(
        f"kernargs[NUM_KERNARGS + {pos}] = (std::uint64_t)size{i};"
        for pos, (i, _ty) in enumerate(ptr_args)
    )
    # Copy every tensor buffer back to its host pointer. We cannot know which
    # argument(s) the kernel writes (Triton does not mark outputs), so copying
    # all of them back is correct regardless of output position; buffers the
    # device did not modify just copy identical bytes. Avoids a fragile
    # "output is the last pointer arg" assumption.
    copy_out = "\n    ".join(
        f"std::memcpy(arg{i}, bufs[{pos}].va, (size_t)size{i});"
        for pos, (i, _ty) in enumerate(ptr_args)
    )

    launch_call_sizes = ", ".join(
        f"tensor_volume{i}" for i, _ty in ptr_args
    )
    if launch_call_sizes:
        launch_call_sizes += ", "
    launch_call_args = ", ".join(
        f"ptr_info{i}.dev_ptr" if ty[0] == "*" else f"_arg{i}"
        for i, ty in signature.items()
    )

    verbosity = 1 if npu_config.debug else 0

    return f"""
#include <assert.h>
#include <fstream>
#include <iostream>
#include <stdbool.h>
#include <Python.h>
#include "ExecutionEngine/CRunnerUtils.h"
#include "ExecutionEngine/CRunnerUtils.cpp"

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hsa/hsa_ext_amd_aie.h"

// Queue depth (capped). Also the kernarg slot count (one slot per ring slot).
static constexpr std::uint32_t QUEUE_SIZE = {HSA_QUEUE_SIZE};

// Number of kernel arguments (tensor pointers) for this specialized launcher.
static constexpr std::uint32_t NUM_KERNARGS = {num_ptr_args};

// Bytes in the AIE dispatch packet after completion_signal up to
// kernarg_address; the ABI requires this to be exactly 24.
static constexpr std::uint16_t AIE_PACKET_COUNT = 24;

#define HSA_CHECK(expr)                                                       \\
  do {{                                                                        \\
    hsa_status_t _s = (expr);                                                  \\
    if (_s != HSA_STATUS_SUCCESS) {{                                           \\
      const char* _m = nullptr;                                               \\
      hsa_status_string(_s, &_m);                                             \\
      throw std::runtime_error(std::string(#expr) + " failed: " +            \\
                               (_m ? _m : "unknown HSA error"));             \\
    }}                                                                         \\
  }} while (0)

namespace {{

// ---- Agent discovery -------------------------------------------------------
// Search state for collect_agents: the device type to match, and the vector
// that matching agents are appended to.
struct AgentSearch {{
  hsa_device_type_t want;
  std::vector<hsa_agent_t>* out;
}};

// hsa_iterate_agents callback: append `agent` to AgentSearch::out when its
// device type equals AgentSearch::want (`data` is the AgentSearch).
hsa_status_t collect_agents(hsa_agent_t agent, void* data) {{
  auto* s = static_cast<AgentSearch*>(data);
  hsa_device_type_t t{{}};
  hsa_status_t st = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &t);
  if (st != HSA_STATUS_SUCCESS) return st;
  if (t == s->want) s->out->push_back(agent);
  return HSA_STATUS_SUCCESS;
}}

// ---- Memory pool discovery -------------------------------------------------
// Search state for find_pool: the global-flag mask and allocatability to match,
// plus the first matching pool (with `found` set true on a hit).
struct PoolSearch {{
  hsa_amd_memory_pool_global_flag_t flags;
  bool allocatable;
  hsa_amd_memory_pool_t pool;
  bool found;
}};

// hsa_amd_agent_iterate_memory_pools callback: select the first GLOBAL pool whose
// flags include PoolSearch::flags and whose allocatability matches, then stop the
// iteration with HSA_STATUS_INFO_BREAK (`data` is the PoolSearch).
hsa_status_t find_pool(hsa_amd_memory_pool_t pool, void* data) {{
  auto* d = static_cast<PoolSearch*>(data);
  hsa_amd_segment_t seg{{}};
  if (hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg) !=
      HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;
  if (seg != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;

  hsa_amd_memory_pool_global_flag_t f{{}};
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
}}

// ---- vmem-backed data buffer ----------------------------------------------
// A tensor I/O buffer allocated through the HSA vmem API: the physical memory
// handle, the mapped virtual address, and the (granule-rounded) mapped size.
struct VmemBuffer {{
  hsa_amd_vmem_alloc_handle_t handle{{}};
  void* va = nullptr;
  std::size_t size = 0;
}};

// PDI / instruction binary loaded into the dev pool (plain HSA allocation).
struct DeviceBinary {{
  void* ptr = nullptr;
  std::size_t size = 0;
}};

// ---- Process-global HSA runtime -------------------------------------------
// Owns every persistent HSA resource this module uses: the AIE agent, the
// dev/data/kernarg memory pools, the single command queue and completion
// signal, the fixed-slot kernarg pool, the pooled vmem I/O buffers, and the
// loaded PDI/insts. One instance per process, reached via runtime().
struct HsaRuntime {{
  hsa_agent_t aie_agent{{}};
  hsa_amd_memory_pool_t dev_pool{{}};
  hsa_amd_memory_pool_t data_pool{{}};
  hsa_amd_memory_pool_t kernarg_pool{{}};
  hsa_queue_t* queue = nullptr;
  hsa_signal_t signal{{}};
  std::vector<hsa_amd_memory_access_desc_t> access_descs;  // RW access, built once
  std::size_t data_granule = 0;

  // Fixed-slot kernarg pool: one backing allocation, one slot per ring slot.
  void* kernarg_buffer = nullptr;
  std::size_t kernarg_slot_size = 0;
  std::uint32_t kernarg_slot_count = 0;

  // Artifact paths for this module's kernel and the loaded PDI / insts binaries,
  // all set once by set_paths(). Kept here so every HSA resource this module
  // uses is owned by (and observable through) this one object. This module loads
  // exactly one PDI + one insts, so no path-keyed cache is needed.
  std::string pdi_path;
  std::string insts_path;
  DeviceBinary pdi_bin{{}};
  DeviceBinary insts_bin{{}};

  // Free-list of vmem data buffers keyed by rounded size. Buffers are reused
  // across launches rather than freed: a vmem region unmapped after an AIE
  // dispatch cannot be reliably re-reserved+mapped (the device-side mapping
  // teardown makes a subsequent hsa_amd_vmem_map on a reused address fail with
  // INVALID_ARGUMENT), and reuse also removes the per-launch allocation cost.
  std::unordered_map<std::size_t, std::vector<VmemBuffer>> vmem_pool;
  std::mutex pool_mtx;

  // Construct by running the full one-time HSA initialization (see init()).
  HsaRuntime() {{ init(); }}

  // Best-effort teardown of all owned HSA resources at process exit.
  ~HsaRuntime() {{
    // Best-effort teardown. Drain in-flight work first.
    if (signal.handle) {{
      hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                                HSA_WAIT_STATE_BLOCKED);
    }}
    for (auto& kv : vmem_pool) {{
      for (auto& b : kv.second) {{
        if (!b.va) continue;
        hsa_amd_vmem_unmap(b.va, b.size);
        hsa_amd_vmem_address_free(b.va, b.size);
        hsa_amd_vmem_handle_release(b.handle);
      }}
    }}
    if (pdi_bin.ptr) hsa_amd_memory_pool_free(pdi_bin.ptr);
    if (insts_bin.ptr) hsa_amd_memory_pool_free(insts_bin.ptr);
    if (kernarg_buffer) hsa_amd_memory_pool_free(kernarg_buffer);
    if (signal.handle) hsa_signal_destroy(signal);
    if (queue) hsa_queue_destroy(queue);
    hsa_shut_down();
  }}

  // One-time setup: initialize HSA, discover the AIE + CPU agents and the
  // dev/data/kernarg pools, build the vmem access-descriptor list, create the
  // queue and completion signal, and allocate the kernarg slot pool.
  void init() {{
    HSA_CHECK(hsa_init());

    std::vector<hsa_agent_t> aies, cpus;
    AgentSearch as_aie{{HSA_DEVICE_TYPE_AIE, &aies}};
    HSA_CHECK(hsa_iterate_agents(collect_agents, &as_aie));
    AgentSearch as_cpu{{HSA_DEVICE_TYPE_CPU, &cpus}};
    HSA_CHECK(hsa_iterate_agents(collect_agents, &as_cpu));
    if (aies.empty())
      throw std::runtime_error("no HSA AIE agent found (is the NPU driver loaded?)");
    aie_agent = aies.front();
    // Every vmem I/O buffer must be RW-accessible to the CPU (host memcpy) and
    // the AIE agent (execution). Build that descriptor list once, here.
    std::vector<hsa_agent_t> access_agents;
    for (auto c : cpus) access_agents.push_back(c);
    for (auto a : aies) access_agents.push_back(a);
    access_descs.reserve(access_agents.size());
    for (auto a : access_agents) access_descs.push_back({{HSA_ACCESS_PERMISSION_RW, a}});

    // dev pool: coarse-grained, non-allocatable (PDI + instructions).
    if (!discover_pool(HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED, false,
                       &dev_pool))
      throw std::runtime_error("no dev memory pool on AIE agent");
    // data pool: coarse-grained, allocatable (tensor data via vmem).
    if (!discover_pool(HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED, true,
                       &data_pool))
      throw std::runtime_error("no data memory pool on AIE agent");
    // kernarg pool: KERNARG_INIT allocatable; fall back to the data pool.
    if (!discover_pool(HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT, true,
                       &kernarg_pool))
      kernarg_pool = data_pool;

    HSA_CHECK(hsa_amd_memory_pool_get_info(
        data_pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &data_granule));

    // Queue, capped at QUEUE_SIZE but clamped into the agent's supported range
    // (the AIE agent requires an exact size, e.g. 64, so clamp up to min_q).
    // NOTE: the AIE agent exposes QUEUES_MAX == 1, so there is one queue per
    // process. Each generated launcher .so owns its own HsaRuntime/queue, so a
    // process that dispatches more than one distinct kernel signature will fail
    // here on the second signature with a clear hsa_queue_create error. Sharing
    // a single queue across signatures would require a common runtime library.
    std::uint32_t min_q = 0, max_q = 0;
    HSA_CHECK(hsa_agent_get_info(aie_agent, HSA_AGENT_INFO_QUEUE_MIN_SIZE, &min_q));
    HSA_CHECK(hsa_agent_get_info(aie_agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &max_q));
    std::uint32_t qsize = QUEUE_SIZE;
    if (qsize < min_q) qsize = min_q;
    if (max_q > 0 && qsize > max_q) qsize = max_q;
    HSA_CHECK(hsa_queue_create(aie_agent, qsize, HSA_QUEUE_TYPE_SINGLE, nullptr,
                               nullptr, 0, 0, &queue));

    // Persistent completion signal: bumped per dispatch, waited on EQ 0.
    HSA_CHECK(hsa_signal_create(0, 0, nullptr, &signal));

    // Kernarg slot pool: one slot per ring slot, sized for this launcher's
    // worst case (N addresses + N sizes), aligned to 64 bytes.
    kernarg_slot_count = static_cast<std::uint32_t>(queue->size);
    std::size_t raw = static_cast<std::size_t>(NUM_KERNARGS) * 2 * sizeof(std::uint64_t);
    if (raw == 0) raw = sizeof(std::uint64_t);
    kernarg_slot_size = (raw + 63u) & ~static_cast<std::size_t>(63u);
    HSA_CHECK(hsa_amd_memory_pool_allocate(
        kernarg_pool, kernarg_slot_size * kernarg_slot_count, 0, &kernarg_buffer));
  }}

  // Find the first memory pool on the AIE agent matching `flags`/`allocatable`
  // (via find_pool); store it in *out and return true, or return false if none.
  bool discover_pool(hsa_amd_memory_pool_global_flag_t flags, bool allocatable,
                     hsa_amd_memory_pool_t* out) {{
    PoolSearch s{{flags, allocatable, {{}}, false}};
    hsa_amd_agent_iterate_memory_pools(aie_agent, find_pool, &s);
    if (s.found) {{
      *out = s.pool;
      return true;
    }}
    return false;
  }}

  // Slot address for ring index -- pure pointer arithmetic, no HSA call.
  void* kernarg_slot(std::uint32_t index) {{
    return static_cast<std::byte*>(kernarg_buffer) +
           static_cast<std::size_t>(index) * kernarg_slot_size;
  }}

  // Read a file into a fresh dev-pool allocation. Called twice, from set_paths.
  DeviceBinary load_binary(const std::string& path) {{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("failed to open '" + path + "'");
    std::streamsize sz = f.tellg();
    f.seekg(0);
    void* buf = nullptr;
    HSA_CHECK(hsa_amd_memory_pool_allocate(dev_pool, static_cast<std::size_t>(sz),
                                           0, &buf));
    if (!f.read(static_cast<char*>(buf), sz)) {{
      hsa_amd_memory_pool_free(buf);
      throw std::runtime_error("short read loading '" + path + "'");
    }}
    return DeviceBinary{{buf, static_cast<std::size_t>(sz)}};
  }}

  // Record the artifact paths and load the PDI + instruction binaries. Called
  // once per module load from set_paths().
  void set_paths(const char* pdi, const char* insts) {{
    pdi_path = pdi;
    insts_path = insts;
    pdi_bin = load_binary(pdi_path);
    insts_bin = load_binary(insts_path);
  }}

  // Allocate a fresh vmem buffer of at least `size` bytes (rounded up to the
  // pool granule), mapped and granted RW access to the CPU and AIE agents.
  VmemBuffer vmem_alloc(std::size_t size) {{
    std::size_t g = data_granule ? data_granule : 4096;
    size = ((size + g - 1) / g) * g;
    VmemBuffer b{{}};
    b.size = size;
    HSA_CHECK(hsa_amd_vmem_handle_create(data_pool, size, MEMORY_TYPE_PINNED, 0,
                                         &b.handle));
    HSA_CHECK(hsa_amd_vmem_address_reserve_align(&b.va, size, 0, 0,
                                                 HSA_AMD_VMEM_ADDRESS_NO_REGISTER));
    HSA_CHECK(hsa_amd_vmem_map(b.va, size, 0, b.handle, 0));
    HSA_CHECK(hsa_amd_vmem_set_access(b.va, size, access_descs.data(),
                                      access_descs.size()));
    return b;
  }}

  // Get a data buffer of at least `size` bytes: reuse a pooled buffer of the
  // matching rounded size, or allocate a new one. Reused buffers stay mapped.
  VmemBuffer acquire(std::size_t size) {{
    std::size_t g = data_granule ? data_granule : 4096;
    std::size_t rounded = ((size + g - 1) / g) * g;
    {{
      std::lock_guard<std::mutex> lock(pool_mtx);
      auto it = vmem_pool.find(rounded);
      if (it != vmem_pool.end() && !it->second.empty()) {{
        VmemBuffer b = it->second.back();
        it->second.pop_back();
        return b;
      }}
    }}
    return vmem_alloc(size);  // rounds to the same `rounded` size
  }}

  // Return a buffer to the free-list for reuse (does not unmap).
  void release(VmemBuffer& b) {{
    if (!b.va) return;
    std::lock_guard<std::mutex> lock(pool_mtx);
    vmem_pool[b.size].push_back(b);
    b.va = nullptr;
  }}
}};

HsaRuntime& runtime() {{
  // Meyers singleton: thread-safe lazy init, one HSA context per process.
  static HsaRuntime rt;
  return rt;
}}

}}  // namespace

// Python-callable set_paths(pdi_path, insts_path): record the artifact paths and
// load the PDI/insts into the runtime singleton. Called once per module load.
static PyObject* py_set_paths(PyObject* self, PyObject* args) {{
  const char* pdi;
  const char* insts;
  if (!PyArg_ParseTuple(args, "ss", &pdi, &insts)) {{
    return NULL;
  }}
  // Store the paths and resolve the binaries on the one runtime singleton.
  try {{
    runtime().set_paths(pdi, insts);
  }} catch (const std::exception& e) {{
    PyErr_SetString(PyExc_RuntimeError,
                    (std::string("HSA set_paths failed: ") + e.what()).c_str());
    return NULL;
  }}
  Py_RETURN_NONE;
}}

// HSA/ROCR AIE dispatch.
static void _launch(int gridX, int gridY, int gridZ, {size_param_decls}{arg_decls}) {{
  if (gridX * gridY * gridZ <= 0) return;
  // Declared outside the try so the catch handler can return any buffers that
  // were already acquired before a later acquire (or other step) threw.
  std::vector<VmemBuffer> bufs;
  try {{
    int verbosity = {verbosity};
    HsaRuntime& rt = runtime();
    bufs.reserve(NUM_KERNARGS);

    // PDI + instructions were resolved once by set_paths() and are owned by the
    // runtime singleton, so the hot path does no path/file/map work.
    if (rt.pdi_bin.ptr == nullptr)
      throw std::runtime_error("set_paths() was not called before launch");
    DeviceBinary& pdi = rt.pdi_bin;
    DeviceBinary& insts = rt.insts_bin;

    // Acquire vmem I/O buffers (from the pool) and copy inputs in.
    {alloc_and_copy_in}

    hsa_queue_t* q = rt.queue;

    // Reserve a ring slot; wait for a free slot if the queue is full (drains
    // completed packets). Single-producer: safe under HSA_QUEUE_TYPE_SINGLE.
    const std::uint64_t wr_idx = hsa_queue_add_write_index_relaxed(q, 1);
    while (wr_idx - hsa_queue_load_read_index_scacquire(q) >= q->size) {{
      // busy-wait for the device to consume a packet
    }}
    const std::uint64_t pkt_idx = wr_idx % q->size;

    // Each ring slot owns a fixed kernarg slot of the same index; no allocation
    // on the hot path. Layout: [addr0..addrN-1, size0..sizeN-1].
    std::uint64_t* kernargs =
        static_cast<std::uint64_t*>(rt.kernarg_slot((std::uint32_t)pkt_idx));
    {set_kernarg_ptrs}
    {set_kernarg_sizes}

    // Build the AIE dispatch packet.
    hsa_amd_aie_kernel_dispatch_packet_t pkt{{}};
    pkt.header = (HSA_AMD_AIE_PACKET_TYPE_READY << HSA_PACKET_HEADER_TYPE) |
                 (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                 (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    pkt.opcode = HSA_AMD_AIE_PACKET_OPCODE_KMQ;
    pkt.count = AIE_PACKET_COUNT;
    pkt.completion_signal = rt.signal;
    pkt.insts_addr_low = reinterpret_cast<std::uintptr_t>(insts.ptr) & 0xFFFFFFFF;
    pkt.insts_addr_high = reinterpret_cast<std::uintptr_t>(insts.ptr) >> 32;
    pkt.num_kernargs = NUM_KERNARGS;
    // The ABI requires kernarg_address to be NULL when num_kernargs is 0.
    pkt.kernarg_address = (NUM_KERNARGS > 0) ? kernargs : nullptr;
    pkt.insts_size = insts.size;
    pkt.pdi_addr = pdi.ptr;

    // Arm the completion signal to 1: the device decrements it to 0 on success,
    // or sets a negative error code on failure. Arm (and publish the packet)
    // before ringing the doorbell so the device cannot run before it is armed.
    // We wait per dispatch, so only one packet is ever in flight and its kernarg
    // slot is free before the next reservation.
    hsa_signal_store_screlease(rt.signal, 1);
    static_cast<hsa_amd_aie_kernel_dispatch_packet_t*>(q->base_address)[pkt_idx] =
        pkt;
    hsa_signal_store_screlease(q->doorbell_signal, wr_idx);

    // Wait for completion, releasing the GIL so other Python threads run during
    // device execution. Wait for value < 1 (rather than == 0) so a device error
    // (negative code) is observed instead of hanging; loop to ignore spurious
    // wake-ups that still see the armed value.
    hsa_signal_value_t sig_val = 1;
    Py_BEGIN_ALLOW_THREADS
    do {{
      sig_val = hsa_signal_wait_scacquire(rt.signal, HSA_SIGNAL_CONDITION_LT, 1,
                                          UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
    }} while (sig_val == 1);
    Py_END_ALLOW_THREADS
    if (sig_val != 0)
      throw std::runtime_error(
          "AIE dispatch failed: completion signal = " +
          std::to_string((long long)sig_val));

    if (verbosity >= 1) std::cout << "HSA dispatch complete." << std::endl;

    // Copy tensor buffers back to their host pointers.
    {copy_out}

    // Return I/O buffers to the pool for reuse (kept mapped).
    for (auto& b : bufs) rt.release(b);
  }} catch (const std::exception& e) {{
    // Return any buffers acquired before the failure so they are not leaked.
    // bufs is only non-empty once runtime() has succeeded, so this is safe even
    // if the failure was runtime() init itself (bufs is then empty).
    for (auto& b : bufs) runtime().release(b);
    std::string msg = std::string("HSA runtime error: ") + e.what();
    PyErr_SetString(PyExc_RuntimeError, msg.c_str());
  }}
}}

#include "npu_dispatch_common.h"

// Python-callable launch(gridX, gridY, gridZ, kernel_metadata, launch_metadata,
// enter_hook, exit_hook, *args): parse the arguments, run the enter/exit hooks,
// resolve each tensor's device pointer and byte size, and call _launch.
static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject* launch_enter_hook = NULL;
  PyObject* launch_exit_hook = NULL;
  PyObject* kernel_metadata = NULL;
  PyObject* launch_metadata = NULL;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if (!PyArg_ParseTuple(args, \"{fmt}\", &gridX, &gridY, &gridZ,
                        &kernel_metadata, &launch_metadata,
                        &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}

  if (launch_enter_hook != Py_None) {{
    PyObject* hook_args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, hook_args);
    Py_DECREF(hook_args);
    if (!ret) return NULL;
  }}

  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  {"; ".join([f"long tensor_volume{i} = getNumElements(_arg{i}) * getElementSizeInBytes(_arg{i}); if (tensor_volume{i} == -1) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  _launch(gridX, gridY, gridZ, {launch_call_sizes}{launch_call_args});

  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if (launch_exit_hook != Py_None) {{
    PyObject* hook_args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, hook_args);
    Py_DECREF(hook_args);
    if (!ret) return NULL;
  }}

  Py_INCREF(Py_None);
  return Py_None;
}}

// Methods exported by this per-signature dispatch extension module.
static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{"set_paths", py_set_paths, METH_VARARGS, "Set paths to aie.pdi and insts.bin"}},
  {{NULL, NULL, 0, NULL}}  // sentinel
}};

// CPython module definition for the "__npu_dispatch" extension.
static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__npu_dispatch\",
  NULL,  // documentation
  -1,    // size
  ModuleMethods
}};

// Module init entry point invoked by CPython when the extension is imported.
PyMODINIT_FUNC PyInit___npu_dispatch(void) {{
  PyObject* m = PyModule_Create(&ModuleDef);
  if (m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""


class HSALauncher(NPULauncher):
    """Launcher that JIT-compiles an HSA/ROCR host dispatcher for a kernel.

    Subclasses :class:`NPULauncher` so ``get_npu_cache_dir`` and the ``__call__``
    protocol keep working, but forces the ``pdi`` output format and the ``hsa``
    link profile.
    """

    def __init__(self, src, metadata):
        # Intentionally does NOT call super().__init__(): NPULauncher.__init__
        # detects the XRT output format and rejects runtime == "hsa". This
        # subclass fixes the format to "pdi" and the link profile to "hsa", then
        # reuses the parent's compile/caching tail via _finalize().
        constants, signature = _extract_signature_and_constants(src)
        self.output_format = "pdi"
        launcher_src = _generate_hsa_launcher(
            constants, signature, self.kernel_placeholder_name
        )
        self._finalize(src, launcher_src, link_profile="hsa")


class HSADriver(NPUDriver):
    """Triton driver that dispatches NPU kernels through HSA/ROCR.

    Activate explicitly::

        import triton
        from triton.backends.amd_triton_npu.hsa_driver import HSADriver
        triton.runtime.driver.set_active(HSADriver())

    or select the runtime via ``AMD_TRITON_NPU_RUNTIME=hsa`` /
    ``npu_config.runtime = "hsa"`` (which also drives ``is_active``).
    """

    def __init__(self):
        # Reuse NPUDriver's setup but dispatch through the HSA launcher.
        super().__init__()
        self.launcher_cls = HSALauncher

    @staticmethod
    def is_active():
        # Note: Triton's backend loader only discovers the driver in driver.py,
        # so it never auto-selects HSADriver via is_active(); activation is always
        # explicit (triton.runtime.driver.set_active(HSADriver())). This reports
        # whether HSA is the configured runtime for callers that ask directly.
        return npu_config.runtime == "hsa"
