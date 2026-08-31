// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// C ABI for the shared HSA/ROCR dispatch runtime.
//
// The implementation (HsaRuntime.cpp) is compiled once into a single shared
// library (libtriton_npu_hsa.so) that every generated per-signature launcher
// links against. Because the dynamic linker loads that shared dependency once
// per process, the HsaRuntime singleton behind these functions is truly
// process-global: one hsa_init, one queue, one completion signal, one kernarg
// pool, one vmem buffer pool, shared across all kernel signatures.
//
// This header exposes only opaque handles and plain C types -- no hsa/*.h -- so
// the launcher never needs the ROCR headers; only HsaRuntime.cpp does.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum number of tensor kernel arguments a single dispatch may carry.
// triton_npu_hsa_dispatch() returns an error if num_tensors exceeds this.
// Sized well above the largest observed design (9 arguments); the only cost of
// headroom is the kernarg slot pool, which is MAX_KERNARGS * 16 bytes per queue
// ring slot (a few KiB in total).
#define TRITON_NPU_HSA_MAX_KERNARGS 16

// Opaque handle to a prepared (pdi, insts) program, owned by the runtime.
typedef struct triton_npu_hsa_program *triton_npu_hsa_program_t;

// Initialize the runtime (idempotent) and write the AIE agent's device name
// into buf, NUL-terminated -- "aie2" on npu1 (Phoenix), "aie2p" on npu2
// (Strix). Returns 0 on success, or a negative value on error (with a message
// written to errbuf). Lets a caller identify the NPU generation straight from
// the HSA agent, without shelling out to xrt-smi.
int triton_npu_hsa_agent_name(char *buf, size_t buf_len, char *errbuf,
                              size_t errbuf_len);

// Initialize the runtime (idempotent) and load + cache the PDI and instruction
// binaries for one kernel. Returns an opaque program handle, or NULL on error
// (with a message written to errbuf). Call once per launcher module.
triton_npu_hsa_program_t triton_npu_hsa_prepare(const char *pdi_path,
                                                const char *insts_path,
                                                char *errbuf,
                                                size_t errbuf_len);

// Dispatch a prepared program: acquire vmem I/O buffers, copy inputs in, fill
// kernargs, enqueue the AIE packet, wait for completion, copy outputs back, and
// return the buffers to the pool. host_ptrs[i]/sizes[i] describe tensor i (i in
// [0, num_tensors)). Returns 0 on success, or a negative value on error (with a
// message written to errbuf). num_tensors must be <=
// TRITON_NPU_HSA_MAX_KERNARGS.
//
// A tensor whose pointer falls inside a shared region (see the shared_* group
// below) skips all of that: it is already memory the AIE agent can reach, so it
// is dispatched on in place, with no staging buffer and neither copy. Both
// halves of one dispatch may be mixed freely.
//
// Setting AMD_TRITON_NPU_HSA_TIMEOUT to a number of seconds (fractional
// accepted) arms a watchdog on the runtime's internal waits. Two caveats:
//
// * It is off by default because recovering from a timeout means permanently
//   abandoning the completion signal and the I/O buffers of the timed-out
//   dispatch -- the device may still write to them -- so a timeout set too low
//   for a legitimately slow kernel leaks memory on every launch.
// * It does NOT currently make a hung dispatch recoverable. Ringing the AIE
//   doorbell submits synchronously, so the whole dispatch happens inside a
//   ROCR call that takes no timeout; the waits the watchdog does bound are
//   microseconds long today. See the note on TIMEOUT_ENV in HsaRuntime.cpp.
int triton_npu_hsa_dispatch(triton_npu_hsa_program_t program,
                            uint32_t num_tensors, void *const *host_ptrs,
                            const uint64_t *sizes, char *errbuf,
                            size_t errbuf_len);

// ---------------------------------------------------------------------------
// Shared regions
// ---------------------------------------------------------------------------
// Memory that both the AIE agent and someone else -- in practice the iGPU, via
// HIP -- can address, so a buffer handed from one to the other never moves.
// The runtime keeps a table of them; a dispatch that names an address inside a
// registered region runs on it in place (see triton_npu_hsa_dispatch above).
//
// The table is keyed by *every address a caller may name the region by*, which
// is why aliases exist: the same pages have one address per runtime holding
// them, and a framework tensor carries whichever one its own runtime handed
// out. All of them map to the single AIE-side address the packet needs.
//
// Freeing is by any registered address, and a region must outlive every
// dispatch naming it -- the caller owns that lifetime, exactly as it owns the
// tensors it dispatches on.

// Allocate `size` bytes reachable by both the CPU and the AIE agent, through
// the vmem API, and register the result as a shared region. Returns the address
// (valid for both), or NULL on error (with a message written to errbuf).
void *triton_npu_hsa_shared_alloc(uint64_t size, char *errbuf,
                                  size_t errbuf_len);

// Map `size` bytes at `ptr` -- memory some other agent on this system owns, in
// practice an iGPU allocation -- for the AIE agent, and register the result as
// a shared region reachable at `ptr` as well as at the address returned.
// Returns the AIE-side address, or NULL on error (with a message written to
// errbuf).
//
// `ptr` must name memory ROCR knows about and can export; pinned host memory
// cannot be exported and is rejected. The dma-buf that carries it across is an
// implementation detail here, handled and closed inside this call.
//
// The imported range is granted to the AIE agents only. Whether the CPU can
// reach it is a property of the memory the owner allocated, not something this
// grant can add -- ROCR rejects a CPU grant on an imported range outright.
void *triton_npu_hsa_shared_import(void *ptr, uint64_t size, char *errbuf,
                                   size_t errbuf_len);

// Register `alias` as another address for the region reachable at `va`, which
// must already be registered and at least `size` bytes long. Returns 0 on
// success, or a negative value on error (with a message written to errbuf).
int triton_npu_hsa_shared_alias(void *alias, void *va, uint64_t size,
                                char *errbuf, size_t errbuf_len);

// Drop one alias, leaving the region and its other addresses in place. Retiring
// an alias before the mapping behind it goes away is what keeps a dispatch from
// resolving an address that is no longer the caller's. Unregistered aliases are
// ignored. Returns 0, or a negative value on error.
int triton_npu_hsa_shared_unalias(void *alias, char *errbuf, size_t errbuf_len);

// Release the region reachable at `va` (any of its registered addresses) and
// forget every address that named it. Unregistered addresses are ignored, so
// this is idempotent. Returns 0 on success, or a negative value on error.
int triton_npu_hsa_shared_free(void *va, char *errbuf, size_t errbuf_len);

// Tensor arguments dispatched since the process started, split by how they got
// to the device: `in_place` were in a shared region, `staged` were copied
// through a pooled buffer. Either pointer may be NULL.
//
// Whether a buffer is *actually* being dispatched on where it lives is not
// otherwise observable -- both paths produce the same answer, one of them
// twice as slowly -- so this is how a caller checks that sharing is doing
// anything, and how the tests check it.
//
// The only entry point here that neither initializes the runtime nor can fail:
// asking what has been dispatched must not itself claim the device, since a
// process that never dispatches through HSA may still ask.
void triton_npu_hsa_dispatch_counts(uint64_t *in_place, uint64_t *staged);

#ifdef __cplusplus
} // extern "C"
#endif
