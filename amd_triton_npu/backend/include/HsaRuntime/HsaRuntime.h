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

#ifdef __cplusplus
} // extern "C"
#endif
