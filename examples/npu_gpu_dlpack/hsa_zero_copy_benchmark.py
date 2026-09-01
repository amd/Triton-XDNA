#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Hand a tensor from the iGPU to the NPU under HSA, with and without copies.

The companion to ``zero_copy_benchmark.py``, which measures the same hand-off
through XRT. Both sides of the boundary differ here, which is why it is a
separate file:

* the NPU is driven through ROCR, so there is no ``xrt::bo`` to bind and no
  multi-launch ELF -- a kernel is dispatched the ordinary Triton way, and the
  runtime decides for itself whether an operand can be used where it lies;
* an operand that cannot be is *staged*: ``HsaRuntime::dispatch`` copies it into
  a pooled vmem buffer before the launch and back out after it. That copy is
  invisible from Python, it happens on every launch, and removing it is what a
  shared buffer is for. ``shared.hsa_dispatch_counts()`` reports how many
  operands took each path, so the claim is checkable without a stopwatch.

The pipeline crosses the boundary twice::

    A   = X @ Y            on the iGPU
    C   = A @ B            on the NPU
    (C consumed)           on the iGPU

Three variants of it are timed:

**staged**  -- ordinary tensors. ``A`` comes back to the host, the runtime
stages every operand in and out around the launch, and the result returns to
the iGPU.

**NPU-owned** -- a shared buffer allocated by the NPU (``device="hsa:0"``) that
the iGPU maps. Nothing crosses and nothing is staged.

**iGPU-owned** -- a shared buffer allocated by the iGPU
(``device="hip:0", share="hsa:0"``), which is a native ``hipMalloc``
allocation the NPU imports. Also zero-copy, and the interesting comparison
against the one above: whether it matters *which* device's memory the pages
are, once neither is copying them.

Run it::

    source scripts/hsa-env.sh
    python examples/npu_gpu_dlpack/hsa_zero_copy_benchmark.py

Exits 77 -- which scripts/run_tests.py grades as a skip -- on a host without
an iGPU, a ROCm build of torch, or an AIE-capable ROCR.
"""

from __future__ import annotations

import os
import sys
import time

import torch
import triton
import triton.language as tl

from triton.backends.amd_triton_npu import shared
from triton.backends.amd_triton_npu.config import npu_config
from triton.backends.amd_triton_npu.driver import NPUDriver, detect_npu_version
from triton.backends.amd_triton_npu.shared import SharedBufferError

# The phase table is the same measurement as the XRT benchmark's, so it is the
# same code: the two files are meant to be read side by side, and a column that
# means something subtly different in each would defeat that.
from zero_copy_benchmark import Phase, SKIP_EXIT_CODE, _sync, report

#: Square, and one 256-wide tile per dimension. The kernel loads the whole K
#: extent in a block, so this is not a knob to turn casually: a larger K has to
#: fit in AIE local memory, and a larger M or N wants a tiling script written
#: for it.
SIZE = 256
ITERS = 50
WARMUP = 5

#: bf16 in, f32 out -- what the AIE matmul does, and what the tiling script
#: below was written for.
IN_DTYPE, OUT_DTYPE = torch.bfloat16, torch.float32
IN_BYTES = SIZE * SIZE * 2
OUT_BYTES = SIZE * SIZE * 4


@triton.jit
def _matmul_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """One tile of C = A @ B; the kernel examples/hsa_matmul dispatches."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b = tl.load(B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    tl.store(
        C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, tl.dot(a, b)
    )


def _grid(meta):
    return (
        triton.cdiv(SIZE, meta["BLOCK_SIZE_M"]),
        triton.cdiv(SIZE, meta["BLOCK_SIZE_N"]),
    )


def _dispatch(a, b, c):
    """C = A @ B on the NPU, on whatever three operands are handed in."""
    _matmul_kernel[_grid](
        a,
        b,
        c,
        SIZE,
        SIZE,
        SIZE,
        SIZE,
        1,
        SIZE,
        1,
        SIZE,
        1,
        BLOCK_SIZE_M=256,
        BLOCK_SIZE_N=256,
        BLOCK_SIZE_K=SIZE,
    )


def run_staged(x, y, b_host, iters):
    """The default path: the host sees every operand, twice per launch."""
    ph = Phase()
    result = None
    for i in range(iters):
        _sync()
        t0 = time.perf_counter()
        a = torch.matmul(x, y)
        _sync()
        t1 = time.perf_counter()

        # iGPU -> host, so the dispatch has something it can name. The runtime
        # then stages this again, into a vmem buffer and back, on every launch.
        a_host = a.cpu()
        c_host = torch.zeros(SIZE, SIZE, dtype=OUT_DTYPE)
        t2 = time.perf_counter()

        _dispatch(a_host, b_host, c_host)
        t3 = time.perf_counter()

        # host -> iGPU, so the next stage can use it.
        result = c_host.to("cuda")
        _sync()
        t4 = time.perf_counter()

        ph.add("igpu matmul", (t1 - t0) * 1e3)
        ph.add("hand-off out (D2H)", (t2 - t1) * 1e3, IN_BYTES)
        ph.add("npu dispatch", (t3 - t2) * 1e3)
        ph.add("hand-off back (H2D)", (t4 - t3) * 1e3, OUT_BYTES)
    return ph, result


def run_shared(x, y, a_buf, b_buf, c_buf, iters):
    """Zero-copy: the iGPU writes where the NPU reads, either way round."""
    ph = Phase()
    a_view, c_view = a_buf.torch(), c_buf.torch()
    result = None
    for i in range(iters):
        _sync()
        t0 = time.perf_counter()
        # out= writes the matmul straight into the NPU's input buffer.
        torch.matmul(x, y, out=a_view)
        _sync()
        t1 = time.perf_counter()

        # No transfer -- only a fence, so the NPU does not read pages the iGPU
        # has not finished writing. See the same phase in zero_copy_benchmark
        # for why it is kept even though the drain above has already paid it.
        torch.cuda.current_stream().synchronize()
        t2 = time.perf_counter()

        _dispatch(a_view, b_buf.torch(), c_view)
        t3 = time.perf_counter()

        # Already an iGPU tensor over the pages the NPU just wrote.
        result = c_view
        _sync()
        t4 = time.perf_counter()

        ph.add("igpu matmul", (t1 - t0) * 1e3)
        ph.add("hand-off out (fence)", (t2 - t1) * 1e3, 0)
        ph.add("npu dispatch", (t3 - t2) * 1e3)
        ph.add("hand-off back (none)", (t4 - t3) * 1e3, 0)
    return ph, result


def _counted(fn, *args):
    """Run a variant and report what the dispatcher did with its operands."""
    before = shared.hsa_dispatch_counts()
    ph, result = fn(*args)
    after = shared.hsa_dispatch_counts()
    return ph, result, (after[0] - before[0], after[1] - before[1])


def main() -> int:
    """Run the three variants, check they agree, and print the comparison."""
    print("=" * 64)
    print("  iGPU matmul -> NPU matmul, through HSA: staged vs shared")
    print("=" * 64)
    if not torch.cuda.is_available():
        print("  SKIP: no ROCm device visible to torch")
        return SKIP_EXIT_CODE
    try:
        triton.runtime.driver.set_active(NPUDriver("hsa"))
        npu_config.transform_tiling_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "matmul_bf16_m64_n64_k64",
            f"transform_{'aie2' if detect_npu_version() == 'npu1' else 'aie2p'}.mlir",
        )
    except Exception as e:
        print(f"  SKIP: the HSA runtime is unavailable: {e}")
        return SKIP_EXIT_CODE

    print(f"  device   : {torch.cuda.get_device_name(0)}")
    print(f"  matmul   : ({SIZE},{SIZE}) @ ({SIZE},{SIZE}) -> ({SIZE},{SIZE})")
    print(
        f"  operands : {IN_BYTES / 1024:.0f} KB in (bf16), "
        f"{OUT_BYTES / 1024:.0f} KB out (f32)"
    )
    print(f"  iters    : {ITERS} (plus {WARMUP} warmup)")

    x = torch.randn(SIZE, SIZE, dtype=IN_DTYPE, device="cuda")
    y = torch.randn(SIZE, SIZE, dtype=IN_DTYPE, device="cuda")
    b_gpu = torch.randn(SIZE, SIZE, dtype=IN_DTYPE, device="cuda")
    _sync()
    reference = torch.matmul(torch.matmul(x, y).float(), b_gpu.float())
    _sync()

    # Allocating is also the availability check, so there is no separate probe.
    owned = {"NPU-owned": ("hsa:0", "hip:0"), "iGPU-owned": ("hip:0", "hsa:0")}
    buffers = {}
    try:
        for label, (primary, secondary) in owned.items():
            trio = [
                shared.empty(
                    SIZE, SIZE, dtype=IN_DTYPE, device=primary, share=secondary
                ),
                shared.empty(
                    SIZE, SIZE, dtype=IN_DTYPE, device=primary, share=secondary
                ),
                shared.zeros(
                    SIZE, SIZE, dtype=OUT_DTYPE, device=primary, share=secondary
                ),
            ]
            trio[1].torch().copy_(b_gpu)
            buffers[label] = trio
    except SharedBufferError as e:
        for trio in buffers.values():
            for buf in trio:
                buf.close()
        print(f"  SKIP: shared NPU/iGPU buffers unavailable: {e}")
        return SKIP_EXIT_CODE

    b_host = b_gpu.cpu()
    try:
        print("\n  warming up (JIT, PDI build, first dispatch)...")
        run_staged(x, y, b_host, WARMUP)
        for trio in buffers.values():
            run_shared(x, y, *trio, WARMUP)

        results = {}
        moved = {}
        ph, res, moved["staged"] = _counted(run_staged, x, y, b_host, ITERS)
        results["staged (via host)"] = (ph, res)
        for label, trio in buffers.items():
            ph, res, moved[label] = _counted(run_shared, x, y, *trio, ITERS)
            results[f"shared, {label} pages"] = (ph, res)

        for label, (ph, _) in results.items():
            report(label, ph, ITERS)

        errors = {
            label: float((res.float().cpu() - reference.cpu()).abs().max())
            for label, (_, res) in results.items()
        }
        print(
            "\n  vs torch reference (bf16 inputs, f32 accumulate): "
            + ", ".join(f"{label} {err:.2f}" for label, err in errors.items())
        )

        staged_ms = sum(results["staged (via host)"][0].per_iter(ITERS)[0].values())
        print("\n" + "=" * 64)
        print(
            f"  {'variant':<26} {'total ms':>9} {'host KB':>9} "
            f"{'in place':>9} {'staged':>7}"
        )
        for label, (ph, _) in results.items():
            ms, by = ph.per_iter(ITERS)
            key = "staged" if label.startswith("staged") else label.split(", ")[1][:-6]
            total = sum(ms.values())
            print(
                f"  {label:<26} {total:>9.3f} {sum(by.values()) / 1024:>9.0f} "
                f"{moved[key][0] // ITERS:>9} {moved[key][1] // ITERS:>7}"
            )
        print(f"\n  Every operand of a shared launch is dispatched where it lies;")
        print(f"  a staged one is copied into a pooled buffer and back, per launch,")
        print(
            f"  on top of the {(IN_BYTES + OUT_BYTES) / 1024:.0f} KB that crossed the host to get there."
        )
        print("=" * 64)

        # bf16 inputs accumulated in f32 against an f32 reference: this is a
        # "did the NPU compute this" tolerance, not a numerical claim.
        return 0 if max(errors.values()) < 20.0 else 1
    finally:
        for trio in buffers.values():
            for buf in trio:
                buf.close()


if __name__ == "__main__":
    sys.exit(main())
