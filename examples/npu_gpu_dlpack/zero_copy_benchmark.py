#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Hand a tensor from the iGPU to the NPU, with and without copies.

The pipeline crosses the boundary twice, and what the NPU does in the middle
depends on how it is being driven::

    A   = X @ Y            on the iGPU
    C   = f(A)             on the NPU
    (C consumed)           on the iGPU

``--runtime xrt`` (the default) dispatches a two-op add chain through a
multi-launch ELF, binding an ``xrt::bo`` per operand. ``--runtime hsa``
dispatches a matmul through ROCR the ordinary Triton way, where there is no BO
to bind and the runtime decides for itself whether each operand can be used
where it lies. Both are the same measurement of the same boundary; they are in
one file so the numbers are produced by one harness rather than two.

The variants, per runtime:

**copy / staged** -- what the boundary costs by default. ``A`` comes back to
the host and the result returns to the iGPU. Under XRT the launch stages each
operand into a BO and back; under HSA the runtime does the same into a pooled
vmem buffer, invisibly, on every launch.

**shared** -- the iGPU writes ``A`` *into* the buffer the NPU will read, the
dispatch names that buffer, and the result is already an iGPU tensor. Nothing
crosses. Under HSA this is measured twice, once with the NPU owning the pages
and once with the iGPU owning them -- the latter being a native ``hipMalloc``
allocation the NPU imports, which answers whether it matters who allocates.

Both are timed per phase, not just end to end. The two variants differ only in
the hand-off, and the hand-off is the small term next to the NPU dispatch that
surrounds it, so a whole-pipeline number mixes the part that changed with a
much larger part that did not. The per-phase rows isolate it, and the byte
columns say the same thing without depending on timing at all. Under HSA the
operand counts from ``shared.hsa_dispatch_counts()`` say it without depending
on the clock either.

Run it::

    source scripts/hsa-env.sh
    python examples/npu_gpu_dlpack/zero_copy_benchmark.py
    python examples/npu_gpu_dlpack/zero_copy_benchmark.py --runtime hsa
    python examples/npu_gpu_dlpack/zero_copy_benchmark.py --rows 512 --cols 2048 -k 1024

Exits 77 -- which scripts/run_tests.py grades as a skip -- on a host without an
iGPU, a ROCm build of torch, or the NPU runtime asked for.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import triton
import triton.language as tl

import add_chain
from triton.backends.amd_triton_npu import shared
from triton.backends.amd_triton_npu.config import npu_config
from triton.backends.amd_triton_npu.driver import NPUDriver, detect_npu_version
from triton.backends.amd_triton_npu.multilaunch import NPUChain
from triton.backends.amd_triton_npu.shared import SharedBuffer, SharedBufferError

#: Told to the harness when the environment cannot run this at all -- no iGPU,
#: no ROCm build of torch, no NPU runtime. Distinct from a failure, which is
#: what any other non-zero status means.
SKIP_EXIT_CODE = 77

#: Shapes each runtime is known to work at. XRT's are free within the element
#: count rule below; HSA's are the ones its tiling script was written for.
DEFAULT_SHAPE = {"xrt": (128, 768, 512), "hsa": (256, 256, 256)}

#: The tile the HSA matmul is dispatched in, fixed by that tiling script.
HSA_TILE = 256

# The chain computes OUT = (C + E1) + E2. C is rewritten every iteration;
# both addends are held fixed, so they stage once per BO set.
I_C, I_E1, I_TMP, I_E2, I_OUT = (
    add_chain.I_A,
    add_chain.I_B,
    add_chain.I_TMP,
    add_chain.I_ADDEND,
    add_chain.I_OUT,
)
STATIC = {I_E1, I_E2}


class Phase:
    """Accumulates per-phase wall time and host-crossing bytes over a run."""

    def __init__(self) -> None:
        self.ms: dict[str, float] = {}
        self.bytes: dict[str, int] = {}

    def add(self, name: str, ms: float, nbytes: int = 0) -> None:
        self.ms[name] = self.ms.get(name, 0.0) + ms
        self.bytes[name] = self.bytes.get(name, 0) + nbytes

    def per_iter(self, iters: int) -> tuple[dict[str, float], dict[str, int]]:
        return (
            {k: v / iters for k, v in self.ms.items()},
            {k: v // iters for k, v in self.bytes.items()},
        )


def _sync() -> None:
    """Drain the iGPU queue so a timestamp measures execution, not enqueue."""
    torch.cuda.synchronize()


def report(label: str, ph: Phase, iters: int) -> tuple[float, float, int]:
    ms, by = ph.per_iter(iters)
    print(f"\n  {label}")
    print(f"    {'phase':<24} {'ms/iter':>9} {'host bytes':>12}")
    print(f"    {'-'*24} {'-'*9} {'-'*12}")
    for k in ms:
        b = f"{by[k]/1024:.0f} KB" if by[k] else "-"
        print(f"    {k:<24} {ms[k]:>9.3f} {b:>12}")
    total_ms = sum(ms.values())
    total_b = sum(by.values())
    handoff = sum(v for k, v in ms.items() if k.startswith("hand-off"))
    print(f"    {'-'*24} {'-'*9} {'-'*12}")
    print(
        f"    {'TOTAL':<24} {total_ms:>9.3f} "
        f"{(str(int(total_b/1024)) + ' KB') if total_b else '0':>12}"
    )
    return total_ms, handoff, total_b


# ---------------------------------------------------------------------------
# XRT: an add chain dispatched through a multi-launch ELF, on bound BOs
# ---------------------------------------------------------------------------
def run_copy(chain, a, b, shape, e1_host, e2_host, iters):
    """The default path: every hand-off goes through host memory."""
    rows, cols, _ = shape
    n = rows * cols
    out_host: np.ndarray = np.zeros(n, dtype=np.float32)
    tmp_host: np.ndarray = np.zeros(n, dtype=np.float32)
    ph = Phase()
    result = None
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        c = torch.matmul(a, b)
        _sync()
        t1 = time.perf_counter()

        # iGPU -> host. The whole tensor crosses. .cpu() already allocates the
        # host buffer and does the transfer, so taking its array directly is one
        # copy -- staging it into a reused array as well would put a second,
        # host-to-host copy inside the very phase this is measuring.
        c_host = c.reshape(-1).cpu().numpy()
        t2 = time.perf_counter()

        # The launch stages c_host into a BO, runs, and stages the result back.
        res = chain.run(
            [c_host, e1_host, tmp_host, e2_host, out_host],
            bo_key="copy",
            static_indices=STATIC,
            intermediate_indices=add_chain.INTERMEDIATE,
            output_indices=add_chain.OUTPUT,
        )
        t3 = time.perf_counter()

        # host -> iGPU, so the next stage can use it.
        result = torch.from_numpy(res[I_OUT]).to("cuda").reshape(rows, cols)
        _sync()
        t4 = time.perf_counter()

        ph.add("igpu matmul", (t1 - t0) * 1e3)
        ph.add("hand-off out (D2H)", (t2 - t1) * 1e3, n * 4)
        ph.add("npu dispatch", (t3 - t2) * 1e3)
        ph.add("hand-off back (H2D)", (t4 - t3) * 1e3, n * 4)
    return ph, result


def run_shared_xrt(chain, a, b, shape, c_buf, out_buf, e1_host, e2_host, iters):
    """Zero-copy: the iGPU writes where the NPU reads."""
    rows, cols, _ = shape
    n = rows * cols
    tmp_host: np.ndarray = np.zeros(n, dtype=np.float32)
    c_view = c_buf.torch().view(rows, cols)
    ph = Phase()
    result = None
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        # out= writes the matmul straight into the NPU's input buffer.
        torch.matmul(a, b, out=c_view)
        _sync()
        t1 = time.perf_counter()

        # No transfer -- only a fence, so the NPU does not read pages the iGPU
        # has not finished writing.
        #
        # It costs nothing here because the _sync() that timed the matmul has
        # already drained the queue, and it is kept anyway: the copy path is
        # measured from that same drained state, so the two hand-off phases are
        # comparable, and a program that is not being timed still needs this
        # fence. Removing it would make the phase read 0 for the wrong reason
        # and leave the example wrong to copy.
        torch.cuda.current_stream().synchronize()
        t2 = time.perf_counter()

        chain.run(
            [c_buf.numpy(), e1_host, tmp_host, e2_host, out_buf.numpy()],
            bo_key="shared",
            static_indices=STATIC,
            intermediate_indices=add_chain.INTERMEDIATE,
            output_indices=add_chain.OUTPUT,
            bound_buffers={I_C: c_buf.bo, I_OUT: out_buf.bo},
        )
        t3 = time.perf_counter()

        # Already an iGPU tensor over the pages the NPU just wrote.
        result = out_buf.torch().view(rows, cols)
        _sync()
        t4 = time.perf_counter()

        ph.add("igpu matmul", (t1 - t0) * 1e3)
        ph.add("hand-off out (fence)", (t2 - t1) * 1e3, 0)
        ph.add("npu dispatch", (t3 - t2) * 1e3)
        ph.add("hand-off back (none)", (t4 - t3) * 1e3, 0)
    return ph, result


def bench_xrt(args):
    """Set up and run both XRT variants; returns (results, reference)."""
    rows, cols, k = args.shape
    n = rows * cols
    a = torch.randn(rows, k, dtype=torch.float32, device="cuda")
    b = torch.randn(k, cols, dtype=torch.float32, device="cuda")
    _sync()
    # The two NPU-side addends. Static operands, so they are staged once per BO
    # set and are not part of the per-iteration hand-off either way.
    e1_host: np.ndarray = np.full(n, 0.5, dtype=np.float32)
    e2_host: np.ndarray = np.full(n, 0.25, dtype=np.float32)
    reference = (torch.matmul(a, b).reshape(-1) + 0.75).reshape(rows, cols)
    _sync()

    chain = add_chain.build("zero_copy_bench_add", n)
    buffers = []
    try:
        # Shared buffers: allocated and registered once, reused every iteration.
        # Allocating is also the availability check, so there is no separate
        # probe. XRT owns the pages (the NPU dispatch names the BO); the iGPU
        # borrows them, which is what lets torch.matmul write straight into the
        # input.
        c_buf = shared.empty(n, dtype=torch.float32, device="xrt:0", share="hip:0")
        buffers.append(c_buf)
        out_buf = shared.empty_like(c_buf)
        buffers.append(out_buf)

        print("\n  warming up (JIT, ELF build, BO allocation)...")
        run_copy(chain, a, b, args.shape, e1_host, e2_host, args.warmup)
        run_shared_xrt(
            chain, a, b, args.shape, c_buf, out_buf, e1_host, e2_host, args.warmup
        )

        results = {}
        ph, res = run_copy(chain, a, b, args.shape, e1_host, e2_host, args.iters)
        results["copy (via host)"] = (ph, res, None)
        ph, res = run_shared_xrt(
            chain, a, b, args.shape, c_buf, out_buf, e1_host, e2_host, args.iters
        )
        results["shared (zero-copy)"] = (ph, res, None)
        return results, reference, buffers, chain
    except Exception:
        for buf in buffers:
            buf.close()
        chain.close()
        raise


# ---------------------------------------------------------------------------
# HSA: a matmul dispatched through ROCR, where the runtime stages for itself
# ---------------------------------------------------------------------------
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


def _dispatch_hsa(a, b, c, shape):
    """C = A @ B on the NPU, on whatever three operands are handed in."""
    rows, cols, k = shape
    grid = lambda meta: (  # noqa: E731 - Triton's own idiom for a grid
        triton.cdiv(rows, meta["BLOCK_SIZE_M"]),
        triton.cdiv(cols, meta["BLOCK_SIZE_N"]),
    )
    _matmul_kernel[grid](
        a,
        b,
        c,
        rows,
        cols,
        k,
        k,
        1,
        cols,
        1,
        cols,
        1,
        BLOCK_SIZE_M=HSA_TILE,
        BLOCK_SIZE_N=HSA_TILE,
        BLOCK_SIZE_K=k,
    )


def run_staged(x, y, b_host, shape, iters):
    """The default path: the host sees every operand, twice per launch."""
    rows, cols, k = shape
    ph = Phase()
    result = None
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        a = torch.matmul(x, y)
        _sync()
        t1 = time.perf_counter()

        # iGPU -> host, so the dispatch has something it can name. The runtime
        # then stages this again, into a vmem buffer and back, every launch.
        a_host = a.cpu()
        c_host = torch.zeros(rows, cols, dtype=torch.float32)
        t2 = time.perf_counter()

        _dispatch_hsa(a_host, b_host, c_host, shape)
        t3 = time.perf_counter()

        # host -> iGPU, so the next stage can use it.
        result = c_host.to("cuda")
        _sync()
        t4 = time.perf_counter()

        ph.add("igpu matmul", (t1 - t0) * 1e3)
        ph.add("hand-off out (D2H)", (t2 - t1) * 1e3, rows * k * 2)
        ph.add("npu dispatch", (t3 - t2) * 1e3)
        ph.add("hand-off back (H2D)", (t4 - t3) * 1e3, rows * cols * 4)
    return ph, result


def run_shared_hsa(x, y, trio, shape, iters):
    """Zero-copy: the iGPU writes where the NPU reads, either way round."""
    a_buf, b_buf, c_buf = trio
    a_view, b_view, c_view = a_buf.torch(), b_buf.torch(), c_buf.torch()
    ph = Phase()
    result = None
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        torch.matmul(x, y, out=a_view)
        _sync()
        t1 = time.perf_counter()

        # See run_shared_xrt for why the fence is kept.
        torch.cuda.current_stream().synchronize()
        t2 = time.perf_counter()

        _dispatch_hsa(a_view, b_view, c_view, shape)
        t3 = time.perf_counter()

        result = c_view
        _sync()
        t4 = time.perf_counter()

        ph.add("igpu matmul", (t1 - t0) * 1e3)
        ph.add("hand-off out (fence)", (t2 - t1) * 1e3, 0)
        ph.add("npu dispatch", (t3 - t2) * 1e3)
        ph.add("hand-off back (none)", (t4 - t3) * 1e3, 0)
    return ph, result


def bench_hsa(args):
    """Set up and run the staged and both shared HSA variants."""
    rows, cols, k = args.shape
    triton.runtime.driver.set_active(NPUDriver("hsa"))
    npu_config.transform_tiling_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "matmul_bf16_m64_n64_k64",
        f"transform_{'aie2' if detect_npu_version() == 'npu1' else 'aie2p'}.mlir",
    )

    x = torch.randn(rows, k, dtype=torch.bfloat16, device="cuda")
    y = torch.randn(k, k, dtype=torch.bfloat16, device="cuda")
    b_gpu = torch.randn(k, cols, dtype=torch.bfloat16, device="cuda")
    _sync()
    reference = torch.matmul(torch.matmul(x, y).float(), b_gpu.float())
    _sync()

    owned = {"NPU-owned": ("hsa:0", "hip:0"), "iGPU-owned": ("hip:0", "hsa:0")}
    buffers: list[SharedBuffer] = []
    try:
        trios = {}
        for label, (primary, secondary) in owned.items():
            trio = (
                shared.empty(
                    rows, k, dtype=torch.bfloat16, device=primary, share=secondary
                ),
                shared.empty(
                    k, cols, dtype=torch.bfloat16, device=primary, share=secondary
                ),
                shared.zeros(
                    rows, cols, dtype=torch.float32, device=primary, share=secondary
                ),
            )
            buffers.extend(trio)
            trio[1].torch().copy_(b_gpu)
            trios[label] = trio
        _sync()

        b_host = b_gpu.cpu()
        print("\n  warming up (JIT, PDI build, first dispatch)...")
        run_staged(x, y, b_host, args.shape, args.warmup)
        for trio in trios.values():
            run_shared_hsa(x, y, trio, args.shape, args.warmup)

        results = {}
        before = shared.hsa_dispatch_counts()
        ph, res = run_staged(x, y, b_host, args.shape, args.iters)
        after = shared.hsa_dispatch_counts()
        results["staged (via host)"] = (ph, res, _per_iter_counts(before, after, args))
        for label, trio in trios.items():
            before = shared.hsa_dispatch_counts()
            ph, res = run_shared_hsa(x, y, trio, args.shape, args.iters)
            after = shared.hsa_dispatch_counts()
            results[f"shared, {label} pages"] = (
                ph,
                res,
                _per_iter_counts(before, after, args),
            )
        return results, reference, buffers, None
    except Exception:
        for buf in buffers:
            buf.close()
        raise


def _per_iter_counts(before, after, args):
    """(in place, staged) operands per launch, from the runtime's own counters."""
    return ((after[0] - before[0]) // args.iters, (after[1] - before[1]) // args.iters)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Measure the iGPU/NPU hand-off with and without copies.",
        epilog="The element count must suit the NPU stage: a multiple of "
        f"{add_chain.BLOCK_SIZE} under xrt (the chain's block), and a multiple "
        f"of {HSA_TILE} per dimension under hsa (the matmul's tile). Only the "
        "defaults are validated on hardware.",
    )
    p.add_argument(
        "--runtime",
        choices=sorted(DEFAULT_SHAPE),
        default="xrt",
        help="how the NPU is driven: an add chain through XRT, or a matmul "
        "through ROCR (default: %(default)s)",
    )
    p.add_argument("--rows", type=int, help="rows of A and of the result")
    p.add_argument("--cols", type=int, help="columns of B and of the result")
    p.add_argument("-k", "--depth", type=int, help="reduction depth of the matmul")
    p.add_argument("--iters", type=int, default=50, help="timed iterations")
    p.add_argument("--warmup", type=int, default=5, help="untimed iterations first")
    args = p.parse_args(argv)

    rows, cols, k = DEFAULT_SHAPE[args.runtime]
    args.shape = (args.rows or rows, args.cols or cols, args.depth or k)
    rows, cols, _ = args.shape
    if args.runtime == "xrt" and (rows * cols) % add_chain.BLOCK_SIZE:
        p.error(
            f"{rows}x{cols} is {rows * cols} elements, which the chain's "
            f"{add_chain.BLOCK_SIZE}-element blocks do not divide"
        )
    if args.runtime == "hsa" and (rows % HSA_TILE or cols % HSA_TILE):
        p.error(f"{rows}x{cols} is not a whole number of {HSA_TILE}-wide tiles")
    return args


def main(argv=None) -> int:
    """Run the variants, check they agree, and print the comparison."""
    args = parse_args(argv)
    rows, cols, k = args.shape
    print("=" * 64)
    print(
        f"  iGPU matmul -> NPU {'add' if args.runtime == 'xrt' else 'matmul'}"
        f" via {args.runtime.upper()}: copies vs shared buffers"
    )
    print("=" * 64)
    # Checked before the first iGPU allocation below, since that is the point
    # where an absent device stops being a question and starts being a
    # traceback. A host with no iGPU has declined this benchmark, not failed
    # it; 77 is what scripts/run_tests.py grades as a skip.
    if not torch.cuda.is_available():
        print("  SKIP: no ROCm device visible to torch")
        return SKIP_EXIT_CODE
    print(f"  device   : {torch.cuda.get_device_name(0)}")
    print(f"  matmul   : ({rows},{k}) @ ({k},{cols}) -> ({rows},{cols})")
    print(f"  npu stage: {rows * cols} elements out")
    print(f"  iters    : {args.iters} (plus {args.warmup} warmup)")

    chain = None
    buffers: list[SharedBuffer] = []
    try:
        bench = bench_xrt if args.runtime == "xrt" else bench_hsa
        results, reference, buffers, chain = bench(args)
    except SharedBufferError as e:
        # The iGPU was there a moment ago, so this is the NPU half: no XRT or
        # ROCR, no device, no pyxrt. Declined rather than failed, as above.
        print(f"  SKIP: shared NPU/iGPU buffers unavailable: {e}")
        return SKIP_EXIT_CODE
    except RuntimeError as e:
        print(f"  SKIP: the {args.runtime} runtime is unavailable: {e}")
        return SKIP_EXIT_CODE

    try:
        for label, (ph, _, _) in results.items():
            report(label, ph, args.iters)

        _sync()
        # Against an independent reference, not just against each other: if the
        # NPU stage were wrong both variants would be wrong identically and a
        # mutual comparison would pass. The tolerance is the NPU stage's, not
        # the hand-off's: bf16 inputs accumulated in f32 under HSA, exact f32
        # adds under XRT.
        tol = 1e-3 if args.runtime == "xrt" else 20.0
        errors = {
            label: float((res.float() - reference).abs().max().cpu())
            for label, (_, res, _) in results.items()
        }
        print(
            "\n  vs torch reference: "
            + ", ".join(
                f"{label.split(' (')[0]} {err:.2e}" for label, err in errors.items()
            )
        )

        baseline = next(iter(results))
        print("\n" + "=" * 64)
        counted = any(counts for _, _, counts in results.values())
        head = f"  {'variant':<26} {'total ms':>9} {'host KB':>9}"
        print(head + (f" {'in place':>9} {'staged':>7}" if counted else ""))
        for label, (ph, _, counts) in results.items():
            ms, by = ph.per_iter(args.iters)
            row = (
                f"  {label:<26} {sum(ms.values()):>9.3f} "
                f"{sum(by.values()) / 1024:>9.0f}"
            )
            print(row + (f" {counts[0]:>9} {counts[1]:>7}" if counts else ""))

        base_ms, base_handoff, base_bytes = _totals(results[baseline][0], args.iters)
        best = min(
            (l for l in results if l != baseline),
            key=lambda l: _totals(results[l][0], args.iters)[0],
        )
        best_ms, best_handoff, _ = _totals(results[best][0], args.iters)
        ratio = f" ({base_handoff / best_handoff:.1f}x)" if best_handoff else ""
        print(f"\n  host bytes per iteration : {base_bytes / 1024:.0f} KB -> 0 B")
        print(
            f"  hand-off cost            : {base_handoff:.3f} ms -> "
            f"{best_handoff:.3f} ms{ratio}"
        )
        print(f"  end-to-end               : {base_ms:.3f} ms -> {best_ms:.3f} ms")
        if counted:
            print(
                "  Every operand of a shared launch is dispatched where it "
                "lies; a staged one\n  is copied into a pooled buffer and "
                "back, per launch, on top of the bytes\n  that crossed the "
                "host to get there."
            )
        print("=" * 64)
        return 0 if max(errors.values()) < tol else 1
    finally:
        for buf in buffers:
            buf.close()
        if chain is not None:
            chain.close()


def _totals(ph: Phase, iters: int) -> tuple[float, float, int]:
    ms, by = ph.per_iter(iters)
    return (
        sum(ms.values()),
        sum(v for k, v in ms.items() if k.startswith("hand-off")),
        sum(by.values()),
    )


if __name__ == "__main__":
    sys.exit(main())
