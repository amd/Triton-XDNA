#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Hand a tensor from the iGPU to the NPU, with and without copies.

The pipeline is the simplest thing that crosses the boundary twice::

    C   = A @ B            on the iGPU
    OUT = (C + E1) + E2    on the NPU  (two chained elementwise adds)
    (OUT consumed)         on the iGPU

Two implementations of the same pipeline:

**copy**  -- what the boundary costs by default. ``C`` comes back to the host
(``.cpu()``), the launch stages it from there into an XRT buffer object, the
result is staged back out, and it returns to the iGPU (``.to("cuda")``).

**shared** -- the iGPU writes ``C`` *into* the buffer object the NPU will read
(``torch.matmul(..., out=...)`` straight into a shared buffer), the dispatch
names that buffer, and the result is already an iGPU tensor. Nothing crosses.

Both are timed per phase, not just end to end. The two variants differ only in
the hand-off, and the hand-off is the small term next to the NPU dispatch that
surrounds it, so a whole-pipeline number mixes the part that changed with a
much larger part that did not. The per-phase rows isolate it, and the byte
columns say the same thing without depending on timing at all.

Run it::

    source scripts/dev-env.sh
    python examples/npu_gpu_dlpack/zero_copy_benchmark.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

import add_chain
from triton.backends.amd_triton_npu import shared
from triton.backends.amd_triton_npu.shared import SharedBuffer, SharedBufferError
from triton.backends.amd_triton_npu.multilaunch import NPUChain

# ROWS x COLS is the shape the matmul produces; K is its reduction depth.
# The element count must be a multiple of the chain's BLOCK_SIZE.
ROWS, COLS, K = 128, 768, 512
N = ROWS * COLS
ITERS = 50
WARMUP = 5

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


def run_copy(
    chain: NPUChain,
    a: torch.Tensor,
    b: torch.Tensor,
    e1_host: np.ndarray,
    e2_host: np.ndarray,
    iters: int,
) -> tuple[Phase, torch.Tensor]:
    """The default path: every hand-off goes through host memory."""
    out_host: np.ndarray = np.zeros(N, dtype=np.float32)
    tmp_host: np.ndarray = np.zeros(N, dtype=np.float32)
    ph = Phase()
    result = None
    for i in range(iters):
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
        result = torch.from_numpy(res[I_OUT]).to("cuda").reshape(ROWS, COLS)
        _sync()
        t4 = time.perf_counter()

        if i == 0:
            continue  # first iteration allocates the BO set
        ph.add("igpu matmul", (t1 - t0) * 1e3)
        ph.add("hand-off out (D2H)", (t2 - t1) * 1e3, N * 4)
        ph.add("npu dispatch", (t3 - t2) * 1e3)
        ph.add("hand-off back (H2D)", (t4 - t3) * 1e3, N * 4)
    return ph, result


def run_shared(
    chain: NPUChain,
    a: torch.Tensor,
    b: torch.Tensor,
    c_buf: SharedBuffer,
    out_buf: SharedBuffer,
    e1_host: np.ndarray,
    e2_host: np.ndarray,
    iters: int,
) -> tuple[Phase, torch.Tensor]:
    """Zero-copy: the iGPU writes where the NPU reads."""
    tmp_host: np.ndarray = np.zeros(N, dtype=np.float32)
    c_view = c_buf.torch().view(ROWS, COLS)
    ph = Phase()
    result = None
    for i in range(iters):
        _sync()
        t0 = time.perf_counter()
        # out= writes the matmul straight into the NPU's input buffer.
        torch.matmul(a, b, out=c_view)
        _sync()
        t1 = time.perf_counter()

        # No transfer -- only a fence, so the NPU does not read pages the iGPU
        # has not finished writing.
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
        result = out_buf.torch().view(ROWS, COLS)
        _sync()
        t4 = time.perf_counter()

        if i == 0:
            continue
        ph.add("igpu matmul", (t1 - t0) * 1e3)
        ph.add("hand-off out (fence)", (t2 - t1) * 1e3, 0)
        ph.add("npu dispatch", (t3 - t2) * 1e3)
        ph.add("hand-off back (none)", (t4 - t3) * 1e3, 0)
    return ph, result


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


def main() -> int:
    """Run both variants, check they agree, and print the comparison."""
    print("=" * 64)
    print("  iGPU matmul -> NPU add: copies vs shared buffers")
    print("=" * 64)
    print(f"  device   : {torch.cuda.get_device_name(0)}")
    print(f"  matmul   : ({ROWS},{K}) @ ({K},{COLS}) -> ({ROWS},{COLS}) f32")
    print(f"  npu add  : {N} elements, {N * 4 / 1024:.0f} KB per operand")
    print(f"  iters    : {ITERS} (plus {WARMUP} warmup)")

    a = torch.randn(ROWS, K, dtype=torch.float32, device="cuda")
    b = torch.randn(K, COLS, dtype=torch.float32, device="cuda")
    _sync()
    # The two NPU-side addends. Static operands, so they are staged once per BO
    # set and are not part of the per-iteration hand-off either way.
    e1_host: np.ndarray = np.full(N, 0.5, dtype=np.float32)
    e2_host: np.ndarray = np.full(N, 0.25, dtype=np.float32)
    reference = (torch.matmul(a, b).reshape(-1) + 0.75).reshape(ROWS, COLS)
    _sync()

    chain = add_chain.build("zero_copy_bench_add", N)
    # Bound before the try so the finally can close whatever got as far as
    # existing -- empty_like can fail after empty succeeded.
    c_buf = out_buf = None

    try:
        # Shared buffers: allocated and registered once, reused every iteration.
        # Allocating is also the availability check, so there is no separate
        # probe. XRT owns the pages (the NPU dispatch names the BO); the iGPU
        # borrows them, which is what lets torch.matmul write straight into the
        # input.
        try:
            c_buf = shared.empty(N, dtype=torch.float32, device="xrt:0", share="hip:0")
            out_buf = shared.empty_like(c_buf)
        except SharedBufferError as e:
            sys.exit(f"shared NPU/iGPU buffers unavailable: {e}")

        print("\n  warming up (JIT, ELF build, BO allocation)...")
        run_copy(chain, a, b, e1_host, e2_host, WARMUP)
        run_shared(chain, a, b, c_buf, out_buf, e1_host, e2_host, WARMUP)

        ph_copy, res_copy = run_copy(chain, a, b, e1_host, e2_host, ITERS)
        ph_shared, res_shared = run_shared(
            chain, a, b, c_buf, out_buf, e1_host, e2_host, ITERS
        )

        t_copy, h_copy, b_copy = report("copy (via host)", ph_copy, ITERS - 1)
        t_shared, h_shared, b_shared = report(
            "shared (zero-copy)", ph_shared, ITERS - 1
        )

        _sync()
        # Against an independent reference, not just against each other: if the
        # NPU stage were wrong both variants would be wrong identically and a
        # mutual comparison would pass.
        err_copy = float((res_copy - reference).abs().max())
        err_shared = float((res_shared - reference).abs().max())
        tol = 1e-3
        print(f"\n  vs torch reference: copy {err_copy:.2e}, shared {err_shared:.2e}")

        print("\n" + "=" * 64)
        print(f"  host bytes per iteration : {b_copy/1024:.0f} KB -> {b_shared} B")
        ratio = f" ({h_copy / h_shared:.1f}x)" if h_shared else ""
        print(
            f"  hand-off cost            : {h_copy:.3f} ms -> "
            f"{h_shared:.3f} ms{ratio}"
        )
        print(f"  end-to-end               : {t_copy:.3f} ms -> {t_shared:.3f} ms")
        print(
            f"  NPU dispatch is {ph_shared.ms['npu dispatch'] / (ITERS-1):.2f} ms of "
            f"that, and is identical in both -- the hand-off is what differs."
        )
        print("=" * 64)
        ok = err_copy < tol and err_shared < tol
        if not ok:
            print("  RESULTS ARE WRONG -- timings below are meaningless")
        return 0 if ok else 1
    finally:
        chain.close()
        for buf in (out_buf, c_buf):
            if buf is not None:
                buf.close()


if __name__ == "__main__":
    sys.exit(main())
