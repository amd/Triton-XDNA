#!/usr/bin/env python3
"""Standalone check of `gemma4_gemm.mm_g` against a torch reference.

Each case is an op of one device, built as `gemma4_triton_fused` builds its
GEMMs and dispatched as mlir-air's runtime dispatches a GEMM. The weights are
packed by mlir-air's own `pack_q4` and `pack_bf16`, and A is laid out as its
runtime lays it out, all in its "bf16" GEMM format, so a change on either side
of that interface shows up here rather than as a wrong prefill. Checked: int4
weights with and without the GELU drain, and bf16 weights at a depth the
device pads.

Not an example: `scripts/run_tests.py` runs every *.py directly under an
example directory, and `llm_q4nx` is excluded from the sweep as a library.
CI runs it from the test action.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import airsrc  # noqa: E402
import gemma4_gemm as G  # noqa: E402
import gemma4_triton_fused as T  # noqa: E402

airsrc.add_air_paths("gemma4_e2b_q4nx")
from air_examples.llms.shared.fused_prefill import device as D  # noqa: E402
from air_examples.llms.shared.fused_prefill import packing as P  # noqa: E402


def tiled_a(x):
    """x [M, k] -> one chunk's A as mlir-air's runtime lays it out: bf16 tiles
    [M / TM, k_pad / TK, TM, TK], zero past k (packing.a_layout)."""
    _, (hr, steps, _, tm, tk) = P.a_layout("bf16", x.shape[1])
    a = torch.zeros(hr * tm, steps * tk, dtype=torch.bfloat16)
    a[:, : x.shape[1]] = x
    return a.reshape(hr, tm, steps, tk).permute(0, 2, 1, 3).contiguous()


def padded(packed, k):
    """Weight packets padded with zero packets to the device's depth, as
    mlir-air's runtime pads them."""
    kp = packed.shape[1] * D.k_pad(k) // k
    return np.pad(packed, ((0, 0), (0, kp - packed.shape[1]), (0, 0)))


def bfp16(x):
    """x [..., K] rounded as the cores convert a bf16 matmul's operands: each
    group of 8 along K shares its largest exponent, with int8 mantissas
    rounded to even."""
    g = np.ascontiguousarray(x, np.float32).reshape(-1, 8)
    e = ((g.view(np.uint32) >> 23) & 0xFF).max(1, keepdims=True).astype(np.int64)
    sc = np.ldexp(np.float32(1), 133 - e).astype(np.float32)
    m = np.rint(g * sc)
    sc = np.where(np.abs(m).max(1, keepdims=True) > 127, sc * 0.5, sc)
    out = np.rint(g * sc) / sc
    out[(e < 8)[:, 0]] = 0
    return torch.from_numpy(out.reshape(x.shape))


def gelu(acc):
    return torch.nn.functional.gelu(acc, approximate="tanh")


def build(proj, n, cases):
    """One device with an op per case, built as gemma4_triton_fused builds
    its GEMMs. Returns its xclbin and TXN builder header."""
    from fused_device import FusedDevice
    from triton.backends.amd_triton_npu.config import config_context

    dev = FusedDevice("fused_prefill", str(proj))
    c = torch.zeros(D.M, n, dtype=torch.bfloat16)
    with config_context(bf16_emulation=True):
        for name, (act, wq, k, a, w, _) in cases.items():
            dev.add(
                name,
                G.mm_g,
                (1,),
                G.args(a, torch.from_numpy(w), c, n, k, act, wq, D.k_pad(k)),
                transform_script=str(T.SCHEDULE),
            )
    return dev.compile(
        runtime_loop_tiling_sizes=(),
        bf16_emulation=True,
        extra_args=T.AIRCC_ARGS,
        stack_size=T.STACK_SIZE,
    )


def run(xclbin, header, cases, n):
    """Each case's op, dispatched as mlir-air's runtime dispatches a GEMM."""
    import pyxrt as xrt
    from air.backend.txn_builder import TxnBuilder
    from air_examples.llms.shared.fused_prefill.engine import FROM, TO, Ctx, view

    ctx = Ctx(xrt.device(0), xclbin)
    txn = TxnBuilder(str(header))
    out = {}
    for name, (act, wq, k, a, w, _) in cases.items():
        ta = a.view(torch.int16).numpy()
        ba, bw, bc = ctx.bo(ta.nbytes, 0), ctx.bo(w.nbytes, 1), ctx.bo(D.M * n * 2, 2)
        ba.write(ta.tobytes(), 0)
        ba.sync(TO)
        bw.write(w.tobytes(), 0)
        bw.sync(TO)
        ins = np.asarray(txn(f"seg_{name}"), np.uint32)
        ib, _ = ctx.insts(ins)
        r = ctx.run(ib, len(ins), [ba, bw, bc])
        r.r.start()
        st = r.r.wait()
        assert "COMPLETED" in str(st), (name, st)
        bc.sync(FROM)
        out[name] = torch.from_numpy(view(bc, bfloat16, (D.M, n)).astype(np.float32))
    ctx.close()
    return out


def check(name, out, ref):
    rel = ((out - ref).norm() / ref.norm()).item()
    print(f"{name}: relerr {rel:.5f}", flush=True)
    assert rel < 4e-3, name


def main():
    rng = np.random.default_rng(0)
    n = P.n_pad(2 * 384)

    def act_rows(k):
        x = torch.from_numpy(rng.standard_normal((D.M, k)).astype(np.float32))
        return x.to(torch.bfloat16)

    # int4: w = q * scale + min, each weight rounded to bf16 as the kernel does
    k4 = 512
    a4 = act_rows(k4)
    q = rng.integers(0, 16, (n, k4)).astype(np.uint8)
    s = (rng.standard_normal((n, k4 // 32)) * 0.02).astype(np.float32)
    m = (rng.standard_normal((n, k4 // 32)) * 0.05).astype(np.float32)
    sb = torch.from_numpy(s).to(torch.bfloat16).float().repeat_interleave(32, 1)
    mb = torch.from_numpy(m).to(torch.bfloat16).float().repeat_interleave(32, 1)
    w4 = ((torch.from_numpy(q).float() + 128) * sb + (mb - 128 * sb)).to(torch.bfloat16)
    # the cores round A and B to bfp16 in the matmul
    acc4 = bfp16(a4.float().numpy()) @ bfp16(w4.float().numpy()).T
    q4 = padded(P.pack_q4(q, s, m, "bf16"), k4)

    # bf16 weights at a depth the device pads, as Gemma4's PLE projection
    kb = 256
    assert D.k_pad(kb) > kb
    ab = act_rows(kb)
    wb = torch.from_numpy(rng.standard_normal((n, kb)).astype(np.float32) * 0.05)
    wb = wb.to(torch.bfloat16)
    b16 = padded(P.pack_bf16(wb.float().numpy(), "bf16"), kb)
    accb = bfp16(ab.float().numpy()) @ bfp16(wb.float().numpy()).T

    cases = {
        "int4": (0, 1, k4, tiled_a(a4), q4, acc4),
        "int4_gelu": (1, 1, k4, tiled_a(a4), q4, gelu(acc4)),
        "bf16": (0, 0, kb, tiled_a(ab), b16, accb),
    }
    with tempfile.TemporaryDirectory() as d:
        xclbin, header = build(Path(d) / "proj", n, cases)
        out = run(xclbin, header, cases, n)
    for name, case in cases.items():
        check(name, out[name], case[-1])
    print("PASS")


if __name__ == "__main__":
    os.environ.setdefault("AMD_TRITON_NPU_TARGET", "npu2")
    main()
