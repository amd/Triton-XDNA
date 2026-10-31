#!/usr/bin/env python3
"""Standalone check of `gemma4_gemm.mm_g` against a torch reference.

Each case is an op of one device, built as `gemma4_triton_fused` builds its
GEMMs and dispatched by `npu_runner` as the prefill dispatches a GEMM. The
weights are packed by `gemma4_gemm.pack_q4` and `pack_bf16` and A is laid out
in the A tiles the host kernels write, so a change on either side of that
interface shows up here rather than as a wrong prefill. Checked: int4 weights
with and without the GELU drain, and bf16 weights at a depth the device
pads.

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
sys.path[:0] = [str(HERE), str(HERE.parent / "gemma4_e2b_q4nx")]
import gemma4_gemm as G  # noqa: E402
import gemma4_triton_fused as T  # noqa: E402


def tiled_a(x):
    """x [M, k] -> one chunk's A tiles: bf16 [M / TM, k_pad / TK, TM, TK],
    zero past k."""
    hr, tm, tk, steps = G.M // G.TM, G.TM, G.TK, G.k_pad(x.shape[1]) // G.TK
    a = torch.zeros(hr * tm, steps * tk, dtype=torch.bfloat16)
    a[:, : x.shape[1]] = x
    return a.reshape(hr, tm, steps, tk).permute(0, 2, 1, 3).contiguous()


def padded(packed, k):
    """Weight packets padded with zero packets to the device's depth, as the
    prefill pads them."""
    kp = packed.shape[1] * G.k_pad(k) // k
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
    from triton.backends.amd_triton_npu.fused_device import FusedDevice
    from triton.backends.amd_triton_npu.config import config_context

    dev = FusedDevice("fused_prefill", str(proj))
    c = torch.zeros(G.M, n, dtype=torch.bfloat16)
    with config_context(bf16_emulation=True):
        for name, (act, wq, k, a, w, _) in cases.items():
            dev.add(
                name,
                G.mm_g,
                (1,),
                G.args(a, torch.from_numpy(w), c, n, k, act, wq, G.k_pad(k)),
                transform_script=str(T.SCHEDULE),
            )
    return T.compile_device(dev)


def run(xclbin, header, cases, n):
    """Each case's op, dispatched as the prefill dispatches a GEMM."""
    from triton.backends.amd_triton_npu.npu_runner import Device, sync_from, sync_to

    dev = Device(xclbin, 3, txn_h=header)
    out = {}
    for name, (act, wq, k, a, w, _) in cases.items():
        ta = a.view(torch.int16).numpy()
        ba, am = dev.array(0, np.int16, ta.shape)
        am[:] = ta
        sync_to(ba)
        bw, wm = dev.array(1, np.uint8, w.shape)
        wm[:] = w
        sync_to(bw)
        bc, cm = dev.array(2, bfloat16, (G.M, n))
        run = dev.launch(name, {0: ba, 1: bw, 2: bc})
        run.start()
        run.wait()
        sync_from(bc)
        out[name] = torch.from_numpy(cm.astype(np.float32))
    dev.close()
    return out


def check(name, out, ref):
    rel = ((out - ref).norm() / ref.norm()).item()
    print(f"{name}: relerr {rel:.5f}", flush=True)
    assert rel < 4e-3, name


def main():
    rng = np.random.default_rng(0)
    n = G.n_pad(2 * 384)

    def act_rows(k):
        x = torch.from_numpy(rng.standard_normal((G.M, k)).astype(np.float32))
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
    q4 = padded(G.pack_q4(q, s, m), k4)

    # bf16 weights at a depth the device pads, as Gemma4's PLE projection
    kb = 256
    assert G.k_pad(kb) > kb
    ab = act_rows(kb)
    wb = torch.from_numpy(rng.standard_normal((n, kb)).astype(np.float32) * 0.05)
    wb = wb.to(torch.bfloat16)
    b16 = padded(G.pack_bf16(wb.float().numpy()), kb)
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
