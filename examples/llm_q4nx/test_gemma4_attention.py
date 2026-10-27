#!/usr/bin/env python3
"""Standalone check of `gemma4_attention.attn` against a numpy reference.

Both of Gemma4-E2B's attention ops are built as `gemma4_triton_fused` builds
them and dispatched as mlir-air's runtime dispatches an attention op: q and o
per head, the KV records of the layer from the chunk's first K block on (made
by mlir-air's `packing.kv_records`), and (nk, q0, k0) at run time. Checked:
the first chunk of a prompt and a later one, for the causal op and for the
sliding-window op past its window.

Not an example: `scripts/run_tests.py` runs every *.py directly under an
example directory, and `llm_q4nx` is excluded from the sweep as a library.
CI runs it from the test action.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import airsrc  # noqa: E402
import gemma4_triton_fused as T  # noqa: E402

airsrc.add_air_paths("gemma4_e2b_q4nx")
import importlib  # noqa: E402

from air_examples.llms.shared.fused_prefill import build as B  # noqa: E402
from air_examples.llms.shared.fused_prefill import device as D  # noqa: E402
from air_examples.llms.shared.fused_prefill import packing as P  # noqa: E402

SPEC = importlib.import_module(T.SPEC)
HEADS = SPEC.ATTN_POINTS["base"][0]
#: Tokens of KV the test holds.
MAX_LEN = 768


def build(proj):
    """The attention ops on one device, as gemma4_triton_fused builds them."""
    from fused_device import FusedDevice
    from triton.backends.amd_triton_npu.config import config_context

    dev = FusedDevice("fused_prefill", str(proj))
    with config_context(bf16_emulation=True):
        T.add_attention(dev, SPEC, HEADS)
    return dev.compile(
        runtime_loop_tiling_sizes=(),
        bf16_emulation=True,
        extra_args=T.AIRCC_ARGS,
        stack_size=T.STACK_SIZE,
    )


def reference(q, k, v, r0, window):
    """softmax(q k^T / sqrt(dh), causal or windowed) v for the chunk at r0, in
    float32. q [heads, t, dh]; k, v [r0 + t, dh]."""
    h, t, dh = q.shape
    qpos = r0 + np.arange(t)[:, None]
    kpos = np.arange(k.shape[0])[None, :]
    d = qpos - kpos
    keep = (d >= 0) & (d < window) if window else d >= 0
    s = np.einsum("htd,kd->htk", q, k) / np.sqrt(dh)
    s = np.where(keep[None], s, -np.inf)
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return np.einsum("htk,kd->htd", p, v)


def run(xclbin, header, cases):
    import pyxrt as xrt
    from air.backend.txn_builder import TxnBuilder
    from air_examples.llms.shared.fused_prefill.engine import FROM, TO, Ctx, view

    ctx = Ctx(xrt.device(0), xclbin)
    txn = TxnBuilder(str(header))
    nargs = 3 + 3 * len(SPEC.CFG.attn)
    out = []
    for op, grp, r0, q, kv, nk, q0, k0 in cases:
        slot = 3 + 3 * list(SPEC.CFG.attn).index(grp)
        bos = [ctx.bo(4096, i) for i in range(nargs)]
        qb = ctx.bo(q.nbytes, slot)
        view(qb, bfloat16, q.shape)[:] = q
        qb.sync(TO)
        rec = kv[k0:]
        kb = ctx.bo(rec.nbytes, slot + 1)
        view(kb, bfloat16, rec.shape)[:] = rec
        kb.sync(TO)
        ob = ctx.bo(q.nbytes, slot + 2)
        bos[slot : slot + 3] = [qb, kb, ob]
        ins = np.asarray(txn(f"seg_{B.seq_name(op, (HEADS,))}", nk, q0, k0), np.uint32)
        ib, _ = ctx.insts(ins)
        r = ctx.run(ib, len(ins), bos)
        r.r.start()
        st = r.r.wait()
        assert "COMPLETED" in str(st), (op, r0, st)
        ob.sync(FROM)
        out.append(view(ob, bfloat16, q.shape).astype(np.float32))
    ctx.close()
    return out


def main():
    rng = np.random.default_rng(0)
    cases, refs, names = [], [], []
    for op, (grp,) in D.attn_ops(SPEC.CFG).items():
        g = SPEC.CFG.attn[grp]
        window = (SPEC.CFG.windows or {}).get(op, g.window)
        k = rng.standard_normal((MAX_LEN, g.dh)).astype(np.float32)
        v = rng.standard_normal((MAX_LEN, g.dh)).astype(np.float32)
        kv = P.kv_records(g, k.astype(bfloat16), v.astype(bfloat16))
        k, v = (x.astype(bfloat16).astype(np.float32) for x in (k, v))
        # the first chunk, and a later one: past the window, for the windowed
        # op
        for r0 in (0, 640 if window else 256):
            t = D.M
            q = rng.standard_normal((HEADS, t, g.dh)).astype(bfloat16)
            q0 = r0 // g.lkp
            nend = -(-(r0 + t) // g.lkp)
            k0 = max(0, q0 - window // g.lkp) if window else 0
            cases.append((op, grp, r0, q, kv, nend - k0, q0, k0))
            n = r0 + t
            refs.append(reference(q.astype(np.float32), k[:n], v[:n], r0, window))
            names.append(f"{op} r0={r0} k0={k0}")
    with tempfile.TemporaryDirectory() as d:
        xclbin, header = build(Path(d) / "proj")
        outs = run(xclbin, header, cases)
    # S and P are bf16 and both matmuls run on bfp16 blocks, against an f32
    # reference.
    ok = True
    for name, got, ref in zip(names, outs, refs):
        rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        print(f"{name}: relerr {rel:.5f}", flush=True)
        ok &= rel < 6e-2
    assert ok
    print("PASS")


if __name__ == "__main__":
    os.environ.setdefault("AMD_TRITON_NPU_TARGET", "npu2")
    main()
