#!/usr/bin/env python3
"""Standalone check of `gemma4_attention.attn` against a numpy reference.

Both of Gemma4-E2B's attention ops are built as `gemma4_triton_fused` builds
them and dispatched by `npu_runner` as the prefill dispatches them: per group
(one per column), q and o for its half of the heads and the KV records of the
layer from the chunk's first K block on, all in 8x8 blocks, and (nk, q0, k0)
at run time. Checked: the
first chunk of a prompt and later ones, for the causal op and for the
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
sys.path[:0] = [str(HERE), str(HERE.parent / "gemma4_e2b_q4nx")]
import gemma4_attention as A  # noqa: E402
import gemma4_triton_fused as T  # noqa: E402

#: Tokens of KV the test holds.
MAX_LEN = 768


def build(proj):
    """The attention ops on one device, as gemma4_triton_fused builds them."""
    from triton.backends.amd_triton_npu.fused_device import FusedDevice
    from triton.backends.amd_triton_npu.config import config_context

    dev = FusedDevice("fused_prefill", str(proj))
    with config_context(bf16_emulation=True):
        T.add_attention(dev)
    return T.compile_device(dev)


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


def blocks(x):
    """[..., rows, cols] -> its 8x8 blocks [..., rows/8, cols/8, 8, 8]."""
    *b, r, c = x.shape
    return x.reshape(*b, r // 8, 8, c // 8, 8).swapaxes(-3, -2)


def unblocks(x):
    *b, r8, c8, _, _ = x.shape
    return x.swapaxes(-3, -2).reshape(*b, r8 * 8, c8 * 8)


def records(k, v, lkp, dvt):
    """KV records: per lkp rows, K's blocks and then V's, a column chunk dvt
    wide at a time."""
    out = []
    for b in range(k.shape[0] // lkp):
        kb, vb = k[b * lkp : (b + 1) * lkp], v[b * lkp : (b + 1) * lkp]
        out.append(blocks(kb).ravel())
        for z in range(vb.shape[1] // dvt):
            out.append(blocks(vb[:, z * dvt : (z + 1) * dvt]).ravel())
    return np.concatenate(out)


def run(xclbin, header, cases):
    from triton.backends.amd_triton_npu.npu_runner import Device, sync_from, sync_to

    dev = Device(xclbin, T.NARGS, txn_h=header)
    out = []
    for op, groups, q, kv, nk, q0, k0 in cases:
        bos, obs = {}, []
        hg = q.shape[0] // len(groups)
        for c, grp in enumerate(groups):
            slot = T.ATTN_SLOT[grp]
            qc = blocks(q[c * hg : (c + 1) * hg])
            qb, qm = dev.array(slot, bfloat16, qc.shape)
            qm[:] = qc
            sync_to(qb)
            kb, km = dev.array(slot + 1, bfloat16, kv.shape)
            km[:] = kv
            sync_to(kb)
            ob, om = dev.array(slot + 2, bfloat16, qc.shape)
            bos.update({slot: qb, slot + 1: kb, slot + 2: ob})
            obs.append((ob, om))
        run = dev.launch(T.attn_name(op), bos, rt=(nk, q0, k0))
        run.start()
        run.wait()
        got = []
        for ob, om in obs:
            sync_from(ob)
            got.append(unblocks(om.astype(np.float32)))
        out.append(np.concatenate(got))
    dev.close()
    return out


def main():
    rng = np.random.default_rng(0)
    cases, refs, names = [], [], []
    for op, groups in T.OPS.items():
        g = T.GROUPS[groups[0]]
        window = T.WINDOWS.get(op)
        k = rng.standard_normal((MAX_LEN, g.dh)).astype(bfloat16)
        v = rng.standard_normal((MAX_LEN, g.dh)).astype(bfloat16)
        kv = records(k, v, g.lkp, A.dvt(g.dh))
        k, v = k.astype(np.float32), v.astype(np.float32)
        # the first chunk and later ones: past the window, for the windowed
        # op
        for r0 in (0, 640) if window else (0, 256, 512):
            t = T.M
            q = rng.standard_normal((T.HEADS, t, g.dh)).astype(bfloat16)
            q0 = r0 // g.lkp
            nend = -(-(r0 + t) // g.lkp)
            k0 = max(0, q0 - window // g.lkp) if window else 0
            # the cores read K blocks in pairs; the extra one is masked
            nk = nend - k0
            nk += nk % 2
            rec = kv[k0 * 2 * g.lkp * g.dh :]
            cases.append((op, groups, q, rec, nk, q0, k0))
            n = r0 + t
            refs.append(reference(q.astype(np.float32), k[:n], v[:n], r0, window))
            names.append(f"{op} r0={r0} k0={k0}")
    with tempfile.TemporaryDirectory() as d:
        xclbin, header = build(Path(d) / "proj")
        outs = run(xclbin, header, cases)
    # S and P are bf16, and so is the output accumulated over K blocks,
    # against an f32 reference.
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
