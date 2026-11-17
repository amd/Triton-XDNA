#!/usr/bin/env python3
"""Standalone check of `dense_attention.attn` against a numpy reference.

For each head dim of the dense models (64, 128), the op is built as
`dense_triton_fused` builds it and called as the prefill calls it, on a
`FusedProgram`: per column, q and o for its half of the heads and the
records of its half of the KV heads, each head's max_blocks apart, from the
chunk's first K block on; (nk, q0, k0) at run time. Checked: the first chunk
of a prompt and later ones.

    python test_dense_attention.py [dh ...]

Not an example: `scripts/run_tests.py` runs every *.py directly under an
example directory, and `llm_q4nx` is excluded from the sweep as a library.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dense_attention as DA  # noqa: E402
import gemma4_attention as A  # noqa: E402
from test_gemma4_attention import blocks, records, reference, unblocks  # noqa: E402

M = 128
#: Tokens of KV the test holds.
MAX_LEN = 640
#: (head dim, q heads, KV heads), as the dense models have them.
SHAPES = {64: (32, 8), 128: (24, 8)}


def group(dh, kv_heads):
    return A.Attn(
        col=0, dh=dh, lkp=32, kv_heads=kv_heads // 2, max_blocks=MAX_LEN // 32 + 1
    )


def build(out, g, heads):
    from triton.backends.amd_triton_npu.config import config_context
    from triton.backends.amd_triton_npu.fused_device import FusedDevice

    dev = FusedDevice("fused_prefill", str(Path(out) / "proj"))
    hg = heads // 2
    nr = hg * M // (A.NQT.value * g.lkp)
    bufs = []
    for _ in range(2):
        q = np.zeros((hg * M, g.dh), bfloat16)
        kv = np.zeros((g.kv_heads * g.max_blocks, 2 * g.lkp * g.dh), bfloat16)
        bufs += [q, kv, q.copy()]
    import torch

    bufs = [torch.from_numpy(b.view(np.int16)).view(torch.bfloat16) for b in bufs]
    with config_context(bf16_emulation=True):
        dev.add(
            "attn",
            DA.attn,
            (1,),
            DA.args(g, nr, 2, 0, 0, bufs),
            transform_script=str(DA.schedule(g, dev.proj)),
            runtime=("nk", "q0", "k0"),
            pin_l2=True,
            multiplex=("L2",),
        )
        dev.compile(
            runtime_loop_tiling_sizes=(),
            bf16_emulation=True,
            extra_args=(
                "--air-channel-multiplexing=L1",
                "--omit-ping-pong-transform=L1",
                "--use-lock-race-condition-fix-auto=false",
            ),
            stack_size=4096,
        )
    dev.save(out)
    return nr


def main(dhs):
    from triton.backends.amd_triton_npu.fused_program import FusedProgram

    rng = np.random.default_rng(0)
    ok = True
    for dh in dhs:
        heads, kvh = SHAPES[dh]
        g = group(dh, kvh)
        hg, gq = heads // 2, heads // kvh
        k = rng.standard_normal((kvh, MAX_LEN, dh)).astype(bfloat16)
        v = rng.standard_normal((kvh, MAX_LEN, dh)).astype(bfloat16)
        dvt = DA.dvt(dh)
        with tempfile.TemporaryDirectory() as d:
            nr = build(d, g, heads)
            prog = FusedProgram(d, "fused_prefill")
            kvb = []
            for c in range(2):
                # each KV head's records max_blocks apart
                nb = g.max_blocks
                b = prog.zeros((g.kv_heads * nb, 2 * g.lkp * dh), bfloat16)
                for h in range(g.kv_heads):
                    hh = c * g.kv_heads + h
                    rec = records(k[hh], v[hh], 32, dvt).reshape(-1, 2 * 32 * dh)
                    b.host(write=True)[h * nb : h * nb + len(rec)] = rec
                kvb.append(b)
            for r0 in (0, 256, 512):
                q = rng.standard_normal((heads, M, dh)).astype(bfloat16)
                q0, k0 = r0 // 32, 0
                nk = -(-(r0 + M) // 32)
                nk += nk % 2
                bufs, outs = [], []
                for c in range(2):
                    qc = blocks(q[c * hg : (c + 1) * hg])
                    qb = prog.empty(qc.shape, bfloat16)
                    qb.host(write=True)[...] = qc
                    ob = prog.empty(qc.shape, bfloat16)
                    bufs += [qb, kvb[c][k0:], ob]
                    outs.append(ob)
                with prog:
                    DA.attn[(1,)](*DA.args(g, nr, nk, q0, k0, bufs))
                got = np.concatenate(
                    [unblocks(o.host().astype(np.float32)) for o in outs]
                )
                n = r0 + M
                ref = np.concatenate(
                    [
                        reference(
                            q[h : h + 1].astype(np.float32),
                            k[h // gq, :n].astype(np.float32),
                            v[h // gq, :n].astype(np.float32),
                            r0,
                            None,
                        )
                        for h in range(heads)
                    ]
                )
                rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
                print(f"dh={dh} r0={r0}: relerr {rel:.5f}", flush=True)
                ok &= rel < 6e-2
    assert ok
    print("PASS")


if __name__ == "__main__":
    os.environ.setdefault("AMD_TRITON_NPU_TARGET", "npu2")
    main([int(a) for a in sys.argv[1:]] or sorted(SHAPES))
