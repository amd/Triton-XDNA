#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Standalone check of `gemma4_vision_attention` against numpy.

The device is built alone (`gemma4_vision_attention.build`) and driven as the
vision tower drives it, through `Attention.attention` on mlir-air's encoder
layout: every head widened to 2 DH columns, the first DH real and the next a
mask column, 1 in q and, in k, 0 for a real key and -big for a padding one.
Checked: a small image, one whose last K block is partly padding, and the
largest the device holds, its q rows over several dispatches.

    python test_gemma4_vision_attention.py

Not run by `scripts/run_tests.py`, which skips `llm_q4nx` as a library.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gemma4_vision_attention as VA  # noqa: E402

H, DH = VA.HEADS, VA.DH
BIG = 1.0e4


def widened(rng, rows, nvalid=None):
    """[rows, H 2DH] bf16: each head's DH columns, then its mask column
    (1 in every row for q, i.e. nvalid None; for k, 0 below nvalid and -BIG
    from there), then zeros."""
    x = np.zeros((rows, H, 2 * DH), np.float32)
    x[:, :, :DH] = rng.standard_normal((rows, H, DH))
    if nvalid is None:
        x[:, :, DH] = 1.0
    else:
        x[nvalid:, :, DH] = -BIG
    return x.reshape(rows, -1).astype(bfloat16)


def reference(q, k, v, nvalid):
    """softmax(q k^T SCALE) v over the first nvalid keys, per head, in
    float32: [Lq, H DH]."""
    f = np.float32
    q = q.astype(f).reshape(len(q), H, 2 * DH)[:, :, :DH]
    k = k.astype(f).reshape(len(k), H, 2 * DH)[:nvalid, :, :DH]
    v = v.astype(f).reshape(len(v), H, DH)[:nvalid]
    s = np.einsum("qhd,khd->hqk", q, k) * VA.SCALE
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return np.einsum("hqk,khd->qhd", p, v).reshape(len(q), H * DH)


def main():
    from triton.backends.amd_triton_npu.fused_program import FusedProgram

    rng = np.random.default_rng(0)
    # (q rows, K rows, real keys). K rows are the real ones rounded up to a
    # K block, as in the encoder.
    cases = [(256, 256, 256), (1536, 704, 700), (2560, 2560, 2520)]
    ok = True
    with tempfile.TemporaryDirectory() as d:
        VA.build(d)
        attn = VA.Attention(FusedProgram(d, VA.NAME))
        for lq, lk, nvalid in cases:
            q = widened(rng, lq)
            k = widened(rng, lk, nvalid)
            v = rng.standard_normal((lk, H * DH)).astype(bfloat16)
            got = attn.attention(q, k, v, VA.SCALE).astype(np.float32)
            ref = reference(q, k, v, nvalid)
            err = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
            # S, P and the accumulator are bf16 against an f32 reference; same
            # bound as test_gemma4_attention.
            good = err < 6e-2
            ok &= good
            print(
                f"  lq={lq} lk={lk} nvalid={nvalid}  rel {err:.2e}  "
                f"{'ok' if good else 'FAIL'}"
            )
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
