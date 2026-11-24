#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""`vision_host_kernels` on triton-cpu against numpy references of mlir-air's
`hostops.c`, at the vision tower's shapes (a 128-row chunk, hidden 768, 12
heads of 64, FFN 3072). bf16 results agree to one ulp: the sums of squares
are taken in another order. No NPU, no weights.

    python test_vision_host_kernels.py
"""

import sys

import numpy as np
from ml_dtypes import bfloat16

import vision_host_kernels as V

M, TM, TK = 128, 32, 128
D, INTER, NH, DH, DK = 768, 3072, 12, 64, 128
EPS = 1e-6
rng = np.random.default_rng(0)


def bf(a):
    return np.asarray(a, np.float32).astype(bfloat16)


def tiles(k):
    return np.zeros((M // TM, k // TK, TM, TK), bfloat16)


def untile(am):
    hr, ks, tm, tk = am.shape
    return am.transpose(0, 2, 1, 3).reshape(hr * tm, ks * tk)


def inv_rms(x):
    return 1.0 / np.sqrt((x * x).sum(-1, keepdims=True) / x.shape[-1] + EPS)


def close(name, got, want, ulp=1):
    """bf16 arrays within `ulp` units in the last place."""
    g = np.asarray(got, bfloat16).view(np.int16).astype(np.int32)
    w = np.asarray(want, bfloat16).view(np.int16).astype(np.int32)
    bad = np.abs(g - w) > ulp
    if bad.any():
        print(f"FAIL {name}: {bad.sum()} of {bad.size} off by more than {ulp} ulp")
        return False
    print(f"ok   {name}")
    return True


def close_f32(name, got, want, rtol=1e-5, atol=1e-6):
    if not np.allclose(got, want, rtol=rtol, atol=atol):
        print(f"FAIL {name}: max abs diff {np.abs(got - want).max():.3g}")
        return False
    print(f"ok   {name}")
    return True


def test_rms_clip():
    x = rng.standard_normal((M, D)).astype(np.float32) * 3
    w = rng.standard_normal(D).astype(np.float32)
    valid, lo, hi = 100, -2.0, 1.5
    am = tiles(D)
    V.rms_clip_tile(x, w, am, valid, D, EPS, lo, hi)
    want = bf(np.clip(x * inv_rms(x) * w, lo, hi))
    want[valid:] = 0
    return close("rms_clip_tile", untile(am), want)


def test_add_rms_clip():
    h = rng.standard_normal((M, D)).astype(np.float32)
    big = bf(rng.standard_normal((M, 2 * D)) * 2)
    y = big[:, :D]  # a strided view, as a GEMM's padded output is read
    w = rng.standard_normal(D).astype(np.float32)
    valid, lo, hi = 90, -1.0, 1.25
    want = h.copy()
    yc = np.clip(y.astype(np.float32), lo, hi)
    want += yc * inv_rms(yc) * w
    want[valid:] = 0
    V.add_rms_clip(h, y, w, valid, EPS, lo, hi)
    return close_f32("add_rms_clip", h, want)


def test_clip_glu():
    ok = True
    x = bf(rng.standard_normal((M, D)) * 3)
    am = tiles(D)
    V.clip_tile(x, am, D, -1.5, 2.0)
    ok &= close(
        "clip_tile", untile(am), bf(np.clip(x.astype(np.float32), -1.5, 2.0)), 0
    )
    g = bf(rng.standard_normal((M, INTER)) * 3)
    u = bf(rng.standard_normal((M, INTER)) * 3)
    am = tiles(INTER)
    V.glu_clip_tile(g, u, am, INTER, 2.5, -2.0, 2.0, -3.0, 3.0)
    gv = np.minimum(g.astype(np.float32), 2.5)
    want = bf(np.clip(gv * np.clip(u.astype(np.float32), -2.0, 2.0), -3.0, 3.0))
    ok &= close("glu_clip_tile", untile(am), want, 0)
    return ok


def ref_head(a, lo, hi, w, cs, sn, scale):
    t = np.clip(a.astype(np.float32), lo, hi)
    t = t * inv_rms(t)
    if w is not None:
        t = t * w
    if cs is None:
        return t * scale
    o = np.empty_like(t)
    per, h2 = DH // 2, DH // 4
    for p in range(2):
        x, c, s = t[:, p * per : (p + 1) * per], cs[:, p * per :], sn[:, p * per :]
        op = o[:, p * per : (p + 1) * per]
        op[:, :h2] = (x[:, :h2] * c[:, :h2] - x[:, h2:] * s[:, :h2]) * scale
        op[:, h2:] = (x[:, h2:] * c[:, h2:per] + x[:, :h2] * s[:, h2:per]) * scale
    return o


def test_qkv_prep():
    qkv = bf(rng.standard_normal((M, 3 * NH * DH)) * 2)
    pos = np.stack([rng.integers(0, 40, M), rng.integers(0, 40, M)], 1).astype(np.int32)
    inv = (
        1.0 / (100.0 ** (np.arange(0, DH // 2, 2, dtype=np.float32) / (DH // 2)))
    ).astype(np.float32)
    cs, sn = V.rope_tables(pos, inv, DH)
    ang = np.concatenate(
        [
            pos[:, :1] * np.concatenate([inv, inv]),
            pos[:, 1:] * np.concatenate([inv, inv]),
        ],
        1,
    ).astype(np.float32)
    ok = close_f32("rope_tables cos", cs, np.cos(ang)) & close_f32(
        "rope_tables sin", sn, np.sin(ang)
    )
    qn, kn = rng.standard_normal(DH).astype(np.float32), rng.standard_normal(DH).astype(
        np.float32
    )
    valid, kv_rows = 70, 128
    bounds = (-2.0, 2.0, -1.5, 1.5, -1.0, 1.0)
    qscale, big = float(np.sqrt(DK)), 1.0e4
    q = np.zeros((M, NH * DK), bfloat16)
    k = np.zeros((kv_rows, NH * DK), bfloat16)
    v = np.zeros((kv_rows, NH * DH), bfloat16)
    V.qkv_prep(qkv, cs, sn, qn, kn, q, k, v, valid, EPS, qscale, big, bounds, NH, DH)
    E = NH * DH
    wq = np.zeros((M, NH, DK), np.float32)
    wk = np.zeros((kv_rows, NH, DK), np.float32)
    wv = np.zeros((kv_rows, NH, DH), np.float32)
    for h in range(NH):
        a = qkv[:, h * DH : (h + 1) * DH]
        wq[:, h, :DH] = ref_head(a, *bounds[0:2], qn, cs, sn, qscale)
        wq[:, h, DH] = 1.0
        wk[:, h, :DH] = ref_head(
            qkv[:, E + h * DH : E + (h + 1) * DH], *bounds[2:4], kn, cs, sn, 1.0
        )
        wv[:, h] = ref_head(
            qkv[:, 2 * E + h * DH : 2 * E + (h + 1) * DH],
            *bounds[4:6],
            None,
            None,
            None,
            1.0,
        )
    wk[valid:] = 0
    wk[valid:, :, DH] = -big
    wv[valid:] = 0
    ok &= close("qkv_prep q", q, bf(wq.reshape(M, -1)))
    ok &= close("qkv_prep k", k, bf(wk.reshape(kv_rows, -1)))
    ok &= close("qkv_prep v", v, bf(wv.reshape(kv_rows, -1)))
    return ok


def test_embed_pool():
    n, valid, rows = 200, 180, 256
    p = rng.random((n, D)).astype(np.float32)
    x = np.full((rows, D), 7, bfloat16)
    V.pixel_rows(p, x, n)
    want = np.zeros((rows, D), np.float32)
    want[:n] = 2.0 * (p - 0.5)
    ok = close("pixel_rows", x, bf(want), 0)
    y = bf(rng.standard_normal((rows, D)))
    pos = np.stack([rng.integers(0, 30, rows), rng.integers(0, 30, rows)], 1).astype(
        np.int32
    )
    pos[n:] = -1
    table = rng.standard_normal((2, 64, D)).astype(np.float32)
    h = np.empty((rows, D), np.float32)
    V.add_positions(h, y, pos, table, n, valid)
    idx = np.maximum(pos[:n], 0)
    wh = y.astype(np.float32).copy()
    wh[:n] += table[0][idx[:, 0]] + table[1][idx[:, 1]]
    wh[valid:] = 0
    ok &= close_f32("add_positions", h, wh, rtol=0, atol=0)
    members = np.full((20, 9), -1, np.int32)
    for s in range(20):
        r = np.sort(rng.choice(valid, rng.integers(1, 10), replace=False))
        members[s, : len(r)] = r
    out = np.full((32, D), 9, bfloat16)
    V.pool_norm(wh, members, out, 9.0, np.sqrt(D), EPS)
    pooled = np.zeros((20, D), np.float32)
    for s in range(20):
        for r in members[s]:
            if r >= 0:
                pooled[s] += wh[r]
    pooled = pooled / 9 * np.float32(np.sqrt(D))
    want = np.zeros((32, D), np.float32)
    want[:20] = pooled * inv_rms(pooled)
    ok &= close("pool_norm", out, bf(want))
    return ok


def main():
    from triton.backends.amd_triton_npu.driver_scope import driver_scope

    with driver_scope("cpu"):
        results = [
            test_rms_clip(),
            test_add_rms_clip(),
            test_clip_glu(),
            test_qkv_prep(),
            test_embed_pool(),
        ]
    ok = all(results)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
