# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""dense_host_kernels against numpy, at the dense models' shapes, on
triton-cpu: with and without q / k norms, RoPE over the whole head and over
part of it (Phi-4), and residual adds with and without a post-norm.

Tolerances as in test_host_kernels. Runs as a script; needs no NPU.
"""

import sys

import numpy as np
from ml_dtypes import bfloat16

import dense_host_kernels as DH
from test_host_kernels import EPS, M, TK, bf, blocks8, check, close, failures, records
from test_host_kernels import rms, rng, tile_a
from triton.backends.amd_triton_npu.driver_scope import driver_scope


def rope(x, pos, inv, rot):
    """x [t, heads, dh] f32: its first rot columns rotated, pairs (i, i +
    rot / 2), as mlir-air's head_post does; cos / sin rounded to bf16."""
    h = rot // 2
    ang = pos[:, None].astype(np.float64) * inv[None, :]
    c = np.cos(ang).astype(bfloat16).astype(np.float32)[:, None]
    s = np.sin(ang).astype(bfloat16).astype(np.float32)[:, None]
    a, b = x[..., :h], x[..., h:rot]
    return np.concatenate([a * c - b * s, b * c + a * s, x[..., rot:]], -1)


def table(inv, r0):
    ang = np.arange(r0, r0 + M)[:, None] * inv[None, :].astype(np.float64)
    return tuple(
        np.ascontiguousarray(f(ang).astype(bfloat16).astype(np.float32))
        for f in (np.cos, np.sin)
    )


def inv_freq(rot, theta=5e5):
    return 1.0 / theta ** (np.arange(0, rot, 2, dtype=np.float64) / rot)


#: (t, r0, dh, rot, norm): Llama-1B, Qwen3, Phi-4 (partial RoPE), Gemma3.
CASES = (
    (128, 0, 64, 64, False),
    (70, 384, 128, 128, True),
    (128, 128, 128, 96, False),
    (33, 256, 256, 256, True),
)


def test_q_post():
    for t, r0, dh, rot, norm in CASES:
        heads, hg, h0 = 8, 4, 4
        npd = heads * dh + 384
        q = bf(rng.standard_normal((M, npd)) * 4)
        w = bf(rng.standard_normal(dh)).astype(np.float32) if norm else None
        inv = inv_freq(rot)
        qb = np.full((hg, M, dh), 7, bfloat16)
        DH.q_post(q, w, table(inv, r0), qb, t, h0, 0.5, EPS)
        x = np.asarray(q[:t, : heads * dh], np.float32).reshape(t, heads, dh)
        if norm:
            x = rms(x, w)
        ref = rope(x, np.arange(r0, r0 + t), inv, rot) * 0.5
        full = np.zeros((hg, M, dh), np.float32)
        full[:, :t] = ref[:, h0 : h0 + hg].transpose(1, 0, 2)
        close(
            qb,
            blocks8(full).astype(bfloat16),
            f"q_post t={t} dh={dh} rot={rot} norm={norm}",
            bf16=True,
        )


def test_kv_post():
    for t, r0, dh, rot, norm in CASES:
        # two heads from head 1 on, each with its records nb + 2 apart
        lkp, dvt, kvh, h0, nh = 32, dh // 4, 4, 1, 2
        dq, dk = 8 * dh, kvh * dh
        x = bf(rng.standard_normal((M, dq + 2 * dk + 384)) * 4)
        w = bf(rng.standard_normal(dh)).astype(np.float32) if norm else None
        inv = inv_freq(rot)
        kf = np.full((t, dk), 7, np.float32)
        vf = np.full((t, dk), 7, np.float32)
        nb = -(-t // lkp)
        rec = np.full((nh * (nb + 2), 2 * lkp * dh), 7, bfloat16)
        span = (nb + 2) * 2 * lkp * dh
        kc, vc = dq + h0 * dh, dq + dk + h0 * dh
        DH.kv_post(
            x, kc, vc, w, table(inv, r0), kf, vf, rec, span, nh, t, lkp, dvt, EPS,
            h0 * dh, dh,
        )  # fmt: skip
        what = f"t={t} dh={dh} rot={rot} norm={norm}"
        for j in range(nh):
            c0 = (h0 + j) * dh
            k = np.asarray(x[:t, dq + c0 : dq + c0 + dh], np.float32)[:, None]
            if norm:
                k = rms(k, w)
            kr = rope(k, np.arange(r0, r0 + t), inv, rot)[:, 0]
            vr = np.asarray(x[:t, dq + dk + c0 : dq + dk + c0 + dh], np.float32)
            close(kf[:, c0 : c0 + dh], bf(kr), f"kv_post k head {j} {what}", bf16=True)
            close(vf[:, c0 : c0 + dh], vr, f"kv_post v head {j} {what}")
            mine = rec[j * (nb + 2) : (j + 1) * (nb + 2)]
            close(
                mine[:nb],
                records(kr, vr, lkp, dvt),
                f"kv_post records head {j} {what}",
                bf16=True,
            )
            check(
                bool(np.all(mine[nb:] == 7)), f"kv_post {what}: later records untouched"
            )
        others = np.r_[: h0 * dh, (h0 + nh) * dh : dk]
        check(
            bool(np.all(kf[:, others] == 7) and np.all(vf[:, others] == 7)),
            f"kv_post {what}: the other heads' columns untouched",
        )


def test_add_rms():
    for t, n, post, tile in (
        (128, 2048, False, True),
        (45, 3072, True, True),
        (100, 2560, False, False),
        (7, 2560, True, False),
    ):
        res = rng.standard_normal((t, n)).astype(np.float32)
        y = bf(rng.standard_normal((M, n + 128)) * 3)
        w = bf(rng.standard_normal(n)).astype(np.float32) if post else None
        w2 = bf(rng.standard_normal(n)).astype(np.float32)
        ks = n // TK
        am = np.full((M // 32, ks, 32, TK), 7, bfloat16) if tile else None
        ref = res + (
            rms(np.asarray(y[:t, :n], np.float32), w)
            if post
            else np.asarray(y[:t, :n], np.float32)
        )
        DH.add_rms(res, y, w, t, EPS, am=am, w2=w2 if tile else None)
        what = f"add_rms t={t} n={n} post={post}"
        close(res, ref, what)
        if tile:
            close(am, tile_a(rms(ref, w2), ks), f"{what} tiles", bf16=True)


def main():
    with driver_scope("cpu"):
        for name, fn in list(globals().items()):
            if name.startswith("test_"):
                print(name)
                fn()
    if failures:
        print(f"FAIL: {len(failures)} check(s)")
        return 1
    print("PASS: dense_host_kernels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
