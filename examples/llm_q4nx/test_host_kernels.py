# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""host_kernels against numpy, at Gemma4-E2B's shapes, on triton-cpu.

f32 results are checked to a relative 1e-5. bf16 results are checked to one
bf16 ulp, plus a floor for values that cancel to near zero, where the last
f32 bits are all that differ. The bytes of a layout a kernel must leave alone
are checked untouched. Runs as a script; needs no NPU.
"""

import sys

import numpy as np
from ml_dtypes import bfloat16

import host_kernels as H
from triton.backends.amd_triton_npu.driver_scope import driver_scope

M, TM, TK = 128, 32, 128
EPS = 1e-6
rng = np.random.default_rng(0)
failures = []


def check(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        failures.append(what)


def close(got, ref, what, bf16=False):
    got, ref = np.asarray(got, np.float32), np.asarray(ref, np.float32)
    if bf16:
        # one bf16 ulp, and the f32 rounding of terms that cancel: RoPE's
        # a * cos - b * sin can land next to zero
        tol = np.abs(ref) * 2.0**-7 + 1e-6 * float(np.abs(ref).max(initial=0.0))
        ok = bool(np.all(np.abs(got - ref) <= tol))
    else:
        ok = bool(np.allclose(got, ref, rtol=1e-5, atol=1e-6))
    err = float(np.abs(got - ref).max()) if got.size else 0.0
    if not ok:
        bad = np.unravel_index(
            np.argmax(np.abs(got - ref) - tol if bf16 else np.abs(got - ref)), got.shape
        )
        what += f"; at {tuple(int(i) for i in bad)} got {got[bad]!r} ref {ref[bad]!r}"
    check(ok, f"{what} (max abs err {err:.3g})")


# ---- numpy references ------------------------------------------------------


def rms(x, w):
    x = np.asarray(x, np.float32)
    y = x / np.sqrt((x * x).sum(-1, keepdims=True) / x.shape[-1] + EPS)
    return y if w is None else y * np.asarray(w, np.float32)


def tile_a(x, ks):
    """x [t, k] -> [M / TM, ks, TM, TK] bf16, zero rows past t and columns
    past k."""
    t, k = x.shape
    full = np.zeros((M, ks * TK), np.float32)
    full[:t, :k] = x
    return full.reshape(M // TM, TM, ks, TK).transpose(0, 2, 1, 3).astype(bfloat16)


def blocks8(x):
    *b, r, c = x.shape
    return x.reshape(*b, r // 8, 8, c // 8, 8).swapaxes(-3, -2).reshape(*b, r, c)


def rope(x, pos, inv):
    """x [t, heads, dh] f32, pairs (i, i + dh / 2)."""
    h = x.shape[-1] // 2
    ang = pos[:, None].astype(np.float32) * inv[None, :]
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    a, b = x[..., :h], x[..., h:]
    return np.concatenate([a * c - b * s, b * c + a * s], -1)


def inv_freq(dh, theta):
    return (1.0 / theta ** (np.arange(0, dh // 2, dtype=np.float32) * 2.0 / dh)).astype(
        np.float32
    )


def records(k, v, lkp, dvt):
    t, dh = k.shape
    nb = -(-t // lkp)
    kp = np.zeros((nb * lkp, dh), np.float32)
    vp = np.zeros((nb * lkp, dh), np.float32)
    kp[:t], vp[:t] = k, v
    out = []
    for b in range(nb):
        kb, vb = kp[b * lkp : (b + 1) * lkp], vp[b * lkp : (b + 1) * lkp]
        parts = [blocks8(kb).reshape(-1)]
        for z in range(dh // dvt):
            parts.append(blocks8(vb[:, z * dvt : (z + 1) * dvt]).reshape(-1))
        out.append(np.concatenate(parts))
    return np.stack(out).astype(bfloat16)


def bf(x):
    return np.asarray(x, np.float32).astype(bfloat16)


# ---- the checks ------------------------------------------------------------


def test_rms_tile():
    for t, k, ks, src in (
        (128, 1536, 12, "f32"),
        (77, 1536, 12, "bf16"),
        (5, 6144, 48, "f32"),
    ):
        x = rng.standard_normal((M, k)).astype(np.float32) * 3
        if src == "bf16":
            x = bf(x)
        w = bf(rng.standard_normal(k))
        am = np.full((M // TM, ks, TM, TK), 7, bfloat16)
        H.rms_tile(x, w, am, t, k, EPS)
        close(am, tile_a(rms(x[:t], w), ks), f"rms_tile t={t} k={k} {src}", bf16=True)
        am = np.zeros((M // TM, ks, TM, TK), bfloat16)
        H.rms_tile(x, None, am, t, k, EPS, norm=False)
        close(
            am,
            tile_a(np.asarray(x[:t], np.float32), ks),
            f"plain tile t={t} k={k}",
            bf16=True,
        )
    # columns past k are the device's K padding, left alone
    x = rng.standard_normal((M, 256)).astype(np.float32)
    am = np.full((M // TM, 4, TM, TK), 7, bfloat16)
    H.rms_tile(x, None, am, 100, 256, EPS, norm=False)
    check(bool(np.all(am[:, 2:] == 7)), "K padding of the A tiles untouched")


def test_add_rms():
    n, npd = 1536, 1536 + 384
    for t, mode in ((128, "none"), (90, "tile"), (33, "rms_tile"), (64, "scale")):
        res = rng.standard_normal((M, n)).astype(np.float32)
        y = bf(rng.standard_normal((M, npd)))
        w, w2 = (bf(rng.standard_normal(n)) for _ in range(2))
        s = 0.37
        ref = res[:t] + rms(y[:t, :n], w)
        if mode == "scale":
            ref = ref * np.float32(s)
        am = np.zeros((M // TM, n // TK, TM, TK), bfloat16)
        got = res.copy()
        H.add_rms(
            got, y, w, t, EPS,
            scale=s if mode == "scale" else None,
            am=am if mode in ("tile", "rms_tile") else None,
            w2=w2 if mode == "rms_tile" else None,
        )  # fmt: skip
        close(got[:t], ref, f"add_rms t={t} {mode}")
        check(np.array_equal(got[t:], res[t:]), f"add_rms t={t} rows past t untouched")
        if mode == "tile":
            close(am, tile_a(ref, n // TK), "add_rms tiles the residual", bf16=True)
        if mode == "rms_tile":
            close(
                am,
                tile_a(rms(ref, w2), n // TK),
                "add_rms tiles rms(residual)",
                bf16=True,
            )


def test_mul_tile():
    for t, k, src in ((128, 6144, "bf16"), (50, 256, "f32 strided")):
        a = bf(rng.standard_normal((M, k + 384)))
        if src == "bf16":
            b = bf(rng.standard_normal((M, k + 384)))
            bv = b
        else:
            b = rng.standard_normal((M, 35, k)).astype(np.float32)
            bv = b[:, 3]
        am = np.zeros((M // TM, k // TK, TM, TK), bfloat16)
        H.mul_tile(a, bv, am, t, k)
        ref = np.asarray(a[:t, :k], np.float32) * np.asarray(bv[:t, :k], np.float32)
        close(am, tile_a(ref, k // TK), f"mul_tile t={t} k={k} {src}", bf16=True)


def test_q_post():
    for t, r0, dh, theta in ((128, 0, 256, 1e4), (70, 384, 512, 1e6)):
        heads, hg, h0 = 8, 4, 4
        npd = heads * dh + 384
        q = bf(rng.standard_normal((M, npd)) * 4)
        w = bf(rng.standard_normal(dh))
        inv = inv_freq(dh, theta)
        scale = 0.7
        qb = np.full((hg, M, dh), 7, bfloat16)
        rp = (np.empty((M, dh // 2), np.float32), np.empty((M, dh // 2), np.float32))
        H.rope_table(inv, *rp, r0)
        H.q_post(q, w, rp, qb, t, h0, scale, EPS)
        x = np.asarray(q[:t, : heads * dh], np.float32).reshape(t, heads, dh)
        ref = rope(rms(x, w), np.arange(r0, r0 + t), inv) * scale
        full = np.zeros((hg, M, dh), np.float32)
        full[:, :t] = ref[:, h0 : h0 + hg].transpose(1, 0, 2)
        close(
            qb,
            blocks8(full).astype(bfloat16),
            f"q_post t={t} r0={r0} dh={dh}",
            bf16=True,
        )


def test_kv_post():
    for t, r0, dh, lkp in ((128, 0, 256, 32), (45, 256, 512, 16)):
        dvt = dh // 4
        npd = dh + 384
        k = bf(rng.standard_normal((M, npd)) * 4)
        v = bf(rng.standard_normal((M, npd)) * 4)
        w = bf(rng.standard_normal(dh))
        inv = inv_freq(dh, 1e4)
        kf = np.zeros((M, dh), np.float32)
        vf = np.zeros((M, dh), np.float32)
        nb = -(-t // lkp)
        rec = np.full((nb + 2, 2 * lkp * dh), 7, bfloat16)
        rp = (np.empty((M, dh // 2), np.float32), np.empty((M, dh // 2), np.float32))
        H.rope_table(inv, *rp, r0)
        H.kv_post(k, v, w, rp, kf, vf, rec, t, lkp, dvt, EPS)
        kr = rope(
            rms(np.asarray(k[:t, :dh], np.float32)[:, None], w),
            np.arange(r0, r0 + t),
            inv,
        )[:, 0]
        vr = rms(np.asarray(v[:t, :dh], np.float32), None)
        close(kf[:t], kr, f"kv_post k t={t} dh={dh}")
        close(vf[:t], vr, f"kv_post v t={t} dh={dh}")
        close(
            rec[:nb],
            records(kr, vr, lkp, dvt),
            f"kv_post records t={t} lkp={lkp}",
            bf16=True,
        )
        check(
            bool(np.all(rec[nb:] == 7)),
            f"kv_post t={t}: records past the last untouched",
        )


def test_o_tile():
    for t, dh in ((128, 256), (61, 512)):
        heads, hg = 8, 4
        o = [bf(rng.standard_normal((hg, M, dh))) for _ in range(2)]
        ks = heads * dh // TK
        am = np.full((M // TM, ks, TM, TK), 7, bfloat16)
        for p in range(2):
            H.o_tile(blocks8(o[p]), am, t, p * hg)
        x = np.concatenate(
            [
                np.asarray(op, np.float32)[:, :t].transpose(1, 0, 2).reshape(t, -1)
                for op in o
            ],
            1,
        )
        close(am, tile_a(x, ks), f"o_tile t={t} dh={dh}", bf16=True)


def test_ple():
    t, nl, pd = 100, 35, 256
    p = bf(rng.standard_normal((M, nl * pd + 128)) * 20)
    w = bf(rng.standard_normal(pd))
    tbl = rng.standard_normal((t, nl * pd)).astype(np.float32)
    pli = np.zeros((t, nl, pd), np.float32)
    H.ple(p, w, tbl, pli, t, 0.0255, 0.7071, EPS)
    proj = np.asarray(p[:t, : nl * pd], np.float32).reshape(t, nl, pd) * np.float32(
        0.0255
    )
    ref = (rms(proj, w) + tbl.reshape(t, nl, pd)) * np.float32(0.7071)
    close(pli, ref, "ple")


def test_embed():
    V, W = 1000, 1536
    q = rng.integers(-128, 128, (V, W), dtype=np.int8)
    s = rng.random((V, W // 32)).astype(np.float32)
    ids = np.array([5, 999, 0, 5, 321], np.int32)
    out = np.zeros((len(ids), W), np.float32)
    H.embed(q, s, ids, out, 1.0)
    ref = q[ids].astype(np.float32) * np.repeat(s[ids], 32, 1)
    close(out, ref, "embed")


def test_rms_rows_softcap():
    x = rng.standard_normal((1, 1536)).astype(np.float32) * 5
    w = bf(rng.standard_normal(1536))
    out = np.zeros(1536, bfloat16)
    H.rms_rows(x, w, out, EPS)
    close(out, rms(x, w)[0].astype(bfloat16), "rms_rows -> bf16", bf16=True)
    nc, tf, nr, fg = 8, 4, 4, 1024
    y = bf(rng.standard_normal((nc, tf, nr, fg)) * 40)
    out = np.zeros(nc * tf * nr * fg, np.float32)
    H.softcap(y, out, 30.0, nr, tf, fg)
    logits = np.asarray(y, np.float32).transpose(0, 2, 1, 3).reshape(-1)
    ref = 30.0 * np.tanh(logits / 30.0)
    check(bool(np.allclose(out, ref, rtol=0, atol=1e-5)), "softcap unpacks and caps")


def main():
    with driver_scope("cpu"):
        for name, fn in list(globals().items()):
            if name.startswith("test_"):
                print(name)
                fn()
    if failures:
        print(f"FAIL: {len(failures)} check(s)")
        return 1
    print("PASS: host_kernels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
