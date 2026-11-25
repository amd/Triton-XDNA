# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""The host side of Gemma4's vision tower, as Triton kernels run by
triton-cpu: what happens between the tower's NPU GEMMs and attention.

Each GEMM of the tower is a ClippableLinear, its input and output clamped to
per-layer bounds. The kernels here apply a GEMM's output clamp as they read
its bf16 output, and its successor's input clamp as they write that GEMM's
bf16 input, rounded once, to nearest even. The math is mlir-air's tower's
(`llms/gemma4_e2b_vision/hostops.c`); every value is f32 between a load and
a store.

Arguments are numpy arrays or a `FusedProgram`'s buffers (`host_kernels.Ptr`);
rows may have any stride. A GEMM's input is written straight into its A
tiles (`host_kernels`' layout), rows past `valid` as zeros.
"""

import numpy as np
import triton
import triton.language as tl

from host_kernels import BR, Ptr, _ch, _inv_rms, _on_host, _tile_off


@triton.jit
def _clamp(x, lo, hi):
    return tl.minimum(tl.maximum(x, lo), hi)


@triton.jit
def _rms_clip_tile(
    X, ldx, W, AM, valid, eps, lo, hi,
    K: tl.constexpr, KS: tl.constexpr, TM: tl.constexpr, TK: tl.constexpr,
    CH: tl.constexpr, BR: tl.constexpr,
):  # fmt: skip
    """A tiles of clip(X / rms(X) * W, lo, hi), X f32; 0 on rows >= valid."""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    live = (r < valid)[:, None]
    inv = _inv_rms(X, ldx, r, live, eps, K, CH)
    for c0 in range(0, K, CH):
        c = c0 + tl.arange(0, CH)
        x = tl.load(X + r[:, None] * ldx + c[None, :], mask=live, other=0.0)
        y = x.to(tl.float32) * inv[:, None] * tl.load(W + c)[None, :]
        y = tl.where(live, _clamp(y, lo, hi), 0.0)
        tl.store(AM + _tile_off(r[:, None], c[None, :], KS, TM, TK), y.to(tl.bfloat16))


@triton.jit
def _add_rms_clip(
    H, Y, ldy, W, valid, eps, lo, hi,
    N: tl.constexpr, CH: tl.constexpr, BR: tl.constexpr,
):  # fmt: skip
    """H += clip(Y, lo, hi) / rms(clip(Y)) * W on rows < valid, Y bf16; H = 0
    on the rows after."""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    live = (r < valid)[:, None]
    acc = tl.zeros((BR, CH), tl.float32)
    for c0 in range(0, N, CH):
        c = c0 + tl.arange(0, CH)
        y = tl.load(Y + r[:, None] * ldy + c[None, :], mask=live, other=0.0)
        y = _clamp(y.to(tl.float32), lo, hi)
        acc += y * y
    inv = 1.0 / tl.sqrt(tl.sum(acc, 1) / N + eps)
    for c0 in range(0, N, CH):
        c = c0 + tl.arange(0, CH)
        y = tl.load(Y + r[:, None] * ldy + c[None, :], mask=live, other=0.0)
        y = _clamp(y.to(tl.float32), lo, hi) * inv[:, None] * tl.load(W + c)[None, :]
        at = H + r[:, None] * N + c[None, :]
        tl.store(at, tl.where(live, tl.load(at) + y, 0.0))


@triton.jit
def _clip_tile(
    X, ldx, AM, lo, hi,
    K: tl.constexpr, KS: tl.constexpr, TM: tl.constexpr, TK: tl.constexpr,
    CH: tl.constexpr, BR: tl.constexpr,
):  # fmt: skip
    """A tiles of clip(X, lo, hi), X bf16."""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    for c0 in range(0, K, CH):
        c = c0 + tl.arange(0, CH)
        x = tl.load(X + r[:, None] * ldx + c[None, :]).to(tl.float32)
        tl.store(
            AM + _tile_off(r[:, None], c[None, :], KS, TM, TK),
            _clamp(x, lo, hi).to(tl.bfloat16),
        )


@triton.jit
def _glu_clip_tile(
    G, ldg, U, ldu, AM, g_hi, u_lo, u_hi, lo, hi,
    K: tl.constexpr, KS: tl.constexpr, TM: tl.constexpr, TK: tl.constexpr,
    CH: tl.constexpr, BR: tl.constexpr,
):  # fmt: skip
    """A tiles of clip(min(G, g_hi) * clip(U, u_lo, u_hi), lo, hi): G the gate
    GEMM's GELU output, whose output clamp min(G, g_hi) stands in for (the
    lower bound is far below where GELU leaves 0), U the up GEMM's."""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    for c0 in range(0, K, CH):
        c = c0 + tl.arange(0, CH)
        g = tl.minimum(tl.load(G + r[:, None] * ldg + c[None, :]).to(tl.float32), g_hi)
        u = _clamp(
            tl.load(U + r[:, None] * ldu + c[None, :]).to(tl.float32), u_lo, u_hi
        )
        tl.store(
            AM + _tile_off(r[:, None], c[None, :], KS, TM, TK),
            _clamp(g * u, lo, hi).to(tl.bfloat16),
        )


@triton.jit
def _head(A, r, h, live, lo, hi, W, eps, CS, SN, scale,
          DH: tl.constexpr, HAS_W: tl.constexpr, ROPE: tl.constexpr):  # fmt: skip
    """One head of rows r at A (the head's first column; row stride in r's
    caller): clip, rmsnorm (times W), and the 2-D rope, scaled. The head's two
    halves rotate on the patch's x and y, each half-split: lane j pairs with
    lane j ^ (DH / 4)."""
    j = tl.arange(0, DH)
    t = _clamp(tl.load(A + j[None, :], mask=live, other=0.0).to(tl.float32), lo, hi)
    inv = 1.0 / tl.sqrt(tl.sum(t * t, 1) / DH + eps)
    if HAS_W:
        w = tl.load(W + j)[None, :]
        t = t * inv[:, None] * w
    else:
        t = t * inv[:, None]
    if ROPE:
        Q: tl.constexpr = DH // 4
        p = j ^ Q
        tp = _clamp(
            tl.load(A + p[None, :], mask=live, other=0.0).to(tl.float32), lo, hi
        )
        if HAS_W:
            tp = tp * inv[:, None] * tl.load(W + p)[None, :]
        else:
            tp = tp * inv[:, None]
        c = tl.load(CS + j[None, :], mask=live, other=0.0)
        s = tl.load(SN + j[None, :], mask=live, other=0.0)
        sign = tl.where((j % (2 * Q)) < Q, -1.0, 1.0)[None, :]
        return (t * c + sign * (tp * s)) * scale
    return t * scale


@triton.jit
def _qkv_prep(
    QKV, ldq, CS, SN, QN, KN, Q, K, V, rows, valid, kv_rows, eps, qscale, big,
    q_lo, q_hi, k_lo, k_hi, v_lo, v_hi,
    NH: tl.constexpr, DH: tl.constexpr, DK: tl.constexpr, BR: tl.constexpr,
):  # fmt: skip
    """The fused q|k|v GEMM's output [rows, 3 NH DH] -> the attention's
    inputs, each head of Q and K widened from DH to DK columns:
      Q [rows, NH DK]: clip, q_norm, rope, * qscale; column DH is 1
      K [kv_rows, NH DK]: clip, k_norm, rope; column DH is 0, or -big on rows
                         >= valid (padding keys, which are 0 otherwise)
      V [kv_rows, NH DH]: clip, rmsnorm without a weight; 0 on rows >= valid"""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    h = tl.program_id(1)
    E: tl.constexpr = NH * DH
    j = tl.arange(0, DH)
    jw = tl.arange(0, DK - DH)
    rq = (r < rows)[:, None]
    rk = (r < kv_rows)[:, None]
    real = (r < valid)[:, None]
    a = QKV + r[:, None] * ldq + h * DH
    cs, sn = CS + r[:, None] * DH, SN + r[:, None] * DH
    q = _head(a, r, h, rq, q_lo, q_hi, QN, eps, cs, sn, qscale, DH, True, True)
    qo = Q + r[:, None] * (NH * DK) + h * DK
    tl.store(qo + j[None, :], q.to(tl.bfloat16), mask=rq)
    one = tl.where(jw == 0, 1.0, 0.0)[None, :] + tl.zeros((BR, DK - DH), tl.float32)
    tl.store(qo + DH + jw[None, :], one.to(tl.bfloat16), mask=rq)
    k = _head(a + E, r, h, rk & real, k_lo, k_hi, KN, eps, cs, sn, 1.0, DH, True, True)
    v = _head(
        a + 2 * E, r, h, rk & real, v_lo, v_hi, KN, eps, cs, sn, 1.0, DH, False, False
    )
    ko = K + r[:, None] * (NH * DK) + h * DK
    tl.store(ko + j[None, :], tl.where(real, k, 0.0).to(tl.bfloat16), mask=rk)
    mask_col = tl.where(real | (jw != 0)[None, :], 0.0, -big)
    tl.store(ko + DH + jw[None, :], mask_col.to(tl.bfloat16), mask=rk)
    vo = V + r[:, None] * E + h * DH
    tl.store(vo + j[None, :], tl.where(real, v, 0.0).to(tl.bfloat16), mask=rk)


@triton.jit
def _rope_tables(POS, INV, CS, SN, rows, DH: tl.constexpr, BR: tl.constexpr):
    """cos/sin [rows, DH] of the 2-D rope: lanes [0, DH/2) on the x position,
    [DH/2, DH) on y, each half cat(pos * INV, pos * INV) for INV's DH/4
    frequencies."""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    j = tl.arange(0, DH)
    live = (r < rows)[:, None]
    pos = tl.load(POS + r[:, None] * 2 + (j // (DH // 2))[None, :], mask=live, other=0)
    ang = pos.to(tl.float32) * tl.load(INV + (j % (DH // 4)))[None, :]
    at = r[:, None] * DH + j[None, :]
    tl.store(CS + at, tl.cos(ang), mask=live)
    tl.store(SN + at, tl.sin(ang), mask=live)


@triton.jit
def _pixel_rows(P, X, n, K: tl.constexpr, CH: tl.constexpr, BR: tl.constexpr):
    """X [*, K] bf16 = 2 (P - 0.5) on rows < n, 0 after; P f32 pixels."""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    live = (r < n)[:, None]
    for c0 in range(0, K, CH):
        c = c0 + tl.arange(0, CH)
        p = tl.load(P + r[:, None] * K + c[None, :], mask=live, other=0.5)
        tl.store(X + r[:, None] * K + c[None, :], (2.0 * (p - 0.5)).to(tl.bfloat16))


@triton.jit
def _add_positions(
    H, Y, ldy, POS, TX, TY, n, valid,
    N: tl.constexpr, CH: tl.constexpr, BR: tl.constexpr,
):  # fmt: skip
    """H [*, N] f32 = Y + TX[x] + TY[y] on rows < valid (x, y the patch's
    position, Y the patch GEMM's bf16 output), 0 after."""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    live = (r < valid)[:, None]
    px = tl.maximum(tl.load(POS + r * 2, mask=r < n, other=0), 0)
    py = tl.maximum(tl.load(POS + r * 2 + 1, mask=r < n, other=0), 0)
    for c0 in range(0, N, CH):
        c = c0 + tl.arange(0, CH)
        y = tl.load(Y + r[:, None] * ldy + c[None, :], mask=live, other=0.0)
        tx = tl.load(TX + px[:, None] * N + c[None, :], mask=live, other=0.0)
        ty = tl.load(TY + py[:, None] * N + c[None, :], mask=live, other=0.0)
        o = y.to(tl.float32) + (tx + ty)
        tl.store(H + r[:, None] * N + c[None, :], tl.where(live, o, 0.0))


@triton.jit
def _pool_norm(
    H, MEMBERS, X, n_out, k, scale, eps,
    MK: tl.constexpr, N: tl.constexpr, CH: tl.constexpr, BR: tl.constexpr,
):  # fmt: skip
    """X [*, N] bf16 = rmsnorm(the sum of H's rows MEMBERS[s, :] / k * scale)
    for output slots s < n_out, 0 after; MEMBERS -1 past a slot's rows,
    which are summed in order."""
    s = tl.program_id(0) * BR + tl.arange(0, BR)
    live = (s < n_out)[:, None]
    sq = tl.zeros((BR, CH), tl.float32)
    for c0 in range(0, N, CH):
        x = _pooled(H, MEMBERS, s, n_out, c0, k, scale, MK, N, CH, BR)
        sq += x * x
    inv = 1.0 / tl.sqrt(tl.sum(sq, 1) / N + eps)
    for c0 in range(0, N, CH):
        c = c0 + tl.arange(0, CH)
        x = _pooled(H, MEMBERS, s, n_out, c0, k, scale, MK, N, CH, BR) * inv[:, None]
        tl.store(
            X + s[:, None] * N + c[None, :], tl.where(live, x, 0.0).to(tl.bfloat16)
        )


@triton.jit
def _pooled(H, MEMBERS, s, n_out, c0, k, scale,
            MK: tl.constexpr, N: tl.constexpr, CH: tl.constexpr, BR: tl.constexpr):  # fmt: skip
    """Columns [c0, c0 + CH) of slots s: the sum of their rows / k * scale."""
    c = c0 + tl.arange(0, CH)
    live = (s < n_out)[:, None]
    acc = tl.zeros((BR, CH), tl.float32)
    for m in range(MK):
        row = tl.load(MEMBERS + s * MK + m, mask=s < n_out, other=-1)
        ok = (row >= 0)[:, None] & live
        acc += tl.load(
            H + tl.maximum(row, 0)[:, None] * N + c[None, :], mask=ok, other=0.0
        )
    return acc / k * scale


# ---- launchers -------------------------------------------------------------


def _ld(a):
    return a.strides[0] // a.itemsize


@_on_host
def rms_clip_tile(x, w, am, valid, k, eps, lo, hi):
    """x [M, k] f32 rows -> am A tiles of clip(rmsnorm(x) * w, lo, hi)."""
    hr, ks, tm, tk = am.shape
    _rms_clip_tile[(hr * tm // BR,)](
        Ptr(x), _ld(x), Ptr(w), Ptr(am), valid, eps, float(lo), float(hi),
        K=k, KS=ks, TM=tm, TK=tk, CH=_ch(k), BR=BR,
    )  # fmt: skip


@_on_host
def add_rms_clip(h, y, w, valid, eps, lo, hi):
    """h [rows, n] f32 += clip(y) / rms * w on rows < valid, 0 after."""
    rows, n = h.shape
    _add_rms_clip[(triton.cdiv(rows, BR),)](
        Ptr(h), Ptr(y), _ld(y), Ptr(w), valid, eps, float(lo), float(hi),
        N=n, CH=_ch(n), BR=BR,
    )  # fmt: skip


@_on_host
def clip_tile(x, am, k, lo, hi):
    hr, ks, tm, tk = am.shape
    _clip_tile[(hr * tm // BR,)](
        Ptr(x), _ld(x), Ptr(am), float(lo), float(hi),
        K=k, KS=ks, TM=tm, TK=tk, CH=_ch(k), BR=BR,
    )  # fmt: skip


@_on_host
def glu_clip_tile(g, u, am, k, g_hi, u_lo, u_hi, lo, hi):
    hr, ks, tm, tk = am.shape
    _glu_clip_tile[(hr * tm // BR,)](
        Ptr(g), _ld(g), Ptr(u), _ld(u), Ptr(am), float(g_hi), float(u_lo),
        float(u_hi), float(lo), float(hi),
        K=k, KS=ks, TM=tm, TK=tk, CH=_ch(k), BR=BR,
    )  # fmt: skip


@_on_host
def qkv_prep(qkv, cs, sn, qn, kn, q, k, v, valid, eps, qscale, big, bounds, nh, dh):
    """`hostops.c`'s qkv_prep: q [rows, nh dk], k [kv_rows, nh dk], v
    [kv_rows, nh dh] from the q|k|v GEMM's output rows."""
    rows, kv_rows = q.shape[0], k.shape[0]
    dk = q.shape[1] // nh
    top = max(rows, kv_rows)
    _qkv_prep[(triton.cdiv(top, BR), nh)](
        Ptr(qkv), _ld(qkv), Ptr(cs), Ptr(sn), Ptr(qn), Ptr(kn),
        Ptr(q), Ptr(k), Ptr(v), rows, valid, kv_rows, eps, float(qscale),
        float(big), *(float(b) for b in bounds),
        NH=nh, DH=dh, DK=dk, BR=BR,
    )  # fmt: skip


@_on_host
def rope_tables(pos, inv, dh):
    """cos, sin [rows, dh] f32 for positions pos [rows, 2] (int32)."""
    rows = pos.shape[0]
    cs = np.empty((rows, dh), np.float32)
    sn = np.empty((rows, dh), np.float32)
    _rope_tables[(triton.cdiv(rows, BR),)](
        Ptr(pos), Ptr(inv), Ptr(cs), Ptr(sn), rows, DH=dh, BR=BR
    )
    return cs, sn


@_on_host
def pixel_rows(p, x, n):
    """x [rows, k] bf16 = 2 (p - 0.5) on rows < n, 0 after; p [n, k] f32."""
    rows, k = x.shape
    _pixel_rows[(triton.cdiv(rows, BR),)](Ptr(p), Ptr(x), n, K=k, CH=_ch(k), BR=BR)


@_on_host
def add_positions(h, y, pos, table, n, valid):
    """h [rows, d] f32 = y + table[0][x] + table[1][y] on rows < valid."""
    rows, d = h.shape
    _add_positions[(triton.cdiv(rows, BR),)](
        Ptr(h), Ptr(y), _ld(y), Ptr(pos), Ptr(table[0]), Ptr(table[1]), n, valid,
        N=d, CH=_ch(d), BR=BR,
    )  # fmt: skip


@_on_host
def pool_norm(h, members, x, k, scale, eps):
    """x [rows, d] bf16 = rmsnorm(pooled slots of h): slot s the sum of h's
    rows members[s] (-1 past them) / k * scale; 0 on rows past the slots."""
    n, mk = members.shape
    d = h.shape[1]
    _pool_norm[(triton.cdiv(x.shape[0], BR),)](
        Ptr(h), Ptr(members), Ptr(x), n, float(k), float(scale), eps,
        MK=mk, N=d, CH=_ch(d), BR=BR,
    )  # fmt: skip
