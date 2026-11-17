# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""The dense models' host steps that `host_kernels` has no form of, as
Triton kernels run by triton-cpu, in the layouts `host_kernels` describes.

A head's RoPE may cover only its first `rot` columns (Phi-4), as pairs
(i, i + rot / 2), and q / k may or may not be RMS-normed (Qwen3, Gemma3);
v is neither. A residual add may or may not norm what it adds (Gemma3).
"""

import triton
import triton.language as tl

from host_kernels import BR, Ptr, _blk_off, _ch, _inv_rms, _on_host, _rec_store
from host_kernels import _tile_off


@triton.jit
def _head(
    X,
    ldx,
    col0,
    W,
    CS,
    SN,
    r,
    live,
    eps,
    DH: tl.constexpr,
    HALF: tl.constexpr,
    PH: tl.constexpr,
    PT: tl.constexpr,
    NORM: tl.constexpr,
    ROPE: tl.constexpr,
):
    """Rows r of one head (columns col0 .. col0 + DH of X): rms times W when
    NORM, its first 2 HALF columns rotated by rows r of the cos / sin table
    when ROPE. Returns the parts [0, HALF), [HALF, 2 HALF) and [2 HALF, DH),
    f32 [rows, PH], [rows, PH] and [rows, PT], each masked past its end."""
    h = tl.arange(0, PH)
    p = tl.arange(0, PT)
    hm = (h < HALF)[None, :]
    pm = (p < DH - 2 * HALF)[None, :]
    base = X + r[:, None] * ldx + col0
    a = tl.load(base + h[None, :], mask=live & hm, other=0.0).to(tl.float32)
    b = tl.load(base + HALF + h[None, :], mask=live & hm, other=0.0).to(tl.float32)
    c = tl.load(base + 2 * HALF + p[None, :], mask=live & pm, other=0.0)
    c = c.to(tl.float32)
    if NORM:
        ss = tl.sum(a * a, 1) + tl.sum(b * b, 1) + tl.sum(c * c, 1)
        inv = 1.0 / tl.sqrt(ss / DH + eps)
        a = a * inv[:, None] * tl.load(W + h, mask=h < HALF, other=0.0)[None, :]
        b = b * inv[:, None] * tl.load(W + HALF + h, mask=h < HALF, other=0.0)[None, :]
        wc = tl.load(W + 2 * HALF + p, mask=p < DH - 2 * HALF, other=0.0)
        c = c * inv[:, None] * wc[None, :]
    if ROPE:
        o = r[:, None] * HALF + h[None, :]
        cs = tl.load(CS + o, mask=hm, other=0.0)
        sn = tl.load(SN + o, mask=hm, other=0.0)
        a, b = a * cs - b * sn, b * cs + a * sn
    return a, b, c


@triton.jit
def _q_post(
    Q,
    ldq,
    W,
    CS,
    SN,
    QB,
    t,
    h0,
    scale,
    eps,
    DH: tl.constexpr,
    HALF: tl.constexpr,
    PH: tl.constexpr,
    PT: tl.constexpr,
    M: tl.constexpr,
    BR: tl.constexpr,
    NORM: tl.constexpr,
):
    """Head h0 + program_id(1) of q: normed when NORM, roped, scaled, as
    bf16, as the head's 8x8 blocks of [M, DH] in QB [heads, M, DH]; rows
    past t zero."""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    hh = tl.program_id(1)
    live = (r < t)[:, None]
    a, b, c = _head(
        Q, ldq, (h0 + hh) * DH, W, CS, SN, r, live, eps,
        DH, HALF, PH, PT, NORM, True,
    )  # fmt: skip
    h = tl.arange(0, PH)[None, :]
    p = tl.arange(0, PT)[None, :]
    base = QB + hh * M * DH
    rr = r[:, None]
    zero = tl.zeros_like(a)
    a, b = tl.where(live, a * scale, zero), tl.where(live, b * scale, zero)
    c = tl.where(live, c * scale, tl.zeros_like(c))
    tl.store(base + _blk_off(rr, h, DH), a.to(tl.bfloat16), mask=h < HALF)
    tl.store(base + _blk_off(rr, HALF + h, DH), b.to(tl.bfloat16), mask=h < HALF)
    tl.store(
        base + _blk_off(rr, 2 * HALF + p, DH), c.to(tl.bfloat16), mask=p < DH - 2 * HALF
    )


@triton.jit
def _kv_post(
    X,
    ld,
    kc,
    vc,
    W,
    CS,
    SN,
    KF,
    VF,
    REC,
    span,
    t,
    rows,
    col0,
    eps,
    DH: tl.constexpr,
    HALF: tl.constexpr,
    PH: tl.constexpr,
    PT: tl.constexpr,
    LKP: tl.constexpr,
    DVT: tl.constexpr,
    BR: tl.constexpr,
    NORM: tl.constexpr,
    LDF: tl.constexpr,
):
    """KV head program_id(1) of a chunk, the heads DH apart from columns kc
    (k) and vc (v) of X: k normed by W when NORM and roped, v as is, each
    rounded to bf16. Rows < t go to the head's columns from col0 on of KF and
    VF (f32 [t, LDF]); rows < `rows` (t rounded up to whole records) to its
    records, `span` elements apart from REC on, zero past t."""
    j = tl.program_id(1)
    kc = kc + j * DH
    vc = vc + j * DH
    col0 = col0 + j * DH
    REC = REC + j * span
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    live = (r < t)[:, None]
    keep = (r < rows)[:, None]
    ka, kb, kt = _head(X, ld, kc, W, CS, SN, r, live, eps, DH, HALF, PH, PT, NORM, True)  # fmt: skip
    va, vb, vt = _head(X, ld, vc, W, CS, SN, r, live, eps, DH, HALF, PH, PT, False, False)  # fmt: skip
    h = tl.arange(0, PH)[None, :]
    p = tl.arange(0, PT)[None, :]
    rr = r[:, None]
    _kv_part(KF, VF, REC, rr, h, ka, va, live, keep, h < HALF, col0, DH, LKP, DVT, LDF)
    _kv_part(KF, VF, REC, rr, HALF + h, kb, vb, live, keep, h < HALF, col0, DH, LKP, DVT, LDF)  # fmt: skip
    _kv_part(KF, VF, REC, rr, 2 * HALF + p, kt, vt, live, keep, p < DH - 2 * HALF, col0, DH, LKP, DVT, LDF)  # fmt: skip


@triton.jit
def _kv_part(
    KF,
    VF,
    REC,
    r,
    c,
    k,
    v,
    live,
    keep,
    m,
    col0,
    DH: tl.constexpr,
    LKP: tl.constexpr,
    DVT: tl.constexpr,
    LDF: tl.constexpr,
):
    """Columns c (where m) of a KV head's rows r: to KF and VF, and its
    records."""
    off = r * LDF + col0 + c
    tl.store(KF + off, k.to(tl.bfloat16).to(tl.float32), mask=live & m)
    tl.store(VF + off, v.to(tl.bfloat16).to(tl.float32), mask=live & m)
    _rec_store(REC, r, c, k, live, keep & m, DH, LKP, DVT, False)
    _rec_store(REC, r, c, v, live, keep & m, DH, LKP, DVT, True)


@triton.jit
def _add_rms(
    R,
    Y,
    ldy,
    W,
    W2,
    AM,
    t,
    eps,
    N: tl.constexpr,
    KS: tl.constexpr,
    TM: tl.constexpr,
    TK: tl.constexpr,
    CH: tl.constexpr,
    BR: tl.constexpr,
    POST: tl.constexpr,
    TILE: tl.constexpr,
):
    """R[:t] += rms(Y) * W when POST, else Y; then when TILE, rms(R) * W2
    into the A tiles."""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    live = (r < t)[:, None]
    if POST:
        inv = _inv_rms(Y, ldy, r, live, eps, N, CH)
    for c0 in range(0, N, CH):
        c = c0 + tl.arange(0, CH)
        y = tl.load(Y + r[:, None] * ldy + c[None, :], mask=live, other=0.0)
        y = y.to(tl.float32)
        if POST:
            y = y * inv[:, None] * tl.load(W + c).to(tl.float32)[None, :]
        o = tl.load(R + r[:, None] * N + c[None, :], mask=live, other=0.0) + y
        tl.store(R + r[:, None] * N + c[None, :], o, mask=live)
    if TILE:
        inv2 = _inv_rms(R, N, r, live, eps, N, CH)
        for c0 in range(0, N, CH):
            c = c0 + tl.arange(0, CH)
            o = tl.load(R + r[:, None] * N + c[None, :], mask=live, other=0.0)
            o = o * inv2[:, None] * tl.load(W2 + c).to(tl.float32)[None, :]
            tl.store(
                AM + _tile_off(r[:, None], c[None, :], KS, TM, TK), o.to(tl.bfloat16)
            )


# ---- launchers -------------------------------------------------------------


def _pow2(n):
    return 1 << max(int(n) - 1, 0).bit_length()


def _rot(dh, rot):
    half = rot // 2
    return dict(HALF=half, PH=_pow2(half), PT=_pow2(max(dh - rot, 1)))


@_on_host
def q_post(q, w, rope, qb, t, h0, scale, eps):
    """Heads h0 .. h0 + len(qb) of q (the qkv GEMM's output rows) into
    qb [heads, M, dh] as 8x8 blocks: normed by w unless it is None, roped by
    rope = (cos, sin) [M, rot / 2], scaled."""
    hg, m, dh = qb.shape
    rot = 2 * rope[0].shape[1]
    _q_post[(m // BR, hg)](
        Ptr(q), q.strides[0] // q.itemsize, Ptr(w if w is not None else q),
        Ptr(rope[0]), Ptr(rope[1]), Ptr(qb), t, h0, scale, eps,
        DH=dh, **_rot(dh, rot), M=m, BR=BR, NORM=w is not None,
    )  # fmt: skip


@_on_host
def kv_post(x, kc, vc, w, rope, kf, vf, rec, span, nh, t, lkp, dvt, eps, col0, dh):
    """nh KV heads, k at columns kc .. kc + nh * dh and v at vc .. of x (the
    qkv GEMM's output rows): columns col0 .. col0 + nh * dh of kf, vf (f32 [t,
    kv heads * dh]) and each head's records, `span` elements apart from rec
    on (the chunk starts a record)."""
    rows = -(-t // lkp) * lkp
    rot = 2 * rope[0].shape[1]
    _kv_post[(triton.cdiv(rows, BR), nh)](
        Ptr(x), x.strides[0] // x.itemsize, kc, vc, Ptr(w if w is not None else x),
        Ptr(rope[0]), Ptr(rope[1]), Ptr(kf), Ptr(vf), Ptr(rec), span, t, rows, col0,
        eps,
        DH=dh, **_rot(dh, rot), LKP=lkp, DVT=dvt, BR=BR, NORM=w is not None,
        LDF=kf.shape[1],
    )  # fmt: skip


@_on_host
def add_rms(res, y, w, t, eps, am=None, w2=None):
    """res[:t] += rms(y) * w, or y when w is None; y the rows of a GEMM's
    output. With am: also rms(res) * w2 into its A tiles."""
    n = res.shape[1]
    hr, ks, tm, tk = am.shape if am is not None else (1, 1, 32, 128)
    rows = hr * tm if am is not None else t
    _add_rms[(triton.cdiv(rows, BR),)](
        Ptr(res), Ptr(y), y.strides[0] // y.itemsize, Ptr(w if w is not None else res),
        Ptr(w2 if w2 is not None else res), Ptr(am if am is not None else res), t, eps,
        N=n, KS=ks, TM=tm, TK=tk, CH=_ch(n), BR=BR, POST=w is not None,
        TILE=am is not None,
    )  # fmt: skip
