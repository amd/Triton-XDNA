# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""The host side of an NPU prefill, as Triton kernels run by triton-cpu.

Between the device's GEMMs and attention the host normalizes, rotates,
multiplies and adds rows, and each result goes straight into the layout the
next device op reads: a GEMM's A tiles, attention's q blocks, a layer's KV
records. Each kernel here is one such step.

Arguments are numpy arrays or a `FusedProgram`'s buffers. The launchers run
on the CPU whichever driver is active around them.

Layouts:
- A tiles: [M / TM, KS, TM, TK], element (r, k) at
  ((r // TM * KS + k // TK) * TM + r % TM) * TK + k % TK. Rows from `t` to M
  are written as zeros; columns past K (the device's K padding) are left
  alone.
- 8x8 blocks of a [rows, cols] matrix, row-block major: element (r, c) at
  (r // 8 * (cols / 8) + c // 8) * 64 + r % 8 * 8 + c % 8.
"""

import functools

import triton
import triton.language as tl
from triton.backends.amd_triton_npu.driver_scope import driver_scope


class Ptr:
    """A numpy array (or a view of one) as a Triton pointer argument. An
    argument that is one already (a `FusedProgram` buffer) is passed as is."""

    __slots__ = ("p", "dtype")

    def __new__(cls, a):
        return a if hasattr(a, "data_ptr") else super().__new__(cls)

    def __init__(self, a):
        if self is not a:
            self.p, self.dtype = a.ctypes.data, a.dtype.name

    def data_ptr(self):
        return self.p


#: rows per program for the row kernels
BR = 4


@triton.jit
def _tile_off(r, c, KS: tl.constexpr, TM: tl.constexpr, TK: tl.constexpr):
    return ((r // TM * KS + c // TK) * TM + r % TM) * TK + c % TK


@triton.jit
def _blk_off(r, c, COLS: tl.constexpr):
    return (r // 8 * (COLS // 8) + c // 8) * 64 + r % 8 * 8 + c % 8


@triton.jit
def _inv_rms(X, ldx, r, live, eps, N: tl.constexpr, CH: tl.constexpr):
    """1 / rms of rows r of X [*, ldx], over N columns, in f32."""
    acc = tl.zeros((r.shape[0], CH), tl.float32)
    for c0 in range(0, N, CH):
        c = c0 + tl.arange(0, CH)
        x = tl.load(X + r[:, None] * ldx + c[None, :], mask=live, other=0.0)
        x = x.to(tl.float32)
        acc += x * x
    return 1.0 / tl.sqrt(tl.sum(acc, 1) / N + eps)


@triton.jit
def _rms_tile(
    X,
    ldx,
    W,
    AM,
    t,
    eps,
    K: tl.constexpr,
    KS: tl.constexpr,
    TM: tl.constexpr,
    TK: tl.constexpr,
    CH: tl.constexpr,
    BR: tl.constexpr,
    NORM: tl.constexpr,
    HAS_W: tl.constexpr,
):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    live = (r < t)[:, None]
    if NORM:
        inv = _inv_rms(X, ldx, r, live, eps, K, CH)
    for c0 in range(0, K, CH):
        c = c0 + tl.arange(0, CH)
        y = tl.load(X + r[:, None] * ldx + c[None, :], mask=live, other=0.0)
        y = y.to(tl.float32)
        if NORM:
            y = y * inv[:, None]
        if HAS_W:
            y = y * tl.load(W + c).to(tl.float32)[None, :]
        tl.store(AM + _tile_off(r[:, None], c[None, :], KS, TM, TK), y.to(tl.bfloat16))


@triton.jit
def _add_rms(
    R,
    Y,
    ldy,
    W,
    W2,
    AM,
    t,
    s,
    eps,
    N: tl.constexpr,
    KS: tl.constexpr,
    TM: tl.constexpr,
    TK: tl.constexpr,
    CH: tl.constexpr,
    BR: tl.constexpr,
    HAS_S: tl.constexpr,
    TILE: tl.constexpr,
):
    """R[:t] = (R + rms(Y) * W) [* s]; then TILE 1 tiles R, TILE 2 rms(R) * W2."""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    live = (r < t)[:, None]
    inv = _inv_rms(Y, ldy, r, live, eps, N, CH)
    for c0 in range(0, N, CH):
        c = c0 + tl.arange(0, CH)
        y = tl.load(Y + r[:, None] * ldy + c[None, :], mask=live, other=0.0)
        y = y.to(tl.float32) * inv[:, None] * tl.load(W + c).to(tl.float32)[None, :]
        o = tl.load(R + r[:, None] * N + c[None, :], mask=live, other=0.0) + y
        if HAS_S:
            o = o * s
        tl.store(R + r[:, None] * N + c[None, :], o, mask=live)
        if TILE == 1:
            tl.store(
                AM + _tile_off(r[:, None], c[None, :], KS, TM, TK), o.to(tl.bfloat16)
            )
    if TILE == 2:
        inv2 = _inv_rms(R, N, r, live, eps, N, CH)
        for c0 in range(0, N, CH):
            c = c0 + tl.arange(0, CH)
            o = tl.load(R + r[:, None] * N + c[None, :], mask=live, other=0.0)
            o = o * inv2[:, None] * tl.load(W2 + c).to(tl.float32)[None, :]
            tl.store(
                AM + _tile_off(r[:, None], c[None, :], KS, TM, TK), o.to(tl.bfloat16)
            )


@triton.jit
def _mul_tile(
    A,
    lda,
    B,
    ldb,
    AM,
    t,
    K: tl.constexpr,
    KS: tl.constexpr,
    TM: tl.constexpr,
    TK: tl.constexpr,
    CH: tl.constexpr,
    BR: tl.constexpr,
):
    """Rows of a * b into the A tiles."""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    live = (r < t)[:, None]
    for c0 in range(0, K, CH):
        c = c0 + tl.arange(0, CH)
        a = tl.load(A + r[:, None] * lda + c[None, :], mask=live, other=0.0)
        b = tl.load(B + r[:, None] * ldb + c[None, :], mask=live, other=0.0)
        y = a.to(tl.float32) * b.to(tl.float32)
        tl.store(AM + _tile_off(r[:, None], c[None, :], KS, TM, TK), y.to(tl.bfloat16))


@triton.jit
def _o_tile(
    OB,
    AM,
    t,
    h0,
    DH: tl.constexpr,
    M: tl.constexpr,
    KS: tl.constexpr,
    TM: tl.constexpr,
    TK: tl.constexpr,
    BR: tl.constexpr,
):
    """Head h0 + program_id(1) of attention's output (8x8 blocks of [M, DH]
    per head in OB) into the A tiles of [t, heads * DH]."""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    hh = tl.program_id(1)
    live = (r < t)[:, None]
    c = tl.arange(0, DH)
    o = tl.load(
        OB + hh * M * DH + _blk_off(r[:, None], c[None, :], DH), mask=live, other=0.0
    )
    col = (h0 + hh) * DH + c
    tl.store(AM + _tile_off(r[:, None], col[None, :], KS, TM, TK), o)


@triton.jit
def _rope_table(INV, CS, SN, r0, HALF: tl.constexpr, BR: tl.constexpr):
    """cos and sin [rows, HALF] of positions r0 + r times INV."""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    h = tl.arange(0, HALF)
    ang = (r0 + r).to(tl.float32)[:, None] * tl.load(INV + h)[None, :]
    o = r[:, None] * HALF + h[None, :]
    tl.store(CS + o, tl.cos(ang))
    tl.store(SN + o, tl.sin(ang))


@triton.jit
def _norm_rope(
    X, ldx, col0, W, CS, SN, r, live, eps, DH: tl.constexpr, ROPE: tl.constexpr
):
    """Rows r of one head (columns col0 .. col0 + DH of X): rms, times W if
    given, rotated by RoPE (rows r of the cos / sin table) when ROPE. Returns
    the halves (i, i + DH / 2), f32 [rows, DH / 2] each."""
    h = tl.arange(0, DH // 2)
    a = tl.load(X + r[:, None] * ldx + col0 + h[None, :], mask=live, other=0.0)
    b = tl.load(
        X + r[:, None] * ldx + col0 + DH // 2 + h[None, :], mask=live, other=0.0
    )
    a, b = a.to(tl.float32), b.to(tl.float32)
    inv = 1.0 / tl.sqrt((tl.sum(a * a, 1) + tl.sum(b * b, 1)) / DH + eps)
    a, b = a * inv[:, None], b * inv[:, None]
    if W is not None:
        a = a * tl.load(W + h).to(tl.float32)[None, :]
        b = b * tl.load(W + DH // 2 + h).to(tl.float32)[None, :]
    if ROPE:
        o = r[:, None] * (DH // 2) + h[None, :]
        cs, sn = tl.load(CS + o), tl.load(SN + o)
        a, b = a * cs - b * sn, b * cs + a * sn
    return a, b


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
    M: tl.constexpr,
    BR: tl.constexpr,
):
    """Heads h0 + program_id(1) of q: normed, roped, scaled, as the head's 8x8
    blocks of [M, DH] in QB [heads, M, DH]."""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    hh = tl.program_id(1)
    live = (r < t)[:, None]
    a, b = _norm_rope(Q, ldq, (h0 + hh) * DH, W, CS, SN, r, live, eps, DH, True)
    h = tl.arange(0, DH // 2)
    base = QB + hh * M * DH
    zero = tl.zeros_like(a)
    a, b = tl.where(live, a * scale, zero), tl.where(live, b * scale, zero)
    tl.store(base + _blk_off(r[:, None], h[None, :], DH), a.to(tl.bfloat16))
    tl.store(base + _blk_off(r[:, None], DH // 2 + h[None, :], DH), b.to(tl.bfloat16))


@triton.jit
def _rec_store(
    REC,
    r,
    c,
    x,
    live,
    keep,
    DH: tl.constexpr,
    LKP: tl.constexpr,
    DVT: tl.constexpr,
    IS_V: tl.constexpr,
):
    """x at rows r, columns c of a KV record stream: per LKP rows, K's 8x8
    blocks of [LKP, DH], then V's, DH / DVT column chunks DVT wide. Rows
    where `keep` and not `live` are written as zeros."""
    rr = r % LKP
    off = (r // LKP) * (2 * LKP * DH)
    if IS_V:
        off = off + LKP * DH + (c // DVT) * (LKP * DVT) + _blk_off(rr, c % DVT, DVT)
    else:
        off = off + _blk_off(rr, c, DH)
    x = tl.where(live, x, tl.zeros_like(x))
    tl.store(REC + off, x.to(tl.bfloat16), mask=keep)


@triton.jit
def _kv_post(
    K,
    V,
    ld,
    W,
    CS,
    SN,
    KF,
    VF,
    REC,
    t,
    rows,
    eps,
    DH: tl.constexpr,
    LKP: tl.constexpr,
    DVT: tl.constexpr,
    BR: tl.constexpr,
):
    """The KV head of a chunk: k normed by W and roped, v normed. Rows < t go
    to KF and VF (f32 [t, DH]); rows < `rows` (t rounded up to whole records)
    to the records at REC, zero past t."""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    live = (r < t)[:, None]
    keep = (r < rows)[:, None]
    h = tl.arange(0, DH // 2)
    ka, kb = _norm_rope(K, ld, 0, W, CS, SN, r, live, eps, DH, True)
    va, vb = _norm_rope(V, ld, 0, None, CS, SN, r, live, eps, DH, False)
    lo, hi = h[None, :], DH // 2 + h[None, :]
    tl.store(KF + r[:, None] * DH + lo, ka, mask=live)
    tl.store(KF + r[:, None] * DH + hi, kb, mask=live)
    tl.store(VF + r[:, None] * DH + lo, va, mask=live)
    tl.store(VF + r[:, None] * DH + hi, vb, mask=live)
    _rec_store(REC, r[:, None], lo, ka, live, keep, DH, LKP, DVT, False)
    _rec_store(REC, r[:, None], hi, kb, live, keep, DH, LKP, DVT, False)
    _rec_store(REC, r[:, None], lo, va, live, keep, DH, LKP, DVT, True)
    _rec_store(REC, r[:, None], hi, vb, live, keep, DH, LKP, DVT, True)


@triton.jit
def _ple(
    P,
    ldp,
    W,
    TBL,
    PLI,
    t,
    pscale,
    iscale,
    eps,
    NL: tl.constexpr,
    PD: tl.constexpr,
    BR: tl.constexpr,
):
    """Layer program_id(1)'s per-layer inputs of rows < t:
    (rms(P * pscale) * W + TBL) * iscale."""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    L = tl.program_id(1)
    live = (r < t)[:, None]
    c = tl.arange(0, PD)
    p = tl.load(P + r[:, None] * ldp + L * PD + c[None, :], mask=live, other=0.0)
    p = p.to(tl.float32) * pscale
    inv = 1.0 / tl.sqrt(tl.sum(p * p, 1) / PD + eps)
    y = p * inv[:, None] * tl.load(W + c).to(tl.float32)[None, :]
    o = r[:, None] * (NL * PD) + L * PD + c[None, :]
    y = (y + tl.load(TBL + o, mask=live, other=0.0)) * iscale
    tl.store(PLI + o, y, mask=live)


@triton.jit
def _embed(Q, S, IDS, OUT, scale, W: tl.constexpr, CH: tl.constexpr):
    """Row program_id(0) of OUT [n, W]: row IDS[i] of an int8 table times its
    per-32-column f32 scales, times `scale`."""
    i = tl.program_id(0)
    tok = tl.load(IDS + i).to(tl.int64)
    for c0 in range(0, W, CH):
        c = c0 + tl.arange(0, CH)
        q = tl.load(Q + tok * W + c).to(tl.float32)
        s = tl.load(S + tok * (W // 32) + c // 32)
        tl.store(OUT + i * W + c, q * s * scale)


@triton.jit
def _rms_rows(X, W, OUT, eps, N: tl.constexpr, CH: tl.constexpr):
    """Row program_id(0) of X [*, N]: rms * W into OUT."""
    r = tl.program_id(0) + tl.arange(0, 1)
    live = (r >= 0)[:, None]
    inv = _inv_rms(X, N, r, live, eps, N, CH)
    for c0 in range(0, N, CH):
        c = c0 + tl.arange(0, CH)
        x = tl.load(X + r[:, None] * N + c[None, :])
        y = x * inv[:, None] * tl.load(W + c).to(tl.float32)[None, :]
        tl.store(OUT + r[:, None] * N + c[None, :], y.to(OUT.dtype.element_ty))


@triton.jit
def _softcap(
    Y, OUT, cap, NR: tl.constexpr, TF: tl.constexpr, FG: tl.constexpr, CH: tl.constexpr
):
    """Logits from the LM GEMV's output [NC, TF, NR, FG] (logit order
    [NC, NR, TF, FG]): cap * tanh(y / cap)."""
    i = tl.program_id(0) * CH + tl.arange(0, CH)
    g = i % FG
    tf = i // FG % TF
    nr = i // (FG * TF) % NR
    nc = i // (FG * TF * NR)
    y = tl.load(Y + ((nc * TF + tf) * NR + nr) * FG + g).to(tl.float32)
    # tanh(x) = 1 - 2 / (exp(2x) + 1), exact at both saturations.
    e = tl.exp(2.0 * (y / cap))
    tl.store(OUT + i, cap * (1.0 - 2.0 / (e + 1.0)))


# ---- launchers -------------------------------------------------------------


def _on_host(fn):
    """Launch on the CPU whichever device is active around the call."""

    @functools.wraps(fn)
    def run(*a, **kw):
        with driver_scope("cpu"):
            return fn(*a, **kw)

    return run


def _ch(n):
    return 256 if n % 256 == 0 else 128


@_on_host
def rms_tile(x, w, am, t, k, eps, norm=True):
    """x [>= t, k] (f32 or bf16 rows, any row stride) -> A tiles am."""
    hr, ks, tm, tk = am.shape
    _rms_tile[(hr * tm // BR,)](
        Ptr(x), x.strides[0] // x.itemsize, Ptr(w if w is not None else x),
        Ptr(am), t, eps,
        K=k, KS=ks, TM=tm, TK=tk, CH=_ch(k), BR=BR, NORM=norm, HAS_W=w is not None,
    )  # fmt: skip


@_on_host
def add_rms(res, y, w, t, eps, scale=None, am=None, w2=None):
    """res[:t] = (res + rms(y) * w) [* scale, a float]; y the rows of a GEMM's
    output.
    With am: also tiles res (w2 None) or rms(res) * w2 into it."""
    n = res.shape[1]
    tile = 0 if am is None else (2 if w2 is not None else 1)
    hr, ks, tm, tk = am.shape if am is not None else (1, 1, 32, 128)
    rows = hr * tm if am is not None else t
    _add_rms[(triton.cdiv(rows, BR),)](
        Ptr(res), Ptr(y), y.strides[0] // y.itemsize, Ptr(w),
        Ptr(w2 if w2 is not None else w), Ptr(am if am is not None else res),
        t, 1.0 if scale is None else float(scale), eps,
        N=n, KS=ks, TM=tm, TK=tk, CH=_ch(n), BR=BR, HAS_S=scale is not None, TILE=tile,
    )  # fmt: skip


@_on_host
def mul_tile(a, b, am, t, k):
    """a * b ([>= t, k] rows each, any row strides) -> A tiles am."""
    hr, ks, tm, tk = am.shape
    _mul_tile[(hr * tm // BR,)](
        Ptr(a), a.strides[0] // a.itemsize, Ptr(b), b.strides[0] // b.itemsize,
        Ptr(am), t, K=k, KS=ks, TM=tm, TK=tk, CH=_ch(k), BR=BR,
    )  # fmt: skip


@_on_host
def rope_table(inv, cs, sn, r0):
    """cs, sn [M, len(inv)] f32: cos and sin of positions r0 .. r0 + M."""
    _rope_table[(cs.shape[0] // BR,)](
        Ptr(inv), Ptr(cs), Ptr(sn), r0, HALF=cs.shape[1], BR=BR,
    )  # fmt: skip


@_on_host
def q_post(q, w, rope, qb, t, h0, scale, eps):
    """Heads h0 .. h0 + len(qb) of q (the q GEMM's output rows) into
    qb [heads, M, dh] as 8x8 blocks; rope = (cos, sin) from rope_table."""
    hg, m, dh = qb.shape
    _q_post[(m // BR, hg)](
        Ptr(q), q.strides[0] // q.itemsize, Ptr(w), Ptr(rope[0]), Ptr(rope[1]),
        Ptr(qb), t, h0, scale, eps, DH=dh, M=m, BR=BR,
    )  # fmt: skip


@_on_host
def kv_post(k, v, w, rope, kf, vf, rec, t, lkp, dvt, eps):
    """One KV head: kf, vf [t, dh] f32 and its records from rec on (the
    chunk starts a record)."""
    dh = kf.shape[1]
    rows = -(-t // lkp) * lkp
    _kv_post[(triton.cdiv(rows, BR),)](
        Ptr(k), Ptr(v), k.strides[0] // k.itemsize, Ptr(w), Ptr(rope[0]),
        Ptr(rope[1]), Ptr(kf), Ptr(vf), Ptr(rec), t, rows, eps,
        DH=dh, LKP=lkp, DVT=dvt, BR=BR,
    )  # fmt: skip


@_on_host
def o_tile(ob, am, t, h0):
    """Attention output ob [heads, M, dh] (8x8 blocks) -> columns
    h0 * dh .. of the A tiles am."""
    hg, m, dh = ob.shape
    hr, ks, tm, tk = am.shape
    _o_tile[(m // BR, hg)](
        Ptr(ob), Ptr(am), t, h0, DH=dh, M=m, KS=ks, TM=tm, TK=tk, BR=BR
    )


@_on_host
def ple(p, w, tbl, pli, t, pscale, iscale, eps):
    """pli[:t] [t, layers, d] from the model projection p's rows and the
    per-layer token embeddings tbl [t, layers * d]."""
    _, nl, pd = pli.shape
    _ple[(triton.cdiv(t, BR), nl)](
        Ptr(p), p.strides[0] // p.itemsize, Ptr(w), Ptr(tbl), Ptr(pli),
        t, pscale, iscale, eps, NL=nl, PD=pd, BR=BR,
    )  # fmt: skip


@_on_host
def embed(q, s, ids, out, scale=1.0):
    """out [len(ids), W] f32 = rows ids of int8 table q [V, W] * scales s
    [V, W / 32]."""
    _embed[(len(ids),)](
        Ptr(q), Ptr(s), Ptr(ids), Ptr(out), scale, W=out.shape[1], CH=_ch(out.shape[1])
    )


@_on_host
def rms_rows(x, w, out, eps):
    n = x.shape[-1]
    _rms_rows[(x.size // n,)](Ptr(x), Ptr(w), Ptr(out), eps, N=n, CH=_ch(n))


@_on_host
def softcap(y, out, cap, nr, tf, fg):
    """out [V] f32 = cap * tanh(logits / cap), the logits unpacked from y."""
    ch = 1024
    _softcap[(out.size // ch,)](Ptr(y), Ptr(out), cap, NR=nr, TF=tf, FG=fg, CH=ch)
