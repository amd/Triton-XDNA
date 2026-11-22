# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""The models' own core functions: SiLU's GLU, the q/k/v bias or per-head
qk-norm and half-split RoPE, q padded with zero heads. The
rest (Q4 products, norms, attention, GELU's GLU) are the Gemma4
superkernel's.
"""

import triton
import triton.language as tl

from gemma4_superkernel.kernels import (
    add_bf,
    bf_parts,
    copy_n,
    exp2_f32,
    f32_key,
    key_f32,
    kv_chunk_mm,
    rms_norm,
    splat,
)


@triton.jit
def silu(gb):
    """silu of 32 bf16 lanes gb, as bf16: g sigmoid(g); with e = exp(-|g|),
    sigmoid(-|g|) = e / (1 + e), which keeps its precision where 1 - e /
    (1 + e) would cancel."""
    L2EH: tl.constexpr = 1.4453125
    L2EM: tl.constexpr = -0.00262451171875
    g = gb.to(tl.float32)
    one = tl.full([32], 1.0, tl.bfloat16)
    sgn = tl.where(g < 0, one, -one).to(tl.float32)
    # exp(-|g|) = 2^t, t to f32 precision
    ah, am = bf_parts(g * sgn)
    t = (am * L2EH + ah * L2EM) + ah * L2EH
    e = exp2_f32(t, 32)
    d = 1.0 + e
    # 1 / d for d in (1, 2]: a linear first guess, then two Newton steps
    r = 1.4571 - 0.5 * d
    r = r * (2.0 - d * r)
    r = r * (2.0 - d * r)
    sn = e * r
    return gb * tl.where(g < 0, sn, 1.0 - sn)


@triton.jit
def glu_silu(y, x, N: tl.constexpr):
    """y[i] = silu(gate[i]) up[i], x = [up (N) | gate (N)]."""
    k = tl.arange(0, 32)
    for c in range(N // 32):
        gb = tl.load(x + N + c * 32 + k)
        u = tl.load(x + c * 32 + k)
        tl.store(y + c * 32 + k, silu(gb) * u)


@triton.jit
def apply_rope(y, x, cs, H: tl.constexpr):
    """y = x rotated, half-split: y[i] = x[i] cos[i] - x[i + H] sin[i],
    y[i + H] = x[i] sin[i] + x[i + H] cos[i], i < H; cos at cs, sin at
    cs + H. 32 lanes at a time, or 16 where H is not 32s (partial
    rotary's 48)."""
    L: tl.constexpr = 32 if H % 32 == 0 else 16
    k = tl.arange(0, L)
    for c in range(H // L):
        o = c * L + k
        x1 = tl.load(x + o).to(tl.float32)
        x2 = tl.load(x + H + o).to(tl.float32)
        co = tl.load(cs + o).to(tl.float32)
        si = tl.load(cs + H + o).to(tl.float32)
        tl.store(y + o, (x1 * co - x2 * si).to(tl.bfloat16))
        tl.store(y + H + o, (x1 * si + x2 * co).to(tl.bfloat16))


@triton.jit
def rotate(y, x, cs, DH: tl.constexpr, RD: tl.constexpr):
    """A head rotated on its first RD dims (all of them, or partial
    rotary's), the rest as they are."""
    apply_rope(y, x, cs, RD // 2)
    if RD < DH:
        copy_n(y + RD, x + RD, DH - RD)


@triton.jit
def rope(
    q,
    kk,
    v,
    qkv,
    w,
    sq,
    NQ: tl.constexpr,
    NKV: tl.constexpr,
    DH: tl.constexpr,
    EPS: tl.constexpr,
    QPAD: tl.constexpr,
    BIAS: tl.constexpr,
    QKNORM: tl.constexpr,
    RD: tl.constexpr,
):
    """q/k/v from the QKV projection's [q | k | v] at qkv (air-FLM's
    rope.cc): with BIAS, the q/k/v biases added first; each q head q-normed
    (with QKNORM) and rotated, each k head k-normed (likewise) and rotated,
    v as it is. q comes out QPAD heads a kv head: its own, then zero heads,
    as air-FLM pads them; a head rotated on its first RD dims. w: [cos RD/2
    | sin RD/2] then [q_norm DH | k_norm DH] or the biases [q | k | v]. The
    heads are normed in place."""
    G: tl.constexpr = NQ // NKV  # q heads a kv head
    if BIAS:
        add_bf(qkv, w + DH, (NQ + 2 * NKV) * DH)
    if G == QPAD:
        for h in range(NQ):
            if QKNORM:
                rms_norm(qkv + h * DH, qkv + h * DH, w + DH, sq, DH, EPS)
            rotate(q + h * DH, qkv + h * DH, w, DH, RD)
    else:
        k = tl.arange(0, 32)
        for h in range(NKV):
            for j in range(G):
                o = (h * G + j) * DH
                if QKNORM:
                    rms_norm(qkv + o, qkv + o, w + DH, sq, DH, EPS)
                rotate(q + (h * QPAD + j) * DH, qkv + o, w, DH, RD)
            for c in range((QPAD - G) * DH // 32):
                tl.store(
                    q + (h * QPAD + G) * DH + c * 32 + k, tl.zeros([32], tl.bfloat16)
                )
    for h in range(NKV):
        o = NQ * DH + h * DH
        if QKNORM:
            rms_norm(qkv + o, qkv + o, w + 2 * DH, sq, DH, EPS)
        rotate(kk + h * DH, qkv + o, w, DH, RD)
    copy_n(v, qkv + (NQ + NKV) * DH, NKV * DH)


@triton.jit
def real_heads(o2, o, NCH: tl.constexpr, NH: tl.constexpr):
    """o2 [NCH][NH][8] = the first NH heads of o [NCH][4][8]."""
    if NH % 2 == 0:
        k = tl.arange(0, NH * 8)
        for c in range(NCH):
            tl.store(o2 + c * NH * 8 + k, tl.load(o + c * 32 + k))
    else:
        k = tl.arange(0, 8)
        for c in range(NCH):
            for h in range(NH):
                tl.store(o2 + (c * NH + h) * 8 + k, tl.load(o + c * 32 + h * 8 + k))


@triton.jit
def real_heads2(
    o2, o, NCH: tl.constexpr, NH: tl.constexpr, QG: tl.constexpr, QPAD: tl.constexpr
):
    """o2 [NCH][NH][8] = the NH real heads of o [2][NCH][4][8], whose heads
    are QPAD a kv head, the first QG of them real. A head's 8 at a time:
    o2's rows of NH 8s leave wider stores unaligned."""
    k = tl.arange(0, 8)
    for c in range(NCH):
        for r in tl.static_range(NH):
            p = (r // QG) * QPAD + r % QG  # the padded head
            tl.store(
                o2 + (c * NH + r) * 8 + k,
                tl.load(o + ((p // 4) * NCH + c) * 32 + (p % 4) * 8 + k),
            )


@triton.jit
def q40_group_b(xq, w, acc, t, g):
    """q4_group_b for Q4_0 weights, w = q scale with q signed: the nibbles
    hold q + 8, which the products take off, and no min (air-FLM's Q4_0
    _qmm_q4k_bf16). x as q4_prep leaves it at xq."""
    r = tl.arange(0, 32)
    tl.store(t + r, tl.zeros([32], tl.float32))
    for j in range(8):
        v = tl.load(w + 512 + g * 256 + j * 32 + r)
        tj = tl.load(t + r)
        c = xq + g * 32 + j * 4
        for s in tl.static_range(4):
            # column 4 j + s, q 2^(4 s): the nibble's word less 8 2^(4 s),
            # in integers (the top nibble's by flipping its sign bit), then
            # exactly as bf16
            if s == 3:
                q = ((v & -4096) ^ -32768).to(tl.bfloat16)
            else:
                q = ((v & (15 << (4 * s))) - (8 << (4 * s))).to(tl.bfloat16)
            tj += q.to(tl.float32) * splat(tl.load(c + s), 32)
        tl.store(t + r, tj)
    part = tl.load(t + r).to(tl.bfloat16).to(tl.float32)
    scale = tl.load(w + g * 32 + r).to(tl.bfloat16, bitcast=True).to(tl.float32)
    tl.store(acc + r, tl.load(acc + r) + part * scale)


@triton.jit
def q40_groups(xq, w, acc, t):
    """A Q4_0 block at w times the x q4_prep left at xq, added to acc."""
    for g in range(8):
        q40_group_b(xq, w, acc, t, g)


@triton.jit
def kv_block(vb, pc, yb, l, ct, NCH: tl.constexpr):
    """attn_kv_block_mm with l kept as [4][8], a head's partial sums across
    its row of 8 (its keys k and k + 8 a lane), added as vectors, not key by
    key; `kv_fin` sums the row."""
    r4 = tl.arange(0, 4)
    r8 = tl.arange(0, 8)
    r32 = tl.arange(0, 32)
    # each head's c across its row of 8 (through ct: a broadcast read does
    # not lower)
    for h in tl.static_range(4):
        tl.store(ct + h * 8 + r8, splat(tl.load(pc + 64 + h), 8))
    # (c is bf16: so the multiplies below are bf16's)
    cv = tl.load(ct + r32).to(tl.bfloat16).to(tl.float32)
    moved = 0
    for h in tl.static_range(4):
        cb = tl.load(pc + 64 + h).to(tl.int16, bitcast=True)
        moved += tl.where(cb != 16256, 1, 0)
    # p as attn_qk_block_mm leaves it, [half][head][8]
    p0 = tl.load(pc + r4[:, None] * 8 + r8[None, :])
    p1 = tl.load(pc + 32 + r4[:, None] * 8 + r8[None, :])
    # l rescaled only if a head's max moved: in three bf16 parts, whose
    # products with c are exact, so l keeps f32's precision as a scalar
    # multiply keeps it (the cores multiply f32 vectors only by bf16 ones)
    for _ in range(tl.where(moved > 0, 1, 0)):
        lv = tl.load(l + r32)
        lh, lm = bf_parts(lv)
        lr = ((lv - lh) - lm).to(tl.bfloat16).to(tl.float32)
        tl.store(l + r32, lh * cv + lm * cv + lr * cv)
    # each lane's sum of its two halves' p: a head's l the sum of its row
    pa = tl.load(pc + r32).to(tl.float32)
    pb = tl.load(pc + 32 + r32).to(tl.float32)
    tl.store(l + r32, tl.load(l + r32) + (pa + pb))
    for _ in range(tl.where(moved > 0, 1, 0)):
        for ch in range(NCH):
            yh, ym = bf_parts(tl.load(yb + ch * 32 + r32))
            a = tl.reshape(yh * cv + ym * cv, [4, 8])
            kv_chunk_mm(vb, yb, a, p0, p1, ch, NCH)
    for _ in range(tl.where(moved > 0, 0, 1)):
        for ch in range(NCH):
            a = tl.reshape(tl.load(yb + ch * 32 + r32), [4, 8])
            kv_chunk_mm(vb, yb, a, p0, p1, ch, NCH)


@triton.jit
def kv_fin(o, yb, l, ct, NCH: tl.constexpr):
    """attn_kv_fin_mm on kv_block's l, a head's the sum of its row."""
    r8 = tl.arange(0, 8)
    r32 = tl.arange(0, 32)
    for h in tl.static_range(4):
        lh = tl.load(l + h * 8)
        for j in tl.static_range(1, 8):
            lh += tl.load(l + h * 8 + j)
        tl.store(ct + h * 8 + r8, tl.full([8], 1.0 / lh, tl.float32))
    rh, rl = bf_parts(tl.load(ct + r32))
    for ch in range(NCH):
        yh, ym = bf_parts(tl.load(yb + ch * 32 + r32))
        v = ym * rh + yh * rl + yh * rh
        tl.store(o + ch * 32 + r32, v.to(tl.bfloat16))


@triton.jit
def qk_block(
    qb,
    kb,
    mst,
    pc,
    acc,
    tmp,
    sb,
    mb,
    blk,
    L,
    LO,
    NCH: tl.constexpr,
    SCALE: tl.constexpr,
):
    """attn_qk_block_mm, its running max taken as `qk_softmax` takes it.
    One 16-key block of 4 heads' attention scores, as 4x8x8 matrix
    products: q blocked [NCH][4][8] (8 columns of the 4 heads, NCH = head
    dim / 8), the keys [2][NCH][8][8] (8 keys by 8 columns). The scores are
    scaled by SCALE in bf16, as air-FLM's are. Keys outside [LO, L) are
    masked.
    Updates the running max mst [4] (f32) and writes to pc: p = exp(s - m)
    as bf16 [half][head][8], then c [4] = exp(m_old - m_new), the factor that
    rescales the sums so far."""
    r4 = tl.arange(0, 4)
    r8 = tl.arange(0, 8)
    # the two halves' sums in buffers of their own (acc, tmp)
    a0 = acc + r4[:, None] * 8 + r8[None, :]
    a1 = tmp + r4[:, None] * 8 + r8[None, :]
    tl.store(a0, tl.zeros([4, 8], tl.float32))
    tl.store(a1, tl.zeros([4, 8], tl.float32))
    # the sums in tmp across the chunks (a loop, not unrolled: a product's
    # code is large)
    for c in range(NCH):
        # K read [key][column]: the transpose folds into the contraction
        qc = tl.load(qb + c * 32 + r4[:, None] * 8 + r8[None, :])
        k0 = tl.load(kb + c * 64 + r8[:, None] * 8 + r8[None, :])
        k1 = tl.load(kb + NCH * 64 + c * 64 + r8[:, None] * 8 + r8[None, :])
        tl.store(a0, tl.dot(qc, tl.trans(k0), tl.load(a0)))
        tl.store(a1, tl.dot(qc, tl.trans(k1), tl.load(a1)))
    # the scores as the products leave them, [half][head][8]
    r32 = tl.arange(0, 32)
    tl.store(sb + r32, tl.load(acc + r32))
    tl.store(sb + 32 + r32, tl.load(tmp + r32))
    if SCALE != 1.0:
        for hp in tl.static_range(2):
            s = tl.load(sb + hp * 32 + r32).to(tl.bfloat16).to(tl.float32) * SCALE
            tl.store(sb + hp * 32 + r32, s.to(tl.bfloat16).to(tl.float32))
    # a block the window's ends cut: its keys outside [LO, L) masked, a
    # scalar at a time (a lane mask would be kept in memory)
    for _ in range(tl.where((blk * 16 < LO) | (blk * 16 + 16 > L), 1, 0)):
        for hh in range(4):
            for key in range(16):
                pos = blk * 16 + key
                at = sb + (key // 8) * 32 + hh * 8 + key % 8
                tl.store(at, tl.where((pos < L) & (pos >= LO), tl.load(at), -1.0e4))
    qk_softmax(mst, pc, tmp, sb, mb)


@triton.jit
def qk_softmax(mst, pc, tmp, sb, mb):
    """qk_softmax_mm with the two halves' keys maxed as vectors of integer
    keys first (in mb, before mb takes the maxima), so that a head's
    max is over 8 lanes, not 16."""
    k = tl.arange(0, 32)
    r8 = tl.arange(0, 8)
    tl.store(tmp + k, tl.full([32], -1.0e4, tl.float32))
    tl.store(tmp + 32 + k, tl.zeros([32], tl.float32))
    # 16 lanes at a time: 32 lanes of i32 do not compare
    k16 = tl.arange(0, 16)
    for c in tl.static_range(2):
        o = c * 16 + k16
        km = tl.maximum(f32_key(tl.load(sb + o)), f32_key(tl.load(sb + 32 + o)))
        tl.store(mb + o, km.to(tl.float32, bitcast=True))
    # each head's max over its 8, then its row of mb the new max (its 8 read
    # before they are written)
    for h in range(4):
        m = tl.load(mb + h * 8).to(tl.int32, bitcast=True)
        for key in range(1, 8):
            m = tl.maximum(m, tl.load(mb + h * 8 + key).to(tl.int32, bitcast=True))
        mo = tl.load(mst + h)
        mn = key_f32(tl.maximum(m, f32_key(mo)))
        tl.store(mst + h, mn)
        tl.store(tmp + h, mo)
        tl.store(tmp + 32 + h, mn)
        row = tl.full([8], mn, tl.float32)
        tl.store(mb + h * 8 + r8, row)
        tl.store(mb + 32 + h * 8 + r8, row)
    L2EH: tl.constexpr = 1.4453125
    L2EM: tl.constexpr = -0.00262451171875
    for hp in range(2):
        d = tl.load(sb + hp * 32 + k) - tl.load(mb + hp * 32 + k)
        dh, dm = bf_parts(d)
        t = (dm * L2EH + dh * L2EM) + dh * L2EH
        tl.store(pc + hp * 32 + k, exp2_f32(t, 32))
    d = tl.load(tmp + k) - tl.load(tmp + 32 + k)
    dh, dm = bf_parts(d)
    t = (dm * L2EH + dh * L2EM) + dh * L2EH
    tl.store(pc + 64 + k, exp2_f32(t, 32))
