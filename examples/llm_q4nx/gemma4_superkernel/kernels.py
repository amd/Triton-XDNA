# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""What the decode's cores compute, as Triton functions on their L1 buffers:
the Q4 projection blocks, the norms, RoPE, attention, GELU and the PLE
products. Written for AIE2P's vector unit: f32 vectors of 16 or 32 lanes,
bf16 products, so f32 precision is kept by splitting values into bf16 parts.
"""

import triton
import triton.language as tl


@triton.jit
def splat(v, N: tl.constexpr):
    """A bf16 scalar as N lanes, widened."""
    return tl.full([N], v, tl.bfloat16).to(tl.float32)


@triton.jit
def bf_parts(a):
    """f32 a as two bf16 values (as f32) whose sum is a to 16 bits."""
    h = a.to(tl.bfloat16).to(tl.float32)
    return h, (a - h).to(tl.bfloat16).to(tl.float32)


@triton.jit
def rms_norm(y, x, w, sq, D: tl.constexpr, EPS: tl.constexpr):
    """y = x w / sqrt(mean(x^2) + EPS), each D bf16 at its pointer, x w
    scaled in f32 and rounded once; sq a buffer of 64 f32."""
    k = tl.arange(0, 32)
    tl.store(sq + k, tl.zeros([32], tl.float32))
    for c in range(D // 32):
        v = tl.load(x + c * 32 + k).to(tl.float32)
        tl.store(sq + k, tl.load(sq + k) + v * v)
    # Fold the upper 16 lanes onto the lower ones with a 32-lane add (the
    # f32 sum of a vector is a 16-lane op, and narrower f32 adds lose an
    # operand), then sum lanes 0 .. 15.
    h = tl.arange(0, 16)
    tl.store(sq + 32 + h, tl.load(sq + 16 + h))
    tl.store(sq + 48 + h, tl.zeros([16], tl.float32))
    tl.store(sq + k, tl.load(sq + k) + tl.load(sq + 32 + k))
    ms = tl.sum(tl.load(sq + h)) * (1.0 / D) + EPS
    # 1 / sqrt(ms): the bit-level first guess, then two Newton steps
    # not >> 1: mlir-aie makes a scalar shift a rounding SRS
    i = 0x5F3759DF - ms.to(tl.int32, bitcast=True) // 2
    r = i.to(tl.float32, bitcast=True)
    r = r * (1.5 - 0.5 * ms * r * r)
    r = r * (1.5 - 0.5 * ms * r * r)
    # r as three bf16 parts, x w (a product of two bf16) as two: their
    # products are the cores' bf16 MACs
    rh = r.to(tl.bfloat16)
    rr = r - rh.to(tl.float32)
    rm = rr.to(tl.bfloat16)
    rl = (rr - rm.to(tl.float32)).to(tl.bfloat16)
    for c in range(D // 32):
        o = c * 32 + k
        xw = tl.load(x + o).to(tl.float32) * tl.load(w + o).to(tl.float32)
        xh = xw.to(tl.bfloat16).to(tl.float32)
        xm = (xw - xh).to(tl.bfloat16).to(tl.float32)
        s = xh * splat(rl, 32) + xm * splat(rm, 32)
        s += xh * splat(rm, 32) + xm * splat(rh, 32)
        s += xh * splat(rh, 32)
        tl.store(y + o, s.to(tl.bfloat16))


@triton.jit
def residual(y, a, b, D: tl.constexpr):
    """y = a + b, D bf16."""
    k = tl.arange(0, 32)
    for c in range(D // 32):
        o = c * 32 + k
        tl.store(y + o, tl.load(a + o) + tl.load(b + o))


@triton.jit
def gelu(gb):
    """gelu_tanh of 32 bf16 lanes gb, as bf16: g sigmoid(2 z), z = c (g +
    0.044715 g^3), z to f32 precision in bf16 MACs; with e = exp(-2 |z|),
    sigmoid(-2 |z|) = e / (1 + e), which keeps its precision where 1 +
    tanh(z) would cancel."""
    # c and c 0.044715, log2(e), each as two bf16 parts
    C1H: tl.constexpr = 0.796875
    C1M: tl.constexpr = 0.001007080078125
    C3H: tl.constexpr = 0.03564453125
    C3M: tl.constexpr = 0.000032901763916015625
    L2EH: tl.constexpr = 1.4453125
    L2EM: tl.constexpr = -0.00262451171875
    g = gb.to(tl.float32)
    g2h, g2m = bf_parts(g * g)
    g3 = g2h * g + g2m * g
    g3h, g3m = bf_parts(g3)
    z = (C1M * g + C3M * g3h) + (C3H * g3m + C3H * g3h) + C1H * g
    two = tl.full([32], 2.0, tl.bfloat16)
    sgn = tl.where(z < 0, two, -two).to(tl.float32)
    zh, zm = bf_parts(z)
    # exp(-2 |z|) = 2^t, t to f32 precision
    ah, am = bf_parts(zh * sgn + zm * sgn)
    t = (am * L2EH + ah * L2EM) + ah * L2EH
    e = exp2_f32(t, 32)
    d = 1.0 + e
    # 1 / d for d in (1, 2]: a linear first guess, then two Newton steps
    r = 1.4571 - 0.5 * d
    r = r * (2.0 - d * r)
    r = r * (2.0 - d * r)
    sn = e * r
    return gb * tl.where(z < 0, sn, 1.0 - sn)


@triton.jit
def glu(y, x, N: tl.constexpr):
    """y[i] = gelu_tanh(gate[i]) up[i], x = [up (N) | gate (N)]."""
    k = tl.arange(0, 32)
    for c in range(N // 32):
        gb = tl.load(x + N + c * 32 + k)
        u = tl.load(x + c * 32 + k)
        tl.store(y + c * 32 + k, gelu(gb) * u)


@triton.jit
def copy_zero(dst, src, N: tl.constexpr, Z: tl.constexpr):
    """dst[0 .. N) = src[0 .. N), then dst[N .. N + Z) = 0, 32 at a time;
    the copy as arithmetic (+ 0, exact): a plain one is not vectorized."""
    k = tl.arange(0, 32)
    for c in tl.static_range(N // 32):
        cv = tl.load(src + c * 32 + k).to(tl.float32) + 0.0
        tl.store(dst + c * 32 + k, cv.to(tl.bfloat16))
    for c in tl.static_range(Z // 32):
        tl.store(dst + N + c * 32 + k, tl.zeros([32], tl.bfloat16))


@triton.jit
def spread_swa(sp, qkv):
    """A sliding layer's 12 compact 256-wide heads (8 q, 2 k, 2 v) at qkv
    into 512-wide slots at sp: q and k as [lo | 0 | hi | 0], so the (i, i +
    256) pairing of RoPE rotates the head's own (i, i + 128) pairs, v as
    [v | 0]."""
    for h in range(10):
        copy_zero(sp + h * 512, qkv + h * 256, 128, 128)
        copy_zero(sp + h * 512 + 256, qkv + h * 256 + 128, 128, 128)
    for h in range(10, 12):
        copy_zero(sp + h * 512, qkv + h * 256, 256, 256)


@triton.jit
def apply_rope(y, x, cs):
    """y = x rotated: y[i] = x[i] cos[i] - x[i + 256] sin[i], y[i + 256] = x[i]
    sin[i] + x[i + 256] cos[i], i < 256; cos at cs, sin at cs + 256."""
    k = tl.arange(0, 32)
    for c in range(8):
        o = c * 32 + k
        x1 = tl.load(x + o).to(tl.float32)
        x2 = tl.load(x + 256 + o).to(tl.float32)
        co = tl.load(cs + o).to(tl.float32)
        si = tl.load(cs + 256 + o).to(tl.float32)
        tl.store(y + o, (x1 * co - x2 * si).to(tl.bfloat16))
        tl.store(y + 256 + o, (x1 * si + x2 * co).to(tl.bfloat16))


@triton.jit
def v_norm(y, x, sq, inv):
    """y = x / sqrt(sum(x^2) * inv + 1e-6), over 512 x, the scale rounded
    to bf16 as the reference does; inv is 1 over the head's real width."""
    k = tl.arange(0, 32)
    tl.store(sq + k, tl.zeros([32], tl.float32))
    for c in range(16):
        v = tl.load(x + c * 32 + k).to(tl.float32)
        tl.store(sq + k, tl.load(sq + k) + v * v)
    h = tl.arange(0, 16)
    tl.store(sq + 32 + h, tl.load(sq + 16 + h))
    tl.store(sq + 48 + h, tl.zeros([16], tl.float32))
    tl.store(sq + k, tl.load(sq + k) + tl.load(sq + 32 + k))
    ms = tl.sum(tl.load(sq + h)) * inv + 1e-6
    # not >> 1: mlir-aie makes a scalar shift a rounding SRS
    i = 0x5F3759DF - ms.to(tl.int32, bitcast=True) // 2
    r = i.to(tl.float32, bitcast=True)
    r = r * (1.5 - 0.5 * ms * r * r)
    r = r * (1.5 - 0.5 * ms * r * r)
    rb = r.to(tl.bfloat16)
    for c in range(16):
        o = c * 32 + k
        tl.store(y + o, (tl.load(x + o).to(tl.float32) * splat(rb, 32)).to(tl.bfloat16))


@triton.jit
def rope_heads(q, kk, v, src, w, sq, inv):
    """rope's per-head work on the 6144 at src (heads in their slots)."""
    for h in range(8):
        rms_norm(src + h * 512, src + h * 512, w + 512, sq, 512, 1e-6)
        apply_rope(q + h * 512, src + h * 512, w)
    for h in range(2):
        rms_norm(src + 4096 + h * 512, src + 4096 + h * 512, w + 1024, sq, 512, 1e-6)
        apply_rope(kk + h * 512, src + 4096 + h * 512, w)
    for h in range(2):
        v_norm(v + h * 512, src + 5120 + h * 512, sq, inv)


@triton.jit
def rope(q, kk, v, qkv, sp, w, sq, swa):
    """Gemma4's q/k/v from the QKV projection's 6144 at qkv: sliding heads
    (swa 1) spread into their slots, each q head q-normed and rotated, each k
    head k-normed and rotated, each v head v-normed. w: [cos 256 | sin 256 |
    q_norm 512 | k_norm 512]; sp a 6144 scratch. Full heads are worked on
    in place (a copy would be element by element)."""
    # loops of 0 or 1 trips, not ifs: pointers keep their structure
    for _ in range(swa):
        spread_swa(sp, qkv)
        rope_heads(q, kk, v, sp, w, sq, 2.0 / 512)
    for _ in range(1 - swa):
        rope_heads(q, kk, v, qkv, w, sq, 1.0 / 512)


@triton.jit
def exp2_f32(t, N: tl.constexpr):
    """2^t for an f32 vector t of N lanes, as bf16: 2^n 2^f, n the integer
    nearest t, the cores' exp2 exact at integers only, 2^f a polynomial."""
    n = (t + 12582912.0) - 12582912.0
    fb = (t - n).to(tl.bfloat16).to(tl.float32)
    p = (fb * 0.0096435546875 + 0.0555041).to(tl.bfloat16).to(tl.float32)
    p = (fb * p + 0.2402265).to(tl.bfloat16).to(tl.float32)
    p = (fb * p + 0.6931472).to(tl.bfloat16).to(tl.float32)
    p = fb * p + 1.0
    e2n = tl.exp2(n).to(tl.bfloat16).to(tl.float32)
    return (e2n * p.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)


@triton.jit
def attn_qk_block_mm(qb, kb, mst, pc, acc, tmp, sb, mb, blk, L, LO):
    """One 16-key block of 4 heads' attention scores, as 4x8x8 matrix
    products: q blocked [64][4][8] (8 columns of the 4 heads), the keys
    [2][64][8][8] (8 keys by 8 columns). Keys outside [LO, L) are masked.
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
    for c in range(64):
        # K read [key][column]: the transpose folds into the contraction
        qc = tl.load(qb + c * 32 + r4[:, None] * 8 + r8[None, :])
        k0 = tl.load(kb + c * 64 + r8[:, None] * 8 + r8[None, :])
        k1 = tl.load(kb + 4096 + c * 64 + r8[:, None] * 8 + r8[None, :])
        tl.store(a0, tl.dot(qc, tl.trans(k0), tl.load(a0)))
        tl.store(a1, tl.dot(qc, tl.trans(k1), tl.load(a1)))
    # the scores as the products leave them, [half][head][8]
    r32 = tl.arange(0, 32)
    tl.store(sb + r32, tl.load(acc + r32))
    tl.store(sb + 32 + r32, tl.load(tmp + r32))
    # a block the window's ends cut: its keys outside [LO, L) masked, a
    # scalar at a time (a lane mask would be kept in memory)
    for _ in range(tl.where((blk * 16 < LO) | (blk * 16 + 16 > L), 1, 0)):
        for hh in range(4):
            for key in range(16):
                pos = blk * 16 + key
                at = sb + (key // 8) * 32 + hh * 8 + key % 8
                tl.store(at, tl.where((pos < L) & (pos >= LO), tl.load(at), -1.0e4))
    qk_softmax_mm(mst, pc, tmp, sb, mb)


@triton.jit
def f32_key(x):
    """An int32 ordered as the f32 x (scalar): the cores' scalar float
    compares are calls, their integer ones instructions."""
    b = x.to(tl.int32, bitcast=True)
    return tl.where(b < 0, b ^ 2147483647, b)


@triton.jit
def key_f32(k):
    """The inverse of f32_key."""
    return tl.where(k < 0, k ^ 2147483647, k).to(tl.float32, bitcast=True)


@triton.jit
def qk_softmax_mm(mst, pc, tmp, sb, mb):
    """From the 4 heads' masked scores at sb, [half][head][8]: the running
    max mst updated, and pc's p and c (see attn_qk_block_mm). The maxima are
    taken on integer keys and m_old - m_new as a vector: no scalar float
    arithmetic."""
    k = tl.arange(0, 32)
    r8 = tl.arange(0, 8)
    # m_old at tmp[h], m_new at tmp[32 + h]; the other lanes give c = 0
    tl.store(tmp + k, tl.full([32], -1.0e4, tl.float32))
    tl.store(tmp + 32 + k, tl.zeros([32], tl.float32))
    for h in range(4):
        km = f32_key(tl.load(sb + h * 8))
        for key in tl.static_range(1, 16):
            km = tl.maximum(
                km, f32_key(tl.load(sb + (key // 8) * 32 + h * 8 + key % 8))
            )
        mo = tl.load(mst + h)
        mn = key_f32(tl.maximum(km, f32_key(mo)))
        tl.store(mst + h, mn)
        # the new max where the head's scores are
        tl.store(mb + h * 8 + r8, tl.full([8], mn, tl.float32))
        tl.store(mb + 32 + h * 8 + r8, tl.full([8], mn, tl.float32))
        tl.store(tmp + h, mo)
        tl.store(tmp + 32 + h, mn)
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


@triton.jit
def attn_kv_block_mm(vb, pc, yb, lst, ct):
    """One 16-key block of 4 heads' attention sums, p V as 4x8x8 matrix
    products: the sums yb [64][4][8] (f32) and l lst [4] (f32) rescaled by c,
    then yb += p V and l += sum(p); the values blocked [2][64][8][8] (8 keys
    by 8 columns), p and c at pc as attn_qk_block_mm writes them; ct a buffer
    of 32 f32."""
    r4 = tl.arange(0, 4)
    r8 = tl.arange(0, 8)
    for h in range(4):
        ls = tl.load(lst + h) * tl.load(pc + 64 + h).to(tl.float32)
        for key in tl.static_range(16):
            ls += tl.load(pc + (key // 8) * 32 + h * 8 + key % 8).to(tl.float32)
        tl.store(lst + h, ls)
    # each head's c across its row of 8 (through ct: a broadcast read does
    # not lower)
    for h in tl.static_range(4):
        tl.store(ct + h * 8 + r8, splat(tl.load(pc + 64 + h), 8))
    # (c is bf16: so the multiplies below are bf16's)
    r32 = tl.arange(0, 32)
    cv = tl.load(ct + r32).to(tl.bfloat16).to(tl.float32)
    # The sums rescaled (in two bf16 parts, exactly) only if a head's max
    # moved, as air-FLM's lazy rescale: once the max settles, c is 1 (bf16
    # 0x3F80) for every head. (Integer compares: scalar float ones are calls.)
    moved = 0
    for h in tl.static_range(4):
        cb = tl.load(pc + 64 + h).to(tl.int16, bitcast=True)
        moved += tl.where(cb != 16256, 1, 0)
    # p as attn_qk_block_mm leaves it, [half][head][8]
    p0 = tl.load(pc + r4[:, None] * 8 + r8[None, :])
    p1 = tl.load(pc + 32 + r4[:, None] * 8 + r8[None, :])
    for _ in range(tl.where(moved > 0, 1, 0)):
        for ch in range(64):
            # (flat: AIE2P computes 1-D vectors)
            yh, ym = bf_parts(tl.load(yb + ch * 32 + r32))
            a = tl.reshape(yh * cv + ym * cv, [4, 8])
            kv_chunk_mm(vb, yb, a, p0, p1, ch)
    for _ in range(tl.where(moved > 0, 0, 1)):
        for ch in range(64):
            a = tl.reshape(tl.load(yb + ch * 32 + r32), [4, 8])
            kv_chunk_mm(vb, yb, a, p0, p1, ch)


@triton.jit
def kv_chunk_mm(vb, yb, a, p0, p1, ch):
    """The block's keys added to chunk ch's sums a ([4][8]) into yb."""
    r8 = tl.arange(0, 8)
    v0 = tl.load(vb + ch * 64 + r8[:, None] * 8 + r8[None, :])
    v1 = tl.load(vb + 4096 + ch * 64 + r8[:, None] * 8 + r8[None, :])
    a = tl.dot(p0, v0, a)
    a = tl.dot(p1, v1, a)
    tl.store(yb + ch * 32 + tl.arange(0, 32), tl.reshape(a, [32]))


@triton.jit
def attn_kv_fin_mm(o, yb, lst, ct):
    """o [64][4][8] (bf16) = y / l of the blocked sums yb, in
    32-lane steps (AIE2P computes f32 vectors 16 lanes or more); ct a
    buffer of 32 f32."""
    r8 = tl.arange(0, 8)
    r32 = tl.arange(0, 32)
    for h in tl.static_range(4):
        tl.store(ct + h * 8 + r8, tl.full([8], 1.0 / tl.load(lst + h), tl.float32))
    rh, rl = bf_parts(tl.load(ct + r32))
    for ch in range(64):
        yh, ym = bf_parts(tl.load(yb + ch * 32 + r32))
        v = ym * rh + yh * rl + yh * rh
        tl.store(o + ch * 32 + r32, v.to(tl.bfloat16))


@triton.jit
def qk_m_out(mst, pcm):
    """The running max (f32, 32 lanes) for the kv core in a score message:
    two bf16 parts, whose sum is it to 16 bits."""
    k = tl.arange(0, 32)
    mh, ml = bf_parts(tl.load(mst + k))
    tl.store(pcm + k, mh.to(tl.bfloat16))
    tl.store(pcm + 32 + k, ml.to(tl.bfloat16))


@triton.jit
def kv_part_out(yb, lst, pcm):
    """A part's sums for the merge: yb [64][4][8] (f32), then l and m (32
    lanes each) after them, m from qk_m_out's message pcm."""
    k = tl.arange(0, 32)
    tl.store(yb + 2048 + k, tl.load(lst + k))
    m = tl.load(pcm + k).to(tl.float32) + tl.load(pcm + 32 + k).to(tl.float32)
    tl.store(yb + 2080 + k, m)


@triton.jit
def attn_merge(o, ba, bb, lst, ct, cs):
    """o [64][4][8] (bf16) from two parts' kv_part_out (over disjoint keys):
    m = max(m_a, m_b), each part's sums and l scaled by exp(m_part - m),
    added, then divided. lst, ct and cs buffers of 32, 32 and 96 f32."""
    k = tl.arange(0, 32)
    r8 = tl.arange(0, 8)
    L2EH: tl.constexpr = 1.4453125
    L2EM: tl.constexpr = -0.00262451171875
    ma = tl.load(ba + 2080 + k)
    mb = tl.load(bb + 2080 + k)
    # m per head on integer keys (a vector maximum's NaN handling ors two
    # lane masks, which do not lower); the other lanes m_a
    tl.store(cs + k, ma)
    for h in tl.static_range(4):
        ka = f32_key(tl.load(ba + 2080 + h))
        kb = f32_key(tl.load(bb + 2080 + h))
        tl.store(cs + h, key_f32(tl.maximum(ka, kb)))
    m = tl.load(cs + k)
    for part in tl.static_range(2):
        d = (ma if part == 0 else mb) - m
        dh, dm = bf_parts(d)
        t = (dm * L2EH + dh * L2EM) + dh * L2EH
        tl.store(cs + part * 32 + k, exp2_f32(t, 32).to(tl.float32))
    # l = l_a f_a + l_b f_b (f bf16, the products bf16's)
    fa = tl.load(cs + k).to(tl.bfloat16).to(tl.float32)
    fb = tl.load(cs + 32 + k).to(tl.bfloat16).to(tl.float32)
    lah, lam = bf_parts(tl.load(ba + 2048 + k))
    lbh, lbm = bf_parts(tl.load(bb + 2048 + k))
    tl.store(lst + k, lah * fa + lam * fa + lbh * fb + lbm * fb)
    # each head's factors across its 8 lanes of a chunk
    for h in tl.static_range(4):
        tl.store(ct + h * 8 + r8, tl.full([8], tl.load(cs + h), tl.float32))
        tl.store(cs + 64 + h * 8 + r8, tl.full([8], tl.load(cs + 32 + h), tl.float32))
    fa = tl.load(ct + k).to(tl.bfloat16).to(tl.float32)
    fb = tl.load(cs + 64 + k).to(tl.bfloat16).to(tl.float32)
    for ch in range(64):
        yah, yam = bf_parts(tl.load(ba + ch * 32 + k))
        ybh, ybm = bf_parts(tl.load(bb + ch * 32 + k))
        tl.store(ba + ch * 32 + k, yah * fa + yam * fa + ybh * fb + ybm * fb)
    attn_kv_fin_mm(o, ba, lst, ct)


@triton.jit
def q4_group_b(xq, w, acc, t, sx, g):
    """Group g (32 columns) of a q4 block at w times x, added to acc: the
    group's 32 rows' nibble products in bf16 MACs, the partial rounded to
    bf16, scaled, plus the group's min times its sum of x. x as q4_prep
    leaves it at xq and sx; t a buffer of 32 f32."""
    r = tl.arange(0, 32)
    tl.store(t + r, tl.zeros([32], tl.float32))
    for j in range(8):
        v = tl.load(w + 512 + g * 256 + j * 32 + r)
        tj = tl.load(t + r)
        c = xq + g * 32 + j * 4
        for s in tl.static_range(4):
            # column 4 j + s times 2^(4 s), exactly, as bf16; the top
            # nibble's word converted unsigned
            if s == 3:
                q = (v & -4096).to(tl.uint16, bitcast=True).to(tl.bfloat16)
            else:
                q = (v & (15 << (4 * s))).to(tl.bfloat16)
            tj += q.to(tl.float32) * splat(tl.load(c + s), 32)
        tl.store(t + r, tj)
    part = tl.load(t + r).to(tl.bfloat16).to(tl.float32)
    red = splat(tl.load(sx + g).to(tl.bfloat16), 32)
    scale = tl.load(w + g * 32 + r).to(tl.bfloat16, bitcast=True).to(tl.float32)
    mn = tl.load(w + 256 + g * 32 + r).to(tl.bfloat16, bitcast=True).to(tl.float32)
    tl.store(acc + r, tl.load(acc + r) + part * scale + mn * red)


@triton.jit
def q4_shift_table(tab):
    """q4_prep's table: column c's exponent shift, 512 (c mod 4), as
    int16 (stores of constants: arange arithmetic does not lower here)."""
    for i in range(64):
        for s in tl.static_range(4):
            tl.store(tab + 4 * i + s, (s * 512))


@triton.jit
def q4_prep(xb, xq, sx, tab):
    """What a q4 block's product needs of its 256 x at xb, the same for
    every block of those columns: x scaled for q4_group_b in xq (256 bf16;
    column c of a word holds its nibble times 2^(4 (c mod 4)), so x[c] is
    scaled by 2^(-4 (c mod 4)), a word's 4 adjacent) and each group's sum
    of x in sx (8 f32); tab q4_shift_table's."""
    k = tl.arange(0, 256)
    xs = tl.reshape(tl.load(xb + k), [8, 32]).to(tl.float32)
    tl.store(sx + tl.arange(0, 8), tl.sum(xs, axis=1))
    # times 2^(-4 (c mod 4)): the bf16 exponent less 4 (c mod 4); x too
    # small for that (below 2^-110, or 0) as 0
    xb16 = tl.load(xb + k).to(tl.int16, bitcast=True)
    sh = tl.load(tab + k)
    ok = (xb16 & 32640) > sh
    tl.store(xq + k, tl.where(ok, xb16 - sh, 0).to(tl.bfloat16, bitcast=True))


@triton.jit
def q4_groups(xq, w, acc, t, sx):
    """A q4 block at w times the x q4_prep left at xq and sx, added to acc."""
    for g in range(8):
        q4_group_b(xq, w, acc, t, sx, g)


@triton.jit
def mvm_half(acc, w, x):
    """mvm_blk over half of a block's columns: w [128][32], x [128]."""
    r = tl.arange(0, 32)
    for k0 in range(8):
        a = tl.load(acc + r)
        for kk in tl.static_range(16):
            c = k0 * 16 + kk
            xv = splat(tl.load(x + c), 32)
            a += tl.load(w + c * 32 + r).to(tl.float32) * xv
        tl.store(acc + r, a)


@triton.jit
def mvm_blk(acc, w, x):
    """acc [32] (f32) += W x for a bf16 block W of 32 rows by 256, laid out
    [256][32] at w, x [256] at x: a column at a time, x broadcast, the sums
    in a register across 16 columns."""
    r = tl.arange(0, 32)
    for k0 in range(16):
        a = tl.load(acc + r)
        for kk in tl.static_range(16):
            c = k0 * 16 + kk
            xv = splat(tl.load(x + c), 32)
            a += tl.load(w + c * 32 + r).to(tl.float32) * xv
        tl.store(acc + r, a)


@triton.jit
def flush32(y, acc):
    """y [32] (bf16) = acc [32] (f32)."""
    r = tl.arange(0, 32)
    tl.store(y + r, tl.load(acc + r).to(tl.bfloat16))


@triton.jit
def scale_bf(y, s, N: tl.constexpr):
    """y [N] *= s, a bf16 scalar, in bf16."""
    r = tl.arange(0, 32)
    for c in range(N // 32):
        tl.store(y + c * 32 + r, tl.load(y + c * 32 + r) * s)


@triton.jit
def add_bf(y, b, N: tl.constexpr):
    r = tl.arange(0, 32)
    for c in range(N // 32):
        tl.store(y + c * 32 + r, tl.load(y + c * 32 + r) + tl.load(b + c * 32 + r))


@triton.jit
def mul_bf(y, b, N: tl.constexpr):
    r = tl.arange(0, 32)
    for c in range(N // 32):
        tl.store(y + c * 32 + r, tl.load(y + c * 32 + r) * tl.load(b + c * 32 + r))


@triton.jit
def gelu_inplace(g, N: tl.constexpr):
    r = tl.arange(0, 32)
    for c in range(N // 32):
        tl.store(g + c * 32 + r, gelu(tl.load(g + c * 32 + r)))


@triton.jit
def copy_n(dst, src, N: tl.constexpr):
    """N bf16 from src to dst, as arithmetic (+ 0, exact): the cores'
    schedule leaves a plain copy unvectorized, an element at a time."""
    r = tl.arange(0, 32)
    for c in range(N // 32):
        v = tl.load(src + c * 32 + r).to(tl.float32) + 0.0
        tl.store(dst + c * 32 + r, v.to(tl.bfloat16))


@triton.jit
def zero32(acc):
    tl.store(acc + tl.arange(0, 32), tl.zeros([32], tl.float32))
