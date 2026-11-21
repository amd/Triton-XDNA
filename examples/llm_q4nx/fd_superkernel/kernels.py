# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""The models' own core functions: SiLU's GLU, the per-head qk-norm and
half-split RoPE, q padded to the attention kernels' 4 heads a kv head. The
rest (Q4 products, norms, attention, GELU's GLU) are the Gemma4
superkernel's.
"""

import triton
import triton.language as tl

from gemma4_superkernel.kernels import bf_parts, copy_n, exp2_f32, rms_norm


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
    cs + H."""
    k = tl.arange(0, 32)
    for c in range(H // 32):
        o = c * 32 + k
        x1 = tl.load(x + o).to(tl.float32)
        x2 = tl.load(x + H + o).to(tl.float32)
        co = tl.load(cs + o).to(tl.float32)
        si = tl.load(cs + H + o).to(tl.float32)
        tl.store(y + o, (x1 * co - x2 * si).to(tl.bfloat16))
        tl.store(y + H + o, (x1 * si + x2 * co).to(tl.bfloat16))


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
):
    """q/k/v from the QKV projection's [q | k | v] at qkv: each q head
    q-normed and rotated, each k head k-normed and rotated, v as it is
    (air-FLM's rope.cc). q comes out 4 heads a kv head: its own, then zero
    heads (Gemma3's 2 a kv head, as air-FLM pads them). w: [cos DH/2 | sin
    DH/2 | q_norm DH | k_norm DH]. The heads are normed in place."""
    G: tl.constexpr = NQ // NKV  # q heads a kv head
    if G == 4:
        for h in range(NQ):
            rms_norm(qkv + h * DH, qkv + h * DH, w + DH, sq, DH, EPS)
            apply_rope(q + h * DH, qkv + h * DH, w, DH // 2)
    else:
        k = tl.arange(0, 32)
        for h in range(NKV):
            for j in range(G):
                o = (h * G + j) * DH
                rms_norm(qkv + o, qkv + o, w + DH, sq, DH, EPS)
                apply_rope(q + (h * 4 + j) * DH, qkv + o, w, DH // 2)
            for c in range((4 - G) * DH // 32):
                tl.store(q + (h * 4 + G) * DH + c * 32 + k, tl.zeros([32], tl.bfloat16))
    for h in range(NKV):
        o = NQ * DH + h * DH
        rms_norm(qkv + o, qkv + o, w + 2 * DH, sq, DH, EPS)
        apply_rope(kk + h * DH, qkv + o, w, DH // 2)
    copy_n(v, qkv + (NQ + NKV) * DH, NKV * DH)


@triton.jit
def real_heads(o2, o, NCH: tl.constexpr, NH: tl.constexpr):
    """o2 [NCH][NH][8] = the first NH heads of o [NCH][4][8]."""
    k = tl.arange(0, NH * 8)
    for c in range(NCH):
        tl.store(o2 + c * NH * 8 + k, tl.load(o + c * 32 + k))
