# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Gemma3's own core functions: the per-head qk-norm and RoPE, its q padded
to the attention kernels' 4 heads a CU. The rest (Q4 products, norms,
attention, GELU's GLU) are the Gemma4 and Qwen3 superkernels'.
"""

import triton
import triton.language as tl

from gemma4_superkernel.kernels import copy_n, rms_norm
from qwen3_superkernel.kernels import apply_rope


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
    """Gemma3's q/k/v from the QKV projection's [q | k | v] at qkv: each q
    head q-normed and rotated, each k head k-normed and rotated, v as it is
    (air-FLM's rope.cc). q comes out a kv head's q heads then zero heads, 4
    heads a kv head. w: [cos DH/2 | sin DH/2 | q_norm DH | k_norm DH]. The
    heads are normed in place."""
    G: tl.constexpr = NQ // NKV  # q heads a kv head
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
