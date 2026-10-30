# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Gemma4 prefill GEMM, `mm_g`, on weights packed by `pack_q4` (int4) or
`pack_bf16`.

A packet holds TN columns of the weights: an int4 packet TK of K, a bf16
packet TK / 4. `mm_g` has a K loop for each format, one of them zero-trip by
`wq`, two K steps a trip. Both stream their packets in pairs, which the cores
take one packet at a time.
"""

import triton
import triton.language as tl

C2 = 2 * (2 / 3.141592653589793) ** 0.5
KG = 0.044715

#: rows per chunk, the core tile TM x TN and K step TK, and the mmul block
M, TM, TN, TK, MM = 128, 32, 64, 128, 8
#: output columns of a round: six cores of TN
NR = 6 * TN
#: bytes of a weight packet
QB = TN * TK // 2 + 2 * 2 * (TK // 32) * TN


def k_pad(k):
    """A GEMM's depth on the device: whole loop trips of 4 K steps."""
    return -(-k // (4 * TK)) * 4 * TK


def n_pad(n):
    """A GEMM's width on the device: whole rounds."""
    return -(-n // NR) * NR


def pack_q4(q, scale, mn):
    """int4 [out, in] as (q, scale, min per 32 in) -> packets [N / TN, K / TK,
    QB] uint8, N padded to whole rounds with zero weights. A packet is the
    nibbles of [TN / MM][TK / MM][MM k][MM n], two per byte along n, low
    first, then scale and min bf16 [TK / 32][TN]."""
    import numpy as np
    from ml_dtypes import bfloat16

    n, k = q.shape
    npd = n_pad(n)
    pad = lambda a: np.pad(np.asarray(a).T, ((0, 0), (0, npd - n)))  # noqa: E731
    q, scale, mn = pad(q), pad(scale), pad(mn)
    qt = q.reshape(k // TK, TK // MM, MM, npd // TN, TN // MM, MM)
    qt = qt.transpose(3, 0, 4, 1, 2, 5).reshape(npd // TN, k // TK, -1)
    qb = (qt[..., 0::2] | (qt[..., 1::2] << 4)).astype(np.uint8)

    def groups(a):
        a = np.asarray(a, bfloat16).reshape(k // TK, TK // 32, npd // TN, TN)
        return a.transpose(2, 0, 1, 3).reshape(npd // TN, k // TK, -1).view(np.uint8)

    return np.ascontiguousarray(np.concatenate([qb, groups(scale), groups(mn)], 2))


def pack_bf16(w):
    """bf16 [out, in] -> packets [N / TN, K / TK * 4, QB] uint8: per K step
    four packets of TN x 32 weights as [TN / MM][32 / MM][MM k][MM n],
    zero-padded to QB bytes."""
    import numpy as np
    from ml_dtypes import bfloat16

    n, k = w.shape
    npd = n_pad(n)
    b = np.zeros((k, npd), bfloat16)
    b[:, :n] = np.asarray(w, np.float32).T
    t = b.reshape(k // TK, 4, 32 // MM, MM, npd // TN, TN // MM, MM)
    t = t.transpose(4, 0, 1, 5, 2, 3, 6).reshape(npd // TN, k // TK * 4, -1)
    out = np.zeros(t.shape[:2] + (QB,), np.uint8)
    out[..., : t.shape[-1] * 2] = t.view(np.uint8)
    return out


@triton.jit(do_not_specialize=["N", "K", "act", "wq"])
def mm_g(
    A,
    W,
    C,
    N,
    K,
    act,
    wq,
    stride_am: tl.constexpr,
    stride_cm: tl.constexpr,
    TN: tl.constexpr,
    PKT: tl.constexpr,
    TS_Q: tl.constexpr,
    TS_B: tl.constexpr,
    TILE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_ENGINES: tl.constexpr,
    NT: tl.constexpr,
    C2: tl.constexpr,
    KG: tl.constexpr,
):
    start = tl.program_id(0)
    num_tiles = N // TILE_N
    # int4: pairs come in pairs (its K is a multiple of 2 * BLOCK_SIZE_K), so
    # L2 can alternate between two buffers; bf16 (K may be one pair) brings
    # each pair's packets into four L2 slots instead
    kq = K // (2 * BLOCK_SIZE_K) * 2 * wq
    kb = K // BLOCK_SIZE_K * (1 - wq)
    om = tl.arange(0, BLOCK_SIZE_M)
    ot = tl.arange(0, BLOCK_SIZE_M // 32)
    orr = tl.arange(0, 32)
    cols = tl.arange(0, BLOCK_SIZE_N)
    live = cols < TILE_N
    opk = tl.arange(0, 2)
    oj = tl.arange(0, 4)
    o128 = tl.arange(0, 128)
    orow = tl.arange(0, 128)
    o32 = tl.arange(0, 32)
    tiles = tl.arange(0, BLOCK_SIZE_N // TN)
    live_t = tiles < TILE_N // TN
    for tile in tl.range(start, num_tiles, NUM_ENGINES):
        on = tile * TILE_N + cols
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        # int4 weights
        for kp in range(kq):
            # A is tiled [M / 32, K / 128, 32, 128]: a row block's K pair is
            # one contiguous [2, 32, 128] block
            a = tl.load(
                A
                + ot[:, None, None, None] * (32 * stride_am)
                + (2 * kp + opk)[None, :, None, None] * (32 * 128)
                + orr[None, None, :, None] * 128
                + o128[None, None, None, :]
            )
            a = a.permute(0, 2, 1, 3).reshape(BLOCK_SIZE_M, BLOCK_SIZE_K)
            # each core column's two packets, as 2560 i16 words each: rows
            # 0-63 q, 64-71 scale, 72-79 min, in one load
            t_off = (tile * (TILE_N // TN) + tiles) * TS_Q + kp * PKT
            p = tl.load(
                W
                + t_off[:, None, None, None]
                + opk[None, :, None, None] * (PKT // 2)
                + orow[None, None, :, None] * 32
                + o32[None, None, None, :],
                mask=live_t[:, None, None, None] & (orow < 80)[None, None, :, None],
                other=0,
            )
            q, r = tl.split(p.reshape(NT, 2, 2, 64, 32).permute(0, 1, 3, 4, 2))
            r, _ = tl.split(r.reshape(NT, 2, 2, 32, 32).permute(0, 1, 3, 4, 2))
            sm, _ = tl.split(r.reshape(NT, 2, 2, 16, 32).permute(0, 1, 3, 4, 2))
            s, mn = tl.split(sm.reshape(NT, 2, 2, 8, 32).permute(0, 1, 3, 4, 2))
            # q words [nb, kb, kk, w] per packet, word w holding columns
            # 4 w .. 4 w + 3 of B tile (nb, kb), K row kk: nb = (nh, nbl),
            # kb = (g, kbm, kbl)
            q = q.reshape(NT, 2, 2, 4, 4, 2, 2, 8, 2, 1)
            sh = (
                (tl.arange(0, 4) * 4).to(tl.int16).reshape(1, 1, 1, 1, 1, 1, 1, 1, 1, 4)
            )
            nib = (q >> sh) & 0xF
            # scale and min [g, n] per packet, n = (nh, nbl, w, i); bf16 bits
            s = s.reshape(NT, 2, 4, 2, 4, 2, 4).permute(0, 1, 3, 4, 2, 5, 6)
            mn = mn.reshape(NT, 2, 4, 2, 4, 2, 4).permute(0, 1, 3, 4, 2, 5, 6)
            s9 = (
                s[:, :, :, :, :, None, None, None, :, :]
                .to(tl.bfloat16, bitcast=True)
                .to(tl.float32)
            )
            m9 = (
                mn[:, :, :, :, :, None, None, None, :, :]
                .to(tl.bfloat16, bitcast=True)
                .to(tl.float32)
            )
            # 0x4300 | v is 128 + v, so w = s (128 + v) + (m - 128 s): one
            # bf16 multiply-add per weight on a per-group base
            x = (nib | 0x4300).to(tl.bfloat16, bitcast=True)
            base = m9 - 128.0 * s9
            b9 = (x.to(tl.float32) * s9 + base).to(
                tl.bfloat16
            )  # tt pk nh nbl g kbm kbl kk w i
            b = b9.permute(1, 4, 5, 6, 7, 0, 2, 3, 8, 9).reshape(
                BLOCK_SIZE_K, BLOCK_SIZE_N
            )
            acc = tl.dot(a, b, acc)
        # bf16 weights
        for kp in range(kb):
            a = tl.load(
                A
                + ot[:, None, None, None] * (32 * stride_am)
                + (2 * kp + opk)[None, :, None, None] * (32 * 128)
                + orr[None, None, :, None] * 128
                + o128[None, None, None, :]
            )
            a = a.permute(0, 2, 1, 3).reshape(BLOCK_SIZE_M, BLOCK_SIZE_K)
            # the pair's eight packets per core column, packet 2 jj + pk, all
            # 80 rows; rows 0-63 hold 64 x 32 bf16 in B-tile order
            # [nb, kbl, kk, nn], kk = (kh, kl)
            t_off = (tile * (TILE_N // TN) + tiles) * TS_B + kp * (8 * (PKT // 2))
            p = tl.load(
                W
                + t_off[:, None, None, None, None]
                + (2 * oj[:, None] + opk[None, :])[None, :, :, None, None] * (PKT // 2)
                + orow[None, None, None, :, None] * 32
                + o32[None, None, None, None, :],
                mask=live_t[:, None, None, None, None]
                & (orow < 80)[None, None, None, :, None],
                other=0,
            )
            q, _ = tl.split(p.reshape(NT, 4, 2, 2, 64, 32).permute(0, 1, 2, 4, 5, 3))
            # jj = 2 h + jj0: K (h, jj0, pk, kbl, kh, kl) x N (tt, nb, nn)
            q = q.reshape(NT, 2, 2, 2, 8, 4, 2, 4, 8)  # tt h jj0 pk nb kbl kh kl nn
            b = (
                q.permute(1, 2, 3, 5, 6, 7, 0, 4, 8)
                .reshape(BLOCK_SIZE_K, BLOCK_SIZE_N)
                .to(tl.bfloat16, bitcast=True)
            )
            acc = tl.dot(a, b, acc)
        y = (C2 * 0.5) * (acc + KG * acc * acc * acc)
        y = tl.where(y > 10.0, 10.0, tl.where(y < -10.0, -10.0, y))
        g = 0.5 * acc * (1.0 + tl.extra.cuda.libdevice.tanh(y))
        out = tl.where(act != 0.0, g, acc)
        tl.store(
            C + om[:, None] * stride_cm + on[None, :],
            out.to(tl.bfloat16),
            mask=live[None, :],
        )


def args(a, w, c, n, k, act, wq, k_pad=None):
    """w: uint8 packets, pack_q4's [N / 64, K / 128, 5120] if wq else
    pack_bf16's [N / 64, K / 32, 5120]; a: A tiled [M / 32, K / 128, 32, 128].
    Both may be padded with zeros to depth `k_pad`; only the first k of K is
    read."""
    import torch

    k_pad = k_pad or k
    return (
        a,
        w.view(torch.int16),
        c,
        n,
        k,
        float(act),
        int(wq),
        k_pad,
        n,
        64,
        5120,
        k_pad // 128 * 2560,
        k_pad // 128 * 4 * 2560,
        384,
        128,
        512,
        256,
        1,
        8,
        C2,
        KG,
    )
