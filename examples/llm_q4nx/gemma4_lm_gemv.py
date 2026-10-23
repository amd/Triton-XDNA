# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Gemma4-E2B's LM head as a Triton GEMV on the buffers of mlir-air's
`lm_gemv` device.

y[V] = W[V, K] x[K] with Q4NX weights (w = q * scale + min). W is in
`lm_gemv.pack`'s stream order, x is bf16 [K], and y is laid out the way
`lm_gemv.unpack_y` reads it. Each of NC x NR cores owns TILES row tiles of GN
rows and takes a row tile's K in KC packets of GK inputs. A packet is

    q      [GN/8][GK/8][8 k][8 n] nibbles, two per byte, low first
    scale  [GK/32][GN] bf16
    min    [GK/32][GN] bf16

and is read as int16 words, four nibbles each. Per 8-row block a core sums
8 k by 8 n products over the row tile's k tiles in f32, with x[k] repeated
over the 8 n of its k tile, and folds the 8 k when the row tile is done.
"""

import triton
import triton.language as tl


@triton.jit
def _packets(W, j, kc0, NKC: tl.constexpr, NC, NR, COL_W, KC, GN, GK):
    """q, scale and min words [NC, NKC, NR, ...] of packets kc0 .. kc0 + NKC
    of row tile j on every core, in the stream's order."""
    NG: tl.constexpr = GK // 32
    # int16 words: a packet's q, then its scale and min
    QW: tl.constexpr = GN * GK // 4
    PW: tl.constexpr = QW + 2 * NG * GN
    oc = tl.arange(0, NC)
    orr = tl.arange(0, NR)
    okc = kc0 + tl.arange(0, NKC)
    # a column's stream: per row tile, per K step, its NR cores' packets
    pkt = W + j * KC * NR * PW + oc[:, None, None, None] * COL_W
    pkt = pkt + okc[None, :, None, None] * (NR * PW) + orr[None, None, :, None] * PW
    q = tl.load(pkt + tl.arange(0, QW)[None, None, None, :])
    og = tl.arange(0, NG * GN)
    s = tl.load(pkt + QW + og[None, None, None, :])
    mn = tl.load(pkt + QW + NG * GN + og[None, None, None, :])
    return q, s, mn


@triton.jit
def _dequant(q, s, mn, NKC: tl.constexpr, NC, NR, GN, GK):
    """w bf16 [NC, NR, GN / 8, NKC * GK / 8, 8, 8]: per core and 8-row block,
    the k tiles of 8 k by 8 n."""
    NG: tl.constexpr = GK // 32
    # q words [nb, kb, kk, h] of a packet; word h of an 8-n group holds
    # n = 4 h .. 4 h + 3 at shifts 0, 4, 8, 12
    sh = (tl.arange(0, 4) * 4).to(tl.int16)
    # In the packets' own order [c, kc, r, nb, g, kb4, kk, h, i], with
    # k = (kc, g, kb4, kk) and n = (r, nb, h, i): nibbles, scale and min meet
    # in one elementwise expression.
    q = q.reshape(NC, NKC, NR, GN // 8, NG, 4, 8, 2, 1)
    nib = (q >> sh.reshape(1, 1, 1, 1, 1, 1, 1, 1, 4)) & 0xF
    s = s.to(tl.bfloat16, bitcast=True).to(tl.float32)
    mn = mn.to(tl.bfloat16, bitcast=True).to(tl.float32)
    # [c, kc, r, g, n] -> [c, kc, r, nb, g, 1, 1, h, i]
    s = s.reshape(NC, NKC, NR, NG, GN // 8, 2, 4).permute(0, 1, 2, 4, 3, 5, 6)
    mn = mn.reshape(NC, NKC, NR, NG, GN // 8, 2, 4).permute(0, 1, 2, 4, 3, 5, 6)
    s = s.reshape(NC, NKC, NR, GN // 8, NG, 1, 1, 2, 4)
    mn = mn.reshape(NC, NKC, NR, GN // 8, NG, 1, 1, 2, 4)
    # 0x4300 | v is 128 + v, so w = s (128 + v) + (m - 128 s), rounded to
    # bf16 once
    v = (nib | 0x4300).to(tl.bfloat16, bitcast=True).to(tl.float32)
    w = (v * s + (mn - 128.0 * s)).to(tl.bfloat16)
    w = w.permute(0, 2, 3, 1, 4, 5, 6, 7, 8)
    return w.reshape(NC, NR, GN // 8, NKC * GK // 8, 8, 8)


@triton.jit
def _xpart(X, k0, NK: tl.constexpr):
    """x[k0 .. k0 + NK] as [NK / 8, 8]."""
    return tl.load(X + k0 + tl.arange(0, NK)).reshape(NK // 8, 8)


@triton.jit
def gemv_q4(
    W,
    X,
    Y,
    NC: tl.constexpr,
    NR: tl.constexpr,
    TILES: tl.constexpr,
    FLUSH: tl.constexpr,
    KC: tl.constexpr,
    GN: tl.constexpr,
    GK: tl.constexpr,
    RUN: tl.constexpr,
):
    NG: tl.constexpr = GK // 32
    PW: tl.constexpr = GN * GK // 4 + 2 * NG * GN
    COL_W: tl.constexpr = TILES * KC * NR * PW
    oc = tl.arange(0, NC)
    orr = tl.arange(0, NR)
    on = tl.arange(0, GN)
    KT: tl.constexpr = RUN * GK // 8
    # x reaches every core once, and each core repeats x[k] over the 8 n of
    # its k tile, a run of packets at a time.
    xes = ()
    for r in tl.static_range(KC // RUN):
        x = _xpart(X, r * RUN * GK, RUN * GK).reshape(1, 1, KT, 8, 1)
        xes = xes + (tl.broadcast_to(x, (NC, NR, KT, 8, 8)),)
    # row tiles in groups of FLUSH, the order the logits are laid out in
    for f in range(TILES // FLUSH):
        for jj in range(FLUSH):
            j = f * FLUSH + jj
            # The row tile's packets in runs of RUN (aranges must be powers of
            # two), all read before any compute, so they leave memory as one
            # stream; runs of one size let the cores take them on one channel.
            ps = ()
            for r in tl.static_range(KC // RUN):
                ps = ps + (_packets(W, j, r * RUN, RUN, NC, NR, COL_W, KC, GN, GK),)
            acc = tl.zeros((NC, NR, GN // 8, 8, 8), tl.float32)
            for r in tl.static_range(KC // RUN):
                q, s, mn = ps[r]
                w = _dequant(q, s, mn, RUN, NC, NR, GN, GK).to(tl.float32)
                xe = xes[r].reshape(NC, NR, 1, KT, 8, 8).to(tl.float32)
                acc += tl.sum(w * xe, 3)
            # y[c, f, r, jj * GN + n]
            out = (
                Y
                + f * (NR * FLUSH * GN)
                + jj * GN
                + oc[:, None, None] * (TILES * NR * GN)
                + orr[None, :, None] * (FLUSH * GN)
                + on[None, None, :]
            )
            y = tl.sum(acc, 3).to(tl.bfloat16)
            tl.store(out, y.reshape(NC, NR, GN))


def args(lm_gemv, w, x, y, run=2):
    """gemv_q4's arguments for the shape of mlir-air's `lm_gemv` module: w is
    `lm_gemv.pack`'s uint8 stream, x bf16 [K], y bf16 as `lm_gemv.unpack_y`
    reads it. `run`: packets per x part and per load."""
    import torch

    L = lm_gemv
    shape = (L.NC, L.NR, L.TILES, L.FLUSH, L.KC, L.GN, L.GK, run)
    return (w.view(torch.int16), x, y, *shape)
