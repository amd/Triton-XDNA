# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Gemma4 prefill attention for the fused prefill device.

Flash attention over one chunk's queries, round by round (a round is one
head's R = NQT * LKP q rows), against nk K blocks of the KV buffer. The
buffers are the op's own: Q and O [rounds * R, DH], and KV as records of a K
tile [LKP, DH] followed by its V as DH / DVT tiles [LKP, DVT]. nk, q0 and k0
(the chunk's first q block and first K block, in LKP rows) are the op's
run-time arguments.

Numerics: S in bf16, the row maximum and the masks on bf16 values, the
probabilities in bf16, and the output accumulator kept in bf16, rescaled and
added to P V in f32 each block.
"""

from pathlib import Path

import triton
import triton.language as tl

LOG2E = 1.4426950408889634
NQT = tl.constexpr(4)  # cores of an attention column, LKP q rows each
SCHEDULE = Path(__file__).resolve().parent / "transform_gemma4_attention_aie2p.mlir.in"


@triton.jit
def _maximum(a, b):
    return tl.where(a > b, a, b)


@triton.jit(do_not_specialize=["nk", "q0", "k0"])
def attn(
    Q,
    KV,
    O,
    nk,
    q0,
    k0,
    LKP: tl.constexpr,
    DH: tl.constexpr,
    DVT: tl.constexpr,
    QR: tl.constexpr,  # rounds per head
    WB: tl.constexpr,  # window in K blocks; 0 is causal
    C: tl.constexpr,  # log2e / sqrt(DH)
    ROUNDS: tl.constexpr,
    KVH: tl.constexpr,  # KV heads, each ROUNDS // KVH consecutive rounds'
    SPAN: tl.constexpr,  # elements of one KV head's records
):
    R: tl.constexpr = NQT * LKP
    REC: tl.constexpr = 2 * LKP * DH
    DVC: tl.constexpr = DH // DVT
    oi = tl.arange(0, LKP)
    orr = tl.arange(0, R)
    od = tl.arange(0, DH)
    # V read in row order, [LKP, DVC, DVT], so it needs no transpose
    oz = tl.arange(0, DVC)
    ov = tl.arange(0, DVT)
    voff = oi[:, None, None] * DVT + oz[None, :, None] * (LKP * DVT) + ov[None, None, :]
    # q row minus K column within a block; a block's masks compare it with
    # the block's offset
    dij = orr[:, None] - oi[None, :]
    # Rounds loop here rather than over programs, so the host sets up the
    # transfers once per dispatch.
    for r in range(ROUNDS):
        rows = r * R + orr
        q = tl.load(Q + rows[:, None] * DH + od[None, :])
        kv = KV + (r // (ROUNDS // KVH)) * SPAN
        # the round's first q row
        qr0 = (q0 + (r % QR) * NQT) * LKP
        # The row maxima are of bf16 scores, so m is bf16: the cores compare
        # in bf16.
        m = tl.full((R,), -1.0e30, tl.bfloat16)
        l = tl.zeros((R,), tl.float32)
        acc = tl.zeros((R, DH), tl.bfloat16)
        for blk in range(nk):
            base = kv + blk * REC
            k = tl.load(base + oi[:, None] * DH + od[None, :])
            v = tl.load(base + LKP * DH + voff).reshape(LKP, DH)
            s = tl.dot(q, tl.trans(k)).to(tl.bfloat16)
            # q row qr0 + i sees K column (k0 + blk) LKP + j when
            # 0 <= d < window
            d = dij + (qr0 - (k0 + blk) * LKP)
            if WB > 0:
                # both bounds in one unsigned compare (d < 0 wraps high), so
                # there is one mask
                keep = d.to(tl.uint32, bitcast=True) < WB * LKP
            else:
                keep = d >= 0
            s = tl.where(keep, s, float("-inf"))
            # a bf16 maximum, without widening: the maximum of bf16 values is
            # exact
            rm = tl.reduce(s, 1, _maximum)
            mn = tl.where(m > rm, m, rm)
            a = tl.exp2(C * (m.to(tl.float32) - mn.to(tl.float32)))
            p = tl.exp2(C * (s.to(tl.float32) - mn[:, None].to(tl.float32))).to(
                tl.bfloat16
            )
            l = l * a + tl.sum(p.to(tl.float32), axis=1)
            acc = (acc.to(tl.float32) * a[:, None] + tl.dot(p, v)).to(tl.bfloat16)
            m = mn
        # one reciprocal per row, then a multiply
        out = acc.to(tl.float32) * (1.0 / l)[:, None]
        tl.store(O + rows[:, None] * DH + od[None, :], out.to(tl.bfloat16))


def schedule(lkp, dh, dvt, out_dir):
    """The transform script for one group's shape, written to `out_dir`."""
    text = SCHEDULE.read_text()
    r = NQT.value * lkp
    keys = {"LKP": lkp, "R": r, "R8": r // 8, "MB": lkp // 8, "KB": dh // 8}
    keys |= {"DVC": dh // dvt, "VB": dvt // 8}
    # With four or more of P's blocks per row block, a block of acc at a time
    # keeps them all live across acc's loop; two by two blocks of acc instead read
    # two of P's and two of V's blocks per step of P V's K loop.
    two = lkp // 8 >= 4
    keys |= {"AT": 2 if two else 1, "AU": 2 if two else 4}
    for k, v in keys.items():
        text = text.replace(f"{{{k}}}", str(v))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"transform_attention_l{lkp}_d{dh}_v{dvt}.mlir"
    path.write_text(text)
    return path


def args(g, rounds, nk, q0, k0, Q, KV, O, rows=128, window=None):
    """attn's arguments for an attention group `g` over `rounds` rounds of a
    chunk of `rows` q rows; `window` overrides the group's window."""
    qr = rows // (NQT.value * g.lkp)
    window = g.window if window is None else window
    wb = window // g.lkp if window else 0
    span = g.max_blocks * 2 * g.lkp * g.dh
    c = LOG2E / g.dh**0.5
    return (
        Q,
        KV,
        O,
        nk,
        q0,
        k0,
        g.lkp,
        g.dh,
        g.dvt,
        qr,
        wb,
        c,
        rounds,
        g.kv_heads,
        span,
    )
