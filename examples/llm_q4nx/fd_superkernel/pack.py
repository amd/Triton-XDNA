# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""The model's weights in the layouts the decode superkernel streams them in:
Q4 blocks of 32 rows by 256 columns ([scale | min | nibbles], 5120 bytes),
in each phase's round order. The LM head is the bundle's own Q4 head, or a
tied one (the embedding) requantized to Q4, as air-FLM's packer takes them.
"""

import numpy as np

from q4_blocks import _blocks, _cat, _rows, bf, pack_blocks

from .geom import D, DK, DQ, GLU_HALF, I2P, INTER, J2P, LM_ROWS, LMR
from .spec import S

GROUP = 32  # the Q4 groups' columns


def phase_mats(m, L):
    """The layer's 4 phase matrices as the cores take them: (q uint8 [M, K],
    scale, min [M, K / 32]); up and gate interleaved GLU_HALF rows at a time,
    as glu takes them."""
    P = m._PROJ

    def q4(nm):
        t, rows, cols = P[nm]
        return m.q4nx(f"model.layers.{L}.{t}.weight", rows, cols)

    qkv = _cat(q4("q"), q4("k"), q4("v"))
    up, gate = q4("up"), q4("gate")
    n = INTER // GLU_HALF
    il = tuple(
        np.concatenate(
            [
                (a if h == 0 else b)[s * GLU_HALF : (s + 1) * GLU_HALF]
                for s in range(n)
                for h in (0, 1)
            ]
        )
        for a, b in zip(up, gate)
    )
    return [qkv, q4("o"), il, q4("down")]


def pack_wave(mats, arm):
    """A wave's weight slab: [stream][phase][fill]."""
    rounds = [LMR] if arm == 0 else list(I2P)
    blocks = [_blocks(mats[p], rounds[p] * 16, J2P[p] * 2) for p in range(len(mats))]
    chunks = []
    for x in range(4):
        for k in range(2):
            for p in range(len(mats)):
                q, sc, mn = blocks[p]
                idx = [
                    (i * 16 + x * 4 + 2 * k + c, j)
                    for i in range(rounds[p])
                    for j in range(J2P[p] * 2)
                    for c in range(2)
                ]
                rbi = np.array([a for a, _ in idx])
                cbi = np.array([b for _, b in idx])
                chunks.append(
                    pack_blocks(q[rbi, cbi], bf(sc[rbi, cbi]), bf(mn[rbi, cbi]))
                )
    return np.concatenate(chunks).reshape(-1)


def lm_mats(lm, c):
    """LM-head wave c's rows of the head (q, scale, min), zero rows past the
    vocabulary."""
    r0 = c * LM_ROWS
    return _rows(tuple(a[r0 : r0 + LM_ROWS] for a in lm), LM_ROWS)


def lm_q4(m, vocab):
    """The LM head in Q4: the bundle's, or the embedding requantized by groups
    of 32 columns as air-FLM's requant does."""
    if S["lm_head"] == "q4":
        return m.q4nx("lm_head.weight", vocab, D)
    from air_examples.llms.shared.infra import host_weights

    lm = m.bf16("model.embed_tokens.weight")[:vocab]
    return host_weights.requant_q4k(np.asarray(lm, np.float32), GROUP)


assert DQ + 2 * DK == I2P[0] * 512 and D == J2P[0] * 512
