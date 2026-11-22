# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""The decode's geometry for the Gemma4 variant the weights module is set up
for (GEMMA4_VARIANT), derived as air-FLM's `fused_decode_ple` derives it:
rounds of 512 rows and pairs of 256-column blocks a phase, what each kind of
layer (arm) streams, and the sizes of the buffers between the cores.
"""

import gemma4_e2b_q4nx_weights as gw
import triton.language as tl

D = gw.D  # the hidden size, K of air-FLM
DH = 512  # every head in a 512 slot: the full layers' head dim
NQ = gw.N_Q_HEADS
DQ = NQ * DH  # q of a full layer
NCU = 2  # attention CUs, 4 q heads each
# The one kv head of E2B is copied to both CUs (KV_REPLICATED); E4B's two
# are one a CU.
KV_REPL = gw.N_KV_HEADS == 1
KVW = DH if KV_REPL else gw.N_KV_HEADS * DH  # a KV row: REGION_W
# E2B splits each CU's context between two sets of cores and merges them
# (KV_PARTS); E4B does not.
PARTS = 2 if KV_REPL else 1
M = DQ + 2 * NCU * DH  # q, k and v rows of a full layer, k/v as the CUs take them
INTER = gw.INTER * (2 if gw.DOUBLE_WIDE_MLP else 1)  # the widest layer's
INTER_NARROW = gw.INTER
NF = INTER // INTER_NARROW  # how much narrower the narrow layers' MLP is
KB = D // 512  # 512s of x
I2P = (M // 512, KB, 2 * INTER // 512, KB)  # rounds of 512 rows a phase
J2P = (KB, DQ // 512, KB, INTER // 512)  # pairs of 256-column blocks a phase
LMR = gw.VARIANTS[gw.VARIANT]["VOCAB_CHUNK_I2"]  # rounds of an LM-head wave
LM_ROWS = LMR * 512  # vocab rows an LM-head wave computes
NL = (gw.VOCAB + LM_ROWS - 1) // LM_ROWS  # LM-head waves


def _arm_rounds(arm):
    """air-FLM's CLASS_GEOM: the rounds of each phase that have weights, for
    a layer of arm 1 (KV-shared, wide, full), 3 (own KV, narrow, full), 5
    (shared, sliding) or 7 (own, sliding). A KV-shared layer has no k/v
    rows; a sliding one half of each head; a narrow one 1/NF of the MLP."""
    own, sliding = arm & 2, arm & 4
    qkv = (M if own else DQ) // (2 if sliding else 1) // 512
    gu = I2P[2] // (NF if own else 1)
    return (qkv, KB, gu, KB)


def _arm_blocks(arm):
    """The column blocks of a round of each phase that have weights, and the
    mask of those without (a sliding layer's o-proj skips the upper halves
    of the heads)."""
    own = arm & 2
    down = 2 * J2P[3] // (NF if own else 1)
    return (2 * KB, 2 * J2P[1], 2 * KB, down)


# Per [phase][arm], what an arm's layer really has: rounds with weights,
# blocks of a round with weights, and the mask of the blocks without. Arm 0:
# an LM-head wave (a vocab chunk).
I2A = tl.constexpr(
    tuple(
        (LMR if (p == 0 and a == 0) else 0) if a % 2 == 0 else _arm_rounds(a)[p]
        for p in range(4)
        for a in range(8)
    )
)
NBA = tl.constexpr(
    tuple(
        (2 * KB if (p == 0 and a == 0) else 0) if a % 2 == 0 else _arm_blocks(a)[p]
        for p in range(4)
        for a in range(8)
    )
)
MASKA = tl.constexpr(
    tuple((4 if (p == 1 and a in (5, 7)) else 0) for p in range(4) for a in range(8))
)
# fills of each weight stream a phase, by arm: rounds x blocks a round
NFA = [
    tl.constexpr(
        tuple(
            I2A.value[p * 8 + a]
            * NBA.value[p * 8 + a]
            // (2 if MASKA.value[p * 8 + a] else 1)
            for a in range(8)
        )
    )
    for p in range(4)
]
NBMAX = max(NBA.value)  # column blocks of the widest phase
NR = sum(I2P)  # rounds of a decode wave
# x landings of a decode wave, 512 at a time: x for QKV and up/gate, the
# attention output for o-proj and glu's for down, each refed a round
NX = I2P[0] * KB + I2P[1] * DQ // 512 + I2P[2] * KB + I2P[3] * INTER // 512
NFILL = sum(I2P[p] * 2 * J2P[p] for p in range(4))  # fills of a weight stream
GLU_SLICES = I2P[2] // 2  # glu's 512-row slices of up/gate
PLE_BLK = D // 256  # PLE projection column blocks
PLE_FILLS = 4 * PLE_BLK + D // 64  # blocks of each PLE weight stream a wave
PXW = gw.PLI_D + gw.PLI_D + D + 32  # [ple emb | ple norm | post-PLE norm | scale]
PK = gw.PLI_D + D + 32  # what pleproj hands plegate
RMS_LAYER = 4 * D  # a layer's 4 norm weights
