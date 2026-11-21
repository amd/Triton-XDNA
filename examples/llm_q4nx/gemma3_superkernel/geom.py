# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""The decode's geometry, derived as air-FLM's `fused_decode` derives it for
its `gemma3-4b` entry: rounds of 512 rows and pairs of 256-column blocks a
phase, the LM head's chunks, the attention CUs and the KV cache's groups.
"""

import gemma3_4b_q4nx_weights as gw
import numpy as np
import triton.language as tl
from ml_dtypes import bfloat16

D = gw.D  # the hidden size, K of air-FLM
DH = gw.DH
NQ, NKV = gw.N_Q_HEADS, gw.N_KV_HEADS
DQ, DK = gw.DQ, gw.DK
M = DQ + 2 * DK  # the QKV projection's rows
INTER = gw.INTER
EPS = gw.RMS_EPS
SWA = gw.SLIDING_WINDOW
# the scores' scale (query_pre_attn_scalar), as the bf16 the kernels multiply by
SCALE = float(np.float32(gw.ATTN_SCALE).astype(bfloat16))
KB = D // 512  # 512s of x
I2P = (M // 512, KB, 2 * INTER // 512, KB)  # rounds of 512 rows a phase
J2P = (KB, DQ // 512, KB, INTER // 512)  # pairs of 256-column blocks a phase
LMR = 5  # rounds of an LM-head wave (VOCAB_CHUNK_I2)
LM_ROWS = LMR * 512  # vocab rows an LM-head wave computes
NL = (gw.VOCAB + LM_ROWS - 1) // LM_ROWS  # LM-head waves
NCU = 4  # attention CUs, a qk and a kv core each, a kv head each
QPC = NQ // NCU  # a CU's q heads
QPAD = 4  # its q heads padded with zero heads, as the attention kernels take 4
NCH = DH // 8  # 8-column chunks of a head
NGRP = 2  # KV groups: the CUs of a column share one, and its memtile
GW = NCU // NGRP * NKV // NCU * DH  # a group's KV row
ROPE_W = 3 * DH  # [cos | sin | q_norm | k_norm] of a layer, at a position
GLU_HALF = 512  # up/gate rows a glu slice takes of each
GLU_SLICES = I2P[2] // (2 * GLU_HALF // 512)
RMS_LAYER = 4 * D  # a layer's 4 norm weights

# Per [phase][arm] (arm 1 a full layer, 2 a sliding one, 0 an LM-head wave):
# rounds with weights, blocks of a round.
NARM = 3
I2A = tl.constexpr(
    tuple(
        (LMR if p == 0 else 0) if a == 0 else I2P[p]
        for p in range(4)
        for a in range(NARM)
    )
)
NBA = tl.constexpr(
    tuple(
        (2 * KB if p == 0 else 0) if a == 0 else 2 * J2P[p]
        for p in range(4)
        for a in range(NARM)
    )
)
NFA = [
    tl.constexpr(
        tuple(I2A.value[p * NARM + a] * NBA.value[p * NARM + a] for a in range(NARM))
    )
    for p in range(4)
]
NBMAX = max(NBA.value)  # column blocks of the widest phase
NR = sum(I2P)  # rounds of a decode wave
# x landings of a decode wave, 512 at a time
NX = I2P[0] * KB + I2P[1] * DQ // 512 + I2P[2] * KB + I2P[3] * INTER // 512
NFILL = sum(I2P[p] * 2 * J2P[p] for p in range(4))  # fills of a weight stream
assert NKV == NCU  # a kv head a CU
