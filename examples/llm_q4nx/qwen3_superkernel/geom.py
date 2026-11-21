# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""The decode's geometry, derived as air-FLM's `fused_decode` derives it for
its `qwen3-4b` entry: rounds of 512 rows and pairs of 256-column blocks a
phase, the LM head's chunks, the attention CUs and the KV cache's groups.
"""

import numpy as np
import qwen3_4b_q4nx_weights as qw
import triton.language as tl
from ml_dtypes import bfloat16

D = qw.D  # the hidden size, K of air-FLM
DH = qw.DH
NQ, NKV = qw.N_Q_HEADS, qw.N_KV_HEADS
DQ, DK = qw.DQ, qw.DK
M = DQ + 2 * DK  # the QKV projection's rows
INTER = qw.INTER
EPS = qw.RMS_EPS
# the scores' scale, 1/sqrt(DH), as the bf16 the kernels multiply by
SCALE = float(np.float32(DH**-0.5).astype(bfloat16))
KB = D // 512  # 512s of x
I2P = (M // 512, KB, 2 * INTER // 512, KB)  # rounds of 512 rows a phase
J2P = (KB, DQ // 512, KB, INTER // 512)  # pairs of 256-column blocks a phase
LMR = 30  # rounds of an LM-head wave (VOCAB_CHUNK_I2)
LM_ROWS = LMR * 512  # vocab rows an LM-head wave computes
NL = (qw.VOCAB + LM_ROWS - 1) // LM_ROWS  # LM-head waves
NCU = 4  # attention CUs, a qk and a kv core each
KVPC = NKV // NCU  # kv heads a CU, 4 q heads each
NCH = DH // 8  # 8-column chunks of a head
NGRP = 2  # KV groups: the CUs of a column share one, and its memtile
GW = NCU // NGRP * KVPC * DH  # a group's KV row
ROPE_W = 3 * DH  # [cos | sin | q_norm | k_norm] of a layer, at a position
GLU_HALF = 256  # up/gate rows a glu slice takes of each
GLU_SLICES = I2P[2] // (2 * GLU_HALF // 512)

# Per [phase][arm] (arm 1 a layer, arm 0 an LM-head wave): rounds with
# weights, blocks of a round, as the Gemma4 superkernel's tables.
I2A = tl.constexpr(
    tuple(
        (LMR if p == 0 else 0) if a == 0 else I2P[p] for p in range(4) for a in range(2)
    )
)
NBA = tl.constexpr(
    tuple(
        (2 * KB if p == 0 else 0) if a == 0 else 2 * J2P[p]
        for p in range(4)
        for a in range(2)
    )
)
NFA = [
    tl.constexpr(tuple(I2A.value[p * 2 + a] * NBA.value[p * 2 + a] for a in range(2)))
    for p in range(4)
]
NBMAX = max(NBA.value)  # column blocks of the widest phase
NR = sum(I2P)  # rounds of a decode wave
# x landings of a decode wave, 512 at a time
NX = I2P[0] * KB + I2P[1] * DQ // 512 + I2P[2] * KB + I2P[3] * INTER // 512
NFILL = sum(I2P[p] * 2 * J2P[p] for p in range(4))  # fills of a weight stream
assert KVPC == 2  # the attention cores take a CU's two kv heads by name
