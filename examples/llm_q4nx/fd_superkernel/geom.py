# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""The decode's geometry for the selected model (`spec`), derived as air-FLM's
`fused_decode` derives it: rounds of 512 rows and pairs of 256-column blocks
a phase, the LM head's chunks, the attention CUs and the KV cache's groups.
"""

import numpy as np
import triton.language as tl
from ml_dtypes import bfloat16

from .spec import S, W

D = W.D  # the hidden size, K of air-FLM
DH = W.DH
NQ, NKV = W.N_Q_HEADS, W.N_KV_HEADS
DQ, DK = W.DQ, W.DK
M = DQ + 2 * DK  # the QKV projection's rows
INTER = getattr(W, "INTER_PAD", W.INTER)  # padded to 512s where it is not
EPS = getattr(W, "RMS_EPS", S.get("eps"))
NDEC = getattr(W, "NUM_LAYERS", S.get("layers"))  # decoder layers
VOCAB = getattr(W, "VOCAB", S.get("vocab"))
SWA = W.SLIDING_WINDOW if S["window"] else 0
# the scores' scale, as the bf16 the kernels multiply by: Gemma3's
# query_pre_attn_scalar, 1/sqrt(DH) otherwise
SCALE = float(np.float32(getattr(W, "ATTN_SCALE", DH**-0.5)).astype(bfloat16))
KB = D // 512  # 512s of x
I2P = (M // 512, KB, 2 * INTER // 512, KB)  # rounds of 512 rows a phase
J2P = (KB, DQ // 512, KB, INTER // 512)  # pairs of 256-column blocks a phase
LMR = S["lm_rounds"]  # rounds of an LM-head wave (VOCAB_CHUNK_I2)
LM_ROWS = LMR * 512  # vocab rows an LM-head wave computes
NL = (VOCAB + LM_ROWS - 1) // LM_ROWS  # LM-head waves
NCU = S.get("cus", 4)  # attention CUs, a qk and a kv core each
KVPC = S["kv_per_cu"]  # kv heads a CU
QG = NQ // NKV  # q heads a kv head
QSEG = S.get("q_seg", 4)
QPAD = -(-QG // QSEG) * QSEG  # those padded with zero heads to QSEG's
NQG = KVPC * QPAD // 4  # a CU's groups of 4 q heads, the attention kernels'
QPC = KVPC * QG  # a CU's (real) q heads
NCH = DH // 8  # 8-column chunks of a head
# KV groups: the CUs of a column share one, and its memtile; 4 CUs take
# columns 3 and 4, 2 column 4 (air-FLM's ATTN_PCOL)
NGRP = NCU // 2
GCOL = 3 if NGRP == 2 else 4  # the first group's column
GW = NCU // NGRP * KVPC * DH  # a group's KV row
ROPE = S.get("rope", "qk_norm")
# a layer's rope weights at a position: [cos | sin], then [q_norm | k_norm] or
# the q/k/v biases where the model has them
RD = S.get("rope_dim", DH)  # the dims a head rotates on
ROPE_W = RD + {"bias": M, "qk_norm": 2 * DH, "lut": 0}[ROPE]
assert RD == DH or ROPE == "lut"
LPW = S.get("layers_per_w", 0)  # layers a W buffer, 0 for all in one
NORMS = S["norms"]  # a layer's norms: 2, or Gemma's sandwich of 4
GLU_HALF = S["glu_half"]  # up/gate rows a glu slice takes of each
GLU_SLICES = I2P[2] // (2 * GLU_HALF // 512)

# Per [phase][arm] (arm 0 an LM-head wave, 1 a layer, 2 a sliding layer):
# rounds with weights, blocks of a round.
NARM = 3 if SWA else 2
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
# The proj cores keep x prepared (q4_prep) for a phase's first NBC blocks,
# all of them where they fit in L1; past NBC (Qwen2.5-7B's down), a block's
# x is prepared every round, into 2 slots of its own.
NBC = min(NBMAX, 64)
NR = sum(I2P)  # rounds of a decode wave
# x landings of a decode wave, 512 at a time
NX = I2P[0] * KB + I2P[1] * DQ // 512 + I2P[2] * KB + I2P[3] * INTER // 512
NFILL = sum(I2P[p] * 2 * J2P[p] for p in range(4))  # fills of a weight stream
assert KVPC in (1, 2) and NQG in (1, 2)
