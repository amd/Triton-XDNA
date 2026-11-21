# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Qwen3-4B's decode as one superkernel: the 36 layers and the LM head of a
token in one dispatch, a wave a layer (or a vocab chunk of the LM head), on
the dataflow of air-FLM's fused decode (`fused_decode`, its `qwen3-4b`
entry).

Each wave streams its Q4 weights from DDR through memtiles to 16 projection
cores (2 blocks of 2x4), which compute its four phases -- QKV, o-proj,
up/gate, down -- a round of 512 rows at a time, each round gathered in a
memtile and sent on to the core that takes it: rope for QKV, the rms core for
o-proj and down, glu for up/gate. Attention runs a qk and a kv core for each
of 4 CUs, a CU 8 q heads on 2 kv heads; the KV cache comes in two groups, a
CU column's each, through that column's memtile. The rms core writes the
layer's output back over x.
"""

import triton
import triton.language as tl
from triton.language.extra import air

from gemma4_superkernel.kernels import (
    attn_kv_block_mm,
    attn_kv_fin_mm,
    attn_qk_block_mm,
    copy_n,
    q4_groups,
    q4_prep,
    q4_shift_table,
    residual,
    rms_norm,
)

from .geom import (
    D,
    DH,
    DK,
    DQ,
    EPS,
    GLU_HALF,
    GLU_SLICES,
    GW,
    I2P,
    INTER,
    J2P,
    KB,
    KVPC,
    LM_ROWS,
    LMR,
    NBMAX,
    NCH,
    NCU,
    NFA,
    NFILL,
    NKV,
    NQ,
    NR,
    NX,
    ROPE_W,
    SCALE,
)
from .kernels import glu_silu, rope

# as constexprs, for the jitted code
D_ = tl.constexpr(D)
DH_ = tl.constexpr(DH)
DQ_ = tl.constexpr(DQ)
DK_ = tl.constexpr(DK)
KB_ = tl.constexpr(KB)
LMR_ = tl.constexpr(LMR)
I2P_ = tl.constexpr(I2P)
J2P_ = tl.constexpr(J2P)
NBMAX_ = tl.constexpr(NBMAX)
NCH_ = tl.constexpr(NCH)
NQ_ = tl.constexpr(NQ)
NKV_ = tl.constexpr(NKV)
KVPC_ = tl.constexpr(KVPC)
GW_ = tl.constexpr(GW)
INTER_ = tl.constexpr(INTER)
GLU_HALF_ = tl.constexpr(GLU_HALF)
GLU_SLICES_ = tl.constexpr(GLU_SLICES)
NR_ = tl.constexpr(NR)
NX_ = tl.constexpr(NX)
NFILL_ = tl.constexpr(NFILL)
LM_ROWS_ = tl.constexpr(LM_ROWS)
ROPE_W_ = tl.constexpr(ROPE_W)
EPS_ = tl.constexpr(EPS)
SCALE_ = tl.constexpr(SCALE)
QPC = tl.constexpr(DQ // NCU)  # q of a CU
NF0, NF1, NF2, NF3 = NFA


def slab_offsets(arms):
    """The first fill of each wave's weight slab, waves of `arms`."""
    out, at = [], 0
    for a in arms:
        out.append(at)
        at += 8 * sum(NFA[p].value[a] for p in range(4))
    return tuple(out)


@triton.jit
def proj_core(arm, CX0: tl.constexpr):
    """A projection core of the 4x4: each round of a phase, its 32 rows' q4
    products with x; then the rows on to the phase's consumer."""
    cx = CX0 + air.herd_id(0)
    cy = air.herd_id(1)
    # x and weights ping-pong, as air-FLM's: the DMA fills one buffer while
    # the core reads the other
    w0 = air.alloc([2560], tl.int16, memory_space=air.L1)
    w1 = air.alloc([2560], tl.int16, memory_space=air.L1)
    x0 = air.alloc([256], tl.bfloat16, memory_space=air.L1)
    x1 = air.alloc([256], tl.bfloat16, memory_space=air.L1)
    acc = air.alloc([32], tl.float32, memory_space=air.L1)
    # x prepared for the products (q4_prep), a column block's per slot: the
    # same in every round of a phase, so prepared in its first
    xqc = air.alloc([NBMAX_ * 256], tl.bfloat16, memory_space=air.L1)
    sxc = air.alloc([NBMAX_ * 8], tl.float32, memory_space=air.L1)
    t = air.alloc([32], tl.float32, memory_space=air.L1)
    y = air.alloc([64], tl.bfloat16, memory_space=air.L1)
    tab = air.alloc([256], tl.int16, memory_space=air.L1)
    r = tl.arange(0, 32)
    q4_shift_table(tab)
    dec = (arm != 0).to(tl.int32)
    for ph in range(1 + 3 * dec):
        i2 = dec * air.table(I2P_, ph) + (1 - dec) * LMR_
        j2 = air.table(J2P_, ph)
        # one select, not arithmetic: the channel address encodes it
        dest = tl.where(dec != 0, air.table((0, 1, 2, 1), ph), 1)
        for i in range(i2):
            tl.store(acc + r, tl.zeros([32], tl.float32))
            first = (i == 0).to(tl.int32)
            for jp in range(j2):
                j = 2 * jp
                air.ChannelGet("inX", x0, indices=[cx, cy], sizes=[256])
                air.ChannelGet("inX", x1, indices=[cx, cy], sizes=[256])
                air.ChannelGet("wL2ToL1", w0, indices=[cx, cy], sizes=[2560])
                for _ in range(first):
                    q4_prep(x0, xqc + j * 256, sxc + j * 8, tab)
                q4_groups(xqc + j * 256, w0, acc, t, sxc + j * 8)
                air.ChannelGet("wL2ToL1", w1, indices=[cx, cy], sizes=[2560])
                for _ in range(first):
                    q4_prep(x1, xqc + j * 256 + 256, sxc + j * 8 + 8, tab)
                q4_groups(xqc + j * 256 + 256, w1, acc, t, sxc + j * 8 + 8)
            tl.store(y + 32 + r, tl.load(acc + r).to(tl.bfloat16))
            air.ChannelPut(
                "outA", y, indices=[cx, cy], offsets=[30], sizes=[34], dest=dest
            )


@triton.jit
def rms_core(arm):
    """The layer's norms and residuals: x normed for the QKV projection;
    o-proj's output added to x, that normed for the MLP; the MLP's output
    added, back to the host as the next layer's x. In an LM wave, the final
    norm and the logits' rounds on to the host."""
    x = air.alloc([D_], tl.bfloat16, memory_space=air.L1)
    w = air.alloc([D_], tl.bfloat16, memory_space=air.L1)
    w2 = air.alloc([D_], tl.bfloat16, memory_space=air.L1)
    xn = air.alloc([D_], tl.bfloat16, memory_space=air.L1)
    o = air.alloc([D_], tl.bfloat16, memory_space=air.L1)
    h = air.alloc([D_], tl.bfloat16, memory_space=air.L1)
    sq = air.alloc([64], tl.float32, memory_space=air.L1)
    lo = air.alloc([D_], tl.bfloat16, memory_space=air.L1)
    air.ChannelGet("rmsX", x, sizes=[D_])
    air.ChannelGet("rmsW", w, sizes=[D_])
    air.ChannelGet("rmsW2", w2, sizes=[D_])
    dec = (arm != 0).to(tl.int32)
    # loops of 0 or 1 trips, not ifs: pointers keep their structure
    for _ in range(dec):
        rms_norm(xn, x, w, sq, D_, EPS_)
        for n in range(I2P_[0]):
            air.ChannelPut("xnorm", xn, offsets=[0], sizes=[D_])
        air.ChannelGet("outY", o, indices=[0, 1], sizes=[D_])
        residual(h, x, o, D_)
        rms_norm(xn, h, w2, sq, D_, EPS_)
        for n in range(I2P_[2]):
            air.ChannelPut("xnorm", xn, offsets=[0], sizes=[D_])
        air.ChannelGet("outY", o, indices=[0, 1], sizes=[D_])
        residual(lo, h, o, D_)
        air.ChannelPut("layerOut", lo, offsets=[0], sizes=[D_])
    # an LM wave: the final norm (w), x for its rounds, and the rounds'
    # logits on to the host, D at a time (KB rounds) as air-FLM
    LB: tl.constexpr = LMR_ // KB_
    for _ in range(1 - dec):
        rms_norm(xn, x, w, sq, D_, EPS_)
        for n in range(KB_):
            air.ChannelPut("xnorm", xn, offsets=[0], sizes=[D_])
        for i in range(LB):
            for _ in range((i < LB - 1).to(tl.int32)):
                for n in range(KB_):
                    air.ChannelPut("xnorm", xn, offsets=[0], sizes=[D_])
            air.ChannelGet("outY", o, indices=[0, 1], sizes=[D_])
            # a buffer of its own, which the core writes before each put
            copy_n(lo, o, D_)
            air.ChannelPut("layerOut", lo, offsets=[0], sizes=[D_])


@triton.jit
def rope_core(arm):
    """q, k and v of the layer from the QKV projection's rows (`rope`); k and
    v also to the host, which appends them to the KV cache."""
    qkv = air.alloc([DQ_ + 2 * DK_], tl.bfloat16, memory_space=air.L1)
    w = air.alloc([ROPE_W_], tl.bfloat16, memory_space=air.L1)
    q = air.alloc([DQ_], tl.bfloat16, memory_space=air.L1)
    kk = air.alloc([DK_], tl.bfloat16, memory_space=air.L1)
    v = air.alloc([DK_], tl.bfloat16, memory_space=air.L1)
    sq = air.alloc([64], tl.float32, memory_space=air.L1)
    for _ in range((arm != 0).to(tl.int32)):
        for n in range(I2P_[0]):
            air.ChannelGet("outY", qkv, indices=[0, 0], offsets=[n * 512], sizes=[512])
        air.ChannelGet("ropeLUT", w, sizes=[ROPE_W_])
        rope(q, kk, v, qkv, w, sq, NQ_, NKV_, DH_, EPS_)
        air.ChannelPut("ropeQ", q, offsets=[0], sizes=[DQ_])
    # in every wave, as the host's gets: an LM wave's are not read
    air.ChannelPut("appendK", kk, offsets=[0], sizes=[DK_])
    air.ChannelPut("appendV", v, offsets=[0], sizes=[DK_])


@triton.jit
def qk_scores(q, kb, mst, pc, acc, tmp, sb, mb, blk, pos, G: tl.constexpr):
    """Group G's (a kv head's 4 q heads') scores of block blk."""
    attn_qk_block_mm(
        q + G * (QPC // KVPC_),
        kb,
        mst + G * 32,
        pc + G * 96,
        acc,
        tmp,
        sb,
        mb,
        blk,
        pos + 1,
        0,
        NCH_,
        SCALE_,
    )


@triton.jit
def qk_core(arm, pos, CU: tl.constexpr):
    """Attention scores of a CU's 8 heads (two groups of 4, one a kv head)
    over the 16-key blocks up to the token's position pos (set each token),
    the keys after it masked, scaled by 1/sqrt(DH): a pair of blocks a trip,
    each kv head's keys of a block into a buffer of their own (the DMA fills
    one while the core reads another), p and c to the kv core."""
    HB: tl.constexpr = 16 * DH_  # a kv head's block
    q = air.alloc([QPC], tl.bfloat16, memory_space=air.L1)
    ka = air.alloc([HB], tl.bfloat16, memory_space=air.L1)
    kb = air.alloc([HB], tl.bfloat16, memory_space=air.L1)
    ka1 = air.alloc([HB], tl.bfloat16, memory_space=air.L1)
    kb1 = air.alloc([HB], tl.bfloat16, memory_space=air.L1)
    mst = air.alloc([32 * KVPC_], tl.float32, memory_space=air.L1)
    pc = air.alloc([96 * KVPC_], tl.bfloat16, memory_space=air.L1)
    pc1 = air.alloc([96 * KVPC_], tl.bfloat16, memory_space=air.L1)
    acc = air.alloc([32], tl.float32, memory_space=air.L1)
    tmp = air.alloc([64], tl.float32, memory_space=air.L1)
    sb = air.alloc([64], tl.float32, memory_space=air.L1)
    mb = air.alloc([64], tl.float32, memory_space=air.L1)
    for _ in range((arm != 0).to(tl.int32)):
        tl.store(
            mst + tl.arange(0, 32 * KVPC_), tl.full([32 * KVPC_], -1.0e4, tl.float32)
        )
        air.ChannelGet("toAttnQ", q, indices=[CU], sizes=[QPC])
        n2 = ((pos + 16) // 16 + 1) // 2
        for i in range(n2):
            b = 2 * i
            air.ChannelGet("toK", ka, indices=[CU], sizes=[HB])
            air.ChannelGet("toK", kb, indices=[CU], sizes=[HB])
            qk_scores(q, ka, mst, pc, acc, tmp, sb, mb, b, pos, 0)
            qk_scores(q, kb, mst, pc, acc, tmp, sb, mb, b, pos, 1)
            air.ChannelPut("sPC", pc, indices=[CU], offsets=[0], sizes=[96 * KVPC_])
            air.ChannelGet("toK", ka1, indices=[CU], sizes=[HB])
            air.ChannelGet("toK", kb1, indices=[CU], sizes=[HB])
            qk_scores(q, ka1, mst, pc1, acc, tmp, sb, mb, b + 1, pos, 0)
            qk_scores(q, kb1, mst, pc1, acc, tmp, sb, mb, b + 1, pos, 1)
            air.ChannelPut("sPC", pc1, indices=[CU], offsets=[0], sizes=[96 * KVPC_])


@triton.jit
def kv_core(arm, pos, CU: tl.constexpr):
    """A CU's attention sums from its qk core's p and c and the values, a kv
    head's values of a block in a buffer of their own; then its output y / l,
    blocked [kv head][NCH][4][8]."""
    HB: tl.constexpr = 16 * DH_
    YB: tl.constexpr = NCH_ * 32  # a group's sums
    va = air.alloc([HB], tl.bfloat16, memory_space=air.L1)
    vb = air.alloc([HB], tl.bfloat16, memory_space=air.L1)
    va1 = air.alloc([HB], tl.bfloat16, memory_space=air.L1)
    vb1 = air.alloc([HB], tl.bfloat16, memory_space=air.L1)
    pc = air.alloc([96 * KVPC_], tl.bfloat16, memory_space=air.L1)
    pc1 = air.alloc([96 * KVPC_], tl.bfloat16, memory_space=air.L1)
    y = air.alloc([YB * KVPC_], tl.float32, memory_space=air.L1)
    lst = air.alloc([32 * KVPC_], tl.float32, memory_space=air.L1)
    ct = air.alloc([32], tl.float32, memory_space=air.L1)
    o = air.alloc([QPC], tl.bfloat16, memory_space=air.L1)
    for _ in range((arm != 0).to(tl.int32)):
        k = tl.arange(0, 32)
        for c in range(YB * KVPC_ // 32):
            tl.store(y + c * 32 + k, tl.zeros([32], tl.float32))
        for g in tl.static_range(KVPC_):
            tl.store(lst + g * 32 + k, tl.zeros([32], tl.float32))
        n2 = ((pos + 16) // 16 + 1) // 2
        for i in range(n2):
            air.ChannelGet("toV", va, indices=[CU], sizes=[HB])
            air.ChannelGet("toV", vb, indices=[CU], sizes=[HB])
            air.ChannelGet("sPC", pc, indices=[CU], sizes=[96 * KVPC_])
            attn_kv_block_mm(va, pc, y, lst, ct, NCH_)
            attn_kv_block_mm(vb, pc + 96, y + YB, lst + 32, ct, NCH_)
            air.ChannelGet("toV", va1, indices=[CU], sizes=[HB])
            air.ChannelGet("toV", vb1, indices=[CU], sizes=[HB])
            air.ChannelGet("sPC", pc1, indices=[CU], sizes=[96 * KVPC_])
            attn_kv_block_mm(va1, pc1, y, lst, ct, NCH_)
            attn_kv_block_mm(vb1, pc1 + 96, y + YB, lst + 32, ct, NCH_)
        for g in tl.static_range(KVPC_):
            attn_kv_fin_mm(o + g * (QPC // KVPC_), y + g * YB, lst + g * 32, ct, NCH_)
        air.ChannelPut("attnO", o, indices=[CU], offsets=[0], sizes=[QPC])


@triton.jit
def glu_core(arm):
    """silu(gate) up of the up/gate projection's rows, GLU_HALF at a time."""
    gu = air.alloc([2 * GLU_HALF_], tl.bfloat16, memory_space=air.L1)
    y = air.alloc([GLU_HALF_], tl.bfloat16, memory_space=air.L1)
    for _ in range((arm != 0).to(tl.int32)):
        for s in range(GLU_SLICES_):
            air.ChannelGet("outY", gu, indices=[0, 2], sizes=[2 * GLU_HALF_])
            glu_silu(y, gu, GLU_HALF_)
            air.ChannelPut("gluOut", y, offsets=[0], sizes=[GLU_HALF_])


@triton.jit
def feed_weights(W, at, NF: tl.constexpr):
    """A wave's NF fills of each of the 8 weight streams, from fill at."""
    for x in tl.static_range(4):
        for k in tl.static_range(2):
            air.ChannelPut(
                "inW",
                W,
                indices=[x, k],
                offsets=[(at + (x * 2 + k) * NF) * 5120],
                sizes=[NF * 5120],
            )


@triton.jit
def feed_arm(W, at, A: tl.constexpr):
    """The weight slab of a wave of arm A, from fill at: [stream][phase][fill],
    so that each stream's fills of the wave are one task."""
    F: tl.constexpr = NF0[A] + NF1[A] + NF2[A] + NF3[A]
    feed_weights(W, at, F)


@triton.jit
def relay_x(N: tl.constexpr):
    """N 512s of x into memtile 3, each to the proj cores as 2 x 256."""
    for n in range(N):
        xl = air.alloc([512], tl.bfloat16, memory_space=air.L2, memtile_col=3)
        air.ChannelGet("xnorm", xl, offsets=[0], sizes=[512])
        for hh in tl.static_range(2):
            air.ChannelPut("inX", xl, indices=[0, 0], offsets=[hh * 256], sizes=[256])


@triton.jit
def gather_rounds(N: tl.constexpr):
    """N rounds' rows, gathered behind the first core's packet header: a
    memtile a projection column, then the main memtile (4)."""
    for i in range(N):
        for x in tl.static_range(4):
            g = air.alloc(
                [130],
                tl.bfloat16,
                memory_space=air.L2,
                memtile_col=x + (x // 2) * 4,
            )
            air.ChannelGet("outA", g, indices=[x, 0], offsets=[0], sizes=[34])
            for c in tl.static_range(1, 4):
                air.ChannelGet(
                    "outA", g, indices=[x, c], offsets=[2 + c * 32], sizes=[32]
                )
            air.ChannelPut("toMain", g, indices=[x], offsets=[0], sizes=[130])
        m = air.alloc([514], tl.bfloat16, memory_space=air.L2, memtile_col=4)
        air.ChannelGet("toMain", m, indices=[0], offsets=[0], sizes=[130])
        for x in tl.static_range(1, 4):
            air.ChannelGet("toMain", m, indices=[x], offsets=[2 + x * 128], sizes=[128])
        air.ChannelPut("outY", m, indices=[0, 0], offsets=[0], sizes=[514])


@triton.jit
def wave(
    XL,
    RMS,
    W,
    ROPE,
    KV,
    pos,
    NW: tl.constexpr,
    S: tl.constexpr,
    SM: tl.constexpr,
    MAXL: tl.constexpr,
    ARMS: tl.constexpr,
    SLABS: tl.constexpr,
    KVAPP: tl.constexpr,
    NDEC: tl.constexpr,
):
    """One token of Qwen3-4B: NW waves, the NDEC decoder layers then the LM
    head's vocab chunks, all in one dispatch. A wave's arm (ARMS) is 1 for a
    layer, 0 for an LM wave. Buffers (bf16 unless said):

    XL    [x | logits]: the host writes the token's embedding into x, each
          layer the next x; an LM wave a chunk of logits
    RMS   each layer's 2 norm weights, then the final norm's
    W     the Q4 weights, a slab a wave (SLABS: its first fill)
    ROPE  each layer's [cos | sin | q_norm | k_norm] at this position
    KV    a layer's [K group 0 | K group 1 | V group 0 | V group 1], each
          MAXL rows of GW (KVAPP: the slab a wave appends to), then a
          scratch slab an LM wave appends to
    pos   the token's position
    S, SM the transform scripts of the cores and of the attention cores.
    """
    air.backend(
        use_lock_race_condition_fix_v2=True,
        merge_contiguous_feeds=False,
        stack_size=6144,
        allow_deadlock_prone_routing=True,
    )
    air.Channel("rmsX", channel_type="npu_dma_packet")
    air.Channel("rmsW")
    air.Channel("rmsW2")
    air.Channel("xnorm", channel_type="npu_dma_packet")
    air.Channel("inX", size=[1, 1], broadcast_shape=[4, 4])
    air.Channel("inW", size=[4, 2])
    air.Channel("wL2ToL1", size=[4, 4])
    air.Channel("outA", size=[4, 4], channel_type="npu_dma_packet")
    air.Channel("toMain", size=[4], channel_type="npu_dma_packet")
    # the consumers take it on their second DMA channel, their inputs on the
    # first
    air.Channel(
        "outY",
        size=[1, 1],
        broadcast_shape=[1, 3],
        channel_type="npu_dma_packet",
        tile_dma_channel=1,
    )
    air.Channel("attnO", size=[4])
    air.Channel("ropeLUT")
    air.Channel("ropeQ")
    air.Channel("toAttnQ", size=[4])
    air.Channel("appendK")
    air.Channel("appendV")
    air.Channel("inKV_K", size=[2])
    air.Channel("inKV_V", size=[2])
    air.Channel("toK", size=[4])
    air.Channel("toV", size=[4])
    air.Channel("sPC", size=[4])
    air.Channel("gluOut")
    air.Channel("layerOut")
    # host
    for wave in range(NW):
        arm = air.table(ARMS, wave)
        # Host branches hold only puts: AIR keeps the waits a branch's
        # transfers have on what comes before it, but nothing after it waits
        # on them, so every get is in all waves.
        lm = arm == 0
        air.ChannelPut("rmsX", XL, offsets=[0], sizes=[D_])
        # an LM wave's rmsW: the final norm; its rmsW2 unused
        air.ChannelPut(
            "rmsW", RMS, offsets=[tl.where(lm, NDEC, wave) * 2 * D_], sizes=[D_]
        )
        air.ChannelPut(
            "rmsW2", RMS, offsets=[tl.where(lm, 0, wave * 2 * D_ + D_)], sizes=[D_]
        )
        if arm != 0:
            air.ChannelPut("ropeLUT", ROPE, offsets=[wave * ROPE_W_], sizes=[ROPE_W_])
        # the wave's weights, their lengths the arm's: a branch an arm, as
        # air-FLM's index_switch. Before the KV append: the host waits for
        # the append before it reads the KV back, and the append needs them.
        at = air.table(SLABS, wave)
        if arm == 1:
            feed_arm(W, at, 1)
        if arm == 0:
            feed_arm(W, at, 0)
        RS: tl.constexpr = MAXL * GW_  # a group's region
        # this token's k and v at row pos of the wave's slab: each group's
        # part of the row in its region
        ks = air.table(KVAPP, wave) * 4 * RS + pos * GW_
        air.ChannelGet("appendK", KV, offsets=[0, ks], sizes=[2, GW_], strides=[RS, 1])
        air.ChannelGet(
            "appendV", KV, offsets=[0, ks + 2 * RS], sizes=[2, GW_], strides=[RS, 1]
        )
        # then the blocks attention reads, of whole pairs, up to pos
        NB: tl.constexpr = MAXL // 16
        nh = ((pos + 16) // 16 + 1) // 2 * 2
        if arm != 0:
            k0 = wave * 4 * RS
            for g in tl.static_range(2):
                air.ChannelPut(
                    "inKV_K",
                    KV,
                    indices=[g],
                    offsets=[k0 + g * RS],
                    sizes=[nh * 16 * GW_],
                )
                air.ChannelPut(
                    "inKV_V",
                    KV,
                    indices=[g],
                    offsets=[k0 + (2 + g) * RS],
                    sizes=[nh * 16 * GW_],
                )
        # x of every round: 512 at a time into memtile 3, broadcast as 2 x 256.
        # The memtiles' relays run without counts, so these loops' bounds also
        # cover an LM wave's rounds: no branches here.
        relay_x(NX_)
        # weights: a fill of a memtile's buffer is a block for each of 2 cores
        for x in tl.static_range(4):
            for k in tl.static_range(2):
                for n in range(NFILL_):
                    wf = air.alloc(
                        [5120],
                        tl.int16,
                        memory_space=air.L2,
                        memtile_col=x + (x // 2) * 4,
                    )
                    air.ChannelGet("inW", wf, indices=[x, k], offsets=[0], sizes=[5120])
                    for c in tl.static_range(2):
                        air.ChannelPut(
                            "wL2ToL1",
                            wf,
                            indices=[x, 2 * k + c],
                            offsets=[c * 2560],
                            sizes=[2560],
                        )
        # q: a CU's 8 heads, as its two groups of 4 blocked [NCH][4][8]
        rq = air.alloc([DQ_], tl.bfloat16, memory_space=air.L2, memtile_col=5)
        air.ChannelGet("ropeQ", rq, offsets=[0], sizes=[DQ_])
        for cu in tl.static_range(4):
            air.ChannelPut(
                "toAttnQ",
                rq,
                indices=[cu],
                offsets=[cu * KVPC_, 0, 0, 0],
                sizes=[KVPC_, NCH_, 4, 8],
                strides=[4 * DH_, 8, DH_, 1],
            )
        # each group's blocks of 16 rows in its column's memtile (3 or 4), to
        # its 2 CUs: a kv head's keys at a time, blocked [2][NCH][8][8] (8
        # keys by 8 columns)
        for g in tl.static_range(2):
            for b in range(NB):
                t = air.alloc(
                    [16 * GW_], tl.bfloat16, memory_space=air.L2, memtile_col=3 + g
                )
                air.ChannelGet("inKV_K", t, indices=[g], offsets=[0], sizes=[16 * GW_])
                for lc in tl.static_range(2):
                    for h in tl.static_range(KVPC_):
                        air.ChannelPut(
                            "toK",
                            t,
                            indices=[2 * g + lc],
                            offsets=[0, 0, 0, (lc * KVPC_ + h) * DH_],
                            sizes=[2, NCH_, 8, 8],
                            strides=[8 * GW_, 8, GW_, 1],
                        )
            for b in range(NB):
                t = air.alloc(
                    [16 * GW_], tl.bfloat16, memory_space=air.L2, memtile_col=3 + g
                )
                air.ChannelGet("inKV_V", t, indices=[g], offsets=[0], sizes=[16 * GW_])
                for lc in tl.static_range(2):
                    for h in tl.static_range(KVPC_):
                        air.ChannelPut(
                            "toV",
                            t,
                            indices=[2 * g + lc],
                            offsets=[0, 0, 0, (lc * KVPC_ + h) * DH_],
                            sizes=[2, NCH_, 8, 8],
                            strides=[8 * GW_, 8, GW_, 1],
                        )
        # the attention output, heads in order, as o-proj's x a round each
        og = air.alloc([DQ_], tl.bfloat16, memory_space=air.L2, memtile_col=5)
        for cu in tl.static_range(4):
            air.ChannelGet(
                "attnO",
                og,
                indices=[cu],
                offsets=[cu * KVPC_, 0, 0, 0],
                sizes=[KVPC_, NCH_, 4, 8],
                strides=[4 * DH_, 8, DH_, 1],
            )
        for n in range(I2P_[1]):
            air.ChannelPut("xnorm", og, offsets=[0], sizes=[DQ_])
        # glu's output, as down's x a round each
        dx = air.alloc([INTER_], tl.bfloat16, memory_space=air.L2, memtile_col=3)
        for s in range(GLU_SLICES_):
            air.ChannelGet("gluOut", dx, offsets=[s * GLU_HALF_], sizes=[GLU_HALF_])
        for n in range(I2P_[3]):
            air.ChannelPut("xnorm", dx, offsets=[0], sizes=[INTER_])
        air.herd(proj_core, (arm, 0), (2, 4), S, name="proj_blk0", at=(0, 2))
        air.herd(proj_core, (arm, 2), (2, 4), S, name="proj_blk1", at=(6, 2))
        gather_rounds(NR_)
        air.herd(rope_core, (arm,), 1, S, name="rope", at=(2, 3))
        # attention: a qk and a kv core per CU, CUs 0 and 1 in column 3, 2
        # and 3 in column 4
        air.herd(qk_core, (arm, pos, 0), 1, SM, name="qk0", at=(3, 2))
        air.herd(kv_core, (arm, pos, 0), 1, SM, name="kv0", at=(3, 3))
        air.herd(qk_core, (arm, pos, 1), 1, SM, name="qk1", at=(3, 4))
        air.herd(kv_core, (arm, pos, 1), 1, SM, name="kv1", at=(3, 5))
        air.herd(qk_core, (arm, pos, 2), 1, SM, name="qk2", at=(4, 2))
        air.herd(kv_core, (arm, pos, 2), 1, SM, name="kv2", at=(4, 3))
        air.herd(qk_core, (arm, pos, 3), 1, SM, name="qk3", at=(4, 4))
        air.herd(kv_core, (arm, pos, 3), 1, SM, name="kv3", at=(4, 5))
        air.herd(rms_core, (arm,), 1, S, name="rms", at=(2, 2))
        air.herd(glu_core, (arm,), 1, S, name="glu", at=(5, 3))
        # the layer's output over x; an LM wave's logits after it
        air.ChannelGet(
            "layerOut",
            XL,
            offsets=[tl.where(lm, D_ + (wave - NDEC) * LM_ROWS_, 0)],
            sizes=[tl.where(lm, LM_ROWS_, D_)],
        )
