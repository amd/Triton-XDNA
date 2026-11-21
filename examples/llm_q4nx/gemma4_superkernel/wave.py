# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Gemma4's decode as one superkernel: all the layers and the LM head of a
token in one dispatch, a wave a layer (or a vocab chunk of the LM head), on
the dataflow of air-FLM's fused decode (`fused_decode_ple`). The sizes are the
variant's (`geom`).

Each wave streams its Q4 weights from DDR through memtiles to 16 projection
cores (2 blocks of 2x4), which compute its four phases -- QKV, o-proj,
up/gate, down -- a round of 512 rows at a time, each round gathered in a
memtile and sent on to the core that takes it: rope for QKV, the rms core for
o-proj and down, glu for up/gate. Attention runs a qk and a kv core for each
of the two head groups; on E2B, whose one kv head both groups share, each
group's context is split in two such parts, and merged. The PLE cores add
the per-layer input after the MLP. Layers differ (wide or narrow MLP, sliding
or full attention, own or shared KV cache); a wave's arm selects which.
"""

import triton
import triton.language as tl
from triton.language.extra import air

from .geom import (
    D,
    DQ,
    GLU_SLICES,
    I2A,
    I2P,
    INTER,
    J2P,
    KB,
    KV_REPL,
    KVW,
    LM_ROWS,
    LMR,
    MASKA,
    NBA,
    NBMAX,
    NFA,
    NFILL,
    NR,
    NX,
    PARTS,
    PK,
    PLE_BLK,
    PLE_FILLS,
    PXW,
    RMS_LAYER,
)
from .kernels import (
    add_bf,
    attn_kv_block_mm,
    attn_kv_fin_mm,
    attn_merge,
    attn_qk_block_mm,
    copy_n,
    flush32,
    gelu_inplace,
    glu,
    kv_part_out,
    mul_bf,
    mvm_blk,
    mvm_half,
    q4_groups,
    q4_prep,
    q4_shift_table,
    qk_m_out,
    residual,
    rms_norm,
    rope,
    scale_bf,
    zero32,
)

EPS = 1e-6
# as constexprs, for the jitted code
D_ = tl.constexpr(D)
DQ_ = tl.constexpr(DQ)
KB_ = tl.constexpr(KB)
LMR_ = tl.constexpr(LMR)
I2P_ = tl.constexpr(I2P)
J2P_ = tl.constexpr(J2P)
NBMAX_ = tl.constexpr(NBMAX)
PARTS_ = tl.constexpr(PARTS)
KV_REPL_ = tl.constexpr(KV_REPL)
KVW_ = tl.constexpr(KVW)
GLU_SLICES_ = tl.constexpr(GLU_SLICES)
INTER_ = tl.constexpr(INTER)
NR_ = tl.constexpr(NR)
NX_ = tl.constexpr(NX)
NFILL_ = tl.constexpr(NFILL)
LM_ROWS_ = tl.constexpr(LM_ROWS)
RMS_LAYER_ = tl.constexpr(RMS_LAYER)
PXW_ = tl.constexpr(PXW)
PLE_FILLS_ = tl.constexpr(PLE_FILLS)
PLE_BLK_ = tl.constexpr(PLE_BLK)
PK_ = tl.constexpr(PK)
PLE_SCALE_ = tl.constexpr(D**-0.5)  # the PLE projection's
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
    products with x, the x of every block streamed, the weights only of the
    blocks the arm's layer has; then the rows on to the phase's consumer."""
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
        i2a = air.table(I2A, ph * 8 + arm)
        nba = air.table(NBA, ph * 8 + arm)
        mask = air.table(MASKA, ph * 8 + arm)
        # x comes for every block; weights only for the arm's own
        for i in range(i2):
            tl.store(acc + r, tl.zeros([32], tl.float32))
            for jp in range(j2):
                j = 2 * jp
                air.ChannelGet("inX", x0, indices=[cx, cy], sizes=[256])
                air.ChannelGet("inX", x1, indices=[cx, cy], sizes=[256])
                # Once or not at all: pointers under an scf.if lose their
                # structure in triton-shared, under a loop they keep it. Blocks
                # j and j + 1 share it (nba is even, the masks skip pairs), so
                # their two weight gets repeat together.
                on = ((i < i2a) & (j < nba) & ((j & mask) == 0)).to(tl.int32)
                for _ in range(on):
                    first = (i == 0).to(tl.int32)
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
def rms_core(shr, arm, EPS: tl.constexpr):
    """The layer's norms and residuals: x normed for the QKV projection; o-proj's
    output normed onto x; that normed for the MLP; the MLP's output normed onto
    it, to plegate through shr. In an LM wave, the final norm and the logits'
    rounds on to the host."""
    x = air.alloc([D_], tl.bfloat16, memory_space=air.L1)
    w = air.alloc([2 * D_], tl.bfloat16, memory_space=air.L1)
    w2 = air.alloc([2 * D_], tl.bfloat16, memory_space=air.L1)
    xn = air.alloc([D_], tl.bfloat16, memory_space=air.L1)
    o = air.alloc([D_], tl.bfloat16, memory_space=air.L1)
    t = air.alloc([D_], tl.bfloat16, memory_space=air.L1)
    h = air.alloc([D_], tl.bfloat16, memory_space=air.L1)
    sq = air.alloc([64], tl.float32, memory_space=air.L1)
    lo = air.alloc([D_], tl.bfloat16, memory_space=air.L1)
    air.ChannelGet("rmsX", x, sizes=[D_])
    air.ChannelGet("rmsW", w, sizes=[2 * D_])
    air.ChannelGet("rmsW2", w2, sizes=[2 * D_])
    dec = (arm != 0).to(tl.int32)
    # loops of 0 or 1 trips, not ifs: pointers keep their structure
    for _ in range(dec):
        rms_norm(xn, x, w, sq, D_, EPS)
        for n in range(I2P_[0]):
            air.ChannelPut("xnorm", xn, offsets=[0], sizes=[D_])
        air.ChannelGet("outY", o, indices=[0, 1], sizes=[D_])
        rms_norm(t, o, w + D_, sq, D_, EPS)
        residual(h, x, t, D_)
        rms_norm(xn, h, w2, sq, D_, EPS)
        for n in range(I2P_[2]):
            air.ChannelPut("xnorm", xn, offsets=[0], sizes=[D_])
        air.ChannelGet("outY", o, indices=[0, 1], sizes=[D_])
        rms_norm(t, o, w2 + D_, sq, D_, EPS)
        residual(x, h, t, D_)
        # to plegate, in one copy loop of the shape it reads it in
        copy_n(shr, x, D_)
    # an LM wave: the final norm (the hi half of w), x for its rounds, and the
    # rounds' logits on to the host, D at a time (KB rounds) as air-FLM. The
    # host's gets are not in branches (their waits would be lost), so a
    # decode wave sends a chunk of logits too, to where nothing reads.
    LB: tl.constexpr = LMR_ // KB_
    for _ in range(1 - dec):
        rms_norm(xn, x, w + D_, sq, D_, EPS)
        for n in range(KB_):
            air.ChannelPut("xnorm", xn, offsets=[0], sizes=[D_])
    # (a decode wave sends one: the host waits for every transfer it gets
    # before the next wave, and each takes the DMA's latency)
    for i in range(LB - (LB - 1) * dec):
        for _ in range((1 - dec) * (i < LB - 1).to(tl.int32)):
            for n in range(KB_):
                air.ChannelPut("xnorm", xn, offsets=[0], sizes=[D_])
        for _ in range(1 - dec):
            air.ChannelGet("outY", o, indices=[0, 1], sizes=[D_])
        # a buffer of its own, which the core writes before each put
        copy_n(lo, o, D_)
        air.ChannelPut("layerOut", lo, offsets=[0], sizes=[D_])


@triton.jit
def rope_core(arm):
    """q, k and v of the layer from the QKV projection's rows (`rope`); k and
    v also to the host, which appends them to the KV cache."""
    qkv = air.alloc([6144], tl.bfloat16, memory_space=air.L1)
    sp = air.alloc([6144], tl.bfloat16, memory_space=air.L1)
    w = air.alloc([1536], tl.bfloat16, memory_space=air.L1)
    q = air.alloc([4096], tl.bfloat16, memory_space=air.L1)
    kk = air.alloc([1024], tl.bfloat16, memory_space=air.L1)
    v = air.alloc([1024], tl.bfloat16, memory_space=air.L1)
    sq = air.alloc([64], tl.float32, memory_space=air.L1)
    for _ in range((arm != 0).to(tl.int32)):
        for n in range(I2P_[0]):
            air.ChannelGet("outY", qkv, indices=[0, 0], offsets=[n * 512], sizes=[512])
        air.ChannelGet("ropeLUT", w, sizes=[1536])
        # arms 5 and 7 slide (no scalar shift: mlir-aie rounds it)
        rope(q, kk, v, qkv, sp, w, sq, ((arm == 5) | (arm == 7)).to(tl.int32))
        air.ChannelPut("ropeQ", q, offsets=[0], sizes=[4096])
    # in every wave, as the host's gets: an LM wave's are not read (E2B's
    # one kv head comes twice, a copy a CU: the first)
    air.ChannelPut("appendK", kk, offsets=[0], sizes=[KVW_])
    air.ChannelPut("appendV", v, offsets=[0], sizes=[KVW_])


@triton.jit
def window_start(arm, pos, SWA: tl.constexpr):
    """The first key a sliding layer (arms 5, 7) sees: pos - SWA + 1."""
    sw = ((arm == 5) | (arm == 7)).to(tl.int32)
    return tl.where(pos >= SWA, pos - SWA + 1, 0) * sw


@triton.jit
def block_pairs(arm, pos, SWA: tl.constexpr):
    """The first KV block attention reads and the pairs each part reads (its
    cores' K and V buffers ping-pong): the window's blocks, in PARTS_ equal
    shares, the count rounded up with blocks whose keys are after pos
    (masked)."""
    b0 = window_start(arm, pos, SWA) // 16
    return b0, ((pos + 16) // 16 - b0 + 2 * PARTS_ - 1) // (2 * PARTS_)


@triton.jit
def qk_core(arm, pos, CU: tl.constexpr, PART: tl.constexpr, SWA: tl.constexpr):
    """Attention scores of a CU's 4 heads over the 16-key blocks up to the
    token's position pos (set each token), the keys after it masked, and
    in a sliding layer (arms 5, 7) those SWA or more before it: the part's
    half of the blocks. q and the keys come
    blocked for the matrix products (attn_qk_block_mm). Then the maxima,
    for the merge."""
    q = air.alloc([2048], tl.bfloat16, memory_space=air.L1)
    kb = air.alloc([8192], tl.bfloat16, memory_space=air.L1)
    kb1 = air.alloc([8192], tl.bfloat16, memory_space=air.L1)
    mst = air.alloc([32], tl.float32, memory_space=air.L1)
    pc = air.alloc([96], tl.bfloat16, memory_space=air.L1)
    pc1 = air.alloc([96], tl.bfloat16, memory_space=air.L1)
    acc = air.alloc([32], tl.float32, memory_space=air.L1)
    tmp = air.alloc([64], tl.float32, memory_space=air.L1)
    sb = air.alloc([64], tl.float32, memory_space=air.L1)
    mb = air.alloc([64], tl.float32, memory_space=air.L1)
    for _ in range((arm != 0).to(tl.int32)):
        tl.store(mst + tl.arange(0, 32), tl.full([32], -1.0e4, tl.float32))
        air.ChannelGet("toAttnQ", q, indices=[CU, PART], sizes=[2048])
        lo = window_start(arm, pos, SWA)
        b0, n4 = block_pairs(arm, pos, SWA)
        # a block into each buffer a trip: one fills while the other is read
        for i in range(n4):
            b = b0 + PART * 2 * n4 + 2 * i
            if PART == 0:
                air.ChannelGet("toK", kb, indices=[0, CU], sizes=[8192])
            else:
                air.ChannelGet("toK1", kb, indices=[0, CU], sizes=[8192])
            attn_qk_block_mm(q, kb, mst, pc, acc, tmp, sb, mb, b, pos + 1, lo)
            air.ChannelPut("sPC", pc, indices=[CU, PART], offsets=[0], sizes=[96])
            if PART == 0:
                air.ChannelGet("toK", kb1, indices=[0, CU], sizes=[8192])
            else:
                air.ChannelGet("toK1", kb1, indices=[0, CU], sizes=[8192])
            attn_qk_block_mm(q, kb1, mst, pc1, acc, tmp, sb, mb, b + 1, pos + 1, lo)
            air.ChannelPut("sPC", pc1, indices=[CU, PART], offsets=[0], sizes=[96])
        if PARTS_ == 2:
            # the maxima, for the merge, in two messages (the buffers
            # alternate)
            qk_m_out(mst, pc)
            air.ChannelPut("sPC", pc, indices=[CU, PART], offsets=[0], sizes=[96])
            air.ChannelPut("sPC", pc1, indices=[CU, PART], offsets=[0], sizes=[96])


@triton.jit
def kv_core(arm, pos, CU: tl.constexpr, PART: tl.constexpr, SWA: tl.constexpr):
    """A part's sums for a CU from its qk core's p and c and the values
    (blocked, attn_kv_block_mm): with two parts, with l and m for the merge
    (kv_part_out); with one, the CU's attention output (attn_kv_fin_mm)."""
    vb = air.alloc([8192], tl.bfloat16, memory_space=air.L1)
    vb1 = air.alloc([8192], tl.bfloat16, memory_space=air.L1)
    pc = air.alloc([96], tl.bfloat16, memory_space=air.L1)
    pc1 = air.alloc([96], tl.bfloat16, memory_space=air.L1)
    y = air.alloc([2112], tl.float32, memory_space=air.L1)
    lst = air.alloc([32], tl.float32, memory_space=air.L1)
    ct = air.alloc([32], tl.float32, memory_space=air.L1)
    for _ in range((arm != 0).to(tl.int32)):
        k = tl.arange(0, 32)
        for c in range(64):
            tl.store(y + c * 32 + k, tl.zeros([32], tl.float32))
        tl.store(lst + k, tl.zeros([32], tl.float32))
        _, n4 = block_pairs(arm, pos, SWA)
        for i in range(n4):
            if PART == 0:
                air.ChannelGet("toV", vb, indices=[0, CU], sizes=[8192])
            else:
                air.ChannelGet("toV1", vb, indices=[0, CU], sizes=[8192])
            air.ChannelGet("sPC", pc, indices=[CU, PART], sizes=[96])
            attn_kv_block_mm(vb, pc, y, lst, ct)
            if PART == 0:
                air.ChannelGet("toV", vb1, indices=[0, CU], sizes=[8192])
            else:
                air.ChannelGet("toV1", vb1, indices=[0, CU], sizes=[8192])
            air.ChannelGet("sPC", pc1, indices=[CU, PART], sizes=[96])
            attn_kv_block_mm(vb1, pc1, y, lst, ct)
        if PARTS_ == 2:
            air.ChannelGet("sPC", pc, indices=[CU, PART], sizes=[96])
            air.ChannelGet("sPC", pc1, indices=[CU, PART], sizes=[96])
            kv_part_out(y, lst, pc)
            air.ChannelPut("part", y, indices=[CU, PART], offsets=[0], sizes=[2112])
        else:
            o = air.alloc([2048], tl.bfloat16, memory_space=air.L1)
            attn_kv_fin_mm(o, y, lst, ct)
            air.ChannelPut("attnO", o, indices=[CU], offsets=[0], sizes=[2048])


@triton.jit
def merge_core(arm, CU: tl.constexpr):
    """A CU's attention output from its two parts' sums: blocked [64][4][8]."""
    ba = air.alloc([2112], tl.float32, memory_space=air.L1)
    bb = air.alloc([2112], tl.float32, memory_space=air.L1)
    o = air.alloc([2048], tl.bfloat16, memory_space=air.L1)
    lst = air.alloc([32], tl.float32, memory_space=air.L1)
    ct = air.alloc([32], tl.float32, memory_space=air.L1)
    cs = air.alloc([96], tl.float32, memory_space=air.L1)
    for _ in range((arm != 0).to(tl.int32)):
        air.ChannelGet("part", ba, indices=[CU, 0], sizes=[2112])
        air.ChannelGet("part", bb, indices=[CU, 1], sizes=[2112])
        attn_merge(o, ba, bb, lst, ct, cs)
        air.ChannelPut("attnO", o, indices=[CU], offsets=[0], sizes=[2048])


@triton.jit
def glu_core(arm):
    """gelu_tanh(gate) up of the up/gate projection's rows, 512 at a time."""
    gu = air.alloc([1024], tl.bfloat16, memory_space=air.L1)
    y = air.alloc([512], tl.bfloat16, memory_space=air.L1)
    for _ in range((arm != 0).to(tl.int32)):
        for s in range(GLU_SLICES_):
            air.ChannelGet("outY", gu, indices=[0, 2], sizes=[1024])
            glu(y, gu, 512)
            air.ChannelPut("gluOut", y, offsets=[0], sizes=[512])


@triton.jit
def pleproj_core(shp, arm):
    """per_layer_input of this layer, to plegate through shared L1 shp:
    [pli (256) | post-PLE norm w (D) | layer scale (32)]."""
    x = air.alloc([D_], tl.bfloat16, memory_space=air.L1)
    proj = air.alloc([256], tl.bfloat16, memory_space=air.L1)
    emb = air.alloc([256], tl.bfloat16, memory_space=air.L1)
    nw = air.alloc([256], tl.bfloat16, memory_space=air.L1)
    pf = air.alloc([D_ + 32], tl.bfloat16, memory_space=air.L1)
    pk = air.alloc([PK_], tl.bfloat16, memory_space=air.L1)
    w = air.alloc([8192], tl.bfloat16, memory_space=air.L1)
    acc = air.alloc([32], tl.float32, memory_space=air.L1)
    sq = air.alloc([64], tl.float32, memory_space=air.L1)
    for _ in range((arm != 0).to(tl.int32)):
        air.ChannelGet("pleX", x, sizes=[D_])
        for i in range(8):
            zero32(acc)
            for j in range(PLE_BLK_):
                air.ChannelGet("pleWp", w, sizes=[8192])
                mvm_blk(acc, w, x + j * 256)
            flush32(proj + i * 32, acc)
        air.ChannelGet("pleX", emb, sizes=[256])
        air.ChannelGet("pleX", nw, sizes=[256])
        air.ChannelGet("pleX", pf, sizes=[D_ + 32])
        scale_bf(proj, PLE_SCALE_, 256)
        rms_norm(proj, proj, nw, sq, 256, 1e-6)
        add_bf(proj, emb, 256)
        scale_bf(proj, 0.7071067811865476, 256)
        # one copy loop of the same shape as the reader's: AIR locks the shared
        # buffer around each access
        copy_n(pk, proj, 256)
        copy_n(pk + 256, pf, D_ + 32)
        copy_n(shp, pk, PK_)


@triton.jit
def plegate_core(shr, shp, arm):
    """The layer's output: residual (from shr) + rmsnorm(up(gelu(gate(res))
    pli)), times the layer scale."""
    P = air.alloc([D_ + PK_], tl.bfloat16, memory_space=air.L1)
    gate = air.alloc([256], tl.bfloat16, memory_space=air.L1)
    out = air.alloc([D_], tl.bfloat16, memory_space=air.L1)
    # each stream's blocks in halves, into two buffers: the DMA fills one
    # while the core reads the other
    wa0 = air.alloc([4096], tl.bfloat16, memory_space=air.L1)
    wa1 = air.alloc([4096], tl.bfloat16, memory_space=air.L1)
    wg0 = air.alloc([4096], tl.bfloat16, memory_space=air.L1)
    wg1 = air.alloc([4096], tl.bfloat16, memory_space=air.L1)
    acca = air.alloc([32], tl.float32, memory_space=air.L1)
    accb = air.alloc([32], tl.float32, memory_space=air.L1)
    sq = air.alloc([64], tl.float32, memory_space=air.L1)
    dec = (arm != 0).to(tl.int32)
    for _ in range(dec):
        copy_n(P, shr, D_)
        for i in range(4):
            zero32(acca)
            zero32(accb)
            for j in range(PLE_BLK_):
                air.ChannelGet("pleWa", wa0, sizes=[4096])
                mvm_half(acca, wa0, P + j * 256)
                air.ChannelGet("pleWa", wa1, sizes=[4096])
                mvm_half(acca, wa1, P + j * 256 + 128)
                air.ChannelGet("pleWg", wg0, sizes=[4096])
                mvm_half(accb, wg0, P + j * 256)
                air.ChannelGet("pleWg", wg1, sizes=[4096])
                mvm_half(accb, wg1, P + j * 256 + 128)
            flush32(gate + i * 32, acca)
            flush32(gate + (i + 4) * 32, accb)
        copy_n(P + D_, shp, PK_)
        gelu_inplace(gate, 256)
        mul_bf(P + D_, gate, 256)
        for i in range(D_ // 64):
            zero32(acca)
            air.ChannelGet("pleWa", wa0, sizes=[4096])
            mvm_half(acca, wa0, P + D_)
            air.ChannelGet("pleWa", wa1, sizes=[4096])
            mvm_half(acca, wa1, P + D_ + 128)
            flush32(out + i * 32, acca)
            zero32(acca)
            air.ChannelGet("pleWg", wg0, sizes=[4096])
            mvm_half(acca, wg0, P + D_)
            air.ChannelGet("pleWg", wg1, sizes=[4096])
            mvm_half(acca, wg1, P + D_ + 128)
            flush32(out + (i + D_ // 64) * 32, acca)
        rms_norm(out, out, P + D_ + 256, sq, D_, 1e-6)
        add_bf(out, P, D_)
        scale_bf(out, tl.load(P + 2 * D_ + 256), D_)
    # in every wave (the host's get is not in a branch): an LM wave's is not
    # read
    air.ChannelPut("pleOut", out, offsets=[0], sizes=[D_])


@triton.jit
def feed_weights(W, at, NF: tl.constexpr):
    """A phase's NF fills of each of the 8 weight streams, from fill at."""
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
    """N rounds' rows, gathered behind the first core's packet header."""
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
        m = air.alloc([514], tl.bfloat16, memory_space=air.L2, memtile_col=2)
        air.ChannelGet("toMain", m, indices=[0], offsets=[0], sizes=[130])
        for x in tl.static_range(1, 4):
            air.ChannelGet("toMain", m, indices=[x], offsets=[2 + x * 128], sizes=[128])
        air.ChannelPut("outY", m, indices=[0, 0], offsets=[0], sizes=[514])


@triton.jit
def wave(
    X,
    RMS,
    W,
    ROPE,
    KV,
    E,
    PX,
    WP,
    WA,
    WB,
    LOG,
    pos,
    NW: tl.constexpr,
    S: tl.constexpr,
    SM: tl.constexpr,
    EPS: tl.constexpr,
    MAXL: tl.constexpr,
    SWA: tl.constexpr,
    ARMS: tl.constexpr,
    SLABS: tl.constexpr,
    KVSRC: tl.constexpr,
    KVAPP: tl.constexpr,
    NDEC: tl.constexpr,
):
    """One token: NW waves, the NDEC decoder layers then the LM head's vocab
    chunks, all in one dispatch. A wave's arm (ARMS) says which kind of layer
    it is; see `geom.I2A`. Buffers (bf16 unless said):

    X     the hidden state [x | final x]; the host writes the token's
          embedding into x
    RMS   each layer's 4 norm weights, then the final norm's
    W     the Q4 weights, a slab a wave (SLABS: its first fill)
    ROPE  each layer's [cos | sin | q_norm | k_norm] at this position
    KV    a [K | V] slab, MAXL rows of KVW, a layer that keeps a cache
          (KVAPP: where a wave appends; KVSRC: which it reads), then a scratch
    E     the token's embedding, for the per-layer input
    PX    each layer's [per-layer embedding | PLE norm | post-PLE norm |
          layer scale]
    WP, WA, WB   the PLE projections, blocked as the PLE cores stream them
    LOG   the logits, a chunk an LM wave
    pos   the token's position
    S, SM the transform scripts of the cores and of the attention cores.
    """
    air.backend(
        use_lock_race_condition_fix_v2=True,
        merge_contiguous_feeds=False,
        stack_size=4096,
        allow_deadlock_prone_routing=True,
    )
    air.Channel("rmsX", channel_type="npu_dma_packet")
    # (shares the rms core's packet shim MM2S, freeing one for the KV's
    # second part, as air-FLM)
    air.Channel("rmsW", channel_type="npu_dma_packet")
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
    air.Channel("attnO", size=[2])
    air.Channel("ropeLUT")
    air.Channel("ropeQ")
    air.Channel("toAttnQ", size=[2, 1], broadcast_shape=[2, PARTS_])
    air.Channel("appendK")
    air.Channel("appendV")
    air.Channel("inKV_K")
    air.Channel("inKV_V")
    if PARTS_ == 2:
        air.Channel("inKV_K1")
        air.Channel("inKV_V1")
    if KV_REPL_:
        # each part's blocks, from memtiles of its own, to both CUs: E2B's
        # one kv head
        air.Channel("toK", size=[1, 1], broadcast_shape=[1, 2])
        air.Channel("toV", size=[1, 1], broadcast_shape=[1, 2])
    else:
        # a kv head a CU
        air.Channel("toK", size=[1, 2])
        air.Channel("toV", size=[1, 2])
    if PARTS_ == 2:
        air.Channel("toK1", size=[1, 1], broadcast_shape=[1, 2])
        air.Channel("toV1", size=[1, 1], broadcast_shape=[1, 2])
    air.Channel("sPC", size=[2, PARTS_])
    if PARTS_ == 2:
        air.Channel("part", size=[2, 2])
    air.Channel("gluOut")
    air.Channel("pleX")
    air.Channel("pleWp")
    air.Channel("pleWa")
    air.Channel("pleWgIn")
    air.Channel("pleWg")
    air.Channel("pleOut")
    air.Channel("layerOut")
    shr = air.alloc([D_], tl.bfloat16, memory_space=air.L1)
    shp = air.alloc([PK_], tl.bfloat16, memory_space=air.L1)
    # host
    # NDEC decoder layers, then LM-head waves (arm 0) of a vocab chunk each
    for wave in range(NW):
        arm = air.table(ARMS, wave)
        # Host branches hold only puts: AIR keeps the waits a branch's
        # transfers have on what comes before it, but nothing after it waits
        # on them, so every get is in all waves (and a loop of 0 trips would
        # not finish AIR's unrolling).
        lm = arm == 0
        air.ChannelPut("rmsX", X, offsets=[0], sizes=[D_])
        # an LM wave's rmsW: the final norm as the hi half; its rmsW2 unused
        air.ChannelPut(
            "rmsW",
            RMS,
            offsets=[tl.where(lm, NDEC, wave) * RMS_LAYER_],
            sizes=[2 * D_],
        )
        air.ChannelPut(
            "rmsW2",
            RMS,
            offsets=[tl.where(lm, 0, wave * RMS_LAYER_ + 2 * D_)],
            sizes=[2 * D_],
        )
        # the wave's weights, their lengths the arm's: a branch an arm, as
        # air-FLM's index_switch. Before the KV append: the host waits for
        # the append before it reads the KV back, and the append needs them.
        at = air.table(SLABS, wave)
        for a in tl.static_range(0, 8):
            if a % 2 == 1 or a == 0:
                if arm == a:
                    feed_arm(W, at, a)
        SL: tl.constexpr = MAXL * KVW_
        BL: tl.constexpr = 16 * KVW_  # a block of 16 rows
        # the KV's 16-row blocks up to pos (the attention cores', in the
        # memtile any count), the keys from pos + 1 on masked
        NB: tl.constexpr = MAXL // 16
        nb = (pos + 16) // 16
        # a sliding layer's from the window's first key's
        sw = tl.where(arm == 5, 1, tl.where(arm == 7, 1, 0))
        lb = tl.where(pos >= SWA, (pos - SWA + 1) // 16, 0) * sw
        # a part's share, of whole pairs (as `block_pairs`)
        nh = (nb - lb + 2 * PARTS_ - 1) // (2 * PARTS_) * 2
        if arm != 0:
            air.ChannelPut("ropeLUT", ROPE, offsets=[wave * 1536], sizes=[1536])
            air.ChannelPut("pleX", E, offsets=[0], sizes=[D_])
            air.ChannelPut("pleX", PX, offsets=[wave * PXW_], sizes=[256])
            air.ChannelPut("pleX", PX, offsets=[wave * PXW_ + 256], sizes=[256])
            air.ChannelPut("pleX", PX, offsets=[wave * PXW_ + 512], sizes=[D_ + 32])
            WPF: tl.constexpr = 8 * PLE_BLK_ * 8192
            WAF: tl.constexpr = PLE_FILLS_ * 8192
            air.ChannelPut("pleWp", WP, offsets=[wave * WPF], sizes=[WPF])
            air.ChannelPut("pleWa", WA, offsets=[wave * WAF], sizes=[WAF])
            air.ChannelPut("pleWgIn", WB, offsets=[wave * WAF], sizes=[WAF])
        # this token's k and v at row pos of the wave's KV slab [K | V], MAXL
        # x KVW each: a layer that keeps a cache its own, the others (KV-shared
        # layers, LM waves) a scratch slab
        ks = air.table(KVAPP, wave)
        air.ChannelGet("appendK", KV, offsets=[ks * 2 * SL + pos * KVW_], sizes=[KVW_])
        air.ChannelGet(
            "appendV", KV, offsets=[ks * 2 * SL + SL + pos * KVW_], sizes=[KVW_]
        )
        # then the blocks attention reads of the layer whose KV it
        # reads: its own, or for a KV-shared layer an earlier one's (its own
        # append then goes to a slab nothing reads)
        if arm != 0:
            src = air.table(KVSRC, wave)
            k0 = src * 2 * SL + lb * BL
            k1 = k0 + nh * BL  # part 1's
            air.ChannelPut("inKV_K", KV, offsets=[k0], sizes=[nh * BL])
            air.ChannelPut("inKV_V", KV, offsets=[k0 + SL], sizes=[nh * BL])
            if PARTS_ == 2:
                air.ChannelPut("inKV_K1", KV, offsets=[k1], sizes=[nh * BL])
                air.ChannelPut("inKV_V1", KV, offsets=[k1 + SL], sizes=[nh * BL])
        for n in range(PLE_FILLS_):
            wbt = air.alloc([8192], tl.bfloat16, memory_space=air.L2, memtile_col=5)
            air.ChannelGet("pleWgIn", wbt, offsets=[0], sizes=[8192])
            # in halves, as plegate takes them
            for h in tl.static_range(2):
                air.ChannelPut("pleWg", wbt, offsets=[h * 4096], sizes=[4096])
        # x of every round: 512 at a time into memtile 3, broadcast as 2 x 256.
        # The memtiles' relays run without counts, so these loops' bounds also
        # cover an LM wave's rounds, and relays an LM wave does not use stay
        # idle: no branches here.
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
        # q, half to each attention CU
        rq = air.alloc([4096], tl.bfloat16, memory_space=air.L2, memtile_col=5)
        air.ChannelGet("ropeQ", rq, offsets=[0], sizes=[4096])
        # each CU's 4 heads blocked [64][4][8] (8 columns of the 4), to
        # both parts
        for cu in tl.static_range(2):
            air.ChannelPut(
                "toAttnQ",
                rq,
                indices=[cu, 0],
                offsets=[0, cu * 4, 0],
                sizes=[64, 4, 8],
                strides=[8, 512, 1],
            )
        # each part's KV blocks, blocked [2][64][8][8] (8 keys by 8
        # columns): part 0's in memtile 4, part 1's K in memtile 2 and V in
        # memtile 5 (air-FLM's). E2B's one kv head goes to both CUs; on E4B
        # a block row holds a head a CU.
        for b in range(NB // PARTS_):
            t = air.alloc([BL], tl.bfloat16, memory_space=air.L2, memtile_col=4)
            air.ChannelGet("inKV_K", t, offsets=[0], sizes=[BL])
            for cu in tl.static_range(1 if KV_REPL_ else 2):
                air.ChannelPut(
                    "toK",
                    t,
                    indices=[0, cu],
                    offsets=[0, 0, 0, cu * 512],
                    sizes=[2, 64, 8, 8],
                    strides=[8 * KVW_, 8, KVW_, 1],
                )
        for b in range(NB // PARTS_):
            t = air.alloc([BL], tl.bfloat16, memory_space=air.L2, memtile_col=4)
            air.ChannelGet("inKV_V", t, offsets=[0], sizes=[BL])
            for cu in tl.static_range(1 if KV_REPL_ else 2):
                air.ChannelPut(
                    "toV",
                    t,
                    indices=[0, cu],
                    offsets=[0, 0, 0, cu * 512],
                    sizes=[2, 64, 8, 8],
                    strides=[8 * KVW_, 8, KVW_, 1],
                )
        for b in range(NB // 2 * (PARTS_ - 1)):
            t = air.alloc([8192], tl.bfloat16, memory_space=air.L2, memtile_col=2)
            air.ChannelGet("inKV_K1", t, offsets=[0], sizes=[8192])
            air.ChannelPut(
                "toK1",
                t,
                indices=[0, 0],
                offsets=[0, 0, 0, 0],
                sizes=[2, 64, 8, 8],
                strides=[4096, 8, 512, 1],
            )
        for b in range(NB // 2 * (PARTS_ - 1)):
            t = air.alloc([8192], tl.bfloat16, memory_space=air.L2, memtile_col=5)
            air.ChannelGet("inKV_V1", t, offsets=[0], sizes=[8192])
            air.ChannelPut(
                "toV1",
                t,
                indices=[0, 0],
                offsets=[0, 0, 0, 0],
                sizes=[2, 64, 8, 8],
                strides=[4096, 8, 512, 1],
            )
        # the attention output, as o-proj's x a round each
        og = air.alloc([DQ_], tl.bfloat16, memory_space=air.L2, memtile_col=5)
        for cu in tl.static_range(2):
            # a CU's 4 heads, blocked [64][4][8], as [half][head][256]
            air.ChannelGet(
                "attnO",
                og,
                indices=[cu],
                offsets=[cu * 2, 0, 0, 0],
                sizes=[2, 32, 4, 8],
                strides=[1024, 8, 256, 1],
            )
        for n in range(I2P_[1]):
            air.ChannelPut("xnorm", og, offsets=[0], sizes=[DQ_])
        # glu's output, as down's x a round each
        dx = air.alloc([INTER_], tl.bfloat16, memory_space=air.L2, memtile_col=3)
        for s in range(GLU_SLICES_):
            air.ChannelGet("gluOut", dx, offsets=[s * 512], sizes=[512])
        for n in range(I2P_[3]):
            air.ChannelPut("xnorm", dx, offsets=[0], sizes=[INTER_])
        air.herd(proj_core, (arm, 0), (2, 4), S, name="proj_blk0", at=(0, 2))
        air.herd(proj_core, (arm, 2), (2, 4), S, name="proj_blk1", at=(6, 2))
        # the rounds' rows, gathered behind the first core's packet header
        gather_rounds(NR_)
        air.herd(rope_core, (arm,), 1, S, name="rope", at=(2, 3))
        # attention: per CU a qk and a kv core; on E2B two of each, for two
        # parts of the context, and a merge
        air.herd(qk_core, (arm, pos, 0, 0, SWA), 1, SM, name="qk0", at=(4, 2))
        air.herd(kv_core, (arm, pos, 0, 0, SWA), 1, SM, name="kv0", at=(4, 3))
        air.herd(qk_core, (arm, pos, 1, 0, SWA), 1, SM, name="qk1", at=(4, 4))
        air.herd(kv_core, (arm, pos, 1, 0, SWA), 1, SM, name="kv1", at=(4, 5))
        if PARTS_ == 2:
            air.herd(qk_core, (arm, pos, 0, 1, SWA), 1, SM, name="qk0b", at=(2, 4))
            air.herd(kv_core, (arm, pos, 0, 1, SWA), 1, SM, name="kv0b", at=(2, 5))
            air.herd(qk_core, (arm, pos, 1, 1, SWA), 1, SM, name="qk1b", at=(3, 4))
            air.herd(kv_core, (arm, pos, 1, 1, SWA), 1, SM, name="kv1b", at=(3, 5))
            air.herd(merge_core, (arm, 0), 1, SM, name="merge0", at=(5, 4))
            air.herd(merge_core, (arm, 1), 1, SM, name="merge1", at=(5, 5))
        air.herd(rms_core, (shr, arm, EPS), 1, S, name="rms", at=(2, 2))
        air.herd(pleproj_core, (shp, arm), 1, S, name="pleproj", at=(3, 3))
        air.herd(plegate_core, (shr, shp, arm), 1, S, name="plegate", at=(3, 2))
        air.herd(glu_core, (arm,), 1, S, name="glu", at=(5, 3))
        # the logits, a decode wave's to the chunk after the LM waves'; the
        # layer output over its input, an LM wave's (nothing) after it
        lc = tl.where(arm == 0, wave - NDEC, NW - NDEC)
        air.ChannelGet(
            "layerOut",
            LOG,
            offsets=[lc * LM_ROWS_],
            sizes=[tl.where(arm == 0, LM_ROWS_, D_)],
        )
        xo = tl.where(arm == 0, D_, 0)
        air.ChannelGet("pleOut", X, offsets=[xo], sizes=[D_])
