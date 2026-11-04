# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Gemma4's vision attention for the vision device: bidirectional flash
attention over an image's patches, every head its own K and V.

`gemma4_attention`'s round loop with one change, the mask: a query sees every
key that is a real patch -- the first `nvalid` -- whatever its position, where
the text's sees the keys at or before it. The buffers are that module's: q and
o as 8x8 blocks [rounds * R / 8, DH / 8], each head's KV as records of a K
tile's blocks [LKP / 8, DH / 8] then its V tile's in DH / DVT column chunks
[LKP / 8, DVT / 8], one head's records SPAN elements after the previous
head's. The heads split over two round loops, one per column, each a herd
of NQT cores, and the schedule is `gemma4_attention`'s template rendered for
this shape.

Numerics as `gemma4_attention`'s: S in bf16, the row maximum and the mask on
bf16 values, probabilities in bf16, the accumulator in bf16 rescaled and added
to P V in f32 each block.
"""

from pathlib import Path

import numpy as np
import triton
import triton.language as tl
from ml_dtypes import bfloat16

import gemma4_attention as A
import gemma4_vision_q4nx as vw

NQT = A.NQT
#: The K block, and a core's q tile: as mlir-air's vision FlashAttention.
LKP = 64
DH = vw.HEAD_DIM
DVT = A.dvt(DH)
#: q rows of a round: one head's, over a column's cores.
R = NQT.value * LKP
COLUMNS = 6
#: Round loops: one per column.
LOOPS = COLUMNS
HEADS_PER_LOOP = vw.N_HEADS // LOOPS
#: Records' bf16 elements per K block: K's then V's.
REC = 2 * LKP * DH
HEADS = vw.N_HEADS
#: The score scale, built in. mlir-air's encoder widens each head to 2 DH
#: columns, scales q on the host by the model's scale times sqrt(2 DH) and
#: has its kernel take 1 / sqrt(2 DH): the same product here.
SCALE = 1.0 / (2 * DH) ** 0.5


@triton.jit
def _rounds_bidir(
    Q,
    KV,
    O,
    nk,
    nvalid,
    nr,
    LKP: tl.constexpr,
    DH: tl.constexpr,
    DVT: tl.constexpr,
    C: tl.constexpr,
    KVH: tl.constexpr,
    SPAN: tl.constexpr,
    S: tl.constexpr,
    KBM: tl.constexpr,
    VBM: tl.constexpr,
):
    """nr rounds of one column, `gemma4_attention._rounds` but bidirectional:
    key column j of block blk is kept when (blk LKP + j) < nvalid. KVH heads,
    each nr // KVH consecutive rounds, its records SPAN elements apart."""
    R: tl.constexpr = NQT * LKP
    REC: tl.constexpr = 2 * LKP * DH
    DVC: tl.constexpr = DH // DVT
    R8: tl.constexpr = R // 8
    MB: tl.constexpr = LKP // 8
    KB: tl.constexpr = DH // 8
    VB: tl.constexpr = DVT // 8
    oi = tl.arange(0, LKP)
    orr = tl.arange(0, R)
    qoff = A._offsets(R8 * KB // (S * KBM), S, KBM)
    koff = A._offsets(MB * KB // (S * KBM), S, KBM)
    voff = tl.arange(0, DVC)[:, None, None, None, None, None] * (
        MB * VB * 64
    ) + A._offsets(MB * VB // (S * VBM), S, VBM).expand_dims(0)
    # Minus each key's column within its block, on every q row, so that a
    # block's mask compares it with how many of its keys are real. Spelled as
    # the q row minus the key column, minus the q row again: the schedule
    # stages a mask of that shape, [R, LKP] from the two ranges, and not a
    # [1, LKP] one broadcast over the rows.
    dkey = (orr[:, None] - oi[None, :]) - orr[:, None]
    for r in range(nr):
        q = tl.reshape(tl.load(Q + r * R * DH + qoff), (R8, KB, 8, 8))
        q = tl.reshape(tl.permute(q, (0, 2, 1, 3)), (R, DH))
        kv = KV + (r // (nr // KVH)) * SPAN
        m = tl.full((R,), -1.0e30, tl.bfloat16)
        l = tl.zeros((R,), tl.float32)
        acc = tl.zeros((R, DH), tl.bfloat16)
        for blk in range(nk):
            base = kv + blk * REC
            k = tl.reshape(tl.load(base + koff), (MB, KB, 8, 8))
            k = tl.reshape(tl.permute(k, (0, 2, 1, 3)), (LKP, DH))
            v = tl.reshape(tl.load(base + LKP * DH + voff), (DVC, MB, VB, 8, 8))
            v = tl.reshape(tl.permute(v, (1, 3, 0, 2, 4)), (LKP, DH))
            s = tl.dot(q, tl.trans(k)).to(tl.bfloat16)
            # key column blk LKP + j is a real patch when it is below nvalid
            d = dkey + (nvalid - 1 - blk * LKP)
            s = tl.where(d >= 0, s, float("-inf"))
            rm = tl.reduce(s, 1, A._maximum)
            mn = tl.where(m > rm, m, rm)
            a = tl.exp2(C * (m.to(tl.float32) - mn.to(tl.float32)))
            p = tl.exp2(C * (s.to(tl.float32) - mn[:, None].to(tl.float32))).to(
                tl.bfloat16
            )
            l = l * a + tl.sum(p.to(tl.float32), axis=1)
            acc = (acc.to(tl.float32) * a[:, None] + tl.dot(p, v)).to(tl.bfloat16)
            m = mn
        out = acc.to(tl.float32) * (1.0 / l)[:, None]
        out = tl.permute(tl.reshape(out.to(tl.bfloat16), (R8, 8, KB, 8)), (0, 2, 1, 3))
        out = tl.reshape(out, (R8 * KB // (S * KBM), S, KBM, 8, 8))
        tl.store(O + r * R * DH + qoff, out)


@triton.jit(do_not_specialize=["nk", "nvalid", "nr"])
def vattn(
    Q,
    KV,
    O,
    nk,
    nvalid,
    nr,
    LKP: tl.constexpr,
    DH: tl.constexpr,
    DVT: tl.constexpr,
    C: tl.constexpr,
    KVH: tl.constexpr,
    SPAN: tl.constexpr,
    QC: tl.constexpr,
    KC: tl.constexpr,
    S: tl.constexpr,
    KBM: tl.constexpr,
    VBM: tl.constexpr,
):
    """The heads of an image in one loop per column: each runs nr rounds of
    its KVH heads against nk K blocks, of which the first nvalid keys are
    real. A column's q and o are QC elements after the previous column's, its
    KV records KC."""
    _rounds_bidir(
        Q + 0 * QC, KV + 0 * KC, O + 0 * QC, nk, nvalid, nr,
        LKP, DH, DVT, C, KVH, SPAN, S, KBM, VBM,
    )  # fmt: skip
    _rounds_bidir(
        Q + 1 * QC, KV + 1 * KC, O + 1 * QC, nk, nvalid, nr,
        LKP, DH, DVT, C, KVH, SPAN, S, KBM, VBM,
    )  # fmt: skip
    _rounds_bidir(
        Q + 2 * QC, KV + 2 * KC, O + 2 * QC, nk, nvalid, nr,
        LKP, DH, DVT, C, KVH, SPAN, S, KBM, VBM,
    )  # fmt: skip
    _rounds_bidir(
        Q + 3 * QC, KV + 3 * KC, O + 3 * QC, nk, nvalid, nr,
        LKP, DH, DVT, C, KVH, SPAN, S, KBM, VBM,
    )  # fmt: skip
    _rounds_bidir(
        Q + 4 * QC, KV + 4 * KC, O + 4 * QC, nk, nvalid, nr,
        LKP, DH, DVT, C, KVH, SPAN, S, KBM, VBM,
    )  # fmt: skip
    _rounds_bidir(
        Q + 5 * QC, KV + 5 * KC, O + 5 * QC, nk, nvalid, nr,
        LKP, DH, DVT, C, KVH, SPAN, S, KBM, VBM,
    )  # fmt: skip


def args(q, kv, o, nk, nvalid, nr, rows=None, max_blocks=None):
    """`vattn`'s arguments on buffers `q`, `kv` and `o`, which hold every
    column's part: per column its heads' q (and o) of `rows` rows each, and
    their KV records, `max_blocks` K blocks a head; nr rounds per column;
    scores scaled by SCALE."""
    rows = rows or ROWS
    max_blocks = max_blocks or MAX_BLOCKS
    return (
        q,
        kv,
        o,
        nk,
        nvalid,
        nr,
        LKP,
        DH,
        DVT,
        float(A.LOG2E * SCALE),
        HEADS_PER_LOOP,
        max_blocks * REC,
        HEADS_PER_LOOP * rows * DH,
        HEADS_PER_LOOP * max_blocks * REC,
        1,
        DH // 8,
        DVT // 8,
    )


def schedule(out_dir):
    """The transform script for `vattn`, written to `out_dir`."""
    t = (LKP // 8, 1, DH // 8, DVT // 8)
    nests = [
        (f"_{n}{c}", f"_{n}", str(c), LKP, DH, DVT, t)
        for c in range(COLUMNS)
        for n in ("a",)
    ]
    path = Path(out_dir) / f"transform_vision_attention_{DH}.mlir"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path.write_text(A.render(nests))
    return path


# ---- host layouts


def blocks(x):
    """[..., rows, DH] -> its 8x8 blocks, row-block major:
    [..., rows / 8, DH / 8, 8, 8]."""
    *b, rows, _ = x.shape
    return x.reshape(*b, rows // 8, 8, DH // 8, 8).swapaxes(-3, -2)


def unblocks(b, rows):
    """`blocks`' inverse: [..., rows * DH] -> [..., rows, DH]."""
    lead = b.shape[:-1]
    b = b.reshape(*lead, rows // 8, DH // 8, 8, 8).swapaxes(-3, -2)
    return b.reshape(*lead, rows, DH)


def records(k, v):
    """K and V [..., n, DH] (n a multiple of LKP) -> their records
    [..., n / LKP * REC]: per block K's 8x8 blocks, then V's in DH / DVT
    column chunks of [LKP / 8, DVT / 8] blocks."""
    *lead, n, _ = k.shape
    nb, mb = n // LKP, LKP // 8
    kb = k.reshape(*lead, nb, mb, 8, DH // 8, 8).swapaxes(-3, -2)
    vb = v.reshape(*lead, nb, mb, 8, DH // DVT, DVT // 8, 8)
    # [..., nb, DVC, MB, VB, 8, 8]
    n0 = len(lead)
    vb = vb.transpose(*range(n0), *(n0 + i for i in (0, 3, 1, 4, 2, 5)))
    rec = np.concatenate(
        [kb.reshape(*lead, nb, -1), vb.reshape(*lead, nb, -1)], axis=-1
    )
    return rec.reshape(*lead, nb * REC).astype(bfloat16)


# ---- a device of its own, for the test


def build(out):
    """`vattn` alone, as device `vattn` in `out`: MAX_BLOCKS K blocks a head
    at most and the rounds of ROWS q rows, nk and nvalid at run time. The
    round count is built in: a run-time one keeps the round loop rolled,
    where every transfer it frees without waiting gives its buffer
    descriptor back while the transfer may still be queued."""
    import torch
    from triton.backends.amd_triton_npu.config import config_context
    from triton.backends.amd_triton_npu.fused_device import FusedDevice

    import npu_device_cache as cache

    out = Path(out)
    dev = FusedDevice("vattn", str(out / "proj"))
    q = torch.zeros(HEADS * ROWS * DH, dtype=torch.bfloat16)
    kv = torch.zeros(HEADS * MAX_BLOCKS * REC, dtype=torch.bfloat16)
    with config_context(bf16_emulation=True):
        dev.add(
            "vattn",
            vattn,
            (1,),
            args(q, kv, q.clone(), 1, 1, NR),
            transform_script=str(schedule(dev.proj)),
            runtime=("nk", "nvalid"),
            key="attn",
            pin_l2=True,
            multiplex=("L2",),
        )
    with cache.toolchain_env():
        dev.compile(
            runtime_loop_tiling_sizes=(),
            bf16_emulation=True,
            extra_args=AIRCC_ARGS,
            stack_size=4096,
        )
    dev.save(out)


#: What aircc builds the device with, beyond the defaults: the vision
#: device's. Without `--fuse-sequential-loops` a core's input channel is fed
#: from two memtile channels.
AIRCC_ARGS = (
    "--air-channel-multiplexing=L1",
    "--omit-ping-pong-transform=L1",
    "--fuse-sequential-loops",
    "--use-lock-race-condition-fix-auto=false",
)

#: The device's shape: up to MAX_BLOCKS K blocks a head, ROWS q rows.
MAX_BLOCKS = 40
ROWS = 256
#: Rounds a column runs: its heads' ROWS q rows, R at a time.
NR = HEADS_PER_LOOP * (ROWS // R)

NAME = "vattn"
FILES = tuple(f"{NAME}.{x}" for x in ("xclbin", "txn.h", "ops.json"))
_SOURCES = (A.SCHEDULE, Path(A.__file__), Path(__file__))


def device(cache_root=None):
    """Build (or reuse) the device. Returns `build_dir` and `cached`."""
    import npu_device_cache as cache

    digest = cache.fingerprint(_SOURCES, [AIRCC_ARGS, MAX_BLOCKS, ROWS])
    return cache.cached("gemma4-vision-attention", digest, FILES, build, cache_root)


class Attention:
    """mlir-air `VisionEncoder`'s attention on the `vattn` device:
    `attention(q, k, v, scale)` over its widened heads.

    q [Lq, H 2DH] and k [Lk, H 2DH] carry each head's DH columns and then
    the encoder's mask columns: column DH is 1 in q and, in k, 0 for a real
    key and -big for a padding one. Here the device masks by key count, so
    only the first DH columns move, and the real keys are the rows whose
    mask column is 0. The q rows go ROWS at a time, every head against the
    whole of its K and V."""

    def __init__(self, prog):
        self.prog = prog
        self.q = prog.empty(HEADS * ROWS * DH, bfloat16)
        self.kv = prog.empty(HEADS * MAX_BLOCKS * REC, bfloat16)
        self.o = prog.empty(HEADS * ROWS * DH, bfloat16)

    def attention(self, q, k, v, scale):
        heads = HEADS
        if abs(scale - SCALE) > 1e-6 * SCALE:
            raise ValueError(f"the device scales scores by {SCALE}, not {scale}")
        lq, lk = q.shape[0], k.shape[0]
        wide = q.shape[1] // heads
        q = np.asarray(q, bfloat16).reshape(lq, heads, wide)[:, :, :DH]
        k = np.asarray(k, bfloat16).reshape(lk, heads, wide)
        nvalid = int(np.count_nonzero(k[:, 0, DH] == 0))
        k = k[:, :, :DH]
        v = np.asarray(v, bfloat16).reshape(lk, heads, DH)
        # Whole K blocks, in pairs: the cores take them two at a time, and the
        # extra one is masked.
        nk = -(-lk // LKP)
        nk += nk % 2
        if nk > MAX_BLOCKS:
            raise ValueError(f"{lk} keys; the device holds {MAX_BLOCKS * LKP}")
        if lk % LKP or lq % 8:
            raise ValueError(f"{lk} keys or {lq} queries: not whole blocks")
        nb = lk // LKP
        # Straight into the device buffers' blocked views: one copy each way.
        kv = self.kv.host(write=True).reshape(heads, MAX_BLOCKS, REC)
        kv[:, nb:nk] = 0
        kv = kv[:, :nb]
        mb, kb, vb, dvc = LKP // 8, DH // 8, DVT // 8, DH // DVT
        # K [nb LKP, H, DH] -> [H, nb, MB, KB, 8, 8]
        kv[..., : LKP * DH].reshape(heads, nb, mb, kb, 8, 8)[...] = k.reshape(
            nb, mb, 8, heads, kb, 8
        ).transpose(3, 0, 1, 4, 2, 5)
        # V -> [H, nb, DVC, MB, VB, 8, 8]
        kv[..., LKP * DH :].reshape(heads, nb, dvc, mb, vb, 8, 8)[...] = v.reshape(
            nb, mb, 8, heads, dvc, vb, 8
        ).transpose(3, 0, 4, 1, 5, 2, 6)
        out = np.empty((lq, heads, DH), bfloat16)
        r8 = ROWS // 8
        for r0 in range(0, lq, ROWS):
            rows = min(ROWS, lq - r0)
            # q [rows, H, DH] -> [H, ROWS / 8, KB, 8, 8], the rows past the
            # image zero
            qv = self.q.host(write=True).reshape(heads, r8, kb, 8, 8)
            qv[:, rows // 8 :] = 0
            qv[:, : rows // 8] = (
                q[r0 : r0 + rows]
                .reshape(rows // 8, 8, heads, kb, 8)
                .transpose(2, 0, 3, 1, 4)
            )
            with self.prog:
                vattn[(1,)](*args(self.q, self.kv, self.o, nk, nvalid, NR))
            ov = self.o.host().reshape(heads, r8, kb, 8, 8)[:, : rows // 8]
            out[r0 : r0 + rows].reshape(rows // 8, 8, heads, kb, 8)[...] = ov.transpose(
                1, 3, 0, 2, 4
            )
        return out.reshape(lq, heads * DH)
