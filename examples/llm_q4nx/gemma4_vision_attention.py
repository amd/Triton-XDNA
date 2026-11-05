# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Gemma4's vision attention for the vision device: bidirectional flash
attention over an image's patches, every head its own K and V.

`gemma4_attention`'s round loop with a different mask: a query sees every
real patch (the first `nvalid` keys) regardless of position, instead of only
the keys at or before it. The buffers are that module's: q and
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
    # Negated key column within the block, on every q row. Written as
    # (row - col) - row so the mask is [R, LKP] built from both ranges, the
    # shape the schedule stages, rather than a [1, LKP] broadcast.
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


# ---- the attention device


def build(out):
    """`vattn` alone, as device `vattn` in `out`: at most MAX_BLOCKS K blocks
    a head, ROWS q rows, nk and nvalid at run time. The round count is a
    compile-time constant because a run-time count keeps the round loop
    rolled, and the rolled loop can release a buffer descriptor while its
    transfer is still queued."""
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


#: aircc options, as for the vision GEMM device. `--fuse-sequential-loops`
#: keeps each core's input channel fed from a single memtile channel.
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
        self.kv = prog.empty(HEADS * MAX_BLOCKS * REC, bfloat16)
        self.qs, self.os = [], []  # one Q and one O buffer per dispatch

    def attention(self, q, k, v, scale):
        heads = HEADS
        if abs(scale - SCALE) > 1e-6 * SCALE:
            raise ValueError(f"the device scales scores by {SCALE}, not {scale}")
        lq, lk = q.shape[0], k.shape[0]
        wide = q.shape[1] // heads
        nvalid = int(np.count_nonzero(np.asarray(k, bfloat16)[:, DH] == 0))
        self.prepare(lq, lk, nvalid)
        self.put_kv(k, v, 0)
        self.put_q(q, 0)
        self.run()
        return self.result()

    # ---- in pieces, so that the host's share can overlap the device's:
    # `prepare`, then `put_kv` and `put_q` row block by row block as the
    # rows are ready, `run`, then `result` (or `rows`).

    def prepare(self, lq, lk, nvalid):
        """For `lq` queries against `lk` keys, the first `nvalid` real."""
        # Whole K blocks, in pairs: the cores take them two at a time, and the
        # extra one is masked.
        nk = -(-lk // LKP)
        nk += nk % 2
        if nk > MAX_BLOCKS:
            raise ValueError(f"{lk} keys; the device holds {MAX_BLOCKS * LKP}")
        if lk % LKP or lq % 8:
            raise ValueError(f"{lk} keys or {lq} queries: not whole blocks")
        self.lq, self.nk, self.nvalid = lq, nk, nvalid
        kv = self.kv.host(write=True).reshape(HEADS, MAX_BLOCKS, REC)
        kv[:, lk // LKP : nk] = 0
        n = -(-lq // ROWS)
        while len(self.qs) < n:
            self.qs.append(self.prog.zeros(HEADS * ROWS * DH, bfloat16))
            self.os.append(self.prog.empty(HEADS * ROWS * DH, bfloat16))
        if lq % ROWS:
            last = self.qs[n - 1].host(write=True)
            last.reshape(HEADS, ROWS // 8, -1)[:, lq % ROWS // 8 :] = 0

    def put_kv(self, k, v, r0):
        """Key rows `r0..` (a whole number of K blocks): k [n, H wide] with the
        mask columns, v [n, H DH]."""
        heads, n = HEADS, k.shape[0]
        if r0 % LKP or n % LKP:
            raise ValueError(f"keys {r0}..{r0 + n}: not whole blocks")
        k = np.asarray(k, bfloat16).reshape(n, heads, -1)[:, :, :DH]
        v = np.asarray(v, bfloat16).reshape(n, heads, DH)
        b0, nb = r0 // LKP, n // LKP
        kv = self.kv.host(write=True).reshape(heads, MAX_BLOCKS, REC)[:, b0 : b0 + nb]
        mb, kb, vb, dvc = LKP // 8, DH // 8, DVT // 8, DH // DVT
        # K [nb LKP, H, DH] -> [H, nb, MB, KB, 8, 8]
        kv[..., : LKP * DH].reshape(heads, nb, mb, kb, 8, 8)[...] = k.reshape(
            nb, mb, 8, heads, kb, 8
        ).transpose(3, 0, 1, 4, 2, 5)
        # V -> [H, nb, DVC, MB, VB, 8, 8]
        kv[..., LKP * DH :].reshape(heads, nb, dvc, mb, vb, 8, 8)[...] = v.reshape(
            nb, mb, 8, heads, dvc, vb, 8
        ).transpose(3, 0, 4, 1, 5, 2, 6)

    def put_q(self, q, r0):
        """Query rows `r0..`, q [n, H wide]; `r0` and n multiples of 8. The
        rows past `lq` in the last dispatch stay zero."""
        heads, kb = HEADS, DH // 8
        q = np.asarray(q, bfloat16).reshape(q.shape[0], heads, -1)[:, :, :DH]
        while len(q):
            d, off = divmod(r0, ROWS)
            n = min(len(q), ROWS - off)
            # [n, H, DH] -> [H, ROWS / 8, KB, 8, 8] at row block off / 8
            qv = self.qs[d].host(write=True).reshape(heads, ROWS // 8, kb, 8, 8)
            qv[:, off // 8 : (off + n) // 8] = (
                q[:n].reshape(n // 8, 8, heads, kb, 8).transpose(2, 0, 3, 1, 4)
            )
            q, r0 = q[n:], r0 + n

    def run(self):
        """Start every dispatch."""
        n = -(-self.lq // ROWS)
        with self.prog:
            for qb, ob in zip(self.qs[:n], self.os[:n]):
                vattn[(1,)](*args(qb, self.kv, ob, self.nk, self.nvalid, NR))

    def rows(self, d):
        """Dispatch `d`'s output rows, [rows, H DH], once it is done."""
        heads, kb = HEADS, DH // 8
        n = min(ROWS, self.lq - d * ROWS)
        ov = self.os[d].host().reshape(heads, ROWS // 8, kb, 8, 8)[:, : n // 8]
        return ov.transpose(1, 3, 0, 2, 4).reshape(n, heads * DH)

    def result(self):
        """The whole output, [lq, H DH]: each dispatch's rows taken while the
        later ones run."""
        out = np.empty((self.lq, HEADS * DH), bfloat16)
        for d in range(-(-self.lq // ROWS)):
            out[d * ROWS : d * ROWS + ROWS] = self.rows(d)
        return out
