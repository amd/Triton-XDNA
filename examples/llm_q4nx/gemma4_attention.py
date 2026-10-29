# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Gemma4 prefill attention for the fused prefill device.

Flash attention over one chunk's queries, round by round (a round is one
head's R = NQT * LKP q rows), against nk K blocks of the KV buffer. The
buffers are those of mlir-air's runtime with the manifest's "attn_format"
"bf16", all in 8x8 blocks, each row-major: Q and O [rounds * R, DH] as blocks
[rounds * R / 8, DH / 8], and KV as records of a K tile's blocks
[LKP / 8, DH / 8] followed by its V tile's in "attn_v_chunks" column chunks
of DVT, each [LKP / 8, DVT / 8]. nk, q0 and k0 (the chunk's first q block and
first K block, in LKP rows) are the op's run-time arguments.

Numerics: S in bf16, the row maximum and the masks on bf16 values, the
probabilities in bf16, and the output accumulator kept in bf16, rescaled and
added to P V in f32 each block.
"""

import re
from pathlib import Path

import triton
import triton.language as tl

LOG2E = 1.4426950408889634
NQT = tl.constexpr(4)  # cores of an attention column, LKP q rows each
SCHEDULE = Path(__file__).resolve().parent / "transform_gemma4_attention_aie2p.mlir.in"


@triton.jit
def _maximum(a, b):
    return tl.where(a > b, a, b)


@triton.jit
def _offsets(A: tl.constexpr, B: tl.constexpr, C: tl.constexpr):
    """Element offsets of [A, B, C] 8x8 blocks stored in order."""
    o8 = tl.arange(0, 8)
    return (
        tl.arange(0, A)[:, None, None, None, None] * (B * C * 64)
        + tl.arange(0, B)[None, :, None, None, None] * (C * 64)
        + tl.arange(0, C)[None, None, :, None, None] * 64
        + o8[None, None, None, :, None] * 8
        + o8[None, None, None, None, :]
    )


@triton.jit
def _rounds(
    Q,
    KV,
    O,
    nk,
    q0,
    k0,
    nr,
    LKP: tl.constexpr,
    DH: tl.constexpr,
    DVT: tl.constexpr,
    QR: tl.constexpr,
    WB: tl.constexpr,
    C: tl.constexpr,
    KVH: tl.constexpr,
    SPAN: tl.constexpr,
    S: tl.constexpr,
    KBM: tl.constexpr,
    VBM: tl.constexpr,
):
    """nr rounds of one group; see `attn`. q, K and o move as [., S, KBM]
    8x8 blocks and V's chunks as [., S, VBM], so both groups of a column use
    the same transfers. The narrower group's rows of blocks are KBM (VBM)
    blocks wide and S of them go together; the wider group's rows are S
    times that wide."""
    R: tl.constexpr = NQT * LKP
    REC: tl.constexpr = 2 * LKP * DH
    DVC: tl.constexpr = DH // DVT
    R8: tl.constexpr = R // 8
    MB: tl.constexpr = LKP // 8
    KB: tl.constexpr = DH // 8
    VB: tl.constexpr = DVT // 8
    oi = tl.arange(0, LKP)
    orr = tl.arange(0, R)
    # 8x8 blocks, row-block major: q and o [R8, KB], K [MB, KB], and V in
    # DVC column chunks [MB, VB]
    qoff = _offsets(R8 * KB // (S * KBM), S, KBM)
    koff = _offsets(MB * KB // (S * KBM), S, KBM)
    voff = tl.arange(0, DVC)[:, None, None, None, None, None] * (
        MB * VB * 64
    ) + _offsets(MB * VB // (S * VBM), S, VBM).expand_dims(0)
    # q row minus K column within a block; a block's masks compare it with
    # the block's offset
    dij = orr[:, None] - oi[None, :]
    # Rounds loop here rather than over programs, so the host sets up the
    # transfers once per dispatch.
    for r in range(nr):
        q = tl.reshape(tl.load(Q + r * R * DH + qoff), (R8, KB, 8, 8))
        q = tl.reshape(tl.permute(q, (0, 2, 1, 3)), (R, DH))
        kv = KV
        if KVH > 1:
            kv = KV + (r // (nr // KVH)) * SPAN
        # the round's first q row
        qr0 = (q0 + (r % QR) * NQT) * LKP
        # The row maxima are of bf16 scores, so m is bf16: the cores compare
        # in bf16.
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
        out = tl.permute(tl.reshape(out.to(tl.bfloat16), (R8, 8, KB, 8)), (0, 2, 1, 3))
        out = tl.reshape(out, (R8 * KB // (S * KBM), S, KBM, 8, 8))
        tl.store(O + r * R * DH + qoff, out)


@triton.jit(do_not_specialize=["nk", "q0", "k0", "nrf", "nrs"])
def attn(
    Qf0,
    KVf0,
    Of0,
    Qs0,
    KVs0,
    Os0,
    Qf1,
    KVf1,
    Of1,
    Qs1,
    KVs1,
    Os1,
    nk,
    q0,
    k0,
    nrf,
    nrs,
    LKP_F: tl.constexpr,
    DH_F: tl.constexpr,
    DVT_F: tl.constexpr,
    QR_F: tl.constexpr,
    WB_F: tl.constexpr,
    C_F: tl.constexpr,
    KVH_F: tl.constexpr,
    SPAN_F: tl.constexpr,
    LKP_S: tl.constexpr,
    DH_S: tl.constexpr,
    DVT_S: tl.constexpr,
    QR_S: tl.constexpr,
    WB_S: tl.constexpr,
    C_S: tl.constexpr,
    KVH_S: tl.constexpr,
    SPAN_S: tl.constexpr,
    S: tl.constexpr,
    KBM: tl.constexpr,
    VBM: tl.constexpr,
):
    """Both attention ops of the device, each head-split over two columns
    whose cores run both: a dispatch runs one op's nrf or nrs rounds per
    column and none of the other's. Per group: QR rounds per head, a window
    of WB K blocks (0 is causal), C = log2e / sqrt(DH), and KVH KV heads,
    each nr // KVH consecutive rounds', SPAN elements of records apart."""
    _rounds(
        Qf0,
        KVf0,
        Of0,
        nk,
        q0,
        k0,
        nrf,
        LKP_F,
        DH_F,
        DVT_F,
        QR_F,
        WB_F,
        C_F,
        KVH_F,
        SPAN_F,
        S,
        KBM,
        VBM,
    )
    _rounds(
        Qs0,
        KVs0,
        Os0,
        nk,
        q0,
        k0,
        nrs,
        LKP_S,
        DH_S,
        DVT_S,
        QR_S,
        WB_S,
        C_S,
        KVH_S,
        SPAN_S,
        S,
        KBM,
        VBM,
    )
    _rounds(
        Qf1,
        KVf1,
        Of1,
        nk,
        q0,
        k0,
        nrf,
        LKP_F,
        DH_F,
        DVT_F,
        QR_F,
        WB_F,
        C_F,
        KVH_F,
        SPAN_F,
        S,
        KBM,
        VBM,
    )
    _rounds(
        Qs1,
        KVs1,
        Os1,
        nk,
        q0,
        k0,
        nrs,
        LKP_S,
        DH_S,
        DVT_S,
        QR_S,
        WB_S,
        C_S,
        KVH_S,
        SPAN_S,
        S,
        KBM,
        VBM,
    )


def _keys(lkp, dh, dvt, t=None):
    """The schedule's constants for one group's shape. `t` is the transfers'
    (rows of 8x8 blocks of a core, S, KBM, VBM) of `_rounds`."""
    r = NQT.value * lkp
    keys = {"LKP": lkp, "R": r, "R8": r // 8, "MB": lkp // 8, "KB": dh // 8}
    keys |= {"DVC": dh // dvt, "VB": dvt // 8}
    # With four or more of P's blocks per row block, a block of acc at a time
    # keeps them all live across acc's loop; two by two blocks of acc instead read
    # two of P's and two of V's blocks per step of P V's K loop.
    two = lkp // 8 >= 4
    keys |= {"AT": 2 if two else 1, "AU": 2 if two else 4}
    tmb, sp, kbm, vbm = t or (lkp // 8, 1, dh // 8, dvt // 8)
    keys |= {"TMB": tmb}
    # The packs on the transfers: [rows, cols] of blocks with a unit dim
    # after each, S on the columns of the group of larger blocks and on the
    # rows of the other.
    big = tmb == lkp // 8

    def pack(rows, cols):
        return "x".join(map(str, rows + [1] + cols + [1]))

    # The L3<->L2 transfers' shapes in 8x8 blocks.
    mb, kb, vb, dvc = lkp // 8, dh // 8, dvt // 8, dh // dvt
    keys["L2Q"] = "x".join(map(str, [(r // 8) * kb // (sp * kbm), sp, kbm]))
    keys["L2K"] = "x".join(map(str, [mb * kb // (sp * kbm), sp, kbm]))
    keys["L2V"] = "x".join(map(str, [dvc, mb * vb // (sp * vbm), sp, vbm]))
    t0 = NQT.value * tmb
    if big:
        keys["QPK"] = pack([t0], [sp, kbm])
        keys["KPK"] = pack([tmb], [sp, kbm])
        keys["VPK"] = pack([tmb], [dh // dvt, sp, vbm])
        keys["VPC"] = pack([tmb], [1, sp, vbm])
    else:
        keys["QPK"] = pack([t0, sp], [kbm])
        keys["KPK"] = pack([tmb, sp], [kbm])
        keys["VPK"] = pack([tmb, sp], [dh // dvt, vbm])
        keys["VPC"] = pack([tmb, sp], [1, vbm])
    return keys


def _fill(text, keys):
    for k, v in keys.items():
        text = text.replace(f"{{{k}}}", str(v))
    return text


def render(nests):
    """The transform script for a kernel of several round loops, in program
    order: `nests` holds each loop's (suffix, group suffix, column suffix,
    lkp, dh, dvt, transfers as `_keys` takes them).

    The template's blocks, each closed by `// @end`: `// @each` is repeated
    for each loop, `// @single` and `// @multi` likewise but only for one loop
    or for several, `// @group` for each group; with several loops,
    `// @shared` is kept once and `// @column` is repeated for each column,
    `{EACH:%x}` there standing for %x of each of the column's loops."""
    groups, columns = {}, {}
    for n, g, c, *shape in nests:
        groups.setdefault(g, shape)
        columns.setdefault(c, []).append(n)
    multi = len(nests) > 1

    def expand(kind, body):
        if kind in ("each", "single", "multi"):
            if (kind == "single" and multi) or (kind == "multi" and not multi):
                return []
            return [
                _fill(
                    body.replace("{N}", n).replace("{G}", g).replace("{C}", c),
                    _keys(*shape),
                )
                for n, g, c, *shape in nests
            ]
        if kind == "group":
            return [
                _fill(body.replace("{G}", g), _keys(*shape))
                for g, shape in groups.items()
            ]
        if not multi:
            return []
        if kind == "shared":
            return [body]
        return [
            re.sub(
                r"\{EACH:(%\w+)\}",
                lambda m: ", ".join(m.group(1) + n for n in on),
                body,
            ).replace("{C}", c)
            for c, on in columns.items()
        ]

    out, block, kind = [], [], None
    for line in SCHEDULE.read_text().split("\n"):
        tag = line.strip()
        if tag.startswith("// @") and tag != "// @end":
            kind, block = tag[4:], []
        elif tag == "// @end":
            out += expand(kind, "\n".join(block))
            kind = None
        elif kind:
            block.append(line)
        elif tag == "{LOOP_SPLIT}":
            res = ", ".join(f"%kv_loop0{n}, %rounds_loop{n}" for n, *_ in nests)
            types = ", ".join(["!transform.any_op"] * (2 * len(nests)))
            out.append(
                f"    {res} = transform.split_handle %loops0 : "
                f"(!transform.any_op) -> ({types})"
            )
        else:
            out.append(line)
    text = "\n".join(out)
    left = re.search(r"\{(N|G|C|EACH:[^}]*|[A-Z0-9]+)\}", text)
    if left:
        raise ValueError(f"unfilled {left.group(0)} in {SCHEDULE.name}")
    return text


def dvt(dh):
    """Width of the column chunks of V a core holds at a time."""
    return dh // 4


def v_chunks(cfg):
    """The column chunks of V in a KV record, the same for every group."""
    (n,) = {g.dh // dvt(g.dh) for g in cfg.attn.values()}
    return n


def _consts(g, rows, window):
    qr = rows // (NQT.value * g.lkp)
    wb = window // g.lkp if window else 0
    span = g.max_blocks * 2 * g.lkp * g.dh
    return (g.lkp, g.dh, dvt(g.dh), qr, wb, LOG2E / g.dh**0.5, g.kv_heads, span)


def transfer(f, s):
    """(S, KBM, VBM) of `_rounds`: both groups' tiles must hold the same
    elements, one group's S times the rows of the other's."""
    big, small = sorted((f, s), key=lambda g: -g.dh)
    k = big.dh // small.dh
    if big.lkp * k != small.lkp or big.dh // dvt(big.dh) != small.dh // dvt(small.dh):
        raise ValueError(f"groups of tiles {f.lkp}x{f.dh} and {s.lkp}x{s.dh}")
    return k, small.dh // 8, dvt(small.dh) // 8


def args(f, s, nrf, nrs, nk, q0, k0, bufs, rows=128, window=None):
    """attn's arguments for groups `f` (causal) and `s` (sliding `window`)
    over `rows` q rows a chunk: `bufs` are the q, KV and o of each group of
    each column, in the kernel's order."""
    return (
        *bufs,
        nk,
        q0,
        k0,
        nrf,
        nrs,
        *_consts(f, rows, None),
        *_consts(s, rows, window),
        *transfer(f, s),
    )


def schedule(f, s, out_dir):
    """The transform script for `attn` with groups `f` and `s`, written to
    `out_dir`."""
    t = (max(f, s, key=lambda g: g.dh).lkp // 8, *transfer(f, s))
    nests = []
    for c in range(2):
        for n, g in (("f", f), ("s", s)):
            nests.append((f"_{n}{c}", f"_{n}", str(c), g.lkp, g.dh, dvt(g.dh), t))
    path = Path(out_dir) / f"transform_attention_{f.dh}_{s.dh}.mlir"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path.write_text(render(nests))
    return path
