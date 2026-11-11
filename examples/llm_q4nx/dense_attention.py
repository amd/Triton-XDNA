# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Causal GQA prefill attention of the dense models' fused prefill device.

One op: the heads split over two columns, each a group of
`gemma4_attention.Attn` (half the KV heads each, their records max_blocks
apart), running `gemma4_attention._rounds` on its cores; the buffers and
numerics are those of `gemma4_attention`, but for V, which a record holds in
one column chunk (`dvt`) and a core takes whole. nk, q0 and k0 are the op's
run-time arguments.
"""

from pathlib import Path

import triton
import triton.language as tl

import gemma4_attention as A

TEMPLATE = Path(__file__).resolve().parent / "transform_dense_attention_aie2p.mlir.in"


def dvt(dh):
    """Width of V's column chunks in a KV record: one chunk, so a core takes
    a block's V whole (dh <= 128 fits L1)."""
    return dh


@triton.jit(do_not_specialize=["nk", "q0", "k0", "nr"])
def attn(
    Q0,
    KV0,
    O0,
    Q1,
    KV1,
    O1,
    nk,
    q0,
    k0,
    nr,
    LKP: tl.constexpr,
    DH: tl.constexpr,
    DVT: tl.constexpr,
    QR: tl.constexpr,
    C: tl.constexpr,
    KVH: tl.constexpr,
    SPAN: tl.constexpr,
    S: tl.constexpr,
    KBM: tl.constexpr,
    VBM: tl.constexpr,
):
    """nr rounds of each column's group: QR rounds per head, C = log2e /
    sqrt(DH), KVH KV heads, each nr // KVH consecutive rounds', SPAN elements
    of records apart."""
    A._rounds(Q0, KV0, O0, nk, q0, k0, nr, LKP, DH, DVT, QR, 0, C, KVH, SPAN, S, KBM, VBM)  # fmt: skip
    A._rounds(Q1, KV1, O1, nk, q0, k0, nr, LKP, DH, DVT, QR, 0, C, KVH, SPAN, S, KBM, VBM)  # fmt: skip


def transfer(g):
    """(S, KBM, VBM) of `_rounds` for one group's tiles."""
    return 1, g.dh // 8, dvt(g.dh) // 8


def args(g, nr, nk, q0, k0, bufs, rows=128):
    """attn's arguments for group shape `g` over `rows` q rows a chunk:
    `bufs` are the q, KV and o of each column's group."""
    lkp, dh, _, qr, _, c, kvh, span = A._consts(g, rows, None)
    return (*bufs, nk, q0, k0, nr, lkp, dh, dvt(dh), qr, c, kvh, span, *transfer(g))


def schedule(g, out_dir):
    """The transform script for `attn` with groups of shape `g`, written to
    `out_dir`."""
    t = (g.lkp // 8, *transfer(g))
    nests = [(f"_a{c}", "_a", str(c), g.lkp, g.dh, dvt(g.dh), t) for c in range(2)]
    path = Path(out_dir) / f"transform_attention_{g.dh}.mlir"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path.write_text(A.render(nests, TEMPLATE))
    return path
