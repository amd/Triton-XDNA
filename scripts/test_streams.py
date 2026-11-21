# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""`streams.lower_streams`: a herd's DDR transfers lowered to shim channels,
memtile rings and core channels.

A 2x2 herd reads A per row (broadcast along it), B per column (its tiles a
stride apart, so one linear shim put per tile) and writes C per core
(gathered per column). Each channel's shim ops, expanded to element offsets,
must give the elements its cores move, in their order; the channels, the
memtile program and the order of the shim ops are checked too, and the IR
the walk can't follow exactly must be rejected. No NPU, no aircc.
"""

import contextlib
import itertools
import re

from air.ir import Context, Module

from triton.backends.amd_triton_npu.streams import (
    _fit,
    _fits,
    _merge,
    _shim_pattern,
    lower_streams,
)
from triton.backends.amd_triton_npu.streams import _shim_ops as _stream_ops


@contextlib.contextmanager
def _raises(match=""):
    """Expect a ValueError whose message has `match`. The file also runs as a
    plain script, without pytest."""
    try:
        yield
    except ValueError as e:
        assert match in str(e), e
    else:
        raise AssertionError("no ValueError")


def _expand(offsets, sizes, strides):
    """The element offsets an access pattern visits, in order."""
    base = sum(o * s for o, s in zip(offsets, strides))
    return [
        base + sum(i * s for i, s in zip(idx, strides))
        for idx in itertools.product(*(range(c) for c in sizes))
    ]


def test_fit():
    assert _fit([5]) == []
    assert _fit([0, 16, 32, 48]) == [(4, 16)]
    # two runs of two, 100 apart
    assert _fit([0, 8, 100, 108]) == [(2, 100), (2, 8)]
    assert _fit([0, 1, 5]) is None


def test_merge():
    # a run of tiles one after another is one dim
    assert _merge([(4, 16), (16, 1)]) == [(64, 1)]
    assert _merge([(1, 7), (4, 32), (16, 1)]) == [(4, 32), (16, 1)]


def test_eval_integer_semantics():
    """The walk's arithmetic is the hardware's: signed division and
    remainder round toward zero, unsigned compares see two's complement."""
    from triton.backends.amd_triton_npu.superkernel import _eval

    text = """
    func.func @f() {
      %a = arith.constant -7 : i64
      %b = arith.constant 2 : i64
      %d = arith.divsi %a, %b : i64
      %r = arith.remsi %a, %b : i64
      %f = arith.floordivsi %a, %b : i64
      %lt = arith.cmpi ult, %a, %b : i64
      %slt = arith.cmpi slt, %a, %b : i64
      %mx = arith.maxsi %a, %b : i64
      %mn = arith.minsi %a, %b : i64
      %big = arith.constant 9007199254740993 : i64
      %one = arith.constant 1 : i64
      %q = arith.divsi %big, %one : i64
      return
    }"""
    with Context():
        m = Module.parse(text)
        ops = list(m.body.operations[0].regions[0].blocks[0].operations)
        vals = [_eval(o.result, []) for o in ops if len(o.results)]
    _, _, d, r, f, lt, slt, mx, mn, _, _, q = vals
    assert (d, r, f) == (-3, -1, -4), vals
    assert (lt, slt) == (0, -1), vals
    assert (mx, mn) == (2, -7), vals
    assert q == 9007199254740993, vals
    text = """
    func.func @f() {
      %m = arith.constant -1 : i32
      %two = arith.constant 2 : i32
      %d = arith.divui %m, %two : i32
      %e = arith.extui %m : i32 to i64
      %t = arith.constant true
      %b = arith.extui %t : i1 to i32
      %w = arith.constant 300 : i32
      %n = arith.trunci %w : i32 to i8
      return
    }"""
    with Context():
        m = Module.parse(text)
        ops = list(m.body.operations[0].regions[0].blocks[0].operations)
        vals = [_eval(o.result, []) for o in ops if len(o.results)]
    _, _, d, e, _, b, _, n = vals
    assert (d, e, b, n) == ((1 << 31) - 1, (1 << 32) - 1, 1, 44), vals
    # a value is its type's bits: casts truncate and sign-extend, and an
    # unsigned result compares as those bits
    text = """
    func.func @f() {
      %big = arith.constant 4294967296 : index
      %t32 = arith.index_cast %big : index to i32
      %t = arith.constant true
      %s = arith.extsi %t : i1 to i32
      %m = arith.constant -1 : i32
      %one = arith.constant 1 : i32
      %q = arith.divui %m, %one : i32
      %eq = arith.cmpi eq, %q, %m : i32
      %neg = arith.cmpi slt, %q, %one : i32
      return
    }"""
    with Context():
        m = Module.parse(text)
        ops = list(m.body.operations[0].regions[0].blocks[0].operations)
        vals = [_eval(o.result, []) for o in ops if len(o.results)]
    _, t32, _, s, _, _, q, eq, neg = vals
    assert (t32, s, q, eq, neg) == (0, -1, -1, -1, -1), vals
    # a compare's result is an i1 like a constant one
    text = """
    func.func @f() {
      %a = arith.constant 3 : i32
      %b = arith.constant 5 : i32
      %lt = arith.cmpi slt, %a, %b : i32
      %s = arith.extsi %lt : i1 to i32
      %z = arith.extui %lt : i1 to i32
      %t = arith.constant true
      %same = arith.cmpi eq, %lt, %t : i1
      return
    }"""
    with Context():
        m = Module.parse(text)
        ops = list(m.body.operations[0].regions[0].blocks[0].operations)
        vals = [_eval(o.result, []) for o in ops if len(o.results)]
    _, _, lt, s, z, _, same = vals
    assert (lt, s, z, same) == (-1, -1, 1, -1), vals


def _seq(bases, sizes, strides):
    return [(b, sizes, strides, i) for i, b in enumerate(bases)]


def test_shim_pattern():
    seq = _seq((0, 32, 64, 96), [16], [1])
    assert _shim_pattern(seq) == (0, [4, 16], [32, 1])
    with _raises("differ in shape"):
        _shim_pattern([(0, [16], [1], 0), (16, [8], [1], 1)])


def test_fits():
    # a GEMM's: A's row repeated per round, a B run, C's column
    assert _fits([11, 147456], [0, 1], 1)
    assert _fits([81920], [1], 2)
    assert _fits([11, 128, 64], [384, 4224, 1], 2)
    # a repeat inside a walking dim; a negative stride; a wrap or a step too
    # big; four addressing dims
    assert not _fits([4, 4, 4, 4], [64, 0, 16, 1], 2)
    assert not _fits([2, 16], [-16, 1], 4)
    assert not _fits([2048, 2], [4, 1], 4)
    assert not _fits([2, 16], [1 << 21, 1], 4)
    assert not _fits([2, 2, 2, 2, 16], [9, 7, 5, 3, 1], 4)


def test_shim_ops():
    # runs that advance: one linear op each, the start in the offset, with
    # the place of its first transfer
    seq = _seq((8, 40, 72), [16], [1])
    assert _stream_ops(seq, 2) == [
        (([8], [16], [1]), 0),
        (([40], [16], [1]), 1),
        (([72], [16], [1]), 2),
    ]
    # two transfers a run
    seq = _seq((0, 16, 100, 116), [16], [1])
    assert _stream_ops(seq, 2) == [(([0], [32], [1]), 0), (([100], [32], [1]), 2)]
    # runs that meet are one
    seq = _seq((0, 32, 16, 48), [16], [1])
    assert _stream_ops(seq, 2) == [
        (([0], [16], [1]), 0),
        (([32], [16], [1]), 1),
        (([16], [16], [1]), 2),
        (([48], [16], [1]), 3),
    ]
    seq = _seq((0, 32, 48, 80), [16], [1])
    assert _stream_ops(seq, 2) == [
        (([0], [16], [1]), 0),
        (([32], [32], [1]), 1),
        (([80], [16], [1]), 3),
    ]
    # one run
    seq = _seq((0, 16, 32), [16], [1])
    assert _stream_ops(seq, 2) == [(([0], [48], [1]), 0)]
    # a run repeated: one op
    seq = _seq((0, 0, 0), [16], [1])
    assert _stream_ops(seq, 2) == [(([0, 0], [3, 16], [0, 1]), 0)]
    # not contiguous inside: one op
    seq = _seq((0, 100), [4, 4], [8, 1])
    assert _stream_ops(seq, 2) == [(([0, 0, 0], [2, 4, 4], [100, 8, 1]), 0)]
    # an innermost stride other than 1: the start where a stride divides it
    seq = _seq((5, 6), [16], [64])
    ops = _stream_ops(seq, 2)
    assert [e for p, _ in ops for e in _expand(*p)] == [
        b + 64 * i for b in (5, 6) for i in range(16)
    ], ops
    # what no BD makes is split along its outer dims, each part with its
    # place: a repeat inside a walking dim
    seq = _seq((0, 0, 64, 64), [4, 4], [16, 1])
    ops = _stream_ops(seq, 2)
    assert [e for p, _ in ops for e in _expand(*p)] == [
        b + 16 * r + c for b in (0, 0, 64, 64) for r in range(4) for c in range(4)
    ], ops
    assert [pl for _, pl in ops] == [0, 2], ops


HEAD = """
module {
  func.func @k(%A: memref<*xbf16>, %B: memref<*xbf16>, %C: memref<*xf32>) {
    %c1 = arith.constant 1 : index
    air.launch (%l0, %l1, %l2) in (%s0=%c1, %s1=%c1, %s2=%c1)
        args(%a=%A, %b=%B, %c=%C) : memref<*xbf16>, memref<*xbf16>, memref<*xf32> {
      air.segment @seg args(%sa=%a, %sb=%b, %sc=%c)
          : memref<*xbf16>, memref<*xbf16>, memref<*xf32> {
        %c2 = arith.constant 2 : index
        air.herd @h tile (%x, %y) in (%nx=%c2, %ny=%c2) args(%ha=%sa, %hb=%sb, %hc=%sc)
            : memref<*xbf16>, memref<*xbf16>, memref<*xf32> {
          %c0 = arith.constant 0 : index
          %one = arith.constant 1 : index
          %c4 = arith.constant 4 : index
          %c8 = arith.constant 8 : index
          %c16 = arith.constant 16 : index
          %c32 = arith.constant 32 : index
          %c64 = arith.constant 64 : index
          %la = memref.alloc() : memref<16xbf16, 2>
          %lb = memref.alloc() : memref<16xbf16, 2>
          %lc = memref.alloc() : memref<8xf32, 2>
          %fa = memref.reinterpret_cast %ha to offset: [0], sizes: [1024], strides: [1]
              : memref<*xbf16> to memref<1024xbf16>
"""
TAIL = """
          memref.dealloc %la : memref<16xbf16, 2>
          memref.dealloc %lb : memref<16xbf16, 2>
          memref.dealloc %lc : memref<8xf32, 2>
        }
      }
    }
    return
  }
}
"""

# A: the row's four tiles one after another, the offset on the transfer.
# B: the column's tiles 32 apart, the offset in the view.
# C: one tile per core, a column's rows 16 apart.
GEMM = HEAD + """
          scf.for %k = %c0 to %c4 step %one {
            %ya = arith.muli %y, %c64 : index
            %ka = arith.muli %k, %c16 : index
            %oa = arith.addi %ya, %ka : index
            air.dma_memcpy_nd (%la[] [] [], %fa[%oa] [%c16] [%one])
                : (memref<16xbf16, 2>, memref<1024xbf16>)
            %xb = arith.muli %x, %c16 : index
            %kb = arith.muli %k, %c32 : index
            %ob = arith.addi %xb, %kb : index
            %vb = memref.reinterpret_cast %hb to offset: [%ob], sizes: [16], strides: [1]
                : memref<*xbf16> to memref<16xbf16, strided<[1], offset: ?>>
            air.dma_memcpy_nd (%lb[] [] [], %vb[] [] [])
                : (memref<16xbf16, 2>, memref<16xbf16, strided<[1], offset: ?>>)
          }
          %xc = arith.muli %x, %c8 : index
          %yc = arith.muli %y, %c16 : index
          %oc = arith.addi %xc, %yc : index
          %vc = memref.reinterpret_cast %hc to offset: [%oc], sizes: [8], strides: [1]
              : memref<*xf32> to memref<8xf32, strided<[1], offset: ?>>
          air.dma_memcpy_nd (%vc[] [] [], %lc[] [] [])
              : (memref<8xf32, strided<[1], offset: ?>>, memref<8xf32, 2>)
""" + TAIL


def _lower(text):
    with Context():
        m = Module.parse(text)
        streams = lower_streams(m, "s")
        return str(m), streams


def _consts(text):
    return dict(re.findall(r"%(\w+) = arith\.constant (\d+) : index", text))


def _shim_ops(text):
    """(get|put, channel, index, offsets, sizes, strides) of the launch's
    channel ops, in order; the lists as ints."""
    launch = text[text.index("air.launch") : text.index("air.segment")]
    pat = (
        r"air\.channel\.(get|put)\s+@(\w+)\[%(\w+)\] \(%\w+\[([^\]]*)\] "
        r"\[([^\]]*)\] \[([^\]]*)\]\)"
    )
    consts = _consts(text)
    ints = lambda s: [int(v) for v in s.split(",")]  # noqa: E731
    return [
        (op, ch, int(consts[i]), ints(o), ints(sz), ints(st))
        for op, ch, i, o, sz, st in re.findall(pat, launch)
    ]


def _moved(text, channel):
    """Per index of shim channel `channel`, the elements its ops move."""
    out = {}
    for _, ch, i, o, sz, st in _shim_ops(text):
        if ch == channel:
            out.setdefault(i, []).extend(_expand(o, sz, st))
    return out


def test_gemm_moves_the_cores_elements():
    text, streams = _lower(GEMM)
    assert [s.dims for s in streams] == [(1,), (0,), (0, 1)], [s.dims for s in streams]
    r = range(16)
    a = {y: [64 * y + 16 * k + i for k in range(4) for i in r] for y in range(2)}
    b = {x: [16 * x + 32 * k + i for k in range(4) for i in r] for x in range(2)}
    c = {x: [8 * x + 16 * y + i for y in range(2) for i in range(8)] for x in range(2)}
    assert _moved(text, "s0_ddr") == a
    assert _moved(text, "s1_ddr") == b
    assert _moved(text, "s2_ddr") == c


def test_gemm_channels():
    text, _ = _lower(GEMM)
    # A broadcast along each row, B along each column, C one per core
    assert (
        "air.channel @s0_l1 [1, 2] {broadcast_shape = [2 : index, 2 : index]}" in text
    )
    assert (
        "air.channel @s1_l1 [2, 1] {broadcast_shape = [2 : index, 2 : index]}" in text
    )
    assert "air.channel @s2_l1 [2, 2]\n" in text
    for k in range(3):
        assert f"air.channel @s{k}_ddr [2]" in text
    # no transfer left, nor the DDR views only they read
    assert "air.dma_memcpy_nd" not in text
    herd = text[text.index("air.herd") :]
    assert "memref.reinterpret_cast" not in herd, herd
    assert herd.count("air.channel.get  @s0_l1[%arg") == 1
    assert herd.count("air.channel.get  @s1_l1[%arg") == 1
    assert herd.count("air.channel.put  @s2_l1[%arg") == 1


def test_gemm_memtile():
    text, _ = _lower(GEMM)
    seg = text[text.index("air.segment") : text.index("air.herd")]
    # a ring of two per instance: per row for A, per column for B, in the
    # memtile of the row's or column's number
    cols = re.findall(
        r'npu\.stream_col = (\d+) : i32, npu\.stream_herd = "h"\} : memref<16xbf16, 1 : i32>',
        seg,
    )
    assert cols == ["0", "0", "1", "1", "0", "0", "1", "1"], cols
    # C: one buffer per column, its rows' tiles one after another
    c = re.findall(
        r'\{air\.no_split, npu\.stream_col = (\d+) : i32, npu\.stream_herd = "h"\}', seg
    )
    assert c == ["0", "1"], c
    offs = re.findall(
        r"air\.channel\.get  @s2_l1\[[^\]]*\] \(%\w+\[(\d+)\] \[8\] \[1\]\)", seg
    )
    assert offs == ["0", "8", "0", "8"], offs


def test_gemm_shim_order():
    text, _ = _lower(GEMM)
    assert "attributes {air.preserve_shim_dma_order}" in text
    ops = [(op, ch, i, o[-1]) for op, ch, i, o, _, _ in _shim_ops(text)]
    # the channels of one op first: A's rows, C's columns; then B, one linear
    # put per tile, the columns in turn
    assert ops == [
        ("put", "s0_ddr", 0, 0),
        ("put", "s0_ddr", 1, 64),
        ("get", "s2_ddr", 0, 0),
        ("get", "s2_ddr", 1, 8),
    ] + [("put", "s1_ddr", k % 2, 16 * k) for k in range(8)], ops


# A and B both split into a put per tile: the shim takes them in the cores'
# order, A's tile and B's of one step together.
TWO_PACED = HEAD + """
          %fb = memref.reinterpret_cast %hb to offset: [0], sizes: [1024], strides: [1]
              : memref<*xbf16> to memref<1024xbf16>
          scf.for %k = %c0 to %c4 step %one {
            %ya = arith.muli %y, %c16 : index
            %ka = arith.muli %k, %c64 : index
            %oa = arith.addi %ya, %ka : index
            air.dma_memcpy_nd (%la[] [] [], %fa[%oa] [%c16] [%one])
                : (memref<16xbf16, 2>, memref<1024xbf16>)
            %xb = arith.muli %x, %c16 : index
            %kb = arith.muli %k, %c32 : index
            %ob = arith.addi %xb, %kb : index
            air.dma_memcpy_nd (%lb[] [] [], %fb[%ob] [%c16] [%one])
                : (memref<16xbf16, 2>, memref<1024xbf16>)
          }
""" + TAIL


def test_paced_streams_in_the_cores_order():
    text, _ = _lower(TWO_PACED)
    ops = [(ch, i) for _, ch, i, _, _, _ in _shim_ops(text)]
    step = [("s0_ddr", 0), ("s0_ddr", 1), ("s1_ddr", 0), ("s1_ddr", 1)]
    assert ops == step * 4, ops


# Each core reads a tile of its own (the offset depends on both coordinates).
OWN = HEAD + """
          %xo = arith.muli %x, %c32 : index
          %yo = arith.muli %y, %c16 : index
          %oo = arith.addi %xo, %yo : index
          %vo = memref.reinterpret_cast %ha to offset: [%oo], sizes: [16], strides: [1]
              : memref<*xbf16> to memref<16xbf16, strided<[1], offset: ?>>
          air.dma_memcpy_nd (%la[] [] [], %vo[] [] [])
              : (memref<16xbf16, 2>, memref<16xbf16, strided<[1], offset: ?>>)
""" + TAIL


def test_read_of_each_core():
    text, streams = _lower(OWN)
    assert [s.dims for s in streams] == [(0, 1)]
    assert "air.channel @s0_l1 [2, 2]\n" in text
    assert "air.channel @s0_ddr [4]" in text
    seg = text[text.index("air.segment") : text.index("air.herd")]
    consts = _consts(text)
    idx = [
        (int(consts[a]), int(consts[b]))
        for a, b in re.findall(r"air\.channel\.put  @s0_l1\[%(\w+), %(\w+)\]", seg)
    ]
    # each instance's ring goes to its own core
    assert idx == [(0, 0)] * 2 + [(0, 1)] * 2 + [(1, 0)] * 2 + [(1, 1)] * 2, idx
    assert _moved(text, "s0_ddr") == {
        j: list(range(16 * j, 16 * j + 16)) for j in range(4)
    }


# A column of A, 64 apart: the innermost stride is not 1.
STRIDED = HEAD + """
          %yo = arith.muli %y, %c4 : index
          air.dma_memcpy_nd (%la[] [] [], %fa[%yo] [%c16] [%c64])
              : (memref<16xbf16, 2>, memref<1024xbf16>)
""" + TAIL


def test_strided_read():
    text, _ = _lower(STRIDED)
    # (an offset counts its dim's stride: row y starts at 4 y 64)
    assert _moved(text, "s0_ddr") == {
        y: [256 * y + 64 * i for i in range(16)] for y in range(2)
    }


# The rows of a column write differently strided tiles.
ROWS = HEAD + """
          %fc = memref.reinterpret_cast %hc to offset: [0], sizes: [1024], strides: [1]
              : memref<*xf32> to memref<1024xf32>
          %xo = arith.muli %x, %c64 : index
          %yo = arith.muli %y, %c16 : index
          %o = arith.addi %xo, %yo : index
          %two = arith.constant 2 : index
          %ys = arith.muli %y, %c4 : index
          %st = arith.addi %ys, %c4 : index
          air.dma_memcpy_nd (%fc[%o, %c0] [%two, %c4] [%st, %one], %lc[] [] [])
              : (memref<1024xf32>, memref<8xf32, 2>)
""" + TAIL


# Fewer offsets than sizes: AIR takes the missing ones as leading zeros, so
# %yo is the offset along the inner dimension.
SHORT_OFFSETS = HEAD + """
          %yo = arith.muli %y, %c4 : index
          air.dma_memcpy_nd (%la[] [] [], %fa[%yo] [%c4, %c4] [%c64, %one])
              : (memref<16xbf16, 2>, memref<1024xbf16>)
""" + TAIL


def test_short_offsets():
    text, _ = _lower(SHORT_OFFSETS)
    assert _moved(text, "s0_ddr") == {
        y: [4 * y + 64 * i + j for i in range(4) for j in range(4)] for y in range(2)
    }


# Each core reads its tile of C and writes it back.
IN_PLACE = HEAD + """
          %fc = memref.reinterpret_cast %hc to offset: [0], sizes: [1024], strides: [1]
              : memref<*xf32> to memref<1024xf32>
          %xo = arith.muli %x, %c64 : index
          %yo = arith.muli %y, %c16 : index
          %o = arith.addi %xo, %yo : index
          air.dma_memcpy_nd (%lc[] [] [], %fc[%o] [%c8] [%one])
              : (memref<8xf32, 2>, memref<1024xf32>)
          air.dma_memcpy_nd (%fc[%o] [%c8] [%one], %lc[] [] [])
              : (memref<1024xf32>, memref<8xf32, 2>)
""" + TAIL


def test_read_and_write_one_buffer():
    with _raises("both read and written"):
        _lower(IN_PLACE)


def test_rows_of_different_shapes():
    with _raises("rows write different shapes"):
        _lower(ROWS)


# One stream's transfers into L1 buffers of 8 and 16 elements; the first's
# loop runs no trips.
SIZES = HEAD + """
          %l8 = memref.alloc() : memref<8xbf16, 2>
          %ya = arith.muli %y, %c64 : index
          scf.for %k = %c0 to %c0 step %one {
            air.dma_memcpy_nd (%l8[] [] [], %fa[%ya] [%c8] [%one])
                : (memref<8xbf16, 2>, memref<1024xbf16>)
          }
          air.dma_memcpy_nd (%la[] [] [], %fa[%ya] [%c16] [%one])
              : (memref<16xbf16, 2>, memref<1024xbf16>)
""" + TAIL


def test_transfers_of_different_sizes():
    with _raises("move different amounts"):
        _lower(SIZES)


# A transfer in an affine.for, which the walk does not run.
AFFINE = HEAD + """
          %o = arith.muli %y, %c16 : index
          affine.for %i = 0 to 4 {
            air.dma_memcpy_nd (%la[] [] [], %fa[%o] [%c16] [%one])
                : (memref<16xbf16, 2>, memref<1024xbf16>)
          }
""" + TAIL


def test_unknown_loop():
    with _raises("can't follow"):
        _lower(AFFINE)


# Every core reads the same tile.
SAME = HEAD + """
          air.dma_memcpy_nd (%la[] [] [], %fa[%c0] [%c16] [%one])
              : (memref<16xbf16, 2>, memref<1024xbf16>)
""" + TAIL


def test_read_the_same_for_every_core():
    with _raises("the same for every core"):
        _lower(SAME)


# A transfer through a view of dynamic strides.
DYNAMIC = HEAD + """
          %fd = memref.reinterpret_cast %ha to offset: [0], sizes: [16, 16], strides: [%c64, 1]
              : memref<*xbf16> to memref<16x16xbf16, strided<[?, 1]>>
          %sv = memref.subview %fd[%y, 0] [1, 16] [1, 1]
              : memref<16x16xbf16, strided<[?, 1]>> to memref<16xbf16, strided<[1], offset: ?>>
          air.dma_memcpy_nd (%la[] [] [], %sv[] [] [])
              : (memref<16xbf16, 2>, memref<16xbf16, strided<[1], offset: ?>>)
""" + TAIL


def test_dynamic_view():
    with _raises("dynamic"):
        _lower(DYNAMIC)


ONE_ARG = """
module {
  func.func @k(%A: memref<*xbf16>) {
    %c1 = arith.constant 1 : index
    air.launch (%l0, %l1, %l2) in (%s0=%c1, %s1=%c1, %s2=%c1) args(%a=%A) : memref<*xbf16> {
      air.segment @seg args(%sa=%a) : memref<*xbf16> {
        %c2 = arith.constant 2 : index
        %c1s = arith.constant 1 : index
        %f = arith.constant false
%BODY%
      }
    }
    return
  }
}
"""


def _herd(name, n):
    """A herd of n x n cores that reads 16 elements of A per row."""
    return f"""
        air.herd @{name} tile (%x, %y) in (%nx={n}, %ny={n}) args(%ha=%sa) : memref<*xbf16> {{
          %one = arith.constant 1 : index
          %c16 = arith.constant 16 : index
          %la = memref.alloc() : memref<16xbf16, 2>
          %fa = memref.reinterpret_cast %ha to offset: [0], sizes: [256], strides: [1]
              : memref<*xbf16> to memref<256xbf16>
          %o = arith.muli %y, %c16 : index
          air.dma_memcpy_nd (%la[] [] [], %fa[%o] [%c16] [%one])
              : (memref<16xbf16, 2>, memref<256xbf16>)
          memref.dealloc %la : memref<16xbf16, 2>
        }}"""


def test_segment_if_runs_one_branch():
    body = f"scf.if %f {{ {_herd('h', '%c2')} }} else {{ {_herd('h', '%c2')} }}"
    text, _ = _lower(ONE_ARG.replace("%BODY%", body))
    assert _moved(text, "s0_ddr") == {
        y: list(range(16 * y, 16 * y + 16)) for y in range(2)
    }


def test_stream_sized_by_its_herd():
    # a 1x1 herd that moves nothing, then the 2x2 one that moves the stream
    empty = "air.herd @p tile (%x, %y) in (%nx=%c1s, %ny=%c1s) {\n}"
    text, _ = _lower(ONE_ARG.replace("%BODY%", empty + _herd("h", "%c2")))
    assert "air.channel @s0_l1 [1, 2]" in text
    assert "air.channel @s0_ddr [2]" in text


def _paced_herd(base):
    """A 2x2 herd @h that reads A per row from element `base` on, a put per
    tile (two tiles 64 apart)."""
    return f"""
        air.herd @h tile (%x, %y) in (%nx=%c2, %ny=%c2) args(%ha=%sa) : memref<*xbf16> {{
          %c0 = arith.constant 0 : index
          %one = arith.constant 1 : index
          %two = arith.constant 2 : index
          %c16 = arith.constant 16 : index
          %c64 = arith.constant 64 : index
          %la = memref.alloc() : memref<16xbf16, 2>
          %fa = memref.reinterpret_cast %ha to offset: [{base}], sizes: [256], strides: [1]
              : memref<*xbf16> to memref<256xbf16, strided<[1], offset: {base}>>
          scf.for %k = %c0 to %two step %one {{
            %ya = arith.muli %y, %c16 : index
            %ka = arith.muli %k, %c64 : index
            %o = arith.addi %ya, %ka : index
            air.dma_memcpy_nd (%la[] [] [], %fa[%o] [%c16] [%one])
                : (memref<16xbf16, 2>, memref<256xbf16, strided<[1], offset: {base}>>)
          }}
          memref.dealloc %la : memref<16xbf16, 2>
        }}"""


def test_paced_over_herds_in_turn():
    # the cores run one herd, then the next: their transfers go on in order
    text, _ = _lower(ONE_ARG.replace("%BODY%", _paced_herd(0) + _paced_herd(128)))
    ops = [(i, o[-1]) for _, _, i, o, _, _ in _shim_ops(text)]
    assert ops == [(y, b + 16 * y) for b in (0, 64, 128, 192) for y in (0, 1)], ops


def test_stream_of_two_herds():
    body = _herd("h", "%c2") + _herd("g", "%c2")
    with _raises("move one stream"):
        _lower(ONE_ARG.replace("%BODY%", body))


# A view of a view: a reinterpret_cast's offset counts from the base, not
# from its source's offset.
RECAST = HEAD + """
          %s = memref.reinterpret_cast %ha to offset: [512], sizes: [256], strides: [1]
              : memref<*xbf16> to memref<256xbf16, strided<[1], offset: 512>>
          %o = arith.muli %y, %c16 : index
          %r = memref.reinterpret_cast %s to offset: [%o], sizes: [16], strides: [1]
              : memref<256xbf16, strided<[1], offset: 512>>
                to memref<16xbf16, strided<[1], offset: ?>>
          air.dma_memcpy_nd (%la[] [] [], %r[] [] [])
              : (memref<16xbf16, 2>, memref<16xbf16, strided<[1], offset: ?>>)
""" + TAIL


def test_view_of_a_view():
    text, _ = _lower(RECAST)
    # both views go with the transfer
    assert "offset: 512" not in text[text.index("air.herd") :]
    assert _moved(text, "s0_ddr") == {
        y: list(range(16 * y, 16 * y + 16)) for y in range(2)
    }


# A column's rows written in reverse order, and a read whose tiles go
# backwards.
BACKWARDS = HEAD + """
          %fc = memref.reinterpret_cast %hc to offset: [0], sizes: [1024], strides: [1]
              : memref<*xf32> to memref<1024xf32>
          %xo = arith.muli %x, %c8 : index
          %yo = arith.muli %y, %c16 : index
          %r = arith.subi %c16, %yo : index
          %o = arith.addi %xo, %r : index
          air.dma_memcpy_nd (%fc[%o] [%c8] [%one], %lc[] [] [])
              : (memref<1024xf32>, memref<8xf32, 2>)
          scf.for %k = %c0 to %c4 step %one {
            %ya = arith.muli %y, %c4 : index
            %kk = arith.subi %c4, %k : index
            %ka = arith.muli %kk, %c64 : index
            %oa = arith.addi %ya, %ka : index
            air.dma_memcpy_nd (%la[] [] [], %fa[%oa, %c0] [%c4, %c4] [%c16, %one])
                : (memref<16xbf16, 2>, memref<1024xbf16>)
          }
""" + TAIL


def test_backwards():
    text, _ = _lower(BACKWARDS)
    assert all(min(st) >= 0 for _, _, _, _, _, st in _shim_ops(text))
    c = {
        x: [8 * x + 16 - 16 * y + i for y in range(2) for i in range(8)]
        for x in range(2)
    }
    assert _moved(text, "s0_ddr") == c
    # (the tile's offset counts its dim's stride, 16)
    a = {
        y: [
            16 * (4 * y + 64 * (4 - k)) + 16 * r + i
            for k in range(4)
            for r in range(4)
            for i in range(4)
        ]
        for y in range(2)
    }
    assert _moved(text, "s1_ddr") == a


# A tile read four times before the next: a repeat inside a walking dim,
# which no BD makes. Its parts are paced, in the cores' order.
REPEAT = HEAD + """
          scf.for %k = %c0 to %c4 step %one {
            scf.for %r = %c0 to %c4 step %one {
              %ya = arith.muli %y, %c4 : index
              %ka = arith.muli %k, %c64 : index
              %oa = arith.addi %ya, %ka : index
              air.dma_memcpy_nd (%la[] [] [], %fa[%oa, %c0] [%c4, %c4] [%c16, %one])
                  : (memref<16xbf16, 2>, memref<1024xbf16>)
            }
          }
""" + TAIL


def test_repeat_in_a_walk():
    text, _ = _lower(REPEAT)
    ops = _shim_ops(text)
    assert all(_fits(sz, st, 2) for _, _, _, _, sz, st in ops), ops
    assert [i for _, _, i, _, _, _ in ops] == [0, 1] * 4, ops
    tile = [16 * r + i for r in range(4) for i in range(4)]
    a = {
        y: [16 * (4 * y + 64 * k) + e for k in range(4) for _ in range(4) for e in tile]
        for y in range(2)
    }
    assert _moved(text, "s0_ddr") == a


# A column's rows write at different points of their programs.
PLACES = HEAD + """
          %fc = memref.reinterpret_cast %hc to offset: [0], sizes: [1024], strides: [1]
              : memref<*xf32> to memref<1024xf32>
          %xo = arith.muli %x, %c8 : index
          %yo = arith.muli %y, %c16 : index
          %o = arith.addi %xo, %yo : index
          %is0 = arith.cmpi eq, %y, %c0 : index
          %ya = arith.muli %y, %c64 : index
          scf.if %is0 {
            air.dma_memcpy_nd (%fc[%o] [%c8] [%one], %lc[] [] [])
                : (memref<1024xf32>, memref<8xf32, 2>)
            air.dma_memcpy_nd (%la[] [] [], %fa[%ya] [%c16] [%one])
                : (memref<16xbf16, 2>, memref<1024xbf16>)
          } else {
            air.dma_memcpy_nd (%la[] [] [], %fa[%ya] [%c16] [%one])
                : (memref<16xbf16, 2>, memref<1024xbf16>)
            air.dma_memcpy_nd (%fc[%o] [%c8] [%one], %lc[] [] [])
                : (memref<1024xf32>, memref<8xf32, 2>)
          }
""" + TAIL


def test_rows_write_at_different_points():
    with _raises("different points"):
        _lower(PLACES)


# The herd's size, read in its body.
SIZE = HEAD + """
          %ya = arith.muli %y, %nx : index
          %o = arith.muli %ya, %c16 : index
          air.dma_memcpy_nd (%la[] [] [], %fa[%o] [%c16] [%one])
              : (memref<16xbf16, 2>, memref<1024xbf16>)
""" + TAIL


def test_herd_size_in_the_body():
    text, _ = _lower(SIZE)
    assert _moved(text, "s0_ddr") == {
        y: list(range(32 * y, 32 * y + 16)) for y in range(2)
    }


# A transfer whose token something waits for.
ASYNC = HEAD + """
          %o = arith.muli %y, %c16 : index
          %t = air.dma_memcpy_nd async (%la[] [] [], %fa[%o] [%c16] [%one])
              : (memref<16xbf16, 2>, memref<1024xbf16>)
          air.wait_all [%t]
""" + TAIL


def test_transfer_waited_for():
    with _raises("token is used"):
        _lower(ASYNC)


# A transfer that waits for another op's token.
WAITS = HEAD + """
          %o = arith.muli %y, %c16 : index
          %t = air.wait_all async
          air.dma_memcpy_nd [%t] (%la[] [] [], %fa[%o] [%c16] [%one])
              : (memref<16xbf16, 2>, memref<1024xbf16>)
""" + TAIL


def test_transfer_that_waits():
    with _raises("waits for tokens"):
        _lower(WAITS)


# A row's read that only the first column's cores make.
FIRST_COLUMN = HEAD + """
          %o = arith.muli %y, %c16 : index
          %x0 = arith.cmpi eq, %x, %c0 : index
          scf.if %x0 {
            air.dma_memcpy_nd (%la[] [] [], %fa[%o] [%c16] [%one])
                : (memref<16xbf16, 2>, memref<1024xbf16>)
          }
""" + TAIL


def test_broadcast_to_cores_that_dont_read():
    with _raises("broadcast differ"):
        _lower(FIRST_COLUMN)


# Each core writes three tiles, 256 apart: three tasks a column, paced in
# the cores' order, the columns in turn.
THREE_WRITES = HEAD + """
          %fc = memref.reinterpret_cast %hc to offset: [0], sizes: [1024], strides: [1]
              : memref<*xf32> to memref<1024xf32>
          %c3 = arith.constant 3 : index
          %c256 = arith.constant 256 : index
          scf.for %k = %c0 to %c3 step %one {
            %xo = arith.muli %x, %c16 : index
            %yo = arith.muli %y, %c8 : index
            %ko = arith.muli %k, %c256 : index
            %xy = arith.addi %xo, %yo : index
            %o = arith.addi %xy, %ko : index
            air.dma_memcpy_nd (%fc[%o] [%c8] [%one], %lc[] [] [])
                : (memref<1024xf32>, memref<8xf32, 2>)
          }
""" + TAIL


def test_paced_writes():
    text, _ = _lower(THREE_WRITES)
    ops = [(i, o[-1]) for _, _, i, o, _, _ in _shim_ops(text)]
    assert ops == [(x, 16 * x + 256 * k) for k in range(3) for x in range(2)], ops


def test_herds_of_different_names():
    # one herd reads A per row, another per column: two streams, but the
    # herds would run side by side
    col = _herd("g", "%c2").replace("%o = arith.muli %y", "%o = arith.muli %x")
    with _raises("different names"):
        _lower(ONE_ARG.replace("%BODY%", _herd("h", "%c2") + col))


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print(f"PASS {name}")
