# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""`fused_device`'s rewrite of a kernel's func before it joins the device.

Its int arguments are bound to the op's values, and the scalars nothing reads
any more are erased from the func and its type, so the op's runtime sequence
takes only buffers and the scalars it still uses. Its buffers are then moved
to the device buffers they bind to, unused buffer arguments taking the places
around them. No NPU and no Triton compile: the kernel is a stand-in for its
parameter list.
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from air.ir import ArrayAttr, Context, FunctionType, Module, TypeAttr  # noqa: E402

import pytest  # noqa: E402

from fused_device import (  # noqa: E402
    FusedDevice,
    _bind_ints,
    _drop_unused_scalars,
    _entry,
    _ops_named,
    _set_multiplexing,
    _shift_buffers,
)

IR = """
func.func private @helper(%x: i32) -> i32 {
  return %x : i32
}
func.func @kern(%a: memref<16xbf16> {tt.divisibility = 16 : i32}, %n: i32,
                %k: i32, %s: f32, %u: f32 {tt.u}, %pid: i32,
                %out: memref<16xbf16> {tt.out}) {
  %0 = arith.addi %n, %k : i32
  %1 = arith.addi %0, %pid : i32
  %2 = func.call @helper(%1) : (i32) -> i32
  %3 = arith.mulf %s, %s : f32
  return
}
"""


def _kernel(*names, constexpr=()):
    params = [SimpleNamespace(name=n, is_constexpr=n in constexpr) for n in names]
    return SimpleNamespace(params=params)


def _signature(fn):
    block = fn.regions[0].blocks[0]
    ftype = FunctionType(TypeAttr(fn.attributes["function_type"]).value)
    return [str(a.type) for a in block.arguments], [str(t) for t in ftype.inputs]


def _arg_attrs(fn):
    return [str(a) for a in ArrayAttr(fn.attributes["arg_attrs"])]


def test_unread_scalars_erased():
    with Context():
        m = Module.parse(IR)
        fn = _entry(m)
        assert str(fn.attributes["sym_name"]) == '"kern"'
        _drop_unused_scalars(fn)
        args, inputs = _signature(fn)
        # %u is the one scalar nothing reads.
        want = ["memref<16xbf16>", "i32", "i32", "f32", "i32", "memref<16xbf16>"]
        assert args == want, args
        assert inputs == want, inputs
        # Each kept argument keeps its own attributes.
        attrs = _arg_attrs(fn)
        assert attrs[0] == "{tt.divisibility = 16 : i32}", attrs
        assert attrs[1:5] == ["{}"] * 4 and attrs[5] == "{tt.out}", attrs
        m.operation.verify()


def test_bound_ints_erased():
    """Bound ints and floats leave only the buffers and the unbound scalar."""
    with Context():
        m = Module.parse(IR)
        # a, n, BLOCK (constexpr, not in the func), k, s, u, then the grid's %pid
        kernel = _kernel("a", "n", "BLOCK", "k", "s", "u", constexpr=("BLOCK",))
        _bind_ints(m, kernel, (None, 128, 64, 256, 0.5, 1.0))
        fn = _entry(m)
        _drop_unused_scalars(fn)
        args, inputs = _signature(fn)
        want = ["memref<16xbf16>", "i32", "memref<16xbf16>"]
        assert args == want, args
        assert inputs == want, inputs
        assert _arg_attrs(fn) == ["{tt.divisibility = 16 : i32}", "{}", "{tt.out}"]
        m.operation.verify()
        text = str(m)
        assert "arith.constant 128 : i32" in text and "arith.constant 256" in text


def test_kept_ints_stay_arguments():
    """An int named in `keep` stays an argument of the func, unbound."""
    with Context():
        m = Module.parse(IR)
        kernel = _kernel("a", "n", "BLOCK", "k", "s", "u", constexpr=("BLOCK",))
        _bind_ints(m, kernel, (None, 128, 64, 256, 0.5, 1.0), keep=("n",))
        fn = _entry(m)
        _drop_unused_scalars(fn)
        args, _ = _signature(fn)
        assert args == ["memref<16xbf16>", "i32", "i32", "memref<16xbf16>"], args
        m.operation.verify()


def test_memref_kept_unread():
    with Context():
        m = Module.parse("func.func @k(%a: memref<4xf32>, %n: i32) {\n  return\n}\n")
        fn = _entry(m)
        _drop_unused_scalars(fn)
        assert _signature(fn) == (["memref<4xf32>"], ["memref<4xf32>"])
        m.operation.verify()


BUFFERS = """
func.func @k(%a: memref<*xbf16> {tt.divisibility = 16 : i32},
             %b: memref<*xbf16> {tt.b}%SCALAR%) {
  return
}
"""


def test_shift_buffers_only():
    """A func of buffers alone gets the unused buffers behind its own too."""
    with Context():
        m = Module.parse(BUFFERS.replace("%SCALAR%", ""))
        _shift_buffers(m, 1, after=2)
        fn = _entry(m)
        args, inputs = _signature(fn)
        assert args == inputs == ["memref<*xbf16>"] * 5
        assert _arg_attrs(fn) == [
            "{}",
            "{tt.divisibility = 16 : i32}",
            "{tt.b}",
            "{}",
            "{}",
        ]
        m.operation.verify()


def test_shift_buffers_ahead_of_scalars():
    with Context():
        m = Module.parse(BUFFERS.replace("%SCALAR%", ", %n: i32 {tt.n}"))
        _shift_buffers(m, 0, after=1)
        fn = _entry(m)
        args, inputs = _signature(fn)
        assert args == inputs == ["memref<*xbf16>"] * 3 + ["i32"]
        assert _arg_attrs(fn)[2:] == ["{}", "{tt.n}"]
        m.operation.verify()


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print(f"PASS {name}")


def test_unknown_kept_name_rejected():
    with Context():
        m = Module.parse(IR)
        kernel = _kernel("a", "n", "k", "s", "u", "pid", "out")
        with pytest.raises(ValueError, match="nn"):
            _bind_ints(m, kernel, (None, 1, 2, 0.5, 0.25, 0, None), keep=("nn",))


def test_shift_ranked_buffers():
    # Ranked buffers count as buffers too, and the filler takes the first
    # buffer's type.
    with Context():
        m = Module.parse(
            "func.func @k(%a: memref<16xbf16>, %b: memref<8xi32>, %n: i32) {\n"
            "  return\n}"
        )
        _shift_buffers(m, 1, after=1)
        args, _ = _signature(_entry(m))
        assert args == [
            "memref<16xbf16>",
            "memref<16xbf16>",
            "memref<8xi32>",
            "memref<16xbf16>",
            "i32",
        ]


def test_shift_buffers_after_scalar_rejected():
    with Context():
        m = Module.parse("func.func @k(%n: i32, %a: memref<16xbf16>) {\n  return\n}")
        with pytest.raises(ValueError):
            _shift_buffers(m, 1)


def test_multiplexing_attribute():
    with Context():
        m = Module.parse(IR)
        _set_multiplexing(m, ("L2",))
        attr = _entry(m).attributes["air.channel_multiplexing"]
        assert str(attr) == '["L2"]'


# An op: `nbuf` buffers, one L2 buffer passed to one herd `cols` columns wide,
# at column `at` if it is placed already.
def _op(nbuf, cols=1, at=None):
    args = ", ".join(f"%b{i}: memref<*xbf16>" for i in range(nbuf))
    loc = "" if at is None else f" attributes {{x_loc = {at} : i64, y_loc = 2 : i64}}"
    return f"""
func.func @k({args}) {{
  %c1 = arith.constant 1 : index
  air.launch (%x) in (%sx=%c1) {{
    air.segment @s {{
      %c1s = arith.constant 1 : index
      %cw = arith.constant {cols} : index
      %l2 = memref.alloc() : memref<64xbf16, 1>
      air.herd @h tile (%tx, %ty) in (%hx=%cw, %hy=%c1s) args(%l=%l2) : memref<64xbf16, 1>{loc} {{
      }}
      memref.dealloc %l2 : memref<64xbf16, 1>
    }}
  }}
  return
}}
"""


def _dev(*ops):
    dev = FusedDevice("dev", "/tmp")
    dev.ops = list(ops)
    return dev


def test_module_text_pads_pins_and_places():
    # Every op's func takes the device's buffers; a pinned op's L2 buffers sit
    # in its column; ops of different keys get columns side by side.
    dev = _dev(("a", "ka", _op(2)), ("b", "kb", _op(3)))
    dev.pinned.add("ka")
    with Context():
        m = Module.parse(dev.module_text())
        funcs = {
            str(f.attributes["sym_name"]): f
            for f in _ops_named(m.operation, "func.func")
        }
        for f in funcs.values():
            args, _ = _signature(f)
            assert args == ["memref<*xbf16>"] * 3
        allocs = {name: _ops_named(f, "memref.alloc")[0] for name, f in funcs.items()}
        assert "air.no_split" in allocs['"a"'].attributes
        assert "air.memtile_col" in allocs['"a"'].attributes
        assert "air.no_split" not in allocs['"b"'].attributes
        cols = sorted(
            str(h.attributes["x_loc"]) for h in _ops_named(m.operation, "air.herd")
        )
        assert len(set(cols)) == 2


def test_pin_l2_needs_one_column():
    dev = _dev(("a", "ka", _op(2, cols=2)))
    dev.pinned.add("ka")
    with pytest.raises(ValueError, match="one column"):
        dev.module_text()


def test_pin_l2_buffer_of_one_column():
    # Two herds side by side, each pinning the L2 buffer it gets to its own
    # column: one buffer passed to both cannot be in both memtiles.
    op = _op(1).replace(
        "      memref.dealloc",
        "      air.herd @g tile (%tx, %ty) in (%hx=%c1s, %hy=%c1s) args(%l=%l2)"
        " : memref<64xbf16, 1> {\n      }\n      memref.dealloc",
    )
    dev = _dev(("a", "ka", op))
    dev.pinned.add("ka")
    with pytest.raises(ValueError, match="two columns"):
        dev.module_text()


def test_prebuilt_op_with_other_buffers_rejected():
    dev = _dev(("a", "ka", _op(3)), ("p", None, _op(2, at=5)))
    with pytest.raises(ValueError, match="other buffers"):
        dev.module_text()
