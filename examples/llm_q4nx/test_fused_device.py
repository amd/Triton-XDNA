# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""`fused_device`'s rewrite of a kernel's func before it joins the device.

Its int arguments are bound to the op's values, and the scalars nothing reads
any more are erased from the func and its type, so the op's runtime sequence
takes only buffers and the scalars it still uses. No NPU and no Triton compile:
the kernel is a stand-in for its parameter list.
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from air.ir import ArrayAttr, Context, FunctionType, Module, TypeAttr  # noqa: E402

from fused_device import _bind_ints, _drop_unused_scalars, _entry  # noqa: E402

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
    params = [SimpleNamespace(is_constexpr=n in constexpr) for n in names]
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


def test_memref_kept_unread():
    with Context():
        m = Module.parse("func.func @k(%a: memref<4xf32>, %n: i32) {\n  return\n}\n")
        fn = _entry(m)
        _drop_unused_scalars(fn)
        assert _signature(fn) == (["memref<4xf32>"], ["memref<4xf32>"])
        m.operation.verify()


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print(f"PASS {name}")
