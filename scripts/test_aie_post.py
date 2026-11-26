#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""A core kernel's functions after its schedule (backend/aie_post.py), on
small modules written by hand:

  intrinsics  llvm.func and llvm.call of aie2p intrinsics become func ops
  rounding    a herd starts by setting conv_even rounding when any op sets
              it, and only then
  names       an outlined function takes its loop's name; different ones
              of one name are told apart; all are never inlined
  calls       an execute_region left around an outlined call goes
  constants   an argument every call passes the same constant becomes the
              function's own; one the calls pass differently stays
  streams     a stream's state every call takes from one fifo_start at one
              offset: the buffer passed instead, the stream started inside
  restrict    llvm.noalias on a restrict function's buffer arguments when no
              other argument reaches the same buffer at any call, and only
              on restrict functions
  finish      only a module aie_ops marked: its marked loops outlined, one
              function for identical ones, its k-loops unrolled by 4

Needs no NPU.

    python3 scripts/test_aie_post.py
"""

import sys

try:
    from air.ir import Context, Location, Module
    from triton.backends.amd_triton_npu import aie_post as P
except ImportError as e:
    print(f"SKIP: {e}")
    sys.exit(77)

failures = 0


def check(ok, what):
    global failures
    print(f"  {'ok  ' if ok else 'FAIL'} {what}")
    failures += not ok


def run(text, step, *args):
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, Location.unknown():
        m = Module.parse(text)
        out = step(m, *args)
        return str(m), out


def test_intrinsics():
    print("intrinsics:")
    text = """
module {
  llvm.func @llvm.aie2p.v32accfloat.to.v32bf16(vector<32xf32>) -> vector<32xbf16>
  llvm.func @llvm.aie2p.tanh(vector<16xf32>) -> vector<16xbf16>
  func.func @k(%x: vector<32xf32>) -> vector<32xbf16> {
    %y = llvm.call @llvm.aie2p.v32accfloat.to.v32bf16(%x) : (vector<32xf32>) -> vector<32xbf16>
    return %y : vector<32xbf16>
  }
}"""
    out, _ = run(text, P._intrinsics_as_func)
    check(
        "func.func private @llvm.aie2p.v32accfloat.to.v32bf16(vector<32xf32>) -> vector<32xbf16>"
        in out,
        "the declaration a func.func",
    )
    check(
        "call @llvm.aie2p.v32accfloat.to.v32bf16(" in out and "llvm.call" not in out,
        "the call a func.call",
    )
    check("llvm.aie2p.tanh" not in out, "a declaration nothing calls goes")


def herd(body):
    return f"""
module {{
  func.func private @llvm.aie2p.set.ctrl.reg(i32, i32)
  func.func @k() {{
    %c1 = arith.constant 1 : index
    air.herd tile (%tx, %ty) in (%sx=%c1, %sy=%c1) {{
{body}
    }}
    return
  }}
}}"""


def test_rounding():
    print("rounding:")
    sets = """
      %r = arith.constant 1 : i32
      %m = arith.constant 12 : i32
      func.call @llvm.aie2p.set.ctrl.reg(%r, %m) : (i32, i32) -> ()"""
    out, _ = run(herd(sets), P._rounding_at_herd_start)
    first = out.split("air.herd", 1)[1].split("\n")[1:4]
    check(
        any("set.ctrl.reg" in ln for ln in first) and out.count("set.ctrl.reg(") == 3,
        "set first in the herd",
    )
    out, _ = run(herd("      %z = arith.constant 0 : i32"), P._rounding_at_herd_start)
    check(out.count("set.ctrl.reg(") == 1, "no op sets it: the herd does not either")


def outlined(fn_attrs, calls):
    """Two functions outlined from loops marked `fn_attrs` (one each), and a
    kernel whose body is `calls`."""
    fns = []
    for i, a in enumerate(fn_attrs):
        name = "aie_fn" if i == 0 else f"aie_fn_{i - 1}"
        fns.append(f"""
  func.func private @{name}(%a: memref<64xf32>, %b: memref<64xf32>) {{
    %c0 = arith.constant 0 : index
    %c{i + 1} = arith.constant {i + 1} : index
    scf.for %i = %c0 to %c{i + 1} step %c{i + 1} {{
      %v = memref.load %a[%i] : memref<64xf32>
      memref.store %v, %b[%i] : memref<64xf32>
    }} {{{a}}}
    return
  }}""")
    return "module {" + "".join(fns) + f"""
  func.func @k() {{
{calls}
    return
  }}
}}"""


def test_names():
    print("names:")
    calls = """    %x = memref.alloc() : memref<64xf32>
    %y = memref.alloc() : memref<64xf32>
    func.call @aie_fn(%x, %y) : (memref<64xf32>, memref<64xf32>) -> ()
    func.call @aie_fn_0(%x, %y) : (memref<64xf32>, memref<64xf32>) -> ()"""
    text = outlined(
        ['aie.fn = "mm", aie.fnloop', 'aie.fn = "mm", aie.fnloop, aie.restrict'], calls
    )
    out, restrict = run(text, P._name_functions)
    check(
        "@mm(" in out and "@mm_0(" in out and "aie_fn" not in out,
        "named mm and mm_0, calls too",
    )
    check(restrict == {"mm_0"}, f"the restrict one noted: {restrict}")
    check(out.count("llvm.no_inline") == 2, "both never inlined")


def test_unwrap():
    print("calls:")
    text = """
module {
  func.func private @f() {
    return
  }
  func.func @k() {
    scf.execute_region {
      func.call @f() : () -> ()
      scf.yield
    }
    return
  }
}"""
    out, _ = run(text, P._unwrap_calls)
    check(
        "execute_region" not in out and "call @f()" in out,
        "the region gone, the call left",
    )


def test_constants():
    print("constants:")
    text = """
module {
  func.func private @f(%a: i32, %b: i32, %m: memref<4xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    memref.store %a, %m[%c0] : memref<4xi32>
    memref.store %b, %m[%c1] : memref<4xi32>
    return
  }
  func.func @k(%n: i32) {
    %m = memref.alloc() : memref<4xi32>
    %c5 = arith.constant 5 : i32
    %d5 = arith.constant 5 : i32
    %c7 = arith.constant 7 : i32
    func.call @f(%c5, %c7, %m) : (i32, i32, memref<4xi32>) -> ()
    func.call @f(%d5, %n, %m) : (i32, i32, memref<4xi32>) -> ()
    return
  }
}"""
    out, _ = run(text, P._sink_constant_arguments)
    check(
        "func.func private @f(%arg0: i32, %arg1: memref<4xi32>)" in out,
        "the argument passed 5 everywhere gone",
    )
    check(
        "arith.constant 5 : i32" in out.split("func.func @k")[0],
        "5 made in the function",
    )
    check(
        out.count("call @f(") == 2 and "(i32, memref<4xi32>) -> ()" in out,
        "the calls pass the rest",
    )


def streams(offsets):
    calls = "\n".join(
        f"""    %o{i} = arith.constant {o} : index
    %p{i}, %s{i}, %c{i} = aievec.fifo_start %m[%o{i}] : memref<9216xi8> -> vector<32xi32>
    func.call @f(%p{i}, %s{i}, %c{i}, %n) : (!llvm.ptr, vector<32xi32>, i32, i32) -> ()"""
        for i, o in enumerate(offsets)
    )
    return f"""
module {{
  func.func private @f(%p: !llvm.ptr, %s: vector<32xi32>, %c: i32, %n: i32) {{
    %v:5 = aievec.fifo_ld_pop_bfp16 %p, %s, %c : vector<32xi32> -> vector<32xi32>, vector<64xi8>, vector<8xi8>
    return
  }}
  func.func @k(%m: memref<9216xi8>, %n: i32) {{
{calls}
    return
  }}
}}"""


def test_streams():
    print("streams:")
    out, _ = run(streams([0, 0]), P._sink_stream_starts)
    head = out.split("func.func @k")[0]
    check(
        "func.func private @f(%arg0: memref<9216xi8>, %arg1: i32)" in head,
        "the buffer in place of the state, the rest after it",
    )
    check("aievec.fifo_start %arg0[" in head, "the stream started in the function")
    check(out.count("call @f(%arg0, %arg1)") == 2, "the calls pass the buffer")
    out, _ = run(streams([0, 72]), P._sink_stream_starts)
    check("@f(%arg0: !llvm.ptr" in out, "starts at different offsets: unchanged")


def restrict_module(same):
    second = "%x" if same else "%y"
    return f"""
module {{
  func.func private @f(%a: memref<64xf32>, %b: memref<64xf32>) {{
    return
  }}
  func.func @k() {{
    %x = memref.alloc() : memref<64xf32>
    %y = memref.alloc() : memref<64xf32>
    func.call @f(%x, {second}) : (memref<64xf32>, memref<64xf32>) -> ()
    return
  }}
}}"""


def test_restrict():
    print("restrict:")
    out, _ = run(restrict_module(False), P._restrict, {"f"})
    check(out.count("llvm.noalias") == 2, "two buffers: both arguments noalias")
    out, _ = run(restrict_module(True), P._restrict, {"f"})
    check("llvm.noalias" not in out, "one buffer passed twice: neither")
    out, _ = run(restrict_module(False), P._restrict, set())
    check("llvm.noalias" not in out, "not restrict: none")


def test_more():
    print("more:")
    text = """
module {
  func.func private @f(%a: i32, %m: memref<4xi32>) {
    %c0 = arith.constant 0 : index
    memref.store %a, %m[%c0] : memref<4xi32>
    return
  }
  func.func @k() {
    %m = memref.alloc() : memref<4xi32>
    %c5 = arith.constant 5 : i32
    %c7 = arith.constant 7 : i32
    func.call @f(%c5, %m) : (i32, memref<4xi32>) -> ()
    func.call @f(%c7, %m) : (i32, memref<4xi32>) -> ()
    return
  }
}"""
    out, _ = run(text, P._sink_constant_arguments)
    check(
        "@f(%arg0: i32, %arg1: memref<4xi32>)" in out,
        "calls pass 5 and 7: the argument stays",
    )

    text = """
module {
  func.func private @f(%a: i32) -> i32 {
    return %a : i32
  }
  func.func @k() -> i32 {
    %c5 = arith.constant 5 : i32
    %r = func.call @f(%c5) : (i32) -> i32
    return %r : i32
  }
}"""
    out, _ = run(text, P._sink_constant_arguments)
    check(
        "@f(%arg0: i32) -> i32" in out,
        "a function whose calls return a value: left as it is",
    )

    def views(same):
        src = "%x" if same else "%y"
        return f"""
module {{
  func.func private @f(%a: memref<32xf32, strided<[1]>>, %b: memref<32xf32, strided<[1], offset: 32>>) {{
    return
  }}
  func.func @k() {{
    %x = memref.alloc() : memref<64xf32>
    %y = memref.alloc() : memref<64xf32>
    %va = memref.subview %x[0] [32] [1] : memref<64xf32> to memref<32xf32, strided<[1]>>
    %vb = memref.subview {src}[32] [32] [1] : memref<64xf32> to memref<32xf32, strided<[1], offset: 32>>
    func.call @f(%va, %vb) : (memref<32xf32, strided<[1]>>, memref<32xf32, strided<[1], offset: 32>>) -> ()
    return
  }}
}}"""

    out, _ = run(views(True), P._restrict, {"f"})
    check(
        "llvm.noalias" not in out, "restrict: two views of one buffer: neither noalias"
    )
    out, _ = run(views(False), P._restrict, {"f"})
    check(
        out.count("llvm.noalias") == 2, "restrict: views of two buffers: both noalias"
    )

    text = streams([0, 0]).replace("@f(%p: !llvm.ptr", "@f(%z: i32, %p: !llvm.ptr")
    text = text.replace("func.call @f(%p", "func.call @f(%n, %p").replace(
        ": (!llvm.ptr, vector<32xi32>, i32, i32) -> ()",
        ": (i32, !llvm.ptr, vector<32xi32>, i32, i32) -> ()",
    )
    out, _ = run(text, P._sink_stream_starts)
    head = out.split("func.func @k")[0]
    check(
        "@f(%arg0: i32, %arg1: memref<9216xi8>, %arg2: i32)" in head,
        "an argument before the state keeps its place",
    )

    text = """
module {
  func.func private @f() {
    return
  }
  func.func @k() {
    scf.execute_region no_inline {
      func.call @f() : () -> ()
      scf.yield
    }
    return
  }
}"""
    out, _ = run(text, P._unwrap_calls)
    check("execute_region" in out, "a no_inline region stays")

    # a kernel aie_ops did not lower, whatever ops its own schedule made
    # and however its loops are marked: left as it is
    text = _HERD.replace("module attributes {aie.code} {", "module {", 1)
    ctx = Context()
    with ctx, Location.unknown():
        before = str(Module.parse(text))
    out, _ = run(text, P.finish)
    check(out == before, "a module aie_ops did not mark: unchanged")


def test_bodies():
    print("bodies:")
    text = """
module {
  func.func private @attn_init(%m: memref<64xf32>) {
    %c0 = arith.constant 0 : index
    %z = arith.constant 0.0 : f32
    %c1 = arith.constant 1 : index
    %y = arith.constant 1.0 : f32
    scf.execute_region no_inline {
      memref.store %z, %m[%c0] : memref<64xf32>
      memref.store %y, %m[%c1] : memref<64xf32>
      scf.yield
    } {aie.fn = "attn_init", aie.fnloop}
    return
  }
  func.func private @other(%m: memref<64xf32>) {
    scf.execute_region no_inline {
      scf.yield
    }
    return
  }
}"""
    out, _ = run(text, P._unwrap_bodies)
    head = out.split("@other")[0]
    first, second = head.find("memref.store %cst,"), head.find("memref.store %cst_0,")
    check(
        "execute_region" not in head and 0 <= first < second,
        "a begin_function region in a function: its ops in its place, in order",
    )
    check(out.count("execute_region") == 1, "a region not marked stays")


def test_stream_skips():
    print("streams left as they are:")
    text = streams([0, 0]).replace(
        "  func.func @k(%m: memref<9216xi8>, %n: i32) {",
        "  func.func @k(%m: memref<9216xi8>, %m2: memref<4608xi8>, %n: i32) {",
    )
    text = text.replace(
        "%p1, %s1, %c1 = aievec.fifo_start %m[%o1] : memref<9216xi8>",
        "%p1, %s1, %c1 = aievec.fifo_start %m2[%o1] : memref<4608xi8>",
    )
    out, _ = run(text, P._sink_stream_starts)
    check("@f(%arg0: !llvm.ptr" in out, "starts on buffers of two types: unchanged")
    text = streams([0, 0]).replace(
        "%o0 = arith.constant 0 : index", "%o0 = arith.index_cast %n : i32 to index"
    )
    out, _ = run(text, P._sink_stream_starts)
    check("@f(%arg0: !llvm.ptr" in out, "a start at an offset not known: unchanged")


def test_restrict_unknown():
    print("restrict, a buffer not known:")
    text = """
module {
  memref.global "private" @g : memref<64xf32> = uninitialized
  func.func private @f(%a: memref<64xf32>, %b: memref<64xf32>) {
    return
  }
  func.func @k() {
    %x = memref.alloc() : memref<64xf32>
    %y = memref.get_global @g : memref<64xf32>
    func.call @f(%x, %y) : (memref<64xf32>, memref<64xf32>) -> ()
    return
  }
}"""
    out, _ = run(text, P._restrict, {"f"})
    check(
        "llvm.noalias" not in out,
        "an argument whose buffer is not known: no argument noalias",
    )


_HERD = """
module attributes {aie.code} {
  func.func @k(%a: memref<64xf32, 2>, %b: memref<64xf32, 2>) {
    %c1 = arith.constant 1 : index
    air.herd tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%x=%a, %y=%b) : memref<64xf32, 2>, memref<64xf32, 2> {
      %c0 = arith.constant 0 : index
      %one = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c64 = arith.constant 64 : index
      %z = arith.constant 0.000000e+00 : f32
      scf.for %i = %c0 to %c64 step %c16 {
        %v = vector.transfer_read %x[%i], %z {in_bounds = [true]} : memref<64xf32, 2>, vector<16xf32>
        vector.transfer_write %v, %y[%i] {in_bounds = [true]} : vector<16xf32>, memref<64xf32, 2>
      } {aie.fn = "copy", aie.fnloop}
      scf.for %i = %c0 to %c64 step %c16 {
        %v = vector.transfer_read %x[%i], %z {in_bounds = [true]} : memref<64xf32, 2>, vector<16xf32>
        vector.transfer_write %v, %y[%i] {in_bounds = [true]} : vector<16xf32>, memref<64xf32, 2>
      } {aie.fn = "copy", aie.fnloop}
      scf.for %i = %c0 to %c4 step %one {
        %acc = vector.transfer_read %y[%c16], %z {in_bounds = [true]} : memref<64xf32, 2>, vector<16xf32>
        %v = vector.transfer_read %x[%c0], %z {in_bounds = [true]} : memref<64xf32, 2>, vector<16xf32>
        %s = arith.addf %acc, %v : vector<16xf32>
        vector.transfer_write %s, %y[%c16] {in_bounds = [true]} : vector<16xf32>, memref<64xf32, 2>
      } {aie.k_loop}
    }
    return
  }
}"""


def test_finish():
    print("finish:")
    out, _ = run(_HERD, P.finish)
    check("aie.code" not in out, "the mark taken off")
    defs = [ln for ln in out.splitlines() if "func.func private @copy" in ln]
    check(
        len(defs) == 1, f"two identical marked loops: one function, named: {len(defs)}"
    )
    check(
        out.count("call @copy(") == 2 and "aie_fn" not in out,
        "called in place of each loop",
    )
    check("llvm.no_inline" in (defs[0] if defs else ""), "never inlined")
    import re

    lines = out.splitlines()
    kl = next((ln for ln in lines if "aie.k_loop" in ln), "")
    alias = re.search(r"loop_annotation = (#\w+)", kl)
    aliases = dict(ln.split(" = ", 1) for ln in lines if re.match(r"#\w+ = ", ln))
    defn = aliases.get(alias[1], "") if alias else ""
    for _ in range(4):
        defn = re.sub(r"#\w+\b(?!<)", lambda m: aliases.get(m[0], m[0]), defn)
    check("count = 4 : i32" in defn, f"the k-loop unrolled by 4: {defn}")
    body = (
        out.split("air.herd", 1)[1].split("aie.k_loop", 1)[0].rsplit("scf.for", 1)[-1]
    )
    check(
        "iter_args" in body and "vector.transfer_write" not in body,
        "a value each trip of the k-loop reads and writes back: carried, written after it",
    )


def main():
    test_intrinsics()
    test_rounding()
    test_names()
    test_unwrap()
    test_constants()
    test_streams()
    test_restrict()
    test_more()
    test_bodies()
    test_stream_skips()
    test_restrict_unknown()
    test_finish()
    if failures:
        print(f"{failures} failed")
        sys.exit(1)
    print("all passed")


if __name__ == "__main__":
    main()
