#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""The lowering of tl.extra.air.aie ops (backend/aie_ops.py), on small
kernels written by hand as triton-shared gives them:

  clang        each op calls the aie2p intrinsics, with the same constant
               arguments, that Peano's clang emits for its aie_api call
               (skipped without Peano and aie_api's headers)
  streams      a stream's pops fill the FIFO unless the reads since the last
               fill are known to be 1 to 7; a loop that reads a stream started
               above it carries its state
  buffers      load_v / store_v on a buffer of their type, and on an int8
               buffer read as another type
  accumulator  an accumulator a loop reads and writes whole is carried as a
               vector
  marks        function, its restrict, and unroll mark their loop; unroll(4)
               makes a k-loop only of a loop that reads a stream
  pointers     a pointer made an int for anything but an aie op is an error
  wiring       each op's operands reach the intrinsic arguments they are for
  templates    each template op calls what clang emits for its aie_api call,
               its operands' lanes in order; a template that does more with
               its output than store to it from byte 0 on is an error, and so
               is an op whose operands do not fill the template's arguments
  spans        begin_function / end_function wrap the code between them
  checks       what a stream, a buffer, an int operand, an op's lanes and a
               mark must be, each an error otherwise
  constants    a splat constant or a fill operand is not read from memory
  mark         a kernel whose aie ops are lowered is marked for aie_post

Needs no NPU.

    python3 scripts/test_aie_ops.py
"""

import glob
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter

try:
    from air.ir import Context, IntegerAttr, Location, Module
    from triton.backends.amd_triton_npu import aie_ops
except ImportError as e:
    print(f"SKIP: {e}")
    sys.exit(77)

failures = 0


def check(ok, what):
    global failures
    print(f"  {'ok  ' if ok else 'FAIL'} {what}")
    failures += not ok


_ARGS = {"b8": "i8", "f32": "f32", "bf": "bf16", "i16": "i16"}


class Kernel:
    """A kernel function, its buffers %b8 (int8), %f32, %bf (bf16) and %i16
    as triton-shared passes them, whose body is aie ops."""

    def __init__(self):
        self.lines, self.n = [], 0

    def tmp(self, p="v"):
        self.n += 1
        return f"%{p}_{self.n}"

    def emit(self, s):
        self.lines.append(s)

    def int8(self, value, rows=8):
        """An [rows, 8] i64 operand of one int (a constant or an SSA i64)."""
        if isinstance(value, int):
            c = self.tmp("c")
            self.emit(f"{c} = arith.constant {value} : i64")
            value = c
        e, f = self.tmp("e"), self.tmp("f")
        self.emit(f"{e} = tensor.empty() : tensor<{rows}x8xi64>")
        self.emit(
            f"{f} = linalg.fill ins({value} : i64) outs({e} : tensor<{rows}x8xi64>) -> tensor<{rows}x8xi64>"
        )
        return f

    def ptr8(self, buf, rows=8, offset=None):
        """The pointer to buffer `buf` (`offset` elements in), made an int,
        as an [rows, 8] i64 operand (triton-shared's ptrtoint of the
        argument)."""
        et = _ARGS[buf]
        c, s, p, i = self.tmp("pc"), self.tmp("ps"), self.tmp("pp"), self.tmp("pi")
        if offset is None:
            mt = f"memref<1x{et}>"
            self.emit(f"{c} = memref.cast %{buf} : memref<*x{et}> to {mt}")
        else:
            mt = f"memref<1x{et}, strided<[1], offset: {offset}>>"
            self.emit(
                f"{c} = memref.reinterpret_cast %{buf} to offset: [{offset}], sizes: [1], strides: [1]"
                f" : memref<*x{et}> to {mt}"
            )
        sp = mt[:-1] + ", #tptr.default_memory_space>"
        self.emit(f"{s} = memref.memory_space_cast {c} : {mt} to {sp}")
        self.emit(f"{p} = ptr.to_ptr {s} : {sp} -> <#tptr.default_memory_space>")
        self.emit(f"{i} = tptr.ptrtoint {p} : <#tptr.default_memory_space> to i64")
        return self.int8(i, rows)

    def broadcast(self, v, et, rows=8):
        """The [1, 8] tensor v broadcast to [rows, 8], as Triton makes an
        operand of an op of [rows, 8] operands."""
        e, b = self.tmp("be"), self.tmp("bb")
        self.emit(f"{e} = tensor.empty() : tensor<{rows}x8x{et}>")
        self.emit(
            f"{b} = linalg.generic {{indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],"
            f' iterator_types = ["parallel", "parallel"]}} ins({v} : tensor<1x8x{et}>) outs({e} : tensor<{rows}x8x{et}>) {{\n'
            f"    ^bb0(%in: {et}, %out: {et}):\n      linalg.yield %in : {et}\n    }} -> tensor<{rows}x8x{et}>"
        )
        return b

    def shape(self, rows=8):
        """The op's extra operand of its shape (the lowering does not read
        it)."""
        r = self.tmp("r")
        self.emit(f"{r} = arith.constant dense<1> : tensor<{rows}x8xi32>")
        return r

    def op(self, name, operands, types, result):
        r = self.tmp()
        cons = "=r" + ",r" * len(operands)
        self.emit(
            f'{r} = tt.elementwise_inline_asm "aie.{name}" {{constraints = "{cons}", packed_element = 1 : i32, pure = false}}'
            f" {', '.join(operands)} : {', '.join(types)} -> {result}"
        )
        return r

    def load(self, buf, off, lanes, et):
        rows = lanes // 8
        return self.op(
            "load_v",
            [self.ptr8(buf, rows), self.int8(off, rows)],
            [f"tensor<{rows}x8xi64>"] * 2,
            f"tensor<{rows}x8x{et}>",
        )

    def store(self, buf, off, v, rows, et):
        return self.op(
            "store_v",
            [self.ptr8(buf, rows), self.int8(off, rows), v],
            [f"tensor<{rows}x8xi64>"] * 2 + [f"tensor<{rows}x8x{et}>"],
            f"tensor<{rows}x8xi32>",
        )

    def text(self):
        args = ", ".join(f"%{n}: memref<*x{t}>" for n, t in _ARGS.items())
        body = "\n    ".join(self.lines)
        return f"module {{\n  func.func @k({args}, %n: i32) {{\n    {body}\n    return\n  }}\n}}\n"


def lower(k):
    return aie_ops.lower(k.text())


def parsed(text):
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    with ctx, Location.unknown():
        return Module.parse(text), ctx


_PARSED = {}


def ops_named(text, name):
    """The ops called `name` in `text`, and what keeps them alive. A text is
    parsed once: values of two parses never compare equal."""
    if text not in _PARSED:
        _PARSED[text] = parsed(text)
    m, ctx = _PARSED[text]
    out = []

    def visit(op):
        if op.name == name:
            out.append(op)
        for r in op.regions:
            for b in r.blocks:
                for o in b.operations:
                    visit(o.operation)

    for o in m.body.operations:
        visit(o.operation)
    return out, (m, ctx)


def intrinsic_calls(text):
    """The aie2p intrinsics the lowered `text` calls: (name, the int
    constants among its arguments), as a multiset."""
    calls, keep = ops_named(text, "llvm.call")
    out = Counter()
    for c in calls:
        name = c.attributes["callee"].value
        if not name.startswith("llvm.aie2p."):
            continue
        consts = []
        for v in c.operands:
            d = v.owner
            if (
                hasattr(d, "name")
                and d.name in ("arith.constant", "llvm.mlir.constant")
                and str(v.type).startswith("i")
            ):
                consts.append(IntegerAttr(d.attributes["value"]).value)
        out[name, tuple(consts)] += 1
    return out


def origin(v):
    """Where the vector or tensor v comes from, back through casts, slices
    and transfers: (kernel argument number, element offset of the read), or
    the name of the op it starts at."""
    from air.ir import Block, IntegerAttr

    for _ in range(64):
        if isinstance(v.owner, Block):
            return v.arg_number
        d = v.owner.operation
        if d.name in (
            "vector.bitcast",
            "vector.shape_cast",
            "vector.extract_strided_slice",
            "vector.transfer_read",
            "vector.transfer_write",
        ):
            v = d.operands[0]
        elif d.name == "memref.reinterpret_cast":
            off = d.operands[1].owner.operation
            return origin(d.operands[0]), IntegerAttr(off.attributes["value"]).value
        else:
            return d.name
    return None


def source(v):
    """origin(v), back also through bitcasts, shuffles and calls (their
    first operand)."""
    from air.ir import Block, IntegerAttr

    for _ in range(64):
        if isinstance(v.owner, Block):
            return v.arg_number
        d = v.owner.operation
        if d.name == "memref.reinterpret_cast":
            off = d.operands[1].owner.operation
            return source(d.operands[0]), IntegerAttr(off.attributes["value"]).value
        if not d.name.startswith("vector.") and d.name not in (
            "llvm.bitcast",
            "llvm.shufflevector",
            "llvm.call",
        ):
            return d.name
        v = d.operands[0]
    return None


def slice_offsets(v):
    """The offsets of the extract_strided_slice ops v comes through."""
    out = []
    for _ in range(64):
        d = v.owner
        if not hasattr(d, "operation"):
            return out
        d = d.operation
        if d.name == "vector.extract_strided_slice":
            out.append(int(str(d.attributes["offsets"]).strip("[]").split(":")[0]))
        if not len(d.operands):
            return out
        v = d.operands[0]
    return out


def calls_to(text, name):
    calls, keep = ops_named(text, "llvm.call")
    return [c for c in calls if c.attributes["callee"].value == name], keep


def _peano():
    for d in [os.environ.get("PEANO_INSTALL_DIR", "")] + glob.glob(
        os.path.join(sys.prefix, "lib", "python3*", "site-packages", "llvm-aie")
    ):
        if d and os.path.exists(os.path.join(d, "bin", "clang++")):
            return os.path.join(d, "bin", "clang++")
    return None


def _aie_api():
    import aie

    for d in [
        os.path.join(os.path.dirname(os.path.dirname(aie.__file__)), "..", "include")
    ] + glob.glob(
        os.path.join(
            sys.prefix, "lib", "python3*", "site-packages", "mlir_aie", "include"
        )
    ):
        if os.path.exists(os.path.join(d, "aie_api", "aie.hpp")):
            return os.path.abspath(d)
    return None


def clang_calls(body):
    """The aie2p intrinsics clang -O2 calls for the C++ `body` (a function
    of float *f, bfloat16 *bf, int16 *i16, uint8 *b8, out pointers *o and
    *ob): (name, its i32 constant arguments), as a multiset."""
    src = (
        "#include <aie_api/aie.hpp>\nusing ACC = aie::accum<accfloat, 64>;\n"
        "using ACC32 = aie::accum<accfloat, 32>;\n"
        'extern "C" void t(float *f, bfloat16 *bf, int16 *i16, uint8 *b8, float *o, bfloat16 *ob) {\n'
        + body
        + "\n}\n"
    )
    with tempfile.TemporaryDirectory() as d:
        cc, ll = os.path.join(d, "t.cc"), os.path.join(d, "t.ll")
        open(cc, "w").write(src)
        subprocess.run(
            [
                CLANG,
                "-O2",
                "-std=c++20",
                "--target=aie2p-none-unknown-elf",
                "-DNDEBUG",
                "-Wno-parentheses",
                "-Wno-attributes",
                "-Wno-macro-redefined",
                "-Wno-empty-body",
                "-Wno-deprecated-declarations",
                f"-I{AIE_API}",
                "-D__AIE_API_AIE_ADF_HPP__",
                "-S",
                "-emit-llvm",
                cc,
                "-o",
                ll,
            ],
            check=True,
        )
        text = open(ll).read()
    out = Counter()
    for m in re.finditer(
        r"call [^@\n]*@(llvm\.aie2p\.[\w.]+)\((.*)\)[^)\n]*$", text, re.M
    ):
        consts = tuple(int(x) for x in re.findall(r"\bi32 (-?\d+)\b", m[2]))
        out[m[1], consts] += 1
    return out


try:
    CLANG, AIE_API = _peano(), _aie_api()
except ImportError:
    CLANG, AIE_API = None, None


def same_calls(what, kernel, body):
    if CLANG is None or AIE_API is None:
        print(f"  skip {what}: no Peano clang or aie_api headers")
        return
    ours, theirs = intrinsic_calls(lower(kernel)), clang_calls(body)
    check(
        ours == theirs,
        f"{what}: {sorted(ours.items())} == clang's {sorted(theirs.items())}",
    )


def test_clang():
    print("clang:")
    k = Kernel()
    v = k.load("f32", 0, 32, "f32")
    k.store(
        "bf",
        0,
        k.op(
            "to_bf16",
            [v, k.shape(4)],
            ["tensor<4x8xf32>", "tensor<4x8xi32>"],
            "tensor<4x8xbf16>",
        ),
        4,
        "bf16",
    )
    same_calls(
        "to_bf16 (32 lanes)",
        k,
        "ACC32 a; a.from_vector(aie::load_v<32>(f)); aie::store_v(ob, a.to_vector<bfloat16>());",
    )

    k = Kernel()
    v = k.load("bf", 0, 64, "bf16")
    k.store(
        "f32",
        0,
        k.op(
            "from_vector",
            [v, k.shape()],
            ["tensor<8x8xbf16>", "tensor<8x8xi32>"],
            "tensor<8x8xf32>",
        ),
        8,
        "f32",
    )
    same_calls(
        "from_vector (64 lanes)",
        k,
        "ACC a; a.from_vector(aie::load_v<64>(bf)); aie::store_v(o, a.to_vector<float>());",
    )

    for op in ("mac", "msc"):
        k = Kernel()
        c = k.load("f32", 0, 64, "f32")
        a, b = k.load("bf", 0, 64, "bf16"), k.load("bf", 64, 64, "bf16")
        r = k.op(
            op,
            [c, a, b, k.shape()],
            [
                "tensor<8x8xf32>",
                "tensor<8x8xbf16>",
                "tensor<8x8xbf16>",
                "tensor<8x8xi32>",
            ],
            "tensor<8x8xf32>",
        )
        k.store("f32", 0, r, 8, "f32")
        same_calls(
            f"{op} (64 lanes)",
            k,
            f"ACC c; c.from_vector(aie::load_v<64>(f)); aie::store_v(o, aie::{op}(c, aie::load_v<64>(bf), aie::load_v<64>(bf + 64)).to_vector<float>());",
        )

    k = Kernel()
    a, b = k.load("bf", 0, 32, "bf16"), k.load("bf", 32, 32, "bf16")
    k.store(
        "f32",
        0,
        k.op(
            "mul",
            [a, b, k.shape(4)],
            ["tensor<4x8xbf16>"] * 2 + ["tensor<4x8xi32>"],
            "tensor<4x8xf32>",
        ),
        4,
        "f32",
    )
    same_calls(
        "mul (32 lanes)",
        k,
        "aie::store_v(o, aie::mul(aie::load_v<32>(bf), aie::load_v<32>(bf + 32)).to_vector<float>());",
    )

    k = Kernel()
    a, b = k.load("bf", 0, 32, "bf16"), k.load("bf", 32, 32, "bf16")
    k.store(
        "bf",
        0,
        k.op(
            "add",
            [a, b, k.shape(4)],
            ["tensor<4x8xbf16>"] * 2 + ["tensor<4x8xi32>"],
            "tensor<4x8xbf16>",
        ),
        4,
        "bf16",
    )
    same_calls(
        "add (32 lanes)",
        k,
        "aie::store_v(ob, aie::add(aie::load_v<32>(bf), aie::load_v<32>(bf + 32)));",
    )

    k = Kernel()
    a, b = k.load("bf", 0, 32, "bf16"), k.load("bf", 32, 32, "bf16")
    m = k.op(
        "lt",
        [a, b, k.shape(4)],
        ["tensor<4x8xbf16>"] * 2 + ["tensor<4x8xi32>"],
        "tensor<4x8xi32>",
    )
    r = k.op(
        "select",
        [a, b, m, k.shape(4)],
        ["tensor<4x8xbf16>"] * 2 + ["tensor<4x8xi32>"] * 2,
        "tensor<4x8xbf16>",
    )
    k.store("bf", 0, r, 4, "bf16")
    same_calls(
        "lt, select (32 lanes)",
        k,
        "auto a = aie::load_v<32>(bf); auto b = aie::load_v<32>(bf + 32); aie::store_v(ob, aie::select(a, b, aie::lt(a, b)));",
    )

    for op in ("tanh", "exp2"):
        k = Kernel()
        x = k.load("f32", 0, 32, "f32")
        k.store(
            "bf",
            0,
            k.op(
                op,
                [x, k.shape(4)],
                ["tensor<4x8xf32>", "tensor<4x8xi32>"],
                "tensor<4x8xbf16>",
            ),
            4,
            "bf16",
        )
        same_calls(
            f"{op} (32 lanes)",
            k,
            f"aie::store_v(ob, aie::{op}<bfloat16>(aie::load_v<32>(f)));",
        )

    k = Kernel()
    x = k.load("b8", 0, 64, "i8")
    k.store(
        "i16",
        0,
        k.op(
            "unpack",
            [x, k.shape()],
            ["tensor<8x8xi8>", "tensor<8x8xi32>"],
            "tensor<8x8xi16>",
        ),
        8,
        "i16",
    )
    same_calls(
        "unpack",
        k,
        "aie::store_v((uint8 *)i16, aie::unpack(aie::load_v<128>((uint4 *)b8)));",
    )

    for first in (True, False):
        k = Kernel()
        a, b = k.load("i16", 0, 64, "i16"), k.load("i16", 64, 64, "i16")
        name = "zip_first" if first else "zip_second"
        r = k.op(
            name,
            [a, b, k.shape()],
            ["tensor<8x8xi16>"] * 2 + ["tensor<8x8xi32>"],
            "tensor<8x8xi16>",
        )
        k.store("i16", 0, r, 8, "i16")
        part = "first" if first else "second"
        same_calls(
            f"interleave_zip ({part})",
            k,
            "auto x = aie::interleave_zip(aie::load_v<128>((uint8 *)i16), aie::load_v<128>((uint8 *)(i16 + 64)), 1);"
            f" aie::store_v((uint8 *)i16, x.{part});",
        )

    k = Kernel()
    v = k.load("bf", 0, 8, "bf16")
    e, b = k.tmp("e"), k.tmp("b")
    k.emit(f"{e} = tensor.empty() : tensor<8x8xbf16>")
    k.emit(
        f"{b} = linalg.generic {{indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],"
        f' iterator_types = ["parallel", "parallel"]}} ins({v} : tensor<1x8xbf16>) outs({e} : tensor<8x8xbf16>) {{\n'
        "    ^bb0(%in: bf16, %out: bf16):\n      linalg.yield %in : bf16\n    } -> tensor<8x8xbf16>"
    )
    k.store(
        "bf",
        0,
        k.op(
            "rows8",
            [b, k.shape()],
            ["tensor<8x8xbf16>", "tensor<8x8xi32>"],
            "tensor<8x8xbf16>",
        ),
        8,
        "bf16",
    )
    same_calls(
        "rows8",
        k,
        "auto v = aie::load_v<8>(bf); aie::vector<bfloat16, 16> x = aie::concat(v, v);"
        " aie::vector<bfloat16, 32> y = aie::concat(x, x); aie::store_v(ob, aie::transpose(aie::concat(y, y), 8, 8));",
    )

    k = Kernel()
    k.op("set_rounding", [k.shape()], ["tensor<8x8xi32>"], "tensor<8x8xi32>")
    same_calls(
        "set_rounding_conv_even", k, "aie::set_rounding(aie::rounding_mode::conv_even);"
    )

    k = Kernel()
    v = k.load("bf", 0, 64, "bf16")
    k.store(
        "bf",
        0,
        k.op(
            "abs",
            [v, k.shape()],
            ["tensor<8x8xbf16>", "tensor<8x8xi32>"],
            "tensor<8x8xbf16>",
        ),
        8,
        "bf16",
    )
    same_calls(
        "abs (no intrinsic)", k, "aie::store_v(ob, aie::abs(aie::load_v<64>(bf)));"
    )


def stream_kernel(pops, loop_pops=None):
    """A kernel that pops `pops` blocks of a stream, then, with
    `loop_pops`, that many each trip of a loop."""
    k = Kernel()
    s = k.op(
        "bfp16_stream",
        [k.ptr8("b8"), k.int8(0), k.int8(9216)],
        ["tensor<8x8xi64>"] * 3,
        "tensor<8x8xi32>",
    )
    acc = k.load("f32", 0, 64, "f32")
    for _ in range(pops):
        b = k.op("pop", [s], ["tensor<8x8xi32>"], "tensor<8x8xi32>")
        acc = k.op(
            "mac_8x8_8x8T",
            [acc, b, b],
            ["tensor<8x8xf32>", "tensor<8x8xi32>", "tensor<8x8xi32>"],
            "tensor<8x8xf32>",
        )
    if loop_pops:
        lb, ub, st = k.tmp("lb"), k.tmp("ub"), k.tmp("st")
        k.emit(f"{lb} = arith.constant 0 : i32")
        k.emit(f"{ub} = arith.constant 4 : i32")
        k.emit(f"{st} = arith.constant 1 : i32")
        res, a = k.tmp("res"), k.tmp("acc")
        k.emit(
            f"{res} = scf.for %i = {lb} to {ub} step {st} iter_args({a} = {acc}) -> (tensor<8x8xf32>) : i32 {{"
        )
        cur = a
        for _ in range(loop_pops):
            b = k.op("pop", [s], ["tensor<8x8xi32>"], "tensor<8x8xi32>")
            cur = k.op(
                "mac_8x8_8x8T",
                [cur, b, b],
                ["tensor<8x8xf32>", "tensor<8x8xi32>", "tensor<8x8xi32>"],
                "tensor<8x8xf32>",
            )
        k.emit(f"scf.yield {cur} : tensor<8x8xf32>")
        k.emit("}")
        acc = res
    k.store("f32", 0, acc, 8, "f32")
    return k


def no_fill(text):
    pops, _ = ops_named(text, "aievec.fifo_ld_pop_bfp16")
    return ["no_fill" in p.attributes for p in pops]


def test_streams():
    print("streams:")
    flags = no_fill(lower(stream_kernel(10)))
    check(
        flags == [False] + [True] * 7 + [False, True],
        f"straight pops fill every 8th: {flags}",
    )
    flags = no_fill(lower(stream_kernel(1, loop_pops=8)))
    check(
        flags == [False, True] + [True] * 6 + [False],
        f"a loop of 8 pops a trip: counts known: {flags}",
    )
    flags = no_fill(lower(stream_kernel(1, loop_pops=4)))
    check(
        flags == [False] + [False] * 4,
        f"a loop of 4 pops a trip: every pop in it fills: {flags}",
    )
    out = lower(stream_kernel(1, loop_pops=1))
    loops, _ = ops_named(out, "scf.for")
    types = [str(t) for t in loops[0].results.types] if loops else []
    check(
        types == ["vector<8x8xf32>", "!llvm.ptr", "vector<32xi32>", "i32"],
        f"one mac a trip: the loop carries the stream's state, and the accumulator as a vector: {types}",
    )
    out4 = lower(stream_kernel(1, loop_pops=4))
    loops, _ = ops_named(out4, "scf.for")
    types = [str(t) for t in loops[0].results.types] if loops else []
    check(
        types[0] == "tensor<8x8xf32>",
        f"macs chained in a trip: the accumulator stays a tensor: {types[0] if types else None}",
    )
    starts, _ = ops_named(out, "aievec.fifo_start")
    check(len(starts) == 1, "one fifo_start for the stream")


def test_buffers():
    print("buffers:")
    k = Kernel()
    k.store("f32", 16, k.load("f32", 32, 64, "f32"), 8, "f32")
    out = lower(k)
    check(
        out.lstrip().startswith("module attributes {aie.code}"),
        "a kernel whose aie ops are lowered: marked for aie_post",
    )
    plain = "module {\n  func.func @k() {\n    return\n  }\n}\n"
    check(aie_ops.lower(plain) == plain, "a kernel without aie ops: as it was")
    check(
        "memref.view" not in out and out.count("memref.reinterpret_cast") == 2,
        "a buffer of the values' type: reinterpret_cast, no view",
    )
    k = Kernel()
    k.store("b8", 4, k.load("b8", 8, 64, "bf16"), 8, "bf16")
    out = lower(k)
    views, _ = ops_named(out, "memref.view")
    check(
        len(views) == 2 and all("bf16" in str(v.results[0].type) for v in views),
        "an int8 buffer read and written as bf16: memref.view",
    )


def test_marks():
    print("marks:")
    for kind, want in (
        ("function:f32_to_bf16_mn", {"aie.fnloop", "aie.fn"}),
        ("function:dq4_bfp16:restrict", {"aie.fnloop", "aie.fn", "aie.restrict"}),
        ("unroll:2", {"loop_annotation"}),
    ):
        k = Kernel()
        k.emit(
            "%lb = arith.constant 0 : i32\n    %ub = arith.constant 4 : i32\n    %st = arith.constant 1 : i32"
        )
        k.emit("scf.for %i = %lb to %ub step %st : i32 {")
        k.op(kind, [k.shape()], ["tensor<8x8xi32>"], "tensor<8x8xi32>")
        k.store("f32", 0, k.load("f32", 0, 64, "f32"), 8, "f32")
        k.emit("}")
        loops, _ = ops_named(lower(k), "scf.for")
        got = set(loops[0].attributes) if loops else set()
        check(want <= got, f"{kind}: {sorted(want)} on the loop")
        if kind.startswith("function"):
            check(
                loops[0].attributes["aie.fn"].value == kind.split(":")[1],
                f"{kind}: its name",
            )
    k = stream_kernel(0, loop_pops=4)
    text = k.text().replace(
        "scf.yield",
        '%u = tt.elementwise_inline_asm "aie.unroll:4" {constraints = "=r,r", packed_element = 1 : i32, pure = false} %ush : tensor<8x8xi32> -> tensor<8x8xi32>\n    scf.yield',
        1,
    )
    text = text.replace(
        "{\n    %", "{\n    %ush = arith.constant dense<1> : tensor<8x8xi32>\n    %", 1
    )
    loops, _ = ops_named(aie_ops.lower(text), "scf.for")
    check(
        loops and "aie.k_loop" in loops[0].attributes,
        "unroll(4) in a loop that reads a stream: aie.k_loop",
    )


def test_pointers():
    print("pointers:")
    k = Kernel()
    p = k.ptr8("f32")
    k.emit(f"%keep = arith.addi {p}, {p} : tensor<8x8xi64>")
    k.store("f32", 0, k.load("f32", 0, 64, "f32"), 8, "f32")
    try:
        lower(k)
        check(False, "a pointer made an int for another op is an error")
    except ValueError as e:
        check(
            "other than an aie op" in str(e),
            f"a pointer made an int for another op is an error: {e}",
        )


def test_wiring():
    print("wiring:")
    k = Kernel()
    a, b = k.load("bf", 0, 32, "bf16"), k.load("bf", 32, 32, "bf16")
    m = k.op(
        "lt",
        [a, b, k.shape(4)],
        ["tensor<4x8xbf16>"] * 2 + ["tensor<4x8xi32>"],
        "tensor<4x8xi32>",
    )
    r = k.op(
        "select",
        [a, b, m, k.shape(4)],
        ["tensor<4x8xbf16>"] * 2 + ["tensor<4x8xi32>"] * 2,
        "tensor<4x8xbf16>",
    )
    k.store("bf", 0, r, 4, "bf16")
    out = lower(k)
    sel, keep = calls_to(out, "llvm.aie2p.vsel16")
    check(
        [origin(v) for v in sel[0].operands[:2]] == [(2, 0), (2, 32)],
        "select(a, b, m): vsel16(a, b, m)",
    )
    lt, keep2 = calls_to(out, "llvm.aie2p.vltbf16")
    check(origin(lt[0].operands[1]) == (2, 32), "lt(a, b): vltbf16 against b")
    check(sel[0].operands[2] == lt[0].results[0], "select takes lt's mask")

    k = Kernel()
    c = k.load("f32", 0, 64, "f32")
    a, b = k.load("bf", 0, 64, "bf16"), k.load("bf", 64, 64, "bf16")
    r = k.op(
        "mac",
        [c, a, b, k.shape()],
        ["tensor<8x8xf32>", "tensor<8x8xbf16>", "tensor<8x8xbf16>", "tensor<8x8xi32>"],
        "tensor<8x8xf32>",
    )
    k.store("f32", 0, r, 8, "f32")
    mac, keep = calls_to(lower(k), "llvm.aie2p.I1024.I1024.ACC2048.bf.mac.conf")
    check(
        [origin(v) for v in mac[0].operands[:3]] == [(2, 0), (2, 64), (1, 0)],
        "mac(acc, a, b): bf.mac.conf(a, b, acc)",
    )

    for name, half in (("zip_first", 0), ("zip_second", 16)):
        k = Kernel()
        a, b = k.load("i16", 0, 64, "i16"), k.load("i16", 64, 64, "i16")
        r = k.op(
            name,
            [a, b, k.shape()],
            ["tensor<8x8xi16>"] * 2 + ["tensor<8x8xi32>"],
            "tensor<8x8xi16>",
        )
        k.store("i16", 0, r, 8, "i16")
        sh, keep = calls_to(lower(k), "llvm.aie2p.vshuffle")
        ok = all(
            [origin(c.operands[0]), origin(c.operands[1])] == [(3, 0), (3, 64)]
            for c in sh
        )
        ok = ok and all(slice_offsets(c.operands[0])[:1] == [half] for c in sh)
        mode = sorted(c.operands[2].owner.attributes["value"].value for c in sh)
        check(
            ok and mode == [20, 21],
            f"{name}: vshuffle 20 and 21 of a's and b's half at {half}",
        )

    k = Kernel()
    v = k.load("bf", 0, 8, "bf16")
    e, b = k.tmp("e"), k.tmp("b")
    k.emit(f"{e} = tensor.empty() : tensor<8x8xbf16>")
    k.emit(
        f"{b} = linalg.generic {{indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],"
        f' iterator_types = ["parallel", "parallel"]}} ins({v} : tensor<1x8xbf16>) outs({e} : tensor<8x8xbf16>) {{\n'
        "    ^bb0(%in: bf16, %out: bf16):\n      linalg.yield %in : bf16\n    } -> tensor<8x8xbf16>"
    )
    k.store(
        "bf",
        0,
        k.op(
            "rows8",
            [b, k.shape()],
            ["tensor<8x8xbf16>", "tensor<8x8xi32>"],
            "tensor<8x8xbf16>",
        ),
        8,
        "bf16",
    )
    vs, keep = calls_to(lower(k), "llvm.aie2p.vshift.bf512.bf512")
    first = [c for c in vs if c.operands[0].owner.operation.name == "llvm.mlir.undef"]
    second = [
        c for c in vs if c.operands[0] == (first[0].results[0] if first else None)
    ]
    check(
        len(first) == 1 and len(second) == 1,
        "rows8: the second vshift shifts into the first's result",
    )

    k = Kernel()
    s = k.op(
        "bfp16_ostream",
        [k.ptr8("b8"), k.int8(2), k.int8(9216)],
        ["tensor<8x8xi64>"] * 3,
        "tensor<8x8xi32>",
    )
    for off in (0, 64):
        blk = k.op(
            "to_bfp16",
            [k.load("f32", off, 64, "f32"), k.shape()],
            ["tensor<8x8xf32>", "tensor<8x8xi32>"],
            "tensor<8x8xi32>",
        )
        k.op("push", [s, blk], ["tensor<8x8xi32>"] * 2, "tensor<8x8xi32>")
    out = lower(k)
    pushes, keep = ops_named(out, "aievec.fifo_st_push_bfp16")
    starts, keep2 = ops_named(out, "aievec.fifo_start")
    ok = len(pushes) == 2 and len(starts) == 1
    ok = ok and list(pushes[0].operands[:3]) == list(starts[0].results)
    ok = ok and list(pushes[1].operands[:3]) == list(pushes[0].results)
    check(
        ok, "pushes: the first takes the stream's start, the second the first's state"
    )
    conv, keep3 = calls_to(out, "llvm.aie2p.v64accfloat.to.v64bfp16ebs8")
    check(
        [origin(c.operands[0]) for c in conv] == [(1, 0), (1, 64)],
        "each push writes its to_bfp16 block",
    )
    off = starts[0].operands[1].owner.operation if starts else None
    check(
        off is not None and off.name == "arith.index_cast",
        "the stream starts at block 2 (72 bytes a block)",
    )

    out = lower(stream_kernel(1, loop_pops=1))
    loops, keep = ops_named(out, "scf.for")
    pops, keep2 = ops_named(out, "aievec.fifo_ld_pop_bfp16")
    lp = loops[0]
    body = lp.regions[0].blocks[0]
    inner = [p for p in pops if p.parent == lp]
    ok = len(inner) == 1 and list(inner[0].operands) == list(body.arguments)[2:5]
    ok = ok and list(lp.operands)[4:7] == list(pops[0].results)[:3]
    ok = (
        ok
        and list(list(lp.regions[0].blocks[0].operations)[-1].operands)[1:4]
        == list(inner[0].results)[:3]
    )
    check(
        ok,
        "the loop's state: the pop before it in, its pop on its arguments, the pop's state out",
    )
    reads = [
        o
        for o in lp.parent.regions[0].blocks[0].operations
        if o.operation.name == "vector.transfer_read"
    ]
    check(
        any(r.results[0] == lp.operands[3] for r in reads),
        "the accumulator read whole before the loop",
    )
    writes = [
        u.owner for u in lp.results[0].uses if u.owner.name == "vector.transfer_write"
    ]
    check(len(writes) == 1, "and written back after it")

    k = Kernel()
    x = k.tmp("x")
    k.emit(f"{x} = arith.extsi %n : i32 to i64")
    k.store(
        "f32",
        0,
        k.op(
            "load_v",
            [k.ptr8("f32"), k.int8(x)],
            ["tensor<8x8xi64>"] * 2,
            "tensor<8x8xf32>",
        ),
        8,
        "f32",
    )
    casts, keep = ops_named(lower(k), "memref.reinterpret_cast")
    ok = any(
        c.operands[1].owner.operation.name == "arith.index_cast"
        for c in casts
        if len(c.operands) > 1
    )
    check(ok, "an offset not known: index_cast of it")


def test_marked_accumulator():
    print("marked accumulator:")
    text = stream_kernel(1, loop_pops=1).text()
    text = text.replace(
        ": i32 {\n",
        ': i32 {\n    %fsh = arith.constant dense<1> : tensor<8x8xi32>\n    %fn = tt.elementwise_inline_asm "aie.function:mm_bfp16" {constraints = "=r,r", packed_element = 1 : i32, pure = false} %fsh : tensor<8x8xi32> -> tensor<8x8xi32>\n',
        1,
    )
    loops, keep = ops_named(aie_ops.lower(text), "scf.for")
    lp = loops[0]
    ok = (
        lp.attributes["aie.fn"].value == "mm_bfp16"
        and str(lp.results[0].type) == "vector<8x8xf32>"
    )
    check(ok, "a function's loop keeps its marks when its accumulator becomes a vector")


def test_errors():
    print("errors:")
    k = Kernel()
    v = k.op(
        "load_v",
        [k.ptr8("f32", offset=64), k.int8(0)],
        ["tensor<8x8xi64>"] * 2,
        "tensor<8x8xf32>",
    )
    k.store("f32", 0, v, 8, "f32")
    text = k.text()
    try:
        aie_ops.lower(text)
        check(False, "a pointer with an offset is an error")
    except ValueError as e:
        check(
            "offsets in `off`" in str(e), f"a pointer with an offset is an error: {e}"
        )
    except Exception as e:  # the hand-written IR itself
        check(False, f"a pointer with an offset: {type(e).__name__}: {str(e)[:200]}")

    k = Kernel()
    s = k.op(
        "bfp16_stream",
        [k.ptr8("b8"), k.int8(0), k.int8(9216)],
        ["tensor<8x8xi64>"] * 3,
        "tensor<8x8xi32>",
    )
    c = k.tmp("c")
    k.emit(f"{c} = arith.cmpi eq, %n, %n : i32")
    k.emit(f"scf.if {c} {{")
    b = k.op("pop", [s], ["tensor<8x8xi32>"], "tensor<8x8xi32>")
    acc = k.load("f32", 0, 64, "f32")
    k.store(
        "f32",
        0,
        k.op(
            "mac_8x8_8x8T",
            [acc, b, b],
            ["tensor<8x8xf32>", "tensor<8x8xi32>", "tensor<8x8xi32>"],
            "tensor<8x8xf32>",
        ),
        8,
        "f32",
    )
    k.emit("}")
    try:
        lower(k)
        check(False, "a stream read in an scf.if is an error")
    except ValueError as e:
        check(
            "other than scf.for" in str(e),
            f"a stream read in an scf.if is an error: {e}",
        )


def test_accumulator_used_in_trip():
    print("accumulator used in its trip:")
    k = Kernel()
    s = k.op(
        "bfp16_stream",
        [k.ptr8("b8"), k.int8(0), k.int8(9216)],
        ["tensor<8x8xi64>"] * 3,
        "tensor<8x8xi32>",
    )
    acc = k.load("f32", 0, 64, "f32")
    k.emit(
        "%lb = arith.constant 0 : i32\n    %ub = arith.constant 4 : i32\n    %st = arith.constant 1 : i32"
    )
    k.emit(
        f"%res = scf.for %i = %lb to %ub step %st iter_args(%a = {acc}) -> (tensor<8x8xf32>) : i32 {{"
    )
    b = k.op("pop", [s], ["tensor<8x8xi32>"], "tensor<8x8xi32>")
    c = k.op(
        "mac_8x8_8x8T",
        ["%a", b, b],
        ["tensor<8x8xf32>", "tensor<8x8xi32>", "tensor<8x8xi32>"],
        "tensor<8x8xf32>",
    )
    k.store("f32", 64, c, 8, "f32")
    k.emit(f"scf.yield {c} : tensor<8x8xf32>")
    k.emit("}")
    k.store("f32", 0, "%res", 8, "f32")
    loops, keep = ops_named(lower(k), "scf.for")
    check(
        str(loops[0].results[0].type) == "tensor<8x8xf32>",
        "an accumulator also stored in its trip stays a tensor",
    )


def test_templates():
    print("templates:")
    gen = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "gen_aie_templates.py"
    )
    if CLANG is None or AIE_API is None:
        print("  skip ops.mlir against ops.cc: no Peano clang or aie_api headers")
    else:
        r = subprocess.run(
            [sys.executable, gen, "--check"], capture_output=True, text=True
        )
        check(r.returncode == 0, f"ops.mlir is ops.cc's IR: {r.stdout.strip()}")

    def two(name, ty="bf16", res="bf16", lanes=64):
        k = Kernel()
        buf = {"bf16": "bf", "f32": "f32"}[ty]
        a, b = k.load(buf, 0, lanes, ty), k.load(buf, lanes, lanes, ty)
        rows = lanes // 8
        r = k.op(
            name,
            [a, b, k.shape(rows)],
            [f"tensor<{rows}x8x{ty}>"] * 2 + [f"tensor<{rows}x8xi32>"],
            f"tensor<{rows}x8x{res}>",
        )
        k.store({"bf16": "bf", "f32": "f32"}[res], 0, r, rows, res)
        return k

    def one(name, ty, res, lanes=64, out_lanes=None):
        k = Kernel()
        buf = {"bf16": "bf", "f32": "f32"}[ty]
        rows = lanes // 8
        x = k.load(buf, 0, lanes, ty)
        r = k.op(
            name,
            [x, k.shape(rows)],
            [f"tensor<{rows}x8x{ty}>", f"tensor<{rows}x8xi32>"],
            f"tensor<{rows}x8x{res}>",
        )
        k.store({"bf16": "bf", "f32": "f32"}[res], 0, r, rows, res)
        return k

    same_calls(
        "max64",
        two("t:op_max_bb:64:64,64"),
        "aie::store_v(ob, aie::max(aie::load_v<64>(bf), aie::load_v<64>(bf + 64)));",
    )
    same_calls(
        "sub64",
        two("t:op_sub_bb:64:64,64"),
        "aie::store_v(ob, aie::sub(aie::load_v<64>(bf), aie::load_v<64>(bf + 64)));",
    )
    same_calls(
        "mul_ff",
        two("t:op_mul_ff:64:64,64", "f32", "f32"),
        "aie::store_v(o, aie::mul(aie::load_v<64>(f), aie::load_v<64>(f + 64)).to_vector<float>());",
    )
    same_calls(
        "inv",
        one("t:op_inv:64:64", "f32", "f32"),
        "aie::store_v(o, aie::inv(aie::load_v<64>(f)));",
    )
    same_calls(
        "exp2_64",
        one("t:op_exp2_64:64:64", "f32", "bf16"),
        "aie::store_v(ob, aie::exp2<bfloat16>(aie::load_v<64>(f)));",
    )
    same_calls(
        "to_bf16_64",
        one("t:op_srs64:64:64", "f32", "bf16"),
        "ACC a; a.from_vector(aie::load_v<64>(f)); aie::store_v(ob, a.to_vector<bfloat16>());",
    )
    same_calls(
        "row_max",
        one("t:op_row_max:8:64", "bf16", "bf16"),
        "auto t = aie::transpose(aie::load_v<64>(bf), 8, 8);"
        " aie::vector<bfloat16, 32> a = aie::max(t.extract<32>(0), t.extract<32>(1));"
        " aie::vector<bfloat16, 16> b = aie::max(a.extract<16>(0), a.extract<16>(1));"
        " aie::store_v(ob, aie::max(b.extract<8>(0), b.extract<8>(1)));",
    )

    k = Kernel()
    x = k.load("bf", 0, 64, "bf16")
    sc, e = k.tmp("sc"), k.tmp("se")
    k.emit(f"{sc} = arith.constant 1.250000e-01 : bf16")
    k.emit(f"{e} = tensor.empty() : tensor<8x8xbf16>")
    f = k.tmp("sf")
    k.emit(
        f"{f} = linalg.fill ins({sc} : bf16) outs({e} : tensor<8x8xbf16>) -> tensor<8x8xbf16>"
    )
    r = k.op(
        "t:op_mulsc64:64:64,s",
        [x, f, k.shape()],
        ["tensor<8x8xbf16>"] * 2 + ["tensor<8x8xi32>"],
        "tensor<8x8xf32>",
    )
    k.store("f32", 0, r, 8, "f32")
    same_calls(
        "mul_scalar_f64",
        k,
        "aie::store_v(o, aie::mul(aie::load_v<64>(bf), bfloat16(0.125f)).to_vector<float>());",
    )

    k = Kernel()
    d = k.load("i16", 0, 64, "i16")
    m = k.op(
        "t:op_le0:m:64",
        [d, k.shape()],
        ["tensor<8x8xi16>", "tensor<8x8xi32>"],
        "tensor<8x8xi32>",
    )
    a, b = k.load("bf", 0, 64, "bf16"), k.load("bf", 64, 64, "bf16")
    r = k.op(
        "t:op_select64:64:64,64,m",
        [a, b, m, k.shape()],
        ["tensor<8x8xbf16>"] * 2 + ["tensor<8x8xi32>"] * 2,
        "tensor<8x8xbf16>",
    )
    k.store("bf", 0, r, 8, "bf16")
    same_calls(
        "le0, select64 (a mask as a value)",
        k,
        "aie::store_v(ob, aie::select(aie::load_v<64>(bf), aie::load_v<64>(bf + 64), aie::le(aie::load_v<64>(i16), int16(0))));",
    )

    k = Kernel()
    a = k.broadcast(k.load("bf", 0, 8, "bf16"), "bf16")
    r = k.op(
        "t:op_rows8a:64:8",
        [a, k.shape()],
        ["tensor<8x8xbf16>", "tensor<8x8xi32>"],
        "tensor<8x8xbf16>",
    )
    k.store("bf", 0, r, 8, "bf16")
    same_calls(
        "rows8a",
        k,
        "auto v = aie::load_v<8>(bf); auto a = aie::interleave_zip(v, v, 1); aie::vector<bfloat16, 16> b = aie::concat(a.first, a.second);"
        " auto c = aie::interleave_zip(b, b, 2); aie::vector<bfloat16, 32> d = aie::concat(c.first, c.second);"
        " auto e = aie::interleave_zip(d, d, 4); aie::store_v(ob, aie::concat(e.first, e.second));",
    )


def test_attention_ops():
    print("attention ops:")
    k = Kernel()
    s = k.op(
        "bfp16_stream",
        [k.ptr8("b8"), k.int8(0), k.int8(9216)],
        ["tensor<8x8xi64>"] * 3,
        "tensor<8x8xi32>",
    )
    a, b = (k.op("pop", [s], ["tensor<8x8xi32>"], "tensor<8x8xi32>") for _ in range(2))
    r = k.op("mul_8x8_8x8T", [a, b], ["tensor<8x8xi32>"] * 2, "tensor<8x8xf32>")
    k.store("f32", 0, r, 8, "f32")
    calls = intrinsic_calls(lower(k))
    check(
        calls == Counter({("llvm.aie2p.BFP576.BFP576.ACC2048.mul.conf", (780,)): 1}),
        f"mul_8x8_8x8T: BFP576 mul conf 780: {dict(calls)}",
    )

    k = Kernel()
    k.op(
        "store_v:8",
        [k.ptr8("bf"), k.int8(16), k.load("bf", 0, 64, "bf16")],
        ["tensor<8x8xi64>"] * 2 + ["tensor<8x8xbf16>"],
        "tensor<8x8xi32>",
    )
    out = lower(k)
    writes, keep = ops_named(out, "vector.transfer_write")
    stored = [
        str(w.operands[0].type) for w in writes if "memref" in str(w.operands[1].type)
    ]
    check(stored == ["vector<8xbf16>"], f"store_v:8 stores 8 lanes: {stored}")

    k = Kernel()
    k.store(
        "i16",
        0,
        k.op("cmr8", [k.shape()], ["tensor<8x8xi32>"], "tensor<8x8xi16>"),
        8,
        "i16",
    )
    out = lower(k)
    gl, keep = ops_named(out, "memref.global")
    gets, keep2 = ops_named(out, "memref.get_global")
    vals = str(gl[0].attributes["initial_value"]) if gl else ""
    check(
        len(gl) == 1 and len(gets) == 1 and vals.startswith("dense<[0, 1, 2"),
        "cmr8: a constant global (col - row), read once",
    )

    for v, want in (("1", -1), ("0", 0)):
        k = Kernel()
        m = k.op(f"mask_all:{v}", [k.shape()], ["tensor<8x8xi32>"], "tensor<8x8xi32>")
        a, b = k.load("bf", 0, 64, "bf16"), k.load("bf", 64, 64, "bf16")
        r = k.op(
            "t:op_select64:64:64,64,m",
            [a, b, m, k.shape()],
            ["tensor<8x8xbf16>"] * 2 + ["tensor<8x8xi32>"] * 2,
            "tensor<8x8xbf16>",
        )
        k.store("bf", 0, r, 8, "bf16")
        consts, keep = ops_named(lower(k), "arith.constant")
        check(
            any(
                str(c.results[0].type) == "i64" and c.attributes["value"].value == want
                for c in consts
            ),
            f"mask_all:{v}: the mask {want}",
        )

    k = Kernel()
    k.emit(
        "%lb = arith.constant 0 : i32\n    %ub = arith.constant 4 : i32\n    %st = arith.constant 1 : i32"
    )
    k.emit("scf.for %i = %lb to %ub step %st : i32 {")
    k.op("unroll:1", [k.shape()], ["tensor<8x8xi32>"], "tensor<8x8xi32>")
    k.store("f32", 0, k.load("f32", 0, 64, "f32"), 8, "f32")
    k.emit("}")
    loops, keep = ops_named(lower(k), "scf.for")
    check(
        "disable = true" in str(loops[0].attributes["loop_annotation"]),
        "unroll(1): unroll disabled",
    )


def span_kernel(after_use=False, end=True):
    k = Kernel()
    k.op(
        "begin_function:attn_init", [k.shape()], ["tensor<8x8xi32>"], "tensor<8x8xi32>"
    )
    v = k.load("f32", 0, 64, "f32")
    k.store("f32", 64, v, 8, "f32")
    if end:
        k.op("end_function", [k.shape()], ["tensor<8x8xi32>"], "tensor<8x8xi32>")
    if after_use:
        k.store("f32", 128, v, 8, "f32")
    return k


def test_spans():
    print("begin_function:")
    regions, keep = ops_named(lower(span_kernel()), "scf.execute_region")
    ok = len(regions) == 1 and regions[0].attributes["aie.fn"].value == "attn_init"
    ok = (
        ok
        and "no_inline" in regions[0].attributes
        and "aie.fnloop" in regions[0].attributes
    )
    inside = (
        [o.operation.name for o in regions[0].regions[0].blocks[0].operations]
        if regions
        else []
    )
    check(
        ok and "vector.transfer_write" in inside,
        "the code between begin and end in an scf.execute_region no_inline, named",
    )
    for what, k, msg in (
        (
            "a value made in it used after it",
            span_kernel(after_use=True),
            "used after it",
        ),
        ("no end_function", span_kernel(end=False), "no end_function"),
    ):
        try:
            lower(k)
            check(False, f"{what}: an error")
        except ValueError as e:
            check(msg in str(e), f"{what}: an error: {e}")


def test_if_results():
    print("scf.if results:")
    k = Kernel()
    a, b = k.load("bf", 0, 64, "bf16"), k.load("bf", 64, 64, "bf16")
    c = k.tmp("c")
    k.emit(f"{c} = arith.cmpi eq, %n, %n : i32")
    r = k.tmp("r")
    k.emit(f"{r} = scf.if {c} -> (tensor<8x8xbf16>) {{")
    t = k.op(
        "t:op_max_bb:64:64,64",
        [a, b, k.shape()],
        ["tensor<8x8xbf16>"] * 2 + ["tensor<8x8xi32>"],
        "tensor<8x8xbf16>",
    )
    k.emit(f"scf.yield {t} : tensor<8x8xbf16>")
    k.emit("} else {")
    k.emit(f"scf.yield {a} : tensor<8x8xbf16>")
    k.emit("}")
    k.store("bf", 0, r, 8, "bf16")
    ifs, keep = ops_named(lower(k), "scf.if")
    types = [str(t) for t in ifs[0].results.types] if ifs else []
    check(
        types == ["vector<64xbf16>"],
        f"an aie value out of an scf.if: a 1-D vector: {types}",
    )


def test_template_halves():
    print("template results in order:")
    k = Kernel()
    a, b = k.load("bf", 0, 64, "bf16"), k.load("bf", 64, 64, "bf16")
    r = k.op(
        "t:op_max_bb:64:64,64",
        [a, b, k.shape()],
        ["tensor<8x8xbf16>"] * 2 + ["tensor<8x8xi32>"],
        "tensor<8x8xbf16>",
    )
    k.store("bf", 0, r, 8, "bf16")
    out = lower(k)
    shuffles, keep = ops_named(out, "vector.shuffle")
    cat = [sh for sh in shuffles if str(sh.results[0].type) == "vector<64xbf16>"]

    def call_of(v):
        for _ in range(16):
            d = v.owner
            if not hasattr(d, "operation"):
                return None
            d = d.operation
            if d.name == "llvm.call":
                return d
            v = d.operands[0]
        return None

    firsts = (
        [slice_offsets(call_of(cat[0].operands[i]).operands[0])[:1] for i in (0, 1)]
        if cat
        else None
    )
    check(
        firsts == [[0], [32]],
        f"max64: the result's first half from the operands' first halves: {firsts}",
    )


def test_mask_layout():
    print("mask layout:")
    k = Kernel()
    d = k.load("i16", 0, 64, "i16")
    m = k.op(
        "t:op_le0:m:64",
        [d, k.shape()],
        ["tensor<8x8xi16>", "tensor<8x8xi32>"],
        "tensor<8x8xi32>",
    )
    a, b = k.load("bf", 0, 64, "bf16"), k.load("bf", 64, 64, "bf16")
    r = k.op(
        "t:op_select64:64:64,64,m",
        [a, b, m, k.shape()],
        ["tensor<8x8xbf16>"] * 2 + ["tensor<8x8xi32>"] * 2,
        "tensor<8x8xbf16>",
    )
    k.store("bf", 0, r, 8, "bf16")
    out = lower(k)
    shuffles, keep = ops_named(out, "llvm.shufflevector")
    put = [sh for sh in shuffles if str(sh.operands[0].type) == "vector<2xi32>"]
    masks = [str(sh.attributes["mask"]) for sh in put]
    check(
        len(put) == 1 and masks[0].startswith("array<i32: 0, 1, -1"),
        f"a mask's two words in lanes 0 and 1, in order: {masks}",
    )
    slices, keep2 = ops_named(out, "vector.extract_strided_slice")
    taken = [sl for sl in slices if str(sl.results[0].type) == "vector<2xi32>"]
    check(
        len(taken) == 1 and str(taken[0].attributes["offsets"]) == "[0]",
        "and read from lanes 0 and 1",
    )


def test_more_wiring():
    print("more wiring:")
    # scf.if: two results, one not [R, 8]; the branches kept apart
    k = Kernel()
    a, b = k.load("bf", 0, 64, "bf16"), k.load("bf", 64, 64, "bf16")
    c = k.tmp("c")
    k.emit(f"{c} = arith.cmpi eq, %n, %n : i32")
    r = k.tmp("r")
    k.emit(f"{r}:2 = scf.if {c} -> (tensor<8x8xbf16>, i32) {{")
    t = k.op(
        "t:op_max_bb:64:64,64",
        [a, b, k.shape()],
        ["tensor<8x8xbf16>"] * 2 + ["tensor<8x8xi32>"],
        "tensor<8x8xbf16>",
    )
    k.emit(f"scf.yield {t}, %n : tensor<8x8xbf16>, i32")
    k.emit("} else {")
    k.emit("%z = arith.constant 0 : i32")
    k.emit(f"scf.yield {b}, %z : tensor<8x8xbf16>, i32")
    k.emit("}")
    k.store("bf", 0, f"{r}#0", 8, "bf16")
    x = k.tmp("x")
    k.emit(f"{x} = arith.extsi {r}#1 : i32 to i64")
    k.store(
        "bf",
        64,
        k.op(
            "load_v",
            [k.ptr8("bf"), k.int8(x)],
            ["tensor<8x8xi64>"] * 2,
            "tensor<8x8xbf16>",
        ),
        8,
        "bf16",
    )
    out = lower(k)
    ifs, keep = ops_named(out, "scf.if")
    ok = len(ifs) == 1 and [str(t) for t in ifs[0].results.types] == [
        "vector<64xbf16>",
        "i32",
    ]
    then_y = list(ifs[0].regions[0].blocks[0].operations)[-1] if ok else None
    else_y = list(ifs[0].regions[1].blocks[0].operations)[-1] if ok else None
    ok = ok and origin(else_y.operands[0]) == (2, 64)
    ok = ok and origin(then_y.operands[0]) not in ((2, 0), (2, 64))
    check(
        ok,
        "an scf.if of an aie value and an i32: the value a vector, each branch its own",
    )
    writes = [u.owner for u in ifs[0].results[0].uses] if ifs else []
    ints = [u.owner.name for u in ifs[0].results[1].uses] if ifs else []
    check(
        any(w.name == "vector.shape_cast" for w in writes) and "arith.extsi" in ints,
        "after it: the vector written back, the i32 as it was",
    )

    # nested ifs
    k = Kernel()
    a = k.load("bf", 0, 64, "bf16")
    c = k.tmp("c")
    k.emit(f"{c} = arith.cmpi eq, %n, %n : i32")
    k.emit(f"%o = scf.if {c} -> (tensor<8x8xbf16>) {{")
    k.emit(f"%i = scf.if {c} -> (tensor<8x8xbf16>) {{")
    t = k.op(
        "t:op_max_bb:64:64,64",
        [a, a, k.shape()],
        ["tensor<8x8xbf16>"] * 2 + ["tensor<8x8xi32>"],
        "tensor<8x8xbf16>",
    )
    k.emit(f"scf.yield {t} : tensor<8x8xbf16>")
    k.emit("} else {")
    k.emit(f"scf.yield {a} : tensor<8x8xbf16>")
    k.emit("}")
    k.emit("scf.yield %i : tensor<8x8xbf16>")
    k.emit("} else {")
    k.emit(f"scf.yield {a} : tensor<8x8xbf16>")
    k.emit("}")
    k.store("bf", 0, "%o", 8, "bf16")
    ifs, keep = ops_named(lower(k), "scf.if")
    check(
        len(ifs) == 2
        and all(str(t) == "vector<64xbf16>" for i in ifs for t in i.results.types),
        "nested ifs: both carry vectors",
    )

    # mul_8x8_8x8T(a, b): a's block first
    k = Kernel()
    s = k.op(
        "bfp16_stream",
        [k.ptr8("b8"), k.int8(0), k.int8(9216)],
        ["tensor<8x8xi64>"] * 3,
        "tensor<8x8xi32>",
    )
    a, b = (k.op("pop", [s], ["tensor<8x8xi32>"], "tensor<8x8xi32>") for _ in range(2))
    k.store(
        "f32",
        0,
        k.op("mul_8x8_8x8T", [a, b], ["tensor<8x8xi32>"] * 2, "tensor<8x8xf32>"),
        8,
        "f32",
    )
    out = lower(k)
    mul, keep = calls_to(out, "llvm.aie2p.BFP576.BFP576.ACC2048.mul.conf")
    pops, keep2 = ops_named(out, "aievec.fifo_ld_pop_bfp16")
    ok = list(mul[0].operands[0:2]) == list(pops[0].results[3:5]) and list(
        mul[0].operands[2:4]
    ) == list(pops[1].results[3:5])
    check(ok, "mul_8x8_8x8T(a, b): a's mantissas and exponents first")

    # a narrow template result: its lanes first, the rest poison
    k = Kernel()
    k.store(
        "bf",
        0,
        k.op(
            "t:op_row_max:8:64",
            [k.load("bf", 0, 64, "bf16"), k.shape()],
            ["tensor<8x8xbf16>", "tensor<8x8xi32>"],
            "tensor<8x8xbf16>",
        ),
        8,
        "bf16",
    )
    sh, keep = ops_named(lower(k), "llvm.shufflevector")
    pads = [
        str(x.attributes["mask"])
        for x in sh
        if str(x.results[0].type) == "vector<64xbf16>"
    ]
    check(
        len(pads) == 1 and pads[0].startswith("array<i32: 0, 1, 2, 3, 4, 5, 6, 7, -1"),
        "8 lanes in an [8, 8] result: lanes 0 to 7, the rest poison",
    )

    # store_v:8 stores lanes 0 to 7
    k = Kernel()
    k.op(
        "store_v:8",
        [k.ptr8("bf"), k.int8(16), k.load("bf", 0, 64, "bf16")],
        ["tensor<8x8xi64>"] * 2 + ["tensor<8x8xbf16>"],
        "tensor<8x8xi32>",
    )
    sl, keep = ops_named(lower(k), "vector.extract_strided_slice")
    check(
        [str(x.attributes["offsets"]) for x in sl] == ["[0]"],
        "store_v:8: the value's first 8 lanes",
    )

    # begin_function: restrict, what is outside stays out, nesting
    k = Kernel()
    k.store("f32", 0, k.load("f32", 0, 64, "f32"), 8, "f32")
    k.op(
        "begin_function:outer:restrict",
        [k.shape()],
        ["tensor<8x8xi32>"],
        "tensor<8x8xi32>",
    )
    k.op("begin_function:inner", [k.shape()], ["tensor<8x8xi32>"], "tensor<8x8xi32>")
    k.store("f32", 64, k.load("f32", 64, 64, "f32"), 8, "f32")
    k.op("end_function", [k.shape()], ["tensor<8x8xi32>"], "tensor<8x8xi32>")
    k.op("end_function", [k.shape()], ["tensor<8x8xi32>"], "tensor<8x8xi32>")
    k.store("f32", 128, k.load("f32", 128, 64, "f32"), 8, "f32")
    out = lower(k)
    regions, keep = ops_named(out, "scf.execute_region")
    names = {r.attributes["aie.fn"].value: r for r in regions}
    ok = (
        set(names) == {"outer", "inner"} and "aie.restrict" in names["outer"].attributes
    )
    ok = (
        ok
        and "aie.restrict" not in names["inner"].attributes
        and names["inner"].parent == names["outer"]
    )
    check(ok, "nested spans: inner in outer, outer restrict")
    fn, keep2 = ops_named(out, "func.func")
    top = [o.operation.name for o in fn[0].regions[0].blocks[0].operations]
    writes_outside = top.count("vector.transfer_write")
    check(
        writes_outside >= 2,
        f"the stores before and after the spans stay outside: {writes_outside}",
    )


def raises(what, fn, msg):
    """fn() raises a ValueError saying `msg`."""
    try:
        fn()
    except ValueError as e:
        check(msg in str(e), f"{what}: {e}")
        return
    check(False, f"{what}: no error")


def test_small_templates():
    print("small templates:")
    k = Kernel()
    a, b = k.load("bf", 0, 8, "bf16"), k.load("bf", 8, 8, "bf16")
    r = k.op(
        "t:op_max8:8:8,8",
        [a, b, k.shape(1)],
        ["tensor<1x8xbf16>"] * 2 + ["tensor<1x8xi32>"],
        "tensor<1x8xbf16>",
    )
    k.store("bf", 0, r, 1, "bf16")
    same_calls(
        "max8",
        k,
        "aie::store_v(ob, aie::max(aie::load_v<8>(bf), aie::load_v<8>(bf + 8)));",
    )

    k = Kernel()
    a, b = k.load("bf", 0, 16, "bf16"), k.load("bf", 16, 16, "bf16")
    r = k.op(
        "t:op_sub16:16:16,16",
        [a, b, k.shape(2)],
        ["tensor<2x8xbf16>"] * 2 + ["tensor<2x8xi32>"],
        "tensor<2x8xbf16>",
    )
    k.store("bf", 0, r, 2, "bf16")
    same_calls(
        "sub16",
        k,
        "aie::store_v(ob, aie::sub(aie::load_v<16>(bf), aie::load_v<16>(bf + 16)));",
    )

    k = Kernel()
    x = k.load("bf", 0, 16, "bf16")
    sc, e, f = k.tmp("sc"), k.tmp("se"), k.tmp("sf")
    k.emit(f"{sc} = arith.constant 1.250000e-01 : bf16")
    k.emit(f"{e} = tensor.empty() : tensor<2x8xbf16>")
    k.emit(
        f"{f} = linalg.fill ins({sc} : bf16) outs({e} : tensor<2x8xbf16>) -> tensor<2x8xbf16>"
    )
    r = k.op(
        "t:op_mulsc16:16:16,s",
        [x, f, k.shape(2)],
        ["tensor<2x8xbf16>"] * 2 + ["tensor<2x8xi32>"],
        "tensor<2x8xf32>",
    )
    k.store("f32", 0, r, 2, "f32")
    same_calls(
        "mul_scalar_f16",
        k,
        "aie::store_v(o, aie::mul(aie::load_v<16>(bf), bfloat16(0.125f)).to_vector<float>());",
    )

    k = Kernel()
    r = k.op(
        "t:op_exp2_16:16:16",
        [k.load("f32", 0, 16, "f32"), k.shape(2)],
        ["tensor<2x8xf32>", "tensor<2x8xi32>"],
        "tensor<2x8xbf16>",
    )
    k.store("bf", 0, r, 2, "bf16")
    same_calls(
        "exp2_16", k, "aie::store_v(ob, aie::exp2<bfloat16>(aie::load_v<16>(f)));"
    )

    k = Kernel()
    d = k.load("i16", 0, 64, "i16")
    m = k.op(
        "t:op_gt0:m:64",
        [d, k.shape()],
        ["tensor<8x8xi16>", "tensor<8x8xi32>"],
        "tensor<8x8xi32>",
    )
    a, b = k.load("bf", 0, 64, "bf16"), k.load("bf", 64, 64, "bf16")
    r = k.op(
        "t:op_select64:64:64,64,m",
        [a, b, m, k.shape()],
        ["tensor<8x8xbf16>"] * 2 + ["tensor<8x8xi32>"] * 2,
        "tensor<8x8xbf16>",
    )
    k.store("bf", 0, r, 8, "bf16")
    same_calls(
        "gt0, select64",
        k,
        "aie::store_v(ob, aie::select(aie::load_v<64>(bf), aie::load_v<64>(bf + 64), aie::gt(aie::load_v<64>(i16), int16(0))));",
    )

    # the lanes each takes, in order
    k = Kernel()
    a, b = k.load("bf", 0, 16, "bf16"), k.load("bf", 16, 16, "bf16")
    r = k.op(
        "t:op_sub16:16:16,16",
        [a, b, k.shape(2)],
        ["tensor<2x8xbf16>"] * 2 + ["tensor<2x8xi32>"],
        "tensor<2x8xbf16>",
    )
    k.store("bf", 0, r, 2, "bf16")
    out = lower(k)
    calls, keep = ops_named(out, "llvm.call")
    subs = [c for c in calls if "sub" in c.attributes["callee"].value]
    firsts = [source(c.operands[0]) for c in subs]
    check(
        subs and all(f == (2, 0) for f in firsts),
        f"sub16(a, b): a first: {firsts}",
    )

    # concat8 of two [1, 8] values (Triton broadcasts them to [2, 8]), and
    # of two broadcast to [8, 8]
    for rows in (2, 8):
        k = Kernel()
        a, b = k.load("bf", 0, 8, "bf16"), k.load("bf", 8, 8, "bf16")
        a, b = k.broadcast(a, "bf16", rows), k.broadcast(b, "bf16", rows)
        t = f"tensor<{rows}x8xbf16>"
        r = k.op(
            "t:op_cat8:16:8,8",
            [a, b, k.shape(rows)],
            [t, t, f"tensor<{rows}x8xi32>"],
            t,
        )
        k.store("bf", 32, r, rows, "bf16")
        vs, keep = calls_to(lower(k), "llvm.aie2p.vshift.bf512.bf512")
        first = [
            c for c in vs if c.operands[0].owner.operation.name == "llvm.mlir.undef"
        ]
        second = [c for c in vs if first and c.operands[0] == first[0].results[0]]
        got = [source(c.operands[1]) for c in first + second]
        check(
            got == [(2, 0), (2, 8)],
            f"concat8 of [{rows}, 8] operands: a shifted in first, then b: {got}",
        )

    k = Kernel()
    r = k.op(
        "t:op_ext8:8:16",
        [k.load("bf", 0, 16, "bf16"), k.shape(2)],
        ["tensor<2x8xbf16>", "tensor<2x8xi32>"],
        "tensor<2x8xbf16>",
    )
    k.store("bf", 64, r, 2, "bf16")
    sh, keep = ops_named(lower(k), "llvm.shufflevector")
    ext = [x for x in sh if str(x.results[0].type) == "vector<4xi32>"]
    masks = [str(x.attributes["mask"]) for x in ext]
    check(
        len(ext) == 1
        and masks[0] == "array<i32: 0, 1, 2, 3>"
        and source(ext[0].operands[0]) == (2, 0),
        f"extract8: the first 4 words (lanes 0 to 7) of its operand: {masks}",
    )


# Templates that break each rule of _TemplateSet.instantiate, and one that
# keeps them all: two 32-lane arguments, the output pointer last.
_BAD_TEMPLATES = """
module {
  llvm.func @t_ok(%a: vector<32xbf16>, %b: vector<32xbf16>, %o: !llvm.ptr) {
    %c = llvm.mlir.constant(64 : i32) : i32
    llvm.store %a, %o : vector<32xbf16>, !llvm.ptr
    %p = llvm.getelementptr %o[%c] : (!llvm.ptr, i32) -> !llvm.ptr, i8
    llvm.store %b, %p : vector<32xbf16>, !llvm.ptr
    llvm.return
  }
  llvm.func @t_elem(%a: vector<32xbf16>, %b: vector<32xbf16>, %o: !llvm.ptr) {
    llvm.store %a, %o : vector<32xbf16>, !llvm.ptr
    %p = llvm.getelementptr %o[1] : (!llvm.ptr) -> !llvm.ptr, vector<32xbf16>
    llvm.store %b, %p : vector<32xbf16>, !llvm.ptr
    llvm.return
  }
  llvm.func @t_unknown(%a: vector<32xbf16>, %b: vector<32xbf16>, %o: !llvm.ptr) {
    %c = llvm.mlir.constant(32 : i32) : i32
    %n = llvm.add %c, %c : i32
    llvm.store %a, %o : vector<32xbf16>, !llvm.ptr
    %p = llvm.getelementptr %o[%n] : (!llvm.ptr, i32) -> !llvm.ptr, i8
    llvm.store %b, %p : vector<32xbf16>, !llvm.ptr
    llvm.return
  }
  llvm.func @t_returns(%a: vector<32xbf16>, %b: vector<32xbf16>, %o: !llvm.ptr) -> vector<32xbf16> {
    %c = llvm.mlir.constant(64 : i32) : i32
    llvm.store %a, %o : vector<32xbf16>, !llvm.ptr
    %p = llvm.getelementptr %o[%c] : (!llvm.ptr, i32) -> !llvm.ptr, i8
    llvm.store %b, %p : vector<32xbf16>, !llvm.ptr
    llvm.return %a : vector<32xbf16>
  }
  llvm.func @t_loads(%a: vector<32xbf16>, %b: vector<32xbf16>, %o: !llvm.ptr) {
    %c = llvm.mlir.constant(64 : i32) : i32
    llvm.store %a, %o : vector<32xbf16>, !llvm.ptr
    %x = llvm.load %o : !llvm.ptr -> vector<32xbf16>
    %p = llvm.getelementptr %o[%c] : (!llvm.ptr, i32) -> !llvm.ptr, i8
    llvm.store %x, %p : vector<32xbf16>, !llvm.ptr
    llvm.return
  }
  llvm.func @t_gap(%a: vector<32xbf16>, %b: vector<32xbf16>, %o: !llvm.ptr) {
    %c = llvm.mlir.constant(128 : i32) : i32
    llvm.store %a, %o : vector<32xbf16>, !llvm.ptr
    %p = llvm.getelementptr %o[%c] : (!llvm.ptr, i32) -> !llvm.ptr, i8
    llvm.store %b, %p : vector<32xbf16>, !llvm.ptr
    llvm.return
  }
  llvm.func @t_uneven(%a: vector<32xbf16>, %b: vector<32xbf16>, %o: !llvm.ptr) {
    %c = llvm.mlir.constant(64 : i32) : i32
    %h = llvm.shufflevector %b, %b [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<32xbf16>
    llvm.store %a, %o : vector<32xbf16>, !llvm.ptr
    %p = llvm.getelementptr %o[%c] : (!llvm.ptr, i32) -> !llvm.ptr, i8
    llvm.store %h, %p : vector<16xbf16>, !llvm.ptr
    llvm.return
  }
}
"""


def t_kernel(spec, operands=2, rows=8, res=None, op_types=None):
    """A kernel of one template op `spec` on `operands` [rows, 8] bf16
    loads, its result stored."""
    k = Kernel()
    xs = [k.load("bf", 64 * i, 8 * rows, "bf16") for i in range(operands)]
    t = f"tensor<{rows}x8xbf16>"
    r = k.op(
        spec, xs + [k.shape(rows)], [t] * operands + [f"tensor<{rows}x8xi32>"], res or t
    )
    k.store("bf", 0, r, rows, (res or t).split("x")[-1].rstrip(">"))
    return k


def test_template_checks():
    print("template checks:")
    saved = aie_ops._TEMPLATES
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
        f.write(_BAD_TEMPLATES)
    aie_ops._TEMPLATES = f.name
    try:
        writes, keep = ops_named(
            lower(t_kernel("t:t_ok:64:64")), "vector.transfer_write"
        )
        check(writes, "a template that keeps the rules lowers")
        for name, msg in (
            ("t_elem", "other than one byte index"),
            ("t_unknown", "by a value not known"),
            ("t_returns", "returns a value"),
            ("t_loads", "uses its output other than by stores"),
            ("t_gap", "gap at byte 64"),
            ("t_uneven", "parts of 32 and 16 lanes"),
        ):
            raises(name, lambda: lower(t_kernel(f"t:{name}:64:64")), msg)
    finally:
        aie_ops._TEMPLATES = saved
        os.unlink(f.name)


def test_operand_checks():
    print("template operands:")
    k = Kernel()
    a, b = k.load("bf", 0, 64, "bf16"), k.load("bf", 64, 64, "bf16")
    r = k.op(
        "t:op_mulsc64:64:64,s",
        [a, b, k.shape()],
        ["tensor<8x8xbf16>"] * 2 + ["tensor<8x8xi32>"],
        "tensor<8x8xf32>",
    )
    k.store("f32", 0, r, 8, "f32")
    raises("s of a value not a fill", lambda: lower(k), "operand 1 is not a tl.full")
    for what, k, msg in (
        (
            "64 lanes of a [4, 8] operand",
            t_kernel("t:op_max_bb:64:64,64", rows=4),
            "operand 0 has fewer than 64 lanes",
        ),
        (
            "a third operand",
            t_kernel("t:op_max_bb:64:64,64,64", operands=3),
            "more operand lanes than its arguments take",
        ),
        (
            "48 lanes into 32-lane arguments",
            t_kernel("t:op_max_bb:64:48,64"),
            "48 lanes do not fill its arguments",
        ),
        (
            "too few operands",
            t_kernel("t:op_max_bb:64:64", operands=1),
            "2 of its 4 arguments",
        ),
        (
            "a mask of a template storing two values",
            t_kernel("t:op_max_bb:m:64,64", res="tensor<8x8xi32>"),
            "a mask template stores 2 values",
        ),
        (
            "a result of other lanes",
            t_kernel("t:op_max_bb:32:64,64"),
            "64 lanes, not 32",
        ),
    ):
        raises(what, lambda: lower(k), msg)


def loop(k, body, iter_acc=None):
    """`body(k)` in an scf.for of 4 trips; with `iter_acc`, the loop carries
    that accumulator and body returns its next value. Returns the loop's
    result."""
    lb, ub, st = k.tmp("lb"), k.tmp("ub"), k.tmp("st")
    k.emit(f"{lb} = arith.constant 0 : i32")
    k.emit(f"{ub} = arith.constant 4 : i32")
    k.emit(f"{st} = arith.constant 1 : i32")
    if iter_acc is None:
        k.emit(f"scf.for %i{k.n} = {lb} to {ub} step {st} : i32 {{")
        body(k)
        k.emit("}")
        return None
    res, a = k.tmp("res"), k.tmp("acc")
    k.emit(
        f"{res} = scf.for %i{k.n} = {lb} to {ub} step {st} iter_args({a} = {iter_acc}) -> (tensor<8x8xf32>) : i32 {{"
    )
    nxt = body(k, a)
    k.emit(f"scf.yield {nxt} : tensor<8x8xf32>")
    k.emit("}")
    return res


def stream(k, block=0):
    return k.op(
        "bfp16_stream",
        [k.ptr8("b8"), k.int8(block), k.int8(9216)],
        ["tensor<8x8xi64>"] * 3,
        "tensor<8x8xi32>",
    )


def pop_mac(k, s, acc):
    b = k.op("pop", [s], ["tensor<8x8xi32>"], "tensor<8x8xi32>")
    return k.op(
        "mac_8x8_8x8T",
        [acc, b, b],
        ["tensor<8x8xf32>", "tensor<8x8xi32>", "tensor<8x8xi32>"],
        "tensor<8x8xf32>",
    )


def test_stream_checks():
    print("stream checks:")
    k = Kernel()
    s = stream(k)
    acc = k.load("f32", 0, 64, "f32")
    loop(
        k, lambda k: loop(k, lambda k: k.store("f32", 0, pop_mac(k, s, acc), 8, "f32"))
    )
    raises(
        "a read two loops below the start",
        lambda: lower(k),
        "nested loops below its start",
    )

    k = Kernel()
    s = stream(k)
    acc = k.load("f32", 0, 64, "f32")
    for _ in range(2):
        loop(k, lambda k: k.store("f32", 0, pop_mac(k, s, acc), 8, "f32"))
    raises("reads in two loops", lambda: lower(k), "read in more than one loop")

    k = Kernel()
    s = stream(k)
    acc = k.load("f32", 0, 64, "f32")
    res = loop(k, lambda k, a: pop_mac(k, s, a), iter_acc=acc)
    k.store("f32", 0, pop_mac(k, s, res), 8, "f32")
    raises("a read after the loop", lambda: lower(k), "after the loop that carries it")

    k = Kernel()
    x = k.tmp("x")
    k.emit(f"{x} = arith.extsi %n : i32 to i64")
    k.op(
        "bfp16_stream",
        [k.ptr8("b8"), k.int8(0), k.int8(x)],
        ["tensor<8x8xi64>"] * 3,
        "tensor<8x8xi32>",
    )
    raises(
        "a buffer size not a constant", lambda: lower(k), "buffer size is a constant"
    )


def test_two_streams_one_loop():
    print("two streams in one loop:")
    k = Kernel()
    s1, s2 = stream(k, 0), stream(k, 64)
    acc = k.load("f32", 0, 64, "f32")
    res = loop(k, lambda k, a: pop_mac(k, s2, pop_mac(k, s1, a)), iter_acc=acc)
    k.store("f32", 0, res, 8, "f32")
    out = lower(k)
    loops, keep = ops_named(out, "scf.for")
    starts, keep2 = ops_named(out, "aievec.fifo_start")
    pops, keep3 = ops_named(out, "aievec.fifo_ld_pop_bfp16")
    lp = loops[0]
    args = list(lp.regions[0].blocks[0].arguments)
    y = list(lp.regions[0].blocks[0].operations)[-1]
    ok = len(starts) == 2 and len(pops) == 2 and len(args) == 1 + 1 + 6
    ok = ok and list(lp.operands)[4:7] == list(starts[0].results)
    ok = ok and list(lp.operands)[7:10] == list(starts[1].results)
    ok = (
        ok
        and list(pops[0].operands) == args[2:5]
        and list(pops[1].operands) == args[5:8]
    )
    ok = ok and list(y.operands)[1:4] == list(pops[0].results)[:3]
    ok = ok and list(y.operands)[4:7] == list(pops[1].results)[:3]
    check(
        ok,
        "each stream its own three iter_args: its start in, its pop on them, its state out",
    )


def test_more_checks():
    print("more checks:")
    k = Kernel()
    k.store("f32", 0, k.load("bf", 0, 64, "f32"), 8, "f32")
    raises(
        "f32 values of a bf16 buffer", lambda: lower(k), "f32 values of a bf16 buffer"
    )

    k = Kernel()
    x = k.tmp("x")
    k.emit(f"{x} = arith.addi {k.int8(1)}, {k.int8(2)} : tensor<8x8xi64>")
    v = k.op("load_v", [k.ptr8("f32"), x], ["tensor<8x8xi64>"] * 2, "tensor<8x8xf32>")
    k.store("f32", 0, v, 8, "f32")
    raises("an offset not a fill or a constant", lambda: lower(k), "not an int operand")

    k = Kernel()
    r = k.op(
        "rows8",
        [k.load("bf", 0, 64, "bf16"), k.shape()],
        ["tensor<8x8xbf16>", "tensor<8x8xi32>"],
        "tensor<8x8xbf16>",
    )
    k.store("bf", 0, r, 8, "bf16")
    raises("rows8 of an [8, 8] value", lambda: lower(k), "aie.rows8: of a [1, 8] value")

    for name in ("add", "lt"):
        k = Kernel()
        a, b = k.load("bf", 0, 64, "bf16"), k.load("bf", 64, 64, "bf16")
        res = "tensor<8x8xi32>" if name == "lt" else "tensor<8x8xbf16>"
        r = k.op(
            name, [a, b, k.shape()], ["tensor<8x8xbf16>"] * 2 + ["tensor<8x8xi32>"], res
        )
        raises(f"{name} of 64 lanes", lambda: lower(k), f"aie.{name}: 32 lanes")
    k = Kernel()
    a, b = k.load("bf", 0, 64, "bf16"), k.load("bf", 64, 64, "bf16")
    m = k.op("mask_all:1", [k.shape()], ["tensor<8x8xi32>"], "tensor<8x8xi32>")
    k.op(
        "select",
        [a, b, m, k.shape()],
        ["tensor<8x8xbf16>"] * 2 + ["tensor<8x8xi32>"] * 2,
        "tensor<8x8xbf16>",
    )
    raises("select of 64 lanes", lambda: lower(k), "aie.select: 32 lanes")

    k = Kernel()
    k.op("unroll:2", [k.shape()], ["tensor<8x8xi32>"], "tensor<8x8xi32>")
    raises("unroll outside a loop", lambda: lower(k), "aie.unroll: not in a loop")

    k = Kernel()
    k.op("end_function", [k.shape()], ["tensor<8x8xi32>"], "tensor<8x8xi32>")
    raises("end_function alone", lambda: lower(k), "no begin_function before it")

    k = Kernel()
    k.op("begin_function:f", [k.shape()], ["tensor<8x8xi32>"], "tensor<8x8xi32>")
    loop(
        k,
        lambda k: k.op(
            "end_function", [k.shape()], ["tensor<8x8xi32>"], "tensor<8x8xi32>"
        ),
    )
    raises(
        "end_function in a loop",
        lambda: lower(k),
        "its end_function is in another block",
    )


def test_unroll4_without_stream():
    print("unroll(4) without a stream:")
    k = Kernel()

    def body(k):
        k.op("unroll:4", [k.shape()], ["tensor<8x8xi32>"], "tensor<8x8xi32>")
        k.store("f32", 0, k.load("f32", 0, 64, "f32"), 8, "f32")

    loop(k, body)
    loops, keep = ops_named(lower(k), "scf.for")
    a = loops[0].attributes
    check(
        "aie.k_loop" not in a and "count = 4" in str(a["loop_annotation"]),
        "a loop that reads no stream: unroll count 4, not a k-loop",
    )


def test_constant_operands():
    print("constant operands:")
    k = Kernel()
    c = k.tmp("c")
    k.emit(f"{c} = arith.constant dense<1.000000e+00> : tensor<8x8xbf16>")
    sc, e, f = k.tmp("sc"), k.tmp("e"), k.tmp("f")
    k.emit(f"{sc} = arith.constant 2.000000e+00 : bf16")
    k.emit(f"{e} = tensor.empty() : tensor<8x8xbf16>")
    k.emit(
        f"{f} = linalg.fill ins({sc} : bf16) outs({e} : tensor<8x8xbf16>) -> tensor<8x8xbf16>"
    )
    r = k.op(
        "t:op_max_bb:64:64,64",
        [c, f, k.shape()],
        ["tensor<8x8xbf16>"] * 2 + ["tensor<8x8xi32>"],
        "tensor<8x8xbf16>",
    )
    k.store("bf", 0, r, 8, "bf16")
    out = lower(k)
    consts, keep = ops_named(out, "arith.constant")
    vec = [c for c in consts if str(c.results[0].type) == "vector<64xbf16>"]
    bcs, keep2 = ops_named(out, "vector.broadcast")
    reads, keep3 = ops_named(out, "vector.transfer_read")
    from_tensor = [
        r
        for r in reads
        if getattr(r.operands[0].owner, "name", None)
        in ("arith.constant", "linalg.fill")
    ]
    check(
        len(vec) == 1
        and len(bcs) == 1
        and str(bcs[0].results[0].type) == "vector<64xbf16>"
        and not from_tensor,
        "a splat constant a vector constant, a fill a broadcast of its scalar, neither read from its tensor",
    )


def test_declared_once():
    print("declarations:")
    k = t_kernel("t:op_max_bb:64:64,64")
    decl = "  llvm.func @llvm.aie2p.vmax.ltbf16(vector<32xbf16>, vector<32xbf16>) -> !llvm.struct<(vector<32xbf16>, i32)>\n"
    text = k.text().replace("module {\n", "module {\n" + decl, 1)
    funcs, keep = ops_named(aie_ops.lower(text), "llvm.func")
    n = sum(
        1 for f in funcs if f.attributes["sym_name"].value == "llvm.aie2p.vmax.ltbf16"
    )
    check(n == 1, f"an intrinsic the kernel declares already: declared once: {n}")


def main():
    test_clang()
    test_streams()
    test_buffers()
    test_marks()
    test_pointers()
    test_wiring()
    test_marked_accumulator()
    test_errors()
    test_accumulator_used_in_trip()
    test_templates()
    test_attention_ops()
    test_spans()
    test_if_results()
    test_template_halves()
    test_mask_layout()
    test_more_wiring()
    test_small_templates()
    test_template_checks()
    test_operand_checks()
    test_stream_checks()
    test_two_streams_one_loop()
    test_more_checks()
    test_unroll4_without_stream()
    test_constant_operands()
    test_declared_once()
    if failures:
        print(f"{failures} failed")
        sys.exit(1)
    print("all passed")


if __name__ == "__main__":
    main()
