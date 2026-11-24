# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lowering of ``tl.extra.air.aie`` ops (``language/air/aie.py``).

Each op arrives as a ``tt.elementwise_inline_asm`` whose string names it,
on tensors, right after triton-shared. It is replaced where it is by the IR
that Peano's clang emits for its aie_api call: vector ops, aievec ops and
calls of aie2p intrinsics, between a ``vector.transfer_read`` of each
operand tensor and a ``vector.transfer_write`` of the result tensor.

A bfp16 stream keeps its state (pointer, buffer and count, as aie_api's
stream object does) in values: each read or write takes the state the one
before it left, and a loop whose trips read a stream started outside it
carries the state in three iter_args, as clang keeps the stream in
registers across the loop.

An accumulator that a loop reads and writes whole once a trip is carried as
a vector rather than a tensor, as clang keeps an ``aie::accum`` in
registers.

A template op is the IR clang emits for a small C function making its
aie_api call (``aie_templates/``): its body in place of the op, its
arguments the operands' lanes cut to their widths, its stores the result.
A mask is a value, its 64 bits lanes 0 and 1 of an [8, 8] int32 tensor; an
scf.if's [R, 8] tensor results become 1-D vector results.

Loops are marked for the backend's post-schedule steps (``aie_post.py``):
``aie.fnloop`` and the name ``aie.fn`` (with ``aie.restrict``) where
``function`` is called, ``aie.k_loop`` for ``unroll(4)`` in a loop that reads
a stream, and a loop annotation for any other ``unroll`` (``unroll(1)``
disables unrolling). The code between ``begin_function`` and
``end_function`` becomes an scf.execute_region marked the same way.

A use of the ops the lowering does not know raises ValueError.
"""

import os
import re

# intrinsics: name -> (argument types, result type or None)
_I = {
    "unpack": ("llvm.aie2p.unpack.I512.I8.I4", "vector<32xi8>, i32", "vector<64xi8>"),
    "vshuffle": (
        "llvm.aie2p.vshuffle",
        "vector<16xi32>, vector<16xi32>, i32",
        "vector<16xi32>",
    ),
    "vshift": (
        "llvm.aie2p.vshift.bf512.bf512",
        "vector<32xbf16>, vector<32xbf16>, i32, i32",
        "vector<32xbf16>",
    ),
    "ups": ("llvm.aie2p.v32bf16.to.v32accfloat", "vector<32xbf16>", "vector<32xf32>"),
    "accadd": (
        "llvm.aie2p.ACC2048.accfloat.add.conf",
        "vector<64xf32>, vector<64xf32>, i32",
        "vector<64xf32>",
    ),
    "srs32": ("llvm.aie2p.v32accfloat.to.v32bf16", "vector<32xf32>", "vector<32xbf16>"),
    "tanh": ("llvm.aie2p.tanh", "vector<16xf32>", "vector<16xbf16>"),
    "exp2": ("llvm.aie2p.exp2", "vector<16xf32>", "vector<16xbf16>"),
    "vlt": ("llvm.aie2p.vltbf16", "vector<32xbf16>, vector<32xbf16>", "i32"),
    "vsel": (
        "llvm.aie2p.vsel16",
        "vector<32xi16>, vector<32xi16>, i32",
        "vector<32xi16>",
    ),
    "set_ctrl": ("llvm.aie2p.set.ctrl.reg", "i32, i32", None),
}
_BFP = "!llvm.struct<(vector<64xi8>, vector<8xi8>)>"
_I["to_bfp16"] = ("llvm.aie2p.v64accfloat.to.v64bfp16ebs8", "vector<64xf32>", _BFP)
_STATE = ("!llvm.ptr", "vector<32xi32>", "i32")
_TPTR_SPACE, _PTR_SPACE = "#tptr.default_memory_space", "#ptr.generic_space"
_BITS = {"i8": 8, "i16": 16, "bf16": 16, "i32": 32, "f32": 32, "i64": 64}
# a buffer operand from its use back to the kernel argument
_POINTER_CHAIN = (
    "linalg.fill",
    "tptr.ptrtoint",
    "ptr.to_ptr",
    "memref.memory_space_cast",
    "memref.cast",
)


_TEMPLATES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "aie_templates", "ops.mlir"
)
# a constant a core loads: col - row of each lane of an 8x8 block (int16)
_CMR8 = "__aie_cmr8"
_CMR8_GLOBAL = (
    '"memref.global"() <{alignment = 64 : i64, constant, initial_value = dense<['
    + ", ".join(str(c - r) for r in range(8) for c in range(8))
    + f']> : tensor<64xi16>, sym_name = "{_CMR8}", sym_visibility = "private", type = memref<64xi16, 2 : i32>}}> : () -> ()'
)


def _bf_intrinsic(op, lanes):
    """aie::mul / mac / msc of bf16 vectors into accfloat: the intrinsic for
    `lanes` lanes (I512 x I512 for 32, I1024 x I1024 for 64)."""
    w = {32: "I512.I512.ACC1024", 64: "I1024.I1024.ACC2048"}[lanes]
    acc = "" if op == "mul" else f", vector<{lanes}xf32>"
    return (
        f"llvm.aie2p.{w}.bf.{op}.conf",
        f"vector<{lanes}xbf16>, vector<{lanes}xbf16>{acc}, i32",
        f"vector<{lanes}xf32>",
    )


def _srs(lanes):
    return (
        f"llvm.aie2p.v{lanes}accfloat.to.v{lanes}bf16",
        f"vector<{lanes}xf32>",
        f"vector<{lanes}xbf16>",
    )


def lower(ttshared):
    """`ttshared` with its aie ops lowered; unchanged if it has none."""
    if '"aie.' not in ttshared:
        return ttshared
    from air.ir import Context, Location, Module

    from .superkernel_jit import _generic

    # The ops' buffer operands are pointers made ints, in the memory space
    # triton-shared names #tptr.default_memory_space, which only its tools
    # parse. Read as the ptr dialect's generic space here; those pointers
    # feed the ops alone, and go with them.
    text = _generic(ttshared).replace(_TPTR_SPACE, _PTR_SPACE)
    from air.ir import UnitAttr

    from .aie_post import MARK

    with Context() as ctx, Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = Module.parse(text)
        _Lowering(m).run()
        m.operation.attributes[MARK] = UnitAttr.get()
        m.operation.verify()
        out = str(m)
    if _PTR_SPACE in out:
        raise ValueError(
            "aie: a pointer made an int for something other than an aie op"
        )
    return out


def _tensor_shape(t):
    """(rows, element type) of an [R, 8] tensor type."""
    m = re.fullmatch(r"tensor<(\d+)x8x(\w+)>", str(t))
    if not m:
        raise ValueError(f"aie: not an [R, 8] tensor: {t}")
    return int(m[1]), m[2]


def _zero(et):
    return "0.0" if et.startswith(("f", "bf")) else "0"


class _Snippet:
    """The IR of one op's expansion, as MLIR text: a function of the values
    it reads, parsed and inlined where the op is (`_Lowering.expand`)."""

    def __init__(self, lowering):
        self.lw = lowering
        self.args, self.lines, self.n = [], [], 0

    def arg(self, v):
        self.args.append(v)
        return f"%a{len(self.args) - 1}"

    def tmp(self):
        self.n += 1
        return f"%t{self.n}"

    def emit(self, line):
        self.lines.append(line)

    def const(self, value, ty):
        t = self.tmp()
        self.emit(f"{t} = arith.constant {value} : {ty}")
        return t

    def read(self, v, rows, et):
        """v, an [rows, 8] tensor, as a vector<8 rows x et>. A splat
        constant becomes a vector constant and a fill a broadcast of its
        scalar, which LLVM keeps in a register as clang does."""
        lanes = 8 * rows
        d = _def(v)
        if d is not None and d.name == "arith.constant":
            m = re.fullmatch(
                r"dense<([^\[][^>]*)> : tensor<.*", str(d.attributes["value"])
            )
            if m:
                return self.const(f"dense<{m[1]}>", f"vector<{lanes}x{et}>")
        if d is not None and d.name == "linalg.fill" and str(d.operands[0].type) == et:
            t = self.tmp()
            self.emit(
                f"{t} = vector.broadcast {self.arg(d.operands[0])} : {et} to vector<{lanes}x{et}>"
            )
            return t
        c0, z = self.const(0, "index"), self.const(_zero(et), et)
        r, t = self.tmp(), self.tmp()
        self.emit(
            f"{r} = vector.transfer_read {self.arg(v)}[{c0}, {c0}], {z} {{in_bounds = [true, true]}}"
            f" : tensor<{rows}x8x{et}>, vector<{rows}x8x{et}>"
        )
        self.emit(
            f"{t} = vector.shape_cast {r} : vector<{rows}x8x{et}> to vector<{lanes}x{et}>"
        )
        return t

    def write(self, v, rows, et):
        """The [rows, 8] tensor of v, a vector<8 rows x et>."""
        s, e, c0, w = self.tmp(), self.tmp(), self.tmp(), self.tmp()
        self.emit(
            f"{s} = vector.shape_cast {v} : vector<{8 * rows}x{et}> to vector<{rows}x8x{et}>"
        )
        self.emit(f"{e} = tensor.empty() : tensor<{rows}x8x{et}>")
        self.emit(f"{c0} = arith.constant 0 : index")
        self.emit(
            f"{w} = vector.transfer_write {s}, {e}[{c0}, {c0}] {{in_bounds = [true, true]}}"
            f" : vector<{rows}x8x{et}>, tensor<{rows}x8x{et}>"
        )
        return w

    def handle(self):
        """The placeholder an op that makes a handle returns: the values
        behind it are the lowering's."""
        t = self.tmp()
        self.emit(f"{t} = tensor.empty() : tensor<8x8xi32>")
        return t

    def call(self, fn, args):
        name, ins, out = fn
        self.lw.decls[name] = (ins, out)
        if out is None:
            self.emit(f"llvm.call @{name}({', '.join(args)}) : ({ins}) -> ()")
            return None
        t = self.tmp()
        self.emit(f"{t} = llvm.call @{name}({', '.join(args)}) : ({ins}) -> {out}")
        return t

    def cat(self, a, b, n, et):
        t = self.tmp()
        idx = ", ".join(map(str, range(2 * n)))
        self.emit(
            f"{t} = vector.shuffle {a}, {b} [{idx}] : vector<{n}x{et}>, vector<{n}x{et}>"
        )
        return t

    def part(self, v, off, n, total, et):
        t = self.tmp()
        self.emit(
            f"{t} = vector.extract_strided_slice {v} offsets = [{off}], sizes = [{n}], strides = [1]"
            f" : vector<{total}x{et}> to vector<{n}x{et}>"
        )
        return t

    def mask_value(self, h):
        """The mask h (i64) as a value: its two words lanes 0 and 1 of an
        [8, 8] int32 tensor, so that it passes through an scf.if as values
        do."""
        w, p = self.tmp(), self.tmp()
        self.emit(f"{w} = llvm.bitcast {h} : i64 to vector<2xi32>")
        self.emit(
            f"{p} = llvm.shufflevector {w}, {w} [0, 1{', -1' * 62}] : vector<2xi32>"
        )
        return self.write(p, 8, "i32")

    def mask_of(self, v):
        """The i64 of the mask value v ([8, 8] int32)."""
        x = self.read(v, 8, "i32")
        w, h = self.part(x, 0, 2, 64, "i32"), self.tmp()
        self.emit(f"{h} = llvm.bitcast {w} : vector<2xi32> to i64")
        return h

    def bitcast(self, v, frm, to):
        t = self.tmp()
        self.emit(f"{t} = vector.bitcast {v} : {frm} to {to}")
        return t

    def text(self, results):
        """The function: its arguments the values read, its results
        `results` [(name, type)]."""
        params = ", ".join(f"%a{i}: {a.type}" for i, a in enumerate(self.args))
        rtys = ", ".join(t for _, t in results)
        body = "\n    ".join(self.lines)
        ret = (
            f"return {', '.join(n for n, _ in results)} : {rtys}"
            if results
            else "return"
        )
        decls = "\n  ".join(
            [_decl(n, ins, out) for n, (ins, out) in self.lw.decls.items()]
            + list(self.lw.globals.values())
        )
        return (
            f"module {{\n  {decls}\n  func.func @f({params}) -> ({rtys}) {{\n"
            f"    {body}\n    {ret}\n  }}\n}}"
        )


def _decl(name, ins, out):
    return f"llvm.func @{name}({ins})" + (f" -> {out}" if out else "")


def _def(v):
    """The op defining v, None for a block argument."""
    from air.ir import Block

    o = v.owner
    return None if isinstance(o, Block) else o.operation


def _ancestors(op):
    out = []
    p = op.parent
    while p is not None:
        out.append(p)
        p = p.parent
    return out


def _loops_around(op):
    """The scf.for ops op is in, innermost first."""
    return [p for p in _ancestors(op) if p.name == "scf.for"]


def _ops_in_order(module):
    out = []

    def visit(op):
        out.append(op)
        for r in op.regions:
            for b in r.blocks:
                for o in b.operations:
                    visit(o.operation)

    for o in module.body.operations:
        visit(o.operation)
    return out


class _Lowering:
    def __init__(self, module):
        self.m = module
        self.decls = {}  # intrinsics called: name -> (argument types, result type)
        self.order = {}  # op -> its position in the module
        self.out = {}  # aie op -> the tensor that replaces its result
        self.streams = {}  # stream op -> [(read or write op, loop or None)]
        self.nofill = set()  # pops that need no FIFO fill
        # stream op -> (init placeholder, block args, yield placeholder)
        self.carry = {}
        self.state = {}  # stream op -> its state after the last op lowered
        self.state_at_entry = {}  # stream op -> its state where its loop starts
        self.blocks = {}  # pop or to_bfp16 op -> the block's (mantissas, exponents)
        self.masks = {}  # lt op -> the mask (i32)
        self.marks = []  # (loop, the marker's kind)
        self.spans = []  # (begin anchor, end anchor, name, restrict)
        self.globals = {}  # module constants: name -> definition
        self._templates = None

    @staticmethod
    def _kind(op):
        s = op.attributes["asm_string"].value
        return s[len("aie.") :] if s.startswith("aie.") else None

    @staticmethod
    def _buffer(v):
        """The kernel argument (a buffer, memref<*xT>) the pointer operand v
        is, made an int as triton-shared makes it: fill, ptrtoint, to_ptr,
        memory_space_cast, cast. Anything else, such as an offset pointer,
        raises."""
        from air.ir import Block

        for name in _POINTER_CHAIN:
            d = _def(v)
            if d is None or d.name != name:
                break
            v = d.operands[0]
        else:
            if (
                isinstance(v.owner, Block)
                and v.owner.owner.operation.name == "func.func"
            ):
                return v
        raise ValueError(
            "aie: a buffer operand is a kernel's own buffer or a cast of it, offsets in `off`"
        )

    @staticmethod
    def _int(v):
        """An int operand (a fill of an i64 scalar, or a splat constant):
        ("int", n) or ("ssa", the scalar)."""
        from air.ir import IntegerAttr

        d = _def(v)
        if d is None:
            raise ValueError("aie: not an int operand")
        if d.name == "linalg.fill":
            s = d.operands[0]
            c = _def(s)
            if c is not None and c.name == "arith.constant":
                return "int", IntegerAttr(c.attributes["value"]).value
            return "ssa", s
        m = d.name == "arith.constant" and re.fullmatch(
            r"dense<(-?\d+)> : .*", str(d.attributes["value"])
        )
        if not m:
            raise ValueError("aie: not an int operand")
        return "int", int(m[1])

    @staticmethod
    def _broadcast_of(v):
        """The [1, 8] tensor v broadcasts to its rows, None if v is not such
        a broadcast."""
        d = _def(v)
        if d is None or d.name != "linalg.generic" or len(d.operands) != 2:
            return None
        maps = [str(m) for m in d.attributes["indexing_maps"]]
        if maps != [
            "affine_map<(d0, d1) -> (0, d1)>",
            "affine_map<(d0, d1) -> (d0, d1)>",
        ]:
            return None
        src = d.operands[0]
        return src if re.fullmatch(r"tensor<1x8x\w+>", str(src.type)) else None

    def expand(self, sn, at, results):
        """Parse snippet `sn` and inline its body before op `at`; returns
        the values of `results` there."""
        from air.ir import Module

        snip = Module.parse(sn.text(results))
        f = next(o for o in snip.body.operations if o.operation.name == "func.func")
        block = f.regions[0].blocks[0]
        for a, v in zip(block.arguments, sn.args):
            a.replace_all_uses_with(v)
        ops = list(block.operations)
        ret = ops[-1]
        vals = list(ret.operands)
        for o in ops[:-1]:
            o.operation.move_before(at)
        return vals

    def run(self):
        all_ops = _ops_in_order(self.m)
        for i, o in enumerate(all_ops):
            self.order[o] = i
        ops = [
            o
            for o in all_ops
            if o.name == "tt.elementwise_inline_asm" and self._kind(o) is not None
        ]
        self._plan_streams(ops)
        self._carry_streams()
        for op in ops:
            self._lower(op)
        self._close_streams()
        self._replace(ops)
        self._mark(ops)
        self._wrap_spans()
        self._vector_if_results()
        self._vector_iter_args()
        self._declare()

    def _plan_streams(self, ops):
        """Per stream: its reads (or writes) in order, the loop below its
        start each is in, and the reads that need no FIFO fill."""
        for s in ops:
            if self._kind(s) not in ("bfp16_stream", "bfp16_ostream"):
                continue
            above = set(_ancestors(s))
            uses = sorted(
                {u.owner for u in s.results[0].uses}, key=lambda o: self.order[o]
            )
            reads = []
            for u in uses:
                below = [p for p in _ancestors(u) if p not in above]
                if any(p.name != "scf.for" for p in below):
                    raise ValueError(
                        "aie: a stream read under an op other than scf.for"
                    )
                if len(below) > 1:
                    raise ValueError(
                        "aie: a stream read in nested loops below its start"
                    )
                reads.append((u, below[0] if below else None))
            loops = {lp for _, lp in reads if lp is not None}
            if len(loops) > 1:
                raise ValueError("aie: a stream read in more than one loop")
            if loops:
                lp = loops.pop()
                if any(
                    l2 is None and self.order[u] > self.order[lp] for u, l2 in reads
                ):
                    raise ValueError(
                        "aie: a stream read after the loop that carries it"
                    )
            self.streams[s] = reads
            # aie_api fills the FIFO (8 pops of 576 bits) unless the reads
            # since the last fill are known to be 1 to 7: known along a
            # straight line, and in a loop whose trips read a multiple of 8
            count, inloop = 0, None
            for u, lp in reads:
                if lp is not None and inloop is None:
                    inloop = lp
                    if sum(1 for _, l2 in reads if l2 is not None and l2 == lp) % 8:
                        count = None
                if count and self._kind(u) == "pop":
                    self.nofill.add(u)
                count = None if count is None else (count + 1) % 8

    def _carry_streams(self):
        """Each loop that reads a stream started above it gets three
        iter_args for the stream's state; their inits and yields are
        placeholders until the stream is lowered."""
        from air.ir import InsertionPoint, Operation, Type

        per_loop = {}
        for s, reads in self.streams.items():
            lp = next((lp for _, lp in reads if lp is not None), None)
            if lp is not None:
                per_loop.setdefault(lp, []).append(s)
        types = [Type.parse(t) for t in _STATE]
        for lp, ss in per_loop.items():
            # (noted before _with_iter_args replaces lp)
            held = {
                s: [i for i, (_, l2) in enumerate(r) if l2 is not None and l2 == lp]
                for s, r in self.streams.items()
            }
            inits = []
            ph = {}
            for s in ss:
                p = Operation.create(
                    "aieops.state", results=types, ip=InsertionPoint(lp)
                )
                ph[s] = p
                inits += list(p.results)
            new, args = _with_iter_args(lp, inits)
            yph = {}
            for k, s in enumerate(ss):
                y = _yield_of(new)
                q = Operation.create(
                    "aieops.state", results=types, ip=InsertionPoint(y)
                )
                yph[s] = q
                self.carry[s] = (ph[s], args[3 * k : 3 * k + 3], q)
            _append_yield(new, [v for s in ss for v in yph[s].results])
            for s, idx in held.items():
                for i in idx:
                    self.streams[s][i] = (self.streams[s][i][0], new)

    def _close_streams(self):
        for s, (init, _, yph) in self.carry.items():
            for a, b in zip(init.results, self.state_at_entry[s]):
                a.replace_all_uses_with(b)
            for a, b in zip(yph.results, self.state[s]):
                a.replace_all_uses_with(b)
            init.erase()
            yph.erase()

    def _replace(self, ops):
        mine = set(ops)
        feeding = []
        for op in reversed(ops):
            new = self.out.get(op)
            if new is not None:
                op.results[0].replace_all_uses_with(new)
            feeding += [
                d for d in map(_def, op.operands) if d is not None and d not in mine
            ]
            op.erase()
        _erase_unused(feeding)

    def _lower(self, op):
        k = self._kind(op)
        kind = k.split(":")[0]
        getattr(self, "_op_" + kind)(op, k)

    def _stream_of(self, op):
        return _def(op.operands[0])

    def _state_in(self, op):
        """The state a read or write of a stream takes: the one before it
        left, or in the loop that carries the stream, that loop's."""
        s = self._stream_of(op)
        if s in self.carry:
            lp = next(l2 for u, l2 in self.streams[s] if u == op)
            if lp is not None and s not in self.state_at_entry:
                self.state_at_entry[s] = self.state[s]
                self.state[s] = self.carry[s][1]
        return s, self.state[s]

    def _op_bfp16_stream(self, op, k):
        sn = _Snippet(self)
        buf = self._buffer(op.operands[0])
        kind, nbytes = self._int(op.operands[2])
        if kind != "int":
            raise ValueError("aie: a stream's buffer size is a constant")
        kind, blk = self._int(op.operands[1])
        b = sn.arg(buf)
        s0 = sn.const(blk, "i64") if kind == "int" else sn.arg(blk)
        c72 = sn.const(72, "i64")
        o64, o, mem = sn.tmp(), sn.tmp(), sn.tmp()
        sn.emit(f"{o64} = arith.muli {s0}, {c72} : i64")
        sn.emit(f"{o} = arith.index_cast {o64} : i64 to index")
        sn.emit(
            f"{mem} = memref.reinterpret_cast {b} to offset: [0], sizes: [{nbytes}], strides: [1]"
            f" : {buf.type} to memref<{nbytes}xi8>"
        )
        p, ps, pc = sn.tmp(), sn.tmp(), sn.tmp()
        sn.emit(
            f"{p}, {ps}, {pc} = aievec.fifo_start {mem}[{o}] : memref<{nbytes}xi8> -> vector<32xi32>"
        )
        h = sn.handle()
        vals = self.expand(
            sn,
            op,
            [(p, _STATE[0]), (ps, _STATE[1]), (pc, _STATE[2]), (h, "tensor<8x8xi32>")],
        )
        self.state[op] = vals[:3]
        self.out[op] = vals[3]

    _op_bfp16_ostream = _op_bfp16_stream

    def _op_pop(self, op, k):
        s, (p, ps, pc) = self._state_in(op)
        sn = _Snippet(self)
        a = [sn.arg(v) for v in (p, ps, pc)]
        nf = " no_fill" if op in self.nofill else ""
        q, qs, qc, m, e = (sn.tmp() for _ in range(5))
        sn.emit(
            f"{q}, {qs}, {qc}, {m}, {e} = aievec.fifo_ld_pop_bfp16 {a[0]}, {a[1]}, {a[2]}{nf}"
            f" : vector<32xi32> -> vector<32xi32>, vector<64xi8>, vector<8xi8>"
        )
        h = sn.handle()
        vals = self.expand(
            sn,
            op,
            [
                (q, _STATE[0]),
                (qs, _STATE[1]),
                (qc, _STATE[2]),
                (m, "vector<64xi8>"),
                (e, "vector<8xi8>"),
                (h, "tensor<8x8xi32>"),
            ],
        )
        self.state[s] = vals[:3]
        self.blocks[op] = vals[3:5]
        self.out[op] = vals[5]

    def _op_push(self, op, k):
        s, (p, ps, pc) = self._state_in(op)
        m, e = self.blocks[_def(op.operands[1])]
        sn = _Snippet(self)
        a = [sn.arg(v) for v in (p, ps, pc, m, e)]
        q, qs, qc = sn.tmp(), sn.tmp(), sn.tmp()
        sn.emit(
            f"{q}, {qs}, {qc} = aievec.fifo_st_push_bfp16 {a[0]}, {a[1]}, {a[2]}, {a[3]}, {a[4]}"
            f" : vector<32xi32>, vector<64xi8>, vector<8xi8> -> vector<32xi32>"
        )
        h = sn.handle()
        vals = self.expand(
            sn,
            op,
            [(q, _STATE[0]), (qs, _STATE[1]), (qc, _STATE[2]), (h, "tensor<8x8xi32>")],
        )
        self.state[s] = vals[:3]
        self.out[op] = vals[3]

    def _op_mac_8x8_8x8T(self, op, k):
        # ::mac_8x8_8x8T_conf into the accumulator itself: a loop's
        # accumulator then bufferizes to one buffer
        acc = op.operands[0]
        (am, ae), (bm, be) = (self.blocks[_def(v)] for v in op.operands[1:3])
        sn = _Snippet(self)
        a = sn.arg(acc)
        x = [sn.arg(v) for v in (am, ae, bm, be)]
        c0, z = sn.const(0, "index"), sn.const("0.0", "f32")
        r, mm, w = sn.tmp(), sn.tmp(), sn.tmp()
        sn.emit(
            f"{r} = vector.transfer_read {a}[{c0}, {c0}], {z} {{in_bounds = [true, true]}} : tensor<8x8xf32>, vector<8x8xf32>"
        )
        sn.emit(
            f"{mm} = aievec.matmul_aie2p_bfp16 {x[0]}, {x[1]}, {x[2]}, {x[3]}, {r}"
            f" : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>"
        )
        sn.emit(
            f"{w} = vector.transfer_write {mm}, {a}[{c0}, {c0}] {{in_bounds = [true, true]}} : vector<8x8xf32>, tensor<8x8xf32>"
        )
        self.out[op] = self.expand(sn, op, [(w, "tensor<8x8xf32>")])[0]

    def _op_to_bfp16(self, op, k):
        sn = _Snippet(self)
        v = sn.read(op.operands[0], 8, "f32")
        b = sn.call(_I["to_bfp16"], [v])
        m, e = sn.tmp(), sn.tmp()
        sn.emit(f"{m} = llvm.extractvalue {b}[0] : {_BFP}")
        sn.emit(f"{e} = llvm.extractvalue {b}[1] : {_BFP}")
        h = sn.handle()
        vals = self.expand(
            sn, op, [(m, "vector<64xi8>"), (e, "vector<8xi8>"), (h, "tensor<8x8xi32>")]
        )
        self.blocks[op] = vals[:2]
        self.out[op] = vals[2]

    def _op_load_v(self, op, k):
        rows, et = _tensor_shape(op.results[0].type)
        sn = _Snippet(self)
        mt, mem = self._view(sn, op, et, 8 * rows)
        c0, z = sn.const(0, "index"), sn.const(_zero(et), et)
        v = sn.tmp()
        sn.emit(
            f"{v} = vector.transfer_read {mem}[{c0}], {z} {{in_bounds = [true]}} : {mt}, vector<{8 * rows}x{et}>"
        )
        w = sn.write(v, rows, et)
        self.out[op] = self.expand(sn, op, [(w, f"tensor<{rows}x8x{et}>")])[0]

    def _op_store_v(self, op, k):
        # store_v:N stores the value's first N lanes
        rows, et = _tensor_shape(op.operands[2].type)
        lanes = int(k.split(":")[1]) if ":" in k else 8 * rows
        sn = _Snippet(self)
        mt, mem = self._view(sn, op, et, lanes)
        c0 = sn.const(0, "index")
        f = sn.read(op.operands[2], rows, et)
        if lanes < 8 * rows:
            f = sn.part(f, 0, lanes, 8 * rows, et)
        sn.emit(
            f"vector.transfer_write {f}, {mem}[{c0}] {{in_bounds = [true]}} : vector<{lanes}x{et}>, {mt}"
        )
        h = sn.tmp()
        sn.emit(f"{h} = tensor.empty() : {op.results[0].type}")
        self.out[op] = self.expand(sn, op, [(h, str(op.results[0].type))])[0]

    def _view(self, sn, op, et, lanes):
        """The `lanes` values of type et at the op's buffer and offset, as a
        memref: (its type, its name). An int8 buffer read as another type
        is viewed as that type (memref.view), as the superkernel makes a
        kernel's pointer cast of its own buffer."""
        buf = self._buffer(op.operands[0])
        bt = re.fullmatch(r"memref<\*x(\w+)>", str(buf.type))[1]
        kind, off = self._int(op.operands[1])
        b = sn.arg(buf)
        if kind == "int":
            o = sn.const(off, "index")
        else:
            o = sn.tmp()
            sn.emit(f"{o} = arith.index_cast {sn.arg(off)} : i64 to index")
        if bt == et:
            mt = f"memref<{lanes}x{et}, strided<[1], offset: ?>>"
            mem = sn.tmp()
            sn.emit(
                f"{mem} = memref.reinterpret_cast {b} to offset: [{o}], sizes: [{lanes}], strides: [1] : memref<*x{et}> to {mt}"
            )
            return mt, mem
        if bt != "i8":
            raise ValueError(f"aie: {et} values of a {bt} buffer")
        size = _BITS[et] // 8
        kb = sn.const(size, "index")
        ob, n, end, raw, mem = (sn.tmp() for _ in range(5))
        sn.emit(f"{ob} = arith.muli {o}, {kb} : index")
        sn.emit(f"{n} = arith.constant {lanes * size} : index")
        sn.emit(f"{end} = arith.addi {ob}, {n} : index")
        sn.emit(
            f"{raw} = memref.reinterpret_cast {b} to offset: [0], sizes: [{end}], strides: [1] : memref<*xi8> to memref<?xi8>"
        )
        mt = f"memref<{lanes}x{et}>"
        sn.emit(f"{mem} = memref.view {raw}[{ob}][] : memref<?xi8> to {mt}")
        return mt, mem

    def _op_to_bf16(self, op, k):
        # the accumulator's srs
        rows, _ = _tensor_shape(op.operands[0].type)
        sn = _Snippet(self)
        v = sn.read(op.operands[0], rows, "f32")
        b = sn.call(_srs(8 * rows), [v])
        w = sn.write(b, rows, "bf16")
        self.out[op] = self.expand(sn, op, [(w, f"tensor<{rows}x8xbf16>")])[0]

    def _op_rows8(self, op, k):
        # concat to 64 lanes and transpose 8 x 8: the 8 values' 16 bytes
        # widened, shifted in twice (vshift 16 and 48 bytes), duplicated,
        # then vshuffle 52 and 53
        src = self._broadcast_of(op.operands[0])
        if src is None:
            raise ValueError("aie.rows8: of a [1, 8] value")
        sn = _Snippet(self)
        v = sn.read(src, 1, "bf16")
        i = sn.bitcast(v, "vector<8xbf16>", "vector<4xi32>")
        x = sn.tmp()
        sn.emit(
            f"{x} = llvm.shufflevector {i}, {i} [0, 1, 2, 3{', -1' * 12}] : vector<4xi32>"
        )
        xb = sn.bitcast(x, "vector<16xi32>", "vector<32xbf16>")
        u = sn.tmp()
        sn.emit(f"{u} = llvm.mlir.undef : vector<32xbf16>")
        k0, k16, k48 = sn.const(0, "i32"), sn.const(16, "i32"), sn.const(48, "i32")
        s1 = sn.call(_I["vshift"], [u, xb, k0, k16])
        s2 = sn.call(_I["vshift"], [s1, xb, k0, k48])
        s2i = sn.bitcast(s2, "vector<32xbf16>", "vector<16xi32>")
        d = sn.tmp()
        sn.emit(
            f"{d} = llvm.shufflevector {s2i}, {s2i} [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7] : vector<16xi32>"
        )
        m52, m53 = sn.const(52, "i32"), sn.const(53, "i32")
        lo = sn.call(_I["vshuffle"], [d, d, m52])
        hi = sn.call(_I["vshuffle"], [d, d, m53])
        c = sn.cat(lo, hi, 16, "i32")
        cb = sn.bitcast(c, "vector<32xi32>", "vector<64xbf16>")
        w = sn.write(cb, 8, "bf16")
        self.out[op] = self.expand(sn, op, [(w, "tensor<8x8xbf16>")])[0]

    def _rows(self, op):
        return _tensor_shape(op.results[0].type)[0]

    def _value(self, op, sn, w, rows, et):
        self.out[op] = self.expand(sn, op, [(w, f"tensor<{rows}x8x{et}>")])[0]

    def _op_from_vector(self, op, k):
        # ups of each 32 lanes
        rows = self._rows(op)
        lanes = 8 * rows
        sn = _Snippet(self)
        v = sn.read(op.operands[0], rows, "bf16")
        parts = []
        for h in range(lanes // 32):
            x = sn.part(v, 32 * h, 32, lanes, "bf16") if lanes > 32 else v
            parts.append(sn.call(_I["ups"], [x]))
        r = parts[0] if len(parts) == 1 else sn.cat(parts[0], parts[1], 32, "f32")
        self._value(op, sn, sn.write(r, rows, "f32"), rows, "f32")

    def _op_mac(self, op, k):
        # aie::mac(acc, a, b): bf.mac.conf(a, b, acc, 828); mul: bf.mul.conf(a, b, 828)
        kind = k
        rows = self._rows(op)
        sn = _Snippet(self)
        if kind == "mul":
            args = [
                sn.read(op.operands[0], rows, "bf16"),
                sn.read(op.operands[1], rows, "bf16"),
            ]
        else:
            c = sn.read(op.operands[0], rows, "f32")
            args = [
                sn.read(op.operands[1], rows, "bf16"),
                sn.read(op.operands[2], rows, "bf16"),
                c,
            ]
        cf = sn.const(828, "i32")
        r = sn.call(_bf_intrinsic(kind, 8 * rows), args + [cf])
        self._value(op, sn, sn.write(r, rows, "f32"), rows, "f32")

    _op_msc = _op_mac
    _op_mul = _op_mac

    def _add_or_lt(self, op, kind):
        # ups, the accfloat add on 64 lanes (conf 60), srs; lt: of a and
        # zero, then vltbf16 against b
        rows = self._rows(op)
        if 8 * rows != 32:
            raise ValueError(f"aie.{kind}: 32 lanes")
        sn = _Snippet(self)
        a = sn.read(op.operands[0], rows, "bf16")
        b = sn.read(op.operands[1], rows, "bf16")
        ua = sn.call(_I["ups"], [a])
        pa = sn.tmp()
        idx = ", ".join(map(str, range(32)))
        sn.emit(
            f"{pa} = llvm.shufflevector {ua}, {ua} [{idx}{', -1' * 32}] : vector<32xf32>"
        )
        if kind == "add":
            ub = sn.call(_I["ups"], [b])
            pb = sn.tmp()
            sn.emit(
                f"{pb} = llvm.shufflevector {ub}, {ub} [{idx}{', -1' * 32}] : vector<32xf32>"
            )
        else:
            pb = sn.const("dense<0.0>", "vector<64xf32>")
        k60 = sn.const(60, "i32")
        s = sn.call(_I["accadd"], [pa, pb, k60])
        h = sn.part(s, 0, 32, 64, "f32")
        r = sn.call(_I["srs32"], [h])
        if kind == "add":
            self._value(op, sn, sn.write(r, rows, "bf16"), rows, "bf16")
            return
        mk = sn.call(_I["vlt"], [r, b])
        ph = sn.tmp()
        sn.emit(f"{ph} = tensor.empty() : {op.results[0].type}")
        vals = self.expand(sn, op, [(mk, "i32"), (ph, str(op.results[0].type))])
        self.masks[op] = vals[0]
        self.out[op] = vals[1]

    def _op_add(self, op, k):
        self._add_or_lt(op, "add")

    def _op_lt(self, op, k):
        self._add_or_lt(op, "lt")

    def _op_abs(self, op, k):
        # the sign bits cleared
        rows = self._rows(op)
        n = 8 * rows
        sn = _Snippet(self)
        v = sn.read(op.operands[0], rows, "bf16")
        i = sn.bitcast(v, f"vector<{n}xbf16>", f"vector<{n // 2}xi32>")
        m = sn.const("dense<2147450879>", f"vector<{n // 2}xi32>")
        a = sn.tmp()
        sn.emit(f"{a} = arith.andi {i}, {m} : vector<{n // 2}xi32>")
        r = sn.bitcast(a, f"vector<{n // 2}xi32>", f"vector<{n}xbf16>")
        self._value(op, sn, sn.write(r, rows, "bf16"), rows, "bf16")

    def _op_tanh(self, op, k):
        # 16 lanes a call
        rows = self._rows(op)
        n = 8 * rows
        sn = _Snippet(self)
        x = sn.read(op.operands[0], rows, "f32")
        parts = [
            sn.call(_I[k], [sn.part(x, 16 * h, 16, n, "f32")]) for h in range(n // 16)
        ]
        width = 16
        while len(parts) > 1:
            parts = [
                sn.cat(parts[j], parts[j + 1], width, "bf16")
                for j in range(0, len(parts), 2)
            ]
            width *= 2
        self._value(op, sn, sn.write(parts[0], rows, "bf16"), rows, "bf16")

    _op_exp2 = _op_tanh

    def _op_bit_and(self, op, k):
        rows = self._rows(op)
        sn = _Snippet(self)
        a = sn.read(op.operands[0], rows, "i16")
        b = sn.read(op.operands[1], rows, "i16")
        r = sn.tmp()
        sn.emit(
            f"{r} = arith.{'andi' if k == 'bit_and' else 'ori'} {a}, {b} : vector<{8 * rows}xi16>"
        )
        self._value(op, sn, sn.write(r, rows, "i16"), rows, "i16")

    _op_bit_or = _op_bit_and

    def _op_select(self, op, k):
        rows = self._rows(op)
        if 8 * rows != 32:
            raise ValueError("aie.select: 32 lanes")
        mk = self.masks[_def(op.operands[2])]
        sn = _Snippet(self)
        a = sn.read(op.operands[0], rows, "bf16")
        b = sn.read(op.operands[1], rows, "bf16")
        ai = sn.bitcast(a, "vector<32xbf16>", "vector<32xi16>")
        bi = sn.bitcast(b, "vector<32xbf16>", "vector<32xi16>")
        r = sn.call(_I["vsel"], [ai, bi, sn.arg(mk)])
        rb = sn.bitcast(r, "vector<32xi16>", "vector<32xbf16>")
        self._value(op, sn, sn.write(rb, rows, "bf16"), rows, "bf16")

    def _op_unpack(self, op, k):
        # each 32 bytes of nibbles to 64 bytes
        sn = _Snippet(self)
        v = sn.read(op.operands[0], 8, "i8")
        k0 = sn.const(0, "i32")
        us = [
            sn.call(_I["unpack"], [sn.part(v, 32 * h, 32, 64, "i8"), k0])
            for h in range(2)
        ]
        c = sn.cat(us[0], us[1], 64, "i8")
        ci = sn.bitcast(c, "vector<128xi8>", "vector<64xi16>")
        self._value(op, sn, sn.write(ci, 8, "i16"), 8, "i16")

    def _op_zip_first(self, op, k):
        # interleave_zip(a, b, 1) of 8-bit lanes: of a's and b's low (first)
        # or high (second) 64 bytes, vshuffle 20 and 21
        h = 0 if k == "zip_first" else 1
        sn = _Snippet(self)
        halves = []
        for v in op.operands[0:2]:
            x = sn.read(v, 8, "i16")
            xi = sn.bitcast(x, "vector<64xi16>", "vector<32xi32>")
            halves.append(sn.part(xi, 16 * h, 16, 32, "i32"))
        m20, m21 = sn.const(20, "i32"), sn.const(21, "i32")
        lo = sn.call(_I["vshuffle"], [halves[0], halves[1], m20])
        hi = sn.call(_I["vshuffle"], [halves[0], halves[1], m21])
        c = sn.cat(lo, hi, 16, "i32")
        ci = sn.bitcast(c, "vector<32xi32>", "vector<64xi16>")
        self._value(op, sn, sn.write(ci, 8, "i16"), 8, "i16")

    _op_zip_second = _op_zip_first

    def _op_vector_cast(self, op, k):
        rows, ei = _tensor_shape(op.operands[0].type)
        _, eo = _tensor_shape(op.results[0].type)
        n = 8 * rows
        sn = _Snippet(self)
        v = sn.read(op.operands[0], rows, ei)
        c = sn.bitcast(v, f"vector<{n}x{ei}>", f"vector<{n}x{eo}>")
        self._value(op, sn, sn.write(c, rows, eo), rows, eo)

    def _op_mul_8x8_8x8T(self, op, k):
        # ::mul_8x8_8x8T: BFP576 x BFP576 -> ACC2048 mul.conf 780
        (am, ae), (bm, be) = (self.blocks[_def(v)] for v in op.operands[0:2])
        sn = _Snippet(self)
        x = [sn.arg(v) for v in (am, ae, bm, be)]
        fn = (
            "llvm.aie2p.BFP576.BFP576.ACC2048.mul.conf",
            "vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8>, i32",
            "vector<64xi32>",
        )
        r = sn.call(fn, x + [sn.const(780, "i32")])
        f = sn.bitcast(r, "vector<64xi32>", "vector<64xf32>")
        self._value(op, sn, sn.write(f, 8, "f32"), 8, "f32")

    def _op_mask_all(self, op, k):
        sn = _Snippet(self)
        h = sn.const(-1 if k.endswith(":1") else 0, "i64")
        self.out[op] = self.expand(sn, op, [(sn.mask_value(h), "tensor<8x8xi32>")])[0]

    def _op_cmr8(self, op, k):
        # a constant the core loads, as a C kernel's static table
        self.globals[_CMR8] = _CMR8_GLOBAL
        sn = _Snippet(self)
        g, v = sn.tmp(), sn.tmp()
        sn.emit(f"{g} = memref.get_global @{_CMR8} : memref<64xi16, 2 : i32>")
        c0, z = sn.const(0, "index"), sn.const(0, "i16")
        sn.emit(
            f"{v} = vector.transfer_read {g}[{c0}], {z} {{in_bounds = [true]}} : memref<64xi16, 2 : i32>, vector<64xi16>"
        )
        self._value(op, sn, sn.write(v, 8, "i16"), 8, "i16")

    def _op_t(self, op, k):
        """A template op, "t:<template>:<result lanes or m>:<operand lanes,
        s (a scalar fill) or m (a mask)>": the template's IR on the
        operands' lanes, cut to its arguments' widths."""
        _, name, rl, spec = k.split(":")
        tm = self.templates()
        widths = tm.arg_widths(name)
        sn = _Snippet(self)
        ins, j = [], 0
        for i, w in enumerate(spec.split(",")):
            v = op.operands[i]
            if w == "s":
                d = _def(v)
                if d is None or d.name != "linalg.fill":
                    raise ValueError(f"aie.{name}: operand {i} is not a tl.full")
                ins.append((sn.arg(d.operands[0]), str(d.operands[0].type)))
                j += 1
                continue
            if w == "m":
                ins.append((sn.mask_of(v), "i64"))
                j += 1
                continue
            lanes, (rows, et) = int(w), _tensor_shape(v.type)
            # (an operand Triton broadcast from [1, 8]: its 8 lanes are its
            # source's)
            src = self._broadcast_of(v) if lanes <= 8 else None
            if src is not None:
                v, rows = src, 1
            if lanes > 8 * rows:
                raise ValueError(
                    f"aie.{name}: operand {i} has fewer than {lanes} lanes"
                )
            x = sn.read(v, rows, et)
            off = 0
            while off < lanes:
                if j >= len(widths):
                    raise ValueError(
                        f"aie.{name}: more operand lanes than its arguments take"
                    )
                n = widths[j]
                part = (
                    x
                    if off == 0 and n == 8 * rows
                    else sn.part(x, off, n, 8 * rows, et)
                )
                ins.append((part, f"vector<{n}x{et}>"))
                off, j = off + n, j + 1
            if off != lanes:
                raise ValueError(
                    f"aie.{name}: operand {i}'s {lanes} lanes do not fill its arguments"
                )
        if j != len(widths):
            raise ValueError(f"aie.{name}: {j} of its {len(widths)} arguments")
        vals = self.expand(sn, op, ins)
        outs = tm.instantiate(name, vals, op, self.decls)
        # the stored parts, as the result's element type, one after another
        sn = _Snippet(self)
        if rl == "m":
            if len(outs) != 1:
                raise ValueError(
                    f"aie.{name}: a mask template stores {len(outs)} values"
                )
            self.out[op] = self.expand(
                sn, op, [(sn.mask_value(sn.arg(outs[0])), "tensor<8x8xi32>")]
            )[0]
            return
        rows, eo = _tensor_shape(op.results[0].type)
        parts = []
        for o in outs:
            n, e = re.fullmatch(r"vector<(\d+)x(\w+)>", str(o.type)).groups()
            n = int(n) * _BITS[e] // _BITS[eo]
            a = sn.arg(o)
            if e != eo:
                t = sn.tmp()
                sn.emit(f"{t} = llvm.bitcast {a} : {o.type} to vector<{n}x{eo}>")
                a = t
            parts.append((a, n))
        while len(parts) > 1:
            (x, n), (y, n2) = parts[0], parts[1]
            if n != n2:
                raise ValueError(f"aie.{name}: parts of {n} and {n2} lanes")
            parts = [(sn.cat(x, y, n, eo), 2 * n)] + parts[2:]
        v, n = parts[0]
        if n != int(rl):
            raise ValueError(f"aie.{name}: {n} lanes, not {rl}")
        if n < 8 * rows:
            t = sn.tmp()
            sn.emit(
                f"{t} = llvm.shufflevector {v}, {v} [{', '.join(map(str, range(n)))}{', -1' * (8 * rows - n)}] : vector<{n}x{eo}>"
            )
            v = t
        self._value(op, sn, sn.write(v, rows, eo), rows, eo)

    def templates(self):
        if self._templates is None:
            self._templates = _TemplateSet(self.m.context)
        return self._templates

    def _op_set_rounding(self, op, k):
        # clang: set.ctrl.reg(1, 12), the rounding mode conv_even
        sn = _Snippet(self)
        r, m = sn.const(1, "i32"), sn.const(12, "i32")
        sn.call(_I["set_ctrl"], [r, m])
        self.out[op] = self.expand(sn, op, [(sn.handle(), "tensor<8x8xi32>")])[0]

    def _marker(self, op, k):
        loops = _loops_around(op)
        if not loops:
            raise ValueError(f"aie.{k.split(':')[0]}: not in a loop")
        sn = _Snippet(self)
        self.out[op] = self.expand(sn, op, [(sn.handle(), "tensor<8x8xi32>")])[0]
        self.marks.append((loops[0], k))

    _op_function = _marker
    _op_unroll = _marker

    def _op_begin_function(self, op, k):
        sn = _Snippet(self)
        h = self.expand(sn, op, [(sn.handle(), "tensor<8x8xi32>")])[0]
        self.out[op] = h
        parts = k.split(":")
        self.spans.append([_def(h), None, parts[1], len(parts) > 2])

    def _op_end_function(self, op, k):
        sn = _Snippet(self)
        h = self.expand(sn, op, [(sn.handle(), "tensor<8x8xi32>")])[0]
        self.out[op] = h
        open_ = [sp for sp in self.spans if sp[1] is None]
        if not open_:
            raise ValueError("aie.end_function: no begin_function before it")
        open_[-1][1] = _def(h)

    def _mark(self, ops):
        """Loop attributes for the backend's post-schedule steps."""
        from air.ir import Attribute, StringAttr, UnitAttr

        for lp, k in self.marks:
            parts = k.split(":")
            if parts[0] == "function":
                lp.attributes["aie.fnloop"] = UnitAttr.get()
                lp.attributes["aie.fn"] = StringAttr.get(parts[1])
                if len(parts) > 2:
                    lp.attributes["aie.restrict"] = UnitAttr.get()
            elif parts[1] == "4" and _reads_stream(lp):
                lp.attributes["aie.k_loop"] = UnitAttr.get()
            elif parts[1] == "1":
                lp.attributes["loop_annotation"] = Attribute.parse(
                    "#llvm.loop_annotation<unroll = <disable = true>>"
                )
            else:
                lp.attributes["loop_annotation"] = Attribute.parse(
                    f"#llvm.loop_annotation<unroll = <count = {parts[1]} : i32>, mustProgress = true>"
                )

    def _vector_iter_args(self):
        """A loop's tensor<8x8xf32> iter_arg that its body reads whole once
        and writes whole once, the write yielded: carried as a vector."""
        changed = True
        while changed:
            changed = False
            for op in _ops_in_order(self.m):
                if op.name == "scf.for" and _vectorize_accumulators(op):
                    changed = True
                    break

    def _declare(self):
        """The intrinsics called, declared as llvm.func: the schedules
        match func.func ops, the kernel's alone."""
        from air.ir import InsertionPoint, Module

        have = {
            o.attributes["sym_name"].value
            for o in self.m.body.operations
            if o.operation.name in ("llvm.func", "func.func")
        }
        first = self.m.body.operations[0]
        for name, (ins, out) in self.decls.items():
            if name in have:
                continue
            d = Module.parse(_decl(name, ins, out))
            d.body.operations[0].operation.move_before(first)
        for name, text in self.globals.items():
            if name not in have:
                d = Module.parse(text)
                d.body.operations[0].operation.move_before(first)

    def _wrap_spans(self):
        """begin_function(NAME) ... end_function(), in one block: the ops
        between them in an scf.execute_region no_inline, marked as a
        function's loop is."""
        from air.ir import InsertionPoint, Operation, StringAttr, UnitAttr

        for b, e, name, restrict in self.spans:
            if e is None:
                raise ValueError(f"aie.begin_function({name}): no end_function")
            blk = [o.operation for o in b.block.operations]
            if e not in blk:
                raise ValueError(
                    f"aie.begin_function({name}): its end_function is in another block"
                )
            i, j = blk.index(b), blk.index(e)
            if j < i:
                raise ValueError(f"aie.begin_function({name}): end_function before it")
            ops = blk[i + 1 : j]
            region_op = Operation.create(
                "scf.execute_region", regions=1, ip=InsertionPoint(b)
            )
            region_op.attributes["no_inline"] = UnitAttr.get()
            region_op.attributes["aie.fnloop"] = UnitAttr.get()
            region_op.attributes["aie.fn"] = StringAttr.get(name)
            if restrict:
                region_op.attributes["aie.restrict"] = UnitAttr.get()
            body = region_op.regions[0].blocks.append()
            y = Operation.create("scf.yield", ip=InsertionPoint(body))
            inner = set(ops)
            for o in ops:
                for r in o.results:
                    if any(not _within(u.owner, inner) for u in r.uses):
                        raise ValueError(
                            f"aie.begin_function({name}): a value made in it is used after it"
                        )
                o.move_before(y)
            b.erase()
            e.erase()

    def _vector_if_results(self):
        """An scf.if's [R, 8] tensor results (aie values, masks) carried as
        1-D vectors: each branch yields its value read whole, written whole
        after the if."""
        ifs = [
            op
            for op in _ops_in_order(self.m)
            if op.name == "scf.if" and any(_is_rows8(t) for t in op.results.types)
        ]
        for op in reversed(ifs):
            self._vectorize_if(op)

    def _vectorize_if(self, op):
        from air.dialects import scf
        from air.ir import InsertionPoint, Type

        types = [str(t) for t in op.results.types]
        conv = {k: _tensor_shape(t) for k, t in enumerate(types) if _is_rows8(t)}
        new_types = [
            Type.parse(f"vector<{8 * conv[k][0]}x{conv[k][1]}>" if k in conv else t)
            for k, t in enumerate(types)
        ]
        new = scf.IfOp(op.operands[0], new_types, has_else=True, ip=InsertionPoint(op))
        for old_b, new_b in (
            (op.regions[0].blocks[0], new.then_block),
            (op.regions[1].blocks[0], new.else_block),
        ):
            y = list(old_b.operations)[-1].operation
            vals = list(y.operands)
            sn = _Snippet(self)
            outs = [(sn.read(vals[k], *conv[k]), str(new_types[k])) for k in conv]
            for k, v in zip(conv, self.expand(sn, y, outs)):
                vals[k] = v
            ny = scf.YieldOp(vals, ip=InsertionPoint(new_b))
            for o in list(old_b.operations)[:-1]:
                o.operation.move_before(ny)
        # after it: each vector written back into a tensor
        sn = _Snippet(self)
        outs = [(sn.write(sn.arg(new.results[k]), *conv[k]), types[k]) for k in conv]
        backs = dict(zip(conv, self.expand(sn, op, outs)))
        for k, r in enumerate(op.results):
            r.replace_all_uses_with(backs[k] if k in conv else new.results[k])
        op.erase()


def _is_rows8(t):
    return re.fullmatch(r"tensor<\d+x8x\w+>", str(t)) is not None


def _within(op, ops):
    """op, or an op it is nested in, is one of `ops`."""
    while op is not None:
        if op in ops:
            return True
        op = op.parent
    return False


def _func_type(t):
    """(argument types, result type or None) of an !llvm.func type."""
    m = re.fullmatch(r"!llvm\.func<(.*?) \((.*)\)>", str(t))
    # (inside an !llvm.func, LLVM types are printed without their prefix)
    full = lambda x: re.sub(r"(?<![!\w.])(struct<|array<|ptr\b)", r"!llvm.\1", x)
    return full(m[2]), (None if m[1] == "void" else full(m[1]))


class _TemplateSet:
    """The template ops' IR (aie_templates/ops.mlir): one llvm.func a
    template, its arguments the operands, its results stored through its
    last argument."""

    def __init__(self, ctx):
        from air.ir import Module

        with open(_TEMPLATES) as f:
            self.m = Module.parse(f.read(), context=ctx)
        self.funcs = {
            o.operation.attributes["sym_name"].value: o.operation
            for o in self.m.body.operations
            if o.operation.name == "llvm.func"
        }
        # the intrinsics' declarations: name -> (argument types, result type)
        self.decls = {
            n: _func_type(f.attributes["function_type"].value)
            for n, f in self.funcs.items()
            if n.startswith("llvm.aie2p.")
        }

    def arg_widths(self, name):
        """The lanes of each argument but the last (1 for a scalar)."""
        if name not in self.funcs:
            raise ValueError(f"aie: no template {name}")
        ins, _ = _func_type(self.funcs[name].attributes["function_type"].value)
        out = []
        for t in re.split(r",\s*(?![^<]*>)", ins)[:-1]:
            m = re.fullmatch(r"vector<(\d+)x\w+>", t.strip())
            out.append(int(m[1]) if m else 1)
        return out

    def instantiate(self, name, operands, at, decls):
        """The template's body before op `at` on `operands`: the values it
        stores, in the order of their offsets, which must cover its output
        from byte 0 without a gap. The intrinsics it calls go into
        `decls`. Anything else the template does with its output (a load,
        a GEP that is not one byte index) raises."""
        from air.ir import DenseI32ArrayAttr, InsertionPoint, IntegerAttr

        c = self.funcs[name].clone(ip=InsertionPoint(self.m.body))
        block = c.regions[0].blocks[0]
        args = list(block.arguments)
        for a, v in zip(args[:-1], operands):
            a.replace_all_uses_with(v)
        offset, stores, consts, drop = {args[-1]: 0}, {}, {}, []
        for o in list(block.operations):
            o = o.operation
            if o.name == "llvm.mlir.constant" and str(o.results[0].type).startswith(
                "i"
            ):
                consts[o.results[0]] = IntegerAttr(o.attributes["value"]).value
            if o.name == "llvm.getelementptr" and o.operands[0] in offset:
                idx = list(DenseI32ArrayAttr(o.attributes["rawConstantIndices"]))
                if len(idx) != 1 or str(o.attributes["elem_type"].value) != "i8":
                    raise ValueError(
                        f"aie: template {name}: a GEP of its output other than one byte index"
                    )
                if idx[0] == -(2**31):
                    if o.operands[1] not in consts:
                        raise ValueError(
                            f"aie: template {name}: a GEP of its output by a value not known"
                        )
                    idx[0] = consts[o.operands[1]]
                offset[o.results[0]] = offset[o.operands[0]] + idx[0]
                drop.append(o)
            elif o.name == "llvm.store" and o.operands[1] in offset:
                stores[offset[o.operands[1]]] = o.operands[0]
                drop.append(o)
            elif o.name == "llvm.return":
                if len(o.operands):
                    raise ValueError(f"aie: template {name} returns a value")
                drop.append(o)
            elif o.name == "llvm.call":
                callee = o.attributes["callee"].value
                decls[callee] = self.decls[callee]
        for v in offset:
            if any(u.owner not in drop for u in v.uses):
                raise ValueError(
                    f"aie: template {name} uses its output other than by stores"
                )
        at_byte = 0
        for k in sorted(stores):
            if k != at_byte:
                raise ValueError(
                    f"aie: template {name}: its output has a gap at byte {at_byte}"
                )
            at_byte += _bytes(stores[k].type)
        for o in list(block.operations):
            if o.operation not in drop:
                o.operation.move_before(at)
        for o in reversed(drop):
            o.erase()
        c.erase()
        return [stores[k] for k in sorted(stores)]


def _bytes(t):
    """The bytes of a vector or scalar type."""
    m = re.fullmatch(r"vector<(\d+)x(\w+)>", str(t))
    n, e = (int(m[1]), m[2]) if m else (1, str(t))
    return n * _BITS[e] // 8


# ops (on values, without effects) that make the aie ops' operands, erased
# once nothing uses them
_PURE = (
    "arith.addi",
    "arith.constant",
    "arith.extsi",
    "arith.muli",
    "linalg.fill",
    "linalg.generic",
    "memref.cast",
    "memref.memory_space_cast",
    "ptr.to_ptr",
    "tensor.empty",
    "tptr.ptrtoint",
)


def _erase_unused(ops):
    """Erase each op of `ops` (None for a block argument) that has no use
    left and only makes a value, then the ones that fed it."""
    # (an op is known by its hash, taken while it lives: a wrapper of an
    # erased op cannot be compared)
    work = [(hash(o), o) for o in ops if o is not None]
    gone = set()
    while work:
        key, op = work.pop()
        if key in gone or op.name not in _PURE:
            continue
        if any(len(list(r.uses)) for r in op.results):
            continue
        work += [(hash(d), d) for d in (_def(v) for v in op.operands) if d is not None]
        gone.add(key)
        op.erase()


def _reads_stream(lp):
    found = []

    def visit(op):
        if op.name == "aievec.fifo_ld_pop_bfp16":
            found.append(op)
        for r in op.regions:
            for b in r.blocks:
                for o in b.operations:
                    visit(o.operation)

    visit(lp)
    return bool(found)


def _copy_attributes(frm, to):
    for name in list(frm.attributes):
        to.attributes[name] = frm.attributes[name]


def _yield_of(for_op):
    return list(for_op.regions[0].blocks[0].operations)[-1].operation


def _with_iter_args(old, inits):
    """old (an scf.for) rebuilt with the iter_args `inits` after its own:
    (the new loop, its new block arguments). Its yield is left as it was,
    for `_append_yield`."""
    from air.dialects import scf
    from air.ir import InsertionPoint, Operation

    old_inits = list(old.operands[3:])
    new = scf.ForOp(
        old.operands[0],
        old.operands[1],
        old.operands[2],
        old_inits + inits,
        ip=InsertionPoint(old),
    )
    nb = new.body
    ob = old.regions[0].blocks[0]
    anchor = Operation.create("aieops.anchor", ip=InsertionPoint(nb))
    for a, b in zip(ob.arguments, list(nb.arguments)[: len(ob.arguments)]):
        a.replace_all_uses_with(b)
    for o in list(ob.operations):
        o.operation.move_before(anchor)
    anchor.erase()
    for a, b in zip(old.results, list(new.results)[: len(old.results)]):
        a.replace_all_uses_with(b)
    _copy_attributes(old, new.operation)
    old.erase()
    return new.operation, list(nb.arguments)[len(ob.arguments) :]


def _append_yield(for_op, values):
    from air.dialects import scf
    from air.ir import InsertionPoint

    y = _yield_of(for_op)
    scf.YieldOp(list(y.operands) + list(values), ip=InsertionPoint(y))
    y.erase()


def _vectorize_accumulators(lp):
    """Carry lp's accumulator iter_args as vectors (_Lowering._vector_iter_args);
    True if it changed lp."""
    from air.dialects import scf
    from air.ir import InsertionPoint, Operation

    tt, vt = "tensor<8x8xf32>", "vector<8x8xf32>"
    body = lp.regions[0].blocks[0]
    args = list(body.arguments)[1:]
    y = _yield_of(lp)
    change = {}
    for k, a in enumerate(args):
        if str(a.type) != tt:
            continue
        users = [u.owner for u in a.uses]
        if len(users) != 2:
            continue
        rd = next((u for u in users if u.name == "vector.transfer_read"), None)
        wr = next((u for u in users if u.name == "vector.transfer_write"), None)
        if rd is None or wr is None or str(rd.results[0].type) != vt:
            continue
        if wr.operands[1] != a or y.operands[k] != wr.results[0]:
            continue
        if len(list(wr.results[0].uses)) != 1 or str(wr.operands[0].type) != vt:
            continue
        if not _whole(rd) or not _whole(wr):
            continue
        change[k] = (rd, wr)
    if not change:
        return False
    inits = list(lp.operands[3:])
    new_inits = list(inits)
    for k, (rd, _) in change.items():
        # the init read whole before the loop, as the body read the arg
        r = rd.clone(ip=InsertionPoint(lp))
        _copy_constants(r, lp)
        r.operands[0] = inits[k]
        new_inits[k] = r.results[0]
    new = scf.ForOp(
        lp.operands[0], lp.operands[1], lp.operands[2], new_inits, ip=InsertionPoint(lp)
    )
    nb = new.body
    nargs = list(nb.arguments)
    body.arguments[0].replace_all_uses_with(nargs[0])
    yvals = list(y.operands)
    backs = {}
    for k, a in enumerate(args):
        if k not in change:
            a.replace_all_uses_with(nargs[k + 1])
            continue
        rd, wr = change[k]
        # after the loop (before the old one, which goes): the vector
        # written back into the init's tensor, as the body wrote the arg
        w = wr.clone(ip=InsertionPoint(lp))
        _copy_constants(w, lp)
        w.operands[0] = new.results[k]
        w.operands[1] = inits[k]
        backs[k] = w.results[0]
        rd.results[0].replace_all_uses_with(nargs[k + 1])
        yvals[k] = wr.operands[0]
    anchor = Operation.create("aieops.anchor", ip=InsertionPoint(nb))
    for o in list(body.operations)[:-1]:
        o.operation.move_before(anchor)
    scf.YieldOp(yvals, ip=InsertionPoint(anchor))
    anchor.erase()
    y.erase()
    for rd, wr in change.values():
        wr.erase()
        rd.erase()
    _copy_attributes(lp, new.operation)
    for k, old in enumerate(lp.results):
        old.replace_all_uses_with(backs[k] if k in change else new.results[k])
    lp.erase()
    return True


def _whole(op):
    """A transfer of a whole [8, 8] tensor at [0, 0], in bounds, unmasked,
    in its own order."""
    from air.ir import IntegerAttr

    write = op.name == "vector.transfer_write"
    if len(op.operands) != 4:  # (a mask)
        return False
    if str(op.attributes["permutation_map"]) != "affine_map<(d0, d1) -> (d0, d1)>":
        return False
    idx = list(op.operands)[2:4] if write else list(op.operands)[1:3]
    for v in idx:
        d = _def(v)
        if (
            d is None
            or d.name != "arith.constant"
            or IntegerAttr(d.attributes["value"]).value != 0
        ):
            return False
    return str(op.attributes["in_bounds"]) == "[true, true]"


def _copy_constants(op, lp):
    """op's operands that are constants defined in lp (op is outside it)
    copied in front of op."""
    from air.ir import InsertionPoint

    for i, v in enumerate(list(op.operands)):
        d = _def(v)
        if d is not None and d.name == "arith.constant" and lp in _ancestors(d) + [d]:
            op.operands[i] = d.clone(ip=InsertionPoint(op)).results[0]
