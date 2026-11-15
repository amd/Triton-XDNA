# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""The AIR module of a superkernel (see `tl.extra.npu.spatial`): several
kernels in one NPU dispatch that loops over waves.

`superkernel_jit` reads a superkernel's TTIR into a `Superkernel`: its
buffers, and one call per kernel in the order its partitions list them. Each
call is compiled like a `FusedDevice` op, and the bodies of their segments are
merged into one segment that the runtime sequence launches once per wave:

    func @name(ddr buffers, scalars) {
      scf.for wave = 0 .. waves {
        air.launch { air.segment @seg { every call's body } }
      }
    }

The cores are configured once and stay resident; each wave only writes their
run-time parameters. An on-chip tensor has no DDR buffer. It is a single L2
buffer: where a call on its own would copy to or from DDR, its writers copy
into the L2 buffer and its readers copy out of it. DDR tensors are the func's
buffer arguments, read and written in place every wave.
"""

import os
import re

from .arg_access import read_args, written_args
from .compiler import _optimize_ttsharedir, _ttir_to_ttsharedir
from .config import config_context
from .driver import _aircc_compile, _ttshared_to_air, detect_npu_version
from .fused_device import (
    SEG,
    _drop_unused_scalars,
    _entry,
    _herd_width,
    _ops_named,
    _pin_l2,
)


class _Wave:
    """An int argument worth `a * wave + b` at wave `wave`."""

    def __init__(self, a=1, b=0):
        self.a, self.b = a, b

    def __repr__(self):
        return f"wave*{self.a}+{self.b}"


class _Table(_Wave):
    """An int argument worth `values[wave]`, for per-wave values that are not
    affine in the wave's index. Lowered as a chain of selects on the wave's
    index."""

    def __init__(self, values):
        super().__init__()
        self.table = tuple(int(v) for v in values)

    def __repr__(self):
        return f"table{self.table}"


class _Scalar:
    """An int argument of the func, set by the host at each dispatch."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"scalar({self.name})"


class Tensor:
    """A flat tensor of `n` elements of `dtype`, an MLIR element type such as
    "bf16"."""

    def __init__(self, name, n, dtype, onchip):
        self.name, self.n, self.dtype, self.onchip = name, n, dtype, onchip

    def __repr__(self):
        return f"{'onchip' if self.onchip else 'ddr'}({self.name})"


class _Call:
    """One call: its kernel's name, its compiled AIR module and how the
    module's func arguments bind to the superkernel's tensors and ints."""

    def __init__(self, name, at, herd, air, binds, read, written):
        self.name, self.at, self.herd, self.air = name, at, herd, air
        self.binds = binds  # per func argument: a Tensor, _Wave or _Scalar
        self.read = read  # the tensors the call reads
        self.written = written  # the tensors the call writes


def _check_phase_order(calls, g):
    """Reject phases `g` that cannot all run where the first one is. The
    phase loop runs every phase at that point, ahead of the calls listed
    between the phases; a DDR buffer that such a call writes must not be
    used by a later phase, nor one it reads be written by a later phase."""
    at = [next(i for i, c in enumerate(calls) if c is x) for x in g]
    for k in range(1, len(g)):
        for c in calls[at[k - 1] + 1 : at[k]]:
            if any(c is x for x in g):
                continue
            for later in g[k:]:
                uses = later.read + later.written
                clash = [t for t in c.written if not t.onchip and t in uses]
                clash += [t for t in c.read if not t.onchip and t in later.written]
                if clash:
                    raise ValueError(
                        f"herd {g[0].herd}: a later phase uses {clash[0]}, "
                        f"which {c.name}, listed between the phases, also "
                        "accesses; the phase loop would run that phase "
                        "before it"
                    )


class Superkernel:
    """A dispatch named `name` that runs `waves` waves of its calls. Build
    artifacts go to `air_project_path`."""

    def __init__(self, name, air_project_path, waves):
        self.name, self.proj, self.waves = (
            name,
            os.path.abspath(air_project_path),
            waves,
        )
        self.tensors, self.calls = [], []
        self.scalars = []  # the func's int arguments after its buffers

    def scalar(self, name):
        """A new int argument of the func, set by the host at each dispatch."""
        v = _Scalar(name)
        self.scalars.append(v)
        return v

    def ddr(self, name, n, dtype):
        """A new buffer argument of the func, in DDR."""
        return self._tensor(name, n, dtype, False)

    def onchip(self, name, n, dtype):
        """A new tensor kept on chip in one L2 buffer."""
        return self._tensor(name, n, dtype, True)

    def _tensor(self, name, n, dtype, onchip):
        if any(t.name == name for t in self.tensors):
            raise ValueError(f"a tensor named {name} exists")
        t = Tensor(name, n, dtype, onchip)
        self.tensors.append(t)
        return t

    @property
    def buffers(self):
        """The func's buffer arguments, in order."""
        return [t for t in self.tensors if not t.onchip]

    def call(self, name, ttir, args, at, herd, schedule, cores):
        """Add a call to the wave: the kernel `name` with TTIR `ttir` (one
        public function), on `args` (per argument of that function: a Tensor,
        _Wave, _Scalar or number), on the herd `herd` with corner `at`, compiled
        with the transform script at path `schedule` into `cores` (columns,
        rows) cores."""
        ttshared = _optimize_ttsharedir(_ttir_to_ttsharedir(ttir), {})
        with config_context(transform_tiling_script=schedule):
            m = _ttshared_to_air(ttshared, 1, 1, 1)
        for h in _ops_named(_entry(m), "air.herd"):
            shape = (_herd_width(h), _herd_height(h))
            if shape != tuple(cores):
                raise ValueError(
                    f"{name}: its schedule makes a {shape[0]}x{shape[1]} herd "
                    f"for {cores[0]}x{cores[1]} cores"
                )
        _bind_constants(m, args)
        kept = _drop_unused_scalars(_entry(m))
        # The func's arguments are the kernel's, then the grid's ints, which
        # triton-shared adds; `kept` says which survived.
        named = list(args) + [None] * (len(kept) - len(args))
        binds = [a for a, k in zip(named, kept) if k]
        names = [f"a{i}" for i in range(len(args))]
        w = written_args(ttir, names)
        if w is None:
            raise ValueError(f"{name}: can't tell what it writes")
        r = read_args(ttir, names)
        if r is None:  # every pointer argument may be read
            r = set(names)
        tensors = [(n, a) for n, a in zip(names, args) if isinstance(a, Tensor)]
        written = [a for n, a in tensors if n in w]
        read = [a for n, a in tensors if n in r]
        for c2 in self.calls:
            if c2.herd == herd and c2.at != tuple(at):
                raise ValueError(f"herd {herd} placed at {c2.at} and {at}")
        self.calls.append(_Call(name, tuple(at), herd, str(m), binds, read, written))

    def _check_writers(self):
        """Every on-chip tensor a call reads is written by some call."""
        for c in self.calls:
            for t in c.read:
                if t.onchip and not any(t in x.written for x in self.calls):
                    raise ValueError(f"{c.name} reads {t}, which no call writes")

    def _phases(self):
        """The calls grouped into phases. Calls of one kernel at one shape on
        the same herd become the trips of one loop over phases. Returns the
        groups, each a list of calls in order."""
        self._check_writers()
        groups = {}
        for c in self.calls:
            # Phases may differ in the numbers the segment computes with, such
            # as trip counts and strides; _phase_numbers checks that nothing
            # else differs. Numbers are masked here, and so are SSA names,
            # which the printer derives from constant values.
            body = re.sub(r"%[\w.]+", "%v", re.sub(r"\d+", "N", c.air))
            groups.setdefault((c.herd, body), []).append(c)
        for g in groups.values():
            _check_phase_order(self.calls, g)
            for i in range(len(g[0].binds)):
                vals = [c.binds[i] for c in g]
                if all(v is vals[0] for v in vals) or all(
                    isinstance(v, _Wave) for v in vals
                ):
                    continue
                if not all(isinstance(v, Tensor) and v.onchip for v in vals):
                    raise ValueError(
                        f"herd {g[0].herd}: its phases take {vals}; only on-chip "
                        "tensors and the wave may differ between phases"
                    )
        return list(groups.values())

    def module_text(self):
        """The AIR module of the whole dispatch, as text."""
        from air.ir import (
            BoolAttr,
            Context,
            IntegerAttr,
            IntegerType,
            Location,
            Module,
        )

        bufs = self.buffers
        onchip = [t for t in self.tensors if t.onchip]
        with Context(), Location.unknown():
            i32 = IntegerType.get_signless(32)
            top = Module.parse(self._skeleton(bufs, onchip))
            seg = _ops_named(_entry(top), "air.segment")[0]
            sblock = seg.regions[0].blocks[0]
            # The segment's arguments: the buffers, the wave, the scalars.
            sargs = list(sblock.arguments)
            allocs = [
                o for o in sblock.operations if o.operation.name == "memref.alloc"
            ]
            end = next(
                o for o in sblock.operations if o.operation.name == "memref.dealloc"
            )
            value = {id(t): sargs[i] for i, t in enumerate(bufs)}
            for t, a in zip(onchip, allocs):
                value[id(t)] = a.result
            for a in allocs:
                # Not shrunk to its current uses: the copies that read and
                # write it are only made later.
                a.attributes["air.shrinkage"] = BoolAttr.get(False)
            wave = sargs[len(bufs)]
            for i, v in enumerate(self.scalars):
                value[id(v)] = sargs[len(bufs) + 1 + i]
            phased = {}  # on-chip tensor id -> (phase loop, "in" or "out", tensors)
            loops = {}  # phase loop key -> (loop, its yield)
            for g in self._phases():
                self._add_phase_group(top, g, end, value, wave, phased, loops)
            _hoist_ddr_reads(sblock, sargs[: len(bufs)], end)
            # A herd whose kernel took a DDR buffer directly now takes the
            # on-chip tensor's L2 buffer instead.
            _retype_herd_args(seg)
            # A one-column herd's own L2 buffers go in its column's memtile.
            # On-chip tensors keep their placement, since herds in several
            # columns may read them.
            for h in _ops_named(seg, "air.herd"):
                if _herd_width(h) == 1:
                    _pin_l2([h], keep=True)
            edge_buffers = [value[id(t)] for t in onchip]
            _check_edge_loops(seg, edge_buffers)
            by_buffer = {}
            for t in self.tensors:
                if id(t) in phased:
                    loop, kind, per = phased[id(t)]
                    per = [value[id(x)] for x in per]
                    by_buffer[_edge_key(value[id(t)], edge_buffers)] = (loop, kind, per)
            _lower_edges(top, seg, edge_buffers, by_buffer)
            for n, d in enumerate(_ops_named(seg, "air.dma_memcpy_nd")):
                d.attributes["id"] = IntegerAttr.get(i32, n + 1)
            return str(top)

    def _add_phase_group(self, top, g, end, value, wave, phased, loops):
        """Move the segment body of the calls `g` (one call, or the phases of
        one herd) into the superkernel's segment before `end`. Phases go into
        a phase loop, shared with other herds whose phases read the same
        on-chip inputs; `loops` holds those loops and `phased` records the
        on-chip tensors that change with the phase."""
        from air.dialects import arith
        from air.dialects.scf import ForOp, yield_
        from air.ir import (
            IndexType,
            InsertionPoint,
            IntegerAttr,
            IntegerType,
            Module,
            StringAttr,
            UnitAttr,
        )

        i64 = IntegerType.get_signless(64)
        i32 = IntegerType.get_signless(32)
        idx = IndexType.get()
        const32 = lambda v: arith.ConstantOp(i32, IntegerAttr.get(i32, v)).result
        c = g[0]
        m = Module.parse(c.air)
        # Bring over the symbols the call's cores use.
        entry = _entry(m)
        have = {str(o.attributes["sym_name"]) for o in top.body.operations}
        for o in list(m.body.operations):
            sym = str(o.attributes["sym_name"])
            if sym != str(entry.attributes["sym_name"]) and sym not in have:
                top.body.append(o)
        cseg = _ops_named(entry, "air.segment")[0]
        cblock = cseg.regions[0].blocks[0]
        at, ph = end, None
        diffs = (
            _phase_numbers(cseg, [Module.parse(x.air) for x in g[1:]])
            if len(g) > 1
            else []
        )
        # Herds whose phases read the same on-chip inputs, phase by phase,
        # share one phase loop, so that each input is filled once per trip
        # for all of them.
        key = tuple(
            tuple(
                id(b)
                for b in x.binds
                if isinstance(b, Tensor) and b.onchip and b not in x.written
            )
            for x in g
        )
        if len(g) > 1 and key in loops:
            loop, at = loops[key]
            ph = loop.induction_variable
            for i, v in enumerate(c.binds):
                if (
                    isinstance(v, Tensor)
                    and v.onchip
                    and any(x.binds[i] is not v for x in g)
                    and v in c.written
                ):
                    phased[id(v)] = (loop, "out", [x.binds[i] for x in g])
        elif len(g) > 1:
            with InsertionPoint(end):
                c0 = arith.ConstantOp(idx, IntegerAttr.get(idx, 0)).result
                cn = arith.ConstantOp(idx, IntegerAttr.get(idx, len(g))).result
                c1 = arith.ConstantOp(idx, IntegerAttr.get(idx, 1)).result
                loop = ForOp(c0, cn, c1)
            # No ping-pong across phases: a core with two phased inputs would
            # receive both phases of one input before the other.
            loop.attributes["air.disable_ping_pong"] = UnitAttr.get()
            with InsertionPoint(loop.body):
                at = yield_([])
            ph = loop.induction_variable
            loops[key] = (loop, at)
            for i, v in enumerate(c.binds):
                if (
                    isinstance(v, Tensor)
                    and v.onchip
                    and any(x.binds[i] is not v for x in g)
                ):
                    if v not in c.written:
                        raise ValueError(
                            f"herd {c.herd}: its phases read different on-chip "
                            "tensors, which is not supported"
                        )
                    phased[id(v)] = (loop, "out", [x.binds[i] for x in g])
        # Per phase, the values the launch binds to the segment's arguments:
        # a call's plain int arguments may differ by phase.
        bound = None
        if ph is not None:
            pms = [Module.parse(x.air) for x in g]  # kept alive
            bound = [
                _func_args_of(_ops_named(_entry(pm), "air.segment")[0]) for pm in pms
            ]
        with InsertionPoint(at):
            for k, (a, src) in enumerate(zip(cblock.arguments, _func_args_of(cseg))):
                if not isinstance(src, int):
                    vals = bound and [_const(b[k]) for b in bound]
                    if vals and None not in vals and len(set(vals)) > 1:
                        a.replace_all_uses_with(_phase_value(vals, ph, a.type))
                    else:
                        a.replace_all_uses_with(src.owner.clone().result)
                    continue
                b = c.binds[src]
                if isinstance(b, _Wave):
                    per = [x.binds[src] for x in g]
                    a.replace_all_uses_with(_wave_value(per, wave, ph, const32))
                else:
                    a.replace_all_uses_with(value[id(b)])
        for op in list(cblock.operations)[:-1]:  # all but the terminator
            op.operation.move_before(at)
            herds = _ops_named(op.operation, "air.herd")
            if op.operation.name == "air.herd":
                herds = [op.operation] + herds
            for h in herds:
                h.attributes["sym_name"] = StringAttr.get(c.herd)
                h.attributes["x_loc"] = IntegerAttr.get(i64, c.at[0])
                h.attributes["y_loc"] = IntegerAttr.get(i64, c.at[1])
        if diffs:
            with InsertionPoint.at_block_begin(loop.body):
                _apply_phase_numbers(diffs, ph)

    def _skeleton(self, bufs, edges):
        """MLIR text of the func with its wave loop, launch and segment. The
        segment holds only the on-chip tensors' L2 buffers."""
        bt = [f"memref<*x{t.dtype}>" for t in bufs]
        ns = len(self.scalars)
        fargs = ", ".join(
            [f"%a{i}: {t}" for i, t in enumerate(bt)]
            + [f"%r{i}: i32" for i in range(ns)]
        )
        types = ", ".join(bt + ["i32"] * (1 + ns))
        # The wave index is passed as an i32: an index argument would be
        # folded to its first wave's value.
        largs = ", ".join(
            [f"%b{i}=%a{i}" for i in range(len(bufs))]
            + ["%bw=%wi"]
            + [f"%br{i}=%r{i}" for i in range(ns)]
        )
        # Do not set air.preserve_shim_dma_order on the launch. It awaits
        # each feed in channel order, and a phase that needs another call's
        # output would then wait for a feed issued only after it.
        sargs = ", ".join(
            [f"%s{i}=%b{i}" for i in range(len(bufs))]
            + ["%sw=%bw"]
            + [f"%sr{i}=%br{i}" for i in range(ns)]
        )
        l2 = [f"memref<{t.n}x{t.dtype}, 1 : i32>" for t in edges]
        alloc = "".join(
            f"        %e{i} = memref.alloc() : {t}\n" for i, t in enumerate(l2)
        )
        free = "".join(
            f"        memref.dealloc %e{i} : {t}\n" for i, t in enumerate(l2)
        )
        # The calls' bodies go before the first dealloc.
        if not edges:
            alloc = "        %e = memref.alloc() : memref<1xi32, 1 : i32>\n"
            free = "        memref.dealloc %e : memref<1xi32, 1 : i32>\n"
        return f"""
module {{
  func.func @{self.name}({fargs}) {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cw = arith.constant {self.waves} : index
    scf.for %w = %c0 to %cw step %c1 {{
      %wi = arith.index_cast %w : index to i32
      air.launch (%l0, %l1, %l2) in (%l3=%c1, %l4=%c1, %l5=%c1) args({largs}) : {types} {{
        air.segment @{SEG} args({sargs}) : {types} attributes {{x_loc = 0 : i64, y_loc = 2 : i64}} {{
{alloc}{free}        }}
      }}
    }}
    return
  }}
}}
"""

    def compile(
        self,
        runtime_loop_tiling_sizes=(4, 4),
        bf16_emulation=None,
        extra_args=(),
        stack_size=2048,
    ):
        """Compile with aircc. Returns the xclbin path and the path of the
        generated TXN header."""
        npu = detect_npu_version()
        os.makedirs(os.path.join(self.proj, "air_project"), exist_ok=True)
        path = os.path.join(self.proj, f"{self.name}.air.mlir")
        with open(path, "w") as f:
            f.write(self.module_text())
        cwd = os.getcwd()
        os.chdir(self.proj)
        try:
            out = _aircc_compile(
                path,
                "xclbin",
                npu,
                self.proj,
                bf16_emulation=bf16_emulation,
                stack_size=stack_size,
                # Unrolled feed loops keep each wave's feeds in their own
                # descriptors; merging contiguous feeds keeps the shim task
                # queue short.
                extra_args=[
                    "--emit-txn-cpp",
                    "--unroll-feed-loops",
                    "--merge-contiguous-feeds",
                    *extra_args,
                ],
                runtime_loop_tiling_sizes=runtime_loop_tiling_sizes,
            )
        finally:
            os.chdir(cwd)
        header = os.path.join(self.proj, "air_project", f"npu.{self.name}.air.txn.h")
        return out["bin_path"], header


def _herd_height(h):
    """The row count of herd `h`, a constant operand."""
    return _const(h.operands[1])


def _bind_constants(m, args):
    """Replace the func's arguments that `args` gives as numbers by
    constants."""
    from air.dialects import arith
    from air.ir import FloatAttr, InsertionPoint, IntegerAttr, Location

    block = _entry(m).regions[0].blocks[0]
    with m.context, Location.unknown(m.context):
        ip = InsertionPoint.at_block_begin(block)
        for a, v in zip(block.arguments, args):
            if isinstance(v, float):
                attr = FloatAttr.get(a.type, v)
            elif isinstance(v, int):
                attr = IntegerAttr.get(a.type, int(v))
            else:
                continue
            c = arith.ConstantOp(a.type, attr, ip=ip)
            a.replace_all_uses_with(c.result)


def _retype_herd_args(seg):
    """Give each herd block argument its operand's type. They differ where an
    on-chip tensor's L2 buffer replaced a DDR buffer."""
    for h in _ops_named(seg, "air.herd"):
        body = h.regions[0].blocks[0]
        n = len(body.arguments) - len(h.operands)
        for i, v in enumerate(h.operands):
            a = body.arguments[n + i]
            if a.type != v.type:
                a.set_type(v.type)


def _func_args_of(seg):
    """For each block argument of segment `seg`, the index of the func
    argument the launch binds it to, or the bound value if it is not a func
    argument."""
    from air.ir import BlockArgument

    launch = seg.parent
    while launch.name != "air.launch":
        launch = launch.parent
    deps, nsizes, _ = list(launch.attributes["operandSegmentSizes"])
    out = []
    for v in seg.operands:
        # A launch's block arguments are its ids, its sizes, then one per
        # operand after its async dependencies and sizes.
        k = BlockArgument(v).arg_number - 2 * nsizes
        o = launch.operands[deps + nsizes + k]
        try:
            out.append(BlockArgument(o).arg_number)
        except ValueError:
            out.append(o)
    return out


def _segment_dmas(op):
    """The DMAs under `op` that are not inside a herd."""
    out = []
    for region in op.regions:
        for block in region:
            for o in block.operations:
                if o.operation.name == "air.dma_memcpy_nd":
                    out.append(o.operation)
                elif o.operation.name != "air.herd":
                    out += _segment_dmas(o.operation)
    return out


def _space(v):
    """The memory space of `v`'s memref type: 0 (DDR) for anything else."""
    from air.ir import IntegerAttr, MemRefType

    if not isinstance(v.type, MemRefType):
        return 0
    s = MemRefType(v.type).memory_space
    return 0 if s is None else IntegerAttr(s).value


def _pattern(static, dynamic):
    """Merge a static index list with its dynamic values: ints where static,
    Values where dynamic."""
    from air.ir import ShapedType

    dyn = iter(dynamic)
    return [next(dyn) if v == ShapedType.get_dynamic_size() else v for v in static]


def _dma_pattern(d, side):
    """Offsets, sizes and strides of one side ("src" or "dst") of DMA `d`."""
    from air.ir import DenseI64ArrayAttr

    return [
        _pattern(
            list(DenseI64ArrayAttr(d.attributes[f"static_{side}_{k}"])),
            getattr(d.opview, f"dynamic_{side}_{k}"),
        )
        for k in ("offsets", "sizes", "strides")
    ]


def _is_whole(op, side, n):
    """Whether `side` of DMA `op` covers a whole buffer of `n` elements in
    order."""
    from air.ir import DenseI64ArrayAttr

    pat = [
        list(DenseI64ArrayAttr(op.attributes[f"static_{side}_{k}"]))
        for k in ("offsets", "sizes", "strides")
    ]
    dyn = [
        len(getattr(op.opview, f"dynamic_{side}_{k}"))
        for k in ("offsets", "sizes", "strides")
    ]
    if any(dyn):
        return False
    return pat == [[], [], []] or pat == [[0], [n], [1]]


def _lower_edges(module, seg, edges, phased=None):
    """Lower the L2-to-L2 copies to and from the on-chip tensors' buffers
    (`edges`). A reader's copy of a whole edge buffer is dropped, the reader
    using the edge buffer directly. Every other copy becomes a channel.

    `phased` maps an edge buffer id to (phase loop, "out", the buffer of each
    phase): a phase loop's output, sent by packet to the buffer its phase
    writes."""
    from air.dialects.air import ChannelGet, ChannelPut, channel as channel_decl
    from air.ir import InsertionPoint, MemRefType

    phased = phased or {}
    n, names = 0, []
    hoisted = []  # (loop, the last copy taken out after it)

    def inside(op, loop):
        p = op.parent
        while p is not None and p.name != "air.segment":
            if p.operation == loop.operation:
                return True
            p = p.parent
        return False

    for d in _segment_dmas(seg):
        dst, src = d.opview.dst, d.opview.src
        if _space(dst) != 1 or _space(src) != 1:
            continue
        if (
            any(src == e for e in edges)
            and dst.owner.name == "memref.alloc"
            and dst.type == src.type
            and _is_whole(d, "src", MemRefType(src.type).shape[0])
            and _is_whole(d, "dst", MemRefType(dst.type).shape[0])
        ):
            alloc = dst.owner
            for u in list(dst.uses):
                if u.owner.name == "memref.dealloc":
                    u.owner.erase()
            d.erase()
            dst.replace_all_uses_with(src)
            alloc.erase()
            continue
        # Copies of the same region of a writer's buffer into the same edge
        # share a channel.
        where = tuple(_static(v) for v in _dma_pattern(d, "src")[0])
        key = (src, dst, where)
        name = next(
            (nm for k, nm in names if k[0] == src and k[1] == dst and k[2] == where),
            None,
        )
        if name is None:
            name = f"edge{n}"
            n += 1
            names.append((key, name))
            with InsertionPoint.at_block_begin(module.body):
                channel_decl(name, size=[1, 1])
        so, ss, st = _dma_pattern(d, "src")
        do, ds, dt = _dma_pattern(d, "dst")
        pdst = phased.get(_edge_key(dst, edges))
        if pdst and pdst[1] == "out" and inside(d, pdst[0]):
            # A phase's output goes by packet to its phase's buffer.
            loop, _, per = pdst
            if not _core_packet_out(module, loop, d, per, (do, ds, dt), f"out{n}"):
                raise ValueError(f"unsupported phase output: {d}")
            n += 1
            continue
        src_outer = _placing_loops(d, "src")
        if src_outer:
            # The writer's end must stay in the loop it shares with the writes
            # into its buffer, or its locks pair with the wrong writes.
            raise ValueError(
                "a copy into an on-chip tensor reads its writer's buffer at a "
                f"place that moves with a loop around cores: {d}"
            )
        with InsertionPoint(d):
            ChannelPut(name, src, offsets=so, sizes=ss, strides=st, indices=[0, 0])
        outer = _placing_loops(d, "dst")
        if not outer:
            with InsertionPoint(d):
                ChannelGet(name, dst, offsets=do, sizes=ds, strides=dt, indices=[0, 0])
        else:
            # The edge's end is a memtile DMA with fixed descriptors. Where a
            # loop moves it, it is taken out into a loop of its own over the
            # same trips, which aircc lowers to a ring of descriptors. The
            # writer's end stays with the writes that fill its buffer.
            anchor = next(
                (c for lp, c in hoisted if lp.operation == outer[0].operation), None
            )
            at = _after(anchor) if anchor is not None else _after(outer[0])
            with InsertionPoint(at):
                last = _get_loop(name, dst, (do, ds, dt), list(_trips(outer)))
            hoisted[:] = [h for h in hoisted if h[0].operation != outer[0].operation]
            hoisted.append((outer[0], last))
        d.erase()


# Offset of the payload in a core's packet buffer; the packet header is
# written just before the payload.
_HDR = 32


def _add_herd_operand(h, v):
    """Rebuild herd `h` with `v` as an extra operand. Returns the new herd and
    the block argument for `v`."""
    from air.dialects.air import HerdOp
    from air.ir import DenseI32ArrayAttr, InsertionPoint

    op = h.operation
    deps, nsizes, _ = list(DenseI32ArrayAttr(op.attributes["operandSegmentSizes"]))
    if deps:
        raise ValueError("an async herd at this stage")
    operands = list(op.operands)
    with InsertionPoint(op):
        new = HerdOp(None, [], operands[:nsizes], operands[nsizes:] + [v])
    for a in _attr_names(op):
        if a != "operandSegmentSizes":
            new.operation.attributes[a] = op.attributes[a]
    old = op.regions[0].blocks[0]
    block = new.regions[0].blocks.append(*([a.type for a in old.arguments] + [v.type]))
    for a, b in zip(old.arguments, block.arguments):
        a.replace_all_uses_with(b)
    ops = list(old.operations)
    for o in ops:
        o.operation.detach_from_parent()
        block.append(o.operation)
    op.erase()
    return new.operation, block.arguments[len(block.arguments) - 1]


def _core_packet_out(module, loop, d, per, dst_pattern, name):
    """Lower phase output copy `d` of a one-core herd whose phases write
    different on-chip tensors (`per`, one per phase). The core sends each
    phase's output as a packet whose destination is the phase's tensor, so
    no memtile program depends on the phase.

    Returns False without changing anything when the writer is not a single
    core or the copy is not a contiguous range of its buffer."""
    from air.dialects import arith, linalg, memref
    from air.dialects.air import Channel, ChannelGet, ChannelPut
    from air.ir import (
        DenseI64ArrayAttr,
        IndexType,
        InsertionPoint,
        IntegerAttr,
        IntegerType,
        MemRefType,
        StringAttr,
    )

    staging = d.opview.src
    herds = [
        h for h in _ops_named(loop, "air.herd") if any(o == staging for o in h.operands)
    ]
    if len(herds) != 1 or _herd_width(herds[0]) != 1:
        return False
    h = herds[0]
    sizes = [_const(v) for v in list(h.operands)[:2]]
    if sizes != [1, 1]:
        return False
    so = list(DenseI64ArrayAttr(d.attributes["static_src_offsets"]))
    ss = list(DenseI64ArrayAttr(d.attributes["static_src_sizes"]))
    st = list(DenseI64ArrayAttr(d.attributes["static_src_strides"]))
    if len(ss) != 1 or st != [1] or so[0] < 0 or ss[0] < 0:
        return False
    off, n = so[0], ss[0]
    k = next(i for i, o in enumerate(list(h.operands)[2:]) if o == staging)
    arg = h.regions[0].blocks[0].arguments[4 + k]
    writes = [
        o for o, _ in _walk(h) if o.name == "air.dma_memcpy_nd" and o.opview.dst == arg
    ]
    if len(writes) != 1:
        return False
    w = writes[0]
    l1 = w.opview.src
    if _space(l1) != 2 or w.attributes["static_src_sizes"] != DenseI64ArrayAttr.get([]):
        return False
    # The 32-bit packet header takes the two elements before the payload.
    if str(MemRefType(l1.type).element_type) not in ("bf16", "f16", "i16"):
        return False
    bufs = []
    for b in per:
        if not any(b == x for x in bufs):
            bufs.append(b)
    dests = [next(i for i, x in enumerate(bufs) if x == b) for b in per]
    # The core sends every phase's packet to one buffer in its column's
    # memtile, which forwards it, header included, to the tensor the header
    # names.
    hop, fan = name + "_hop", name
    with InsertionPoint.at_block_begin(module.body):
        hc = Channel(hop, size=[1, 1])
        hc.operation.attributes["channel_type"] = StringAttr.get("npu_dma_packet")
        ch = Channel(fan, size=[1, 1], broadcast_shape=[1, len(bufs)])
        ch.operation.attributes["channel_type"] = StringAttr.get("npu_dma_packet")
    idx = IndexType.get()
    et = MemRefType(l1.type).element_type
    i32t = IntegerType.get_signless(32)
    col = IntegerAttr(h.attributes["x_loc"]).value
    h, ph = _add_herd_operand(h, loop.induction_variable)
    w = next(
        o
        for o, _ in _walk(h)
        if o.name == "air.dma_memcpy_nd"
        and o.opview.src == l1
        and _space(o.opview.dst) == 1
    )
    with InsertionPoint(w):
        pk = memref.AllocOp(
            MemRefType.get([_HDR + n], et, memory_space=l1.type.memory_space), [], []
        ).result
        c = lambda v: arith.ConstantOp(idx, IntegerAttr.get(idx, v)).result
        # A linalg.copy rather than a view, so that AIR sees the core write
        # the packet buffer and makes it wait until the DMA has sent the
        # previous packet.
        src_v = memref.subview(l1, [off], [n], [1])
        dst_v = memref.subview(pk, [_HDR], [n], [1])
        linalg.copy(src_v, outs=[dst_v])
        dsel = _phase_value(dests, ph, idx) if len(set(dests)) > 1 else c(dests[0])
        ChannelPut(
            hop,
            pk,
            offsets=[_HDR - 2],
            sizes=[n + 2],
            strides=[1],
            indices=[0, 0],
            dest=dsel,
        )
        memref.DeallocOp(pk)
    w.erase()
    with InsertionPoint(_after(h)):
        hub = memref.AllocOp(
            MemRefType.get([n + 2], et, memory_space=IntegerAttr.get(i32t, 1)),
            [],
            [],
        ).result
        hub.owner.attributes["air.memtile_col"] = IntegerAttr.get(i32t, col)
        ChannelGet(hop, hub, offsets=[0], sizes=[n + 2], strides=[1], indices=[0, 0])
        ChannelPut(fan, hub, offsets=[0], sizes=[n + 2], strides=[1], indices=[0, 0])
        memref.DeallocOp(hub)
    do, ds, dt = dst_pattern
    with InsertionPoint(_after(loop)):
        for j, b in enumerate(bufs):
            ChannelGet(fan, b, offsets=do, sizes=ds, strides=dt, indices=[0, j])
    d.erase()
    return True


def _edge_key(v, edges):
    """The key of edge buffer `v` in `phased`, or None if `v` is not one."""
    for e in edges:
        if v == e:
            return id(e)
    return None


def _check_edge_loops(seg, edges):
    """Reject copies to or from on-chip tensors whose offsets move with a
    loop that contains no herd. A memtile DMA program is fixed, so such a
    copy would get one buffer descriptor, at the first trip's offsets. Loops
    around herds are handled by _lower_edges."""
    for d in _segment_dmas(seg):
        dst, src = d.opview.dst, d.opview.src
        if _space(dst) != 1 or _space(src) != 1:
            continue
        if not any(dst == e or src == e for e in edges):
            continue
        view = d.opview
        values = [
            v
            for side in ("src", "dst")
            for k in ("offsets", "sizes", "strides")
            for v in getattr(view, f"dynamic_{side}_{k}")
        ]
        p = d.parent
        while p.name != "air.segment":
            if p.name == "scf.for" and not _ops_named(p, "air.herd"):
                iv = p.regions[0].blocks[0].arguments[0]
                if any(_depends_on(v, iv) for v in values):
                    raise ValueError(
                        "a copy to or from an on-chip tensor moves with a loop "
                        f"outside the cores: {d}"
                    )
            p = p.parent


def _depends_on(v, iv, seen=None):
    """Whether `v` depends on `iv`."""
    from air.ir import OpResult

    if v == iv:
        return True
    if not isinstance(v, OpResult):
        return False
    seen = [] if seen is None else seen
    op = v.owner.operation
    if any(op == x for x in seen):
        return False
    seen.append(op)
    return any(_depends_on(o, iv, seen) for o in op.operands)


def _const(v):
    """The value of `v` if it is an arith.constant, else None."""
    from air.ir import IntegerAttr, OpResult

    if isinstance(v, OpResult) and v.owner.name == "arith.constant":
        return IntegerAttr(v.owner.attributes["value"]).value
    return None


def _placing_loops(d, side):
    """The scf.for loops around DMA `d`, outermost first, starting from the
    outermost one whose index `side` of `d` depends on. Empty if none."""
    view = d.opview
    values = [
        v
        for k in ("offsets", "sizes", "strides")
        for v in getattr(view, f"dynamic_{side}_{k}")
    ]
    loops, p = [], d.parent
    while p.name != "air.segment":
        if p.name == "scf.for":
            loops.append(p)
        p = p.parent
    loops.reverse()
    for i, loop in enumerate(loops):
        iv = loop.regions[0].blocks[0].arguments[0]
        if any(_depends_on(v, iv) for v in values):
            return loops[i:]
    return []


def _trips(loops, env=()):
    """Yield every trip of loop nest `loops` as [(iv, value), ...], outer
    index slowest. Bounds may depend on the indices of outer loops."""
    env = list(env)
    if not loops:
        yield env
        return
    loop = loops[0]
    try:
        lo, hi, st = (_eval(v, env) for v in list(loop.operands)[:3])
    except ValueError:
        raise ValueError("an on-chip tensor is placed by a loop of unknown trip count")
    iv = loop.regions[0].blocks[0].arguments[0]
    for x in range(lo, hi, st):
        yield from _trips(loops[1:], env + [(iv, x)])


def _after(op):
    """The operation after `op` in its block."""
    block_ops = op.operation.block.operations
    found = False
    for o in block_ops:
        if found:
            return o
        if o.operation == op.operation:
            found = True
    raise ValueError("no operation after the loop")


def _eval(v, env):
    """Evaluate index computation `v`, with loop index values from `env`.
    Raises ValueError for anything that is not constant under `env`."""
    from air.ir import AffineMapAttr, IntegerAttr, OpResult

    if isinstance(v, int):
        return v
    for iv, x in env:
        if v == iv:
            return x
    if not isinstance(v, OpResult):
        raise ValueError(f"can't evaluate {v}")
    op = v.owner
    name = op.name
    if name == "arith.constant":
        return IntegerAttr(op.attributes["value"]).value
    a = [_eval(o, env) for o in op.operands]
    if name in (
        "arith.index_cast",
        "arith.index_castui",
        "arith.extsi",
        "arith.extui",
        "arith.trunci",
    ):
        return a[0]
    ops = {
        "arith.addi": lambda: a[0] + a[1],
        "arith.subi": lambda: a[0] - a[1],
        "arith.muli": lambda: a[0] * a[1],
        "arith.divsi": lambda: int(a[0] / a[1]),
        "arith.divui": lambda: a[0] // a[1],
        "arith.floordivsi": lambda: a[0] // a[1],
        "arith.remsi": lambda: a[0] % a[1],
        "arith.remui": lambda: a[0] % a[1],
    }
    if name in ops:
        return ops[name]()
    if name == "arith.select":
        return a[1] if a[0] else a[2]
    if name == "arith.cmpi":
        pred = IntegerAttr(op.attributes["predicate"]).value
        x, y = a
        cmp = [x == y, x != y, x < y, x <= y, x > y, x >= y]
        return (cmp + cmp[2:])[pred]
    if name == "affine.apply":
        m = AffineMapAttr(op.attributes["map"]).value
        return _affine(m.results[0], a[: m.n_dims], a[m.n_dims :])
    raise ValueError(f"can't evaluate {name}")


def _affine(e, dims, syms):
    from air.ir import (
        AffineAddExpr,
        AffineCeilDivExpr,
        AffineConstantExpr,
        AffineDimExpr,
        AffineFloorDivExpr,
        AffineModExpr,
        AffineMulExpr,
        AffineSymbolExpr,
    )

    if isinstance(e, AffineConstantExpr):
        return e.value
    if isinstance(e, AffineDimExpr):
        return dims[e.position]
    if isinstance(e, AffineSymbolExpr):
        return syms[e.position]
    for cls, f in (
        (AffineAddExpr, lambda x, y: x + y),
        (AffineMulExpr, lambda x, y: x * y),
        (AffineModExpr, lambda x, y: x % y),
        (AffineFloorDivExpr, lambda x, y: x // y),
        (AffineCeilDivExpr, lambda x, y: -(-x // y)),
    ):
        if isinstance(e, cls):
            return f(_affine(e.lhs, dims, syms), _affine(e.rhs, dims, syms))
    raise ValueError(f"can't evaluate {e}")


def _static(v):
    """`v` as a hashable key: its value if constant, else its text."""
    try:
        return _eval(v, [])
    except ValueError:
        return str(v)


def _get_loop(name, dst, pattern, trips):
    """Emit one channel get into `dst` per trip in `trips`: an scf.for with affine offsets when the offsets move
    linearly and sizes and strides are constant, otherwise one op per trip.
    Returns the last op emitted."""
    from air.dialects import arith
    from air.dialects.affine import apply as affine_apply
    from air.dialects.air import ChannelGet
    from air.dialects.scf import for_, yield_
    from air.ir import AffineConstantExpr, AffineDimExpr, AffineMap

    vals = [[[_eval(v, env) for v in part] for part in pattern] for env in trips]
    first = vals[0]
    offs = []
    for k in range(len(first[0])):
        a = first[0][k]
        b = vals[1][0][k] - a if len(vals) > 1 else 0
        offs.append((a, b))
    linear = all(
        v[0][k] == a + b * t
        for t, v in enumerate(vals)
        for k, (a, b) in enumerate(offs)
    ) and all(v[1] == first[1] and v[2] == first[2] for v in vals)
    if not linear or len(vals) < 2:
        last = None
        for v in vals:
            last = ChannelGet(
                name, dst, offsets=v[0], sizes=v[1], strides=v[2], indices=[0, 0]
            )
        return last
    c = arith.ConstantOp.create_index
    for t in for_(c(0), c(len(vals)), c(1)):
        o = [
            (
                a
                if not b
                else affine_apply(
                    AffineMap.get(
                        1,
                        0,
                        [
                            AffineDimExpr.get(0) * AffineConstantExpr.get(b)
                            + AffineConstantExpr.get(a)
                        ],
                    ),
                    [t],
                )
            )
            for a, b in offs
        ]
        ChannelGet(
            name,
            dst,
            offsets=o,
            sizes=first[1],
            strides=first[2],
            indices=[0, 0],
        )
        yield_([])
    return t.owner.owner


def _wave_value(per_phase, wave, ph, const32):
    """The i32 value of a call's wave argument at wave `wave`. With a phase
    loop (index `ph`), `per_phase` holds each phase's argument."""
    from air.dialects import arith
    from air.ir import IndexType, IntegerAttr

    def pick(vals):
        if ph is None or len(set(vals)) == 1:
            return const32(vals[-1])
        step = vals[1] - vals[0]
        if all(v == vals[0] + step * k for k, v in enumerate(vals)):
            # Linear in the phase, so that DMA patterns stay affine.
            k = arith.index_cast(const32(0).type, ph)
            return arith.addi(const32(vals[0]), arith.muli(k, const32(step)))
        out = const32(vals[-1])
        idx = IndexType.get()
        for k in reversed(range(len(vals) - 1)):
            if vals[k] == vals[-1]:
                continue
            is_k = arith.cmpi(
                arith.CmpIPredicate.eq,
                ph,
                arith.ConstantOp(idx, IntegerAttr.get(idx, k)).result,
            )
            out = arith.select(is_k, const32(vals[k]), out)
        return out

    if any(isinstance(w, _Table) for w in per_phase):
        n = max(len(w.table) for w in per_phase if isinstance(w, _Table))

        def chain(w):
            vals = (
                w.table if isinstance(w, _Table) else [w.a * i + w.b for i in range(n)]
            )
            out = const32(vals[-1])
            for i in reversed(range(len(vals) - 1)):
                if vals[i] != vals[-1]:
                    is_i = arith.cmpi(arith.CmpIPredicate.eq, wave, const32(i))
                    out = arith.select(is_i, const32(vals[i]), out)
            return out

        per = [chain(w) for w in per_phase]
        if ph is None or len(per) == 1:
            return per[-1]
        out = per[-1]
        idx = IndexType.get()
        for k in reversed(range(len(per) - 1)):
            is_k = arith.cmpi(
                arith.CmpIPredicate.eq,
                ph,
                arith.ConstantOp(idx, IntegerAttr.get(idx, k)).result,
            )
            out = arith.select(is_k, per[k], out)
        return out
    a = [w.a for w in per_phase]
    b = [w.b for w in per_phase]
    if a == [1] * len(a) and b == [0] * len(b):
        return wave
    return arith.addi(arith.muli(pick(a), wave), pick(b))


def _trip_value(env, iv):
    """The value of loop index `iv` in trip `env`."""
    return next(v for k, v in env if k == iv)


def _hoist_ddr_reads(block, ddr_args, end):
    """Move the reads from DDR at the top level of `block` to its start,
    after the edge buffers' allocs, keeping their order.

    The runtime sequence issues a wave's DDR transfers in segment order and
    waits on some before issuing the next. A call that runs between two
    phases of another herd would otherwise have its inputs issued after the
    later phase's transfers, which wait for that call: a deadlock. A read of
    a DDR buffer that an earlier copy in the wave writes stays in place."""
    ops = list(block.operations)
    written = []  # DDR args written so far, in program order
    moves = []
    for op in ops:
        if op.operation == end.operation:
            break
        if op.name != "air.dma_memcpy_nd":
            # DDR buffers written inside a nested op count from here on.
            for d in _segment_dmas(op):
                if any(d.opview.dst == a for a in ddr_args):
                    written.append(d.opview.dst)
            continue
        src, dst = op.opview.src, op.opview.dst
        if any(dst == a for a in ddr_args):
            written.append(dst)
            continue
        if any(src == a for a in ddr_args) and not any(src == w for w in written):
            moves.append(op)
    # Insert after the leading allocs, in order.
    prev = None
    for o in ops:
        if o.name != "memref.alloc":
            break
        prev = o
    placed = []  # ops already moved along with an earlier read
    for op in moves:
        for o in _defining_ops(op, block) + [op]:
            if any(o.operation == x for x in placed):
                continue
            if prev is None:
                o.operation.move_before(ops[0])
            elif o.operation != prev.operation:
                o.operation.move_after(prev)
            placed.append(o.operation)
            prev = o


def _defining_ops(op, block):
    """The ops of `block` that compute `op`'s operands, transitively, in
    program order."""
    from air.ir import OpResult

    found = []  # operations

    def visit(v):
        if not isinstance(v, OpResult):
            return
        d = v.owner.operation
        if d.block != block or any(d == x for x in found):
            return
        found.append(d)
        for o in d.operands:
            visit(o)

    for v in op.operands:
        visit(v)
    return [o for o in block.operations if any(o.operation == x for x in found)]


def _walk(op):
    """All ops nested under `op` in order, each with whether it is inside a
    herd."""
    out = []

    def go(o, herd):
        for r in o.regions:
            for b in r:
                for x in b.operations:
                    out.append((x, herd))
                    go(x, herd or x.name == "air.herd")

    go(op, False)
    return out


def _phase_numbers(seg, others):
    """How the segments of a herd's phases differ from the first phase's
    `seg`: the integer constants and static DMA patterns of the `others`.
    Raises ValueError if anything else differs, or anything inside a herd
    body, since the phases share the cores' code."""
    from air.ir import DenseI64ArrayAttr, IntegerAttr

    mine = _walk(seg)
    theirs = [_walk(_ops_named(_entry(m), "air.segment")[0]) for m in others]
    wiring = _wiring(mine)
    if any(len(t) != len(mine) or _wiring(t) != wiring for t in theirs):
        raise ValueError("the phases of a herd differ in their structure")
    diffs = []
    for k, (op, in_herd) in enumerate(mine):
        peers = [t[k][0] for t in theirs]
        if any(p.name != op.name for p in peers):
            raise ValueError("the phases of a herd differ in their structure")
        if op.name == "arith.constant":
            vals = [str(x.attributes["value"]) for x in [op] + peers]
            if len(set(vals)) > 1:
                if in_herd:
                    raise ValueError("the phases of a herd differ in its code")
                diffs.append(
                    (
                        "const",
                        op,
                        [
                            IntegerAttr(x.attributes["value"]).value
                            for x in [op] + peers
                        ],
                    )
                )
            continue
        if op.name == "air.dma_memcpy_nd":
            for side in ("dst", "src"):
                for kind in ("offsets", "sizes", "strides"):
                    key = f"static_{side}_{kind}"
                    arrs = [
                        list(DenseI64ArrayAttr(x.attributes[key])) for x in [op] + peers
                    ]
                    if any(a != arrs[0] for a in arrs):
                        if in_herd:
                            raise ValueError("the phases of a herd differ in its code")
                        diffs.append(("dma", op, (side, kind), arrs))
            continue
        attrs = [
            sorted((k, str(x.attributes[k])) for k in _attr_names(x) if k != "id")
            for x in [op] + peers
        ]
        if any(a != attrs[0] for a in attrs) or any(
            [str(r.type) for r in x.results] != [str(r.type) for r in op.results]
            for x in peers
        ):
            raise ValueError(f"the phases of a herd differ in {op.name}")
    return diffs


def _wiring(ops):
    """For each op of `ops` (as _walk lists them), where each operand comes
    from: the position of its defining op and the result number, or the
    position of the op owning its block and the argument number (-1 for ops
    outside `ops`)."""
    from air.ir import BlockArgument, OpResult

    pos = {op.operation: k for k, (op, _) in enumerate(ops)}
    out = []
    for op, _ in ops:
        keys = []
        for v in op.operands:
            if isinstance(v, OpResult):
                keys.append(("r", pos.get(v.owner.operation, -1), v.result_number))
            elif isinstance(v, BlockArgument):
                keys.append(("a", pos.get(v.owner.owner.operation, -1), v.arg_number))
            else:
                keys.append(("?", -1, -1))
        out.append(keys)
    return out


def _phase_value(vals, ph, type_):
    """A value of type `type_` that is vals[ph] at phase `ph`: affine when
    `vals` is linear, otherwise a chain of selects."""
    from air.dialects import arith
    from air.ir import IndexType, IntegerAttr

    idx = IndexType.get()
    c = lambda v: arith.ConstantOp(type_, IntegerAttr.get(type_, v)).result
    step = vals[1] - vals[0]
    if all(v == vals[0] + step * k for k, v in enumerate(vals)):
        k = ph if type_ == idx else arith.index_cast(type_, ph)
        return arith.addi(c(vals[0]), arith.muli(k, c(step)))
    out = c(vals[-1])
    for k in reversed(range(len(vals) - 1)):
        is_k = arith.cmpi(
            arith.CmpIPredicate.eq,
            ph,
            arith.ConstantOp(idx, IntegerAttr.get(idx, k)).result,
        )
        out = arith.select(is_k, c(vals[k]), out)
    return out


def _apply_phase_numbers(diffs, ph):
    """Rewrite the first phase's segment to compute each phase's numbers
    (`diffs`, from _phase_numbers) from phase index `ph`."""
    from air.ir import (
        DenseI32ArrayAttr,
        DenseI64ArrayAttr,
        IndexType,
        InsertionPoint,
        Operation,
        ShapedType,
    )

    for d in diffs:
        if d[0] == "const":
            _, op, vals = d
            v = _phase_value(vals, ph, op.results[0].type)
            op.results[0].replace_all_uses_with(v)
            op.operation.erase()
    by_dma = {}
    for d in diffs:
        if d[0] == "dma":
            by_dma.setdefault(id(d[1]), (d[1], []))[1].append(d[2:])
    dyn = ShapedType.get_dynamic_size()
    idx = IndexType.get()
    for op, changes in by_dma.values():
        # Operand groups: deps, dst, dst offsets/sizes/strides, src, src
        # offsets/sizes/strides.
        seg_sizes = list(DenseI32ArrayAttr(op.attributes["operandSegmentSizes"]))
        operands = list(op.operands)
        groups, i = [], 0
        for n in seg_sizes:
            groups.append(operands[i : i + n])
            i += n
        names = [
            "deps",
            "dst",
            "dst_offsets",
            "dst_sizes",
            "dst_strides",
            "src",
            "src_offsets",
            "src_sizes",
            "src_strides",
        ]
        attrs = {k: op.attributes[k] for k in _attr_names(op)}
        for (side, kind), arrs in changes:
            key = f"static_{side}_{kind}"
            static = list(DenseI64ArrayAttr(attrs[key]))
            g = names.index(f"{side}_{kind}")
            dynamic = list(groups[g])
            new_static, new_dyn, j = [], [], 0
            for e, v in enumerate(static):
                if v == dyn:
                    new_static.append(dyn)
                    new_dyn.append(dynamic[j])
                    j += 1
                elif any(a[e] != v for a in arrs):
                    new_static.append(dyn)
                    new_dyn.append(_phase_value([a[e] for a in arrs], ph, idx))
                else:
                    new_static.append(v)
            attrs[key] = DenseI64ArrayAttr.get(new_static)
            groups[g] = new_dyn
        attrs["operandSegmentSizes"] = DenseI32ArrayAttr.get([len(x) for x in groups])
        with InsertionPoint(op):
            new = Operation.create(
                "air.dma_memcpy_nd",
                results=[r.type for r in op.results],
                operands=[v for x in groups for v in x],
                attributes=attrs,
            )
        for a, b in zip(op.results, new.results):
            a.replace_all_uses_with(b)
        op.operation.erase()


def _attr_names(op):
    """The names of an op's attributes."""
    out = []
    for a in op.attributes:
        out.append(a if isinstance(a, str) else a.name)
    return out
