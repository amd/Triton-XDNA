# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""The AIR module of a superkernel (see `tl.extra.air.spatial`): several
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

    def __init__(self, name, n, dtype, onchip, space=1, col=None):
        self.name, self.n, self.dtype, self.onchip = name, n, dtype, onchip
        self.space = space  # an on-chip tensor's memory space: 1 L2, 2 L1
        self.col = col  # an L2 tensor's memtile column, if pinned

    def __repr__(self):
        return f"{'onchip' if self.onchip else 'ddr'}({self.name})"


class _TileId:
    """A program id of a kernel that runs one program a core: the core's
    index in its herd along `axis`."""

    def __init__(self, axis):
        self.axis = axis

    def __repr__(self):
        return f"program_id({self.axis})"


_TILE = {a: _TileId(a) for a in range(3)}


class _Local:
    """An L1 buffer of each core of the herds whose kernel allocates it, of
    `n` elements of `dtype`. Stands as a placeholder buffer of the segment
    until the herds get buffers of their own."""

    def __init__(self, n, dtype, symbol):
        self.n, self.dtype, self.space = n, dtype, 2
        self.symbol = symbol  # the function that stands for it in the TTIR

    def __repr__(self):
        return f"local({self.n}x{self.dtype})"


class _Channel:
    """An AIR channel: a bundle of FIFOs of shape `size` between cores.
    `n` and `dtype` give the placeholder buffer that stands for it until the
    copies through it become puts and gets."""

    def __init__(
        self,
        name,
        size,
        broadcast_shape,
        channel_type,
        buffer_resources,
        tile_dma_channel=None,
    ):
        self.name, self.size = name, tuple(size)
        self.broadcast_shape, self.channel_type = broadcast_shape, channel_type
        self.buffer_resources = buffer_resources
        self.tile_dma_channel = tile_dma_channel
        self.n, self.dtype = 1, "i8"

    def __repr__(self):
        return f"channel({self.name})"


class _Segment:
    """Code of the segment between herds: the text of a func whose arguments
    are the values it uses from outside (`binds`, each an int, a _Wave, a
    _Scalar, an on-chip Tensor or the name of a buffer an earlier segment
    code exports) and whose results are the buffers it exports by name."""

    def __init__(self, text, binds, exports, where="segment"):
        self.text, self.binds, self.exports = text, binds, exports
        self.where = where  # "segment", or "launch" for the DDR transfers


class _Call:
    """One call: its kernel's name, its compiled AIR module and how the
    module's func arguments bind to the superkernel's tensors and ints."""

    def __init__(self, name, at, herd, air, binds, read, written, cores, spmd):
        self.name, self.at, self.herd, self.air = name, at, herd, air
        self.cores, self.spmd = cores, spmd
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
        self.channels = []
        self.items = []  # ("call", _Call) and ("segment", _Segment), in order
        self.backend = None  # air.backend's options, if given

    def scalar(self, name):
        """A new int argument of the func, set by the host at each dispatch."""
        v = _Scalar(name)
        self.scalars.append(v)
        return v

    def channel(
        self,
        name,
        size,
        broadcast_shape=None,
        channel_type=None,
        buffer_resources=None,
        tile_dma_channel=None,
    ):
        """A new AIR channel named `name`."""
        if any(c.name == name for c in self.channels):
            raise ValueError(f"a channel named {name} exists")
        c = _Channel(
            name,
            size,
            broadcast_shape,
            channel_type,
            buffer_resources,
            tile_dma_channel,
        )
        self.channels.append(c)
        return c

    def ddr(self, name, n, dtype):
        """A new buffer argument of the func, in DDR."""
        return self._tensor(name, n, dtype, False)

    def onchip(self, name, n, dtype, space=1, col=None):
        """A new tensor kept on chip in one buffer of the segment, in L2 (in
        the memtile of column `col`, if given) or, for neighbouring cores, L1
        (`space` 2)."""
        return self._tensor(name, n, dtype, True, space, col)

    def _tensor(self, name, n, dtype, onchip, space=1, col=None):
        if any(t.name == name for t in self.tensors):
            raise ValueError(f"a tensor named {name} exists")
        t = Tensor(name, n, dtype, onchip, space, col)
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
        ttshared = _allocate_locals(ttshared, args)
        with config_context(transform_tiling_script=schedule):
            m = _ttshared_to_air(ttshared, 1, 1, 1)
        # A kernel that reads its program ids runs one program a core: its
        # schedule compiles one program. Any other is split over its cores
        # by its schedule.
        spmd = any(isinstance(a, _TileId) for a in args)
        want = (1, 1) if spmd else tuple(cores)
        for h in _ops_named(_entry(m), "air.herd"):
            shape = (_herd_width(h), _herd_height(h))
            if shape != want:
                raise ValueError(
                    f"{name}: its schedule makes a {shape[0]}x{shape[1]} herd "
                    f"for {want[0]}x{want[1]} cores"
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
        call = _Call(
            name, tuple(at), herd, str(m), binds, read, written, tuple(cores), spmd
        )
        self.calls.append(call)
        self.items.append(("call", call))

    def segment(self, seg):
        """Add segment code, a `_Segment`, to the wave after the calls so far."""
        self.items.append(("segment", seg))

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
        from air.dialects import arith, memref
        from air.ir import (
            BoolAttr,
            Context,
            InsertionPoint,
            IntegerAttr,
            IntegerType,
            Location,
            Module,
        )

        bufs = self.buffers
        onchip = [t for t in self.tensors if t.onchip]
        with Context(), Location.unknown():
            i32 = IntegerType.get_signless(32)
            owns = []  # the kernels' own L1 buffers
            for c in self.calls:
                owns += [b for b in c.binds if isinstance(b, _Local) and b not in owns]
            holders = onchip + self.channels + owns
            top = Module.parse(self._skeleton(bufs, holders))
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
            for t, a in zip(holders, allocs):
                value[id(t)] = a.result
            for a in allocs:
                # Not shrunk to its current uses: the copies that read and
                # write it are only made later.
                a.attributes["air.shrinkage"] = BoolAttr.get(False)
            for t, a in zip(holders, allocs):
                if getattr(t, "col", None) is not None:
                    a.attributes["air.memtile_col"] = IntegerAttr.get(i32, t.col)
            wave = sargs[len(bufs)]
            for i, v in enumerate(self.scalars):
                value[id(v)] = sargs[len(bufs) + 1 + i]
            phased = {}  # on-chip tensor id -> (phase loop, "in" or "out", tensors)
            loops = {}  # phase loop key -> (loop, its yield)
            tiles = {}
            with InsertionPoint.at_block_begin(sblock):
                for a, t in _TILE.items():
                    # Stand-ins for the program ids until the herds are
                    # sized.
                    c = arith.ConstantOp(i32, IntegerAttr.get(i32, -1 - a))
                    tiles[a] = c.result
                    value[id(t)] = c.result
            groups = self._phases()
            first = {id(g[0]): g for g in groups}
            made = {}  # a segment's exported buffers, by name
            # The launch's arguments: the buffers, the wave, the scalars.
            launch = seg.operation.parent
            nargs = len(bufs) + 1 + len(self.scalars)
            largs = list(launch.regions[0].blocks[0].arguments)[-nargs:]
            lvalue = {id(t): largs[i] for i, t in enumerate(bufs)}
            for i, v in enumerate(self.scalars):
                lvalue[id(v)] = largs[len(bufs) + 1 + i]
            lwave = largs[len(bufs)]
            for kind, item in self.items:
                if kind == "segment" and item.where == "launch":
                    _emit_segment(top, item, seg, lvalue, lwave, {}, self.channels)
                elif kind == "segment":
                    _emit_segment(top, item, end, value, wave, made, self.channels)
                elif id(item) in first:
                    self._add_phase_group(
                        top, first[id(item)], end, value, wave, phased, loops
                    )
            with InsertionPoint(end):
                for v in made.values():
                    memref.DeallocOp(v)
            spmd = {c.herd: c.cores for c in self.calls if c.spmd}
            _bind_tile_ids(seg, tiles, spmd)
            _lower_channels(top, seg, [(c, value[id(c)]) for c in self.channels])
            homes = [
                v for _, v in _lower_locals(seg, [(b, value[id(b)]) for b in owns])
            ]
            # and those the kernels allocate themselves, before their schedule
            homes += [
                o.results[0]
                for o in _ops_named(seg, "memref.alloc")
                if "air.local" in o.attributes
            ]
            shared = [value[id(t)] for t in onchip if t.space == 2]
            _in_place(seg, homes, shared)
            _forward_private_vectors(seg)
            _scalar_transfers(seg)
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
            _fold_constant_casts(top.operation, seg)
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
        l2 = [f"memref<{t.n}x{t.dtype}, {getattr(t, 'space', 1)} : i32>" for t in edges]
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
        feed_args=("--unroll-feed-loops", "--merge-contiguous-feeds"),
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
                    *feed_args,
                    *extra_args,
                ],
                runtime_loop_tiling_sizes=runtime_loop_tiling_sizes,
            )
        finally:
            os.chdir(cwd)
        header = os.path.join(self.proj, "air_project", f"npu.{self.name}.air.txn.h")
        return out["bin_path"], header


def _emit_segment(top, sg, end, value, wave, made, channels):
    """Put segment code `sg` before `end`, its arguments bound."""
    from air.dialects import arith
    from air.ir import InsertionPoint, IntegerAttr, IntegerType, Module

    decls = "".join(
        f"air.channel @{c.name} [{', '.join(map(str, c.size))}]\n" for c in channels
    )
    m = Module.parse(decls + sg.text)
    f = next(o for o in m.body.operations if o.operation.name == "func.func")
    block = f.regions[0].blocks[0]
    i32 = IntegerType.get_signless(32)
    const32 = lambda v: arith.ConstantOp(
        i32, IntegerAttr.get(i32, v)
    ).result  # noqa: E731
    with InsertionPoint(end):
        for a, b in zip(block.arguments, sg.binds):
            if isinstance(b, str):
                v = made[b]
            elif isinstance(b, bool):
                v = arith.ConstantOp(a.type, IntegerAttr.get(a.type, int(b))).result
            elif isinstance(b, int):
                v = arith.ConstantOp(a.type, IntegerAttr.get(a.type, b)).result
            elif isinstance(b, _Wave):
                v = _wave_value([b], wave, None, const32)
                if str(a.type) == "i1":
                    v = arith.cmpi(arith.CmpIPredicate.ne, v, const32(0))
                elif str(a.type) != "i32":
                    v = arith.extsi(a.type, v)
            else:
                v = value[id(b)]
            a.replace_all_uses_with(v)
    ops = list(block.operations)
    ret = ops[-1]
    for name, v in zip(sg.exports, ret.operands):
        made[name] = v
    for o in ops[:-1]:
        o.operation.move_before(end)


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


_RADIX = 256  # as in tl.extra.air: a channel index's digits in its address

_CASTS = (
    "arith.index_cast",
    "arith.index_castui",
    "arith.extsi",
    "arith.extui",
    "arith.trunci",
)


def _linear(v):
    """`v` as a constant plus a sum of values times constants: (constant,
    [(value, factor)]), looking through additions, products with constants
    and casts."""
    from air.ir import IntegerAttr, OpResult

    if isinstance(v, int):
        return v, []
    if isinstance(v, OpResult):
        op, name = v.owner, v.owner.name
        if name == "arith.constant":
            return IntegerAttr(op.attributes["value"]).value, []
        if name in _CASTS:
            return _linear(op.operands[0])
        if name in ("arith.addi", "arith.subi"):
            c1, t1 = _linear(op.operands[0])
            c2, t2 = _linear(op.operands[1])
            sign = 1 if name == "arith.addi" else -1
            return c1 + sign * c2, t1 + [(x, sign * k) for x, k in t2]
        if name == "arith.muli":
            c1, t1 = _linear(op.operands[0])
            c2, t2 = _linear(op.operands[1])
            if not t1:
                return c1 * c2, [(x, c1 * k) for x, k in t2]
            if not t2:
                return c1 * c2, [(x, c2 * k) for x, k in t1]
    return 0, [(v, 1)]


def _channel_indices(c, off, n):
    """The indices into channel `c` that a copy of `n` elements at offset
    `off` of its placeholder stands for: the offset's digits of radix
    _RADIX, each digit's place a multiple of `n`; and a put's dest, the
    digit above them less one, or None. Emitted before the copy."""
    from air.dialects import arith
    from air.ir import IndexType, IntegerAttr

    idx = IndexType.get()
    rank = len(c.size) + 1
    places = [n * _RADIX ** (rank - 1 - k) for k in range(rank)]
    const, terms = _linear(off)
    per = [[] for _ in range(rank)]
    for x, k in terms:
        dim = next(
            (d for d, p in enumerate(places) if k % p == 0 and 0 < k // p < _RADIX),
            None,
        )
        if dim is None:
            raise ValueError(f"{c}: an index the address does not encode: {off}")
        per[dim].append((x, k // places[dim]))
    if const % n:
        raise ValueError(f"{c}: an offset of {const}, not a whole block of {n}")
    q, digits = const // n, []
    for _ in range(rank):
        digits.insert(0, q % _RADIX)
        q //= _RADIX
    if q:
        raise ValueError(f"{c}: more indices than its rank {rank - 1}")
    out = []
    for terms_, digit in zip(per, digits):
        v = None
        for x, m in terms_:
            if x.type != idx:
                x = arith.IndexCastOp(idx, x).result
            if m != 1:
                x = arith.MulIOp(
                    x, arith.ConstantOp(idx, IntegerAttr.get(idx, m)).result
                ).result
            v = x if v is None else arith.AddIOp(v, x).result
        if v is None or digit:
            d = arith.ConstantOp(idx, IntegerAttr.get(idx, digit)).result
            v = d if v is None else arith.AddIOp(v, d).result
        out.append(v)
    dest = None
    if per[0] or digits[0]:
        one = arith.ConstantOp(idx, IntegerAttr.get(idx, 1)).result
        dest = arith.SubIOp(out[0], one).result
    return out[1:], dest


def _lower_channels(module, seg, chans):
    """Declare each channel and turn the herds' copies to and from its
    placeholder buffer into its puts and gets; drop the placeholders."""
    from air.dialects.air import ChannelGet, ChannelPut, channel as channel_decl
    from air.ir import ArrayAttr, InsertionPoint, IntegerAttr, IntegerType

    i64 = IntegerType.get_signless(64)
    for c, holder in chans:
        with InsertionPoint.at_block_begin(module.body):
            decl = channel_decl(c.name, size=list(c.size), channel_type=c.channel_type)
        if c.broadcast_shape is not None:
            decl.attributes["broadcast_shape"] = ArrayAttr.get(
                [IntegerAttr.get(i64, d) for d in c.broadcast_shape]
            )
        if c.buffer_resources is not None:
            decl.attributes["buffer_resources"] = IntegerAttr.get(
                i64, c.buffer_resources
            )
        if c.tile_dma_channel is not None:
            decl.attributes["air.tile_dma_channel"] = IntegerAttr.get(
                i64, c.tile_dma_channel
            )
        for h in _ops_named(seg, "air.herd"):
            body = h.regions[0].blocks[0]
            first = len(body.arguments) - len(h.operands)
            for i, v in enumerate(h.operands):
                if v != holder:
                    continue
                for u in list(body.arguments[first + i].uses):
                    d = u.owner
                    if d.name != "air.dma_memcpy_nd":
                        raise ValueError(f"{c}: used by {d.name}")
                    put = d.opview.dst == body.arguments[first + i]
                    side, other = ("dst", "src") if put else ("src", "dst")
                    offs, sizes, strides = _dma_pattern(d, side)
                    if len(offs) != 1 or strides[0] != 1:
                        raise ValueError(f"{c}: a block that is not contiguous: {d}")
                    o, sz, st = _dma_pattern(d, other)
                    mem = getattr(d.opview, other)
                    with InsertionPoint(d):
                        ends, dest = _channel_indices(c, offs[0], sizes[0])
                        if put:
                            ChannelPut(
                                c.name,
                                mem,
                                offsets=o,
                                sizes=sz,
                                strides=st,
                                indices=ends,
                                dest=dest,
                            )
                        else:
                            if dest is not None:
                                raise ValueError(f"{c}: a get with a dest: {d}")
                            ChannelGet(
                                c.name,
                                mem,
                                offsets=o,
                                sizes=sz,
                                strides=st,
                                indices=ends,
                            )
                    d.erase()
        for h in _ops_named(seg, "air.herd"):
            ops = list(h.operands)
            if any(v == holder for v in ops):
                _drop_herd_operands(h, [i for i, v in enumerate(ops) if v == holder])
        for u in list(holder.uses):
            if u.owner.name != "memref.dealloc":
                raise ValueError(f"{c}: its placeholder is used by {u.owner.name}")
            u.owner.erase()
        holder.owner.erase()


def _lower_locals(seg, owns):
    """Give each herd that takes a kernel's own L1 buffer (placeholders
    `owns`) a buffer of its own, in its body. Returns those buffers, each
    with the symbol of the kernel's allocation."""
    from air.dialects import memref
    from air.ir import InsertionPoint, IntegerAttr, IntegerType, MemRefType, Type

    homes = []
    i32 = IntegerType.get_signless(32)
    for b, holder in owns:
        t = MemRefType.get(
            [b.n], Type.parse(b.dtype), memory_space=IntegerAttr.get(i32, 2)
        )
        for h in _ops_named(seg, "air.herd"):
            ops = list(h.operands)
            drop = [i for i, v in enumerate(ops) if v == holder]
            if not drop:
                continue
            body = h.regions[0].blocks[0]
            first = len(body.arguments) - len(ops)
            for i in drop:
                with InsertionPoint.at_block_begin(body):
                    mine = memref.AllocOp(t, [], [])
                body.arguments[first + i].replace_all_uses_with(mine.result)
                _respace_views(mine.result)
                homes.append((b.symbol, mine.result))
            _drop_herd_operands(h, drop)
        for u in list(holder.uses):
            u.owner.erase()
        holder.owner.erase()
    return homes


_VIEWS = ("memref.reinterpret_cast", "memref.subview", "memref.cast", "memref.view")


def _allocate_locals(ttshared, args):
    """Have kernel function `ttshared` allocate its own L1 buffers (the
    arguments `args` gives as _Local) in its body, so that its schedule
    places them with its other buffers. Each allocation is tagged with its
    buffer's symbol. A kernel may read an int8 buffer as another type
    (a pointer cast, which triton-shared leaves as an unrealized cast): that
    use becomes a `memref.view` of the bytes."""
    from air.dialects import arith, memref
    from air.ir import Context, IndexType, InsertionPoint, IntegerAttr, Location
    from air.ir import MemRefType, Module, StringAttr, Type, UnrankedMemRefType

    own = [(i, a) for i, a in enumerate(args) if isinstance(a, _Local)]
    if not own:
        return ttshared
    with Context() as ctx, Location.unknown():
        ctx.allow_unregistered_dialects = True
        m = Module.parse(ttshared)
        f = next(
            o
            for o in m.body.operations
            if o.operation.name == "func.func" and len(o.regions[0].blocks)
        )
        block = f.regions[0].blocks[0]
        for i, b in own:
            with InsertionPoint.at_block_begin(block):
                a = memref.AllocOp(MemRefType.get([b.n], Type.parse(b.dtype)), [], [])
            a.attributes["air.local"] = StringAttr.get(b.symbol)
            arg = block.arguments[i]
            for u in list(arg.uses):
                op = u.owner
                if op.name == "memref.reinterpret_cast":
                    continue
                if op.name != "builtin.unrealized_conversion_cast":
                    raise ValueError(f"an L1 buffer of the kernel used by {op.name}")
                if b.dtype != "i8":
                    raise ValueError(
                        "a kernel reads an L1 buffer as another type only if the "
                        "buffer is int8"
                    )
                to = UnrankedMemRefType(op.results[0].type)
                width = to.element_type.width // 8
                with InsertionPoint(op):
                    zero = arith.ConstantOp(
                        IndexType.get(), IntegerAttr.get(IndexType.get(), 0)
                    )
                    view = memref.ViewOp(
                        MemRefType.get([b.n // width], to.element_type),
                        a.result,
                        zero,
                        [],
                    )
                    cast = memref.CastOp(op.results[0].type, view.result)
                op.results[0].replace_all_uses_with(cast.result)
                op.erase()
            arg.replace_all_uses_with(a.result)
        return str(m)


def _respace_views(v):
    """Give the views of buffer `v` (`memref.reinterpret_cast` and the
    like, which triton-shared makes of a kernel's arguments) `v`'s memory
    space."""
    from air.ir import MemRefType

    space = MemRefType(v.type).memory_space
    for u in list(v.uses):
        op = u.owner
        if op.name not in _VIEWS:
            continue
        r = op.results[0]
        t = MemRefType(r.type)
        r.set_type(
            MemRefType.get(
                list(t.shape), t.element_type, layout=t.layout, memory_space=space
            )
        )
        _respace_views(r)


def _in_place(seg, homes, shared):
    """Have the herds use their L1 buffers in place: their own (`homes`) and
    the segment's shared L1 buffers (`shared`). A kernel compiled alone
    copies between such a buffer and a buffer of its own, which a DMA of
    the core cannot do. Where that buffer only takes a channel's get or
    feeds its put, the get or put moves to the part of the L1 buffer and the
    copy goes; otherwise the core copies."""
    from air.dialects import linalg
    from air.ir import InsertionPoint, MemRefType, OpResult

    for h in _ops_named(seg, "air.herd"):
        body = h.regions[0].blocks[0]
        first = len(body.arguments) - len(h.operands)
        mine = [x for x in homes if _within(x.owner, h)]
        mine += [
            body.arguments[first + i]
            for i, v in enumerate(h.operands)
            if any(v == x for x in shared)
        ]
        for c in _l1_copies(h, mine):
            _copy_in_place(*c)
        for d in _ops_named(h, "air.dma_memcpy_nd"):
            dst, src = d.opview.dst, d.opview.src
            side = next(
                (
                    k
                    for k, v in (("dst", dst), ("src", src))
                    if any(v == x for x in mine)
                ),
                None,
            )
            if side is None:
                continue
            home = dst if side == "dst" else src
            tmp = src if side == "dst" else dst
            if not (isinstance(tmp, OpResult) and tmp.owner.name == "memref.alloc"):
                raise ValueError(f"a copy between a core's L1 buffer and {tmp}: {d}")
            offs, sizes, strides = _dma_pattern(d, side)
            tt = MemRefType(tmp.type)
            if not sizes:
                sizes, strides, offs = list(tt.shape), [1], [0]
            others = [
                u.owner
                for u in tmp.uses
                if u.owner != d.operation and u.owner.name != "memref.dealloc"
            ]
            fills = [o for o in others if o.name == "linalg.fill"]
            transfers = [o for o in others if o.name.startswith("air.channel.")]
            if others and len(fills) + len(transfers) == len(others) and transfers:
                # The channel moves the data to or from the L1 buffer itself.
                for o in offs:
                    if not isinstance(o, int):
                        _move_before(o, tmp.owner)
                with InsertionPoint(tmp.owner):
                    view = _view(home, offs, sizes, strides, tt)
                for f in fills:
                    f.erase()
                alloc = tmp.owner
                d.erase()
                for u in list(tmp.uses):
                    if u.owner.name == "memref.dealloc":
                        u.owner.erase()
                tmp.replace_all_uses_with(view)
                alloc.erase()
                continue
            if list(tt.shape) != list(sizes):
                raise ValueError(
                    f"a copy of part of an L1 buffer into another shape: {d}"
                )
            with InsertionPoint(d):
                view = _view(home, offs, sizes, strides, tt)
                if side == "dst":
                    linalg.copy(tmp, outs=[view])
                else:
                    linalg.copy(view, outs=[tmp])
            d.erase()


def _home_of(v, mine):
    """The buffer of `mine` that `v` is a view of (through
    `memref.reinterpret_cast` and `memref.subview`), and the view's static
    element offset into it (None where it is not static); else None."""
    from air.ir import OpResult

    off = 0
    while isinstance(v, OpResult) and v.owner.name in (
        "memref.reinterpret_cast",
        "memref.subview",
    ):
        op = v.owner.opview
        so = list(op.static_offsets)
        if v.owner.name == "memref.subview":
            st = _strides_of(op.source.type)
            if off is not None and st is not None and all(o >= 0 for o in so):
                off += sum(o * k for o, k in zip(so, st))
            else:
                off = None
        elif off is not None and so[0] >= 0:
            off += so[0]
        else:
            off = None
        v = op.source
    if any(v == x for x in mine):
        return v, off
    return None


def _strides_of(t):
    """A memref type's static strides, or None."""
    from air.ir import MemRefType

    st, _ = MemRefType(t).get_strides_and_offset()
    return None if any(k < 0 for k in st) else st


def _l1_copies(h, mine):
    """The copies in herd `h` between a channel's buffer of the core (an
    alloc whose other users are `air.channel` ops) and a view of one of the
    L1 buffers `mine`: `memref.copy` ops, or a whole-buffer
    `vector.transfer_read` written whole by a `vector.transfer_write`. Each
    as (ops to erase, the channel's buffer, the view)."""
    from air.ir import MemRefType, OpResult

    def channel_buffer(v):
        if not (isinstance(v, OpResult) and v.owner.name == "memref.alloc"):
            return False
        users = [u.owner for u in v.uses]
        return any(u.name.startswith("air.channel.") for u in users) and all(
            u.name.startswith("air.channel.")
            or u.name in ("memref.dealloc", "memref.copy", "linalg.copy")
            or u.name == "memref.subview"
            or u.name in ("vector.transfer_read", "vector.transfer_write")
            for u in users
        )

    out = []
    for loop in _ops_named(h, "scf.for"):
        c = _tiled_copy(loop)
        if c is None:
            continue
        src, dst, n = c
        for tmp, view in ((src, dst), (dst, src)):
            if (
                channel_buffer(tmp)
                and MemRefType(tmp.type).shape == [n]
                and _home_of(view, mine)
            ):
                out.append(([loop], tmp, _prefix(view, n)))
    for c in _ops_named(h, "memref.copy") + _ops_named(h, "linalg.copy"):
        src, dst = c.operands[0], c.operands[1]
        for tmp, view in ((src, dst), (dst, src)):
            if channel_buffer(tmp) and _home_of(view, mine):
                out.append(([c], tmp, view))
    for r in _ops_named(h, "vector.transfer_read"):
        res = r.results[0]
        users = [u.owner for u in res.uses]
        if len(users) != 1 or users[0].name != "vector.transfer_write":
            continue
        w = users[0]
        src, dst = r.operands[0], w.operands[1]
        for tmp, view in ((src, dst), (dst, src)):
            if channel_buffer(tmp) and _home_of(view, mine):
                out.append(([w, r], tmp, view))
    # A buffer copied more than once is not just a channel's: it keeps them.
    return [c for c in out if sum(c[1] == d[1] for d in out) == 1]


def _fold_constant_casts(op, seg):
    """Index casts of integer constants in `op`, as index constants, and
    unused casts in segment `seg` dropped: AIR reads loop bounds and offsets
    as constants only when they are, and a loop of the segment that only
    re-sends a buffer as one re-sent buffer only when its body holds nothing
    else that reads the loop's index. (The host's loops keep theirs: a
    transfer of the host is not re-sent.)"""
    from air.dialects import arith
    from air.ir import IndexType, InsertionPoint, IntegerAttr

    idx = IndexType.get()
    for c in _ops_named(seg, "arith.index_cast"):
        if not list(c.results[0].uses):
            c.erase()
    for c in _ops_named(op, "arith.index_cast"):
        src = c.operands[0]
        d = getattr(src.owner, "name", None)
        if d != "arith.constant":
            continue
        v = IntegerAttr(src.owner.attributes["value"]).value
        with InsertionPoint(c):
            k = arith.ConstantOp(idx, IntegerAttr.get(idx, v)).result
        c.results[0].replace_all_uses_with(k)
        c.erase()


def _scalar_transfers(seg):
    """In the herds, vector reads and writes of a 0-d buffer (a reduction's
    scalar result) as a scalar load and store."""
    from air.dialects import memref, vector
    from air.ir import InsertionPoint, MemRefType, VectorType

    for h in _ops_named(seg, "air.herd"):
        for r in _ops_named(h, "vector.transfer_read"):
            src = r.operands[0]
            if MemRefType(src.type).rank != 0:
                continue
            with InsertionPoint(r):
                v = memref.LoadOp(src, []).result
                b = vector.BroadcastOp(VectorType(r.results[0].type), v).result
            r.results[0].replace_all_uses_with(b)
            r.erase()
        for w in _ops_named(h, "vector.transfer_write"):
            dst = w.operands[1]
            if MemRefType(dst.type).rank != 0:
                continue
            with InsertionPoint(w):
                v = vector.ExtractOp(w.operands[0], [], []).result
                memref.StoreOp(v, dst, [])
            w.erase()


def _forward_private_vectors(seg):
    """In the herds, a buffer of the core that one vector write fills whole
    and vector reads then read whole, in the same block: the reads take the
    written vector, and the buffer goes."""
    from air.ir import MemRefType, VectorType

    for h in _ops_named(seg, "air.herd"):
        for a in _ops_named(h, "memref.alloc"):
            buf = a.results[0]
            t = MemRefType(buf.type)
            users = [u.owner.operation for u in buf.uses]
            writes = [u for u in users if u.name == "vector.transfer_write"]
            reads = [u for u in users if u.name == "vector.transfer_read"]
            if len(writes) != 1 or not reads:
                continue
            if any(
                u.name
                not in (
                    "vector.transfer_write",
                    "vector.transfer_read",
                    "memref.dealloc",
                )
                for u in users
            ):
                continue
            w = writes[0]
            val = w.operands[0]
            if VectorType(val.type).shape != t.shape or w.operands[1] != buf:
                continue
            if not all(
                r.block == w.block
                and w.is_before_in_block(r)
                and r.results[0].type == val.type
                and r.operands[0] == buf
                and all(_const(i) == 0 for i in r.operands[1:-1])
                for r in reads
            ):
                continue
            if not all(_const(i) == 0 for i in w.operands[2:]):
                continue
            for r in reads:
                r.results[0].replace_all_uses_with(val)
                r.erase()
            w.erase()
            for u in list(buf.uses):
                u.owner.erase()
            a.erase()


def _tiled_copy(loop):
    """(source, destination, n) where `loop` copies the first n elements of
    a 1-D buffer into another, a vector at a time: `scf.for` from 0 to n
    whose body reads a subview of one at the induction variable and writes
    it to the same subview of the other; else None."""
    from air.ir import MemRefType

    body = loop.regions[0].blocks[0]
    ops = [o for o in body.operations if o.operation.name != "scf.yield"]
    names = [o.operation.name for o in ops]
    if sorted(names) != sorted(
        [
            "memref.subview",
            "memref.subview",
            "vector.transfer_read",
            "vector.transfer_write",
        ]
    ):
        return None
    iv = body.arguments[0]
    read = next(o for o in ops if o.operation.name == "vector.transfer_read")
    write = next(o for o in ops if o.operation.name == "vector.transfer_write")
    if write.operands[0] != read.results[0]:
        return None
    ends = []
    for v in (read.operands[0], write.operands[1]):
        sv = v.owner
        if sv.name != "memref.subview" or list(sv.opview.offsets) != [iv]:
            return None
        if list(sv.opview.static_strides) != [1]:
            return None
        ends.append(sv.opview.source)
    lb, ub = loop.operands[0], loop.operands[1]
    n = [MemRefType(e.type).shape for e in ends]
    if _const(lb) != 0 or any(len(x) != 1 or x[0] < _const(ub) for x in n):
        return None
    return ends[0], ends[1], _const(ub)


def _prefix(view, n):
    """The first `n` elements of 1-D `view`: itself, or a subview of it."""
    from air.dialects import memref
    from air.ir import InsertionPoint, MemRefType, StridedLayoutAttr

    t = MemRefType(view.type)
    if t.shape == [n]:
        return view
    st, off = t.get_strides_and_offset()
    rt = MemRefType.get(
        [n],
        t.element_type,
        layout=StridedLayoutAttr.get(off, st),
        memory_space=t.memory_space,
    )
    with InsertionPoint.after(view.owner):
        return memref.SubViewOp(rt, view, [], [], [], [0], [n], [1]).result


def _copy_in_place(ops, tmp, view):
    """Have the channel ops on buffer `tmp` move the data to or from `view`
    itself (or the whole buffer it views), and drop the copy `ops`."""
    from air.ir import MemRefType

    for o in ops:
        o.erase()
    home, off = _home_of(view, _roots(view))
    ht, vt = MemRefType(home.type), MemRefType(view.type)
    st = _strides_of(view.type)
    whole = (
        off == 0
        and st is not None
        and list(vt.shape) == list(ht.shape)
        and _contiguous(list(vt.shape), st)
    )
    target = home if whole else view
    # The view, and what it is computed from, before the channel ops.
    _move_before(target, tmp.owner)
    for u in list(tmp.uses):
        if u.owner.name == "memref.dealloc":
            u.owner.erase()
    tmp.replace_all_uses_with(target)
    tmp.owner.erase()


def _roots(v):
    """The buffer at the root of view `v`."""
    from air.ir import OpResult

    while isinstance(v, OpResult) and v.owner.name in (
        "memref.reinterpret_cast",
        "memref.subview",
    ):
        v = v.owner.opview.source
    return [v]


def _view(home, offs, sizes, strides, like):
    """A view of the part of buffer `home` at `offs` with `sizes` and
    `strides`, of the element type and memory space of `like`."""
    from air.dialects import arith, memref
    from air.ir import (
        IndexType,
        IntegerAttr,
        MemRefType,
        ShapedType,
        StridedLayoutAttr,
    )

    t = home.type if isinstance(home.type, MemRefType) else None
    if (
        t is not None
        and t
        == MemRefType.get(list(sizes), t.element_type, memory_space=t.memory_space)
        and all(o == 0 for o in offs)
        and _contiguous(sizes, strides)
    ):
        # The whole buffer: itself, so that AIR's locks on the buffer cover
        # the transfer.
        return home
    idx = IndexType.get()
    dyn = ShapedType.get_dynamic_stride_or_offset()
    const = sum(o * s for o, s in zip(offs, strides) if isinstance(o, int))
    at = arith.ConstantOp(idx, IntegerAttr.get(idx, const)).result
    for o, st in zip(offs, strides):
        if not isinstance(o, int):
            k = arith.ConstantOp(idx, IntegerAttr.get(idx, st)).result
            at = arith.AddIOp(at, arith.MulIOp(o, k).result).result
    layout = StridedLayoutAttr.get(dyn, list(strides))
    vt = MemRefType.get(
        list(sizes), like.element_type, layout=layout, memory_space=like.memory_space
    )
    return memref.ReinterpretCastOp(
        vt, home, [at], [], [], [dyn], list(sizes), list(strides)
    ).result


def _contiguous(sizes, strides):
    want = 1
    for n, st in reversed(list(zip(sizes, strides))):
        if st != want:
            return False
        want *= n
    return True


def _move_before(v, anchor):
    """Move the arithmetic and views that compute `v`, where they follow
    `anchor` in the same block, before `anchor`."""
    from air.ir import OpResult

    if not isinstance(v, OpResult):
        return
    op = v.owner.operation
    if op.block != anchor.operation.block or not (
        op.name.startswith("arith.") or op.name in _VIEWS
    ):
        return
    for x in op.operands:
        _move_before(x, anchor)
    if anchor.operation.is_before_in_block(op):
        op.move_before(anchor.operation)


def _within(op, h):
    p = op.parent
    while p is not None:
        if p == h.operation:
            return True
        p = p.parent
    return False


def _bind_tile_ids(seg, tiles, shapes):
    """Give the herds that run one program a core the shape `shapes` names
    for them, and their program ids, stand-ins `tiles` by axis, the core's
    index in the herd. The stand-ins are then erased."""
    from air.dialects import arith
    from air.ir import InsertionPoint, IntegerType

    i32 = IntegerType.get_signless(32)
    for h in _ops_named(seg, "air.herd"):
        ops = list(h.operands)
        body = h.regions[0].blocks[0]
        first = len(body.arguments) - len(ops)
        drop = []
        for i, v in enumerate(ops):
            a = next((ax for ax, t in tiles.items() if t == v), None)
            if a is None:
                continue
            if a > 1:
                raise ValueError("a herd has two axes; program_id(2) is not one")
            with InsertionPoint.at_block_begin(body):
                c = arith.IndexCastOp(i32, body.arguments[a])
            body.arguments[first + i].replace_all_uses_with(c.result)
            drop.append(i)
        name = str(h.attributes["sym_name"]).strip('"')
        if name in shapes:
            _drop_herd_operands(h, drop, shapes[name])
        elif drop:
            raise ValueError(f"herd {name} reads program ids but runs one program")
    for t in tiles.values():
        if list(t.uses):
            raise ValueError("a program id is used outside a herd")
        t.owner.erase()


def _drop_herd_operands(h, drop, shape=None):
    """Rebuild herd `h` without its operands at positions `drop`, whose block
    arguments must be unused, and as `shape` (columns, rows) if given."""
    from air.dialects import arith
    from air.dialects.air import HerdOp
    from air.ir import DenseI32ArrayAttr, IndexType, InsertionPoint, IntegerAttr

    op = h.operation
    deps, nsizes, _ = list(DenseI32ArrayAttr(op.attributes["operandSegmentSizes"]))
    if deps:
        raise ValueError("an async herd at this stage")
    operands = list(op.operands)
    keep = [i for i in range(nsizes, len(operands)) if i not in drop]
    sizes = operands[:nsizes]
    with InsertionPoint(op):
        if shape is not None:
            idx = IndexType.get()
            sizes = [
                arith.ConstantOp(idx, IntegerAttr.get(idx, n)).result for n in shape
            ]
        new = HerdOp(None, [], sizes, [operands[i] for i in keep])
    for a in _attr_names(op):
        if a != "operandSegmentSizes":
            new.operation.attributes[a] = op.attributes[a]
    old = op.regions[0].blocks[0]
    first = len(old.arguments) - len(operands)
    kept_args = list(old.arguments[: first + nsizes]) + [
        old.arguments[first + i] for i in keep
    ]
    block = new.regions[0].blocks.append(*[a.type for a in kept_args])
    for a, b in zip(kept_args, block.arguments):
        a.replace_all_uses_with(b)
    for o in list(old.operations):
        o.operation.detach_from_parent()
        block.append(o.operation)
    op.erase()


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
