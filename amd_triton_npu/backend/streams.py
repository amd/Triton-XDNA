# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""DDR streams: lower a herd's transfers between DDR and its cores' local
memory to channels.

A schedule can leave a herd's cores reading operands from DDR and writing
results to it directly (air.dma_memcpy_nd between an L3 view and an L1
buffer inside the herd). `lower_streams` turns each such transfer into

    shim --(<s>_ddr)--> memtile ring --(<s>_l1)--> cores       (reads)
    cores --(<s>_l1)--> memtile buffer --(<s>_ddr)--> shim     (writes)

The cores get or put on the core channel where the transfer was. The memtile
program has no loop; its buffers form a fixed ring. The shim side gets its
access patterns by walking the op's loops at the op's sizes, and IR the walk
cannot evaluate exactly is rejected.

A stream is the transfers on one DDR argument, in one direction, that depend
on the same herd coordinates. A read that depends only on the row is
broadcast to that row's cores, one that depends only on the column to that
column's cores. A write is gathered per column into one memtile buffer.

The launch keeps the shim ops in the order emitted, one task each, and a
channel's later tasks may wait for its earlier ones. Channels whose tasks
never wait (a read channel with one task, a write channel with at most two)
go first; the remaining tasks follow in the order the cores consume them.
"""

from .superkernel import _const, _eval, _ops_named, _space

# the ops a DDR view is taken through, their source first
_VIEWS = (
    "memref.reinterpret_cast",
    "memref.subview",
    "memref.expand_shape",
    "memref.collapse_shape",
    "memref.cast",
)


def _parts(op):
    """The number of async dependences and of sizes of hierarchy op `op`."""
    from air.ir import DenseI32ArrayAttr

    seg = list(DenseI32ArrayAttr(op.attributes["operandSegmentSizes"]))
    return seg[0], seg[1]


def _hier_operand(arg):
    """The operand of an air hierarchy op (herd, segment, launch) bound to its
    block argument `arg`, or None for its ids and sizes."""
    op = arg.owner.owner.operation
    if op.name not in ("air.herd", "air.segment", "air.launch"):
        return None
    deps, nsizes = _parts(op)
    k = arg.arg_number - 2 * nsizes
    if k < 0:
        return None
    return op.operands[deps + nsizes + k]


def _hier_arg(v, env):
    """A hierarchy op's block argument, evaluated: its operand's or size's
    value; a launch's own id is 0, as the dispatch launches it once."""
    o = _hier_operand(v)
    if o is not None:
        return _ev(o, env)
    op = v.owner.owner.operation
    if op.name in ("air.herd", "air.segment", "air.launch"):
        deps, nsizes = _parts(op)
        sizes = list(op.operands)[deps : deps + nsizes]
        if nsizes <= v.arg_number < 2 * nsizes:
            return _ev(sizes[v.arg_number - nsizes], env)
        if op.name == "air.launch" and v.arg_number < nsizes:
            if any(_ev(x, env) != 1 for x in sizes):
                raise ValueError("a launch of more than one iteration")
            return 0
    raise ValueError(f"can't evaluate {v}")


def _ev(v, env):
    """`v` evaluated with `env` (block arguments -> ints), through the
    hierarchy ops' arguments; ValueError if it is not a known constant."""
    return _eval(v, env, _hier_arg)


def _root(v):
    """The func argument behind memref `v`, through views and the hierarchy
    ops' arguments, or None."""
    from air.ir import BlockArgument, OpResult

    while True:
        if isinstance(v, BlockArgument):
            if v.owner.owner.operation.name == "func.func":
                return v
            v = _hier_operand(v)
            if v is None:
                return None
        elif isinstance(v, OpResult) and v.owner.name in _VIEWS:
            v = v.owner.operands[0]
        else:
            return None


def _static(xs, what):
    """`xs`, which must not hold the dynamic marker."""
    from air.ir import ShapedType

    if (
        ShapedType.get_dynamic_size() in xs
        or ShapedType.get_dynamic_stride_or_offset() in xs
    ):
        raise ValueError(f"a DDR view of dynamic {what}")
    return xs


def _strides_of(t):
    """The layout strides of memref type `t`."""
    from air.ir import MemRefType

    strides, _ = MemRefType(t).get_strides_and_offset()
    return _static(list(strides), "strides")


def _offset(v, env):
    """The element offset of memref view `v` from its func argument."""
    from air.ir import BlockArgument, DenseI64ArrayAttr, ShapedType

    if isinstance(v, BlockArgument):
        if v.owner.owner.operation.name == "func.func":
            return 0
        return _offset(_hier_operand(v), env)
    op = v.owner
    name = op.name
    src = op.operands[0]
    dynamic = ShapedType.get_dynamic_stride_or_offset()
    if name == "memref.reinterpret_cast":
        static = list(DenseI64ArrayAttr(op.attributes["static_offsets"]))
        dyn = iter(op.opview.offsets)
        off = static[0]
        if off == dynamic:
            off = _ev(next(dyn), env)
        # from the source's base, which every view keeps: the func argument's
        return off
    if name == "memref.subview":
        static = list(DenseI64ArrayAttr(op.attributes["static_offsets"]))
        dyn = iter(op.opview.offsets)
        off = 0
        for s, st in zip(static, _strides_of(src.type)):
            off += (_ev(next(dyn), env) if s == dynamic else s) * st
        return _offset(src, env) + off
    if name in ("memref.expand_shape", "memref.collapse_shape", "memref.cast"):
        return _offset(src, env)
    raise ValueError(f"can't follow {name}")


def _dma_side(d, side, env):
    """The element offset of `side` ("src" or "dst") of DMA `d` in that side's
    memref, and its sizes and strides."""
    from air.ir import DenseI64ArrayAttr, MemRefType, ShapedType

    def part(k):
        static = list(DenseI64ArrayAttr(d.attributes[f"static_{side}_{k}"]))
        dyn = iter(getattr(d.opview, f"dynamic_{side}_{k}"))
        dynv = ShapedType.get_dynamic_size()
        return [_ev(next(dyn), env) if s == dynv else s for s in static]

    offs, sizes, strides = part("offsets"), part("sizes"), part("strides")
    if not sizes:
        t = getattr(d.opview, side).type
        sizes, strides = _static(list(MemRefType(t).shape), "sizes"), _strides_of(t)
        offs = [0] * len(sizes)
    # missing offsets are the leading ones, as AIR reads them
    offs = [0] * (len(sizes) - len(offs)) + offs
    return sum(o * s for o, s in zip(offs, strides)), sizes, strides


def _bytes(t):
    """The bytes of an element of type `t`."""
    from air.ir import FloatType, IntegerType

    for cls in (IntegerType, FloatType):
        try:
            return cls(t).width // 8
        except ValueError:
            pass
    raise ValueError(f"a stream of {t}")


def _elements(d, side):
    """The number of elements DMA `d` moves, from its static `side`."""
    from air.ir import DenseI64ArrayAttr, MemRefType

    sizes = list(DenseI64ArrayAttr(d.attributes[f"static_{side}_sizes"]))
    sizes = sizes or list(MemRefType(getattr(d.opview, side).type).shape)
    n = 1
    for c in sizes:
        n *= c
    return n


class _Stream:
    """The transfers of one stream. `seq` maps (instance, core) to the core's
    transfers, in program order: (DDR element offset, sizes, strides, place in
    the core's program). An instance is the coordinates the stream depends
    on."""

    def __init__(self, arg, read, dims, et):
        self.arg, self.read, self.dims, self.et = arg, read, dims, et
        self.seq = {}
        self.dmas = []
        self.herd = None  # (sym_name, nx, ny) of the herds that move it


def _herd_dims(d, herd):
    """Which herd coordinates (0: column, 1: row) DMA `d`'s DDR side depends
    on."""
    from air.ir import OpResult

    from .superkernel import _depends_on

    blk = herd.regions[0].blocks[0]
    side = "src" if _space(d.opview.src) != 2 else "dst"
    vals = [
        v
        for k in ("offsets", "sizes", "strides")
        for v in getattr(d.opview, f"dynamic_{side}_{k}")
    ]
    # and the operands of the views the DDR side is taken through
    v = getattr(d.opview, side)
    while isinstance(v, OpResult) and v.owner.name in _VIEWS:
        vals += list(v.owner.operands)[1:]
        v = v.owner.operands[0]
    return tuple(
        i for i in (0, 1) if any(_depends_on(v, blk.arguments[i]) for v in vals)
    )


def _ddr_transfer(d):
    """(read, DDR side) of DMA `d`, or None if it moves within L1."""
    src, dst = d.opview.src, d.opview.dst
    if _space(src) == 2 and _space(dst) == 2:
        return None
    read = _space(dst) == 2
    ddr, l1 = (src, dst) if read else (dst, src)
    if _space(ddr) != 0 or _space(l1) != 2:
        raise ValueError(f"a core transfer that is not DDR <-> L1: {d}")
    return read, ddr


def _collect(seg):
    """Walk the segment in program order, its loops at their trip counts and
    each herd's cores in turn, and gather the streams of its DDR transfers."""
    from air.ir import MemRefType, StringAttr

    streams = {}
    holds = {}  # op -> whether it holds a DMA
    places = {}  # core -> the transfers it has made, over the segment's herds

    def has_dma(op):
        if op not in holds:
            holds[op] = bool(_ops_named(op, "air.dma_memcpy_nd"))
        return holds[op]

    def branch(op, env):
        """The block scf.if `op` runs under `env`, or None."""
        taken = 0 if _ev(op.operands[0], env) else 1
        if taken < len(op.regions) and op.regions[taken].blocks:
            return op.regions[taken].blocks[0]
        return None

    def visit_block(block, env):
        for o in block.operations:
            op = o.operation
            if op.name == "air.herd":
                visit_herd(op, env)
            elif op.name == "air.dma_memcpy_nd":
                raise ValueError(f"a segment transfer outside the cores: {op}")
            elif not op.regions or not has_dma(op):
                continue
            elif op.name == "scf.for":
                lo, hi, st = (_ev(v, env) for v in list(op.operands)[:3])
                iv = op.regions[0].blocks[0].arguments[0]
                for x in range(lo, hi, st):
                    visit_block(op.regions[0].blocks[0], env + [(iv, x)])
            elif op.name == "scf.if":
                b = branch(op, env)
                if b is not None:
                    visit_block(b, env)
            else:
                raise ValueError(
                    f"a transfer under {op.name}, which the walk can't follow"
                )

    # The streams, from every herd's transfers, whether or not their loops run
    # at this op's sizes: every op of a kernel gets the same channels.
    for h in _ops_named(seg, "air.herd"):
        nx, ny = (_const(v) for v in list(h.operands)[:2])
        for d in _ops_named(h, "air.dma_memcpy_nd"):
            t = _ddr_transfer(d)
            if t is None:
                continue
            read, ddr = t
            if d.results and list(d.results[0].uses):
                raise ValueError(f"a transfer whose token is used: {d}")
            if list(d.opview.async_dependencies):
                raise ValueError(f"a transfer that waits for tokens: {d}")
            if nx is None or ny is None:
                raise ValueError("a herd of non-constant size moves DDR data")
            arg = _root(ddr)
            if arg is None:
                raise ValueError(f"no func argument behind {d}")
            dims = _herd_dims(d, h)
            l1 = d.opview.dst if read else d.opview.src
            key = (arg.arg_number, read, dims)
            if key not in streams:
                streams[key] = _Stream(
                    arg, read, dims, MemRefType(l1.type).element_type
                )
            s = streams[key]
            herd = (StringAttr(h.attributes["sym_name"]).value, nx, ny)
            if s.herd not in (None, herd):
                raise ValueError(f"herds {s.herd} and {herd} move one stream")
            s.herd = herd
            if not any(x == d for x in s.dmas):
                s.dmas.append(d)

    def visit_herd(h, env):
        blk = h.regions[0].blocks[0]
        nx, ny = (_ev(v, env) for v in list(h.operands)[:2])

        # each core runs the herd body: the DMAs in program order, with loops
        # (only the ops that hold a DMA: the rest moves nothing)
        def run(block, env2, core, place):
            for o in block.operations:
                op = o.operation
                if op.name == "air.dma_memcpy_nd":
                    t = _ddr_transfer(op)
                    if t is None:
                        continue
                    read, ddr = t
                    s = streams[(_root(ddr).arg_number, read, _herd_dims(op, h))]
                    off, sizes, strides = _dma_side(op, "src" if read else "dst", env2)
                    inst = tuple(core[k] for k in s.dims)
                    s.seq.setdefault((inst, core), []).append(
                        (_offset(ddr, env2) + off, sizes, strides, place[0])
                    )
                    place[0] += 1
                elif not op.regions or not has_dma(op):
                    continue
                elif op.name == "scf.for":
                    lo, hi, st = (_ev(v, env2) for v in list(op.operands)[:3])
                    iv = op.regions[0].blocks[0].arguments[0]
                    for x in range(lo, hi, st):
                        run(op.regions[0].blocks[0], env2 + [(iv, x)], core, place)
                elif op.name == "scf.if":
                    b = branch(op, env2)
                    if b is not None:
                        run(b, env2, core, place)
                else:
                    raise ValueError(
                        f"a transfer under {op.name}, which the walk can't follow"
                    )

        for cx in range(nx):
            for cy in range(ny):
                env2 = [(blk.arguments[0], cx), (blk.arguments[1], cy)] + env
                run(blk, env2, (cx, cy), places.setdefault((cx, cy), [0]))

    visit_block(seg.regions[0].blocks[0], [])
    return list(streams.values())


def _fit(starts):
    """Nested (count, stride) dims, outermost first, that generate `starts`
    from starts[0]; None if they do not nest."""
    n = len(starts)
    if n == 1:
        return []
    d = starts[1] - starts[0]
    if all(starts[i + 1] - starts[i] == d for i in range(n - 1)):
        return [(n, d)]
    for p in range(2, n // 2 + 1):
        if n % p:
            continue
        blocks = [starts[i * p : (i + 1) * p] for i in range(n // p)]
        rel = [b - blocks[0][0] for b in blocks[0]]
        if any([b - blk[0] for b in blk] != rel for blk in blocks):
            continue
        inner = _fit(blocks[0])
        outer = _fit([blk[0] for blk in blocks])
        if inner is not None and outer is not None:
            return outer + inner
    return None


def _merge(dims):
    """Drop unit dims and merge a dim into the next where they are one run."""
    dims = [(c, s) for c, s in dims if c != 1]
    out = []
    for c, s in reversed(dims):
        if out and out[0][1] * out[0][0] == s:
            out[0] = (out[0][0] * c, out[0][1])
        else:
            out.insert(0, (c, s))
    return out or [(1, 1)]


# A shim BD (npu1, npu2): three addressing dims of wraps up to 1023 and an
# iteration dim of up to 64, steps of 20 bits, in 32-bit words.
_BD_DIMS, _BD_WRAP, _BD_ITER, _BD_STEP = 3, 1023, 64, 1 << 20


def _fits(sizes, strides, eb):
    """Whether the pattern `sizes`/`strides`, of elements of `eb` bytes, is one
    shim BD: a linear run, or addressing dims with an iteration dim (a repeat,
    or the outermost of four) around them."""
    dims = [(c, s) for c, s in zip(sizes, strides) if c != 1]
    if any(s < 0 for _, s in dims):
        return False
    words = lambda n: n * eb / 4  # noqa: E731
    if dims and (dims[0][1] == 0 or len(dims) > _BD_DIMS):
        (c, s), dims = dims[0], dims[1:]
        if c > _BD_ITER or words(s) >= _BD_STEP or words(s) != int(words(s)):
            return False
    if len(dims) <= 1 and (not dims or dims[0][1] == 1):
        return True
    if len(dims) > _BD_DIMS or any(s == 0 for _, s in dims):
        return False
    inner = dims[-1]
    if inner[1] == 1:  # its wrap and the others' steps count words
        dims = dims[:-1] + [(words(inner[0]), 1)]
        steps = [words(s) for _, s in dims[:-1]]
    else:
        steps = [words(s) for _, s in dims]
    return all(c <= _BD_WRAP and c == int(c) for c, _ in dims) and all(
        st < _BD_STEP and st == int(st) for st in steps
    )


def _offsets(base, sizes, strides):
    """A pattern starting at element `base`, as AIR spells it: an offset
    counts its dim's stride, so the start goes in a dim whose stride divides
    it, the innermost such, else in a leading dim of one element."""
    for i in reversed(range(len(sizes))):
        if strides[i] and base % strides[i] == 0:
            offs = [0] * len(sizes)
            offs[i] = base // strides[i]
            return offs, list(sizes), list(strides)
    return [base] + [0] * len(sizes), [1] + list(sizes), [1] + list(strides)


def _tasks(base, sizes, strides, eb):
    """The pattern split along its outer dims into shim BDs: (offsets, sizes,
    strides) and the index of its first element in the pattern's order."""
    if _fits(sizes, strides, eb) or len(sizes) == 1:
        if not _fits(sizes, strides, eb):
            # a single dim that no BD can describe
            raise ValueError(f"a transfer of {sizes} {strides} that no BD can make")
        return [(_offsets(base, sizes, strides), 0)]
    n = 1
    for c in sizes[1:]:
        n *= c
    return [
        (p, i * n + e)
        for i in range(sizes[0])
        for p, e in _tasks(base + i * strides[0], sizes[1:], strides[1:], eb)
    ]


def _shim_pattern(seq):
    """One access pattern for a sequence of transfers of the same shape: its
    first element and its sizes and strides, in elements of the DDR
    argument."""
    shapes = {(tuple(sz), tuple(st)) for _, sz, st, *_ in seq}
    if len(shapes) != 1:
        raise ValueError("a stream's transfers differ in shape")
    sizes, strides = map(list, shapes.pop())
    outer = _fit([b for b, *_ in seq])
    if outer is None:
        raise ValueError("a stream's transfers do not form one access pattern")
    dims = _merge(outer + list(zip(sizes, strides)))
    return seq[0][0], [c for c, _ in dims], [s for _, s in dims]


def _shim_ops(seq, eb):
    """The shim ops of a sequence of transfers (elements of `eb` bytes), each
    with the place in the cores' program of its first transfer. When every
    transfer is one contiguous run and the runs advance through the argument,
    one linear op per run (runs that touch are merged), so no BD stride field
    has to hold the distance between runs. Otherwise one op, split into as
    many BDs as it needs."""
    base, sz, st = _shim_pattern(seq)
    n = 1
    for c in seq[0][1]:
        n *= c
    one = _merge(list(zip(seq[0][1], seq[0][2])))
    if len(one) == 1 and one[0][1] == 1 and len(sz) > 1 and 0 not in st[:-1]:
        starts = [base]
        for c, d in zip(sz[:-1], st[:-1]):
            starts = [b + i * d for b in starts for i in range(c)]
        runs = []  # [start, elements, first element in the sequence]
        for i, b in enumerate(starts):
            if runs and runs[-1][0] + runs[-1][1] == b:
                runs[-1][1] += sz[-1]
            else:
                runs.append([b, sz[-1], i * sz[-1]])
    else:
        runs = None
    ops = []
    for b, sizes, strides, e0 in (
        [(b, [c], [1], e) for b, c, e in runs] if runs else [(base, sz, st, 0)]
    ):
        for p, e in _tasks(b, sizes, strides, eb):
            ops.append((p, seq[(e0 + e) // n][3]))
    return ops


def _launch_value(launch, arg):
    """The launch's block argument bound to func argument `arg`."""
    deps, nsizes = _parts(launch)
    ops = list(launch.operands)[deps + nsizes :]
    k = next(i for i, o in enumerate(ops) if o == arg)
    return launch.regions[0].blocks[0].arguments[2 * nsizes + k]


def lower_streams(module, name="s"):
    """See _lower_streams; under `module`'s context."""
    from air.ir import Location

    with module.context, Location.unknown():
        return _lower_streams(module, name)


def _lower_streams(module, name):
    """Lower the DDR transfers of the herds of `module` (see the module's
    documentation). Its channels are named `<name><k>_l1` (cores) and
    `<name><k>_ddr` (shim). The memtile buffers carry `npu.stream_col`, the
    column of herd `npu.stream_herd` whose memtile they go in. Returns the
    streams."""
    from air.dialects import memref
    from air.dialects.air import Channel, ChannelGet, ChannelPut
    from air.ir import (
        DenseI64ArrayAttr,
        InsertionPoint,
        IntegerAttr,
        IntegerType,
        MemRefType,
        OpResult,
        StringAttr,
        UnitAttr,
    )

    segs = _ops_named(module.operation, "air.segment")
    launches = _ops_named(module.operation, "air.launch")
    if len(segs) != 1 or len(launches) != 1:
        raise ValueError("streams need one launch of one segment")
    seg, launch = segs[0], launches[0]
    streams = _collect(seg)
    # the shim issues each stream's tasks on its own, so a write is not
    # ordered before a later read of the same buffer
    reads = {s.arg.arg_number for s in streams if s.read}
    if any(not s.read and s.arg.arg_number in reads for s in streams):
        raise ValueError("a DDR buffer both read and written by streams")
    if len({s.herd[0] for s in streams}) > 1:
        # such herds run concurrently, but the shim ops are emitted in one order
        raise ValueError("herds of different names move DDR data")
    i32 = IntegerType.get_signless(32)
    sblock = seg.regions[0].blocks[0]
    first = list(sblock.operations)[0]
    dead = []  # the DDR views only the transfers used, erased at the end
    term = list(sblock.operations)[-1]
    seg_op = next(
        o for o in launch.regions[0].blocks[0].operations if o.operation == seg
    )
    l2 = IntegerAttr.get(i32, 1)
    # keep the shim ops in the order emitted, one task each, paced per channel
    launch.attributes["air.preserve_shim_dma_order"] = UnitAttr.get()
    once = []  # the shim ops of channels of one task: (k, op, channel, ...)
    paced = []  # the others: (place, k, ...)

    def alloc(shape, et, col, herd):
        a = memref.AllocOp(MemRefType.get(shape, et, memory_space=l2), [], [])
        a.attributes["npu.stream_col"] = IntegerAttr.get(i32, col)
        a.attributes["npu.stream_herd"] = StringAttr.get(herd)
        return a.result

    for k, s in enumerate(streams):
        nm = f"{name}{k}"
        hname, nx, ny = s.herd
        side = "dst" if s.read else "src"
        if not s.dims:
            raise ValueError(f"stream {nm}: a transfer the same for every core")
        if not s.read and s.dims != (0, 1):
            raise ValueError(f"stream {nm}: a write must depend on both coordinates")
        # the cores of one instance must see one sequence
        per = {}
        for core in ((cx, cy) for cx in range(nx) for cy in range(ny)):
            inst = tuple(core[d] for d in s.dims)
            seq = s.seq.get((inst, core), [])
            if inst in per and per[inst] != seq:
                raise ValueError(f"stream {nm}: the cores of one broadcast differ")
            per.setdefault(inst, seq)
        # every instance, whether or not it moves anything at this op's sizes
        ranges = [
            range(nx) if 0 in s.dims else [None],
            range(ny) if 1 in s.dims else [None],
        ]
        insts = [
            tuple(c for c in (x, y) if c is not None)
            for x in ranges[0]
            for y in ranges[1]
        ]
        for inst in insts:
            per.setdefault(inst, [])
        if s.read and any(inst[0] >= nx for inst in insts):
            raise ValueError(
                f"stream {nm}: more rows than the herd has memtile columns"
            )
        # one channel to the cores, broadcast along the coordinates the stream
        # does not depend on; one from or to the shim per memtile buffer
        shape = [nx if 0 in s.dims else 1, ny if 1 in s.dims else 1]
        with InsertionPoint.at_block_begin(module.body):
            Channel(
                f"{nm}_l1",
                size=shape,
                broadcast_shape=None if shape == [nx, ny] else [nx, ny],
            )
            Channel(f"{nm}_ddr", size=[len(insts) if s.read else nx])
        # the elements of one transfer: a memtile slot
        counts = {_elements(d, side) for d in s.dmas}
        if len(counts) != 1:
            raise ValueError(f"stream {nm}: its transfers move different amounts")
        n = counts.pop()
        eb = _bytes(s.et)
        # the cores' side: a get or put where the transfer was
        for d in s.dmas:
            h = d.parent
            while h.name != "air.herd":
                h = h.parent
            hb = h.regions[0].blocks[0]
            if any(
                list(getattr(d.opview, f"dynamic_{side}_{q}"))
                for q in ("offsets", "sizes", "strides")
            ):
                raise ValueError(f"stream {nm}: the cores' side moves: {d}")
            pat = [
                list(DenseI64ArrayAttr(d.attributes[f"static_{side}_{q}"]))
                for q in ("offsets", "sizes", "strides")
            ]
            op = ChannelGet if s.read else ChannelPut
            with InsertionPoint(d):
                op(
                    f"{nm}_l1",
                    getattr(d.opview, side),
                    offsets=pat[0],
                    sizes=pat[1],
                    strides=pat[2],
                    indices=[hb.arguments[0], hb.arguments[1]],
                )
            ddr = getattr(d.opview, "src" if s.read else "dst")
            d.erase()
            # and the DDR views only it used (with the views they are of)
            if (
                isinstance(ddr, OpResult)
                and ddr.owner.name in _VIEWS
                and not list(ddr.uses)
            ):
                dead.append(ddr.owner)
        ddr = _launch_value(launch, s.arg)
        if s.read:
            # memtile: a ring of two buffers per instance, in the memtile of
            # its column (for a row's stream, the column with the row's number)
            puts = []
            for j, inst in enumerate(insts):
                with InsertionPoint(first):
                    ring = [alloc([n], s.et, inst[0], hname) for _ in range(2)]
                    li = [0, 0]
                    for dim, c in zip(s.dims, inst):
                        li[dim] = c
                    for b in ring:
                        ChannelGet(f"{nm}_ddr", b, indices=[j])
                    for b in ring:
                        ChannelPut(f"{nm}_l1", b, indices=li)
                with InsertionPoint(term):
                    for b in ring:
                        memref.DeallocOp(b)
                if per[inst]:
                    puts += [(place, j, p) for p, place in _shim_ops(per[inst], eb)]
            if all(
                sum(1 for _, j, _ in puts if j == i) <= 1 for i in range(len(insts))
            ):
                once += [(k, ChannelPut, f"{nm}_ddr", ddr, p, j) for _, j, p in puts]
            else:
                paced += [
                    (place, k, j, ChannelPut, f"{nm}_ddr", ddr, p)
                    for place, j, p in puts
                ]
        else:
            # memtile: per column one buffer, the rows' tiles one after another;
            # shim: the column's buffers in turn, each the rows' tiles
            for x in range(nx):
                with InsertionPoint(first):
                    b = alloc([ny * n], s.et, x, hname)
                    b.owner.attributes["air.no_split"] = UnitAttr.get()
                    for y in range(ny):
                        ChannelGet(
                            f"{nm}_l1",
                            b,
                            offsets=[y * n],
                            sizes=[n],
                            strides=[1],
                            indices=[x, y],
                        )
                    ChannelPut(f"{nm}_ddr", b, indices=[x])
                with InsertionPoint(term):
                    memref.DeallocOp(b)
                rows = [per[(x, y)] for y in range(ny)]
                if len({len(r) for r in rows}) != 1:
                    raise ValueError(f"stream {nm}: rows write different counts")
                seq = []
                for t in range(len(rows[0])):
                    if len({(tuple(r[t][1]), tuple(r[t][2])) for r in rows}) != 1:
                        raise ValueError(f"stream {nm}: rows write different shapes")
                    bases = [r[t][0] for r in rows]
                    dy = bases[1] - bases[0] if ny > 1 else 0
                    if any(bases[y] != bases[0] + y * dy for y in range(ny)):
                        raise ValueError(
                            f"stream {nm}: a column's rows are not one tile"
                        )
                    if len({r[t][3] for r in rows}) != 1:
                        raise ValueError(
                            f"stream {nm}: a column's rows write at different points"
                        )
                    _, sz, st, place = rows[0][t]
                    seq.append((bases[0], [ny] + list(sz), [dy] + list(st), place))
                # a channel's writes of up to two tasks are never waited for
                # before the end: they go first; more are paced
                gets = _shim_ops(seq, eb) if seq else []
                if len(gets) <= 2:
                    once += [(k, ChannelGet, f"{nm}_ddr", ddr, p, x) for p, _ in gets]
                else:
                    paced += [
                        (place, k, x, ChannelGet, f"{nm}_ddr", ddr, p)
                        for p, place in gets
                    ]
    # the channels whose tasks are never waited for first, stream by stream;
    # then the paced ones in the order the cores take them, a round's
    # instances in turn
    ops = [(op, ch, ddr, p, j) for _, op, ch, ddr, p, j in once]
    ops += [
        (op, ch, ddr, p, j)
        for _, _, j, op, ch, ddr, p in sorted(paced, key=lambda t: t[:3])
    ]
    for op, ch, ddr, (o, sz, st), j in ops:
        with InsertionPoint(seg_op):
            op(ch, ddr, offsets=o, sizes=sz, strides=st, indices=[j])
    for view in dead:
        src = view.operands[0]
        view.erase()
        while (
            isinstance(src, OpResult)
            and src.owner.name in _VIEWS
            and not list(src.uses)
        ):
            view = src.owner
            src = view.operands[0]
            view.erase()
    return streams
