# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Superkernels: one `@triton.jit` function that runs kernels as AIR herds,
side by side, wave after wave, as a single dispatch.

    @triton.jit
    def layer(X, W, NW: tl.constexpr, D: tl.constexpr, DP: tl.constexpr):
        h = air.alloc([DP], tl.bfloat16, memory_space=air.L2)
        for wave in range(NW):
            air.herd(scale, (X, W, h, wave, D, DP), [1, 1], SCHEDULE)
            air.herd(add, (X, h, X, D, DP), [1, 1], SCHEDULE)

    layer[(1,)](x, w, NW=4, D=1536, DP=2048)

The function's body is AIR's segment, looped over the waves. `herd` runs a
kernel on its own cores; the herds of a wave all run at the same time, as in
a segment, and their program order is the order of the DDR writes they see
and of the transfers the dispatch issues. A herd's function is a kernel and
its arguments, or a `@triton.jit` function that calls kernels in turn on the
same cores; calls of one kernel that differ only in their numbers and in the
L2 buffers they write become one loop over phases. Herds of one name run in
turn on the same cores.

Herds exchange data through L2 buffers from `alloc` (a buffer read in a wave
must be written by some herd in that wave, which the cores wait for) and
through channels (`Channel`, `ChannelPut`, `ChannelGet`, as AIR's). DDR
buffers, the pointer arguments, are updated in place.

A herd's `size` is its core count, an int or (columns, rows). A kernel that
reads `tl.program_id` runs one program a core, its program ids the core's
index in the herd. Any other kernel is split over its cores by its
`schedule`: the path of a transform script whose `@__transform_main` takes,
after the payload, one `!transform.param<i64>` per core dimension, for use
as `num_threads`. Herds are placed on the array in order unless `at` gives
the (column, row) of a herd's corner.

The loop over the waves must have a constexpr trip count. Ints computed from
the wave index, such as `wave * 2 + 1` or `air.table(values, wave)`, are
evaluated per wave when the dispatch is built; int arguments of the function
are set by the host at each launch (mark them `do_not_specialize` so that a
new value does not recompile the dispatch).
"""

from triton._C.libtriton import ir
from triton.language import core

__all__ = ["Channel", "ChannelGet", "ChannelPut", "L1", "L2", "alloc", "herd", "table"]

#: AIR's memory spaces
L1 = core.constexpr(2)
L2 = core.constexpr(1)


class _PartitionContext:
    """Compiles a herd's function apart from the caller's, under a name of
    its own, and marks it with the herd's settings. The functions it
    calls stay out of line as well, so that the backend finds the kernels."""

    def __init__(self, step, index, cores, schedule, at, name):
        self.step, self.index = step, index
        self.cores, self.schedule, self.at, self.name = cores, schedule, at, name
        self.marked = False

    def mangle(self):
        return f"_H{self.step}"

    def initialize_callee(self, fn, builder):
        fn.set_attr("noinline", builder.get_bool_attr(True))
        if self.marked:
            return
        self.marked = True
        i32 = builder.get_int32_attr
        fn.set_attr("air.step", i32(self.step))
        fn.set_attr("air.herd", i32(self.index))
        fn.set_attr("air.cores_x", i32(self.cores[0]))
        fn.set_attr("air.cores_y", i32(self.cores[1]))
        fn.set_attr("air.schedule", builder.get_string_attr(self.schedule))
        fn.set_attr("air.name", builder.get_string_attr(self.name))
        if self.at is not None:
            fn.set_attr("air.at_col", i32(self.at[0]))
            fn.set_attr("air.at_row", i32(self.at[1]))


def _values(seq):
    seq = core._unwrap_if_constexpr(seq)
    return [core._unwrap_if_constexpr(v) for v in seq]


def _shape(size):
    size = core._unwrap_if_constexpr(size)
    if isinstance(size, int):
        return (size, 1)
    size = tuple(_values(size))
    if len(size) != 2 or not all(isinstance(c, int) and c > 0 for c in size):
        raise ValueError(f"a herd's size is an int or (columns, rows), not {size}")
    return size


def _next_id(generator, counter):
    n = getattr(generator, counter, 0)
    setattr(generator, counter, n + 1)
    return n


@core.builtin
def herd(fn, args, size, schedule, name=None, at=None, _semantic=None, _generator=None):
    """Run `fn` on `args` as a herd of `size` cores; see the module's
    documentation."""
    fn = core._unwrap_if_constexpr(fn)
    schedule = core._unwrap_if_constexpr(schedule)
    if not isinstance(schedule, str):
        raise ValueError(f"herd {fn.__name__}: the schedule must be a path")
    step = _next_id(_generator, "_air_herds")
    at = core._unwrap_if_constexpr(at)
    corner = None if at is None else tuple(_values(at))
    name = core._unwrap_if_constexpr(name) or f"{fn.__name__}_{step}"
    ctx = _PartitionContext(step, 0, _shape(size), schedule, corner, name)
    _generator.call_JitFunction(fn, list(_values(args)), kwargs={}, caller_context=ctx)


@core.builtin
def alloc(
    shape, dtype, memory_space=L2, memtile_col=None, _semantic=None, _generator=None
):
    """A new buffer of the segment, of `shape` elements of `dtype` in
    `memory_space`, as a pointer to its first element; only its herds and
    segment code access it. An L2 buffer is in a memtile, which
    `memtile_col` picks (AIR's `air.memtile_col`). An L1 buffer is in one
    core's memory, which the cores next to it access as their own: herds
    passed it must be neighbours."""
    memory_space = core._unwrap_if_constexpr(memory_space)
    if memory_space not in (L1.value, L2.value):
        raise ValueError("alloc: memory_space is air.L1 or air.L2")
    shape = core._unwrap_if_constexpr(shape)
    n = 1
    for d in [shape] if isinstance(shape, int) else _values(shape):
        n *= d
    dtype = core._unwrap_if_constexpr(dtype)
    b = _semantic.builder
    ptr = core.pointer_type(dtype)
    # A call of a function declared for this buffer alone: the backend finds
    # the buffer's size there, and no two buffers fold into one value.
    name = f"air.alloc.{_next_id(_generator, '_air_allocs')}"
    module = _generator.module
    decl = ir.builder(_generator.context).get_or_insert_function(
        module, name, b.get_function_ty([], [ptr.to_ir(b)]), "private", None
    )
    module.push_back(decl)
    decl.set_attr("air.alloc_elements", b.get_int32_attr(n))
    decl.set_attr("air.memory_space", b.get_int32_attr(memory_space))
    memtile_col = core._unwrap_if_constexpr(memtile_col)
    if memtile_col is not None:
        decl.set_attr("air.memtile_col", b.get_int32_attr(memtile_col))
    call = b.call(module.get_function(name), [])
    return core.tensor(call.get_result(0), ptr)


@core.builtin
def table(values, index, _semantic=None, _generator=None):
    """`values[index]`, for a list of constant ints and an int index."""
    values = _values(values)
    index = _semantic.to_tensor(index)
    out = _semantic.to_tensor(values[-1])
    for i in range(len(values) - 2, -1, -1):
        hit = _semantic.equal(index, _semantic.to_tensor(i))
        out = _semantic.where(hit, _semantic.to_tensor(values[i]), out)
    return out


#: the radix of a channel index's digits in the address that encodes it
_RADIX = 256


def _ints(v):
    v = core._unwrap_if_constexpr(v)
    return [v] if isinstance(v, int) else _values(v)


@core.builtin
def Channel(
    sym_name,
    size=(1,),
    broadcast_shape=None,
    channel_type=None,
    buffer_resources=None,
    _semantic=None,
    _generator=None,
):
    """Declare the channel `sym_name`, as AIR's `Channel`: a bundle of FIFOs
    of shape `size`, of any rank, that carry blocks between cores with no
    buffer of their own in between. `broadcast_shape`, of the bundle's rank,
    is the shape of cores that get each block put at one index;
    `channel_type` is AIR's ("npu_dma_packet", "npu_cascade", ...);
    `buffer_resources` the FIFO's depth. Declared in the superkernel, outside
    the loop over the waves; its kernels use it by name."""
    name = core._unwrap_if_constexpr(sym_name)
    size = _ints(size)
    if any(d >= _RADIX for d in size):
        raise ValueError(f"channel {name}: each extent of {size} below {_RADIX}")
    b = _semantic.builder
    module = _generator.module
    symbol = f"air.channel.{name}"
    if module.has_function(symbol):
        raise ValueError(f"channel {name} is declared twice")
    decl = ir.builder(_generator.context).get_or_insert_function(
        module,
        symbol,
        b.get_function_ty([], [b.get_ptr_ty(b.get_int8_ty(), "global")]),
        "private",
        None,
    )
    module.push_back(decl)
    text = lambda v: b.get_string_attr(",".join(map(str, v)))  # noqa: E731
    decl.set_attr("air.channel_size", text(size))
    broadcast_shape = core._unwrap_if_constexpr(broadcast_shape)
    if broadcast_shape is not None:
        broadcast_shape = _ints(broadcast_shape)
        if len(broadcast_shape) != len(size):
            raise ValueError(f"channel {name}: broadcast_shape is not of size's rank")
        decl.set_attr("air.channel_broadcast_shape", text(broadcast_shape))
    channel_type = core._unwrap_if_constexpr(channel_type)
    if channel_type is not None:
        decl.set_attr("air.channel_type", b.get_string_attr(channel_type))
    buffer_resources = core._unwrap_if_constexpr(buffer_resources)
    if buffer_resources is not None:
        decl.set_attr(
            "air.channel_buffer_resources", b.get_int32_attr(buffer_resources)
        )


def _fifo(chan_name, n, dtype, indices, _semantic, _generator):
    """The addresses of a block of `n` elements of `dtype` at `indices` of
    channel `chan_name`. The indices go into the address as digits of radix
    _RADIX, each digit's place a multiple of the block, in 64 bits."""
    name = core._unwrap_if_constexpr(chan_name)
    module = _generator.module
    symbol = f"air.channel.{name}"
    if not module.has_function(symbol):
        raise ValueError(f"channel {name} is not declared")
    b = _semantic.builder
    call = b.call(module.get_function(symbol), [])
    ptr = core.tensor(call.get_result(0), core.pointer_type(core.int8))
    ptr = ptr.to(core.pointer_type(dtype), bitcast=True, _semantic=_semantic)
    indices = core._unwrap_if_constexpr(indices)
    indices = (
        list(indices) if isinstance(indices, (tuple, list, core.tuple)) else [indices]
    )
    const, off = 0, None
    for k, v in enumerate(indices):
        v = core._unwrap_if_constexpr(v)
        place = n * _RADIX ** (len(indices) - 1 - k)
        if isinstance(v, int):
            if not 0 <= v < _RADIX:
                raise ValueError(f"channel {name}: index {v} not in [0, {_RADIX})")
            const += v * place
            continue
        term = v.to(core.int64, _semantic=_semantic).__mul__(place, _semantic=_semantic)
        off = term if off is None else off.__add__(term, _semantic=_semantic)
    if off is not None:
        ptr = ptr.__add__(off, _semantic=_semantic)
    if const:
        ptr = ptr.__add__(const, _semantic=_semantic)
    return ptr.__add__(core.arange(0, n, _semantic=_semantic), _semantic=_semantic)


def _dma(kind, chan_name, buf, offsets, sizes, strides, indices, _semantic, _generator):
    """A channel op on a buffer, as AIR's: `sizes` elements of the buffer
    `buf` points to, at `offsets` with `strides`. Recorded as a call of a
    function declared for it, whose attributes hold the static parts."""
    name = core._unwrap_if_constexpr(chan_name)
    if not _generator.module.has_function(f"air.channel.{name}"):
        raise ValueError(f"channel {name} is not declared")
    sizes, strides = _ints(sizes), _ints(
        strides if strides is not None else [1] * len(_ints(sizes))
    )
    offsets = [0] * len(sizes) if offsets is None else _values(offsets)
    indices = core._unwrap_if_constexpr(indices)
    indices = (
        list(indices) if isinstance(indices, (tuple, list, core.tuple)) else [indices]
    )
    if not (len(offsets) == len(sizes) == len(strides)):
        raise ValueError(f"{kind} {name}: offsets, sizes and strides of one rank")
    b = _semantic.builder
    i32 = lambda v: (  # noqa: E731
        _semantic.to_tensor(v) if isinstance(v, int) else v
    ).to(core.int32, _semantic=_semantic)
    operands = [buf] + [i32(v) for v in offsets] + [i32(v) for v in indices]
    module = _generator.module
    symbol = f"air.dma.{_next_id(_generator, '_air_dmas')}"
    decl = ir.builder(_generator.context).get_or_insert_function(
        module,
        symbol,
        b.get_function_ty([o.type.to_ir(b) for o in operands], []),
        "private",
        None,
    )
    module.push_back(decl)
    text = lambda v: b.get_string_attr(",".join(map(str, v)))  # noqa: E731
    decl.set_attr("air.dma_channel", b.get_string_attr(name))
    decl.set_attr("air.dma_put", b.get_bool_attr(kind == "ChannelPut"))
    decl.set_attr("air.dma_sizes", text(sizes))
    decl.set_attr("air.dma_strides", text(strides))
    decl.set_attr("air.dma_indices", b.get_int32_attr(len(indices)))
    b.call(module.get_function(symbol), [o.handle for o in operands])


@core.builtin
def ChannelPut(
    chan_name,
    src,
    indices=(),
    offsets=None,
    sizes=None,
    strides=None,
    _semantic=None,
    _generator=None,
):
    """Put into channel `chan_name` at `indices`, as AIR's `ChannelPut`.
    In a kernel, `src` is a 1-D block of the core's. With `sizes`, `src`
    points to a buffer, of DDR or L2, and the put sends `sizes` elements of
    it at `offsets` with `strides`: a transfer of the segment, not of a core.
    Waits while the FIFO is full."""
    if sizes is not None:
        return _dma(
            "ChannelPut",
            chan_name,
            src,
            offsets,
            sizes,
            strides,
            indices,
            _semantic,
            _generator,
        )
    addr = _fifo(chan_name, src.shape[0], src.dtype, indices, _semantic, _generator)
    core.store(addr, src, _semantic=_semantic)


@core.builtin
def ChannelGet(
    chan_name,
    dst,
    dtype=None,
    indices=(),
    offsets=None,
    sizes=None,
    strides=None,
    _semantic=None,
    _generator=None,
):
    """Get from channel `chan_name` at `indices`, as AIR's `ChannelGet`. In
    a kernel, `dst` is the shape of a 1-D block of `dtype`, which the get
    returns. With `sizes`, `dst` points to a buffer, of DDR or L2, that the
    get fills: `sizes` elements at `offsets` with `strides`. Waits for the
    data."""
    if sizes is not None:
        return _dma(
            "ChannelGet",
            chan_name,
            dst,
            offsets,
            sizes,
            strides,
            indices,
            _semantic,
            _generator,
        )
    n = _ints(dst)
    if len(n) != 1:
        raise ValueError("ChannelGet: a 1-D block")
    dtype = core._unwrap_if_constexpr(dtype)
    addr = _fifo(chan_name, n[0], dtype, indices, _semantic, _generator)
    return core.load(addr, volatile=True, _semantic=_semantic)
