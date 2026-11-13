# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Superkernels: one `@triton.jit` function that runs several kernels side by
side on the NPU's cores, wave after wave, as a single dispatch.

    @triton.jit
    def layer(X, W, NW: tl.constexpr, D: tl.constexpr, DP: tl.constexpr):
        h = air.allocate_l2([DP], tl.bfloat16)
        for wave in range(NW):
            air.spatial(
                [(scale, (X, W, h, wave, D, DP)), (add, (X, h, X, D, DP))],
                cores=[1, 1],
                schedules=[SCHEDULE, SCHEDULE],
            )

    layer[(1,)](x, w, NW=4, D=1536, DP=2048)

`spatial` runs each partition on its own cores, all at the same time. A
partition is a kernel and its arguments, or a `@triton.jit` function that calls
kernels in turn on the same cores; calls of one kernel that differ only in
their numbers and in the on-chip buffers they write become one loop over
phases. Partitions exchange data through L2 buffers from `allocate_l2`: a
buffer read in a wave must be written by some partition in that wave, which
the cores wait for. DDR buffers, the pointer arguments, are updated in place;
a partition sees the DDR writes of the partitions before it.

A wave may call `spatial` several times. The calls take effect in program
order, which is the order of the DDR writes the partitions see and of the
transfers the dispatch issues; L2 buffers synchronize by themselves.
Partitions of one name, within a call or across calls, run in turn on the
same cores.

`cores[i]` is partition i's core count, an int or (columns, rows). It drives
the partition's schedule: `schedules[i]` is the path of a transform script
whose `@__transform_main` takes, after the payload, one `!transform.param<i64>`
per core dimension, for use as `num_threads`. The partitions are placed on the
array in order unless `at[i]` gives the (column, row) of partition i's corner.

The loop over the waves must have a constexpr trip count. Ints computed from
the wave index, such as `wave * 2 + 1` or `air.table(values, wave)`, are
evaluated per wave when the dispatch is built; int arguments of the function
are set by the host at each launch (mark them `do_not_specialize` so that a
new value does not recompile the dispatch).
"""

from triton._C.libtriton import ir
from triton.language import core

__all__ = ["Channel", "ChannelGet", "ChannelPut", "allocate_l2", "spatial", "table"]


class _PartitionContext:
    """Compiles a partition's function apart from the caller's, under a name
    of its own, and marks it with the partition's settings. The functions it
    calls stay out of line as well, so that the backend finds the kernels."""

    def __init__(self, step, index, cores, schedule, at, name):
        self.step, self.index = step, index
        self.cores, self.schedule, self.at, self.name = cores, schedule, at, name
        self.marked = False

    def mangle(self):
        return f"_SK{self.step}P{self.index}"

    def initialize_callee(self, fn, builder):
        fn.set_attr("noinline", builder.get_bool_attr(True))
        if self.marked:
            return
        self.marked = True
        i32 = builder.get_int32_attr
        fn.set_attr("air.step", i32(self.step))
        fn.set_attr("air.partition", i32(self.index))
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


def _shape(cores):
    cores = core._unwrap_if_constexpr(cores)
    if isinstance(cores, int):
        return (cores, 1)
    cores = tuple(_values(cores))
    if len(cores) != 2 or not all(isinstance(c, int) and c > 0 for c in cores):
        raise ValueError(f"cores must be an int or (columns, rows), not {cores}")
    return cores


def _next_id(generator, counter):
    n = getattr(generator, counter, 0)
    setattr(generator, counter, n + 1)
    return n


@core.builtin
def spatial(
    partitions, cores, schedules, at=None, names=None, _semantic=None, _generator=None
):
    """Run `partitions`, a list of (function, arguments), side by side; see
    the module's documentation."""
    parts = list(core._unwrap_if_constexpr(partitions))
    cores = [_shape(c) for c in core._unwrap_if_constexpr(cores)]
    schedules = _values(schedules)
    at = [None] * len(parts) if at is None else _values(at)
    names = [None] * len(parts) if names is None else _values(names)
    for what, v in (
        ("cores", cores),
        ("schedules", schedules),
        ("at", at),
        ("names", names),
    ):
        if len(v) != len(parts):
            raise ValueError(f"spatial: {len(parts)} partitions but {len(v)} {what}")
    step = _next_id(_generator, "_npu_spatial_steps")
    for i, part in enumerate(parts):
        fn, args = _values(part)
        if not isinstance(schedules[i], str):
            raise ValueError(f"spatial: schedules[{i}] must be a path")
        corner = None if at[i] is None else tuple(_values(at[i]))
        name = names[i] or f"{fn.__name__}_{step}_{i}"
        ctx = _PartitionContext(step, i, cores[i], schedules[i], corner, name)
        _generator.call_JitFunction(fn, list(args), kwargs={}, caller_context=ctx)


@core.builtin
def allocate_l2(shape, dtype, _semantic=None, _generator=None):
    """A new buffer of `shape` elements of `dtype` in L2, as a pointer to its
    first element. Only partitions of a `spatial` may access it."""
    shape = core._unwrap_if_constexpr(shape)
    n = 1
    for d in [shape] if isinstance(shape, int) else _values(shape):
        n *= d
    dtype = core._unwrap_if_constexpr(dtype)
    b = _semantic.builder
    ptr = core.pointer_type(dtype)
    # A call of a function declared for this buffer alone: the backend finds
    # the buffer's size there, and no two buffers fold into one value.
    name = f"air.l2.{_next_id(_generator, '_npu_l2_buffers')}"
    module = _generator.module
    decl = ir.builder(_generator.context).get_or_insert_function(
        module, name, b.get_function_ty([], [ptr.to_ir(b)]), "private", None
    )
    module.push_back(decl)
    decl.set_attr("air.l2_elements", b.get_int32_attr(n))
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


@core.builtin
def ChannelPut(chan_name, src, indices=(), _semantic=None, _generator=None):
    """Put the 1-D block `src` into channel `chan_name` at `indices`, as
    AIR's `ChannelPut`. Waits while the FIFO is full."""
    addr = _fifo(chan_name, src.shape[0], src.dtype, indices, _semantic, _generator)
    core.store(addr, src, _semantic=_semantic)


@core.builtin
def ChannelGet(chan_name, shape, dtype, indices=(), _semantic=None, _generator=None):
    """The next block of `shape` (1-D) elements of `dtype` from channel
    `chan_name` at `indices`, as AIR's `ChannelGet`. Waits for it."""
    n = _ints(shape)
    if len(n) != 1:
        raise ValueError("ChannelGet: a 1-D block")
    dtype = core._unwrap_if_constexpr(dtype)
    addr = _fifo(chan_name, n[0], dtype, indices, _semantic, _generator)
    return core.load(addr, volatile=True, _semantic=_semantic)
