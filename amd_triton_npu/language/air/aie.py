# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""AIE core code in Triton: ``tl.extra.air.aie``.

Each op is one aie_api call, and the backend lowers it to the IR that
Peano's clang emits for that call, so that a kernel written with these ops
compiles to the code of the same kernel written in C++ with aie_api.

Values are held as the cores hold them, in 8x8 blocks:

=====================================================  ======================
aie_api                                                Triton
=====================================================  ======================
``aie::vector<T, N>``, ``aie::accum<accfloat, N>``     ``[N // 8, 8]`` tensor
``aie::vector<uint8, 128>``                            ``[8, 8]`` int16
``aie::block_vector<bfp16ebs8, 64>``, a stream, a mask  a handle
=====================================================  ======================

A handle is an ``[8, 8]`` int32 tensor that only these ops read. A buffer
argument is a core's own buffer (``air.alloc(..., memory_space=air.L1)``) or
a cast of it; offsets are elements of the buffer's type.

Every op is an ``elementwise_inline_asm`` that is not pure, so that Triton
neither merges nor moves ops across the functions the backend outlines
(``function``). An op whose operands may all be splats takes one more
operand of its shape, which the lowering does not read: Triton would
otherwise turn it into a scalar op.
"""

import triton
import triton.language as tl

__all__ = [
    "abs",
    "add",
    "begin_function",
    "bfp16_ostream",
    "bfp16_stream",
    "bit_and",
    "bit_or",
    "cmr8",
    "concat8",
    "end_function",
    "exp2",
    "exp2_16",
    "exp2_64",
    "extract8",
    "from_vector",
    "function",
    "gt0",
    "interleave_zip",
    "inv",
    "le0",
    "load_v",
    "lt",
    "mac",
    "mac_8x8_8x8T",
    "mask_false",
    "mask_true",
    "max64",
    "max8",
    "msc",
    "mul",
    "mul_8x8_8x8T",
    "mul_ff",
    "mul_scalar_f16",
    "mul_scalar_f64",
    "pop",
    "push",
    "row_max",
    "rows8",
    "rows8a",
    "select",
    "select64",
    "set_rounding_conv_even",
    "store_v",
    "store_v8",
    "sub16",
    "sub64",
    "tanh",
    "to_bf16",
    "to_bf16_64",
    "to_bfp16",
    "unpack",
    "unroll",
    "vector_cast_bf16",
    "vector_cast_u16",
]


@triton.jit
def _int8(x, R: tl.constexpr = 8):
    return tl.zeros((R, 8), tl.int64) + x


@triton.jit
def _ptr8(p, R: tl.constexpr = 8):
    return tl.zeros((R, 8), tl.int64) + p.to(tl.int64, bitcast=True)


@triton.jit
def _shape(v):
    R: tl.constexpr = v.shape[0]
    return tl.reshape(tl.arange(0, 8 * R), (R, 8))


@triton.jit
def _blocks8():
    return tl.reshape(tl.arange(0, 64), (8, 8))


@triton.jit
def bfp16_stream(p, block, NBYTES: tl.constexpr):
    """``aie::block_vector_input_buffer_stream<bfp16ebs8, 64>(p + 72 *
    block)``: the 72-byte bfp16 blocks of the int8 buffer ``p``, ``NBYTES``
    long, read from block ``block`` on."""
    return tl.inline_asm_elementwise(
        "aie.bfp16_stream",
        "=r,r,r,r",
        [_ptr8(p), _int8(block), _int8(NBYTES)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def bfp16_ostream(p, block, NBYTES: tl.constexpr):
    """``aie::block_vector_output_buffer_stream<bfp16ebs8, 64>(p + 72 *
    block)``: bfp16 blocks written into the int8 buffer ``p``, ``NBYTES``
    long, from block ``block`` on."""
    return tl.inline_asm_elementwise(
        "aie.bfp16_ostream",
        "=r,r,r,r",
        [_ptr8(p), _int8(block), _int8(NBYTES)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def pop(s):
    """``s >> blk``: the stream's next block."""
    return tl.inline_asm_elementwise(
        "aie.pop", "=r,r", [s], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def push(s, blk):
    """``s << blk``."""
    return tl.inline_asm_elementwise(
        "aie.push", "=r,r,r", [s, blk], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def mac_8x8_8x8T(acc, a, b):
    """``::mac_8x8_8x8T_conf(a, b, acc, 0, 0, 0)``: ``acc`` ([8, 8] f32,
    lanes ``m * 8 + n``) plus a times b transposed, of the bfp16 blocks
    ``a`` (rows m) and ``b`` (rows n)."""
    return tl.inline_asm_elementwise(
        "aie.mac_8x8_8x8T",
        "=r,r,r,r",
        [acc, a, b],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def load_v(p, off, N: tl.constexpr = 64):
    """``aie::load_v<N>(p + off)``, as an ``[N // 8, 8]`` tensor of ``p``'s
    element type."""
    return tl.inline_asm_elementwise(
        "aie.load_v",
        "=r,r,r",
        [_ptr8(p, N // 8), _int8(off, N // 8)],
        dtype=p.dtype.element_ty,
        is_pure=False,
        pack=1,
    )


@triton.jit
def store_v(p, off, v):
    """``aie::store_v(p + off, v)``."""
    R: tl.constexpr = v.shape[0]
    tl.inline_asm_elementwise(
        "aie.store_v",
        "=r,r,r,r",
        [_ptr8(p, R), _int8(off, R), v],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def to_bf16(v):
    """``acc.to_vector<bfloat16>()`` of an accumulator ``v`` (f32), rounded
    as ``set_rounding_conv_even`` sets."""
    return tl.inline_asm_elementwise(
        "aie.to_bf16",
        "=r,r,r",
        [v, _shape(v)],
        dtype=tl.bfloat16,
        is_pure=False,
        pack=1,
    )


@triton.jit
def to_bfp16(acc):
    """``acc.to_vector<bfp16ebs8>()`` of an [8, 8] f32 accumulator: a block."""
    return tl.inline_asm_elementwise(
        "aie.to_bfp16",
        "=r,r,r",
        [acc, _shape(acc)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def from_vector(v):
    """``acc.from_vector(v)`` of bf16 lanes: an f32 accumulator."""
    return tl.inline_asm_elementwise(
        "aie.from_vector",
        "=r,r,r",
        [v, _shape(v)],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def rows8(v):
    """8 bf16 lanes ``v`` ([1, 8]) concatenated to 64 and transposed 8 x 8:
    an [8, 8] tensor whose row r is all ``v[r]``."""
    return tl.inline_asm_elementwise(
        "aie.rows8", "=r,r,r", [v, _blocks8()], dtype=tl.bfloat16, is_pure=False, pack=1
    )


@triton.jit
def mac(acc, a, b):
    """``aie::mac(acc, a, b)`` of bf16 ``a`` and ``b``."""
    return tl.inline_asm_elementwise(
        "aie.mac",
        "=r,r,r,r,r",
        [acc, a, b, _shape(acc)],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def msc(acc, a, b):
    """``aie::msc(acc, a, b)`` of bf16 ``a`` and ``b``."""
    return tl.inline_asm_elementwise(
        "aie.msc",
        "=r,r,r,r,r",
        [acc, a, b, _shape(acc)],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def mul(a, b):
    """``aie::mul(a, b)`` of bf16 ``a`` and ``b``: an f32 accumulator."""
    return tl.inline_asm_elementwise(
        "aie.mul",
        "=r,r,r,r",
        [a, b, _shape(a)],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def add(a, b):
    """``aie::add(a, b)`` of 32 bf16 lanes."""
    return tl.inline_asm_elementwise(
        "aie.add",
        "=r,r,r,r",
        [a, b, _shape(a)],
        dtype=tl.bfloat16,
        is_pure=False,
        pack=1,
    )


@triton.jit
def abs(v):
    """``aie::abs(v)`` of bf16 lanes."""
    return tl.inline_asm_elementwise(
        "aie.abs", "=r,r,r", [v, _shape(v)], dtype=tl.bfloat16, is_pure=False, pack=1
    )


@triton.jit
def tanh(x):
    """``aie::tanh<bfloat16>(x)`` of f32 lanes."""
    return tl.inline_asm_elementwise(
        "aie.tanh", "=r,r,r", [x, _shape(x)], dtype=tl.bfloat16, is_pure=False, pack=1
    )


@triton.jit
def exp2(x):
    """``aie::exp2<bfloat16>(x)`` of f32 lanes."""
    return tl.inline_asm_elementwise(
        "aie.exp2", "=r,r,r", [x, _shape(x)], dtype=tl.bfloat16, is_pure=False, pack=1
    )


@triton.jit
def bit_and(a, b):
    """``aie::bit_and(a, b)`` of int16 lanes."""
    return tl.inline_asm_elementwise(
        "aie.bit_and",
        "=r,r,r,r",
        [a, b, _shape(a)],
        dtype=tl.int16,
        is_pure=False,
        pack=1,
    )


@triton.jit
def bit_or(a, b):
    """``aie::bit_or(a, b)`` of int16 lanes."""
    return tl.inline_asm_elementwise(
        "aie.bit_or",
        "=r,r,r,r",
        [a, b, _shape(a)],
        dtype=tl.int16,
        is_pure=False,
        pack=1,
    )


@triton.jit
def lt(a, b):
    """``aie::lt(a, b)`` of 32 bf16 lanes: a mask."""
    return tl.inline_asm_elementwise(
        "aie.lt", "=r,r,r,r", [a, b, _shape(a)], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def select(a, b, m):
    """``aie::select(a, b, m)`` of 32 bf16 lanes: ``b`` where the mask ``m``
    is set, ``a`` elsewhere."""
    return tl.inline_asm_elementwise(
        "aie.select",
        "=r,r,r,r,r",
        [a, b, m, _shape(a)],
        dtype=tl.bfloat16,
        is_pure=False,
        pack=1,
    )


@triton.jit
def unpack(v):
    """``aie::unpack`` of 128 uint4 (an [8, 8] int8 tensor, low nibble
    first): 128 uint8, two to an int16 lane."""
    return tl.inline_asm_elementwise(
        "aie.unpack", "=r,r,r", [v, _shape(v)], dtype=tl.int16, is_pure=False, pack=1
    )


@triton.jit
def interleave_zip(a, b):
    """``aie::interleave_zip(a, b, 1)`` of 128-lane uint8 vectors (as
    ``unpack`` gives them): the pair of vectors."""
    f = tl.inline_asm_elementwise(
        "aie.zip_first",
        "=r,r,r,r",
        [a, b, _shape(a)],
        dtype=tl.int16,
        is_pure=False,
        pack=1,
    )
    s = tl.inline_asm_elementwise(
        "aie.zip_second",
        "=r,r,r,r",
        [a, b, _shape(a)],
        dtype=tl.int16,
        is_pure=False,
        pack=1,
    )
    return f, s


@triton.jit
def vector_cast_bf16(v):
    """``aie::vector_cast<bfloat16>(v)`` of int16 lanes."""
    return tl.inline_asm_elementwise(
        "aie.vector_cast",
        "=r,r,r",
        [v, _shape(v)],
        dtype=tl.bfloat16,
        is_pure=False,
        pack=1,
    )


@triton.jit
def vector_cast_u16(v):
    """``v.cast_to<uint16>()`` of bf16 lanes, as int16."""
    return tl.inline_asm_elementwise(
        "aie.vector_cast",
        "=r,r,r",
        [v, _shape(v)],
        dtype=tl.int16,
        is_pure=False,
        pack=1,
    )


@triton.jit
def set_rounding_conv_even():
    """``aie::set_rounding(aie::rounding_mode::conv_even)``."""
    tl.inline_asm_elementwise(
        "aie.set_rounding", "=r,r", [_blocks8()], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def unroll(N: tl.constexpr):
    """``#pragma clang loop unroll_count(N)`` on the innermost loop this is
    in, N 2 or 4; 1 is ``unroll(disable)``."""
    if N == 1:
        tl.inline_asm_elementwise(
            "aie.unroll:1", "=r,r", [_blocks8()], dtype=tl.int32, is_pure=False, pack=1
        )
    elif N == 2:
        tl.inline_asm_elementwise(
            "aie.unroll:2", "=r,r", [_blocks8()], dtype=tl.int32, is_pure=False, pack=1
        )
    else:
        tl.static_assert(N == 4, "unroll: 1, 2 or 4")
        tl.inline_asm_elementwise(
            "aie.unroll:4", "=r,r", [_blocks8()], dtype=tl.int32, is_pure=False, pack=1
        )


@triton.jit
def function(NAME: tl.constexpr, RESTRICT: tl.constexpr = False):
    """The innermost loop this is in is the C function ``NAME``:
    the backend outlines it, one function for all its calls, never inlined.
    With ``RESTRICT``, its buffer arguments are ``__restrict``."""
    if RESTRICT:
        tl.inline_asm_elementwise(
            "aie.function:" + NAME + ":restrict",
            "=r,r",
            [_blocks8()],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    else:
        tl.inline_asm_elementwise(
            "aie.function:" + NAME,
            "=r,r",
            [_blocks8()],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )


@triton.jit
def begin_function(NAME: tl.constexpr, RESTRICT: tl.constexpr = False):
    """The code from here to ``end_function()``, in one block, is the C
    function ``NAME`` (as ``function``)."""
    if RESTRICT:
        tl.inline_asm_elementwise(
            "aie.begin_function:" + NAME + ":restrict",
            "=r,r",
            [_blocks8()],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    else:
        tl.inline_asm_elementwise(
            "aie.begin_function:" + NAME,
            "=r,r",
            [_blocks8()],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )


@triton.jit
def end_function():
    """The end of the code ``begin_function`` starts."""
    tl.inline_asm_elementwise(
        "aie.end_function", "=r,r", [_blocks8()], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def mul_8x8_8x8T(a, b):
    """``::mul_8x8_8x8T(a, b)``: a times b transposed, of bfp16 blocks, as
    an [8, 8] f32 accumulator."""
    return tl.inline_asm_elementwise(
        "aie.mul_8x8_8x8T", "=r,r,r", [a, b], dtype=tl.float32, is_pure=False, pack=1
    )


@triton.jit
def store_v8(p, off, v):
    """``aie::store_v(p + off, v)`` of an 8-lane vector (``v``'s first 8
    lanes)."""
    R: tl.constexpr = v.shape[0]
    tl.inline_asm_elementwise(
        "aie.store_v:8",
        "=r,r,r,r",
        [_ptr8(p, R), _int8(off, R), v],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def mask_true():
    """``aie::mask<64>(true)``."""
    return tl.inline_asm_elementwise(
        "aie.mask_all:1", "=r,r", [_blocks8()], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def mask_false():
    """``aie::mask<64>(false)``."""
    return tl.inline_asm_elementwise(
        "aie.mask_all:0", "=r,r", [_blocks8()], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def cmr8():
    """The table col - row of each lane of an 8x8 block (int16), a constant
    the core loads."""
    return tl.inline_asm_elementwise(
        "aie.cmr8", "=r,r", [_blocks8()], dtype=tl.int16, is_pure=False, pack=1
    )


# Template ops: lowered to the IR clang emits for their aie_api call
# (backend/aie_templates). "aie.t:<template>:<result lanes>:<operand lanes>",
# s a scalar operand (a tl.full), m a mask. A result narrower than its
# operands keeps their [R, 8] shape, its lanes first.


@triton.jit
def _rows(R: tl.constexpr):
    return tl.reshape(tl.arange(0, 8 * R), (R, 8))


@triton.jit
def row_max(v):
    """The max of each row of an 8x8 block of bf16: 8 lanes."""
    return tl.inline_asm_elementwise(
        "aie.t:op_row_max:8:64",
        "=r,r,r",
        [v, _shape(v)],
        dtype=tl.bfloat16,
        is_pure=False,
        pack=1,
    )


@triton.jit
def rows8a(v):
    """8 bf16 lanes as an 8x8 block, row r all ``v[r]`` (by
    ``aie::interleave_zip``)."""
    return tl.inline_asm_elementwise(
        "aie.t:op_rows8a:64:8",
        "=r,r,r",
        [v, _rows(8)],
        dtype=tl.bfloat16,
        is_pure=False,
        pack=1,
    )


@triton.jit
def max8(a, b):
    """``aie::max`` of 8 bf16 lanes."""
    return tl.inline_asm_elementwise(
        "aie.t:op_max8:8:8,8",
        "=r,r,r,r",
        [a, b, _shape(a)],
        dtype=tl.bfloat16,
        is_pure=False,
        pack=1,
    )


@triton.jit
def max64(a, b):
    """``aie::max`` of 64 bf16 lanes."""
    return tl.inline_asm_elementwise(
        "aie.t:op_max_bb:64:64,64",
        "=r,r,r,r",
        [a, b, _shape(a)],
        dtype=tl.bfloat16,
        is_pure=False,
        pack=1,
    )


@triton.jit
def concat8(a, b):
    """``aie::concat`` of two 8-lane bf16 vectors."""
    if (a.shape[0] == 1) & (b.shape[0] == 1):
        r = tl.inline_asm_elementwise(
            "aie.t:op_cat8:16:8,8",
            "=r,r,r,r",
            [a, b, _rows(2)],
            dtype=tl.bfloat16,
            is_pure=False,
            pack=1,
        )
    else:
        r = tl.inline_asm_elementwise(
            "aie.t:op_cat8:16:8,8",
            "=r,r,r,r",
            [a, b, _rows(8)],
            dtype=tl.bfloat16,
            is_pure=False,
            pack=1,
        )
    return r


@triton.jit
def sub16(a, b):
    """``aie::sub`` of 16 bf16 lanes."""
    return tl.inline_asm_elementwise(
        "aie.t:op_sub16:16:16,16",
        "=r,r,r,r",
        [a, b, _shape(a)],
        dtype=tl.bfloat16,
        is_pure=False,
        pack=1,
    )


@triton.jit
def sub64(a, b):
    """``aie::sub`` of 64 bf16 lanes."""
    return tl.inline_asm_elementwise(
        "aie.t:op_sub_bb:64:64,64",
        "=r,r,r,r",
        [a, b, _shape(a)],
        dtype=tl.bfloat16,
        is_pure=False,
        pack=1,
    )


@triton.jit
def mul_scalar_f16(a, s):
    """``aie::mul(a, s).to_vector<float>()`` of 16 bf16 lanes and a bf16
    scalar ``s`` (a ``tl.full``)."""
    return tl.inline_asm_elementwise(
        "aie.t:op_mulsc16:16:16,s",
        "=r,r,r,r",
        [a, s, _shape(a)],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def mul_scalar_f64(a, s):
    """``aie::mul(a, s).to_vector<float>()`` of 64 bf16 lanes."""
    return tl.inline_asm_elementwise(
        "aie.t:op_mulsc64:64:64,s",
        "=r,r,r,r",
        [a, s, _shape(a)],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def exp2_16(x):
    """``aie::exp2<bfloat16>`` of 16 f32 lanes."""
    return tl.inline_asm_elementwise(
        "aie.t:op_exp2_16:16:16",
        "=r,r,r",
        [x, _shape(x)],
        dtype=tl.bfloat16,
        is_pure=False,
        pack=1,
    )


@triton.jit
def exp2_64(x):
    """``aie::exp2<bfloat16>`` of 64 f32 lanes."""
    return tl.inline_asm_elementwise(
        "aie.t:op_exp2_64:64:64",
        "=r,r,r",
        [x, _shape(x)],
        dtype=tl.bfloat16,
        is_pure=False,
        pack=1,
    )


@triton.jit
def extract8(v):
    """``v.extract<8>(0)`` of 16 bf16 lanes."""
    return tl.inline_asm_elementwise(
        "aie.t:op_ext8:8:16",
        "=r,r,r",
        [v, _shape(v)],
        dtype=tl.bfloat16,
        is_pure=False,
        pack=1,
    )


@triton.jit
def mul_ff(a, b):
    """``aie::mul`` of 64 f32 lanes, ``to_vector<float>()``."""
    return tl.inline_asm_elementwise(
        "aie.t:op_mul_ff:64:64,64",
        "=r,r,r,r",
        [a, b, _shape(a)],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def inv(a):
    """``aie::inv`` of 64 f32 lanes."""
    return tl.inline_asm_elementwise(
        "aie.t:op_inv:64:64",
        "=r,r,r",
        [a, _shape(a)],
        dtype=tl.float32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def to_bf16_64(a):
    """``acc.to_vector<bfloat16>()`` of a 64-lane accumulator."""
    return tl.inline_asm_elementwise(
        "aie.t:op_srs64:64:64",
        "=r,r,r",
        [a, _shape(a)],
        dtype=tl.bfloat16,
        is_pure=False,
        pack=1,
    )


@triton.jit
def le0(d):
    """``aie::le(d, int16(0))`` of 64 int16 lanes: a mask."""
    return tl.inline_asm_elementwise(
        "aie.t:op_le0:m:64",
        "=r,r,r",
        [d, _shape(d)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def gt0(d):
    """``aie::gt(d, int16(0))`` of 64 int16 lanes: a mask."""
    return tl.inline_asm_elementwise(
        "aie.t:op_gt0:m:64",
        "=r,r,r",
        [d, _shape(d)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def select64(a, b, m):
    """``aie::select(a, b, m)`` of 64 bf16 lanes: ``b`` where the mask
    ``m`` is set."""
    return tl.inline_asm_elementwise(
        "aie.t:op_select64:64:64,64,m",
        "=r,r,r,r,r",
        [a, b, m, _shape(a)],
        dtype=tl.bfloat16,
        is_pure=False,
        pack=1,
    )
