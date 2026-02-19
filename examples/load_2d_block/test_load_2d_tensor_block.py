# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import torch

import triton
import triton.language as tl
import sys, os

sys.path.append(os.path.abspath(".."))
import benchmark

"""

|-----|-----|-----|-----|
|     |     |     |     |
|-----|-----|-----|-----|
|     |     |     |     |
|-----|-----|-----|-----|

Each instance loads VEC_REG_SIZE_ROW * VEC_REG_SIZE_COL
"""


@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    stride_0: tl.constexpr,
    stride_1: tl.constexpr,
    VEC_REG_SIZE_ROW: tl.constexpr,
    VEC_REG_SIZE_COL: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    block_offset0 = pid0 * BLOCK_SIZE_ROW
    block_offset1 = pid1 * BLOCK_SIZE_COL

    input_x_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=[1, 1, n_rows, n_cols],
        strides=[VEC_REG_SIZE_ROW * stride_0, VEC_REG_SIZE_COL, stride_0, stride_1],
        offsets=[0, 0, block_offset0, block_offset1],
        block_shape=[
            BLOCK_SIZE_ROW // VEC_REG_SIZE_ROW,
            BLOCK_SIZE_COL // VEC_REG_SIZE_COL,
            VEC_REG_SIZE_ROW,
            VEC_REG_SIZE_COL,
        ],
        order=[3, 2, 1, 0],
    )
    x = tl.load(input_x_ptr)
    input_y_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=[1, 1, n_rows, n_cols],
        strides=[VEC_REG_SIZE_ROW * stride_0, VEC_REG_SIZE_COL, stride_0, stride_1],
        offsets=[0, 0, block_offset0, block_offset1],
        block_shape=[
            BLOCK_SIZE_ROW // VEC_REG_SIZE_ROW,
            BLOCK_SIZE_COL // VEC_REG_SIZE_COL,
            VEC_REG_SIZE_ROW,
            VEC_REG_SIZE_COL,
        ],
        order=[3, 2, 1, 0],
    )
    y = tl.load(input_y_ptr)
    z = x * y
    output_ptr = tl.make_block_ptr(
        base=z_ptr,
        shape=[1, 1, n_rows, n_cols],
        strides=[VEC_REG_SIZE_ROW * stride_0, VEC_REG_SIZE_COL, stride_0, stride_1],
        offsets=[0, 0, block_offset0, block_offset1],
        block_shape=[
            BLOCK_SIZE_ROW // VEC_REG_SIZE_ROW,
            BLOCK_SIZE_COL // VEC_REG_SIZE_COL,
            VEC_REG_SIZE_ROW,
            VEC_REG_SIZE_COL,
        ],
        order=[3, 2, 1, 0],
    )
    tl.store(output_ptr, z)


def test():
    device = "cpu"
    n_rows = 128
    n_cols = 64
    x = torch.arange(0, n_rows * n_cols, 1, device=device, dtype=torch.float32).reshape(
        [n_rows, n_cols]
    )
    y = torch.arange(0, n_rows * n_cols, 1, device=device, dtype=torch.float32).reshape(
        [n_rows, n_cols]
    )
    output = torch.full([n_rows, n_cols], -1, device=device, dtype=x.dtype)
    VEC_REG_SIZE_ROW = 4
    VEC_REG_SIZE_COL = 2
    BLOCK_SIZE_ROW = 32
    BLOCK_SIZE_COL = 32

    grid = lambda meta: (n_rows // BLOCK_SIZE_ROW, n_cols // BLOCK_SIZE_COL)

    kernel[grid](
        x,
        y,
        output,
        n_rows,
        n_cols,
        x.stride(0),
        x.stride(1),
        VEC_REG_SIZE_ROW=VEC_REG_SIZE_ROW,
        VEC_REG_SIZE_COL=VEC_REG_SIZE_COL,
        BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL,
    )
    expected = x * y

    torch.testing.assert_close(output, expected, rtol=0.001, atol=1e-5)


if __name__ == "__main__":
    benchmark.select_npu_backend()
    test()
