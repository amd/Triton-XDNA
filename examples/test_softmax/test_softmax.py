# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import torch
import triton
import triton.language as tl
import sys, os

sys.path.append(os.path.abspath(".."))
import benchmark


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_stride_row: tl.constexpr,
    input_stride_col: tl.constexpr,
    output_stride_row: tl.constexpr,
    output_stride_col: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_row = tl.program_id(0)  # block row id

    offs_row = pid_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_col = tl.arange(0, n_cols)

    a_block = tl.load(
        input_ptr
        + offs_row[:, None] * input_stride_row
        + offs_col[None, :] * input_stride_col
    )
    # Subtract maximum for numerical stability
    row_minus_max = a_block - tl.max(a_block, axis=1, keep_dims=True)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1, keep_dims=True)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    tl.store(
        output_ptr
        + offs_row[:, None] * output_stride_row
        + offs_col[None, :] * output_stride_col,
        softmax_output,
    )


def softmax(x, y):
    n_rows, n_cols = x.shape
    # Enqueue kernel. The 1D launch grid is simple: we have one kernel instance per row o
    # f the input matrix
    grid = lambda META: (
        n_rows // 4,
        1,
    )
    softmax_kernel[grid](
        x,
        y,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        n_cols,
        BLOCK_SIZE=4,
    )
    return y


# @benchmark.measure()
def bench_softmax(size, provider):
    torch.manual_seed(0)
    dtype_in = torch.bfloat16
    x = torch.randn(size, size, device="cpu", dtype=dtype_in)
    # Allocate output
    y = torch.empty_like(x)
    if provider == "torch" or provider == "test":
        y_ref = torch.softmax(x, axis=1)
    if provider == "triton" or provider == "test":
        softmax(x, y)
        if provider == "test":
            torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    benchmark.select_npu_backend()
    bench_softmax(256, "test")
