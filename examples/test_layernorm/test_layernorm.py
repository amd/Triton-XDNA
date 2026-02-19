# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import torch
import triton
import triton.language as tl
import sys, os

sys.path.append(os.path.abspath(".."))
import benchmark


@triton.jit
def _layer_norm_fwd_fused(
    input_ptr,  # pointer to the input
    output_ptr,  # pointer to the output
    input_stride_row: tl.constexpr,
    input_stride_col: tl.constexpr,
    output_stride_row: tl.constexpr,
    output_stride_col: tl.constexpr,
    n_cols: tl.constexpr,  # number of columns in X
    eps: tl.constexpr,  # epsilon to avoid division by zero
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

    sum_val = tl.sum(a_block, axis=1, keep_dims=True)
    sum_sq_val = tl.sum(a_block * a_block, axis=1, keep_dims=True)

    # Fixed normalization parameters (matching C++ kernel)
    eps_val = eps

    mean = sum_val / n_cols
    mean_sq = mean * mean
    variance = (sum_sq_val / n_cols) - mean_sq
    inv_std = 1.0 / tl.sqrt(variance + eps_val)

    gamma = 1.0
    beta = 0.0

    diff = a_block - mean
    norm = diff * inv_std
    scaled = norm * gamma
    out = scaled + beta
    # Write output
    tl.store(
        output_ptr
        + offs_row[:, None] * output_stride_row
        + offs_col[None, :] * output_stride_col,
        out,
    )


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps, device, tile_size=4):
        """
        Simple layer norm matching C++ kernel.

        Args:
            x: Input tensor
            eps: Epsilon for numerical stability
            device: Device to run on
            tile_size: Number of rows per dispatch tile (for hierarchical tiling)
        """
        # allocate output
        y = torch.empty_like(x)

        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape

        grid = lambda META: (
            M // 4,
            1,
        )
        _layer_norm_fwd_fused[grid](
            x_arg,
            y,
            x_arg.stride(0),
            x_arg.stride(1),
            y.stride(0),
            y.stride(1),
            N,
            eps,
            BLOCK_SIZE=4,
        )

        return y


# @benchmark.measure()
def bench_layernorm(size, provider):
    layer_norm = LayerNorm.apply
    device = "cpu"
    eps = 1e-5
    dtype = torch.float32
    x_shape = (size, size)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    x.requires_grad_(False)

    # forward pass
    y_tri = layer_norm(x, eps, device)


if __name__ == "__main__":
    benchmark.select_npu_backend()
    for X in [2**i for i in range(10, 13, 1)]:
        for provider in ["triton"]:
            bench_layernorm(X, provider)
