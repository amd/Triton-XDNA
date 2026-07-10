# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
LayerNorm kernel for GPT-2 with learnable gamma (weight) and beta (bias).
GPU: Fused layernorm with per-row mean/variance.
NPU: Block-based layernorm adapted from test_layernorm example.
"""

import os
import math
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# GPU kernel: fused layernorm with gamma/beta
# ---------------------------------------------------------------------------
@triton.jit
def layernorm_kernel_gpu(
    X,
    Y,
    Gamma,
    Beta,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    stride_x: tl.constexpr,
    stride_y: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per row."""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x_ptr = X + row_idx * stride_x
    y_ptr = Y + row_idx * stride_y

    # Load row
    x = tl.load(x_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute mean and variance
    mean = tl.sum(x, axis=0) / n_cols
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize
    x_norm = x_centered * inv_std

    # Scale and shift with learnable parameters
    gamma = tl.load(Gamma + col_offsets, mask=mask, other=1.0).to(tl.float32)
    beta = tl.load(Beta + col_offsets, mask=mask, other=0.0).to(tl.float32)
    y = x_norm * gamma + beta

    tl.store(y_ptr + col_offsets, y.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# NPU kernel: bare layernorm (no gamma/beta)
# ---------------------------------------------------------------------------
# Matches the test_layernorm / mlir-aie C++ kernel design: gamma=1, beta=0.
# The affine transform (y * gamma + beta) is applied in the Python wrapper
# to avoid routing pressure from extra DMA channels on the AIE shim tile.
@triton.jit
def layernorm_kernel_npu(
    X,
    Y,
    input_stride_row: tl.constexpr,
    input_stride_col: tl.constexpr,
    output_stride_row: tl.constexpr,
    output_stride_col: tl.constexpr,
    n_cols: tl.constexpr,
    n_cols_real: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """n_cols is the padded (power-of-2) size; n_cols_real is the true column count."""
    pid_row = tl.program_id(0)
    offs_row = pid_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_col = tl.arange(0, n_cols)

    a_block = tl.load(
        X + offs_row[:, None] * input_stride_row + offs_col[None, :] * input_stride_col
    )

    # Use n_cols_real for statistics so zero-padded columns don't affect mean/var
    sum_val = tl.sum(a_block, axis=1, keep_dims=True)
    sum_sq_val = tl.sum(a_block * a_block, axis=1, keep_dims=True)

    mean = sum_val / n_cols_real
    variance = (sum_sq_val / n_cols_real) - (mean * mean)
    inv_std = 1.0 / tl.sqrt(variance + eps)

    out = (a_block - mean) * inv_std

    tl.store(
        Y
        + offs_row[:, None] * output_stride_row
        + offs_col[None, :] * output_stride_col,
        out,
    )


# ---------------------------------------------------------------------------
# Wrapper: triton_layernorm
# ---------------------------------------------------------------------------
from .backend_utils import CachedNPUKernel

_layernorm_npu_cached = CachedNPUKernel()


def triton_layernorm(x, gamma, beta, eps=1e-5, backend="gpu", transform_script=None):
    """
    Layer normalization over the last dimension.

    Args:
        x: Input tensor of shape (*, n_cols)
        gamma: Scale parameter of shape (n_cols,)
        beta: Shift parameter of shape (n_cols,)
        eps: Epsilon for numerical stability
        backend: "gpu" or "npu"
        transform_script: Path to transform script (NPU only)

    Returns:
        Normalized tensor of same shape as x.
    """
    orig_shape = x.shape
    n_cols = x.shape[-1]
    x_2d = x.reshape(-1, n_cols).contiguous()
    n_rows = x_2d.shape[0]

    if backend == "gpu":
        device = "cuda"
        if x_2d.device.type == "cuda":
            x_dev = x_2d.to(dtype=torch.bfloat16).contiguous()
        else:
            x_dev = x_2d.to(device=device, dtype=torch.bfloat16)
        if gamma.device.type == "cuda":
            gamma_dev = gamma.to(dtype=torch.bfloat16).contiguous()
        else:
            gamma_dev = gamma.to(device=device, dtype=torch.bfloat16).contiguous()
        if beta.device.type == "cuda":
            beta_dev = beta.to(dtype=torch.bfloat16).contiguous()
        else:
            beta_dev = beta.to(device=device, dtype=torch.bfloat16).contiguous()
        output = torch.empty_like(x_dev)

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        layernorm_kernel_gpu[(n_rows,)](
            x_dev,
            output,
            gamma_dev,
            beta_dev,
            n_cols,
            eps,
            x_dev.stride(0),
            output.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # NPU path: bare layernorm (no gamma/beta) + CPU affine transform.
        # The NPU kernel computes (x - mean) / sqrt(var + eps) only.
        # gamma/beta are applied afterward to avoid AIE routing pressure
        # from extra DMA channels on the shim tile.
        BLOCK_SIZE = 4  # rows per program, matching test_layernorm pattern

        # NPU requires power-of-2 arange sizes for n_cols
        n_cols_padded = 1 << (n_cols - 1).bit_length() if n_cols > 1 else 1

        n_rows_padded = math.ceil(n_rows / BLOCK_SIZE) * BLOCK_SIZE

        # Pad x with zeros in col dim, pad rows
        pad_cols = n_cols_padded - n_cols
        pad_rows = n_rows_padded - n_rows
        if pad_cols > 0 or pad_rows > 0:
            x_2d = torch.nn.functional.pad(x_2d, (0, pad_cols, 0, pad_rows))

        x_npu = x_2d.to(torch.float32).contiguous()
        output = torch.empty((n_rows_padded, n_cols_padded), dtype=torch.float32)

        old_script = os.environ.get("AIR_TRANSFORM_TILING_SCRIPT")
        if transform_script:
            os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = transform_script

        grid = (n_rows_padded // BLOCK_SIZE, 1)
        _layernorm_npu_cached(
            layernorm_kernel_npu,
            grid,
            x_npu,
            output,
            x_npu.stride(0),
            x_npu.stride(1),
            output.stride(0),
            output.stride(1),
            n_cols_padded,
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        if old_script is not None:
            os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = old_script
        elif transform_script:
            del os.environ["AIR_TRANSFORM_TILING_SCRIPT"]

        # Slice back to original dims
        output = output[:n_rows, :n_cols]

        # Apply learnable affine transform: y = norm * gamma + beta
        gamma_f32 = gamma.to(torch.float32)
        beta_f32 = beta.to(torch.float32)
        output = output * gamma_f32 + beta_f32

    return output.reshape(orig_shape)
