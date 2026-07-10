# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
RMSNorm kernel for Qwen2.5 with learnable weight (gamma).

RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

Unlike LayerNorm there is no mean subtraction and no bias: a single
reduction (sum of squares) drives the normalization. Both GPU and NPU
kernels compute the normalization in float32.

NPU kernel computes the bare normalization (x / sqrt(mean(x^2)+eps)) and the
learnable weight is applied in the Python wrapper, mirroring the layernorm
design that keeps extra DMA channels off the AIE shim tile.
"""

import os
import math
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# GPU kernel: fused RMSNorm with weight
# ---------------------------------------------------------------------------
@triton.jit
def rmsnorm_kernel_gpu(
    X,
    Y,
    Gamma,
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

    x = tl.load(x_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / n_cols
    inv_rms = 1.0 / tl.sqrt(mean_sq + eps)
    x_norm = x * inv_rms

    gamma = tl.load(Gamma + col_offsets, mask=mask, other=0.0).to(tl.float32)
    y = x_norm * gamma

    tl.store(y_ptr + col_offsets, y.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# NPU kernel: bare RMSNorm (no weight); weight applied in wrapper
# ---------------------------------------------------------------------------
@triton.jit
def rmsnorm_kernel_npu(
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

    sum_sq_val = tl.sum(a_block * a_block, axis=1, keep_dims=True)
    mean_sq = sum_sq_val / n_cols_real
    inv_rms = 1.0 / tl.sqrt(mean_sq + eps)

    out = a_block * inv_rms

    tl.store(
        Y
        + offs_row[:, None] * output_stride_row
        + offs_col[None, :] * output_stride_col,
        out,
    )


# ---------------------------------------------------------------------------
# Wrapper: triton_rmsnorm
# ---------------------------------------------------------------------------
from .backend_utils import CachedNPUKernel

_rmsnorm_npu_cached = CachedNPUKernel()


def triton_rmsnorm(x, gamma, eps=1e-6, backend="gpu", transform_script=None):
    """
    Root-mean-square layer normalization over the last dimension.

    Args:
        x: Input tensor of shape (*, n_cols)
        gamma: Scale parameter of shape (n_cols,)
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
        output = torch.empty_like(x_dev)

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        rmsnorm_kernel_gpu[(n_rows,)](
            x_dev,
            output,
            gamma_dev,
            n_cols,
            eps,
            x_dev.stride(0),
            output.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # NPU path: bare rmsnorm (no weight) + CPU weight multiply.
        BLOCK_SIZE = 4  # rows per program

        n_cols_padded = 1 << (n_cols - 1).bit_length() if n_cols > 1 else 1
        n_rows_padded = math.ceil(n_rows / BLOCK_SIZE) * BLOCK_SIZE

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
        _rmsnorm_npu_cached(
            rmsnorm_kernel_npu,
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

        # Slice back and apply learnable weight: y = norm * gamma
        output = output[:n_rows, :n_cols]
        gamma_f32 = gamma.to(torch.float32)
        output = output * gamma_f32

    return output.reshape(orig_shape)
