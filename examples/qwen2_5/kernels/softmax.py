# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Softmax kernel for Qwen2.5 attention.
GPU: Online softmax with masking support for causal attention.
NPU: Block-based softmax adapted from test_softmax example.
"""

import os
import math
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# GPU kernel: row-wise softmax with optional causal masking
# ---------------------------------------------------------------------------
@triton.jit
def softmax_kernel_gpu(
    input_ptr,
    output_ptr,
    n_cols,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per row. BLOCK_SIZE >= n_cols."""
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load row
    row = tl.load(
        input_ptr + row_start + col_offsets, mask=mask, other=float("-inf")
    ).to(tl.float32)

    # Numerically stable softmax
    row_max = tl.max(row, axis=0)
    row = row - row_max
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    tl.store(output_ptr + row_start + col_offsets, result.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# NPU kernel: block-based softmax
# ---------------------------------------------------------------------------
@triton.jit
def softmax_kernel_npu(
    input_ptr,
    output_ptr,
    input_stride_row: tl.constexpr,
    input_stride_col: tl.constexpr,
    output_stride_row: tl.constexpr,
    output_stride_col: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_row = tl.program_id(0)
    offs_row = pid_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_col = tl.arange(0, n_cols)

    a_block = tl.load(
        input_ptr
        + offs_row[:, None] * input_stride_row
        + offs_col[None, :] * input_stride_col
    )
    row_minus_max = a_block - tl.max(a_block, axis=1, keep_dims=True)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1, keep_dims=True)
    softmax_output = numerator / denominator

    tl.store(
        output_ptr
        + offs_row[:, None] * output_stride_row
        + offs_col[None, :] * output_stride_col,
        softmax_output,
    )


# ---------------------------------------------------------------------------
# Wrapper: triton_softmax
# ---------------------------------------------------------------------------
from .backend_utils import CachedNPUKernel

_softmax_npu_cached = CachedNPUKernel()


def triton_softmax(x, causal_mask=None, backend="gpu", transform_script=None):
    """
    Row-wise softmax.

    Args:
        x: Input tensor of shape (*, n_cols). Softmax applied over last dim.
        causal_mask: Optional bool mask of shape broadcastable to x.
                     True = keep, False = mask to -inf. Applied before softmax.
        backend: "gpu" or "npu"
        transform_script: Path to transform script (NPU only)

    Returns:
        Softmax output of same shape as x.
    """
    if causal_mask is not None:
        x = x.masked_fill(~causal_mask, float("-inf"))

    orig_shape = x.shape
    n_cols = x.shape[-1]
    x_2d = x.reshape(-1, n_cols).contiguous()
    n_rows = x_2d.shape[0]

    if backend == "gpu":
        device = "cuda"
        if x_2d.device.type == "cuda":
            x_2d = x_2d.to(dtype=torch.bfloat16).contiguous()
        else:
            x_2d = x_2d.to(device=device, dtype=torch.bfloat16)
        output = torch.empty_like(x_2d)
        # BLOCK_SIZE must be power of 2 >= n_cols
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        softmax_kernel_gpu[(n_rows,)](
            x_2d,
            output,
            n_cols,
            x_2d.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # NPU path
        # The NPU softmax transform only supports grid=(1,1) (single block launch).
        # Process rows in chunks of BLOCK_SIZE, one kernel call per chunk.
        BLOCK_SIZE = 4

        # NPU requires power-of-2 arange sizes for n_cols.
        # Minimum 128: the transform script vectorizes at tile_sizes [0, 32],
        # so n_cols < 32 leaves math.exp as a scalar op that Peano can't legalize.
        # Additionally, DMA transfers require buffers >= 256 bytes (128 bf16 elements);
        # n_cols 32 and 64 compile but hang at runtime (ERT_CMD_STATE_TIMEOUT).
        MIN_NPU_COLS = 128
        n_cols_padded = 1 << (n_cols - 1).bit_length() if n_cols > 1 else 1
        n_cols_padded = max(n_cols_padded, MIN_NPU_COLS)

        # Pad rows to multiple of BLOCK_SIZE
        n_rows_padded = math.ceil(n_rows / BLOCK_SIZE) * BLOCK_SIZE
        pad_rows = n_rows_padded - n_rows
        pad_cols = n_cols_padded - n_cols
        if pad_rows > 0 or pad_cols > 0:
            x_2d = torch.nn.functional.pad(
                x_2d, (0, pad_cols, 0, pad_rows), value=float("-inf")
            )

        old_script = os.environ.get("AIR_TRANSFORM_TILING_SCRIPT")
        if transform_script:
            os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = transform_script

        # Process in chunks of BLOCK_SIZE rows (1 kernel call each)
        output_chunks = []
        for row_start in range(0, n_rows_padded, BLOCK_SIZE):
            chunk = (
                x_2d[row_start : row_start + BLOCK_SIZE].to(torch.bfloat16).contiguous()
            )
            out_chunk = torch.empty_like(chunk)
            _softmax_npu_cached(
                softmax_kernel_npu,
                (1, 1),
                chunk,
                out_chunk,
                chunk.stride(0),
                chunk.stride(1),
                out_chunk.stride(0),
                out_chunk.stride(1),
                n_cols_padded,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            output_chunks.append(out_chunk)
        output = torch.cat(output_chunks, dim=0)

        if old_script is not None:
            os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = old_script
        elif transform_script:
            del os.environ["AIR_TRANSFORM_TILING_SCRIPT"]

        # Slice back to original dims and renormalize
        output = output[:n_rows, :n_cols].to(torch.float32)
        output = output / output.sum(dim=-1, keepdim=True).clamp(min=1e-9)

    return output.reshape(orig_shape)
