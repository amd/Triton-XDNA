# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# RMS Normalization kernel for AMD XDNA NPU
# Computes: y = x * rsqrt(mean(x^2) + eps) per row
#
# Processes BLOCK_M rows per invocation (2D tiling) so the Linalg IR has
# a row dimension that can be tiled at [1] (like softmax). This avoids
# the scalar chain issue where tl.sum produces a scalar that can't be
# fused into a forall.

import torch
import triton
import triton.language as tl
import sys, os

sys.path.append(os.path.abspath(".."))
import benchmark

EPS = 1e-5


@triton.jit
def rms_norm_kernel(
    X,
    Y,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    # Load BLOCK_M rows at once (2D block)
    offsets = rows[:, None] * N + cols[None, :]
    x = tl.load(X + offsets)

    # Compute mean of squares per row in bf16 (AIE2P only supports bf16 vector add)
    x_f32 = x.to(tl.float32)
    x_sq = x_f32 * x_f32
    x_sq_bf16 = x_sq.to(x.dtype)  # truncate to bf16 for AIE2P vector support
    sum_sq_bf16 = tl.sum(x_sq_bf16, axis=1)  # bf16 sum
    sum_sq = sum_sq_bf16.to(tl.float32)  # convert back to f32 for divf/rsqrt

    # Compute rsqrt per row (element-wise on 1D tensor, NO scalar chain)
    mean_sq = sum_sq / N
    rstd = tl.math.rsqrt(mean_sq + eps)  # shape: [BLOCK_M]

    # Normalize: broadcast rstd from [BLOCK_M] to [BLOCK_M, BLOCK_N]
    y = x_f32 * rstd[:, None]
    y = y.to(x.dtype)
    tl.store(Y + offsets, y)


def bench_rms_norm(M, N, provider):
    device = "cpu"
    dtype = torch.bfloat16
    BLOCK_M = 2  # Process 2 rows per invocation
    x = torch.randn(M, N, device=device, dtype=dtype)
    y = torch.empty(M, N, device=device, dtype=dtype)
    if provider == "torch" or provider == "test":
        x_f32 = x.float()
        mean_sq = (x_f32 * x_f32).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(mean_sq + EPS)
        y_ref = (x_f32 * rstd).to(dtype)
    if provider == "triton" or provider == "test":
        grid = (M // BLOCK_M,)
        compiled_kernel = rms_norm_kernel[grid](
            x,
            y,
            N,
            EPS,
            BLOCK_M=BLOCK_M,
            BLOCK_N=N,
        )
        with open("tt.shared.mlir", "w") as f:
            f.write(str(compiled_kernel.asm["ttsharedir"]))
        if provider == "test":
            torch.testing.assert_close(y, y_ref, atol=5e-1, rtol=1e-1)


if __name__ == "__main__":
    benchmark.select_npu_backend()
    # N >= 256 required for proper 2D DMA patterns in aircc runtime sequence
    for M in [32, 64]:
        for N in [256]:
            bench_rms_norm(M, N, "test")
