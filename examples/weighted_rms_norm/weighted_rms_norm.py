# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Weighted RMS Normalization kernel for AMD XDNA NPU
# Computes: y = x * rsqrt(mean(x^2) + eps) * w per row
#
# Extends rms_norm by multiplying each element by a learned weight vector w.
# Uses BLOCK_M=2 (2D tiling) to avoid the scalar chain issue.

import torch
import triton
import triton.language as tl
import sys, os

sys.path.append(os.path.abspath(".."))
import benchmark

EPS = 1e-5


@triton.jit
def weighted_rms_norm_kernel(
    X,
    W,
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

    # Load weight vector (same for all rows, 1D [BLOCK_N])
    w = tl.load(W + cols)

    # Compute mean of squares per row in bf16
    x_f32 = x.to(tl.float32)
    x_sq = x_f32 * x_f32
    x_sq_bf16 = x_sq.to(x.dtype)
    sum_sq_bf16 = tl.sum(x_sq_bf16, axis=1)
    sum_sq = sum_sq_bf16.to(tl.float32)

    # Compute rsqrt per row
    mean_sq = sum_sq / N
    rstd = tl.math.rsqrt(mean_sq + eps)

    # Normalize and multiply by weight: y = x * rstd * w
    w_f32 = w.to(tl.float32)
    y = x_f32 * rstd[:, None] * w_f32[None, :]
    y = y.to(x.dtype)
    tl.store(Y + offsets, y)


def bench_weighted_rms_norm(M, N, provider):
    device = "cpu"
    dtype = torch.bfloat16
    BLOCK_M = 2
    x = torch.randn(M, N, device=device, dtype=dtype)
    w = torch.randn(N, device=device, dtype=dtype)
    y = torch.empty(M, N, device=device, dtype=dtype)
    if provider == "torch" or provider == "test":
        x_f32 = x.float()
        w_f32 = w.float()
        mean_sq = (x_f32 * x_f32).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(mean_sq + EPS)
        y_ref = (x_f32 * rstd * w_f32.unsqueeze(0)).to(dtype)
    if provider == "triton" or provider == "test":
        grid = (M // BLOCK_M,)
        compiled_kernel = weighted_rms_norm_kernel[grid](
            x,
            w,
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
    for M in [32, 64]:
        for N in [256]:
            bench_weighted_rms_norm(M, N, "test")
