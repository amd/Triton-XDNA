# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Mean subtraction (centering) kernel for AMD XDNA NPU
# Computes: y[i,j] = x[i,j] - mean(x[i,:]) per row
#
# This is the 2D-output form of average pooling that matches the rms_norm
# reduction pattern. The 1D output form (just storing the mean) hits a
# 4-byte DMA alignment constraint on AIE (memref<1xbf16> = 2 bytes < 4).
# Broadcasting the mean back to [BLOCK_M, BLOCK_N] via subtraction avoids
# this constraint (output DMA is [1, 256] = 512 bytes per tile).
#
# Uses BLOCK_M=2 (2D tiling) to avoid the scalar chain issue.

import torch
import triton
import triton.language as tl
import sys, os

sys.path.append(os.path.abspath(".."))
import benchmark


@triton.jit
def avg_pool_kernel(
    X,
    Y,
    N: tl.constexpr,
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

    # Sum per row in bf16 (AIE2P only supports bf16 vector add)
    row_sum = tl.sum(x, axis=1)  # [BLOCK_M], bf16

    # Divide by N in f32 (divf is f32-only on AIE2P)
    mean = row_sum.to(tl.float32) / N

    # Subtract mean from input (2D output, broadcasts mean across columns)
    x_f32 = x.to(tl.float32)
    y = x_f32 - mean[:, None]
    y = y.to(x.dtype)

    tl.store(Y + offsets, y)


def bench_avg_pool(M, N, provider):
    device = "cpu"
    dtype = torch.bfloat16
    BLOCK_M = 2
    x = torch.randn(M, N, device=device, dtype=dtype)
    y = torch.empty(M, N, device=device, dtype=dtype)
    if provider == "torch" or provider == "test":
        x_f32 = x.float()
        mean = x_f32.mean(dim=-1, keepdim=True)
        y_ref = (x_f32 - mean).to(dtype)
    if provider == "triton" or provider == "test":
        grid = (M // BLOCK_M,)
        compiled_kernel = avg_pool_kernel[grid](
            x,
            y,
            N,
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
            bench_avg_pool(M, N, "test")
