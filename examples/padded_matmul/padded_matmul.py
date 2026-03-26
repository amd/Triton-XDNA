# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# F32 matmul with BF16 emulation for NPU2 (AIE2P/Strix).
# A is stored in K x M layout (transposed). Non-tile-aligned dimensions
# are handled via air-split-launch-for-padding.
#
# Target: NPU2/Strix only (ELF output format, bf16 emulation).
# Data types: F32 inputs/outputs, bf16 emulation on hardware
#   (hardware truncates f32 -> bf16 before multiply, f32 accumulation).
# Tile sizes: TILE_M=64, TILE_N=32, HERD=4x4, LAUNCH_TILE=256x128.

import math
import os
import sys

import torch
import triton
import triton.language as tl
import numpy as np
from ml_dtypes import bfloat16

sys.path.append(os.path.abspath(".."))
import benchmark

# === Tile parameters (must match transform_aie2p.mlir) ===
TILE_M = 64
TILE_N = 32
K_L2_TILE = 16
HERD_M = 4
HERD_N = 4
LAUNCH_TILE_M = TILE_M * HERD_M  # 256
LAUNCH_TILE_N = TILE_N * HERD_N  # 128
INNER_BLOCK = 8

# === Problem dimensions ===
# M and N can be non-tile-aligned; padding is handled by air-split-launch-for-padding.
# K must be a power of 2 (Triton requires tl.arange sizes to be powers of 2)
# and a multiple of K_L2_TILE.
M_actual = 500
N_actual = 500
K_val = 1024

assert K_val % K_L2_TILE == 0, (
    f"K={K_val} must be divisible by K_L2_TILE={K_L2_TILE}"
)

# === Padded/allocated dimensions ===
M_padded = math.ceil(M_actual / LAUNCH_TILE_M) * LAUNCH_TILE_M  # 512
N_padded = math.ceil(N_actual / LAUNCH_TILE_N) * LAUNCH_TILE_N  # 512
M_alloc = math.ceil(M_actual / INNER_BLOCK) * INNER_BLOCK  # 504
N_alloc = math.ceil(N_actual / INNER_BLOCK) * INNER_BLOCK  # 504


@triton.jit
def padded_matmul_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_block = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_block = tl.load(B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    c_block = tl.dot(a_block, b_block)

    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c_block)


def run_padded_matmul():
    np.random.seed(42)

    # Host data: A is K x M_alloc (transposed, block-aligned).
    # B is K x N_alloc. Zero-padded beyond M_actual/N_actual.
    A_np = np.zeros((K_val, M_alloc), dtype=np.float32)
    A_np[:, :M_actual] = (np.random.rand(K_val, M_actual) * 4).astype(np.float32)
    B_np = np.zeros((K_val, N_alloc), dtype=np.float32)
    B_np[:, :N_actual] = (np.random.rand(K_val, N_actual) * 4).astype(np.float32)

    A = torch.from_numpy(A_np)
    B = torch.from_numpy(B_np)
    C = torch.zeros((M_padded, N_padded), dtype=torch.float32)

    # Enable BF16 emulation for aircc
    os.environ["AMD_TRITON_NPU_BF16_EMULATION"] = "1"

    grid = (
        triton.cdiv(M_actual, LAUNCH_TILE_M),
        triton.cdiv(N_actual, LAUNCH_TILE_N),
    )

    compiled_kernel = padded_matmul_kernel[grid](
        A,
        B,
        C,
        M_actual,
        N_actual,
        K_val,
        1,  # stride_am = 1 (A transposed: stored K x M)
        M_alloc,  # stride_ak = M_alloc
        N_alloc,  # stride_bk = N_alloc
        1,  # stride_bn = 1
        N_padded,  # stride_cm = N_padded
        1,  # stride_cn = 1
        BLOCK_SIZE_M=LAUNCH_TILE_M,  # 256
        BLOCK_SIZE_N=LAUNCH_TILE_N,  # 128
        BLOCK_SIZE_K=K_val,  # full K
    )

    # Dump intermediate IR for debugging
    with open("tt.shared.mlir", "w") as f:
        f.write(str(compiled_kernel.asm["ttsharedir"]))

    # Validate with stochastic sampling.
    # Golden: truncate f32 inputs to bf16 (matching hardware bf16_emulation
    # truncf_op), then compute dot product with f32 accumulation.
    A_bf16 = A_np.astype(bfloat16)
    B_bf16 = B_np.astype(bfloat16)

    num_samples = 100
    sample_m = np.random.randint(0, M_actual, num_samples)
    sample_n = np.random.randint(0, N_actual, num_samples)

    # Add deterministic boundary-tile samples to catch padding errors.
    boundary_m = list(
        set(
            [
                min(M_actual - 1, m)
                for m in [M_actual - 1, M_actual - TILE_M + 1, 0]
                if m >= 0
            ]
        )
    )
    boundary_n = list(
        set(
            [
                min(N_actual - 1, n)
                for n in [N_actual - 1, N_actual - TILE_N + 1, 0]
                if n >= 0
            ]
        )
    )
    for bm in boundary_m:
        for bn in boundary_n:
            sample_m = np.append(sample_m, bm)
            sample_n = np.append(sample_n, bn)

    C_np = C.numpy()
    errors = 0
    for i in range(len(sample_m)):
        m, n = int(sample_m[i]), int(sample_n[i])
        expected = np.sum(
            A_bf16[:, m].astype(np.float32) * B_bf16[:, n].astype(np.float32),
            dtype=np.float32,
        )
        actual = C_np[m, n]
        if not np.isclose(actual, expected, rtol=0.1, atol=10.0):
            errors += 1
            if errors <= 5:
                print(
                    f"Mismatch at ({m}, {n}): actual={actual}, expected={expected}"
                )

    total = len(sample_m)
    if errors == 0:
        print(
            f"PASS: All {total} sampled elements match "
            f"(M={M_actual}, N={N_actual}, K={K_val})"
        )
    else:
        print(f"FAIL: {errors}/{total} samples mismatched")
        sys.exit(1)


if __name__ == "__main__":
    benchmark.select_npu_backend()
    run_padded_matmul()
