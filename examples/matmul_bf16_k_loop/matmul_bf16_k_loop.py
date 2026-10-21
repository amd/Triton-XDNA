# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# A GEMM written the way Triton's matmul tutorial writes it: each program
# loops over K in BLOCK_SIZE_K steps, advancing its pointers and accumulating
# with tl.dot(a, b, acc). K only has to be a multiple of BLOCK_SIZE_K, so it
# need not be a power of two, and it may be a single step.
#
# The schedule (transform_aie2p.mlir) is generated with --k-loop. It stages one
# BLOCK_SIZE_K slice in L2 per step, so its l2_k must equal BLOCK_SIZE_K.

import torch
import triton
import triton.language as tl
import sys, os

sys.path.append(os.path.abspath(".."))
import benchmark

BLOCK_SIZE_M = 256
BLOCK_SIZE_N = 256
BLOCK_SIZE_K = 64


@triton.jit
def matmul_k_loop(
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
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    tl.static_assert(K % BLOCK_SIZE_K == 0)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, K // BLOCK_SIZE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc)


def test_matmul(M, N, K):
    a = torch.randn((M, K), dtype=torch.bfloat16)
    b = torch.randn((K, N), dtype=torch.bfloat16)
    c = torch.empty((M, N), dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    matmul_k_loop[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    ref = a.float() @ b.float()
    torch.testing.assert_close(c, ref, atol=1e-1 * K**0.5, rtol=5e-2)
    print(f"M={M} N={N} K={K}: PASS")


if __name__ == "__main__":
    benchmark.select_npu_backend()
    torch.manual_seed(0)
    for K in (64, 256, 1024, 1536):
        test_matmul(512, 512, K)
