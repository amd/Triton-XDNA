# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# this is a benchmark which multiplies square matrices with maximum block size
# to check the performance of tl.dot operation

import torch
import triton
import triton.language as tl
import sys, os

sys.path.append(os.path.abspath(".."))
import benchmark

configs = [triton.Config(kwargs={'BLOCK_SIZE_M': 256}), triton.Config(kwargs={'BLOCK_SIZE_M': 128})]

@triton.autotune(configs=configs, key=["M"])
@triton.jit
def bare_matmul(
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
    pid_m = tl.program_id(0)  # block row id
    pid_n = tl.program_id(1)  # block column id

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_block = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_block = tl.load(B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    c_block = tl.dot(a_block, b_block)

    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c_block)


# @benchmark.measure()
def bench_matmul(M, N, K, provider):
    device = "cpu"
    dtype_in = torch.bfloat16
    dtype_out = torch.float32
    a = torch.randn((M, K), device=device, dtype=dtype_in)
    b = torch.randn((K, N), device=device, dtype=dtype_in)
    c = torch.empty((M, N), device=device, dtype=dtype_out)
    if provider == "torch" or provider == "test":
        c_ref = torch.matmul(a, b).to(dtype_out)
    if provider == "triton" or provider == "test":
        # 2D launch kernel where each block gets its own program.
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )
        compiled_kernel = bare_matmul[grid](
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
            # BLOCK_SIZE_M=256,
            BLOCK_SIZE_N=256,
            BLOCK_SIZE_K=K,
        )
        with open("tt.shared.mlir", "w") as f:
            f.write(str(compiled_kernel.asm["ttsharedir"]))
        if provider == "test":
            torch.testing.assert_close(c, c_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    benchmark.select_npu_backend()
    bench_matmul(256,256,256, "test")
