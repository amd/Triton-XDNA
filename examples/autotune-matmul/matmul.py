# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# this is a benchmark which multiplies square matrices with maximum block size
# to check the performance of tl.dot operation

import math
import torch
import triton
import triton.language as tl
import sys, os

sys.path.append(os.path.abspath(".."))
import benchmark

configs = [
    triton.Config(kwargs={"BLOCK_SIZE_M": 256}),
    triton.Config(kwargs={"BLOCK_SIZE_M": 128}),
]


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
    # Scale inputs by 1/sqrt(K) so |c| stays O(1) independent of K, and take
    # the reference in f32 from the same bf16 inputs. Both follow mlir-air's
    # bf16_in_fp32_out GEMM, and together they make a tight tolerance
    # meaningful: an unscaled reference grows with K, and a bf16 reference
    # rounds the very result it is compared against.
    scale = 1.0 / math.sqrt(K)
    a = (torch.randn((M, K), device=device) * scale).to(dtype_in)
    b = (torch.randn((K, N), device=device) * scale).to(dtype_in)
    c = torch.empty((M, N), device=device, dtype=dtype_out)
    if provider == "torch" or provider == "test":
        c_ref = a.to(dtype_out) @ b.to(dtype_out)
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
            # Tolerance follows mlir-air's bf16_in_fp32_out tier A: rtol is
            # PyTorch's bf16 standard (1.6e-2), because the GEMM computes in
            # bf16 whatever the storage type. Their atol=1.5e-3 is not a
            # constant to copy: it was measured at K=8192, and with inputs
            # scaled by 1/sqrt(K) the error also falls as 1/sqrt(K) (verified
            # here at K=256/512/1024). Rescaled to K=256 their bound is
            # ~8.5e-3, so 5e-3 is the tighter of the two and still leaves
            # ~1.7x over the worst element observed on npu2.
            torch.testing.assert_close(c, c_ref, atol=5e-3, rtol=1.6e-2)


if __name__ == "__main__":
    # Fixed seed: the tolerance above is tight enough that unseeded inputs
    # would make a failure awkward to reproduce.
    torch.manual_seed(0)
    benchmark.select_npu_backend()
    bench_matmul(256, 256, 256, "test")
