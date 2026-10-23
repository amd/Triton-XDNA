# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# A GEMM followed by an elementwise epilogue in the same kernel: the
# 1/sqrt(head_dim) scale applied to attention scores, a SiLU, and a scale with
# a bf16 store.
#
# The schedule (transform_aie2p.mlir) is generated with --epilogue. One
# schedule serves all three epilogues.

import torch
import triton
import triton.language as tl
import sys, os

sys.path.append(os.path.abspath(".."))
import benchmark

BLOCK_SIZE_M = 256
BLOCK_SIZE_N = 256

HEAD_DIM = 128
SCALE = HEAD_DIM**-0.5


@triton.jit
def matmul_epilogue(
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
    SCALE: tl.constexpr,
    EPILOGUE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, K)
    a = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b = tl.load(B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    c = tl.dot(a, b)

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    if EPILOGUE == "scale":
        tl.store(c_ptrs, c * SCALE)
    elif EPILOGUE == "silu":
        tl.store(c_ptrs, c / (1.0 + tl.exp(-c)))
    else:  # "scale_bf16"
        tl.store(c_ptrs, (c * SCALE).to(tl.bfloat16))


def test_matmul(M, N, K, epilogue):
    a = torch.randn((M, K), dtype=torch.bfloat16)
    b = torch.randn((K, N), dtype=torch.bfloat16)
    out_dtype = torch.bfloat16 if epilogue == "scale_bf16" else torch.float32
    c = torch.empty((M, N), dtype=out_dtype)
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    matmul_epilogue[grid](
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
        SCALE=SCALE,
        EPILOGUE=epilogue,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    ref = a.float() @ b.float()
    # The plain GEMM's tolerance, scaled along with the output.
    atol = 1e-1 * K**0.5
    if epilogue == "silu":
        ref = torch.nn.functional.silu(ref)
    else:
        ref = ref * SCALE
        atol *= SCALE
    torch.testing.assert_close(c.float(), ref, atol=atol, rtol=5e-2)
    print(f"M={M} N={N} K={K} {epilogue}: PASS")


if __name__ == "__main__":
    benchmark.select_npu_backend()
    torch.manual_seed(0)
    for epilogue in ("scale", "silu", "scale_bf16"):
        test_matmul(1024, 1024, HEAD_DIM, epilogue)
