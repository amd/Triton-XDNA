# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# this is a benchmark for adding vectors with maximum block size
# to check the performance of tl.dot operation

import torch
import triton
import triton.language as tl
import sys, os

sys.path.append(os.path.abspath(".."))
import benchmark


@triton.jit
def vecadd(
    A,
    B,
    C,
    n_elements: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)  # block row id
    block_start = pid * BLOCK_SIZE_N
    offsets = block_start + tl.arange(0, BLOCK_SIZE_N)

    # mask = offsets < n_elements    #AMK - in triton example, do we need?

    a_block = tl.load(A + offsets[:])
    b_block = tl.load(B + offsets[:])

    c_block = a_block + b_block

    tl.store(C + offsets[:], c_block)


# @benchmark.measure()
def bench_vecadd(N, provider):
    device = "cpu"
    dtype_in = torch.bfloat16
    dtype_out = (
        torch.bfloat16
    )  # torch.float32 won't work due to unsupported `%33 = fpext <8 x bfloat> %32 to <8 x float>`
    a = torch.randn(N, device=device, dtype=dtype_in)
    b = torch.randn(N, device=device, dtype=dtype_in)
    c = torch.empty(N, device=device, dtype=dtype_out)
    if provider == "torch" or provider == "test":
        c_ref = torch.add(a, b)
    if provider == "triton" or provider == "test":
        # 2D launch kernel where each block gets its own program.
        grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE_N"]),)
        compiled_kernel = vecadd[grid](
            a,
            b,
            c,
            N,
            BLOCK_SIZE_N=1024,  # TODO: small tile sizes currently face errors due to lock race condition at memtiles
        )
        with open("tt.shared.mlir", "w") as f:
            f.write(str(compiled_kernel.asm["ttsharedir"]))
        if provider == "test":
            torch.testing.assert_close(c, c_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    benchmark.select_npu_backend()
    for N in [2**i for i in range(10, 16, 1)]:
        bench_vecadd(N, "test")
