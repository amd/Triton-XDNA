# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import torch
import triton
import triton.language as tl
import sys, os

sys.path.append(os.path.abspath(".."))
import benchmark


@triton.jit
def erf_kernel(X, Y, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offsets[:])
    y = tl.erf(x.to(tl.float32))
    tl.store(Y + offsets[:], y.to(x.dtype))


@triton.jit
def gelu_erf_kernel(X, Y, BLOCK_SIZE: tl.constexpr):
    # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offsets[:])
    xf = x.to(tl.float32)
    y = 0.5 * xf * (1.0 + tl.erf(xf * 0.7071067811865476))
    tl.store(Y + offsets[:], y.to(x.dtype))


def reference(name, x):
    if name == "erf":
        return torch.erf(x.float()).to(x.dtype)
    return torch.nn.functional.gelu(x.float(), approximate="none").to(x.dtype)


def bench_erf(kernel, name, N, provider):
    x = torch.randn(N, dtype=torch.bfloat16)
    y = torch.empty(N, dtype=torch.bfloat16)
    if provider == "torch" or provider == "test":
        y_ref = reference(name, x)
    if provider == "triton" or provider == "test":
        grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
        kernel[grid](x, y, BLOCK_SIZE=1024)
        if provider == "test":
            # Sized to the error measured on npu2.
            torch.testing.assert_close(y, y_ref, atol=4e-2, rtol=1.6e-2)


if __name__ == "__main__":
    benchmark.select_npu_backend()
    for kernel, name in [(erf_kernel, "erf"), (gelu_erf_kernel, "gelu")]:
        for N in [2**i for i in range(10, 16, 1)]:
            bench_erf(kernel, name, N, "test")
