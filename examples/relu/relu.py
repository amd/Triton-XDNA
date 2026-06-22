# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# ReLU benchmark: y = max(x, 0)
# Supports bf16 (default), f32 (via bf16-emulation), i8, and i16.

import argparse
import torch
import triton
import triton.language as tl
import sys
import os

sys.path.append(os.path.abspath(".."))
import benchmark

DTYPE_CONFIG = {
    "bf16": {
        "torch_dtype": torch.bfloat16,
        "is_float": True,
        "atol": 1e-2,
        "rtol": 1e-2,
        "bf16_emulation": False,
    },
    "f32": {
        "torch_dtype": torch.float32,
        "is_float": True,
        "atol": 1e-1,
        "rtol": 5e-2,
        "bf16_emulation": True,
    },
    "i8": {
        "torch_dtype": torch.int8,
        "is_float": False,
        "atol": 0,
        "rtol": 0,
        "bf16_emulation": False,
    },
    "i16": {
        "torch_dtype": torch.int16,
        "is_float": False,
        "atol": 0,
        "rtol": 0,
        "bf16_emulation": False,
    },
}


@triton.jit
def relu_kernel(
    X,
    Y,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    x = tl.load(X + offsets[:])
    # x * 0 produces a dtype-compatible zero for both float and int types.
    y = tl.maximum(x, x * 0)
    tl.store(Y + offsets[:], y)


def bench_relu(N, provider, cfg):
    device = "cpu"
    torch_dtype = cfg["torch_dtype"]

    if cfg["is_float"]:
        x = torch.randn(N, device=device, dtype=torch_dtype)
    else:
        iinfo = torch.iinfo(torch_dtype)
        x = torch.randint(iinfo.min, iinfo.max, (N,), device=device, dtype=torch_dtype)

    y = torch.empty(N, device=device, dtype=torch_dtype)

    if provider == "torch" or provider == "test":
        y_ref = torch.relu(x)
    if provider == "triton" or provider == "test":
        grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
        compiled_kernel = relu_kernel[grid](
            x,
            y,
            N,
            BLOCK_SIZE=1024,
        )
        with open("tt.shared.mlir", "w") as f:
            f.write(str(compiled_kernel.asm["ttsharedir"]))
        if provider == "test":
            torch.testing.assert_close(y, y_ref, atol=cfg["atol"], rtol=cfg["rtol"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReLU benchmark for AMD NPU")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=list(DTYPE_CONFIG.keys()),
        default="bf16",
        help="Element data type (default: bf16)",
    )
    parser.add_argument(
        "--bf16-emulation",
        dest="bf16_emulation",
        default=False,
        action="store_true",
        help="Use f32 data type with bf16 emulation on AIE cores",
    )
    args = parser.parse_args()

    if args.bf16_emulation:
        args.dtype = "f32"

    cfg = DTYPE_CONFIG[args.dtype]

    if cfg["bf16_emulation"]:
        os.environ["AMD_TRITON_NPU_BF16_EMULATION"] = "1"

    benchmark.select_npu_backend()
    for N in [2**i for i in range(10, 16, 1)]:
        bench_relu(N, "test", cfg)
