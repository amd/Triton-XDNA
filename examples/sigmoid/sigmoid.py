# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Sigmoid benchmark: y = 1 / (1 + exp(-x))
# Supports bf16 (default) and f32 (via bf16-emulation).

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
        "atol": 1e-1,
        "rtol": 1e-1,
        "bf16_emulation": False,
    },
    "f32": {
        "torch_dtype": torch.float32,
        "atol": 2e-1,
        "rtol": 1e-1,
        "bf16_emulation": True,
    },
}


@triton.jit
def sigmoid_kernel(
    X,
    Y,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    x = tl.load(X + offsets[:])
    # sigmoid(x) = 1 / (1 + exp(-x))
    # Implemented manually to control the Linalg IR structure.
    # Use f32 for intermediate computation (Triton promotes bf16).
    x_f32 = x.to(tl.float32)
    neg_x = -x_f32
    exp_neg_x = tl.exp(neg_x)
    one_plus_exp = 1.0 + exp_neg_x
    y_f32 = 1.0 / one_plus_exp
    y = y_f32.to(x.dtype)
    tl.store(Y + offsets[:], y)


def bench_sigmoid(N, provider, cfg):
    device = "cpu"
    torch_dtype = cfg["torch_dtype"]
    x = torch.randn(N, device=device, dtype=torch_dtype)
    y = torch.empty(N, device=device, dtype=torch_dtype)
    if provider == "torch" or provider == "test":
        y_ref = torch.sigmoid(x)
    if provider == "triton" or provider == "test":
        grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
        compiled_kernel = sigmoid_kernel[grid](
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
    parser = argparse.ArgumentParser(description="Sigmoid benchmark for AMD NPU")
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
        bench_sigmoid(N, "test", cfg)
