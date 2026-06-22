# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Leaky ReLU benchmark: y = x if x >= 0, else alpha * x
# Supports bf16 (default) and f32 (via bf16-emulation).

import argparse
import torch
import triton
import triton.language as tl
import sys
import os

sys.path.append(os.path.abspath(".."))
import benchmark

ALPHA = 0.01  # Standard leaky relu negative slope

DTYPE_CONFIG = {
    "bf16": {
        "torch_dtype": torch.bfloat16,
        "atol": 1e-2,
        "rtol": 1e-2,
        "bf16_emulation": False,
    },
    "f32": {
        "torch_dtype": torch.float32,
        "atol": 1e-1,
        "rtol": 5e-2,
        "bf16_emulation": True,
    },
}


@triton.jit
def leaky_relu_kernel(
    X,
    Y,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    x = tl.load(X + offsets[:])
    # Leaky ReLU: y = x if x >= 0, else alpha * x
    # tl.where produces arith.cmpf + arith.select in Linalg IR.
    # AIE2 supports vector select (vselect intrinsic).
    y = tl.where(x >= 0, x, 0.01 * x)
    tl.store(Y + offsets[:], y)


def bench_leaky_relu(N, provider, cfg):
    device = "cpu"
    torch_dtype = cfg["torch_dtype"]
    x = torch.randn(N, device=device, dtype=torch_dtype)
    y = torch.empty(N, device=device, dtype=torch_dtype)
    if provider == "torch" or provider == "test":
        y_ref = torch.nn.functional.leaky_relu(x, negative_slope=ALPHA)
    if provider == "triton" or provider == "test":
        grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
        compiled_kernel = leaky_relu_kernel[grid](
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
    parser = argparse.ArgumentParser(description="Leaky ReLU benchmark for AMD NPU")
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
        bench_leaky_relu(N, "test", cfg)
