# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# AXPY benchmark: out = alpha * x + y
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
        "alpha": 2.0,
        "atol": 1e-2,
        "rtol": 1e-2,
        "bf16_emulation": False,
    },
    "f32": {
        "torch_dtype": torch.float32,
        "is_float": True,
        "alpha": 2.0,
        "atol": 1e-1,
        "rtol": 5e-2,
        "bf16_emulation": True,
    },
    "i8": {
        "torch_dtype": torch.int8,
        "is_float": False,
        "alpha": 2,
        "atol": 0,
        "rtol": 0,
        "bf16_emulation": False,
    },
    "i16": {
        "torch_dtype": torch.int16,
        "is_float": False,
        "alpha": 2,
        "atol": 0,
        "rtol": 0,
        "bf16_emulation": False,
    },
}


@triton.jit
def axpy_kernel(
    X,
    Y,
    OUT,
    alpha: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    x = tl.load(X + offsets[:])
    y = tl.load(Y + offsets[:])
    out = alpha * x + y
    tl.store(OUT + offsets[:], out)


def bench_axpy(N, provider, cfg):
    device = "cpu"
    torch_dtype = cfg["torch_dtype"]
    alpha = cfg["alpha"]

    if cfg["is_float"]:
        x = torch.randn(N, device=device, dtype=torch_dtype)
        y = torch.randn(N, device=device, dtype=torch_dtype)
    else:
        iinfo = torch.iinfo(torch_dtype)
        # Keep values small enough that alpha*x+y doesn't overflow
        quarter_max = iinfo.max // 4
        x = torch.randint(0, quarter_max, (N,), device=device, dtype=torch_dtype)
        y = torch.randint(0, quarter_max, (N,), device=device, dtype=torch_dtype)

    out = torch.empty(N, device=device, dtype=torch_dtype)

    if provider == "torch" or provider == "test":
        out_ref = alpha * x + y
    if provider == "triton" or provider == "test":
        grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
        compiled_kernel = axpy_kernel[grid](
            x,
            y,
            out,
            alpha,
            N,
            BLOCK_SIZE=1024,
        )
        with open("tt.shared.mlir", "w") as f:
            f.write(str(compiled_kernel.asm["ttsharedir"]))
        if provider == "test":
            torch.testing.assert_close(out, out_ref, atol=cfg["atol"], rtol=cfg["rtol"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AXPY benchmark for AMD NPU")
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
        bench_axpy(N, "test", cfg)
