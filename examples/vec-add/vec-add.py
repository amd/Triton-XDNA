# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Vector addition benchmark supporting multiple data types.
# Supports bf16 (default), f32 (via bf16-emulation), i8, and i16.

import argparse
import torch
import triton
import triton.language as tl
import sys
import os

sys.path.append(os.path.abspath(".."))
import benchmark

# Dtype configuration: torch type, whether it's a float, tolerances.
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
        "bf16_emulation": True,  # f32 addf not native on AIE; requires bf16-emulation
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

    a_block = tl.load(A + offsets[:])
    b_block = tl.load(B + offsets[:])

    c_block = a_block + b_block

    tl.store(C + offsets[:], c_block)


def bench_vecadd(N, provider, cfg):
    device = "cpu"
    torch_dtype = cfg["torch_dtype"]

    if cfg["is_float"]:
        a = torch.randn(N, device=device, dtype=torch_dtype)
        b = torch.randn(N, device=device, dtype=torch_dtype)
    else:
        # Clamp to half-max to avoid overflow on addition
        iinfo = torch.iinfo(torch_dtype)
        half_max = iinfo.max // 2
        a = torch.randint(0, half_max, (N,), device=device, dtype=torch_dtype)
        b = torch.randint(0, half_max, (N,), device=device, dtype=torch_dtype)

    c = torch.empty(N, device=device, dtype=torch_dtype)

    if provider == "torch" or provider == "test":
        c_ref = torch.add(a, b)
    if provider == "triton" or provider == "test":
        grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE_N"]),)
        compiled_kernel = vecadd[grid](
            a,
            b,
            c,
            N,
            BLOCK_SIZE_N=1024,
        )
        with open("tt.shared.mlir", "w") as f:
            f.write(str(compiled_kernel.asm["ttsharedir"]))
        if provider == "test":
            torch.testing.assert_close(c, c_ref, atol=cfg["atol"], rtol=cfg["rtol"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vector addition benchmark for AMD NPU"
    )
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

    # --bf16-emulation is shorthand for --dtype f32
    if args.bf16_emulation:
        args.dtype = "f32"

    cfg = DTYPE_CONFIG[args.dtype]

    # Enable bf16 emulation env var when needed
    if cfg["bf16_emulation"]:
        os.environ["AMD_TRITON_NPU_BF16_EMULATION"] = "1"

    benchmark.select_npu_backend()
    for N in [2**i for i in range(10, 16, 1)]:
        bench_vecadd(N, "test", cfg)
