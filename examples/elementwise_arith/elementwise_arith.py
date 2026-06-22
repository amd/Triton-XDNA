# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Elementwise arithmetic benchmark: sub, mul, div, square.
# Supports bf16 (default) and f32 (via bf16-emulation).
# Not all ops support all dtypes:
#   sub: bf16, f32
#   mul: bf16, f32
#   div: f32 only (hardware constraint: arith.divf is f32-only on AIE2P)
#   square: bf16, f32 (implemented as x * x)

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

# Which dtypes each op supports.
# Integer types (i16) fail at aircc for subi/muli on AIE2P (only addi works).
OP_DTYPES = {
    "sub": ["bf16", "f32"],
    "mul": ["bf16", "f32"],
    "div": ["f32"],  # arith.divf is f32-only on AIE2P; bf16 divf not supported
    "square": ["bf16", "f32"],
}


# --- Triton kernels ---


@triton.jit
def sub_kernel(X, Y, OUT, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offsets[:])
    y = tl.load(Y + offsets[:])
    tl.store(OUT + offsets[:], x - y)


@triton.jit
def mul_kernel(X, Y, OUT, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offsets[:])
    y = tl.load(Y + offsets[:])
    tl.store(OUT + offsets[:], x * y)


@triton.jit
def div_kernel(X, Y, OUT, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offsets[:])
    y = tl.load(Y + offsets[:])
    tl.store(OUT + offsets[:], x / y)


@triton.jit
def square_kernel(X, OUT, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offsets[:])
    tl.store(OUT + offsets[:], x * x)


# --- Kernel dispatch table ---

KERNELS = {
    "sub": sub_kernel,
    "mul": mul_kernel,
    "div": div_kernel,
    "square": square_kernel,
}

# --- Torch reference functions ---

TORCH_REF = {
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "div": lambda x, y: x / y,
    "square": lambda x, y: x * x,
}


def bench_op(op, N, provider, cfg):
    device = "cpu"
    torch_dtype = cfg["torch_dtype"]
    is_unary = op == "square"

    x = torch.randn(N, device=device, dtype=torch_dtype)
    if not is_unary:
        if op == "div":
            # Avoid division by zero; use values in [0.5, 1.5]
            y = 0.5 + torch.rand(N, device=device, dtype=torch_dtype)
        else:
            y = torch.randn(N, device=device, dtype=torch_dtype)

    out = torch.empty(N, device=device, dtype=torch_dtype)

    if provider == "torch" or provider == "test":
        out_ref = TORCH_REF[op](x, y if not is_unary else None)

    if provider == "triton" or provider == "test":
        grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
        kernel = KERNELS[op]
        if is_unary:
            compiled_kernel = kernel[grid](x, out, N, BLOCK_SIZE=1024)
        else:
            compiled_kernel = kernel[grid](x, y, out, N, BLOCK_SIZE=1024)
        with open("tt.shared.mlir", "w") as f:
            f.write(str(compiled_kernel.asm["ttsharedir"]))
        if provider == "test":
            torch.testing.assert_close(out, out_ref, atol=cfg["atol"], rtol=cfg["rtol"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Elementwise arithmetic benchmark for AMD NPU"
    )
    parser.add_argument(
        "--op",
        type=str,
        choices=list(KERNELS.keys()),
        required=True,
        help="Operation to benchmark",
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

    if args.bf16_emulation:
        args.dtype = "f32"

    # Validate op + dtype combination
    if args.dtype not in OP_DTYPES[args.op]:
        supported = ", ".join(OP_DTYPES[args.op])
        print(f"Error: --op {args.op} does not support --dtype {args.dtype}.")
        print(f"Supported dtypes for {args.op}: {supported}")
        sys.exit(1)

    cfg = DTYPE_CONFIG[args.dtype]

    if cfg["bf16_emulation"]:
        os.environ["AMD_TRITON_NPU_BF16_EMULATION"] = "1"

    # Select the right transform script based on op arity and NPU version.
    # If AIR_TRANSFORM_TILING_SCRIPT is already set, respect it.
    if not os.environ.get("AIR_TRANSFORM_TILING_SCRIPT"):
        from triton.backends.amd_triton_npu.driver import detect_npu_version

        is_unary = args.op == "square"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        arity = "unary" if is_unary else "binary"
        npu = detect_npu_version()
        suffix = "aie2" if npu == "npu1" else "aie2p"
        os.environ["AIR_TRANSFORM_TILING_SCRIPT"] = os.path.join(
            script_dir, f"transform_{arity}_{suffix}.mlir"
        )

    benchmark.select_npu_backend()
    for N in [2**i for i in range(10, 16, 1)]:
        bench_op(args.op, N, "test", cfg)
