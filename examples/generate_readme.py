#!/usr/bin/env python3

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Generates an operator dashboard for Triton-XDNA examples.
#
# Device support is determined by the presence of transform files:
#   transform_aie2.mlir  → AIE2 (NPU1) supported
#   transform_aie2p.mlir → AIE2P (NPU2) supported
#
# Usage:
#   python3 examples/generate_readme.py
#   python3 examples/generate_readme.py --output _site/index.md --base-url URL
#   python3 examples/generate_readme.py --verify

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# ── Example registry ──────────────────────────────────────────────────
# Category, human-readable name, directory path (relative to examples/),
# and datatypes are manually maintained here.  When adding a new example,
# add an entry to this list.

EXAMPLES = [
    {
        "category": "Matrix",
        "name": "Matrix Multiplication (BF16)",
        "path": "matmul_bf16_m64_n64_k64",
        "datatypes": "bf16",
    },
    {
        "category": "Matrix",
        "name": "Padded Matrix Multiplication (F32, A Transposed)",
        "path": "matmul_f32_m64_n32_k16_padded_atransposed",
        "datatypes": "f32 (bf16 emulation)",
    },
    {
        "category": "Matrix",
        "name": "Matrix Multiplication (INT8)",
        "path": "matmul_i8_m64_n64_k64",
        "datatypes": "i8",
    },
    {
        "category": "Matrix",
        "name": "Matrix Multiplication (INT8, Large Tile)",
        "path": "matmul_i8_m128_n64_k64",
        "datatypes": "i8",
    },
    {
        "category": "Matrix",
        "name": "Matrix Multiplication (Autotune)",
        "path": "autotune-matmul",
        "datatypes": "bf16",
    },
    {
        "category": "Element-wise",
        "name": "ReLU",
        "path": "relu",
        "datatypes": "bf16, f32, i8, i16",
    },
    {
        "category": "Element-wise",
        "name": "Sigmoid",
        "path": "sigmoid",
        "datatypes": "bf16",
    },
    {
        "category": "Element-wise",
        "name": "SiLU",
        "path": "silu",
        "datatypes": "bf16",
    },
    {
        "category": "Element-wise",
        "name": "GELU",
        "path": "gelu",
        "datatypes": "bf16",
    },
    {
        "category": "Element-wise",
        "name": "Leaky ReLU",
        "path": "leaky_relu",
        "datatypes": "bf16",
    },
    {
        "category": "Element-wise",
        "name": "SwiGLU",
        "path": "swiglu",
        "datatypes": "bf16",
    },
    {
        "category": "Element-wise",
        "name": "AXPY",
        "path": "axpy",
        "datatypes": "bf16, f32, i8, i16",
    },
    {
        "category": "Element-wise",
        "name": "Vector Add",
        "path": "vec-add",
        "datatypes": "bf16, f32, i8, i16",
    },
    {
        "category": "Normalization",
        "name": "RMS Normalization",
        "path": "rms_norm",
        "datatypes": "bf16",
    },
    {
        "category": "Normalization",
        "name": "Weighted RMS Normalization",
        "path": "weighted_rms_norm",
        "datatypes": "bf16",
    },
    {
        "category": "Normalization",
        "name": "Softmax",
        "path": "test_softmax",
        "datatypes": "bf16",
    },
    {
        "category": "Normalization",
        "name": "Layer Normalization",
        "path": "test_layernorm",
        "datatypes": "f32",
    },
    {
        "category": "Pooling",
        "name": "Average Pool",
        "path": "average_pool",
        "datatypes": "bf16",
    },
    {
        "category": "Special",
        "name": "2D Block Load",
        "path": "load_2d_block",
        "datatypes": "f32",
    },
    {
        "category": "Special",
        "name": "Multi-Driver",
        "path": "multi_drivers",
        "datatypes": "bf16",
    },
]

# Directories to ignore when verifying registry completeness
VERIFY_IGNORE = {"__pycache__"}


def get_device_support(example_dir):
    """Check which device targets have transform files.

    Returns (has_aie2, has_aie2p) as booleans.
    """
    has_aie2 = (example_dir / "transform_aie2.mlir").exists()
    has_aie2p = (example_dir / "transform_aie2p.mlir").exists()
    return has_aie2, has_aie2p


def generate_dashboard_table(base_url=""):
    """Generate markdown table rows for the operator dashboard."""
    rows = []
    for ex in EXAMPLES:
        example_dir = SCRIPT_DIR / ex["path"]
        if example_dir.is_dir():
            aie2, aie2p = get_device_support(example_dir)
        else:
            aie2, aie2p = False, False

        path = ex["path"]
        link = f"{base_url}{path}/"
        aie2_cell = "\u2705" if aie2 else "\u2014"
        aie2p_cell = "\u2705" if aie2p else "\u2014"
        row = (
            f'| {ex["category"]} '
            f'| [{ex["name"]}]({link}) '
            f'| {ex["datatypes"]} '
            f"| {aie2_cell} "
            f"| {aie2p_cell} "
            f"| [{path}/]({link}) |"
        )
        rows.append(row)
    return rows


def generate_readme(base_url=""):
    """Generate the full dashboard markdown content."""
    table_rows = generate_dashboard_table(base_url=base_url)
    table_body = "\n".join(table_rows)

    return f"""\
<!-- This file is auto-generated by generate_readme.py. Do not edit manually. -->

# Triton-XDNA Examples

These examples demonstrate how to write \
[Triton](https://github.com/triton-lang/triton) kernels that compile and \
run on AMD XDNA\u2122 NPUs via the \
[MLIR-AIR](https://github.com/Xilinx/mlir-air) compilation flow.

## Operator Dashboard

| Category | Operation | Datatype(s) | AIE2 | AIE2P | Example |
|:---------|:----------|:------------|:----:|:-----:|:--------|
{table_body}

### Legend

- \u2705 Transform file available (device target supported)
- \u2014 Not yet available

**AIE2** = AMD Ryzen\u2122 AI (Phoenix, NPU1) &nbsp;&nbsp; \
**AIE2P** = AMD Ryzen\u2122 AI (Strix, NPU2)

## Running Examples

Make sure XRT is sourced and a virtual environment with `triton-xdna` \
is active (see top-level [README](../README.md)):

```bash
source /opt/xilinx/xrt/setup.sh

# Run an example on AIE2 (NPU1):
cd matmul_bf16_m64_n64_k64
AIR_TRANSFORM_TILING_SCRIPT=transform_aie2.mlir python matmul_bf16_m64_n64_k64.py

# Run on AIE2P (NPU2):
AIR_TRANSFORM_TILING_SCRIPT=transform_aie2p.mlir python matmul_bf16_m64_n64_k64.py
```

## Running All Tests

```bash
python scripts/run_tests.py --device aie2 --verbose
python scripts/run_tests.py --device aie2p --verbose
```
"""


def verify_registry():
    """Check for example directories not listed in the EXAMPLES registry.

    Returns True if all directories are accounted for, False otherwise.
    """
    registered = {ex["path"] for ex in EXAMPLES}
    missing = []
    for entry in sorted(SCRIPT_DIR.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in VERIFY_IGNORE:
            continue
        # Skip build artifact directories
        if entry.name == "air_project":
            continue
        if entry.name not in registered:
            missing.append(entry.name)

    if missing:
        for name in missing:
            print(f"WARNING: {name}/ is not in the EXAMPLES registry")
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the Triton-XDNA examples operator dashboard."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "README.md",
        help="Output file path (default: examples/README.md)",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="Base URL prefix for example links (default: relative links)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify all example directories are in the registry (exit 1 if not)",
    )
    args = parser.parse_args()

    if args.verify:
        if not verify_registry():
            print("Registry verification failed.")
            sys.exit(1)
        print("Registry verification passed.")

    content = generate_readme(base_url=args.base_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)
    print(f"Generated {args.output}")
