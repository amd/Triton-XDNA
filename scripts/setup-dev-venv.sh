#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Create a development venv for the heterogeneous (iGPU + NPU) examples.
#
# Installs, in order:
#   1. ROCm PyTorch + the gfx1151 device payload from TheRock's multi-arch
#      nightly index (the per-family /v2/<target>/ indexes are frozen).
#   2. The prebuilt triton-xdna wheel, which brings mlir-air/mlir-aie/llvm-aie
#      with it. Installed AFTER torch so its `triton` wins over the
#      `pytorch-triton-rocm` that the torch wheel pulls in.
#   3. transformers (GPT-2 weights + reference) and pyxrt (NPUChain dispatch).
#
# Usage:  bash scripts/setup-dev-venv.sh [venv_dir]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${1:-$REPO/.venv}"
ROCM_INDEX="https://rocm.nightlies.amd.com/whl-multi-arch/"
GFX="gfx1151"

echo "=== [1/6] venv at $VENV ==="
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip wheel setuptools

echo "=== [2/6] ROCm PyTorch ($GFX) from TheRock ==="
"$VENV/bin/pip" install --index-url "$ROCM_INDEX" \
    torch "amd-torch-device-$GFX"

echo "=== [3/6] rocm-sdk runtime libraries ==="
"$VENV/bin/pip" install --index-url "$ROCM_INDEX" \
    "rocm[libraries,devel,device-$GFX]"

echo "=== [4/6] triton-xdna (+ mlir-air / mlir-aie / llvm-aie) ==="
"$VENV/bin/pip" install triton-xdna \
    --find-links https://github.com/amd/Triton-XDNA/releases/expanded_assets/latest-wheels \
    --find-links https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-no-rtti-2 \
    --find-links https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly \
    --find-links https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels-no-rtti

echo "=== [5/6] transformers ==="
"$VENV/bin/pip" install transformers

echo "=== [6/6] pyxrt (from the system XRT install) ==="
# XRT ships pyxrt as a prebuilt .so under its python dir; there is no wheel.
# Link it into the venv rather than copying so an XRT upgrade is picked up.
XRT_PY="${XILINX_XRT:-/opt/xilinx/xrt}/python"
SITE="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
if [[ -d "$XRT_PY" ]]; then
    echo "$XRT_PY" > "$SITE/xrt-python.pth"
    echo "linked $XRT_PY -> $SITE/xrt-python.pth"
else
    echo "WARNING: $XRT_PY not found; pyxrt (NPUChain / fused MLP) will be unavailable" >&2
fi

echo
echo "=== versions ==="
"$VENV/bin/python" - <<'PY'
import importlib, sys
for mod in ("torch", "triton", "transformers", "pyxrt", "ml_dtypes"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:14s} {getattr(m, '__version__', '?'):20s} {m.__file__}")
    except Exception as e:
        print(f"  {mod:14s} MISSING ({type(e).__name__}: {e})")
import torch
print(f"  torch.version.hip   = {torch.version.hip}")
print(f"  torch.cuda.available= {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device              = {torch.cuda.get_device_name(0)}")
PY

echo
echo "Done. Activate with:  source $VENV/bin/activate"
