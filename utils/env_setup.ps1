# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Windows environment setup for Triton-XDNA
# Usage: . .\utils\env_setup.ps1
#
# Prerequisites:
#   - Python 3.10, 3.11, or 3.12 (Xilinx Windows wheels do not yet support 3.13+)
#   - A virtual environment activated (e.g. python -m venv venv && .\venv\Scripts\Activate.ps1)
#   - XRT SDK at C:\Program Files\AMD\xrt (download xrt_windows_sdk.zip from XRT releases)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

function Read-HashField {
    param(
        [Parameter(Mandatory)] [string]$Path,
        [Parameter(Mandatory)] [string]$Field
    )
    $line = Select-String -Path $Path -Pattern "^${Field}:" | Select-Object -First 1
    if (-not $line) {
        throw "Field '${Field}' not found in $Path"
    }
    return ($line.Line -split ":\s+", 2)[1].Trim()
}

# =============================================================================
# Install triton-windows
# =============================================================================

Write-Host "Installing triton-windows..."
python -m pip install triton-windows

# =============================================================================
# Install mlir-air with [aie] extra
# =============================================================================
# The mlir-air wheel pins the matching mlir-aie commit and requires llvm-aie,
# so a single pip install resolves the whole MLIR-AIE/AIR/LLVM-AIE stack with
# a guaranteed-compatible mlir-aie.

$HashFile           = Join-Path $ScriptDir "mlir-air-hash.txt"
$MLIR_AIR_COMMIT    = Read-HashField -Path $HashFile -Field "Commit"
$SHORT_AIR_COMMIT   = $MLIR_AIR_COMMIT.Substring(0, 7)
$MLIR_AIR_VERSION   = Read-HashField -Path $HashFile -Field "Version"
$MLIR_AIR_TIMESTAMP = Read-HashField -Path $HashFile -Field "Timestamp"

Write-Host "Using mlir-air hash: $SHORT_AIR_COMMIT"
Write-Host "mlir-air version: $MLIR_AIR_VERSION"
Write-Host "mlir-air timestamp: $MLIR_AIR_TIMESTAMP"

python -m pip install "mlir_air[aie]==$MLIR_AIR_VERSION.$MLIR_AIR_TIMESTAMP+$SHORT_AIR_COMMIT.no.rtti" `
    -f https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels-no-rtti `
    -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-no-rtti `
    -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly

# The [aie] extra requires llvm-aie without a version pin. To track the
# nightly wheel, force-upgrade llvm-aie explicitly so an existing installation
# doesn't silently satisfy the unpinned requirement.
python -m pip install --upgrade llvm-aie `
    -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly

if (-not $env:MLIR_AIE_INSTALL_DIR) {
    $MLIR_AIE_INSTALL_DIR = (python -c "import importlib.util; spec = importlib.util.find_spec('mlir_aie'); print(spec.submodule_search_locations[0])")
    $env:MLIR_AIE_INSTALL_DIR = $MLIR_AIE_INSTALL_DIR
}

$env:PATH = "$env:MLIR_AIE_INSTALL_DIR\bin;$env:PATH"
$env:PYTHONPATH = "$env:MLIR_AIE_INSTALL_DIR\python;$env:PYTHONPATH"

# =============================================================================
# Install triton-xdna (copies backend into triton-windows)
# =============================================================================

Write-Host "Installing triton-xdna..."
Push-Location $ProjectDir
python -m pip install -e . --no-build-isolation
Pop-Location

# pip install -e . with setuptools build_meta does not invoke the custom
# TritonXdnaDevelop command class, so the triton_shared backend is not
# automatically copied into the installed triton package.  Do it manually.
Write-Host "Copying triton_shared backend into triton..."
$TritonBackendsDir = python -c "import triton, os; print(os.path.join(os.path.dirname(triton.__file__), 'backends'))"
$TritonSharedSrc   = Join-Path $ProjectDir "third_party\triton_shared\backend"
$TritonSharedDst   = Join-Path $TritonBackendsDir "triton_shared"
if (-not (Test-Path $TritonSharedDst)) {
    Copy-Item -Recurse -Force $TritonSharedSrc $TritonSharedDst
    Write-Host "  Copied triton_shared backend to: $TritonSharedDst"
} else {
    Write-Host "  triton_shared backend already present at: $TritonSharedDst"
}

# =============================================================================
# Install PyTorch (CPU)
# =============================================================================

Write-Host "Installing PyTorch (CPU)..."
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

Write-Host ""
Write-Host "Environment setup complete."

# =============================================================================
# XRT Development Files
# =============================================================================
# Download xrt_windows_sdk.zip from https://github.com/Xilinx/XRT/releases and
# extract the inner xrt_sdk/xrt/ directory to C:\Program Files\AMD\xrt
# (the zip's top-level folder is xrt_sdk/, not xrt/).
$xrtDefault = Join-Path $env:PROGRAMFILES "AMD\xrt"
if (Test-Path (Join-Path $xrtDefault "include\xrt\xrt_bo.h")) {
    Write-Host "XRT SDK found at: $xrtDefault"
} else {
    Write-Warning "XRT SDK not found at $xrtDefault. Download xrt_windows_sdk.zip from XRT releases and move the inner xrt_sdk/xrt/ folder there."
}
