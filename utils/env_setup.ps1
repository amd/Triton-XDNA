# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Windows environment setup for Triton-XDNA
# Usage: . .\utils\env_setup.ps1
#
# Prerequisites:
#   - Python 3.12 (required by mlir-air Windows wheel)
#   - A virtual environment activated (e.g. python -m venv venv312 && .\venv312\Scripts\Activate.ps1)
#   - XRT SDK at C:\Program Files\AMD\xrt (download xrt_windows_sdk.zip from XRT releases)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

# =============================================================================
# Install triton-windows
# =============================================================================

Write-Host "Installing triton-windows..."
python -m pip install triton-windows

# =============================================================================
# Install mlir-aie
# =============================================================================

if (-not $env:MLIR_AIE_INSTALL_DIR) {
    $HashFile = Join-Path $ScriptDir "mlir-aie-hash-windows.txt"
    $MLIR_AIE_COMMIT = (Select-String -Path $HashFile -Pattern "^Commit:" | ForEach-Object { ($_ -split ":\s+")[1] }).Trim()
    $SHORT_COMMIT = $MLIR_AIE_COMMIT.Substring(0, 7)
    $MLIR_AIE_VERSION = (Select-String -Path $HashFile -Pattern "^Version:" | ForEach-Object { ($_ -split ":\s+")[1] }).Trim()
    $TIMESTAMP = (Select-String -Path $HashFile -Pattern "^Timestamp:" | ForEach-Object { ($_ -split ":\s+")[1] }).Trim()

    Write-Host "Using mlir-aie hash: $SHORT_COMMIT"
    Write-Host "Version: $MLIR_AIE_VERSION"
    Write-Host "Timestamp: $TIMESTAMP"

    python -m pip install "mlir_aie==$MLIR_AIE_VERSION.$TIMESTAMP+$SHORT_COMMIT.no.rtti" -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-no-rtti

    $MLIR_AIE_INSTALL_DIR = (python -c "import importlib.util; spec = importlib.util.find_spec('mlir_aie'); print(spec.submodule_search_locations[0])") 
    $env:MLIR_AIE_INSTALL_DIR = $MLIR_AIE_INSTALL_DIR
}

$env:PATH = "$env:MLIR_AIE_INSTALL_DIR\bin;$env:PATH"
$env:PYTHONPATH = "$env:MLIR_AIE_INSTALL_DIR\python;$env:PYTHONPATH"

# =============================================================================
# Install llvm-aie
# =============================================================================

$HashFile = Join-Path $ScriptDir "llvm-aie-hash-windows.txt"
$LLVM_AIE_COMMIT = (Select-String -Path $HashFile -Pattern "^Commit:" | ForEach-Object { ($_ -split ":\s+")[1] }).Trim()
$LLVM_AIE_VERSION = (Select-String -Path $HashFile -Pattern "^Version:" | ForEach-Object { ($_ -split ":\s+")[1] }).Trim()
$LLVM_AIE_TIMESTAMP = (Select-String -Path $HashFile -Pattern "^Timestamp:" | ForEach-Object { ($_ -split ":\s+")[1] }).Trim()

Write-Host "Using llvm-aie hash: $LLVM_AIE_COMMIT"
Write-Host "llvm-aie version: $LLVM_AIE_VERSION"
Write-Host "llvm-aie timestamp: $LLVM_AIE_TIMESTAMP"

python -m pip install "llvm_aie==$LLVM_AIE_VERSION.$LLVM_AIE_TIMESTAMP+$LLVM_AIE_COMMIT" -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly

# =============================================================================
# Install mlir-air
# =============================================================================

$HashFile = Join-Path $ScriptDir "mlir-air-hash-windows.txt"
$MLIR_AIR_COMMIT = (Select-String -Path $HashFile -Pattern "^Commit:" | ForEach-Object { ($_ -split ":\s+")[1] }).Trim()
$SHORT_AIR_COMMIT = $MLIR_AIR_COMMIT.Substring(0, 7)
$MLIR_AIR_VERSION = (Select-String -Path $HashFile -Pattern "^Version:" | ForEach-Object { ($_ -split ":\s+")[1] }).Trim()
$MLIR_AIR_TIMESTAMP = (Select-String -Path $HashFile -Pattern "^Timestamp:" | ForEach-Object { ($_ -split ":\s+")[1] }).Trim()

Write-Host "Using mlir-air hash: $SHORT_AIR_COMMIT"
Write-Host "mlir-air version: $MLIR_AIR_VERSION"
Write-Host "mlir-air timestamp: $MLIR_AIR_TIMESTAMP"

python -m pip install "mlir_air==$MLIR_AIR_VERSION.$MLIR_AIR_TIMESTAMP+$SHORT_AIR_COMMIT.no.rtti" -f https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels-no-rtti

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
# Download xrt_windows_sdk.zip from https://github.com/Xilinx/XRT/releases
# and extract the xrt/ directory to C:\Program Files\AMD\xrt.
# The driver.py auto-detect will find it there without any env var.
$xrtDefault = Join-Path $env:PROGRAMFILES "AMD\xrt"
if (Test-Path (Join-Path $xrtDefault "include\xrt\xrt_bo.h")) {
    Write-Host "XRT SDK found at: $xrtDefault"
} else {
    Write-Warning "XRT SDK not found at $xrtDefault. Download xrt_windows_sdk.zip from XRT releases and extract xrt/ there."
}
