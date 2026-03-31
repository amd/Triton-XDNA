# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Windows environment setup for Triton-XDNA
# Usage: . .\utils\env_setup.ps1

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# =============================================================================
# Install mlir-aie
# =============================================================================

if (-not $env:MLIR_AIE_INSTALL_DIR) {
    $HashFile = Join-Path $ScriptDir "mlir-aie-hash.txt"
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

$HashFile = Join-Path $ScriptDir "llvm-aie-hash.txt"
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

$HashFile = Join-Path $ScriptDir "mlir-air-hash.txt"
$MLIR_AIR_COMMIT = (Select-String -Path $HashFile -Pattern "^Commit:" | ForEach-Object { ($_ -split ":\s+")[1] }).Trim()
$SHORT_AIR_COMMIT = $MLIR_AIR_COMMIT.Substring(0, 7)
$MLIR_AIR_VERSION = (Select-String -Path $HashFile -Pattern "^Version:" | ForEach-Object { ($_ -split ":\s+")[1] }).Trim()
$MLIR_AIR_TIMESTAMP = (Select-String -Path $HashFile -Pattern "^Timestamp:" | ForEach-Object { ($_ -split ":\s+")[1] }).Trim()

Write-Host "Using mlir-air hash: $SHORT_AIR_COMMIT"
Write-Host "mlir-air version: $MLIR_AIR_VERSION"
Write-Host "mlir-air timestamp: $MLIR_AIR_TIMESTAMP"

python -m pip install "mlir_air==$MLIR_AIR_VERSION.$MLIR_AIR_TIMESTAMP+$SHORT_AIR_COMMIT.no.rtti" -f https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels-no-rtti

Write-Host ""
Write-Host "Environment setup complete."

# =============================================================================
# XRT Development Files
# =============================================================================
# The Ryzen AI SDK ships only the runtime DLL. Triton-XDNA needs headers and
# an import library for JIT compilation. Run setup_xrt_dev.ps1 to prepare them:
#
#   .\utils\setup_xrt_dev.ps1
#
# Then set the environment variable (the script prints the path):
#   $env:XILINX_XRT = "<path_to_xrt_dev>"
if (-not $env:XILINX_XRT -and -not $env:XRT_DEV_DIR) {
    $xrtDevDefault = Join-Path $ScriptDir "..\xrt-dev"
    if (Test-Path (Join-Path $xrtDevDefault "include\xrt\xrt_bo.h")) {
        $env:XILINX_XRT = (Resolve-Path $xrtDevDefault).Path
        Write-Host "Auto-detected XRT dev dir: $env:XILINX_XRT"
    } else {
        Write-Warning "XILINX_XRT / XRT_DEV_DIR not set. Run .\utils\setup_xrt_dev.ps1 first."
    }
}
