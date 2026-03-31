# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Set up XRT development files (headers + import library) on Windows.
#
# The Ryzen AI SDK ships only the runtime DLL (xrt_coreutil.dll).
# Triton-XDNA's JIT compiler needs C++ headers and an import library
# to compile the NPU launch shim at runtime.
#
# This script:
#   1. Sparse-clones the XRT headers from github.com/Xilinx/XRT
#   2. Generates xrt_coreutil.lib from the runtime DLL using MSVC tools
#   3. Produces a self-contained directory with include/ and lib/
#
# Usage:
#   .\utils\setup_xrt_dev.ps1 [-XrtDll <path>] [-OutputDir <path>]
#
# After running, set the environment variable:
#   $env:XILINX_XRT = "<OutputDir>"
#   # -- or --
#   $env:XRT_DEV_DIR = "<OutputDir>"
#
# Prerequisites:
#   - Git
#   - Visual Studio (for dumpbin.exe / lib.exe) -- Build Tools edition is fine
#   - Ryzen AI SDK installed (provides xrt_coreutil.dll)

param(
    [string]$XrtDll = "",
    [string]$OutputDir = "$PSScriptRoot\..\xrt-dev",
    [string]$XrtBranch = "master"
)

$ErrorActionPreference = "Stop"

# ── Resolve the runtime DLL ────────────────────────────────────────────────
if (-not $XrtDll) {
    # Auto-detect: check Ryzen AI SDK, then System32
    $candidates = @()
    if (Test-Path "C:\RyzenAI") {
        $versions = Get-ChildItem "C:\RyzenAI" -Directory | Sort-Object Name -Descending
        foreach ($v in $versions) {
            $candidates += Join-Path $v.FullName "xrt\xrt_coreutil.dll"
        }
    }
    $candidates += "C:\Windows\System32\xrt_coreutil.dll"

    foreach ($c in $candidates) {
        if (Test-Path $c) { $XrtDll = $c; break }
    }
    if (-not $XrtDll) {
        Write-Error "Could not find xrt_coreutil.dll. Pass -XrtDll <path> or install the Ryzen AI SDK."
        exit 1
    }
}
Write-Host "Using DLL: $XrtDll"

# ── Resolve output directory ───────────────────────────────────────────────
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
New-Item "$OutputDir\include\xrt\detail" -ItemType Directory -Force | Out-Null
New-Item "$OutputDir\include\xrt\experimental" -ItemType Directory -Force | Out-Null
New-Item "$OutputDir\lib" -ItemType Directory -Force | Out-Null
Write-Host "Output directory: $OutputDir"

# ── Clone XRT headers (sparse checkout) ────────────────────────────────────
$cloneDir = Join-Path $OutputDir "_xrt_src"
if (-not (Test-Path "$cloneDir\.git")) {
    Write-Host "Cloning XRT headers (sparse checkout)..."
    New-Item $cloneDir -ItemType Directory -Force | Out-Null
    Push-Location $cloneDir
    git init
    git remote add origin https://github.com/Xilinx/XRT.git
    git config core.sparseCheckout true
    @(
        "src/runtime_src/core/include/xrt"
        "src/runtime_src/core/include/xrt/detail"
        "src/runtime_src/core/include/xrt/experimental"
    ) | Out-File -Encoding ascii .git/info/sparse-checkout
    git pull --depth 1 origin $XrtBranch
    Pop-Location
} else {
    Write-Host "XRT source already cloned, updating..."
    Push-Location $cloneDir
    git pull --depth 1 origin $XrtBranch 2>$null
    Pop-Location
}

# Copy headers into include/xrt/
$srcInclude = "$cloneDir\src\runtime_src\core\include\xrt"
Copy-Item "$srcInclude\*.h" "$OutputDir\include\xrt\" -Force
if (Test-Path "$srcInclude\detail") {
    Copy-Item "$srcInclude\detail\*" "$OutputDir\include\xrt\detail\" -Force
}
if (Test-Path "$srcInclude\experimental") {
    Copy-Item "$srcInclude\experimental\*" "$OutputDir\include\xrt\experimental\" -Force
}
Write-Host "Headers installed."

# ── Copy runtime DLL ──────────────────────────────────────────────────────
Copy-Item $XrtDll "$OutputDir\lib\" -Force

# ── Find Visual Studio tools ──────────────────────────────────────────────
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vsWhere)) {
    Write-Error "vswhere.exe not found -- is Visual Studio or Build Tools installed?"
    exit 1
}
$vsPath = & $vsWhere -latest -property installationPath
$vcvars = "$vsPath\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) {
    Write-Error "vcvars64.bat not found at $vcvars"
    exit 1
}
Write-Host "Using Visual Studio: $vsPath"

# ── Generate import library ───────────────────────────────────────────────
Write-Host "Generating xrt_coreutil.lib..."

$batScript = @"
@echo off
call "$vcvars" >nul 2>&1

echo LIBRARY xrt_coreutil > "$OutputDir\lib\xrt_coreutil.def"
echo EXPORTS >> "$OutputDir\lib\xrt_coreutil.def"
dumpbin /EXPORTS "$XrtDll" | findstr /R "^  *[0-9][0-9]*  *[0-9A-Fa-f]" > "$OutputDir\lib\_exports_raw.txt"
for /f "tokens=4" %%%%i in ("$OutputDir\lib\_exports_raw.txt") do echo     %%%%i >> "$OutputDir\lib\xrt_coreutil.def"

lib /DEF:"$OutputDir\lib\xrt_coreutil.def" /OUT:"$OutputDir\lib\xrt_coreutil.lib" /MACHINE:X64 >nul
exit /b %ERRORLEVEL%
"@

$batPath = Join-Path $OutputDir "_gen_lib.bat"
$batScript | Out-File -Encoding ascii $batPath
$result = cmd /c $batPath
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to generate import library (exit code $LASTEXITCODE)"
    exit 1
}

# Clean up temp files
Remove-Item "$OutputDir\lib\_exports_raw.txt" -ErrorAction SilentlyContinue
Remove-Item $batPath -ErrorAction SilentlyContinue

# ── Verify ─────────────────────────────────────────────────────────────────
$headerOk = Test-Path "$OutputDir\include\xrt\xrt_bo.h"
$libOk    = Test-Path "$OutputDir\lib\xrt_coreutil.lib"
$dllOk    = Test-Path "$OutputDir\lib\xrt_coreutil.dll"

if ($headerOk -and $libOk -and $dllOk) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host " XRT development files ready!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Set one of these environment variables:"
    Write-Host "  `$env:XILINX_XRT = `"$OutputDir`""
    Write-Host "  `$env:XRT_DEV_DIR = `"$OutputDir`""
    Write-Host ""
    Write-Host "Contents:"
    Write-Host "  include/xrt/       - XRT C++ headers"
    Write-Host "  lib/xrt_coreutil.lib - Import library"
    Write-Host "  lib/xrt_coreutil.dll - Runtime DLL"
} else {
    Write-Error "Verification failed: header=$headerOk lib=$libOk dll=$dllOk"
    exit 1
}
