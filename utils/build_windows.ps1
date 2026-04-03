# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# End-to-end Windows build script for Triton-XDNA
#
# This script:
#   1. Installs pre-built wheels (mlir-aie, llvm-aie, triton-windows, torch)
#   2. Downloads the MLIR distro wheel (needed to build mlir-air)
#   3. Builds mlir-air from source with MSVC + Ninja
#   4. Installs triton-xdna with plugins
#
# Prerequisites:
#   - Visual Studio 2022 (or Build Tools) with C++ workload
#   - CMake (3.20+), Ninja, Git
#   - Python 3.13 venv activated
#   - XRT Windows SDK (download xrt_windows_sdk.zip from Xilinx/XRT releases)
#
# Usage:
#   cd C:\projects\Triton-XDNA
#   .\venv\Scripts\activate
#   .\utils\build_windows.ps1 [-SkipInstallWheels] [-SkipBuildMlirAir] [-Jobs 8]
#

param(
    [switch]$SkipInstallWheels,
    [switch]$SkipBuildMlirAir,
    [int]$Jobs = 0,           # 0 = auto-detect
    [string]$BuildDir = "$PSScriptRoot\..\mlir-air-build",
    [string]$XrtDir = ""
)

$ErrorActionPreference = "Stop"
$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

# ── Helpers ────────────────────────────────────────────────────────────────
function Write-Step($n, $msg) {
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host " [$n] $msg" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
}

function Assert-Command($cmd) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Error "'$cmd' not found. Please install it and ensure it's on PATH."
        exit 1
    }
}

# ── Pre-flight checks ─────────────────────────────────────────────────────
Write-Host "Triton-XDNA Windows Build" -ForegroundColor Green
Write-Host "Project root: $PROJECT_ROOT"
Write-Host "Python:       $(python --version 2>&1)"
Write-Host "Python exe:   $(python -c 'import sys; print(sys.executable)')"

Assert-Command "cmake"
Assert-Command "ninja"
Assert-Command "git"

# Find MSVC
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vsWhere)) {
    Write-Error "Visual Studio not found. Install VS 2022 with C++ workload."
    exit 1
}
$vsPath = & $vsWhere -latest -property installationPath
$vcvars = "$vsPath\VC\Auxiliary\Build\vcvars64.bat"
Write-Host "Visual Studio: $vsPath"

if ($Jobs -eq 0) {
    $Jobs = [Environment]::ProcessorCount
}
Write-Host "Parallel jobs: $Jobs"

# ── Read hash files ────────────────────────────────────────────────────────
function Read-Hash($file, $key) {
    $content = Get-Content $file
    foreach ($line in $content) {
        if ($line -match "^${key}:\s*(.+)$") { return $Matches[1].Trim() }
    }
    throw "Key '$key' not found in $file"
}

$AIE_COMMIT  = Read-Hash "$PROJECT_ROOT\utils\mlir-aie-hash.txt" "Commit"
$AIE_TS      = Read-Hash "$PROJECT_ROOT\utils\mlir-aie-hash.txt" "Timestamp"
$AIE_VER     = Read-Hash "$PROJECT_ROOT\utils\mlir-aie-hash.txt" "Version"
$AIE_SHORT   = $AIE_COMMIT.Substring(0, 7)

$AIR_COMMIT  = Read-Hash "$PROJECT_ROOT\utils\mlir-air-hash.txt" "Commit"
$AIR_TS      = Read-Hash "$PROJECT_ROOT\utils\mlir-air-hash.txt" "Timestamp"
$AIR_VER     = Read-Hash "$PROJECT_ROOT\utils\mlir-air-hash.txt" "Version"
$AIR_SHORT   = $AIR_COMMIT.Substring(0, 7)

$PEANO_COMMIT = Read-Hash "$PROJECT_ROOT\utils\llvm-aie-hash.txt" "Commit"
$PEANO_TS     = Read-Hash "$PROJECT_ROOT\utils\llvm-aie-hash.txt" "Timestamp"
$PEANO_VER    = Read-Hash "$PROJECT_ROOT\utils\llvm-aie-hash.txt" "Version"

Write-Host ""
Write-Host "Pinned versions:"
Write-Host "  mlir-aie: $AIE_VER.$AIE_TS+$AIE_SHORT"
Write-Host "  mlir-air: $AIR_VER.$AIR_TS+$AIR_SHORT (build from source)"
Write-Host "  llvm-aie: $PEANO_VER.$PEANO_TS+$PEANO_COMMIT"

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: Install pre-built wheels
# ══════════════════════════════════════════════════════════════════════════
if (-not $SkipInstallWheels) {
    Write-Step 1 "Installing pre-built Python wheels"

    # Core build/runtime dependencies
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install cmake ninja lit numpy PyYAML nanobind scipy

    # PyTorch (CPU is fine for Triton compilation)
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

    # triton-windows (instead of upstream Linux-only triton)
    python -m pip install triton-windows

    # mlir-aie — find the closest matching version
    Write-Host "Installing mlir-aie..."
    $aie_no_rtti = "$AIE_VER.$AIE_TS+$AIE_SHORT.no.rtti"
    # Try exact version first, then with timestamp ±1
    $aie_installed = $false
    foreach ($ts_offset in @(0, 1, -1, 2, -2)) {
        $ts_try = [long]$AIE_TS + $ts_offset
        $ver_try = "$AIE_VER.$ts_try+$AIE_SHORT.no.rtti"
        $result = python -m pip install "mlir-aie==$ver_try" `
            -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-no-rtti `
            2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Installed mlir-aie==$ver_try"
            $aie_installed = $true
            break
        }
    }
    if (-not $aie_installed) {
        Write-Error "Failed to install mlir-aie. Check available versions."
        exit 1
    }

    # llvm-aie (Peano) — use latest available version matching the commit
    Write-Host "Installing llvm-aie..."
    $peano_installed = $false
    foreach ($ts_offset in @(0, 1, -1, 2, -2)) {
        $ts_try = [long]$PEANO_TS + $ts_offset
        $ver_try = "$PEANO_VER.$ts_try+$PEANO_COMMIT"
        $result = python -m pip install "llvm-aie==$ver_try" `
            -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly `
            2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Installed llvm-aie==$ver_try"
            $peano_installed = $true
            break
        }
    }
    if (-not $peano_installed) {
        # The pinned hash is very old — try latest version
        Write-Host "  Pinned version not found, installing latest llvm-aie..."
        python -m pip install llvm-aie `
            -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to install llvm-aie."
            exit 1
        }
    }

    Write-Host ""
    Write-Host "Pre-built wheels installed." -ForegroundColor Green
}

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: Download MLIR distro wheel (for building mlir-air)
# ══════════════════════════════════════════════════════════════════════════
if (-not $SkipBuildMlirAir) {
    Write-Step 2 "Preparing MLIR distro for mlir-air build"

    $mlirDir = "$BuildDir\mlir"
    if (-not (Test-Path "$mlirDir\lib\cmake\mlir")) {
        New-Item $BuildDir -ItemType Directory -Force | Out-Null

        # Get the MLIR distro version from mlir-air's clone-llvm.sh
        # We'll try to download it programmatically
        Write-Host "Downloading MLIR distro wheel..."

        # First clone mlir-air to read its version requirements
        $airSrcDir = "$BuildDir\mlir-air"
        if (-not (Test-Path "$airSrcDir\.git")) {
            git clone https://github.com/Xilinx/mlir-air.git $airSrcDir
        }
        Push-Location $airSrcDir
        git fetch origin
        git checkout $AIR_COMMIT
        git submodule update --init --recursive
        Pop-Location

        # Extract MLIR wheel version from clone-llvm.sh
        $cloneLlvmSh = Get-Content "$airSrcDir\utils\clone-llvm.sh" -Raw
        if ($cloneLlvmSh -match 'MLIR_WHEEL_VERSION="([^"]+)"') {
            $mlirWheelVersion = $Matches[1]
        } elseif ($cloneLlvmSh -match "MLIR_WHEEL_VERSION='([^']+)'") {
            $mlirWheelVersion = $Matches[1]
        } else {
            # Try to extract from the --get-wheel-version logic
            Write-Host "  Could not parse MLIR wheel version from clone-llvm.sh"
            Write-Host "  Attempting to run clone-llvm.sh --get-wheel-version via Git Bash..."
            $gitBash = "C:\Program Files\Git\bin\bash.exe"
            if (Test-Path $gitBash) {
                $mlirWheelVersion = & $gitBash -c "cd '$airSrcDir' && bash utils/clone-llvm.sh --get-wheel-version 2>/dev/null"
                $mlirWheelVersion = $mlirWheelVersion.Trim()
            }
        }

        if (-not $mlirWheelVersion) {
            Write-Error "Could not determine MLIR wheel version. Check $airSrcDir\utils\clone-llvm.sh"
            exit 1
        }
        Write-Host "  MLIR distro version: $mlirWheelVersion"

        # Download and extract the MLIR wheel
        $mlirWheelDir = "$BuildDir\mlir_wheel"
        New-Item $mlirWheelDir -ItemType Directory -Force | Out-Null
        python -m pip download "mlir==$mlirWheelVersion" `
            -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/mlir-distro `
            --no-deps --only-binary :all: -d $mlirWheelDir
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to download MLIR distro wheel."
            exit 1
        }

        # Extract the wheel (it's a zip)
        $mlirWhl = Get-ChildItem "$mlirWheelDir\mlir-*.whl" | Select-Object -First 1
        if (-not $mlirWhl) {
            Write-Error "No MLIR wheel matching 'mlir-*.whl' was found in $mlirWheelDir."
            Write-Host "Contents of ${mlirWheelDir}:" -ForegroundColor Yellow
            $dirContents = Get-ChildItem $mlirWheelDir -ErrorAction SilentlyContinue
            if ($dirContents) {
                $dirContents | ForEach-Object { Write-Host "  $($_.Name)" }
            } else {
                Write-Host "  (directory is empty)"
            }
            Write-Error "Please verify that 'pip download' produced an MLIR wheel and that its name matches the expected pattern."
            exit 1
        } else {
            Write-Host "  Extracting $($mlirWhl.Name)..."
            Expand-Archive $mlirWhl.FullName -DestinationPath $BuildDir -Force
        }

        if (-not (Test-Path "$mlirDir\lib\cmake\mlir")) {
            Write-Error "MLIR extraction failed — lib\cmake\mlir not found in $mlirDir"
            exit 1
        }

        # The MLIR distro wheel was built with a specific VS edition (e.g. Enterprise).
        # Patch hardcoded DIA SDK paths to match the local VS installation.
        $llvmExports = "$mlirDir\lib\cmake\llvm\LLVMExports.cmake"
        if (Test-Path $llvmExports) {
            $vsPath = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath
            $localDiaLib = "$vsPath/DIA SDK/lib/amd64/diaguids.lib" -replace '\\','/'
            $exportContent = Get-Content $llvmExports -Raw
            if ($exportContent -match 'C:/Program Files/Microsoft Visual Studio/2022/[^/]+/DIA SDK/lib/amd64/diaguids\.lib') {
                $exportContent = $exportContent -replace 'C:/Program Files/Microsoft Visual Studio/2022/[^/]+/DIA SDK/lib/amd64/diaguids\.lib', $localDiaLib
                $exportContent | Set-Content $llvmExports -NoNewline
                Write-Host "  Patched DIA SDK path in LLVMExports.cmake to match local VS"
            }
        }

        Write-Host "  MLIR distro extracted to $mlirDir" -ForegroundColor Green
    } else {
        Write-Host "MLIR distro already present at $mlirDir"
    }

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: Build mlir-air from source
    # ══════════════════════════════════════════════════════════════════════
    Write-Step 3 "Building mlir-air from source (MSVC + Ninja)"

    $airSrcDir = "$BuildDir\mlir-air"
    $airBuildDir = "$BuildDir\air-build"
    $airInstallDir = "$BuildDir\air-install"

    # Clone cmakeModules if needed
    $cmakeModulesDir = "$BuildDir\cmakeModules"
    if (-not (Test-Path $cmakeModulesDir)) {
        git clone --depth 1 https://github.com/Xilinx/cmakeModules.git $cmakeModulesDir
    }

    # Get mlir-aie install path
    $mlirAieDir = python -c "import mlir_aie; import pathlib; print(pathlib.Path(mlir_aie.__path__[0]))"
    $mlirAieDir = $mlirAieDir.Trim()
    Write-Host "  mlir-aie at: $mlirAieDir"

    # Get lit path
    $litPath = (python -c "import shutil; print(shutil.which('lit'))").Trim()
    Write-Host "  lit at: $litPath"

    # Create and enter build dir
    New-Item $airBuildDir -ItemType Directory -Force | Out-Null

    # Run CMake configure + build inside a MSVC Developer environment
    # We create a batch script that sets up vcvars64 and runs cmake+ninja

    # CMake requires forward slashes in paths to avoid escape-character issues
    $cmakeAirSrcDir    = $airSrcDir    -replace '\\','/'
    $cmakeAirBuildDir  = $airBuildDir  -replace '\\','/'
    $cmakeModulesDirFw = $cmakeModulesDir -replace '\\','/'
    $cmakeMlirDir      = $mlirDir      -replace '\\','/'
    $cmakeAieDir       = $mlirAieDir   -replace '\\','/'
    $cmakeLitPath      = $litPath      -replace '\\','/'
    $cmakeAirInstDir   = $airInstallDir -replace '\\','/'
    $cmakePythonExe    = (python -c "import sys; print(sys.executable)").Trim() -replace '\\','/'

    # ── MSVC linker RSP fix ─────────────────────────────────────────────
    # CMake+Ninja generates response files (.rsp) where LINK_LIBRARIES is
    # space-separated on a SINGLE 100K+ char line.  MSVC link.exe silently
    # truncates / corrupts paths on such extreme lines, resulting in
    # LNK1181 errors with mangled filenames.
    #
    # fix_rsp.ps1  – converts spaces to newlines (quote-aware) in an RSP.
    # patch_rules.ps1 – injects a call to fix_rsp.ps1 into every Ninja
    #                   linker rule *command* so the fix runs automatically
    #                   each time Ninja regenerates the RSP.
    # ────────────────────────────────────────────────────────────────────

    $fixRspContent = @'
param([string]$RspFile)
if (-not $RspFile -or -not (Test-Path $RspFile)) { exit 0 }
$c = [IO.File]::ReadAllText($RspFile)
$sb = [System.Text.StringBuilder]::new($c.Length)
$inQuote = $false
for ($i = 0; $i -lt $c.Length; $i++) {
    $ch = $c[$i]
    if ($ch -eq '"') {
        $inQuote = -not $inQuote; [void]$sb.Append($ch)
    } elseif ($ch -eq ' ' -and -not $inQuote) {
        [void]$sb.Append([char]10)
    } else {
        [void]$sb.Append($ch)
    }
}
[IO.File]::WriteAllText($RspFile, $sb.ToString())
'@
    Set-Content "$airBuildDir\fix_rsp.ps1" -Value $fixRspContent -Force

    $fixRspAbsolute = "$airBuildDir\fix_rsp.ps1" -replace '\\','/'
    $patchRulesContent = @"
`$rulesFile = "CMakeFiles\rules.ninja"
if (-not (Test-Path `$rulesFile)) { Write-Host "  rules.ninja not found, skipping patch"; exit 0 }
`$content = [IO.File]::ReadAllText(`$rulesFile)
`$old = 'cmd.exe /C "`$PRE_LINK'
`$new = 'cmd.exe /C "powershell -NoProfile -File $fixRspAbsolute `$RSP_FILE && `$PRE_LINK'
if (`$content.Contains(`$old)) {
    `$content = `$content.Replace(`$old, `$new)
    [IO.File]::WriteAllText(`$rulesFile, `$content)
    Write-Host "  Patched rules.ninja for MSVC RSP file compatibility"
} else {
    Write-Host "  rules.ninja already patched or pattern not found"
}
"@
    Set-Content "$airBuildDir\patch_rules.ps1" -Value $patchRulesContent -Force

    $cmakeConfig = @"
@echo off
call "$vcvars" >nul 2>&1

cd /d "$airBuildDir"

cmake "$cmakeAirSrcDir" ^
    -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_C_COMPILER=cl ^
    -DCMAKE_CXX_COMPILER=cl ^
    -DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON ^
    -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON ^
    -DCMAKE_C_VISIBILITY_PRESET=hidden ^
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden ^
    -DCMAKE_MODULE_PATH="$cmakeModulesDirFw" ^
    -DMLIR_DIR="$cmakeMlirDir/lib/cmake/mlir" ^
    -DLLVM_DIR="$cmakeMlirDir/lib/cmake/llvm" ^
    -DAIE_DIR="$cmakeAieDir/lib/cmake/aie" ^
    -DLLVM_EXTERNAL_LIT="$cmakeLitPath" ^
    -DPython3_EXECUTABLE="$cmakePythonExe" ^
    -DCMAKE_INSTALL_PREFIX="$cmakeAirInstDir" ^
    -DLLVM_ENABLE_RTTI=OFF ^
    -DBUILD_SHARED_LIBS=OFF ^
    -DAIR_RUNTIME_TARGETS="" ^
    -DAIR_ENABLE_GPU=OFF

if %ERRORLEVEL% neq 0 (
    echo CMake configure failed!
    exit /b 1
)

echo.
echo Patching build rules for MSVC linker RSP compatibility...
powershell -NoProfile -File patch_rules.ps1

echo.
echo Building with $Jobs parallel jobs...
ninja -j $Jobs

if %ERRORLEVEL% neq 0 (
    echo Build failed!
    exit /b 1
)

echo.
echo Installing...
ninja install

if %ERRORLEVEL% neq 0 (
    echo Install failed!
    exit /b 1
)

echo Build complete!
"@

    $batPath = "$BuildDir\build_air.bat"
    $cmakeConfig | Out-File -Encoding ascii $batPath
    Write-Host "  Running CMake configure + build..."
    Write-Host "  Build dir: $airBuildDir"
    Write-Host "  Install dir: $airInstallDir"
    Write-Host "  (This may take 15-30 minutes)"
    Write-Host ""

    & $batPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "mlir-air build failed. Check output above."
        exit 1
    }

    # Install the Python package from the build
    Write-Host ""
    Write-Host "  Installing mlir-air Python package..."

    # The Python bindings are built into the install directory
    # Add them to the Python path via a .pth file
    $sitePackages = python -c "import site; print(site.getsitepackages()[0])"
    $sitePackages = $sitePackages.Trim()

    # Create .pth file to add the air install to Python path
    "$airInstallDir\python" | Out-File -Encoding ascii "$sitePackages\mlir-air.pth"

    # Also add the bin directory to PATH for air-opt.exe and aircc.exe
    $env:PATH = "$airInstallDir\bin;$env:PATH"

    Write-Host "  mlir-air installed." -ForegroundColor Green
}

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: Set up environment variables
# ══════════════════════════════════════════════════════════════════════════
Write-Step 4 "Setting environment variables"

# XRT development files — auto-detected from C:\Program Files\AMD\xrt
$xrtDefault = Join-Path $env:PROGRAMFILES "AMD\xrt"
if ($XrtDir -and (Test-Path "$XrtDir\include\xrt\xrt_bo.h")) {
    Write-Host "  XRT SDK (override): $XrtDir"
} elseif (Test-Path (Join-Path $xrtDefault "include\xrt\xrt_bo.h")) {
    Write-Host "  XRT SDK: $xrtDefault"
} else {
    Write-Warning "XRT SDK not found. Download xrt_windows_sdk.zip from https://github.com/Xilinx/XRT/releases"
    Write-Warning "and extract the xrt/ directory to $xrtDefault"
}

# LLVM binary dir (for llc, etc.)
$llvmAieDir = python -c "import llvm_aie; import pathlib; print(pathlib.Path(llvm_aie.__path__[0]))" 2>$null
if ($llvmAieDir) {
    $llvmAieDir = $llvmAieDir.Trim()
    $env:LLVM_BINARY_DIR = "$llvmAieDir\bin"
    Write-Host "  LLVM_BINARY_DIR = $env:LLVM_BINARY_DIR"
}

# ══════════════════════════════════════════════════════════════════════════
# STEP 5: Initialize submodules and build triton-xdna
# ══════════════════════════════════════════════════════════════════════════
Write-Step 5 "Building Triton-XDNA"

Push-Location $PROJECT_ROOT

# Initialize submodules
if (-not (Test-Path "third_party\triton\CMakeLists.txt")) {
    Write-Host "  Initializing git submodules..."
    git submodule update --init --recursive
}

# Install triton-xdna in development mode
Write-Host "  Installing triton-xdna (develop mode)..."
$env:TRITON_PLUGIN_DIRS = "$PROJECT_ROOT\third_party\triton_shared;$PROJECT_ROOT\amd_triton_npu"
python -m pip install -e . --no-build-isolation -v

Pop-Location

# ══════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host " Build complete!" -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host ""

if ($llvmAieDir) {
    Write-Host "Environment variables to set in your shell:"
    Write-Host "  `$env:LLVM_BINARY_DIR = `"$llvmAieDir\bin`""
    Write-Host ""
}
Write-Host "To test:"
Write-Host "  cd $PROJECT_ROOT\examples\vec-add"
Write-Host "  `$env:AIR_TRANSFORM_TILING_SCRIPT = `"transform_aie2p.mlir`""
Write-Host "  python vec-add.py"
