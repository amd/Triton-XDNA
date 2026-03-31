# Triton-XDNA on Windows

Native Windows support for AMD Ryzen AI NPU kernel compilation and dispatch via Triton.

This fork adds Windows platform support to Triton-XDNA, enabling the full compilation pipeline
(Triton → MLIR → xclbin → XRT dispatch) to run natively on Windows with MSVC — no WSL or Linux required.

## Performance

Using SmolLM2-1.7B-Instruct (bf16) with SwiGLU offloaded to the NPU:

| Metric | Value |
|--------|-------|
| Decode throughput | **11 tokens/s** |
| Per-token latency | **0.09s** |
| Prefill (25-token prompt) | **0.28s** |
| NPU dispatches/token | 24 (one per MLP layer) |

Tested on AMD Ryzen AI Max+ 395 (Strix Halo), XRT 2.19.0, NPU Driver 32.0.203.329.

## Requirements

### Hardware
- AMD Ryzen AI processor with XDNA NPU (Strix, Strix Halo, or newer)
- NPU driver installed (AMD Adrenalin or standalone NPU driver)

### Software
- **Windows 10/11** (x64)
- **Visual Studio 2022** (Community or higher) with "Desktop development with C++" workload
- **Python 3.13** (3.12+ should also work)
- **CMake 3.20+** and **Ninja** (via pip or standalone)
- **Git** with submodule support
- **Ryzen AI SDK** (provides `xrt_coreutil.dll` runtime)

## Quick Start

### 1. Clone and set up Python environment

```powershell
git clone https://github.com/rwfsmith/Triton-XDNA.git
cd Triton-XDNA
git submodule update --init

python -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip setuptools wheel
```

### 2. Prepare XRT development files

The Ryzen AI SDK ships only the runtime DLL. We need headers and an import library:

```powershell
.\utils\setup_xrt_dev.ps1
```

This will:
- Sparse-clone XRT C++ headers from GitHub
- Generate `xrt_coreutil.lib` from the runtime DLL
- Create a `xrt-dev/` directory with `include/` and `lib/`

### 3. Run the automated build

```powershell
.\utils\build_windows.ps1
```

This script handles everything:
1. Installs pre-built wheels (triton-windows, mlir-aie, llvm-aie, PyTorch)
2. Downloads the MLIR distro and builds mlir-air from source
3. Installs the Triton-XDNA backend

The build takes approximately 30–60 minutes (mostly mlir-air compilation).

### 4. Set environment and run a test

```powershell
# Set XRT dev directory (if not auto-detected)
$env:XILINX_XRT = "C:\projects\Triton-XDNA\xrt-dev"  # adjust path

# Run vec-add test
cd examples\vec-add
$env:AIR_TRANSFORM_TILING_SCRIPT = "transform_aie2p.mlir"
python vec-add.py
```

You should see output like:
```
Testing size 1024... PASS
Testing size 2048... PASS
...
```

## Manual Build (Step by Step)

If the automated script doesn't work for your setup, here are the individual steps:

### Install Python dependencies

```powershell
pip install cmake ninja lit numpy PyYAML nanobind scipy
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install triton-windows
```

### Install mlir-aie and llvm-aie wheels

```powershell
# Read pinned versions from hash files, or install latest:
pip install mlir-aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-no-rtti
pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
```

### Build mlir-air from source

mlir-air doesn't ship Windows wheels yet, so it must be built from source:

```powershell
# 1. Download MLIR distro wheel and extract it
pip download mlir==<version> -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/mlir-distro --no-deps -d mlir_wheel
# Extract the .whl (it's a zip) to get lib/cmake/mlir/

# 2. Clone mlir-air
git clone https://github.com/Xilinx/mlir-air.git
cd mlir-air
git checkout <commit-from-utils/mlir-air-hash.txt>
git submodule update --init --recursive

# 3. Configure and build with MSVC
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl ^
  -DMLIR_DIR=<mlir-distro>/lib/cmake/mlir ^
  -DLLVM_DIR=<mlir-distro>/lib/cmake/llvm ^
  -DAIE_DIR=<mlir-aie-python-pkg>/lib/cmake/aie ^
  -DLLVM_ENABLE_RTTI=OFF -DBUILD_SHARED_LIBS=OFF ^
  -DAIR_RUNTIME_TARGETS="" -DAIR_ENABLE_GPU=OFF ^
  -B build -S .
ninja -C build -j%NUMBER_OF_PROCESSORS%
ninja -C build install
```

### Install Triton-XDNA

```powershell
cd Triton-XDNA
$env:TRITON_PLUGIN_DIRS = "$PWD\third_party\triton_shared;$PWD\amd_triton_npu"
pip install -e . --no-build-isolation -v
```

## Additional Tools

### xclbinutil

The MLIR-AIE pipeline needs `xclbinutil` to package device binaries. Options:

1. **Build from XRT source** (recommended): Clone [Xilinx/XRT](https://github.com/Xilinx/XRT),
   build just the `xclbinutil` target with CMake/MSVC
2. **Use the included Python assembler** as a fallback: `utils/xclbin_assemble.py`

Place the executable or make it available at:
```
<mlir_aie_install>/bin/xclbinutil.exe
```

### aiebu-asm

Required for AIE binary generation. Build from [Xilinx/aiebu](https://github.com/Xilinx/aiebu)
with CMake/MSVC and place `aiebu-asm.exe` in the mlir-aie bin directory.

### DIA SDK (for mlir-air build)

The MLIR distro uses PDB debug info, which requires the DIA SDK. If cmake can't find it:

```powershell
# Create a drive letter mapping (session-only, not persistent across reboots)
subst Z: "C:\Program Files\Microsoft Visual Studio\2022\Community\DIA SDK"
```

## LLM Inference Demo

Run real LLM inference with NPU-offloaded SwiGLU activations:

```powershell
pip install transformers accelerate

cd examples\llm_inference
python llm_real.py
```

This loads SmolLM2-1.7B-Instruct and offloads the SwiGLU activation function
in all 24 MLP layers to the NPU. Linear projections and attention stay on CPU.

The demo includes:
- **Auto-capture fast dispatch**: First dispatch per tensor size goes through the full
  Triton JIT pipeline; subsequent dispatches bypass it entirely via cached C extensions
- **Per-token timing**: Shows prefill and decode latency breakdown
- **Correctness verification**: Generates coherent text proving end-to-end correctness

### LLM Kernel Tests

Test individual NPU kernels used in LLM inference:

```powershell
python examples\llm_inference\llm_npu.py
```

Tests SiLU, GELU, SwiGLU, residual add, and RMSNorm with numerical verification.

## What Changed (vs upstream)

### Modified files
- **`setup.py`** — Windows-compatible paths (`/` vs `\`), dynamic library extension
  (`.pyd` vs `.so`), MSVC compiler flags
- **`CMakeLists.txt`** — MSVC compatibility, skip `-fPIC` on Windows
- **`pyproject.toml`** — Added `triton-windows` as Windows alternative to `triton`
- **`amd_triton_npu/backend/compiler.py`** — Windows path handling for MLIR tools
- **`amd_triton_npu/backend/driver.py`** — Major changes:
  - MSVC JIT compilation (replaces GCC): `/std:c++latest`, `/O2`, proper link flags
  - XRT header/library discovery via `XILINX_XRT` / `XRT_DEV_DIR`
  - Cached `xrt::run` objects (reuse via `start()`/`wait()` instead of re-creating)
  - Skip output buffer device-sync (NPU writes directly)
  - Global module cache with `_last_dispatched_module` for fast-path capture
  - Optional per-dispatch profiling (`AMD_TRITON_NPU_PROFILE_DISPATCH=1`)

### New files
- **`utils/build_windows.ps1`** — End-to-end automated build script
- **`utils/setup_xrt_dev.ps1`** — XRT development file preparation
- **`utils/env_setup.ps1`** — Windows environment setup (installs pinned wheels)
- **`utils/xclbin_assemble.py`** — Pure-Python xclbin assembler (fallback for xclbinutil)
- **`examples/llm_inference/`** — LLM inference demos and profiling tools

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `XILINX_XRT` or `XRT_DEV_DIR` | XRT headers + import lib directory | `C:\projects\xrt-dev` |
| `AIR_TRANSFORM_TILING_SCRIPT` | Path to MLIR transform dialect IR | `transform_aie2p.mlir` |
| `AMD_TRITON_NPU_OUTPUT_FORMAT` | Binary format: `xclbin` (default) or `elf` | `xclbin` |
| `AMD_TRITON_NPU_PROFILE_DISPATCH` | Enable per-dispatch C++ timing | `1` |
| `AMD_TRITON_NPU_BF16_EMULATION` | Force bf16 emulation | `0` |
| `AMD_TRITON_NPU_AIR_PROJECT_PATH` | Custom path for intermediate MLIR artifacts | `./air_project` |

## Known Limitations

- **mlir-air must be built from source** — no Windows wheels are published yet
- **xclbinutil and aiebu-asm must be built separately** — not yet included in wheels
- **NPU driver required** — the XDNA kernel module must be installed
- **Linear projections stay on CPU** — only activation functions are offloaded to NPU
- **Single-batch only** — no batched inference support yet

## License

MIT — same as upstream Triton-XDNA.
