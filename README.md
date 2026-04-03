# Triton-XDNA

**An experimental open-source project demonstrating compiler-driven kernel generation for AMD XDNA NPUs using [Triton](https://github.com/triton-lang/triton) and [MLIR-AIR](https://github.com/Xilinx/mlir-air).**

Triton-XDNA provides an end-to-end compilation flow that lowers standard Triton kernels directly to AMD NPU hardware — no prebuilt kernel libraries required. It bridges Triton's high-level parallel programming model with AMD's MLIR-AIR/AIE compilation stack, producing XRT-compatible binaries for AMD AI Engine architectures (AIE2 and AIE2P).

### How it works

Triton kernels are first lowered to compact Linalg compute graphs via [triton-shared](https://github.com/facebookincubator/triton-shared), then tiled and mapped onto parallel NPU cores using the MLIR Transform dialect, and finally compiled through [MLIR-AIR](https://github.com/Xilinx/mlir-air) and [MLIR-AIE](https://github.com/Xilinx/mlir-aie) to produce device binaries.

```
Triton kernel (@triton.jit)
  -> triton-shared (Linalg)
    -> MLIR Transform dialect (tiling, bufferization, vectorization)
      -> MLIR-AIR / MLIR-AIE
        -> XRT binary (aie.xclbin)
```

### Key results

- For dense matrix multiplication (I8/I16/BF16), compiler-generated kernels achieve **performance parity with handwritten NPU implementations**
- Over **90% of tested matmul configurations reach at least 90% of baseline throughput**; no configuration falls below 80%
- Currently supports matrix multiplication, elementwise operations, softmax, and layer normalization
- Complex compute graphs with reductions and broadcasts are mapped onto parallel NPU tiles

### Contributing

This is an experimental project and we welcome community contributions. Whether it's adding support for new kernel types, improving performance, or extending platform support — we'd love to collaborate.

## Usage

### Clone the repository
```
git clone https://github.com/amd/Triton-XDNA.git
cd Triton-XDNA
git submodule update --init
```

### Install XRT

Please follow the instructions in [mlir-aie project](https://github.com/Xilinx/mlir-aie/blob/main/README.md) on how to install the XDNA driver.

### Setup build environment 

#### Option 1: Install Pre-built Wheel (Recommended)

The easiest way to get started is to install the pre-built wheel from GitHub Releases:

```bash
python3 -m venv sandbox
source sandbox/bin/activate
python3 -m pip install --upgrade pip

# Install triton-xdna from GitHub Releases
pip install triton-xdna \
  --find-links https://github.com/amd/Triton-XDNA/releases/expanded_assets/latest-wheels \
  --find-links https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-no-rtti \
  --find-links https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly \
  --find-links https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels-no-rtti
```

**Note:** To install from a local wheel file:
```bash
pip install /path/to/triton_xdna-*.whl \
  --find-links https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-no-rtti \
  --find-links https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly \
  --find-links https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels-no-rtti
```

#### Option 2: Build from Source (Using Pip)

Starting from the root of the repository:

```bash
python3 -m venv sandbox
source sandbox/bin/activate
python3 -m pip install --upgrade pip
pip install cmake pybind11 nanobind wheel ninja pytest setuptools Cython

# Install triton-xdna from source and all dependencies automatically
pip install . --no-build-isolation \
  --find-links https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-no-rtti \
  --find-links https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly \
  --find-links https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels-no-rtti
```

This will automatically install all required dependencies:
- mlir-aie
- llvm-aie
- mlir-air

The versions are managed in `utils/mlir-aie-hash.txt`, `utils/llvm-aie-hash.txt`, and `utils/mlir-air-hash.txt`.

#### Option 3: Build from Source (Using Cmake)

```bash
python3 -m venv sandbox
source sandbox/bin/activate
python3 -m pip install --upgrade pip
pip install cmake pybind11 nanobind wheel ninja pytest setuptools Cython
source utils/env_setup.sh

cmake -GNinja -S . -Bbuild
cd build
ninja
```

Cmake shall install the C++ binaries under `third_party/triton/python/build`.
A triton python package with a new amd_triton_npu backend is also pip installed to the virtual environment `sandbox`.

### Run examples

Please make sure to run `source {path_to_xrt}/setup.sh` before running examples.
The test also depends on PyTorch as CPU reference.

```
cd examples/matmul_bf16_m64_n64_k64
AIR_TRANSFORM_TILING_SCRIPT=transform_aie2.mlir python matmul_bf16_m64_n64_k64.py
```

**Note:** The `transform_aie2.mlir` transform dialect IR is specifically designed for the AIE2 architecture. For AIE2P architecture, use `transform_aie2p.mlir` instead.

## Windows Support

Native Windows builds are supported using MSVC — no WSL or Linux required. The full
compilation pipeline (Triton → MLIR → xclbin → XRT dispatch) runs natively on Windows.

### Windows Requirements

- **Windows 10/11** (x64)
- **Visual Studio 2022** with "Desktop development with C++" workload
- **Python 3.12+**
- **CMake 3.20+** and **Ninja** (via pip or standalone)
- **AMD NPU driver** (installs `xrt_coreutil.dll` runtime)

### Windows Quick Start

```powershell
git clone https://github.com/amd/Triton-XDNA.git
cd Triton-XDNA
git submodule update --init

python -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip setuptools wheel
```

Prepare XRT development files (headers, import library, xclbinutil). Download
`xrt_windows_sdk.zip` from [Xilinx/XRT releases](https://github.com/Xilinx/XRT/releases)
and extract the `xrt/` directory to `C:\Program Files\AMD\xrt`:

```powershell
# The xrt/ folder inside the zip should end up at:
#   C:\Program Files\AMD\xrt\include\xrt\xrt_bo.h
#   C:\Program Files\AMD\xrt\lib\xrt_coreutil.lib
```

Run the automated build:

```powershell
.\utils\build_windows.ps1
```

This installs pre-built wheels (triton-windows, mlir-aie, llvm-aie), builds mlir-air
from source, and installs the Triton-XDNA backend. Takes approximately 30–60 minutes.

### Windows Manual Build

```powershell
pip install cmake ninja lit numpy PyYAML nanobind scipy
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install triton-windows
pip install mlir-aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-no-rtti
pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
```

mlir-air must be built from source (no Windows wheels yet):

```powershell
git clone https://github.com/Xilinx/mlir-air.git
cd mlir-air
git checkout <commit-from-utils/mlir-air-hash.txt>
git submodule update --init --recursive

cmake -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl `
  -DMLIR_DIR=<mlir-distro>/lib/cmake/mlir `
  -DLLVM_DIR=<mlir-distro>/lib/cmake/llvm `
  -DAIE_DIR=<mlir-aie-python-pkg>/lib/cmake/aie `
  -DLLVM_ENABLE_RTTI=OFF -DBUILD_SHARED_LIBS=OFF `
  -DAIR_RUNTIME_TARGETS="" -DAIR_ENABLE_GPU=OFF `
  -B build -S .
ninja -C build -j $env:NUMBER_OF_PROCESSORS
ninja -C build install
```

Install Triton-XDNA:

```powershell
$env:TRITON_PLUGIN_DIRS = "$PWD\third_party\triton_shared;$PWD\amd_triton_npu"
pip install -e . --no-build-isolation -v
```

### Additional Windows Tools

**xclbinutil** and **aiebu-asm** — Included in the XRT Windows SDK zip. Ensure they
are on PATH or in `<mlir_aie_install>/bin/`.

**DIA SDK** — If the mlir-air cmake build can't find DIA SDK:
```powershell
subst Z: "C:\Program Files\Microsoft Visual Studio\2022\Community\DIA SDK"
```

### Run examples (Windows)

```powershell
cd examples\vec-add
$env:AIR_TRANSFORM_TILING_SCRIPT = "transform_aie2p.mlir"
python vec-add.py
```

### Windows Environment Variables

| Variable | Purpose |
|----------|---------|
| `AIR_TRANSFORM_TILING_SCRIPT` | Path to MLIR transform dialect IR |
| `XILINX_XRT` | (Optional) Override XRT SDK location if not in `C:\Program Files\AMD\xrt` |

### Windows Known Limitations

- mlir-air must be built from source (no Windows wheels published)
- xclbinutil and aiebu-asm must be on PATH (from XRT Windows SDK)
- NPU driver must be installed
