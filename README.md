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
cd examples/matmul
AIR_TRANSFORM_TILING_SCRIPT=transform_aie2.mlir python matmul.py
```

**Note:** The `transform_aie2.mlir` transform dialect IR is specifically designed for the AIE2 architecture. For AIE2P architecture, use `transform_aie2p.mlir` instead.
