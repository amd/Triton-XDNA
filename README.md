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
  --find-links https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-no-rtti-2 \
  --find-links https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly \
  --find-links https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels-no-rtti
```

**Note:** To install from a local wheel file:
```bash
pip install /path/to/triton_xdna-*.whl \
  --find-links https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-no-rtti-2 \
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
  --find-links https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-no-rtti-2 \
  --find-links https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly \
  --find-links https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels-no-rtti
```

This will automatically install all required dependencies:
- mlir-aie
- llvm-aie
- mlir-air

The mlir-air version is pinned in `utils/mlir-air-hash.txt`. The matching mlir-aie commit is pinned by the mlir-air wheel's `[aie]` extra, so it's resolved transitively. llvm-aie uses the latest nightly release.

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

Browse the full set of available operators, their supported datatypes, and AIE2/AIE2P coverage in the live [examples dashboard](https://amd.github.io/Triton-XDNA/).

Please make sure to run `source {path_to_xrt}/setup.sh` before running examples.
The test also depends on PyTorch as CPU reference.

```bash
cd examples/matmul_bf16_m64_n64_k64
AIR_TRANSFORM_TILING_SCRIPT=transform_aie2.mlir python matmul_bf16_m64_n64_k64.py
```

**Note:** The `transform_aie2.mlir` transform dialect IR is specifically designed for the AIE2 architecture. For AIE2P architecture, use `transform_aie2p.mlir` instead.

### Launch runtime: XRT (default) or HSA

By default kernels are dispatched through XRT (xclbin on npu1, ELF on npu2). An alternative HSA via ROCR runtime dispatches Triton-generated kernels through the AIE agent path (`hsa_amd_aie_kernel_dispatch_packet_t`). Select the runtime by passing it to the driver — `NPUDriver("hsa")` or `NPUDriver("xrt")` — or via the `AMD_TRITON_NPU_RUNTIME` environment variable (honored by a bare `NPUDriver()`):

| Value | Behavior |
| --- | --- |
| `xrt` (default) | Dispatch via XRT; artifact is `xclbin` (npu1) or `elf` (npu2). |
| `hsa` | Dispatch via HSA; the backend produces `pdi` + `insts.bin` and launches them on the AIE agent. |

Under HSA the output format is `pdi`. The HSA runtime is Linux-only and requires
an **AIE-capable ROCR** — one that provides the AIE dispatch extension header
(`include/hsa/hsa_ext_amd_aie.h`) and `libhsa-runtime64.so`. A stock `/opt/rocm`
usually lacks AIE support, so you typically build ROCR from source (below) and
point the backend at it with `AMD_NPU_ROCR_PATH`.

The backend searches, in order: `AMD_NPU_ROCR_PATH`, `ROCM_PATH`, a pip-installed ROCm (TheRock's `rocm-sdk` wheels), then `/opt/rocm`. A candidate is accepted only if it provides *all* the headers the runtime includes — including `hsa/hsa_ext_amd_aie.h` — plus `libhsa-runtime64`, so an installation without AIE support is reported at startup rather than failing later in the compile. If nothing qualifies, the error lists every candidate and what each was missing.

```bash
cd examples/hsa_matmul
python hsa_matmul.py
```

Or activate it programmatically:

```python
import triton
from triton.backends.amd_triton_npu.driver import NPUDriver

triton.runtime.driver.set_active(NPUDriver("hsa"))
```

#### Building an AIE-capable ROCR

Build rocr-runtime from source and install it to a private prefix. The helper
`scripts/build-rocr.sh` automates this (defaults install to `<workspace>/opt/rocm`,
where `<workspace>` is the directory containing this checkout):

```bash
# Requires the rocm-systems repo (rocr-runtime, with AIE support) checked out.
# Override ROCR_SRC / PREFIX / CLANG_DIR / JOBS via the environment if needed.
./scripts/build-rocr.sh
```

Equivalently, by hand:

```bash
ROCR_SRC=/path/to/rocm-systems/projects/rocr-runtime
PREFIX=$HOME/opt/rocm                 # install prefix -> AMD_NPU_ROCR_PATH
cmake -S "$ROCR_SRC" -B "$ROCR_SRC/build" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_BUILD_TYPE=Release \
  # Match your installed clang/LLVM
  -DClang_DIR=/usr/lib/cmake/clang-22 \
  -DIMAGE_SUPPORT=OFF \
  -DBUILD_SHARED_LIBS=ON
cmake --build "$ROCR_SRC/build" -j"$(nproc)"
cmake --install "$ROCR_SRC/build"
```

This installs `include/hsa/hsa_ext_amd_aie.h` and `lib/libhsa-runtime64.so` under
`$PREFIX`. Point the backend at it and put its `lib/` ahead of any system ROCR at
runtime:

```bash
export AMD_NPU_ROCR_PATH="$PREFIX"                       # build/link the HSA runtime against it
export LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH}"  # load its libhsa-runtime64 first
```

`AMD_NPU_ROCR_PATH` selects which ROCR the backend compiles and links the shared
HSA runtime (`libtriton_npu_hsa.so`) against. `LD_LIBRARY_PATH` then guarantees
the freshly built `libhsa-runtime64.so` is loaded ahead of a system-installed one
(e.g. a stock `/opt/rocm` without AIE support), avoiding a mismatch at dispatch
time.

## Windows Support

Native Windows builds are supported using MSVC — no WSL or Linux required. The full
compilation pipeline (Triton → MLIR → xclbin/ELF → XRT dispatch) runs natively on Windows.

**Kernel execution on Windows is currently validated on npu2 (AIE2P) devices only.**
On npu1 (AIE2) the compile pipeline completes, but dispatch goes through the older
xclbin/DPU path, which is not yet supported on the Windows NPU driver stack; kernels
abort with `ERT_CMD_STATE_ABORT` and produce zeroed output. Use Linux for npu1 for
now. See [#88](https://github.com/amd/Triton-XDNA/issues/88).

### Windows Requirements

- **Windows 10/11** (x64)
- **AMD NPU: npu2 (AIE2P)** — npu1 (AIE2) is compile-only on Windows for now
- **Visual Studio 2022** with "Desktop development with C++" workload
- **Python 3.10–3.14** (3.13 recommended). Prebuilt Windows wheels are published
  for all of these versions; 3.13 is recommended because it matches the prebuilt
  `pyxrt.pyd` in the current XRT Windows SDK (see below), so the runtime binding
  works without building it from source.
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
and extract the inner `xrt_sdk/xrt/` directory (note the zip's top-level
folder is `xrt_sdk/`) to `C:\Program Files\AMD\xrt`:

```powershell
# The contents of xrt_sdk/xrt/ inside the zip should end up at:
#   C:\Program Files\AMD\xrt\include\xrt\xrt_bo.h
#   C:\Program Files\AMD\xrt\lib\xrt_coreutil.lib
```

The same zip also contains the runtime Python binding `pyxrt.pyd` at
`xrt_sdk/xrt/python/pyxrt.pyd`, which is required at execution time (the NPU
launcher does `import pyxrt`). Copy it onto your interpreter's import path:

```powershell
# The current pyxrt.pyd targets Python 3.13. On a 3.13 venv, copy it into
# site-packages (or any directory on PYTHONPATH):
Copy-Item "path\to\xrt_sdk\xrt\python\pyxrt.pyd" ".\venv\Lib\site-packages\"
```

`pyxrt.pyd` loads `xrt_coreutil.dll` at import time, so ensure the AMD NPU driver
is installed (it provides that DLL) and on `PATH`. On a Python version other than
the one the `.pyd` targets, importing it raises `ImportError: DLL load failed`;
in that case, build `pyxrt` from source against your interpreter.

Run the automated environment setup (must be dot-sourced so PATH/env vars
persist in the current shell):

```powershell
. .\utils\env_setup.ps1
```

This installs the pre-built wheels (`triton-windows`, `mlir-air[aie]` which
transitively pulls `mlir-aie` and `llvm-aie`) and the Triton-XDNA backend.

### Windows Manual Build

Install build tools, PyTorch, and the MLIR-AIE/AIR/LLVM-AIE stack. The
`mlir_air[aie]` extra transitively pins matching `mlir-aie` and pulls
`llvm-aie`, so a single resolver pass installs the whole stack from the
Xilinx release pages:

```powershell
pip install cmake ninja lit numpy PyYAML nanobind scipy
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install triton-windows
pip install "mlir_air[aie]" `
  -f https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels-no-rtti `
  -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-no-rtti-2 `
  -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
```

To pin a specific mlir-air version, use the values from
`utils/mlir-air-hash.txt`:
`mlir_air[aie]==<Version>.<Timestamp>+<short-commit>.no.rtti`.

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

`transform_aie2p.mlir` targets npu2 (AIE2P). The npu1 (AIE2) equivalents —
`transform_aie2.mlir` and `AMD_TRITON_NPU_TARGET=npu1` — compile successfully on
Windows but do not yet execute; see the limitation noted under
[Windows Support](#windows-support).

### Windows Environment Variables

| Variable | Purpose |
|----------|---------|
| `AIR_TRANSFORM_TILING_SCRIPT` | Path to MLIR transform dialect IR |
| `XILINX_XRT` | (Optional) Override XRT SDK location if not in `C:\Program Files\AMD\xrt` |

### Windows Known Limitations

- Kernel execution is currently validated on npu2 (AIE2P) only. npu1 (AIE2)
  compiles but does not yet execute on Windows — dispatch uses the xclbin/DPU
  path, which the Windows NPU driver stack does not yet support, so kernels abort
  (`ERT_CMD_STATE_ABORT`) and outputs come back zeroed. Use Linux for npu1 for
  now. See [#88](https://github.com/amd/Triton-XDNA/issues/88)
- Python 3.10–3.14 supported; 3.13 recommended so the prebuilt `pyxrt.pyd` in the
  XRT Windows SDK can be used as-is. On other versions, build `pyxrt` from source
  to match your interpreter
- `pyxrt.pyd` (from the XRT Windows SDK zip) must be on `PYTHONPATH` /
  site-packages, and the AMD NPU driver's `xrt_coreutil.dll` must be on `PATH`
- xclbinutil and aiebu-asm must be on PATH (from XRT Windows SDK)
- NPU driver must be installed
