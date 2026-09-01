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

The instructions below are for Linux. For Windows, skip to
[Windows Support](#windows-support).

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
(`include/hsa/hsa_ext_amd_aie.h`) and `libhsa-runtime64`.

Install one from [TheRock](https://github.com/ROCm/TheRock)'s nightly ROCm wheels;
the backend picks it up automatically, with no environment variables to set:

```bash
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ rocm-sdk-core
```

`rocm-sdk-core` is the only package needed — it carries both the headers and
`libhsa-runtime64`. AIE dispatch requires a build from **2026-07-16 or later**;
earlier ones lack the memory-handle resolution fix and abort at dispatch. Note
that the older per-GPU-family indexes (`https://rocm.nightlies.amd.com/v2/<target>/`)
are deprecated and frozen — they still resolve under pip, so pointing at one
silently installs a runtime too old to work.

The backend searches, in order: `AMD_NPU_ROCR_PATH`, `ROCM_PATH`, a pip-installed ROCm (TheRock's `rocm-sdk` wheels), then `/opt/rocm`. A candidate is accepted only if it provides *all* the headers the runtime includes — including `hsa/hsa_ext_amd_aie.h` — plus `libhsa-runtime64`, so an installation without AIE support is reported at startup rather than failing later in the compile. If nothing qualifies, the error lists every candidate and what each was missing. Set `AMD_NPU_ROCR_PATH` to override the search with a specific prefix — a locally built rocr-runtime, for instance.

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

`AMD_NPU_ROCR_PATH` selects which ROCR the backend compiles and links the shared
HSA runtime (`libtriton_npu_hsa.so`) against; that path is baked in as an rpath, so
the matching `libhsa-runtime64` is loaded without setting `LD_LIBRARY_PATH`. Set
`LD_LIBRARY_PATH` only to force a different one ahead of it — for example when a
system ROCR without AIE support would otherwise be picked up first.

#### Sharing a process with PyTorch

An rpath decides where a library is *found*, not whether it is looked for at
all: a `libhsa-runtime64.so.1` already loaded satisfies the dependency first. A
ROCm build of PyTorch bundles its own copy and loads it at import, so in a
process that uses both the iGPU and the NPU, `import torch` hands the AIE agent
to a ROCR that was never built for it — and that aborts inside ROCR rather than
raising. Preload the right one to settle it first:

```bash
LD_PRELOAD=$(python -c "from triton.backends.amd_triton_npu.driver import \
  _get_rocr_install; print(_get_rocr_install().lib_path)") python your_script.py
```

HIP then uses that ROCR too, so one runtime serves both devices.
`scripts/hsa-env.sh` sets this for the heterogeneous examples, and the backend
reports the problem with this instruction when the preload is missing.

#### Shared buffers

Under HSA a buffer from `triton.backends.amd_triton_npu.shared` is dispatched on
where it lives — the runtime recognises it and skips the staging copies it makes
for an ordinary tensor. It works in both directions: the NPU can allocate pages
the iGPU maps (`device="hsa:0", share="hip:0"`), and the iGPU can allocate pages
the NPU maps (`device="hip:0", share="hsa:0"`). A buffer names one NPU runtime
or the other; XRT and HSA cannot map each other's pages, and asking for both is
refused. `shared.hsa_dispatch_counts()` reports how many tensor arguments were
dispatched in place and how many were staged, which is the way to confirm a
buffer is doing what it was allocated for.

## Windows Support

Triton-XDNA runs natively on Windows — no WSL, no Linux VM. The whole flow,
from `@triton.jit` through MLIR to an NPU binary and kernel dispatch, executes
on the Windows host using MSVC.

Kernel execution is validated on **npu2 (AIE2P)** devices. Compilation itself
needs no NPU at all, so any Windows machine can build and cross-compile
artifacts.

### Requirements

| | |
|---|---|
| OS | Windows 10 or 11 (x64) |
| NPU | npu2 (AIE2P) to run kernels; none required to compile |
| Python | 3.11–3.14 (3.13 recommended — see [Set up XRT](#set-up-xrt)) |
| Compiler | Visual Studio 2022 or newer, with "Desktop development with C++" |
| Driver | AMD NPU driver |

Visual Studio does not need to be activated first — Triton-XDNA locates
`vcvars64.bat` on its own, so an ordinary PowerShell prompt is enough. Running
from an "x64 Native Tools Command Prompt" also works; that environment is used
as-is when present.

For the NPU driver itself, follow the
[mlir-aie instructions](https://github.com/Xilinx/mlir-aie/blob/main/README.md).

### Set up XRT

Triton-XDNA compiles a small host shim for every kernel, so it needs the XRT
SDK (headers and import library) in addition to the runtime DLL the driver
installs.

Download `xrt_windows_sdk.zip` from the
[Xilinx/XRT releases](https://github.com/Xilinx/XRT/releases) page and extract
the inner `xrt_sdk/xrt/` directory — note the zip's top-level folder is
`xrt_sdk/` — to `C:\Program Files\AMD\xrt`, so that you have:

```
C:\Program Files\AMD\xrt\include\xrt\xrt_bo.h
C:\Program Files\AMD\xrt\lib\xrt_coreutil.lib
```

To keep it somewhere else, point `XRT_DEV_DIR` at that directory instead:

```powershell
$env:XRT_DEV_DIR = "<path-to>\xrt"
```

The same zip ships the Python binding `pyxrt.pyd` at
`xrt_sdk/xrt/python/pyxrt.pyd`, which is used to identify the NPU. Copy it onto
your interpreter's import path:

```powershell
Copy-Item "<path-to>\xrt_sdk\xrt\python\pyxrt.pyd" ".\venv\Lib\site-packages\"
```

The published `pyxrt.pyd` targets Python 3.13, which is why that version is
recommended — on 3.13 it works as shipped. On other versions, build `pyxrt`
from source against your interpreter.

### Option 1: Install Pre-built Wheel (Recommended)

Windows wheels are published for every supported Python version, so no compiler
or source build is involved:

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip

pip install triton-xdna `
  --find-links https://github.com/amd/Triton-XDNA/releases/expanded_assets/latest-wheels `
  --find-links https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-no-rtti-2 `
  --find-links https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly `
  --find-links https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels-no-rtti

pip install torch --index-url https://download.pytorch.org/whl/cpu
```

PyTorch is used by the examples as a CPU reference.

### Option 2: Build from Source

Check the sources out with LF line endings, which is what the vendored
submodules expect:

```powershell
git config --global core.autocrlf false
git config --global core.eol lf

git clone https://github.com/amd/Triton-XDNA.git
cd Triton-XDNA
git submodule update --init

python -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip setuptools wheel
```

Then run the environment setup, which installs the MLIR-AIE/AIR/LLVM-AIE stack
and the Triton-XDNA backend. Dot-source it so the environment persists in your
shell:

```powershell
. .\utils\env_setup.ps1
```

To drive the install yourself instead, the `mlir_air[aie]` extra pins a matching
`mlir-aie` and pulls `llvm-aie`, so one resolver pass installs the whole stack:

```powershell
pip install cmake ninja lit numpy PyYAML nanobind scipy
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install triton-windows
pip install "mlir_air[aie]" `
  -f https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels-no-rtti `
  -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-no-rtti-2 `
  -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly

pip install . --no-build-isolation
```

To pin a specific mlir-air version, use the values from
`utils/mlir-air-hash.txt`:
`mlir_air[aie]==<Version>.<Timestamp>+<short-commit>.no.rtti`.

### Run examples

Browse the available operators and their AIE2/AIE2P coverage in the live
[examples dashboard](https://amd.github.io/Triton-XDNA/).

```powershell
cd examples\vec-add
$env:AIR_TRANSFORM_TILING_SCRIPT = "transform_aie2p.mlir"
python vec-add.py
```

Examples check their results against PyTorch and raise on mismatch, so a clean
exit with no output means the kernel ran correctly on the NPU. The compiled
artifacts and intermediate IR are left in `air_project\` next to the example.

To run the whole suite:

```powershell
python scripts\run_tests.py --device aie2p
```

`transform_aie2p.mlir` targets npu2 (AIE2P); `transform_aie2.mlir` targets npu1
(AIE2).

Compilation does not require matching hardware. Setting
`AMD_TRITON_NPU_TARGET` selects the device to compile for and
`AMD_TRITON_NPU_COMPILE_ONLY=1` stops before dispatch, so any Windows machine
can produce artifacts for either generation — they are written to
`air_project\`. Note that the examples verify their results against PyTorch, so
they are meant to be run with dispatch enabled; use compile-only when you want
the artifacts rather than a pass/fail result.

### Windows Environment Variables

| Variable | Purpose |
|----------|---------|
| `AIR_TRANSFORM_TILING_SCRIPT` | Path to the MLIR transform dialect tiling script |
| `XRT_DEV_DIR` | XRT SDK location, if not at `C:\Program Files\AMD\xrt` |
| `AMD_TRITON_NPU_XRT_DIR` | Same, taking precedence over `XRT_DEV_DIR`; also settable as `npu_config.xrt_dir` |
| `AMD_TRITON_NPU_TARGET` | Force `npu1` or `npu2` instead of detecting the installed device |
| `AMD_TRITON_NPU_COMPILE_ONLY` | `1` to compile without dispatching — build on a machine with no NPU |
| `AMD_TRITON_NPU_OUTPUT_FORMAT` | Force `elf` or `xclbin`; defaults to `elf` on npu2, `xclbin` on npu1 |
| `AMD_TRITON_NPU_BF16_EMULATION` | `1` to truncate f32 to bf16 before multiply, accumulating in f32 |
| `AMD_TRITON_NPU_AIR_PROJECT_PATH` | Where intermediate IR and artifacts are written (default `.\air_project`) |
| `AMD_TRITON_NPU_DEBUG` | `1` for verbose compiler and launcher output |

Each of these is also settable from Python via `npu_config` — see
`amd_triton_npu/backend/config.py`. The HSA runtime (`AMD_TRITON_NPU_RUNTIME`)
is Linux-only.

### Troubleshooting

**`ImportError: DLL load failed` when importing `pyxrt`** — the prebuilt
`pyxrt.pyd` targets Python 3.13. Either use a 3.13 interpreter or build `pyxrt`
from source against the version you are running.

**`pyxrt` imports but finds no device** — `pyxrt.pyd` loads `xrt_coreutil.dll`
from the AMD NPU driver. Confirm the driver is installed and that
`xrt-smi examine` lists the device.

**"XRT development files not found"** — the XRT SDK is missing or incomplete.
The error lists every location that was searched; extract
`xrt_windows_sdk.zip` as described in [Set up XRT](#set-up-xrt), or point
`XRT_DEV_DIR` at it. A runtime-only install (just the driver's DLLs) is not
enough to compile.

**npu1 (AIE2) kernels abort with `ERT_CMD_STATE_ABORT` and return zeros** —
npu1 dispatch uses the older xclbin/DPU path, which the Windows NPU driver stack
does not yet support. npu1 compiles on Windows but must be run on Linux; see
[#88](https://github.com/amd/Triton-XDNA/issues/88).
