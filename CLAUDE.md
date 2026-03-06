# CLAUDE.md - Triton-XDNA Development Guide

## Project Overview

Triton-XDNA is an AMD-developed Triton compiler backend plugin that enables compilation and execution of Triton kernels on AMD XDNA NPU (Neural Processing Unit) hardware. It bridges Triton's high-level parallel programming model with AMD's MLIR-AIR/AIE compilation stack, producing XRT-compatible binaries for AMD's AI Engine architecture (AIE2 and AIE2P devices).

The project packages two Triton backend plugins (`amd_triton_npu` and `triton_shared`) together with a patched Triton compiler into a single distributable Python wheel called `triton-xdna`.

## Repository Structure

```
Triton-XDNA/
├── amd_triton_npu/              # NPU backend plugin (main project code)
│   ├── amd_triton_npu.cc        # pybind11 plugin registration (minimal)
│   ├── CMakeLists.txt           # Builds plugin, links TritonSharedAnalysis + TritonTilingExtIR
│   └── backend/
│       ├── compiler.py          # NPUBackend: TTIR → TritonShared dialect pipeline
│       ├── driver.py            # NPUDriver: AIR transforms, aircc compilation, XRT launcher
│       ├── name.conf            # Backend identifier: "amd_triton_npu"
│       └── include/ExecutionEngine/  # C++ runtime utilities (CRunnerUtils, Msan)
├── third_party/
│   ├── triton/                  # Git submodule: github.com/triton-lang/triton
│   ├── triton_shared/           # Git submodule: github.com/facebookincubator/triton-shared
│   ├── triton.patch             # Autotuner patch (backend-aware cache keys)
│   └── triton_shared.patch      # CPU backend enhancements (IR dumping, type handling)
├── examples/                    # Example Triton kernels for NPU
│   ├── matmul/                  # Matrix multiplication (BF16, primary example)
│   ├── vec-add/                 # Vector addition (BF16)
│   ├── test_softmax/            # Softmax with numerical stability
│   ├── test_layernorm/          # Fused layer normalization
│   ├── load_2d_block/           # 2D block pointer patterns
│   ├── autotune-matmul/         # Autotuned matmul variant
│   ├── multi_drivers/           # Multi-backend (CPU + NPU) comparison
│   └── benchmark.py             # Benchmarking utilities and backend selectors
├── scripts/
│   ├── apply_patches.py         # Patch management for submodules
│   └── run_tests.py             # Test runner (discovers and executes examples)
├── utils/
│   ├── env_setup.sh             # Environment setup (installs mlir-aie/air/llvm-aie)
│   ├── mlir-aie-hash.txt        # Pinned mlir-aie commit/version
│   ├── mlir-air-hash.txt        # Pinned mlir-air commit/version
│   ├── llvm-aie-hash.txt        # Pinned llvm-aie commit/version
│   └── requirements_wheel.txt   # Build dependencies for wheel creation
├── ci/docker-based/             # Docker CI runner setup
├── .github/
│   ├── workflows/
│   │   ├── build.yml            # Main CI build (ubuntu-latest)
│   │   ├── small.yml            # Self-hosted NPU tests (docker runner)
│   │   └── nightly-wheels.yml   # Wheel building pipeline (matrix: py3.10-3.14)
│   └── actions/
│       ├── build/action.yaml    # Reusable build action
│       └── test/action.yaml     # Reusable test action (runs on NPU hardware)
├── CMakeLists.txt               # Root build orchestration (patch targets + triton-build)
├── setup.py                     # Custom wheel builder (renames triton → triton-xdna)
├── pyproject.toml               # Package metadata, entry points, cibuildwheel config
└── LICENSE                      # MIT License (AMD Inc.)
```

## Compilation Pipeline

The full compilation flow from Triton kernel to NPU execution:

```
Triton Kernel (@triton.jit)
    │
    ▼
NPUBackend.make_ttir()          → Triton IR (TTIR)
    │
    ▼
triton-shared-opt                → TritonShared / Linalg dialect
  --triton-to-linalg-experimental
    │
    ▼
air-opt (3 passes)              → AIR dialect
  ├─ resolve-tensor-conflicts + memory-space override
  ├─ tiling via Transform dialect IR (AIE2 or AIE2P)
  └─ wrap-func-with-parallel + copy-to-dma
    │
    ▼
aircc (aiecc)                   → aie.xclbin + insts.bin
    │
    ▼
C++ launcher generation         → __npu_dispatch.so
  (links XRT, Python, MLIR)
    │
    ▼
XRT kernel invocation           → NPU execution
```

## Build Instructions

### Prerequisites

- Python 3.10+
- CMake >= 3.20, Ninja
- Clang, LLD (for Triton compilation)
- XRT (Xilinx Runtime) installed at `/opt/xilinx/xrt/`

Always follow the README for build instructions. The key requirement is using a **Python virtual environment** (`python3 -m venv sandbox`). There are two main build options:

### Option 1: Pip Build (Recommended)

```bash
python3 -m venv sandbox && source sandbox/bin/activate
pip install --upgrade pip
pip install cmake pybind11 nanobind wheel ninja pytest setuptools Cython
pip install . --no-build-isolation \
  --find-links https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-no-rtti \
  --find-links https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly \
  --find-links https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels-no-rtti
```

This handles everything: applies patches, builds triton + plugins, installs mlir-aie/air/llvm-aie wheels.

### Option 2: CMake Build (Development)

```bash
python3 -m venv sandbox && source sandbox/bin/activate
pip install --upgrade pip
pip install cmake pybind11 nanobind wheel ninja pytest setuptools Cython
source utils/env_setup.sh
cmake -GNinja -S . -Bbuild
cd build && ninja
```

The `env_setup.sh` script installs mlir-aie, mlir-air, and llvm-aie from their respective release pages.

### Running examples

```bash
source /opt/xilinx/xrt/setup.sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
cd examples/matmul
AIR_TRANSFORM_TILING_SCRIPT=transform_aie2.mlir python matmul.py
```

## Key Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `LLVM_BINARY_DIR` | Yes | Path to LLVM tools (llc, opt, etc.) |
| `TRITON_PLUGIN_DIRS` | Build-time | Semicolon-separated paths to backend plugins |
| `TRITON_BUILD_WITH_CLANG_LLD` | Build-time | Set `true` to use clang/lld toolchain |
| `TRITON_SHARED_OPT_PATH` | Runtime | Path to `triton-shared-opt` binary |
| `AMD_TRITON_NPU_AIR_PROJECT_PATH` | Optional | Directory for intermediate IR dumps (default: `./air_project/`) |
| `AIR_TRANSFORM_TILING_SCRIPT` | Optional | Custom MLIR transform dialect file for tiling |
| `AMD_TRITON_NPU_COMPILE_ONLY` | Optional | Set `1` to compile without executing |
| `MLIR_AIE_INSTALL_DIR` | Runtime | Path to mlir-aie installation |
| `XILINX_XRT` | Runtime | Path to XRT installation |

## Running Examples and Tests

### Run a single example

```bash
source /opt/xilinx/xrt/setup.sh
export LLVM_BINARY_DIR=<path-to-llvm-bin>
cd examples/matmul
python matmul.py
```

Each example directory contains `transform_aie2.mlir` and `transform_aie2p.mlir` files specifying architecture-specific tiling strategies.

### Run the test suite

```bash
python scripts/run_tests.py --device aie2 --verbose --timeout 1200
```

The test runner auto-discovers `.py` files in example subdirectories and sets `AIR_TRANSFORM_TILING_SCRIPT` per example. Default skipped tests: layernorm, load_2d_block, multi_drivers.

## Architecture & Key Components

### Backend Plugin System

Triton discovers backends via Python entry points defined in `pyproject.toml`:

```toml
[project.entry-points."triton.backends"]
amd_triton_npu = "triton.backends.amd_triton_npu"
triton_shared = "triton.backends.triton_shared"
```

Each backend must provide:
- `compiler.py` with a class extending `BaseBackend` (compilation stages)
- `driver.py` with a class extending `DriverBase` (runtime execution)

Backends are symlinked into `third_party/triton/python/triton/backends/` during build.

### NPU Backend (`amd_triton_npu/backend/`)

**compiler.py** (`NPUBackend`):
- `NPUOptions` dataclass holds compilation config (num_warps, debug, cluster dims)
- Stages: `make_ttir` → `ttsharedir` (via `triton-shared-opt`)
- Target string: `"npu"`

**driver.py** (`NPUDriver`, `NPULauncher`, `NPUUtils`):
- NPU version detection via `xrt-smi examine` (npu1 = Phoenix/RyzenAI, npu2 = Strix)
- Transform IR generation with architecture-specific tiling (AIE2: 8-core, AIE2P: 16+ core)
- AIR transformation passes via `air-opt`
- C++ launcher code generation linking XRT for buffer management and kernel dispatch
- Binary caching via Triton's cache manager (MD5 hash of AIR IR as key)
- Auto-tuning support with execution timing

### TritonShared Backend (`third_party/triton_shared/`)

Provides a CPU reference backend and the TritonShared MLIR dialect used as an intermediate representation by both backends. The `triton-shared-opt` binary is the key tool for converting Triton IR to Linalg.

### Submodule Patch System

Patches in `third_party/*.patch` are applied via `scripts/apply_patches.py`:
- **triton.patch** (35 lines, 1 file): Two autotuner fixes — changes `@cached_property` to `@property` on `do_bench` so backend switches are detected, and adds backend name to the autotuner cache key so switching backends triggers retuning. Cannot be moved to backend plugin code because the Autotuner class has no extension points for cache key customization or property behavior override.
- **triton_shared.patch** (79 lines, 2 files): Adds `do_bench()` method to `CPUDriver` for autotuner benchmarking support (needed for multi-backend comparison), and fixes trailing newline in `triton-hash.txt`. Cannot be moved to the NPU plugin because it modifies triton_shared's CPU backend, not ours.

Marker files (`.patches_applied`) prevent re-application. The CMake build runs `apply-patches` before `triton-build`.

## Dependency Version Management

External dependencies are pinned via hash files in `utils/`:

| File | Package | Fields |
|------|---------|--------|
| `mlir-aie-hash.txt` | mlir-aie | Commit, Timestamp, Version |
| `mlir-air-hash.txt` | mlir-air | Commit, Timestamp, Version |
| `llvm-aie-hash.txt` | llvm-aie | Commit, Timestamp, Version |

These are parsed by `setup.py` to generate pinned `install_requires` entries and by CI workflows to install exact dependency versions.

## Wheel Building

The custom `setup.py` implements `TritonXdnaBdistWheel` which:

1. Applies patches to submodules
2. Builds a Triton wheel with `TRITON_PLUGIN_DIRS` set to both plugins
3. Unpacks the wheel and adds the `triton-shared-opt` binary
4. Renames the package from `triton` to `triton-xdna`
5. Updates METADATA with mlir-aie/llvm-aie/mlir-air dependencies
6. Repacks into a final distributable wheel

Version format: `{base_version}.{timestamp}+{commit_hash}` (e.g., `3.6.0.2026020404+abc1234`)

## CI/CD Pipeline

### build.yml (ubuntu-latest)
Standard build validation: checkout with submodules → install apt deps → install MLIR stack from hash files → cmake + ninja build.

### small.yml (self-hosted docker runner)
NPU hardware tests: checkout → `.github/actions/build` → `.github/actions/test` (sources XRT, runs `scripts/run_tests.py`).

### nightly-wheels.yml
Matrix wheel building (Python 3.10-3.14) via cibuildwheel on manylinux_2_28. Uploads to GitHub Releases (`latest-wheels` or tagged releases).

### Docker CI Runner
`ci/docker-based/` provides ephemeral GitHub Actions runners with XRT device passthrough (`/dev/accel/accel0`). The loop script continuously spawns fresh runners.

## Development Workflow

### Adding a new example kernel

1. Create a directory under `examples/` with your Triton kernel `.py` file
2. Add `transform_aie2.mlir` and `transform_aie2p.mlir` with tiling strategies
3. The test runner auto-discovers new examples unless added to the skip list in `scripts/run_tests.py`

### Modifying submodules

1. Make changes in `third_party/triton/` or `third_party/triton_shared/`
2. Generate a new patch: `cd third_party/triton && git diff > ../triton.patch`
3. Commit the updated `.patch` file
4. Run `ninja -Cbuild reapply-patches` to verify

### Updating submodule versions (triton or triton-shared)

When upstream triton-shared bumps its triton pin, use the rebase approach to regenerate `triton_shared.patch`:

1. `cd third_party/triton_shared && git checkout <old_commit>`
2. `git apply ../triton_shared.patch && git add -A && git commit -m "Apply old patch"`
3. `git rebase --onto <new_commit> <old_commit> HEAD`
4. Resolve any conflicts (usually just `triton-hash.txt`)
5. `git diff <new_commit> HEAD > ../triton_shared.patch`
6. Verify: `git checkout <new_commit> && git apply --check ../triton_shared.patch`

When updating the triton submodule to a new commit, clear the cached LLVM download: `rm -rf ~/.triton/llvm/` — otherwise the build reuses the old LLVM and compilation fails with API mismatch errors.

### Updating dependency versions

Edit the corresponding hash file in `utils/` (e.g., `mlir-aie-hash.txt`) with the new commit, timestamp, and version. CI and `setup.py` will pick up the changes automatically.

When updating mlir-air, check which mlir-aie commit it pins to via `utils/clone-mlir-aie.sh` in the mlir-air repo, and update mlir-aie to match. The llvm-aie version is independently managed and doesn't need to match triton's LLVM.

### Debugging compilation

Set `AMD_TRITON_NPU_AIR_PROJECT_PATH=./debug_ir/` to dump intermediate MLIR at each stage. Key files produced:
- `asm_src.mlir` - TritonShared dialect output
- `asm_air_output.mlir` - Post-AIR-transform output

## Code Conventions

- Python backend code lives in `amd_triton_npu/backend/` (compiler.py ~200 LOC, driver.py ~1000 LOC)
- C++ plugin registration is minimal (empty pybind11 module in `amd_triton_npu.cc`)
- Examples use BF16 data types with F32 accumulation as the standard pattern
- Transform MLIR files use the Transform dialect for architecture-specific tiling
- Tests validate against PyTorch CPU reference with relaxed tolerances (rtol=1e-1, atol=1e1 for BF16)
- Backend names are defined in `name.conf` files (plain text, single line)

## Hardware Targets

| Device | Architecture | NPU Version | Cores | Default Transform |
|--------|-------------|-------------|-------|-------------------|
| Phoenix / RyzenAI | AIE2 | npu1 | 8 (4x2) | `transform_aie2.mlir` |
| Strix | AIE2P | npu2 | 16+ | `transform_aie2p.mlir` |

NPU version is auto-detected at runtime via `xrt-smi examine` output parsing.

## LLVM Version Architecture

The project uses **two independent LLVM toolchains** that do NOT need to match:

1. **Triton's LLVM** (compile-time): Downloaded automatically by triton's `setup.py` from Azure blob storage based on `third_party/triton/cmake/llvm-hash.txt`. Used to build `triton-shared-opt` and triton's compiler. Cached at `~/.triton/llvm/`.

2. **mlir-aie/mlir-air's LLVM**: Embedded in pre-built wheels from Xilinx GitHub releases. Used by `air-opt`, `aie-opt`, and `aircc`. Version pinned via `utils/clone-llvm.sh` in the mlir-aie repo.

These two LLVM versions can diverge because the tools exchange data via **textual MLIR IR** using stable dialects (Linalg, memref, func). The output of `triton-shared-opt` (Linalg IR) is consumed by `air-opt` — as long as the standard MLIR dialect serialization format is compatible, different LLVM versions work fine. This has been verified with triton using LLVM `979132a0` and mlir-aie/mlir-air using LLVM `ebf5d9ef`.

## Build Architecture

`triton-shared-opt` is built **within triton's CMake build tree** as a plugin via `TRITON_PLUGIN_DIRS`. It links statically against triton's LLVM/MLIR libraries. The `triton-hash.txt` in triton-shared is patched to match our triton submodule commit, but this file is only informational — the actual build uses whichever triton submodule Triton-XDNA has checked out.

## Common Issues

- **XRT not found**: Ensure `/opt/xilinx/xrt/setup.sh` is sourced before running
- **Patches fail to apply**: Run `ninja -Cbuild reset-submodules` then `ninja -Cbuild apply-patches`
- **Missing triton-shared-opt**: Binary is built during CMake/pip build; check `build/` directory
- **Lock race conditions**: Small tile sizes on vec-add can cause memtile lock races (known limitation)
- **Build takes long**: Use ccache (CI uses 1GB cache); set `PARALLEL_LEVEL` for parallel compilation
- **Build fails with MLIR API errors after triton update**: Clear `~/.triton/llvm/` — the old cached LLVM doesn't match the new triton version
- **`pip install` fails resolving mlir-aie/llvm-aie**: These packages are on Xilinx GitHub releases, not PyPI. Use `--find-links` URLs as shown in the README, or `--no-deps` if deps are already installed
- **Always use a venv**: The README instructions require `python3 -m venv sandbox`. Building outside a venv causes `libpython.a` linking issues and externally-managed-environment errors
- **Matmul BF16 assertion failures (~2-3% mismatch)**: This is a pre-existing numerical precision issue with BF16 matmul on AIE2 hardware, not a build or code bug. The tolerances in the test (`atol=1e1, rtol=1e-1`) sometimes don't capture all outlier elements

## LLM-Assisted Transform Dialect Script Generation

This section documents findings from analyzing the paper "LLM-Driven Optimization of MLIR Transform Dialect for High-Level Synthesis" (CF '26, `cf26-paper356.pdf` in this repo) and how its approach relates to writing transform dialect scripts for MLIR-AIR in Triton-XDNA.

### Paper Summary

The paper demonstrates using GPT-4o to automatically generate MLIR Transform dialect schedules that optimize Linalg-on-tensors IR for FPGA/ASIC targets via the Bambu HLS backend. The core insight is that the Transform dialect separates **what** to optimize (the schedule) from **how** to verify correctness (the MLIR infrastructure) -- the LLM generates the schedule, and the MLIR transform interpreter enforces semantic preservation. If the LLM produces a malformed or illegal schedule, the MLIR verifier rejects it rather than producing silently wrong code.

**4-stage Chain-of-Thought prompting workflow:**

1. **Structure Analysis** -- The LLM is given the input Linalg MLIR and asked to describe its structure: operation types, tensor shapes, loop nesting, data dependencies, and reduction patterns. This forces the model to "read" the IR before attempting to transform it.
2. **HLS Expert Explanation** -- The LLM is given a structured hardware specification (target frequency, available resources, memory bandwidth) and asked to reason about which optimizations would improve the design: which loops to tile, what tile sizes fit in on-chip memory, which loops to unroll for parallelism.
3. **Critic** -- The LLM reviews its own proposed strategy for correctness: are the tile sizes legal (divide evenly), does the unroll factor exceed the trip count, do the memory promotions exceed BRAM capacity? This self-review stage catches common errors before code generation.
4. **Implementation** -- The LLM generates the actual Transform dialect MLIR code using only the primitives it was shown in few-shot examples.

**Primitive library:** The paper defines 15 "transformation primitives" -- pairs of (original MLIR + transform schedule + natural language description) covering: `tile_using_for`, `tile_using_forall`, `loop.unroll`, `bufferize_to_allocation` (memory promotion), `fuse_into_containing_op`, `vectorize`, `interchange`, and `generalize`. These serve as few-shot examples in the prompt, teaching the LLM the exact syntax and semantics of each operation.

**Results:** On PolyBench (30 kernels) and an EELS autoencoder: up to 7.53x speedup over unoptimized HLS baseline, 75% syntactically valid schedule generation rate, 6.6 minutes average per kernel, ~$4.50 API cost per kernel. Failures were primarily: incorrect `split_handle` indexing (wrong position assumptions), tile sizes that don't divide loop bounds, and unroll factors exceeding trip counts.

### What Transfers Directly to Triton-XDNA

**The schedule/verifier separation.** Our architecture already has this: `run_transform()` in `driver.py:291` executes a transform module against the payload IR, and the MLIR infrastructure rejects illegal transforms. An LLM-generated transform script that produces invalid IR would fail at `run_transform()` or at the subsequent `air-opt` passes, rather than silently producing wrong results. This is the key property that makes LLM generation viable.

**The primitive library concept.** The paper's 15 primitives map directly to our needs. We would build an equivalent library from our ~60 AIR transform ops (defined in `mlir-air/mlir/include/air/Dialect/AIR/AIRTransformOps.td`) plus the standard `transform.structured.*` ops. Each primitive entry would include:
- The op's MLIR syntax
- A before/after IR snippet showing what it does
- Natural language description of when to use it
- Constraints (e.g., `air.linalg_promote` requires that operands satisfy `promoteSubviewsPrecondition`)

Our existing transform scripts (`examples/matmul/transform_aie2.mlir` etc.) already have step-by-step annotations that serve as a starting point for this library.

**Hardware spec format.** The paper provides target hardware as structured natural language. We would adapt this for AIE2/AIE2P:

```
Target: AMD AIE2 (Phoenix), 4x2 core array (8 cores)
Per-core L1 memory: 16 KB (memory_space = 2)
Shared L2 MemTile: 512 KB (memory_space = 1)
Vector unit: 256-bit, BF16x16 or I8x32 per cycle
Accumulator: 8x F32 registers
DMA engines: 2 per core (L1<->L2), 2 per MemTile (L2<->L3)
Data types: BF16 inputs, F32 accumulation
```

**Iterative feedback loop.** The paper uses MLIR verifier errors as feedback to retry generation. We can extend this: dump IR at each step via `AMD_TRITON_NPU_AIR_PROJECT_PATH`, feed error messages or unexpected IR shapes back to the LLM for correction.

### What Is Harder for Triton-XDNA

**Script complexity.** The paper's generated schedules are 3-5 transform operations (tile + unroll + promote). Our matmul script is 34 steps across 11 phases. An LLM cannot reliably generate a 34-step schedule in one shot -- the IR changes at every step, and later steps depend on the exact structure produced by earlier ones. The paper itself notes that "the number of transformations... is moderate" as a factor in their success.

**Custom dialect ops.** No LLM has training data for `transform.air.par_to_herd`, `transform.air.herd_vectorize`, `transform.air.hoist_loop_invariant_transfers`, or any of the ~60 AIR-specific operations. The paper's primitives are all standard upstream MLIR transforms that GPT-4o has seen in open-source MLIR code. Teaching an LLM our custom ops requires extensive few-shot context, which competes with context window limits.

**Multi-level memory staging.** The paper's memory promotion is single-level (main memory -> BRAM). Our scripts implement 3-level staging (DDR -> L2 -> L1) with explicit pingpong buffering via loop fusion. The LLM must reason about which data goes where, when to fuse copy loops for double-buffering, and how tile sizes interact with memory capacity at each level. This is a qualitatively harder planning problem.

**Multi-core parallelism.** The paper targets single-threaded HLS. Our scripts must partition work across 8-16 AIE cores via `tile_using_forall` + `par_to_herd`, which requires reasoning about workload balance, data ownership, and L1 memory per core. HLS has no equivalent concept.

**Positional handle manipulation.** The paper's own failure analysis identifies incorrect `split_handle` indexing as a primary failure mode, which is exactly the fragility problem in our scripts. The paper works around this because their schedules are short enough that positional reasoning is tractable. Our 8-way read split + 4-way write split for accumulator hoisting is beyond what current LLMs can reliably get right.

**IR-state-aware reasoning.** Each step in our scripts operates on the IR produced by all previous steps. The LLM would need to mentally simulate the effect of each transform to reason about what operations exist at step 20. The paper avoids this because their schedules are short and mostly independent. For a 34-step pipeline, this becomes a fundamental limitation.

### Concrete Recommendations for This Project

**1. Build an AIR-specific primitive library.** Document each of the ~60 `transform.air.*` ops with before/after IR examples, following the paper's format. Source material already exists in the tablegen descriptions in `AIRTransformOps.td` and in the step annotations in existing transform scripts. Organize by category: memory (promote, copy, DMA), parallelism (herd, launch, segment), vectorization (herd_vectorize, vector_type_cast), loop (hoist, fuse, normalize), and math (rsqrt, broadcast). This library serves double duty: as few-shot context for LLM generation and as human reference documentation.

**2. Decompose scripts into independently-generable phases.** Rather than asking an LLM to generate a full 34-step matmul script, decompose the problem:
- **Phase prompt 1:** "Given this Linalg matmul IR and this AIE2 hardware spec, generate tiling and packing transforms for L2 staging." (Phases 1-3)
- **Phase prompt 2:** "Given this packed/tiled IR, generate K-dimension tiling and producer fusion." (Phase 4)
- **Phase prompt 3:** "Given this tiled IR, generate multi-core parallelization for an 8-core array." (Phase 5)
- Between each phase, run the generated script fragment, dump the resulting IR, and feed it as input to the next phase prompt. This mirrors the paper's single-kernel approach but applies it incrementally.

**3. Use the LLM for the structured phases, keep manual control for the fragile phases.** Phases 1-7 (tiling, packing, promotion, bufferization) are regular and well-suited to LLM generation -- they use standard MLIR transforms with clear rules. Phases 8-11 (pingpong fusion, vectorization tiling, herd mapping, accumulator hoisting) require fine-grained handle manipulation and AIE-specific knowledge that current LLMs lack. A practical workflow: LLM generates the first 70% of the script, human expert writes the final 30%.

**4. Adopt the 4-stage CoT prompt structure.** The paper's Structure Analysis -> Expert -> Critic -> Implementation workflow is directly applicable. For an AIR context:
- **Structure:** "Describe the Linalg operations, tensor shapes, and data dependencies in this MLIR."
- **Expert:** "Given AIE2 with 8 cores, 16KB L1, 512KB L2, and BF16x16 vector units, what tiling sizes and memory placement would maximize throughput?"
- **Critic:** "Do the proposed tile sizes fit in L1? Does pack size [4,4,8] of BF16 produce 4*8*2=64 bytes per tile, fitting in 16KB? Does forall tile [16,16] divide the problem dimensions?"
- **Implementation:** "Generate the Transform dialect MLIR using only these primitives: [primitive library]."

**5. Use MLIR verifier + IR dumps as a feedback loop.** When a generated script fails:
- If the transform interpreter rejects it: feed the error message back to the LLM with the failing step and ask for correction.
- If it succeeds but produces unexpected IR: dump the IR at the failing step via `AMD_TRITON_NPU_AIR_PROJECT_PATH` and show it to the LLM with the expected structure.
- If it compiles but produces wrong results: compare against the reference output and identify which phase diverged.

**6. Start with vec-add, not matmul.** The vec-add script (~15 steps, no packing, no reduction, no accumulator hoisting) is the closest analog to the paper's PolyBench kernels. Use it as the first test case for LLM-generated scripts. Graduate to softmax (reductions, multi-op fusion) and then matmul (packing, multi-core, pingpong) as confidence builds.

### Limitations Noted by the Paper That Apply Here

- **Regular compute patterns only.** The paper works on dense linear algebra (PolyBench). Irregular patterns (sparse, data-dependent control flow) are not addressed. Our kernels are also regular, so this is not a blocker.
- **Single-kernel scope.** The paper optimizes one kernel at a time. Our examples are also single-kernel, but a full Triton program may have multiple fused kernels. Cross-kernel optimization is out of scope for this approach.
- **Dependence on example diversity.** The LLM can only generate transforms it has seen in the primitive library. If a kernel needs an operation not in the library (e.g., `air.hoist_cast_pair` for mixed-precision accumulation), the LLM cannot invent it. The library must be comprehensive.
- **No autotuning integration.** The paper does not search over tile sizes -- it relies on the LLM's reasoning about hardware specs. Combining LLM-generated script skeletons with Triton's existing autotuner (which already supports NPU via the `do_bench` patch) could address this.

## AIE2P Vector Type Constraints

This section documents the AIE2P (Strix/NPU2) vector operation type constraints discovered through hardware testing and validated against `mlir-air/programming_examples/primitives/`. Understanding these constraints is essential for writing correct transform scripts.

### Type Mapping

AIE2P vector operations use a **mixed type system** -- not uniformly bf16 or f32:

| Operation | bf16 vector | f32 vector | Vector sizes |
|-----------|:-----------:|:----------:|:------------:|
| `arith.addf` | Yes | **No** | 16, 32 |
| `arith.subf` | Yes | **No** | 16, 32 |
| `arith.mulf` | Yes | **No** | 16, 64 |
| `arith.divf` | **No** | Yes | 16, 64 |
| `arith.maximumf` | Yes | **No** | 16, 32 |
| `arith.cmpf` | Yes (inputs) | **No** | 16, 32 |
| `arith.select` | Yes (values) | **No** | 16, 32 |
| `math.exp` | Yes | **No** | 16, 32 |
| `math.tanh` | Yes | **No** | 16 (AIE2P only) |
| `math.rsqrt` | **No** | Yes | 16, 32 |
| `vector.fma` | Yes | **No** | 16, 32 |
| `vector.reduction(add)` | Yes | **No** | 16, 32 |
| `vector.reduction(max)` | Yes | **No** | 16, 32 |
| reciprocal (`1/x`) | **No** | Yes | 16, 32 |

**Key rule**: Most arithmetic and transcendental ops are **bf16 only**. Only `divf`, `rsqrt`, and reciprocal are **f32 only**. There is no f32 `math.exp` on AIE2P -- not even as a scalar instruction.

### Impact on Transform Scripts

Triton-shared-opt produces Linalg IR with **f32 intermediates** for BF16 kernels (the Triton frontend promotes bf16 to f32 for operations like `tl.sigmoid`, `tl.exp`, and even `tl.maximum(x, 0.0)` where the `0.0` literal is f32). After `fuse_elementwise_linalg` merges the extf/compute/truncf chain, the fused generic body still contains f32 operations internally.

After `herd_vectorize`, these become `vector<16xf32>` operations which fail to legalize on AIE2P for most op types. The fix is `transform.air.vector_type_cast` applied selectively after vectorization:

```mlir
// Cast bf16-only ops from f32 to bf16 after vectorization
%vector_exps = transform.structured.match ops{["math.exp"]} in %vectorized_herd
%exp_cast = transform.air.vector_type_cast %vector_exps {target_element_type = bf16}

%vector_subs = transform.structured.match ops{["arith.subf"]} in %vectorized_herd
%sub_cast = transform.air.vector_type_cast %vector_subs {target_element_type = bf16}

%vector_adds = transform.structured.match ops{["arith.addf"]} in %vectorized_herd
%add_cast = transform.air.vector_type_cast %vector_adds {target_element_type = bf16}

%vector_muls = transform.structured.match ops{["arith.mulf"]} in %vectorized_herd
%mul_cast = transform.air.vector_type_cast %vector_muls {target_element_type = bf16}

// arith.divf stays f32 -- AIE2P has native f32 vector division
```

For `arith.cmpf` (used in leaky_relu), only the **inputs** should be cast to bf16; the result must stay `i1`:
```mlir
%vector_cmps = transform.structured.match ops{["arith.cmpf"]} in %vectorized_herd
%cmp_cast = transform.air.vector_type_cast %vector_cmps
    {target_element_type = bf16, input_indices = [0, 1]}

%vector_selects = transform.structured.match ops{["arith.select"]} in %vectorized_herd
%sel_cast = transform.air.vector_type_cast %vector_selects
    {target_element_type = bf16, input_indices = [1, 2], output_indices = [0]}
```

### Failure Modes

If the type constraints are violated, failures occur at different pipeline stages:

| Violation | Error | Stage |
|-----------|-------|-------|
| f32 `math.exp` | `unable to legalize instruction: G_FEXP` | llc (LLVM codegen) |
| f32 `arith.maximumf` | `aievec.max conversion fails due to unsupported element data type` | aie-opt (convert-aievec-to-llvm) |
| f32 `arith.cmpf` | `failed to legalize operation 'aievec.cmp'` | aie-opt |
| f32 `arith.mulf` | `failed to legalize operation 'arith.mulf'` | aie-opt |
| bf16 `arith.divf` | `unable to legalize instruction: G_FDIV <16 x s16>` | llc |
| Too many L1 `memref.alloc` inside herd | `undefined symbol: __air_herd_arg_N` | ld.lld (linker) |

## Elementwise Activation Function Transform Pattern

This section documents the proven transform pattern for elementwise activation functions (relu, sigmoid, silu, gelu, leaky_relu), verified on NPU2 (Strix/AIE2P) hardware.

### The Problem

Triton kernels for activation functions produce Linalg IR with this structure after `triton-shared-opt`:

```
linalg.fill (zero/one constants in f32)
linalg.generic: extf (bf16 -> f32)
linalg.generic: compute_op (f32)     -- e.g., maxnumf, negf+exp, etc.
linalg.generic: truncf (f32 -> bf16)
```

The multiple linalg.generic ops create intermediate f32 buffers. If each gets its own L1 allocation via `bufferize_to_allocation`, the AIE lowering fails with `__air_herd_arg_N` linker errors (too many L1 allocs inside the herd body).

### The Solution

1. **`fuse_elementwise_linalg`** merges the extf + compute + truncf chain into a single `linalg.generic` with bf16 input and bf16 output. Intermediate f32 computation stays inside the generic body but no separate buffers are allocated.

2. **Vec-add-style tiling**: flatten -> bufferize result to L2 -> tile_using_forall [256] -> pad -> promote input/output to L1 -> bufferize -> vectorize at 16-lane.

3. **Selective `vector_type_cast`** after `herd_vectorize`: cast each bf16-only op from f32 to bf16 (exp, addf, subf, mulf, maxnumf), keep f32-only ops as f32 (divf).

### Worked Example: Sigmoid

```
Triton kernel: sigmoid(x) = 1/(1+exp(-x))

Linalg IR (6 generics):
  extf -> subf(0,x) -> exp -> addf(exp,1) -> divf(1,sum) -> truncf

After fuse_elementwise_linalg (1 generic):
  bf16_input -> extf -> subf -> exp -> addf -> divf -> truncf -> bf16_output

After vectorize + vector_type_cast:
  transfer_read bf16 -> extf -> [subf bf16] -> [exp bf16] -> [addf bf16] -> divf f32 -> truncf -> transfer_write bf16
```

The resulting AIR has 2 L1 buffers (input bf16, output bf16) and a single vectorized loop with DMA in/out.

### Transform Script Template

The common 10-phase structure for elementwise ops:

| Phase | Purpose | Key Transform Ops |
|-------|---------|-------------------|
| 1 | Canonicalization | `apply_patterns`, `apply_cse` |
| 2 | Fuse elementwise chain | `air.fuse_elementwise_linalg` |
| 3 | Tile for multi-core | `flatten_elementwise`, `bufferize_to_allocation {memory_space=1}`, `tile_using_forall [256]` |
| 4 | Canonicalization | `apply_patterns`, `apply_cse` |
| 5 | Pad and promote to L1 | `structured.pad`, `bufferize_to_allocation {memory_space=2}` |
| 6 | Canonicalization | `apply_patterns`, `apply_cse` |
| 7 | Bufferization | `one_shot_bufferize` |
| 8 | Post-bufferization cleanup | `remove_uninitialized_copy`, `eliminate_cascade_memcpy` |
| 9 | Vectorization tiling | `tile_using_for [16]` |
| 10 | AIR mapping + type casts | `par_to_herd`, `copy_to_dma`, `herd_vectorize`, `vector_type_cast` per op |

### Op-Specific Casts Needed

| Kernel | Ops requiring bf16 cast | Ops staying f32 |
|--------|------------------------|-----------------|
| relu | `arith.maxnumf` | -- |
| sigmoid | `math.exp`, `arith.subf`, `arith.addf`, `arith.mulf` | `arith.divf` |
| silu | same as sigmoid + `arith.mulf` (x * sigmoid) | `arith.divf` |
| gelu | same as sigmoid + `arith.mulf` | `arith.divf` |
| leaky_relu | `arith.maxnumf`, `arith.cmpf` (inputs only), `arith.select` (values only) | -- |

### Triton Frontend Considerations

- `tl.sigmoid()` requires f32 input; bf16 raises `ValueError`. Cast explicitly: `x_f32 = x.to(tl.float32)`.
- `tl.exp()` also requires f32. Cast before calling.
- `tl.maximum(x, 0.0)` promotes to f32 because `0.0` is an f32 literal.
- `tl.math.tanh` does not exist in the current Triton version. Use `tl.sigmoid`-based formulations instead.
- After `.to(x.dtype)` to truncate back to bf16, the Linalg IR will contain the extf/truncf wrappers that `fuse_elementwise_linalg` merges.

### mlir-air Programming Examples as Reference

The `mlir-air/programming_examples/primitives/` directory contains standalone tests for each vector/scalar operation on AIE2/AIE2P hardware. These are the ground truth for type support:

- `vector_examples/vector_exp/` -- bf16 exp, vec16/32 (AIE2P hardware intrinsic)
- `vector_examples/vector_tanh/` -- bf16 tanh, AIE2P-only (`__builtin_aie2p_tanh`)
- `vector_examples/vector_div/` -- f32 div, vec16/64 (AIE2P only, NOT AIE2)
- `vector_examples/vector_max/` -- bf16 max, vec16/32
- `vector_examples/vector_select/` -- bf16 cmpf+select, vec16/32
- `vector_examples/vector_rsqrt/` -- f32 rsqrt, vec16/32
- `vector_examples/vector_reciprocal/` -- f32 reciprocal (1/x), vec16/32

The full-kernel examples (`programming_examples/relu/`, `sigmoid/`, `silu/`, `gelu/`, `softmax/`, `layer_norm/`, etc.) show the target AIR structure for each operation type and can be used to validate the Triton-XDNA AIR output via `AMD_TRITON_NPU_AIR_PROJECT_PATH`.
