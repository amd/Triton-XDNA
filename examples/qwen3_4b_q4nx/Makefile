# Qwen3-4B Q4NX on the NPU: a Triton prefill feeding mlir-air's fused decode.
#
#   make run             One prompt, N_TOKENS of output  <-- start here
#   make prefill         Prefill only + the Paris gate (no decode build needed)
#   make profile         Per-operator prefill timing
#   make compile-decode  One-time: build the fused decode templates
#
# Run these two lines FIRST, from the repo root. A virtualenv is required --
# `utils/env_setup.sh` pip-installs into whatever Python is active, and the
# system one is externally managed (PEP 668), so it fails there:
#
#   source sandbox/bin/activate          # or however your venv is named
#   source /opt/xilinx/xrt/setup.sh && source utils/env_setup.sh
#
# `make preflight` checks all of this and says what is missing.

PYTHON  ?= python3
srcdir  := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
DRIVER  := $(srcdir)/qwen3_4b_q4nx_inference.py
# The harness shared with the other Q4NX model families.
shared  := $(srcdir)/../llm_q4nx
# This example's model family, as ../llm_q4nx/registry.py names it.
MODEL_NAME := qwen3-4b

# Dispatch runtime. Only xrt here: the HSA decode adapter subclasses mlir-air's
# `FusedDecoder`, which this driver does name -- but nothing has run it on
# hardware for this model, so the harness refuses rather than pretending
# (registry.py: supports_hsa). Kept as a variable so enabling it is a one-line
# change once it is tested.
RUNTIME  ?= xrt
export AMD_TRITON_NPU_RUNTIME = $(RUNTIME)
DECODE_FORMAT := $(if $(filter hsa,$(RUNTIME)),pdi,xclbin)

OPS      ?= all
BACKEND  ?= npu
N_TOKENS ?= 200
PROMPT   ?= The capital of France is
MODEL    ?=

MODEL_ARG  := $(if $(MODEL),--model "$(MODEL)",)
COMMON     := --backend $(BACKEND) --ops $(OPS) $(MODEL_ARG)

.PHONY: help run prefill profile compile-decode \
	recompile-decode decode-kernels air-src preflight clean

## Fail early, and say what is actually wrong.
preflight:
	@$(PYTHON) -c "import triton" 2>/dev/null || { \
	  echo "ERROR: '$(PYTHON)' cannot import triton."; \
	  echo "  A virtualenv with triton-xdna installed must be active. From the repo root:"; \
	  echo "    source sandbox/bin/activate"; \
	  echo "    source /opt/xilinx/xrt/setup.sh && source ../../utils/env_setup.sh"; \
	  exit 1; }
	@$(PYTHON) -c "import mlir_air, mlir_aie" 2>/dev/null || { \
	  echo "ERROR: mlir_air / mlir_aie are not installed in this environment."; \
	  echo "  With the venv ACTIVE, run:  source ../../utils/env_setup.sh"; \
	  exit 1; }
	@test -n "$$XILINX_XRT" || { \
	  echo "ERROR: XRT is not set up.  Run: source /opt/xilinx/xrt/setup.sh"; exit 1; }
	@test -n "$$MLIR_AIE_INSTALL_DIR" || { \
	  echo "ERROR: MLIR_AIE_INSTALL_DIR is unset.  Run: source ../../utils/env_setup.sh"; exit 1; }

help:
	@echo "Qwen3-4B Q4NX -- Triton prefill + mlir-air fused decode"
	@echo ""
	@echo "  make run              One prompt: PROMPT=\"...\" N_TOKENS=$(N_TOKENS)"
	@echo "  make prefill          Prefill only + Paris gate (no decode build)"
	@echo "  make profile          Per-operator prefill timing"
	@echo "  make compile-decode   Build the decode from nothing (~22 s, run once)"
	@echo "  make recompile-decode Relower just the decode xclbins (~9 s)"
	@echo "  make decode-kernels   Rebuild just the decode's AIE kernels (~13 s)"
	@echo "  make air-src          Fetch mlir-air sources at the pinned commit"
	@echo "  make preflight        Check the environment is set up"
	@echo ""
	@echo "  BACKEND=npu|cpu       Where the prefill runs (default: npu)"
	@echo "  OPS=all|matmul,...    Which operators go to the NPU (default: all)"

# No `chat` target. mlir-air's Session/interactive_chat pair exists only on its
# npz-API drivers; Qwen3-4B's has neither, so there is nothing to host our
# prefill in and the harness refuses --interactive up front. A target that can
# only fail is worse than no target.

## One prompt, greedy, N_TOKENS of output.
run: preflight
	@cd $(srcdir) && $(PYTHON) $(DRIVER) $(COMMON) \
		--text "$(PROMPT)" --max-tokens $(N_TOKENS) --greedy

## Prefill on its own -- needs no decode build. Gates on first token == 12095.
prefill: preflight
	@cd $(srcdir) && $(PYTHON) $(DRIVER) --prefill-only $(COMMON)

## Per-operator timing for one prefill.
profile: preflight
	@cd $(srcdir) && $(PYTHON) $(DRIVER) --prefill-only --profile $(COMMON)

## Everything the decode needs, from nothing. Run once, before `run`.
##
## Two steps, both owned here:
##
##   1. ../llm_q4nx/decode_kernels.py -- the six AIE kernels, compiled with Peano (~13 s).
##      Byte-identical to what mlir-air's Makefile produces, and cached on the
##      source and flags, so a re-run is free.
##   2. ../llm_q4nx/decode_build.py -- the xclbins, lowered through FusedDecodeOp: mlir-air's
##      build_module() supplies the AIR IR, this backend's _aircc_compile turns
##      it into an artifact exactly as for every other AIR design (~9 s).
##
## This used to delegate to mlir-air's `compile-decode`, which took ~15 min --
## almost all of it building templates we then relowered anyway. What is needed
## from mlir-air now is SOURCE, not its build system.
compile-decode: preflight air-src
	@cd $(shared) && $(PYTHON) decode_kernels.py --model $(MODEL_NAME)
	@cd $(shared) && $(PYTHON) decode_build.py --model $(MODEL_NAME) --format $(DECODE_FORMAT)

## Just step 2 -- relower the xclbins from the AIR IR (~9 s), reusing the AIE
## kernels already on disk. The one to run while iterating on the builder or
## the lowering.
recompile-decode: preflight air-src
	@cd $(shared) && $(PYTHON) decode_build.py --model $(MODEL_NAME) --format $(DECODE_FORMAT)

## Just step 1 -- rebuild the AIE kernels. Only needed if their C++ changed.
decode-kernels: preflight air-src
	@cd $(shared) && $(PYTHON) decode_kernels.py --model $(MODEL_NAME) --force

## mlir-air's sources at the commit utils/mlir-air-hash.txt already pins -- the
## same one env_setup.sh builds the wheel version from. Sparse and blobless
## (~1.7 MB of sources), idempotent, and skipped entirely if you have your own
## checkout at mlir-air-local/ or AIR_LLMS_ROOT.
air-src:
	@$(PYTHON) $(srcdir)/../../utils/fetch_mlir_air_src.py

clean:
	rm -rf $(srcdir)/air_project $(srcdir)/__pycache__ $(srcdir)/params.txt
