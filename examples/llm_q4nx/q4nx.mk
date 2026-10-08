# The Makefile every Q4NX example includes. A model's own Makefile sets what is
# its alone, then includes this:
#
#   MODEL_NAME := llama-3.2-1b     # as ../llm_q4nx/registry.py names it
#   TITLE      := Llama-3.2-1B     # for `make help`
#   PROMPT     ?= ...              # `make run`'s default question (text)
#   N_TOKENS   ?= 200
#   PROMPT_IDS ?= 2,818,...        # optional: run these ids when PROMPT is empty
#   DECODE_L   ?= 128              # optional: the decode's ATTN_MAXL
#   CHAT       := 1                # optional: offer `make chat`
#   include $(dir $(lastword $(MAKEFILE_LIST)))../llm_q4nx/q4nx.mk
#
# The driver is `<example dir>_inference.py`, beside that Makefile.

PYTHON  ?= python3
srcdir  := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
DRIVER  := $(srcdir)/$(notdir $(srcdir))_inference.py
# The harness shared by every Q4NX model family; this file lives in it.
shared  := $(realpath $(srcdir)/../llm_q4nx)
# Absolute, so the remediation messages in `preflight` name a path that works
# from wherever make was invoked.
root    := $(realpath $(srcdir)/../..)

# Dispatch runtime: xrt, or hsa where registry.py's `supports_hsa` says the HSA
# decode has been made to work for the model (the harness refuses it
# elsewhere). HSA runs both halves on the HSA runtime: the prefill's kernels
# as PDI, and the decode with its instruction stream patched per token
# (../llm_q4nx/hsa_decode.py), so the decode is built as PDI too.
RUNTIME  ?= xrt
export AMD_TRITON_NPU_RUNTIME = $(RUNTIME)
DECODE_FORMAT := $(if $(filter hsa,$(RUNTIME)),pdi,xclbin)

# The prefill engine. triton (default here): this repository's prefill, its
# operators placed by BACKEND and OPS. auto: the harness's own default --
# mlir-air's fused prefill where it has one, which BACKEND/OPS do not apply to.
ENGINE   ?= triton
BACKEND  ?= npu
# Operators to run on the NPU: 'all', or a comma list of the model's NPU_OPS.
# Empty takes the model's own default (its DEFAULT_OPS), as the CLI does.
OPS      ?=
N_TOKENS ?= 200
MODEL    ?=
SYSTEM   ?=

MODEL_ARG  := $(if $(MODEL),--model "$(MODEL)",)
SYSTEM_ARG := $(if $(SYSTEM),--system "$(SYSTEM)",)
COMMON     := --prefill-engine $(ENGINE) \
	$(if $(filter triton,$(ENGINE)),--backend $(BACKEND) $(if $(OPS),--ops $(OPS))) \
	$(MODEL_ARG)
DECODE_L_ARG := $(if $(DECODE_L),--max-context-length $(DECODE_L))
# With PROMPT_IDS, an empty PROMPT runs those ids -- `config.PROMPT`, which is
# what the harness gates the decode on. A model whose tokenizer does not
# reproduce `config.PROMPT` from text (Gemma4 emits no <bos>) needs that, or
# `make run` would pass by skipping its own check.
RUN_PROMPT := $(if $(PROMPT),--text "$(PROMPT)",$(if $(PROMPT_IDS),--prompt "$(PROMPT_IDS)"))

.PHONY: help $(if $(CHAT),chat) run prefill profile compile-decode \
	recompile-decode decode-kernels air-src preflight clean

## Fail early, and say what is actually wrong.
preflight:
	@$(PYTHON) -c "import triton" 2>/dev/null || { \
	  echo "ERROR: '$(PYTHON)' cannot import triton."; \
	  echo "  A virtualenv with triton-xdna installed must be active:"; \
	  echo "    source $(root)/sandbox/bin/activate"; \
	  echo "    source /opt/xilinx/xrt/setup.sh && source $(root)/utils/env_setup.sh"; \
	  exit 1; }
	@$(PYTHON) -c "import mlir_air, mlir_aie" 2>/dev/null || { \
	  echo "ERROR: mlir_air / mlir_aie are not installed in this environment."; \
	  echo "  With the venv ACTIVE, run:  source $(root)/utils/env_setup.sh"; \
	  exit 1; }
	@test -n "$$XILINX_XRT" || { \
	  echo "ERROR: XRT is not set up.  Run: source /opt/xilinx/xrt/setup.sh"; exit 1; }
	@test -n "$$MLIR_AIE_INSTALL_DIR" || { \
	  echo "ERROR: MLIR_AIE_INSTALL_DIR is unset.  Run: source $(root)/utils/env_setup.sh"; exit 1; }

help:
	@echo "$(TITLE) Q4NX -- Triton prefill + mlir-air fused decode"
	@echo ""
	$(if $(CHAT),@echo "  make chat             Interactive REPL (/clear resets, /exit quits)")
	@echo "  make run              One prompt, N_TOKENS=$(N_TOKENS)$(if $(PROMPT_IDS), (empty PROMPT: the gated ids))"
	@echo "  make prefill          Prefill only + Paris gate (no decode build)"
	@echo "  make profile          Per-operator prefill timing"
	@echo "  make compile-decode   Build the decode from nothing (run once)"
	@echo "  make recompile-decode Relower just the decode templates"
	@echo "  make decode-kernels   Rebuild just the decode's AIE kernels"
	@echo "  make air-src          Fetch mlir-air sources at the pinned commit"
	@echo "  make preflight        Check the environment is set up"
	@echo ""
	@echo "  RUNTIME=xrt|hsa       Dispatch runtime (default: xrt)"
	@echo "  ENGINE=triton|auto    Prefill engine (default: triton; auto = fused where available)"
	@echo "  BACKEND=npu|hetero|cpu Where the Triton prefill's operators run (default: npu)"
	@echo "  OPS=all|matmul,...    Which operators go to the NPU (default: the model's)"
	$(if $(CHAT),@echo "  SYSTEM=\"...\"          System prompt for chat")

ifneq ($(CHAT),)
## Interactive chat. Weights load once; every turn is prefill compute only.
chat: preflight
	@cd $(srcdir) && $(PYTHON) $(DRIVER) --interactive $(COMMON) $(SYSTEM_ARG)
endif

## One prompt, greedy, N_TOKENS of output.
run: preflight
	@cd $(srcdir) && $(PYTHON) $(DRIVER) $(COMMON) $(RUN_PROMPT) \
		--max-tokens $(N_TOKENS) --greedy

## Prefill on its own -- needs no decode build. Gates on config.EXPECT_FIRST.
prefill: preflight
	@cd $(srcdir) && $(PYTHON) $(DRIVER) --prefill-only $(COMMON)

## Per-operator timing for one prefill.
profile: preflight
	@cd $(srcdir) && $(PYTHON) $(DRIVER) --prefill-only --profile $(COMMON)

## Everything the decode needs, from nothing. Run once, before `run`.
##
## Two steps, both owned here:
##
##   1. ../llm_q4nx/decode_kernels.py -- the AIE kernels the model's decode
##      engine links, compiled with Peano. Byte-identical to what mlir-air's
##      Makefile produces, and cached on the source and flags, so a re-run is
##      free.
##   2. ../llm_q4nx/decode_build.py -- the templates, lowered through
##      FusedDecodeOp: mlir-air's build_module() supplies the AIR IR, this
##      backend's _aircc_compile turns it into an artifact exactly as for every
##      other AIR design.
##
## What is needed from mlir-air is SOURCE, not its build system.
compile-decode: preflight air-src
	@cd $(shared) && $(PYTHON) decode_kernels.py --model $(MODEL_NAME)
	@cd $(shared) && $(PYTHON) decode_build.py --model $(MODEL_NAME) --format $(DECODE_FORMAT) $(DECODE_L_ARG)

## Just step 2 -- relower the templates from the AIR IR, reusing the AIE
## kernels already on disk. The one to run while iterating on the builder or
## the lowering.
recompile-decode: preflight air-src
	@cd $(shared) && $(PYTHON) decode_build.py --model $(MODEL_NAME) --format $(DECODE_FORMAT) $(DECODE_L_ARG)

## Just step 1 -- rebuild the AIE kernels. Only needed if their C++ changed.
decode-kernels: preflight air-src
	@cd $(shared) && $(PYTHON) decode_kernels.py --model $(MODEL_NAME) --force

## mlir-air's sources at the commit utils/mlir-air-hash.txt already pins -- the
## same one env_setup.sh builds the wheel version from. Sparse and blobless
## (~1.7 MB of sources), idempotent, and skipped entirely if you have your own
## checkout at mlir-air-local/ or AIR_LLMS_ROOT.
air-src:
	@$(PYTHON) $(root)/utils/fetch_mlir_air_src.py

clean:
	rm -rf $(srcdir)/air_project $(srcdir)/__pycache__ $(srcdir)/params.txt
