# Included by each Q4NX example's Makefile, which sets these first:
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
shared  := $(realpath $(srcdir)/../llm_q4nx)
root    := $(realpath $(srcdir)/../..)

# Dispatch runtime: xrt, or hsa for models with `supports_hsa` in
# registry.py. Under hsa the decode is a full ELF if the loaded ROCR can run it
# (hsa_decode.use_scratchpad), and the PDI template pair otherwise. Deferred, so
# only a decode build pays for asking.
RUNTIME  ?= xrt
export AMD_TRITON_NPU_RUNTIME = $(RUNTIME)
HSA_DECODE_FORMAT = $(shell cd $(shared) && $(PYTHON) -c "import hsa_decode, registry; \
  print('elf' if hsa_decode.use_scratchpad(registry.spec('$(MODEL_NAME)').engine) else 'pdi')" \
  2>/dev/null || echo pdi)
DECODE_FORMAT = $(if $(filter hsa,$(RUNTIME)),$(HSA_DECODE_FORMAT),xclbin)

# Prefill engine. triton: this repository's prefill, placed by BACKEND and
# OPS. auto: mlir-air's fused prefill where the model has one (BACKEND and OPS
# are then not passed).
ENGINE   ?= triton
BACKEND  ?= npu
# Operators to run on the NPU: 'all', or a comma list of the model's NPU_OPS.
# Empty uses the model's DEFAULT_OPS.
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
# With PROMPT_IDS set, an empty PROMPT runs those ids. The harness only gates
# the decode when the ids equal `config.PROMPT`, which text does not reproduce
# for every tokenizer.
RUN_PROMPT := $(if $(PROMPT),--text "$(PROMPT)",$(if $(PROMPT_IDS),--prompt "$(PROMPT_IDS)"))

.PHONY: help $(if $(CHAT),chat) run prefill profile compile-decode \
	recompile-decode decode-kernels air-src preflight clean

## Check the environment and say what is missing.
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

## Build the decode from scratch; run once before `run`. Two steps:
##   1. decode_kernels.py: the AIE kernels the decode links, compiled with
##      Peano and cached on their sources and flags.
##   2. decode_build.py: the decode templates, lowered from mlir-air's
##      build_module() through this backend.
compile-decode: preflight air-src
	@cd $(shared) && $(PYTHON) decode_kernels.py --model $(MODEL_NAME)
	@cd $(shared) && $(PYTHON) decode_build.py --model $(MODEL_NAME) --format $(DECODE_FORMAT) $(DECODE_L_ARG)

## Step 2 only, reusing the AIE kernels already built.
recompile-decode: preflight air-src
	@cd $(shared) && $(PYTHON) decode_build.py --model $(MODEL_NAME) --format $(DECODE_FORMAT) $(DECODE_L_ARG)

## Step 1 only, forcing a rebuild of the AIE kernels.
decode-kernels: preflight air-src
	@cd $(shared) && $(PYTHON) decode_kernels.py --model $(MODEL_NAME) --force

## mlir-air's sources at the commit pinned in utils/mlir-air-hash.txt.
## Skipped when mlir-air-local/ or AIR_LLMS_ROOT provides them.
air-src:
	@$(PYTHON) $(root)/utils/fetch_mlir_air_src.py

clean:
	rm -rf $(srcdir)/air_project $(srcdir)/__pycache__ $(srcdir)/params.txt
