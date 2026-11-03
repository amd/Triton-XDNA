# Gemma4-E2B "flash" -- an IMAGE through a Triton vision tower and prefill,
# feeding mlir-air's fused Q4NX decode.
#
#   make vision          Gate the tower against transformers  <-- start here
#   make prefill         Image + prompt, prefill only (no decode build)
#   make run             Image + prompt, N_TOKENS of caption
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
#
# The LANGUAGE half is ../gemma4_e2b_q4nx's, unchanged -- same checkpoint, same
# decode, same `registry.spec`. Only the image path is new, so there is no
# second decode build here: `compile-decode` writes the same artifacts that
# example's does, and either directory's target serves both.

PYTHON  ?= python3
srcdir  := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
DRIVER  := $(srcdir)/gemma4_e2b_flash_inference.py
shared  := $(srcdir)/../llm_q4nx
MODEL_NAME := gemma4-e2b
root    := $(realpath $(srcdir)/../..)

RUNTIME  ?= xrt
export AMD_TRITON_NPU_RUNTIME = $(RUNTIME)
DECODE_FORMAT := $(if $(filter hsa,$(RUNTIME)),pdi,xclbin)

# 512, where the text example takes 128. An image is 256 soft tokens before a
# word of prompt, so a template at 128 cannot hold one -- this is the floor for
# this path, not a preference.
DECODE_L ?= 512
MAX_SEQ  ?= 512

OPS      ?=
BACKEND  ?= npu
# triton, or air-fused for mlir-air's fused NPU prefill (vision on the iGPU).
ENGINE   ?= triton
N_TOKENS ?= 48
MODEL    ?=

# The two-shape test image: a red square upper-left, a blue circle lower-right
# on a light background. Unambiguous, needs no dataset, and a correct answer
# names both shapes, both colours and both positions -- which is a far stronger
# signal than any text prompt, because a broken image path produces a fluent
# wrong description rather than obvious garbage.
IMAGE  ?= /tmp/shapes.png
PROMPT ?= Describe the shapes and their colors and positions.

# Image resolution budget: 70, 140, 280, 560, 1120. Empty takes the
# checkpoint's own `default_output_length` (280), which on a 448x448 input is
# 2520 patches and a 276-token prompt.
#
# FastFlowLM ships the **70** tier -- measured, not assumed: its prefill log
# reads "1/1 with 84 tokens" on this image, and 70 is the only budget that
# reproduces that. Use SOFT=70 for any comparison against `flm run`, or the
# two sides are not running the same workload: 280 is 4x the patches and,
# because the tower's attention is quadratic, ~16x the attention.
SOFT   ?=

MODEL_ARG := $(if $(MODEL),--model "$(MODEL)",)
OPS_ARG   := $(if $(OPS),--ops $(OPS),)
SOFT_ARG  := $(if $(SOFT),--max-soft-tokens $(SOFT),)
COMMON    := --backend $(BACKEND) $(OPS_ARG) $(MODEL_ARG) $(SOFT_ARG)

.PHONY: help vision run prefill test-image compile-decode \
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
	@$(PYTHON) -c "import PIL, transformers" 2>/dev/null || { \
	  echo "ERROR: the image path needs pillow and transformers."; \
	  echo "  With the venv ACTIVE, run:  pip install pillow transformers"; \
	  exit 1; }
	@test -n "$$XILINX_XRT" || { \
	  echo "ERROR: XRT is not set up.  Run: source /opt/xilinx/xrt/setup.sh"; exit 1; }
	@test -n "$$MLIR_AIE_INSTALL_DIR" || { \
	  echo "ERROR: MLIR_AIE_INSTALL_DIR is unset.  Run: source $(root)/utils/env_setup.sh"; exit 1; }

help:
	@echo "Gemma4-E2B flash -- Triton vision tower + prefill, mlir-air decode"
	@echo ""
	@echo "  make vision           Gate the tower against transformers, per stage"
	@echo "  make prefill          Image + prompt, prefill only (no decode build)"
	@echo "  make run              Image + prompt, N_TOKENS=$(N_TOKENS) of caption"
	@echo "  make test-image       Write the two-shape test image to $(IMAGE)"
	@echo "  make compile-decode   Build the decode at DECODE_L=$(DECODE_L) (run once)"
	@echo "  make preflight        Check the environment is set up"
	@echo ""
	@echo "  BACKEND=npu|cpu       Where the tower and prefill run (default: npu)"
	@echo "  OPS=all|matmul,...    Which operators go to the NPU"
	@echo "                        DO NOT use OPS=all: the per-head norm is a"
	@echo "                        [30240, 64] problem and the kernel times out"
	@echo "  SOFT=70|140|280|...   Image resolution budget (default: 280)"
	@echo "                        FLM ships 70; use it for any A/B vs flm"
	@echo "  IMAGE=path            The image (default: $(IMAGE))"
	@echo "  PROMPT=\"...\"          The question about it"

## The two-shape test image. No dataset, no download.
test-image:
	@$(PYTHON) -c "from PIL import Image, ImageDraw; \
im = Image.new('RGB', (448, 448), (245, 245, 245)); \
d = ImageDraw.Draw(im); \
d.rectangle([60, 60, 200, 200], fill=(220, 30, 30)); \
d.ellipse([250, 250, 400, 400], fill=(30, 80, 220)); \
im.save('$(IMAGE)')"
	@echo "wrote $(IMAGE)"

## The correctness gate: our tower against transformers' own, stage by stage.
##
## Needs no decode build and no NPU (BACKEND=cpu runs it all in torch). This is
## what to run after touching anything in ../llm_q4nx/gemma4_vision*.py -- the
## end-to-end caption cannot tell you WHICH stage broke, and this can.
vision: preflight
	@cd $(srcdir) && $(PYTHON) vision.py --image $(IMAGE) $(COMMON)

## Image + prompt through the prefill only. Needs no decode build.
prefill: preflight
	@cd $(srcdir) && $(PYTHON) $(DRIVER) --prefill-only $(COMMON) --prefill-engine $(ENGINE) \
		--image $(IMAGE) --text "$(PROMPT)" --max-seq $(MAX_SEQ)

## Image + prompt, all the way to a caption.
##
## Not gated on an exact string: generation is not deterministic here, and the
## same weights give different wording run to run. Judge it by eye -- both
## shapes, both colours, both positions. Mixed-script or incoherent output is a
## correctness failure; different-but-fluent is not.
run: preflight
	@cd $(srcdir) && $(PYTHON) $(DRIVER) $(COMMON) --prefill-engine $(ENGINE) \
		--image $(IMAGE) --text "$(PROMPT)" \
		--max-seq $(MAX_SEQ) --max-tokens $(N_TOKENS)

## Everything the decode needs, from nothing. Shares its artifacts with
## ../gemma4_e2b_q4nx -- but at DECODE_L=$(DECODE_L), which that example's
## default of 128 is too short for an image prompt.
compile-decode: preflight air-src
	@cd $(shared) && $(PYTHON) decode_kernels.py --model $(MODEL_NAME)
	@cd $(shared) && $(PYTHON) decode_build.py --model $(MODEL_NAME) --format $(DECODE_FORMAT) --max-context-length $(DECODE_L)

## Just relower the xclbins, reusing the AIE kernels on disk.
recompile-decode: preflight air-src
	@cd $(shared) && $(PYTHON) decode_build.py --model $(MODEL_NAME) --format $(DECODE_FORMAT) --max-context-length $(DECODE_L)

## Just rebuild the AIE kernels. Only needed if their C++ changed.
decode-kernels: preflight air-src
	@cd $(shared) && $(PYTHON) decode_kernels.py --model $(MODEL_NAME) --force

## mlir-air's sources at the pinned commit.
air-src:
	@$(PYTHON) $(srcdir)/../../utils/fetch_mlir_air_src.py

clean:
	rm -rf $(srcdir)/air_project $(srcdir)/__pycache__ $(srcdir)/params.txt
