# Shared Makefile for the Gemma4 image-turn examples (gemma4_*_flash).
#
# The including Makefile sets:
#   TEXT  := gemma4_e2b_q4nx        # the text example whose model this is
#   TITLE := Gemma4-E2B
#
# Each stage of an image turn is placed on its own:
#   VISION=auto|air|triton|igpu|host  ENGINE=air-fused|triton-fused|triton|igpu
#   DECODE=npu|gpu
#   TAIL=npu|igpu  (ENGINE=triton-fused: the layers that write no KV cache, and
#                   the head, for the prompt's last row)
# Whatever the placement, the prefill writes K/V into the slab the decode
# reads, in pages both the NPU and the iGPU map.

PYTHON  ?= python3
srcdir  := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
DRIVER  := $(srcdir)/$(notdir $(srcdir))_inference.py
shared  := $(realpath $(srcdir)/../llm_q4nx)
textdir := $(realpath $(srcdir)/../$(TEXT))
root    := $(realpath $(srcdir)/../..)

# The decode is the text example's build, made long enough for an image
# prompt: an image at the default budget is 256 soft tokens before any text.
DECODE_L ?= 512
MAX_SEQ  ?= $(DECODE_L)

VISION   ?= auto
ENGINE   ?= air-fused
DECODE   ?= npu
TAIL     ?= npu
## The image resolution budget (70, 140, 280, 560, 1120); empty = 280.
SOFT     ?=
N_TOKENS ?= 100
## Timed vision + prefill turns after a warm-up, for TTFT.
REPEAT   ?= 1
BACKEND  ?=
OPS      ?=
MODEL    ?=
IMAGE    ?= /tmp/shapes.png
PROMPT   ?= Describe the shapes and their colors and positions.
## `make vision`'s tower: host (scored per stage), air, triton or igpu.
TOWER    ?= host

OPT_ARGS := $(if $(SOFT),--max-soft-tokens $(SOFT)) $(if $(MODEL),--model "$(MODEL)") \
            $(if $(BACKEND),--backend $(BACKEND)) $(if $(OPS),--ops $(OPS))
RUN_ARGS := --image $(IMAGE) --text "$(PROMPT)" --vision $(VISION) \
            --prefill-engine $(ENGINE) --prefill-tail $(TAIL) --decode $(DECODE) \
            --max-seq $(MAX_SEQ) \
            --repeat $(REPEAT) $(OPT_ARGS)

.PHONY: help preflight test-image vision prefill run verify compile-decode \
        recompile-decode air-src clean

help:
	@echo "$(TITLE) flash -- an image turn: vision tower, prefill and decode"
	@echo ""
	@echo "  make run              Describe IMAGE, N_TOKENS=$(N_TOKENS) tokens"
	@echo "  make verify           run, gated on the shapes image's description"
	@echo "  make prefill          Vision + prefill only (no decode build)"
	@echo "  make vision           Gate a vision TOWER against transformers"
	@echo "  make compile-decode   Build the decode at DECODE_L=$(DECODE_L) (once)"
	@echo "  make test-image       Write the two-shape test image to $(IMAGE)"
	@echo ""
	@echo "  VISION=auto|air|triton|igpu|host     (default: $(VISION))"
	@echo "  ENGINE=air-fused|triton-fused|triton|igpu  prefill (default: $(ENGINE))"
	@echo "  DECODE=npu|gpu                       (default: $(DECODE))"
	@echo "  TAIL=npu|igpu                        triton-fused prefill tail (default: $(TAIL))"
	@echo "  SOFT=70|140|280|560|1120             image budget (default: 280)"
	@echo "  REPEAT=n                             median TTFT over n turns"
	@echo "  IMAGE=path PROMPT=\"...\""

preflight:
	@$(MAKE) --no-print-directory -C $(textdir) preflight
	@$(PYTHON) -c "import PIL, transformers" 2>/dev/null || { \
	  echo "ERROR: an image turn needs pillow and transformers."; \
	  echo "  With the venv active: pip install pillow transformers"; \
	  exit 1; }

test-image:
	@$(PYTHON) -c "import sys; sys.path.insert(0, '$(shared)'); \
import gemma4_image; print('wrote', gemma4_image.shapes_image('$(IMAGE)'))"

$(IMAGE):
	@$(MAKE) --no-print-directory test-image

# Without MODEL, the gate is given this example's default model rather than
# the loader's.
vision: preflight $(IMAGE)
	@cd $(srcdir) && $(PYTHON) $(shared)/gemma4_vision_gate.py --image $(IMAGE) \
		--tower $(TOWER) --repeat $(REPEAT) $(OPT_ARGS) \
		$(if $(MODEL),,--model "$$($(PYTHON) -c 'import config; print(config.MODEL_DEFAULT)')")

prefill: preflight $(IMAGE)
	@cd $(srcdir) && $(PYTHON) $(DRIVER) --prefill-only $(RUN_ARGS)

run: preflight $(IMAGE)
	@cd $(srcdir) && $(PYTHON) $(DRIVER) $(RUN_ARGS) --max-tokens $(N_TOKENS)

verify: preflight $(IMAGE)
	@cd $(srcdir) && $(PYTHON) $(DRIVER) $(RUN_ARGS) --max-tokens $(N_TOKENS) --gate

compile-decode recompile-decode air-src:
	@$(MAKE) --no-print-directory -C $(textdir) $@ DECODE_L=$(DECODE_L)

clean:
	rm -rf $(srcdir)/air_project $(srcdir)/__pycache__
