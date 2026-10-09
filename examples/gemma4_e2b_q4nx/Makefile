# Gemma4-E2B Q4NX on the NPU: a Triton prefill feeding mlir-air's fused decode.
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

MODEL_NAME := gemma4-e2b
TITLE      := Gemma4-E2B
N_TOKENS   ?= 4
# The decode template's ATTN_MAXL, matching mlir-air's own build of this
# model. Every token streams a cache padded to it; raise it for longer sessions.
DECODE_L   ?= 128
# Empty by default so `make run` sends PROMPT_IDS (`config.PROMPT`) and the
# decode is gated: this tokenizer emits no <bos>, so text never reproduces
# those ids. Setting PROMPT sends text, ungated.
PROMPT     ?=
PROMPT_IDS ?= 2,818,5279,529,7001,563

# No `chat` target: the harness only supports --interactive for npz-API
# drivers, and this model's is not one.

include $(dir $(lastword $(MAKEFILE_LIST)))../llm_q4nx/q4nx.mk
