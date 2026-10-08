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
# ATTN_MAXL the decode template is built for. 128, where the siblings take
# decode_build.py's 2048 default: that is what mlir-air's own token driver
# builds this model at (RUN_LBUILD), and every token streams a cache padded to
# it, so a 35-layer model pays for the depth on every dispatch. Raise it for
# longer sessions.
DECODE_L   ?= 128
# EMPTY by default, and that is load-bearing: an empty PROMPT runs the
# canonical token IDS below, which is what lets `make run` gate the decode.
# This tokenizer emits no <bos>, so `--text` of the same sentence gives ids
# starting at 818 where `config.PROMPT` starts at 2, and the harness applies
# the recorded continuation only when the ids match exactly. Setting PROMPT
# sends text, as on every sibling, and correctly does not gate.
PROMPT     ?=
# `config.PROMPT`, as ids. Keep the two in step.
PROMPT_IDS ?= 2,818,5279,529,7001,563

# No `chat` target. mlir-air's Session/interactive_chat pair exists only on its
# npz-API drivers; Gemma4-E2B's has neither, so there is nothing to host our
# prefill in and the harness refuses --interactive up front.

include $(dir $(lastword $(MAKEFILE_LIST)))../llm_q4nx/q4nx.mk
