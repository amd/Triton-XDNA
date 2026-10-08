# Llama-3.1-8B Q4NX on the NPU: a Triton prefill feeding mlir-air's fused decode.
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

MODEL_NAME := llama-3.1-8b
TITLE      := Llama-3.1-8B
# This model's own gate prompt, not the siblings'. The 1B and 3B use "The
# capital of France is"; on the 8B that ranks " a" above " Paris", which is
# why config.PROMPT differs -- so `make run` has to differ with it.
PROMPT     ?= The capital city of France is called

include $(dir $(lastword $(MAKEFILE_LIST)))../llm_q4nx/q4nx.mk
