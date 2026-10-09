# Qwen2.5-7B Q4NX on the NPU: a Triton prefill feeding mlir-air's fused decode.
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
#
# Unlike every other example here, the weights are NOT a Q4NX bundle: there is
# no FastFlowLM Qwen2.5-7B NPU2 bundle, so an upstream HF checkpoint is rounded
# onto the Q4NX grid at load time. That costs both time and host memory --
# `registry.py` records the measured requirement, and the example declines
# (exit 77) rather than being SIGKILLed partway through the load.

MODEL_NAME := qwen2.5-7b
TITLE      := Qwen2.5-7B
PROMPT     ?= The capital of France is

include $(dir $(lastword $(MAKEFILE_LIST)))../llm_q4nx/q4nx.mk
