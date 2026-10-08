# Qwen2.5-3B Q4NX on the NPU: a Triton prefill feeding mlir-air's fused decode.
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
# The default weight source IS a FastFlowLM model.q4nx bundle, unlike the 7B's;
# an HF checkpoint also works and is Q4_0-quantized on load, which is what the
# `_q4` in mlir-air's own directory name refers to. `registry.py` records the
# host memory the prefill needs, and the example declines (exit 77) rather than
# being SIGKILLed partway through the load.

MODEL_NAME := qwen2.5-3b
TITLE      := Qwen2.5-3B
PROMPT     ?= The capital of France is

include $(dir $(lastword $(MAKEFILE_LIST)))../llm_q4nx/q4nx.mk
