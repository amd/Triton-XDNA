# Gemma4-E2B flash: an image turn -- vision tower, prefill and decode, each on
# the NPU or the iGPU, sharing one KV cache in DDR.
#
#   make run             Describe the shapes image  <-- start here
#   make verify          The same, gated on the description
#   make prefill         Vision + prefill only (no decode build needed)
#   make vision          Gate a vision tower against transformers
#   make compile-decode  One-time: build the decode (../gemma4_e2b_q4nx's)
#
# Run these two lines FIRST, from the repo root. A virtualenv is required, and
# the iGPU stages need a ROCm build of torch in it:
#
#   source sandbox/bin/activate
#   source /opt/xilinx/xrt/setup.sh && source utils/env_setup.sh

TEXT  := gemma4_e2b_q4nx
TITLE := Gemma4-E2B

include $(dir $(lastword $(MAKEFILE_LIST)))../llm_q4nx/gemma4_flash.mk
