#!/bin/bash
# Environment for the gemma4-hetero worktree.
#   source env.sh
#
# Unlike the sibling checkout, this venv holds triton-xdna built FROM THIS
# BRANCH's source (`pip install . --no-build-isolation`), not the prebuilt
# wheel. That matters: the wheel on the release index predates PR #133, so its
# `tl.extra.npu` has no DecodeEngine and scripts/test_tl_extra_npu.py fails 4
# of its 10 checks against it for reasons that have nothing to do with the
# working tree.

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${_here}/.venv/bin/activate"
source /opt/xilinx/xrt/setup.sh >/dev/null

# $HOME is an NFS mount with a 5 GB quota and every one of these defaults into
# it; an over-quota home fails unrelated writes with "Errno 122 / EDQUOT".
# Its own TRITON_HOME, so the source build's kernel cache never mixes with the
# wheel venv's.
export TRITON_HOME="/scratch/erweiw/triton-home-src"
export TRITON_CACHE_DIR="${TRITON_HOME}/.triton/cache"
export HF_HOME="/scratch/erweiw/hf-home"
export PIP_CACHE_DIR="/scratch/erweiw/pip-cache"
export XDG_CACHE_HOME="/scratch/erweiw/xdg-cache"
mkdir -p "${TRITON_CACHE_DIR}" "${HF_HOME}" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}"

# Exports only -- the stack is already installed, and letting this reinstall
# would pull whatever llvm-aie nightly is current today.
TRITON_XDNA_ENV_INSTALL=0 source "${_here}/utils/env_setup.sh"

# gfx1151 (Radeon 8060S), for the iGPU half when it lands.
export HSA_OVERRIDE_GFX_VERSION=11.5.1

echo "venv:      $(which python)"
echo "triton:    $(python -c 'import triton; print(triton.__version__)' 2>/dev/null) (source build)"
echo "MLIR-AIE:  ${MLIR_AIE_INSTALL_DIR}"
