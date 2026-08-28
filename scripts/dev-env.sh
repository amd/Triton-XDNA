#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Environment for running the heterogeneous (iGPU + NPU) examples from the
# dev venv built by scripts/setup-dev-venv.sh.  Source it, don't execute it.
#
#     source scripts/dev-env.sh
#     python examples/gpt2/gpt2_inference.py --backend hetero
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO/.venv}"

source "$VENV/bin/activate"
source /opt/xilinx/xrt/setup.sh >/dev/null

# Resolve mlir-aie the way utils/env_setup.sh does: the [aie] extra installs the
# no-RTTI wheel (dist name mlir_aie_no_rtti) and falls back to the RTTI one, but
# both unpack to <site-packages>/mlir_aie. Guessing the path breaks on the RTTI
# wheel, so ask pip.
if [[ -z "${MLIR_AIE_INSTALL_DIR:-}" ]]; then
    LOC="$(python -m pip show mlir_aie_no_rtti 2>/dev/null | awk '/^Location:/{print $2}')"
    [[ -n "$LOC" ]] || LOC="$(python -m pip show mlir_aie 2>/dev/null | awk '/^Location:/{print $2}')"
    [[ -n "$LOC" ]] || LOC="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    export MLIR_AIE_INSTALL_DIR="$LOC/mlir_aie"
fi
export PATH="$MLIR_AIE_INSTALL_DIR/bin:$PATH"
# ${VAR:+:$VAR} rather than :${VAR:-}: with the variable unset the latter leaves
# a trailing colon, and an empty entry means the current directory -- so every
# shell that sourced this would search $PWD for modules and shared objects.
export PYTHONPATH="$MLIR_AIE_INSTALL_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$MLIR_AIE_INSTALL_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# The JIT cache grows fast and ~/.triton is often a quota-limited network home,
# so default it next to the checkout. Override either var to move it.
export TRITON_HOME="${TRITON_HOME:-$REPO/.triton-home}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$TRITON_HOME/.triton/cache}"
export TMPDIR="${TMPDIR:-$TRITON_HOME/tmp}"
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"
