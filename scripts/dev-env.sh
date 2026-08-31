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

# Settle which ROCR the process gets, for the runs that use the NPU through HSA
# *and* the iGPU through torch: whichever ROCR loads first claims the SONAME for
# the whole process, and torch bundles one. See README, "Sharing a process with
# PyTorch". Set whenever a usable ROCR is installed, not only for the HSA runs:
# a pure-XRT run has no HSA runtime to bind, and having HIP use this ROCR rather
# than the one torch ships changes nothing it can observe.
#
# Asked of the backend rather than guessed, so this honors AMD_NPU_ROCR_PATH and
# ROCM_PATH and picks the same library the HSA runtime will be linked against --
# preloading a different one would be worse than preloading nothing.
ROCR_LIB="$(python -c 'from triton.backends.amd_triton_npu.driver import _get_rocr_install
print(_get_rocr_install().lib_path)' 2>/dev/null)"
if [[ -f "$ROCR_LIB" ]]; then
    export LD_PRELOAD="$ROCR_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
fi

# The JIT cache grows fast and ~/.triton is often a quota-limited network home,
# so default it next to the checkout. Override either var to move it.
export TRITON_HOME="${TRITON_HOME:-$REPO/.triton-home}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$TRITON_HOME/.triton/cache}"
export TMPDIR="${TMPDIR:-$TRITON_HOME/tmp}"
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"
