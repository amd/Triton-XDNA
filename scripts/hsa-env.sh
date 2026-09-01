#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Environment for driving the NPU through HSA/ROCR alongside the iGPU -- the
# heterogeneous examples under examples/zero_copy and examples/hsa_matmul.
# Source it, don't execute it.
#
#     source scripts/hsa-env.sh
#     python examples/zero_copy/shared_buffer_test.py
#
# A pure-XRT run needs none of this; utils/env_setup.sh plus the XRT setup
# script is enough. What is specific to HSA is the LD_PRELOAD below.
#
# Expects an environment that already has the backend installed (VENV, default
# .venv) and the MLIR-AIE stack (install it with `source utils/env_setup.sh`).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO/.venv}"

if [[ -f "$VENV/bin/activate" ]]; then
    source "$VENV/bin/activate"
fi
source /opt/xilinx/xrt/setup.sh >/dev/null

# MLIR-AIE paths come from the repo's own env_setup.sh rather than a second
# copy of its logic. Its install half is skipped: this file is sourced once per
# shell, and installing re-resolves two wheel sets over the network and
# upgrades llvm-aie to the current nightly -- changing the toolchain under a
# checkout that was working a minute ago. Run `source utils/env_setup.sh` when
# installing is what you actually want.
export TRITON_XDNA_ENV_INSTALL=0
source "$REPO/utils/env_setup.sh" ||
    echo "hsa-env.sh: no MLIR-AIE stack found; run 'source utils/env_setup.sh'" >&2
unset TRITON_XDNA_ENV_INSTALL

# Settle which ROCR the process gets, which is the reason this file exists.
# Both the NPU's HSA runtime and torch's HIP want libhsa-runtime64.so.1, and
# whichever is loaded first claims that SONAME for the whole process -- so
# importing torch, which bundles its own copy, would hand the AIE agent to a
# ROCR that aborts on it. See README, "Sharing a process with PyTorch".
#
# Asked of the backend rather than guessed, so this honors AMD_NPU_ROCR_PATH and
# ROCM_PATH and picks the same library the HSA runtime will be linked against --
# preloading a different one would be worse than preloading nothing.
ROCR_LIB="$(python -c 'from triton.backends.amd_triton_npu.driver import _get_rocr_install
print(_get_rocr_install().lib_path)' 2>/dev/null)"
if [[ -f "$ROCR_LIB" ]]; then
    export LD_PRELOAD="$ROCR_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
else
    echo "hsa-env.sh: no AIE-capable ROCR found; the HSA runtime will report why" >&2
fi

# The JIT cache grows fast and ~/.triton is often a quota-limited network home,
# so default it next to the checkout. Override either var to move it.
export TRITON_HOME="${TRITON_HOME:-$REPO/.triton-home}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$TRITON_HOME/.triton/cache}"
export TMPDIR="${TMPDIR:-$TRITON_HOME/tmp}"
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"
