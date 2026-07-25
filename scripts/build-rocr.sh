#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Build an AIE-capable ROCR (rocr-runtime) from source and install it to a
# private prefix, for use with the HSA launch runtime
# (AMD_TRITON_NPU_RUNTIME=hsa / NPUDriver("hsa")). A stock /opt/rocm often lacks
# the AIE dispatch extension header (hsa/hsa_ext_amd_aie.h), so building ROCR
# from source is usually required.
#
# After it finishes, point the backend at the install prefix and put its lib
# ahead of any system-installed ROCR at runtime:
#   export AMD_NPU_ROCR_PATH=<PREFIX>
#   export LD_LIBRARY_PATH=<PREFIX>/lib:${LD_LIBRARY_PATH}
#
# Override any of these via the environment:
#   ROCR_SRC   rocr-runtime source dir
#              (default: <workspace>/rocm-systems/projects/rocr-runtime)
#   PREFIX     install prefix -> AMD_NPU_ROCR_PATH (default: <workspace>/opt/rocm)
#   CLANG_DIR  Clang CMake package dir (default: /usr/lib/cmake/clang-22)
#   JOBS       parallel build jobs (default: nproc)
#
# where <workspace> is the directory containing this Triton-XDNA checkout.
set -euo pipefail

REPO_ROOT=$(realpath "$(dirname -- "${BASH_SOURCE[0]}")/..")
WORKSPACE=$(realpath "${REPO_ROOT}/..")

ROCR_SRC=${ROCR_SRC:-${WORKSPACE}/rocm-systems/projects/rocr-runtime}
PREFIX=${PREFIX:-${WORKSPACE}/opt/rocm}
CLANG_DIR=${CLANG_DIR:-/usr/lib/cmake/clang-22}
JOBS=${JOBS:-$(nproc)}
BUILD_DIR=${ROCR_SRC}/build

echo "ROCR source : ${ROCR_SRC}"
echo "Install to  : ${PREFIX}"
echo "Clang_DIR   : ${CLANG_DIR}"

if [ ! -d "${ROCR_SRC}" ]; then
  echo "error: rocr-runtime source not found at ${ROCR_SRC}" >&2
  echo "       clone the rocm-systems repo or set ROCR_SRC." >&2
  exit 1
fi

cmake -S "${ROCR_SRC}" -B "${BUILD_DIR}" \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DClang_DIR="${CLANG_DIR}" \
  -DIMAGE_SUPPORT=OFF \
  -DBUILD_SHARED_LIBS=ON
cmake --build "${BUILD_DIR}" -j "${JOBS}"
cmake --install "${BUILD_DIR}"

echo
echo "Done. Use this ROCR with:"
echo "  export AMD_NPU_ROCR_PATH=${PREFIX}"
echo "  export LD_LIBRARY_PATH=${PREFIX}/lib:\${LD_LIBRARY_PATH}"
