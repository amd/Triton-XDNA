#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT


SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"

if [[ $MLIR_AIE_INSTALL_DIR == "" ]]; then
    SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
    MLIR_AIE_COMMIT_HASH_FILE="$(dirname ${SCRIPT_PATH})/mlir-aie-hash.txt"
    # Install mlir-aie
    MLIR_AIE_COMMIT_HASH=$(awk -v kw="Commit:" '$0 ~ kw {for (i=1; i<NF; i++) if ($i == kw) print $(i+1)}' "$MLIR_AIE_COMMIT_HASH_FILE")
    SHORT_MLIR_AIE_COMMIT_HASH="${MLIR_AIE_COMMIT_HASH:0:7}"
    echo "Using mlir-aie hash: $SHORT_MLIR_AIE_COMMIT_HASH"
    MLIR_AIE_VERSION=$(awk -v kw="Version:" '$0 ~ kw {for (i=1; i<NF; i++) if ($i == kw) print $(i+1)}' "$MLIR_AIE_COMMIT_HASH_FILE")
    echo "Version: $MLIR_AIE_VERSION"
    TIMESTAMP=$(awk -v kw="Timestamp:" '$0 ~ kw {for (i=1; i<NF; i++) if ($i == kw) print $(i+1)}' "$MLIR_AIE_COMMIT_HASH_FILE")
    echo "Timestamp: $TIMESTAMP"
    python3 -m pip install mlir_aie==$MLIR_AIE_VERSION.$TIMESTAMP+$SHORT_MLIR_AIE_COMMIT_HASH.no.rtti -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-no-rtti
    export MLIR_AIE_INSTALL_DIR="$(python3 -m pip show mlir_aie | grep ^Location: | awk '{print $2}')/mlir_aie"
fi

export PATH=${MLIR_AIE_INSTALL_DIR}/bin:${PATH} 
export PYTHONPATH=${MLIR_AIE_INSTALL_DIR}/python:${PYTHONPATH}
export LD_LIBRARY_PATH=${MLIR_AIE_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}

# Install llvm-aie (latest nightly)
python3 -m pip install --upgrade --force-reinstall llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly

# Install mlir-air
MLIR_AIR_HASH_FILE="$(dirname ${SCRIPT_PATH})/mlir-air-hash.txt"
MLIR_AIR_COMMIT_HASH=$(awk -v kw="Commit:" '$0 ~ kw {for (i=1; i<NF; i++) if ($i == kw) print $(i+1)}' "$MLIR_AIR_HASH_FILE")
SHORT_MLIR_AIR_COMMIT_HASH="${MLIR_AIR_COMMIT_HASH:0:7}"
echo "Using mlir-air hash: $SHORT_MLIR_AIR_COMMIT_HASH"
MLIR_AIR_VERSION=$(awk -v kw="Version:" '$0 ~ kw {for (i=1; i<NF; i++) if ($i == kw) print $(i+1)}' "$MLIR_AIR_HASH_FILE")
echo "mlir-air version: $MLIR_AIR_VERSION"
MLIR_AIR_TIMESTAMP=$(awk -v kw="Timestamp:" '$0 ~ kw {for (i=1; i<NF; i++) if ($i == kw) print $(i+1)}' "$MLIR_AIR_HASH_FILE")
echo "mlir-air timestamp: $MLIR_AIR_TIMESTAMP"
python3 -m pip install mlir_air==$MLIR_AIR_VERSION.$MLIR_AIR_TIMESTAMP+$SHORT_MLIR_AIR_COMMIT_HASH.no.rtti -f https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels-no-rtti
