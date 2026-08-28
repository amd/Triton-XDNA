#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Copy the working tree's amd_triton_npu backend Python into the venv's
# installed backend, so edits take effect without rebuilding the wheel (the
# C++ half of the install is untouched). Run after every backend edit.
#
# Symlinking instead of copying does NOT work: compiler.py locates the
# triton-shared binaries with Path(__file__).resolve(), which follows the
# symlink back into the repo and then looks for them at the wrong prefix.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO/.venv}"
SRC="$REPO/amd_triton_npu/backend"
DST="$("$VENV/bin/python" -c 'import sysconfig,os; print(os.path.join(sysconfig.get_paths()["purelib"], "triton", "backends", "amd_triton_npu"))')"

[[ -d "$DST" ]] || { echo "ERROR: backend not installed at $DST" >&2; exit 1; }

# Drop any symlinks a previous version of this script left behind.
find "$DST" -maxdepth 1 -type l -delete
rm -rf "$DST/__pycache__"
# Mirror the manifest in setup.py::_copy_backend_to_triton, which is what a
# real install uses. Keep the two in step: a file added there and missed here
# shows up as an ImportError only in the dev venv. (*.py already covers
# __init__.py, which setup.py handles separately only so it can create one.)
cp -f "$SRC"/*.py "$DST/"
if [[ -f "$SRC/name.conf" ]]; then
    cp -f "$SRC/name.conf" "$DST/"
fi
cp -rf "$SRC/transform_library" "$DST/"
# include/ carries the C/C++ sources the backend compiles at runtime (the HSA
# runtime and the dispatch glue). The DLPack producer is not here -- it is
# compiled into libtriton, so changing it needs a real rebuild, not a sync.
cp -rf "$SRC/include" "$DST/"
echo "synced $(ls "$SRC"/*.py | wc -l) backend files + include/ -> $DST"
