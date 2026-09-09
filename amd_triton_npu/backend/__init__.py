# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Force mlir-air's statically-linked LLVM to load before any ROCm SDK LLVM.
# A ROCm PyTorch install (e.g. TheRock/gfx1151) transitively pulls in its own
# libLLVM.so.23 via libamdhip64 -> libhipblaslt. Under ELF first-loaded-wins
# symbol resolution, whichever LLVM loads first wins the global namespace; if
# ROCm's wins, mlir-air's libAirAggregateCAPI.so constructor binds the wrong
# LLVM globals and segfaults (issue #102). Importing air's own LLVM-linked
# bindings here makes mlir-air's symbols win before triton_shared is loaded.
try:
    import air._mlir_libs._mlir  # noqa: F401
except Exception:
    pass
