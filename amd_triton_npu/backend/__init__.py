# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# air statically links LLVM; a ROCm PyTorch install pulls in its own
# libLLVM.so.23. When a script imports torch before triton (as every example
# does), ROCm's LLVM wins the global namespace and air's constructor binds the
# wrong globals and segfaults (issue #102). Load air's bindings with
# RTLD_DEEPBIND so air resolves its own LLVM regardless of torch.
try:
    import os
    import sys

    _prev_flags = sys.getdlopenflags()
    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL | os.RTLD_DEEPBIND)
    try:
        import air._mlir_libs._mlir  # noqa: F401
    finally:
        sys.setdlopenflags(_prev_flags)
except Exception:
    pass
