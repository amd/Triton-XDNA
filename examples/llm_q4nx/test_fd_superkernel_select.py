# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""fd_superkernel serves one model a process: a model's entry module selects
it, a second, different selection raises rather than leaving one entry's
decoder with another's sizes, and the same one again is allowed. No NPU, no
weights, no mlir-air sources: only the package's selection is imported."""

import importlib
import sys


def fresh():
    for m in [m for m in sys.modules if m.split(".")[0] == "fd_superkernel"]:
        del sys.modules[m]
    return importlib.import_module("fd_superkernel")


def main():
    pkg = fresh()
    pkg.select("qwen3-4b")
    pkg.select("qwen3-4b")
    try:
        pkg.select("gemma3-4b")
    except RuntimeError as e:
        assert "qwen3-4b" in str(e) and "gemma3-4b" in str(e), e
    else:
        raise AssertionError("a second model was selected")
    assert pkg._model == "qwen3-4b"

    # no selection: the model's sizes cannot be read
    fresh()
    try:
        importlib.import_module("fd_superkernel.spec")
    except RuntimeError as e:
        assert "no model selected" in str(e), e
    else:
        raise AssertionError("spec imported with no model selected")
    print("test_fd_superkernel_select: PASS")


if __name__ == "__main__":
    main()
