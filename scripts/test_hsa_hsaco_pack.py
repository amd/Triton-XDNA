#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Pack an HSA kernel into an hsaco and verify its section, without hardware.

Forces npu_config.compile_only so compile_module stops right after caching
the packed hsaco, before set_paths/dispatch -- so this needs the aiecc
toolchain but not a live NPU device or ROCR build.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "examples", "zero_copy"))


def skip(msg):
    print(f"SKIP: {msg}")
    sys.exit(77)


def main():
    try:
        import triton
        from aie.compiler.hsaco import dump
        from triton.backends.amd_triton_npu.driver import (
            NPUDriver,
            get_npu_cache_dir,
            npu_config,
        )
        from common import add_chain
    except ImportError as e:
        skip(f"cannot import the backend or aie.compiler.hsaco: {e}")

    npu_config.compile_only = True
    triton.runtime.driver.set_active(NPUDriver("hsa"))

    import torch

    zeros = torch.zeros(4096, dtype=torch.float32)
    grid = (4096 // add_chain.BLOCK_SIZE,)
    compiled = add_chain.add_kernel[grid](
        zeros, zeros, zeros, 4096, BLOCK_SIZE=add_chain.BLOCK_SIZE
    )
    cache_dir = get_npu_cache_dir(compiled)
    if cache_dir is None:
        skip("no NPU cache dir -- compile_only path did not run")
    hsaco_path = os.path.join(cache_dir, "kernel.hsaco")
    if not os.path.isfile(hsaco_path):
        print(f"FAIL: {hsaco_path} was not produced")
        return 1

    sections = dump.read_sections_from_hsaco(hsaco_path)
    ok = True
    for arch, data in sections:
        info = dump.parse_section(data)
        print(f"arch={arch} kernels={[k['name'] for k in info['kernels']]}")
        for k in info["kernels"]:
            if k["num_cols"] <= 0:
                print(f"FAIL: kernel {k['name']!r} has num_cols={k['num_cols']}")
                ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
