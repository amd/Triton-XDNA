#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""An NPUChain's own bf16_emulation setting reaches aircc.

An f32 vector multiply has no aie2p lowering of its own; it compiles only
with bf16 emulation. The chain turns emulation on while the global setting
stays off, and must compile, run, and match the product of the inputs rounded
to bf16. The cache key must also differ between the two settings, or one
would load the other's binary. Needs an npu2 device; exits 77 otherwise.
"""

from __future__ import annotations

import os
import sys

N = 4096
BLOCK_SIZE = 1024

_EXAMPLES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"
)
TRANSFORM_SCRIPT = os.path.join(_EXAMPLES, "gpt2", "transform_add_f32_aie2p.mlir")


def skip(msg):
    print(f"SKIP: {msg}")
    sys.exit(77)


def main():
    try:
        import numpy as np
        import torch
        import triton
        import triton.language as tl
        from triton.backends.amd_triton_npu.config import npu_config
        from triton.backends.amd_triton_npu.driver import NPUDriver, detect_npu_version
        from triton.backends.amd_triton_npu.multilaunch import (
            MultiLaunchBuilder,
            NPUChain,
        )
    except ImportError as e:
        skip(f"cannot import the backend: {e}")
    try:
        import pyxrt

        pyxrt.device(0)
    except Exception as e:
        skip(f"no NPU: {e}")
    triton.runtime.driver.set_active(NPUDriver())
    if detect_npu_version() != "npu2":
        skip("needs npu2")

    @triton.jit
    def mul_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offs)
        y = tl.load(y_ptr + offs)
        tl.store(out_ptr + offs, x * y)

    failures = 0

    def check(ok, what):
        nonlocal failures
        print(f"  {'ok  ' if ok else 'FAIL'} {what}", flush=True)
        failures += not ok

    npu_config.bf16_emulation = False
    on = MultiLaunchBuilder("k", bf16_emulation=True)
    off = MultiLaunchBuilder("k", bf16_emulation=False)
    default = MultiLaunchBuilder("k")
    keys = {
        name: b.cache_key("elf", "npu2", text="module {}")
        for name, b in (("on", on), ("off", off), ("default", default))
    }
    check(keys["on"] != keys["off"], "cache key depends on the chain's setting")
    check(keys["default"] == keys["off"], "unset follows the global setting")

    zeros = torch.zeros(N, dtype=torch.float32)
    chain = NPUChain("bf16_emulation_mul", bf16_emulation=True)
    chain.add(
        mul_kernel,
        grid=(N // BLOCK_SIZE,),
        arg_map={0: 0, 1: 1, 2: 2},
        args=(zeros, zeros, zeros, N),
        constexprs={"BLOCK_SIZE": BLOCK_SIZE},
        transform_script=TRANSFORM_SCRIPT,
    )
    rng = np.random.default_rng(0)
    a = rng.standard_normal(N).astype(np.float32)
    b = rng.standard_normal(N).astype(np.float32)
    out = chain.run([a, b, np.zeros(N, np.float32)], bo_key="k")[2]

    def to_bf16(x):
        return torch.from_numpy(x).to(torch.bfloat16).to(torch.float32).numpy()

    ref = to_bf16(a) * to_bf16(b)
    # Older emulation also rounds the product to bf16, and may truncate.
    check(
        bool(np.allclose(out, ref, rtol=2e-2, atol=1e-6)),
        "emulated f32 multiply matches the bf16-rounded product",
    )
    check(npu_config.bf16_emulation is False, "global setting left unchanged")
    chain.close()

    print("\nRESULT:", "PASS" if not failures else f"FAIL ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
