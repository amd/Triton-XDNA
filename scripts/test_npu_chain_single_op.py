#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Re-dispatch a one-op NPUChain and check every result.

Cases: plain host staging, a static operand, and bound shared buffers
(skipped without the interop). Needs an npu2 device; exits 77 otherwise.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "examples", "zero_copy"))

N = 4096
DISPATCHES = 8


def skip(msg):
    print(f"SKIP: {msg}")
    sys.exit(77)


def main():
    try:
        import numpy as np
        import torch
        import triton
        from triton.backends.amd_triton_npu.driver import NPUDriver
        from triton.backends.amd_triton_npu.multilaunch import NPUChain
        from common import add_chain
    except ImportError as e:
        skip(f"cannot import the backend: {e}")
    try:
        import pyxrt

        pyxrt.device(0)
    except Exception as e:
        skip(f"no NPU: {e}")
    triton.runtime.driver.set_active(NPUDriver())

    zeros = torch.zeros(N, dtype=torch.float32)
    chain = NPUChain("single_op_add")
    chain.add(
        add_chain.add_kernel,
        grid=(N // add_chain.BLOCK_SIZE,),
        arg_map={0: 0, 1: 1, 2: 2},
        args=(zeros, zeros, zeros, N),
        constexprs={"BLOCK_SIZE": add_chain.BLOCK_SIZE},
        transform_script=add_chain.TRANSFORM_SCRIPT,
    )
    rng = np.random.default_rng(0)
    failures = 0

    def check(ok, what):
        nonlocal failures
        print(f"  {'ok  ' if ok else 'FAIL'} {what}", flush=True)
        failures += not ok

    def close(out, ref):
        return bool(np.allclose(out, ref, rtol=0, atol=1e-5))

    ok = True
    for _ in range(DISPATCHES):
        a = rng.standard_normal(N).astype(np.float32)
        b = rng.standard_normal(N).astype(np.float32)
        out = chain.run([a, b, np.zeros(N, np.float32)], bo_key="staged")[2]
        ok &= close(out, a + b)
    check(ok, f"{DISPATCHES} dispatches, plain host staging, fresh inputs")

    b = rng.standard_normal(N).astype(np.float32)
    ok = True
    for _ in range(DISPATCHES):
        a = rng.standard_normal(N).astype(np.float32)
        out = chain.run(
            [a, b, np.zeros(N, np.float32)], bo_key="static", static_indices={1}
        )[2]
        ok &= close(out, a + b)
    check(ok, f"{DISPATCHES} dispatches with a static operand")

    try:
        from triton.backends.amd_triton_npu import shared

        bufs = [shared.zeros(N, dtype=torch.float32, device="xrt:0") for _ in range(3)]
        if any(t.bo is None for t in bufs):
            raise RuntimeError("no BO behind the shared buffer")
    except Exception as e:  # noqa: BLE001
        print(f"  skip bound buffers: {e}")
        bufs = None
    if bufs is not None:
        ok = True
        for _ in range(DISPATCHES):
            a = rng.standard_normal(N).astype(np.float32)
            b = rng.standard_normal(N).astype(np.float32)
            bufs[0].numpy()[:] = a
            bufs[1].numpy()[:] = b
            chain.run(
                [t.numpy() for t in bufs],
                bo_key="bound",
                intermediate_indices={2},
                output_indices={2},
                bound_buffers={i: t.bo for i, t in enumerate(bufs)},
            )
            ok &= close(bufs[2].numpy(), a + b)
        check(ok, f"{DISPATCHES} dispatches on bound shared buffers")

    chain.close()
    print("\nRESULT:", "PASS" if not failures else f"FAIL ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
