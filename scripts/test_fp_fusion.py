#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""`enable_fp_fusion` (on unless set off): a kernel compiled with it has its
float multiplies, adds and subtracts marked `contract`, which lets mlir-aie
fuse a multiply into the add that takes it; without it they carry no such
flag. Checked on the ttsharedir of a kernel, and on the AIR module of a
superkernel, whose kernels are compiled when it is first launched.

Needs no NPU.

    python3 scripts/test_fp_fusion.py
"""

import os
import sys
import tempfile

try:
    import torch
    import triton
    import triton.language as tl
    from triton.language.extra import air
except ImportError as e:
    print(f"SKIP: {e}")
    sys.exit(77)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_superkernel as T  # noqa: E402

D, DP = T.D, T.DP


@triton.jit
def fma(X, Y, Z, D: tl.constexpr):
    r = tl.arange(0, D)
    z = tl.load(Z + r)
    tl.store(Z + r, tl.load(X + r) * tl.load(Y + r) + z - z * 0.5)


@triton.jit
def arith_superkernel(X, W, Y, Z, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr):
    for wave in range(1):
        air.herd(T.scale, (X, W, Y, 0, D, DP), 1, S, name="mul")
        air.herd(T.add, (X, Y, Z, D, DP), 1, S, name="add")
        air.herd(T.sub, (X, Y, Z, D, DP, False), 1, S, name="sub")


def contracted(text):
    """Whether every float multiply, add and subtract of `text` is marked
    contract, and whether any is."""
    ops = [
        line
        for line in text.splitlines()
        if any(f"arith.{op} " in line for op in ("mulf", "addf", "subf"))
    ]
    marked = ["contract" in line for line in ops]
    return bool(ops) and all(marked), any(marked)


def main():
    from triton.backends.amd_triton_npu.config import config_context
    from triton.backends.amd_triton_npu.driver_scope import driver_scope

    os.environ.setdefault("AMD_TRITON_NPU_TARGET", "npu2")
    x, w, y, z = (torch.zeros(DP, dtype=torch.bfloat16) for _ in range(4))
    with tempfile.TemporaryDirectory() as tmp:
        script = os.path.join(tmp, "elementwise.mlir")
        with open(script, "w") as f:
            f.write(T.ELEMENTWISE)
        for fusion in (False, True, None):
            opt = {} if fusion is None else dict(enable_fp_fusion=fusion)
            fusion = fusion is not False
            with driver_scope("npu"), config_context(compile_only=True):
                c = fma.warmup(x, y, z, D=DP, grid=(1,), **opt)
            text = c.asm["ttsharedir"]
            text = text.decode() if isinstance(text, bytes) else text
            every, some = contracted(text)
            T.check(
                every if fusion else not some,
                f"kernel, {opt or 'default'}: "
                + ("every" if fusion else "no")
                + " float op marked contract",
            )
            text = T.module_text(
                arith_superkernel,
                (x, w, y, z),
                dict(D=D, DP=DP, S=script, **opt),
                os.path.join(tmp, f"sk{fusion}"),
            )
            every, some = contracted(text)
            T.check(
                every if fusion else not some,
                f"superkernel, {opt or 'default'}: "
                + ("every" if fusion else "no")
                + " float op marked contract",
            )
    if T.failures:
        print(f"{T.failures} check(s) failed")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
