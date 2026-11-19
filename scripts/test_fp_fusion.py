#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""`enable_fp_fusion` (on unless set off): a kernel compiled with it has its
float multiplies, adds and subtracts marked `contract`, which lets mlir-aie
fuse a multiply into the add that takes it; without it they carry no such
flag. Checked on the ttsharedir of a kernel, also of one holding
triton-shared's tptr, and on the AIR module of a superkernel, whose kernels
are compiled when it is first launched.

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


@triton.jit
def ptr_select(X, Y, F, D: tl.constexpr):
    r = tl.arange(0, D)
    p = tl.where(tl.load(F) > 0, X, Y)
    tl.store(Y + r, tl.load(p + r) * 2.0 + 1.0)


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


def flags_kept():
    """Flags an op already has stay, with `contract` added."""
    from triton.backends.amd_triton_npu.compiler import _allow_fp_contraction

    text = (
        "%0 = arith.mulf %a, %b fastmath<fast> : vector<16xbf16>\n"
        "%1 = arith.addf %0, %c fastmath<none> : bf16\n"
        "%2 = arith.subf %1, %d : f32 loc(#loc3)\n"
        "%3 = arith.divf %2, %e : f32\n"
    )
    want = (
        "%0 = arith.mulf %a, %b fastmath<contract,fast> : vector<16xbf16>\n"
        "%1 = arith.addf %0, %c fastmath<contract> : bf16\n"
        "%2 = arith.subf %1, %d fastmath<contract> : f32 loc(#loc3)\n"
        "%3 = arith.divf %2, %e : f32\n"
    )
    T.check(_allow_fp_contraction(text) == want, "flags an op has are kept")


def main():
    from triton.backends.amd_triton_npu.config import config_context
    from triton.backends.amd_triton_npu.driver_scope import driver_scope

    flags_kept()

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
    # triton-shared leaves tptr in the ttsharedir of a kernel that selects
    # between pointers; no parser outside triton-shared's tools reads it.
    with driver_scope("npu"), config_context(compile_only=True):
        c = ptr_select.warmup(x, y, w, D=DP, grid=(1,))
    text = c.asm["ttsharedir"]
    text = text.decode() if isinstance(text, bytes) else text
    every, _ = contracted(text)
    T.check(
        "tptr" in text and every,
        "a kernel whose ttsharedir holds tptr: every float op marked contract",
    )
    if T.failures:
        print(f"{T.failures} check(s) failed")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
