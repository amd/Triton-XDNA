# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Checks for `arg_access.written_args` on kernels compiled by triton-cpu.

Needs no NPU: the analysis reads TTIR, which every backend produces.
"""

import sys

import numpy as np
import triton
import triton.language as tl

from triton.backends.amd_triton_npu.arg_access import written_args
from triton.backends.amd_triton_npu.driver_scope import driver_scope

failures = []


def check(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        failures.append(what)


class Ptr:
    __slots__ = ("p", "dtype")

    def __init__(self, a):
        self.p, self.dtype = a.ctypes.data, a.dtype.name

    def data_ptr(self):
        return self.p


@triton.jit
def _put(dst, i, v):
    tl.store(dst + i, v)


@triton.jit
def copy(X, Y, B: tl.constexpr):
    i = tl.arange(0, B)
    tl.store(Y + i, tl.load(X + i))


@triton.jit
def through_helper(X, Y, Z, B: tl.constexpr):
    i = tl.arange(0, B)
    _put(Z, i, tl.load(X + i) + tl.load(Y + i))


@triton.jit
def gather(T, IDX, OUT, B: tl.constexpr):
    """OUT[i] = T[IDX[i]]: IDX feeds the load's address, nothing writes it."""
    i = tl.arange(0, B)
    tl.store(OUT + i, tl.load(T + tl.load(IDX + i)))


@triton.jit
def accumulate(X, ACC, B: tl.constexpr):
    i = tl.arange(0, B)
    tl.atomic_add(ACC + i, tl.load(X + i))


@triton.jit
def reads_only(X, n, B: tl.constexpr):
    i = tl.arange(0, B)
    tl.load(X + i, mask=i < n)


@triton.jit
def loop_carried(X, Y, n, B: tl.constexpr):
    i = tl.arange(0, B)
    p = Y
    for _ in range(n):
        tl.store(p + i, tl.load(X + i))
        p += B


@triton.jit
def after_loop(X, Y, Z, n, B: tl.constexpr):
    """Stores through the pointer a loop leaves, which may be Z's."""
    i = tl.arange(0, B)
    p = Y
    for _ in range(n):
        p = Z
    tl.store(p + i, tl.load(X + i))


def names(kernel, ck):
    ps = [p.name for p in kernel.params if not p.is_constexpr]
    w = written_args(ck.asm["ttir"], ps)
    return None if w is None else sorted(w)


def main():
    f = np.zeros(16, np.float32)
    idx = np.zeros(16, np.int32)
    with driver_scope("cpu"):
        cases = [
            (copy, (Ptr(f), Ptr(f)), ["Y"]),
            (through_helper, (Ptr(f), Ptr(f), Ptr(f)), ["Z"]),
            (gather, (Ptr(f), Ptr(idx), Ptr(f)), ["OUT"]),
            (accumulate, (Ptr(f), Ptr(f)), ["ACC"]),
            (reads_only, (Ptr(f), 16), []),
            (loop_carried, (Ptr(f), Ptr(f), 4), None),
            (after_loop, (Ptr(f), Ptr(f), Ptr(f), 4), None),
        ]
        for kernel, args, want in cases:
            got = names(kernel, kernel.warmup(*args, B=16, grid=(1,)))
            check(got == want, f"{kernel.fn.__name__}: {got} (want {want})")
    if failures:
        print(f"FAIL: {len(failures)} check(s)")
        return 1
    print("PASS: arg_access")
    return 0


if __name__ == "__main__":
    sys.exit(main())
