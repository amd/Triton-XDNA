# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Calls of a fused device's ops as Triton kernels (`FusedProgram`), on the
NPU: which op a call reaches, that the host and the device see each other's
writes, that runs of one op can be in flight together, and that the host
waits only for runs on the bytes it touches.

The device holds one vector add as two ops, told apart by an int bound into
each (`tag`), which the kernel does not read, so both share one body. Needs
an npu2.
"""

import os
import sys
import tempfile

import numpy as np
import triton
import triton.language as tl
from ml_dtypes import bfloat16

from triton.backends.amd_triton_npu import fused_program as FP
from triton.backends.amd_triton_npu.config import config_context
from triton.backends.amd_triton_npu.driver_scope import driver_scope
from triton.backends.amd_triton_npu.fused_device import FusedDevice
from triton.backends.amd_triton_npu.fused_program import FusedProgram

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "examples", "vec-add", "transform_aie2p.mlir")
N = 1024
TAGS = (1, 2)
failures = []


def check(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        failures.append(what)


@triton.jit
def vecadd(A, B, C, tag, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(C + i, tl.load(A + i) + tl.load(B + i))


@triton.jit
def double(X, Y, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(Y + i, tl.load(X + i) * 2)


def build(out):
    import torch

    dev = FusedDevice("dev", os.path.join(out, "proj"))
    with config_context(bf16_emulation=True):
        z = torch.zeros(N, dtype=torch.bfloat16)
        for tag in TAGS:
            dev.add(
                f"add_{tag}", vecadd, (1,), (z, z, z, tag, N), transform_script=SCRIPT
            )
    dev.compile(runtime_loop_tiling_sizes=(), bf16_emulation=True)
    dev.save(out)


def filled(prog, n, rng):
    b = prog.empty(n, bfloat16)
    b.host(write=True)[...] = rng.standard_normal(n).astype(bfloat16)
    return b


def ref(a, b):
    return (a.host().astype(np.float32) + b.host().astype(np.float32)).astype(bfloat16)


def main():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        build(d)
        prog = FusedProgram(d, "dev")
        n = N
        a, b, c = filled(prog, n, rng), filled(prog, n, rng), prog.empty(n, bfloat16)
        with prog:
            vecadd[(1,)](a, b, c, 1, n)
        check(
            np.array_equal(c.host(), ref(a, b)),
            "a call reaches its op; the host reads its output",
        )
        picked = [prog.find(vecadd, dict(tag=tag, N=n)).name for tag in TAGS]
        check(
            picked == ["add_1", "add_2"], f"the bound values pick between ops: {picked}"
        )

        try:
            with prog:
                vecadd[(1,)](a, b, c, 3, n)
            check(False, "a call no op was built for is refused")
        except KeyError as e:
            check("'tag': 3" in str(e), "a call no op was built for is refused")

        # the host writes an input after a run; the next run sees it
        a.host(write=True)[...] = rng.standard_normal(n).astype(bfloat16)
        with prog:
            vecadd[(1,)](a, b, c, 1, n)
        check(
            np.array_equal(c.host(), ref(a, b)), "a run sees what the host wrote since"
        )

        # runs of one op in flight together, each on its own output
        outs = [prog.empty(n, bfloat16) for _ in range(6)]
        with prog:
            for o in outs:
                vecadd[(1,)](a, b, o, 2, n)
        op = prog.find(vecadd, dict(tag=2, N=n))
        launches = len(op.busy) + len(op.idle)
        check(launches > 1, "in-flight runs of one op take launches of their own")
        check(
            all(np.array_equal(o.host(), ref(a, b)) for o in outs),
            "every in-flight run's output is right",
        )

        # views: a run on rows of a buffer, and the host waiting only for
        # runs on the bytes it touches
        big = prog.empty((2, n), bfloat16)
        big.host(write=True)[...] = rng.standard_normal((2, n)).astype(bfloat16)
        waited = []
        real = FP._Run.wait

        def spy(run):
            waited.append(run)
            real(run)

        FP._Run.wait = spy
        try:
            with prog:
                vecadd[(1,)](big[0:1].reshape(n), b, c, 1, n)
            big[1:2].host(write=True)[...] = 0
            check(
                not waited, "writing other rows than a run reads does not wait for it"
            )
            big[0:1].host(write=True)
            check(len(waited) == 1, "writing rows a run reads waits for it")
        finally:
            FP._Run.wait = real
        with prog:
            vecadd[(1,)](big[1:2].reshape(n), b, c, 1, n)
        want = (big.host()[1].astype(np.float32) + b.host().astype(np.float32)).astype(
            bfloat16
        )
        check(np.array_equal(c.host(), want), "a run on a view reads the view's bytes")

        # a host Triton kernel on a device buffer: it waits for the run and
        # reads what the device wrote; the device then reads what it wrote
        y, z = prog.empty(n, bfloat16), prog.empty(n, bfloat16)
        with prog:
            vecadd[(1,)](a, b, c, 1, n)
            with driver_scope("cpu"):
                double[(1,)](c, y, n)
            vecadd[(1,)](y, b, z, 1, n)
        want = (ref(a, b).astype(np.float32) * 2).astype(bfloat16)
        check(
            np.array_equal(y.host(), want), "a host kernel reads what the device wrote"
        )
        back = (want.astype(np.float32) + b.host().astype(np.float32)).astype(bfloat16)
        check(
            np.array_equal(z.host(), back), "the device reads what a host kernel wrote"
        )

    if failures:
        print(f"FAIL: {len(failures)} check(s)")
        return 1
    print("PASS: fused_program")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("AMD_TRITON_NPU_TARGET", "npu2")
    sys.exit(main())
