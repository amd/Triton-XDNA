#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Superkernels: one @triton.jit function that runs kernels side by side on
the NPU, wave after wave (`tl.extra.npu.spatial`).

Each case checks the AIR module its superkernel lowers to, then launches it
on the NPU and compares with numpy:

  chain      one partition writes an L2 buffer that the other reads, and a
             DDR buffer is updated in place every wave
  phases     a partition calls one kernel twice per wave, writing two L2
             buffers: a loop over phases
  table      an int per wave from `npu.table`, and an int argument set by
             the host
  multicore  a partition on two cores, its schedule given the core count
  steps      a wave of four spatials: the cores of one name run two phases,
             another partition's call between them

Also checks the errors reading a superkernel reports. Needs an npu2 device
for the runs; with no NPU only the IR checks run.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile

try:
    import torch
    import triton
    import triton.language as tl
    from triton.language.extra import npu
except ImportError as e:
    print(f"SKIP: {e}")
    sys.exit(77)

D, DP = 1536, 2048  # vector length, and the power of two it is padded to
# A vector the two cores of the multicore case split evenly: an L2 buffer only
# holds a masked vector's valid elements.
D2 = 2048
WAVES = 4
ROWS = (3, 0, 2, 1)

# The vector-add schedule, on as many cores as the partition is given.
ELEMENTWISE = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly},
      %cores: !transform.param<i64>) {
    %op = transform.structured.match ops{["linalg.generic"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %flat = transform.structured.flatten_elementwise %op
        : (!transform.any_op) -> !transform.any_op
    %res, %new = transform.structured.bufferize_to_allocation %flat
        <{memory_space = 1, bufferize_destination_only, emit_dealloc}>
        : !transform.any_op
    %op1 = transform.structured.match ops{["linalg.generic"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %tiled, %forall = transform.structured.tile_using_forall %op1
        num_threads [%cores] (mapping = [#gpu.thread<x>])
        : (!transform.any_op, !transform.param<i64>)
        -> (!transform.any_op, !transform.any_op)
    transform.include @canonicalize_with_cse failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @pad_and_promote_binary_bf16 failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @canonicalize_with_cse failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @one_shot_bufferize failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @post_bufferize_cleanup failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @vectorize_generics_at_32 failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    %vh = transform.include @air_herd_mapping_and_vectorize
        failures(propagate) (%arg1) : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
"""

failures = 0


def check(ok, what):
    global failures
    print(f"  {'ok  ' if ok else 'FAIL'} {what}", flush=True)
    failures += not ok


@triton.jit
def scale(X, W, Y, row, D: tl.constexpr, DP: tl.constexpr):
    i = tl.arange(0, DP)
    m = i < D
    x = tl.load(X + i, mask=m)
    tl.store(Y + i, x * tl.load(W + row * D + i, mask=m), mask=m)


@triton.jit
def add(X, Y, Z, D: tl.constexpr, DP: tl.constexpr):
    i = tl.arange(0, DP)
    m = i < D
    tl.store(Z + i, tl.load(X + i, mask=m) + tl.load(Y + i, mask=m), mask=m)


@triton.jit
def sub(X, Y, Z, D: tl.constexpr, DP: tl.constexpr, SWAP: tl.constexpr):
    i = tl.arange(0, DP)
    m = i < D
    a, b = tl.load(X + i, mask=m), tl.load(Y + i, mask=m)
    tl.store(Z + i, b - a if SWAP else a - b, mask=m)


@triton.jit
def chain(X, W, NW: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr):
    h = npu.allocate_l2([D], tl.bfloat16)
    for wave in range(NW):
        npu.spatial(
            [(scale, (X, W, h, wave, D, DP)), (add, (X, h, X, D, DP))],
            cores=[1, 1],
            schedules=[S, S],
        )


@triton.jit
def scale_twice(X, W, h0, h1, wave, D: tl.constexpr, DP: tl.constexpr):
    hs = (h0, h1)
    for p in tl.static_range(2):
        scale(X, W, hs[p], wave * 2 + p, D, DP)


@triton.jit
def phases(X, W, NW: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr):
    h0 = npu.allocate_l2([D], tl.bfloat16)
    h1 = npu.allocate_l2([D], tl.bfloat16)
    for wave in range(NW):
        npu.spatial(
            [(scale_twice, (X, W, h0, h1, wave, D, DP)), (add, (h0, h1, X, D, DP))],
            cores=[1, 1],
            schedules=[S, S],
        )


@triton.jit(do_not_specialize=["k"])
def table(
    X,
    W,
    k,
    NW: tl.constexpr,
    D: tl.constexpr,
    DP: tl.constexpr,
    S: tl.constexpr,
    ROWS: tl.constexpr,
):
    h = npu.allocate_l2([D], tl.bfloat16)
    for wave in range(NW):
        npu.spatial(
            [
                (scale, (X, W, h, npu.table(ROWS, wave), D, DP)),
                (scale, (h, W, X, k, D, DP)),
            ],
            cores=[1, 1],
            schedules=[S, S],
        )


@triton.jit
def multicore(
    X, W, NW: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr
):
    h = npu.allocate_l2([D], tl.bfloat16)
    for wave in range(NW):
        npu.spatial(
            [(scale, (X, W, h, wave, D, DP)), (add, (X, h, X, D, DP))],
            cores=[1, 2],
            schedules=[S, S],
        )


@triton.jit
def steps(X, W, NW: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr):
    h0 = npu.allocate_l2([D], tl.bfloat16)
    h1 = npu.allocate_l2([D], tl.bfloat16)
    g = npu.allocate_l2([D], tl.bfloat16)
    for wave in range(NW):
        npu.spatial([(scale, (X, W, h0, wave * 2, D, DP))], [1], [S], names=["s"])
        npu.spatial([(add, (h0, X, g, D, DP))], [1], [S], names=["b"])
        npu.spatial([(scale, (X, W, h1, wave * 2 + 1, D, DP))], [1], [S], names=["s"])
        npu.spatial([(add, (g, h1, X, D, DP))], [1], [S], names=["a"])


# Superkernels that reading rejects.


@triton.jit
def unwritten(X, NW: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr):
    h = npu.allocate_l2([D], tl.bfloat16)
    for wave in range(NW):
        npu.spatial([(add, (h, X, X, D, DP))], cores=[1], schedules=[S])


@triton.jit
def overlap(X, NW: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr):
    for wave in range(NW):
        npu.spatial(
            [(add, (X, X, X, D, DP)), (add, (X, X, X, D, DP))],
            cores=[2, 1],
            schedules=[S, S],
            at=[(2, 2), (3, 2)],
        )


@triton.jit
def two_shapes(X, NW: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr):
    for wave in range(NW):
        npu.spatial([(add, (X, X, X, D, DP))], [1], [S], names=["a"])
        npu.spatial([(add, (X, X, X, D, DP))], [2], [S], names=["a"])


@triton.jit
def wave_and_scalar(
    X, W, k, NW: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr
):
    for wave in range(NW):
        npu.spatial([(scale, (X, W, X, wave + k, D, DP))], cores=[1], schedules=[S])


@triton.jit
def read_two(X, h0, h1, o, D: tl.constexpr, DP: tl.constexpr):
    add(h0, X, o, D, DP)
    add(h1, X, o, D, DP)


@triton.jit
def phased_inputs(
    X, NW: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr
):
    h0 = npu.allocate_l2([D], tl.bfloat16)
    h1 = npu.allocate_l2([D], tl.bfloat16)
    o = npu.allocate_l2([D], tl.bfloat16)
    for wave in range(NW):
        npu.spatial(
            [
                (scale_twice, (X, X, h0, h1, wave, D, DP)),
                (read_two, (X, h0, h1, o, D, DP)),
                (add, (o, X, X, D, DP)),
            ],
            cores=[1, 1, 1],
            schedules=[S, S, S],
        )


@triton.jit
def interleaved(X, W, h0, h1, wave, D: tl.constexpr, DP: tl.constexpr):
    scale(X, W, h0, wave * 2, D, DP)
    add(X, h0, X, D, DP)
    scale(X, W, h1, wave * 2 + 1, D, DP)


@triton.jit
def phase_order(
    X, W, NW: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr
):
    h0 = npu.allocate_l2([D], tl.bfloat16)
    h1 = npu.allocate_l2([D], tl.bfloat16)
    for wave in range(NW):
        npu.spatial(
            [(interleaved, (X, W, h0, h1, wave, D, DP)), (add, (X, h1, X, D, DP))],
            cores=[1, 1],
            schedules=[S, S],
        )


@triton.jit
def both_ways(X, Y, o0, o1, D: tl.constexpr, DP: tl.constexpr):
    sub(X, Y, o0, D, DP, False)
    sub(X, Y, o1, D, DP, True)


@triton.jit
def wiring(X, Y, NW: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr):
    o0 = npu.allocate_l2([D], tl.bfloat16)
    o1 = npu.allocate_l2([D], tl.bfloat16)
    for wave in range(NW):
        npu.spatial(
            [(both_ways, (X, Y, o0, o1, D, DP)), (add, (o0, o1, X, D, DP))],
            cores=[1, 1],
            schedules=[S, S],
        )


def bf(a):
    """Round float32 values to bfloat16, as float32."""
    import numpy as np

    u = np.asarray(a, np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.view(np.float32)


def ref_chain(x, w):
    w = w.reshape(-1, x.size)
    for l in range(WAVES):
        x = bf(x + bf(x * w[l]))
    return x


def ref_phases(x, w):
    w = w.reshape(-1, x.size)
    for l in range(WAVES):
        x = bf(bf(x * w[2 * l]) + bf(x * w[2 * l + 1]))
    return x


def ref_table(x, w, k):
    w = w.reshape(-1, x.size)
    for l in range(WAVES):
        x = bf(bf(x * w[ROWS[l]]) * w[k])
    return x


def ref_steps(x, w):
    w = w.reshape(-1, x.size)
    for l in range(WAVES):
        x = bf(bf(bf(x * w[2 * l]) + x) + bf(x * w[2 * l + 1]))
    return x


def ref_multicore(x, w):
    w = w.reshape(-1, x.size)
    for l in range(WAVES):
        x = bf(x + bf(x * w[l]))
    return x


CASES = {
    "chain": (chain, (), {}, ref_chain, D),
    "phases": (phases, (), {}, ref_phases, D),
    "table": (table, (2,), {"ROWS": ROWS}, ref_table, D),
    "multicore": (multicore, (), {}, ref_multicore, D2),
    "steps": (steps, (), {}, ref_steps, D),
}


def module_text(kernel, args, kwargs, proj):
    """The AIR module the superkernel `kernel` lowers to, as text."""
    from triton.backends.amd_triton_npu.config import config_context
    from triton.backends.amd_triton_npu.driver_scope import driver_scope
    from triton.backends.amd_triton_npu.superkernel_jit import read

    with driver_scope("npu"), config_context(compile_only=True):
        c = kernel.warmup(*args, grid=(1,), **kwargs)
    ttir = c.asm["ttsharedir"]
    ttir = ttir.decode() if isinstance(ttir, bytes) else ttir
    return read(ttir, proj)[0].module_text()


def ir_checks(name, text):
    """What each case's module must show."""
    func = text[text.index("func.func") :].split("\n", 1)[0]
    check(text.count("air.segment @") == 1, f"{name}: one segment")
    loop = re.search(r"scf\.for %\S+ = %\S+ to (%\S+) step", text)
    trips = loop and re.search(
        rf"{re.escape(loop[1])} = arith.constant (\d+) : index", text
    )
    check(bool(trips) and int(trips[1]) == WAVES, f"{name}: a loop over the waves")
    check("air.channel @" in text, f"{name}: L2 buffers moved by channels")
    check(func.count("memref<*xbf16>") == 2, f"{name}: only X and W are buffers")
    if name == "phases":
        check("air.disable_ping_pong" in text, "phases: a phase loop")
        check("npu_dma_packet" in text, "phases: outputs routed by packet")
    if name == "table":
        check("i32" in func, "table: k is an argument of the func")
        check("arith.select" in text, "table: per-wave values selected")
    if name == "steps":
        check("air.disable_ping_pong" in text, "steps: a phase loop for s")
    if name == "multicore":
        sizes = re.findall(r"air\.herd @(\w+) .*? in \(%\w+=(%\w+)", text)
        two = [h for h, s in sizes if re.search(rf"{s} = arith.constant 2 :", text)]
        check(bool(two), "multicore: a herd on two cores")


def error_checks(script, proj):
    x = torch.empty(D, dtype=torch.bfloat16)
    common = dict(NW=WAVES, D=D, DP=DP, S=script)
    bad = [
        ("unwritten", unwritten, (x,), "which no call writes"),
        ("overlap", overlap, (x,), "overlaps"),
        ("two shapes", two_shapes, (x,), "is given"),
        ("wave and scalar", wave_and_scalar, (x, x, 3), "only ints from constants"),
        ("phased inputs", phased_inputs, (x,), "not supported"),
        ("phase order", phase_order, (x, x), "listed between the phases"),
        ("phase wiring", wiring, (x, x), "differ"),
    ]
    for what, kernel, args, want in bad:
        try:
            module_text(kernel, args, common, os.path.join(proj, what))
            check(False, f"{what}: no error")
        except ValueError as e:
            check(want in str(e), f"{what}: {e}")
    with open(script) as f:
        fixed = (
            f.read()
            .replace(",\n      %cores: !transform.param<i64>) {", ") {")
            .replace("[%cores]", "[1]")
            .replace("!transform.any_op, !transform.param<i64>", "!transform.any_op")
        )
    lit = os.path.join(proj, "fixed.mlir")
    with open(lit, "w") as f:
        f.write(fixed)
    try:
        module_text(chain, (x, x), dict(common, S=lit), os.path.join(proj, "fixed"))
        check(False, "a schedule without core counts is an error")
    except ValueError as e:
        check("does not take the core counts" in str(e), f"fixed cores: {e}")


def run(name, kernel, extra, kw, ref, n):
    import numpy as np

    rng = np.random.default_rng(0)
    x0 = bf(rng.standard_normal(n))
    w0 = bf(1 + 0.1 * rng.standard_normal(2 * WAVES * n))
    x = torch.from_numpy(x0).to(torch.bfloat16)
    w = torch.from_numpy(w0).to(torch.bfloat16)
    kernel[(1,)](x, w, *extra, **kw)
    got = x.to(torch.float32).numpy()
    want = ref(x0, w0, *extra)
    err = float(np.max(np.abs(got - want) / (np.abs(want) + 1e-2)))
    check(err < 2e-2, f"{name}: NPU result matches numpy (max rel err {err:.2e})")


def main():
    try:
        import pyxrt

        pyxrt.device(0)
        has_npu = True
    except Exception as e:
        print(f"no NPU ({e}): IR checks only")
        has_npu = False
    os.environ.setdefault("AMD_TRITON_NPU_TARGET", "npu2")
    from triton.backends.amd_triton_npu.config import config_context
    from triton.backends.amd_triton_npu.driver_scope import driver_scope

    with tempfile.TemporaryDirectory() as tmp:
        script = os.path.join(tmp, "elementwise.mlir")
        with open(script, "w") as f:
            f.write(ELEMENTWISE)
        with config_context(bf16_emulation=False):
            error_checks(script, os.path.join(tmp, "bad"))
            for name, (kernel, extra, kwargs, ref, n) in CASES.items():
                print(f"{name}:", flush=True)
                x = torch.empty(n, dtype=torch.bfloat16)
                w = torch.empty(2 * WAVES * n, dtype=torch.bfloat16)
                dp = 1 << (n - 1).bit_length()
                kw = dict(NW=WAVES, D=n, DP=dp, S=script, **kwargs)
                text = module_text(kernel, (x, w, *extra), kw, os.path.join(tmp, name))
                ir_checks(name, text)
                if has_npu:
                    with driver_scope("npu"):
                        run(name, kernel, extra, kw, ref, n)
    if failures:
        print(f"{failures} check(s) failed")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
