#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Superkernels: one @triton.jit function that runs kernels side by side on
the NPU, wave after wave (`tl.extra.air`).

Each case checks the AIR module its superkernel lowers to, then launches it
on the NPU and compares with numpy:

  chain      one herd writes an L2 buffer that the other reads, and a
             DDR buffer is updated in place every wave
  phases     a herd calls one kernel twice per wave, writing two L2
             buffers: a loop over phases
  table      an int per wave from `air.table`, and an int argument set by
             the host
  multicore  a herd on two cores, its schedule given the core count
  steps      a wave of four herds: the cores of one name run two phases,
             another herd's call between them
  stream     a core puts blocks into a channel that another core gets
  spmd       a kernel run as one program a core, program_id its core
  bcast      one core's blocks broadcast to the two cores of another
  merge      two cores put into a FIFO each, one core gets both

Also checks the errors reading a superkernel reports. Needs an npu2 device
for the runs; with no NPU only the IR checks run.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile

try:
    import numpy as np
    import torch
    import triton
    import triton.language as tl
    from triton.language.extra import air
except ImportError as e:
    print(f"SKIP: {e}")
    sys.exit(77)

D, DP = 1536, 2048  # vector length, and the power of two it is padded to
# A vector the two cores of the multicore case split evenly: an L2 buffer only
# holds a masked vector's valid elements.
D2 = 2048
WAVES = 4
ROWS = (3, 0, 2, 1)

# The vector-add schedule, on as many cores as the herd is given.
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

# A kernel whole on each of its cores, for kernels that move their own data
# (channels, program ids): every copy a DMA of the core.
DIRECT = """
// A kernel whole on one core: its body in a 1x1 herd, every buffer it
// allocates in the core's L1, every copy a DMA of the core.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly},
      %cores_x: !transform.param<i64>, %cores_y: !transform.param<i64>) {
    // A masked load's fill and box copy as tensor ops: the box gets a buffer
    // of its own, so no DMA writes a buffer the core also fills.
    %fr = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fr1 = transform.air.raise_load_buffer_to_tensor %fr : (!transform.any_op) -> !transform.any_op
    %f0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f0 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %f0 : !transform.any_op
    %f1 = transform.air.fuse_elementwise_linalg %f0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f1 { transform.apply_patterns.canonicalization } : !transform.any_op
    %te = transform.structured.match ops{["tensor.empty"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %te1 = transform.cast %te : !transform.any_op to !transform.op<"tensor.empty">
    %te2 = transform.bufferization.empty_tensor_to_alloc_tensor %te1 : (!transform.op<"tensor.empty">) -> !transform.op<"bufferization.alloc_tensor">
    transform.include @one_shot_bufferize failures(propagate) (%arg1) : (!transform.any_op) -> ()
    // post_bufferize_cleanup, less eliminate_cascade_memcpy: a buffer each
    // value stored, none written again while its DMA may still read it
    %fb = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %fb {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %fb : !transform.any_op
    %lc = transform.structured.match ops{["linalg.copy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %mc = transform.structured.linalg_copy_to_memref %lc : (!transform.any_op) -> !transform.any_op
    %fb2 = transform.air.remove_uninitialized_copy %fb : (!transform.any_op) -> !transform.any_op
    // (a one-trip parallel canonicalizes away: made only now, and taken to
    // a herd at once)
    %f2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %f3 = transform.apply_registered_pass "air-wrap-func-with-parallel"
        with options = { "loop-bounds" = "1,1" } to %f2
        : (!transform.any_op) -> !transform.any_op
    %p = transform.structured.match ops{["scf.parallel"]} in %f3 : (!transform.any_op) -> !transform.any_op
    %h = transform.air.par_to_herd %p : (!transform.any_op) -> !transform.any_op
    %h1 = transform.air.override_memref_memory_space %h <{memory_space = 2 : i32}> : (!transform.any_op) -> !transform.any_op
    %cp = transform.structured.match ops{["memref.copy", "linalg.copy"]} in %h1 : (!transform.any_op) -> !transform.any_op
    transform.foreach %cp : !transform.any_op {
    ^bb0(%c: !transform.any_op):
      %dm = transform.air.copy_to_dma %c : (!transform.any_op) -> !transform.any_op
    }
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
    h = air.alloc([D], tl.bfloat16, memory_space=air.L2)
    for wave in range(NW):
        air.herd(scale, (X, W, h, wave, D, DP), 1, S)
        air.herd(add, (X, h, X, D, DP), 1, S)


@triton.jit
def scale_twice(X, W, h0, h1, wave, D: tl.constexpr, DP: tl.constexpr):
    hs = (h0, h1)
    for p in tl.static_range(2):
        scale(X, W, hs[p], wave * 2 + p, D, DP)


@triton.jit
def phases(X, W, NW: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr):
    h0 = air.alloc([D], tl.bfloat16, memory_space=air.L2)
    h1 = air.alloc([D], tl.bfloat16, memory_space=air.L2)
    for wave in range(NW):
        air.herd(scale_twice, (X, W, h0, h1, wave, D, DP), 1, S)
        air.herd(add, (h0, h1, X, D, DP), 1, S)


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
    h = air.alloc([D], tl.bfloat16, memory_space=air.L2)
    for wave in range(NW):
        air.herd(scale, (X, W, h, air.table(ROWS, wave), D, DP), 1, S)
        air.herd(scale, (h, W, X, k, D, DP), 1, S)


@triton.jit
def multicore(
    X, W, NW: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr
):
    h = air.alloc([D], tl.bfloat16, memory_space=air.L2)
    for wave in range(NW):
        air.herd(scale, (X, W, h, wave, D, DP), 1, S)
        air.herd(add, (X, h, X, D, DP), 2, S)


@triton.jit
def steps(X, W, NW: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr):
    h0 = air.alloc([D], tl.bfloat16, memory_space=air.L2)
    h1 = air.alloc([D], tl.bfloat16, memory_space=air.L2)
    g = air.alloc([D], tl.bfloat16, memory_space=air.L2)
    for wave in range(NW):
        air.herd(scale, (X, W, h0, wave * 2, D, DP), 1, S, name="s")
        air.herd(add, (h0, X, g, D, DP), 1, S, name="b")
        air.herd(scale, (X, W, h1, wave * 2 + 1, D, DP), 1, S, name="s")
        air.herd(add, (g, h1, X, D, DP), 1, S, name="a")


@triton.jit
def src(X, D: tl.constexpr, T: tl.constexpr):
    for t in range(D // T):
        air.ChannelPut("toSink", tl.load(X + t * T + tl.arange(0, T)) * 2)


@triton.jit
def sink(Y, D: tl.constexpr, T: tl.constexpr):
    for t in range(D // T):
        y = air.ChannelGet("toSink", [T], tl.bfloat16)
        tl.store(Y + t * T + tl.arange(0, T), y + 1)


@triton.jit
def stream(X, Y, NW: tl.constexpr, D: tl.constexpr, T: tl.constexpr, S: tl.constexpr):
    air.Channel("toSink")
    for wave in range(NW):
        air.herd(src, (X, D, T), 1, S)
        air.herd(sink, (Y, D, T), 1, S)


@triton.jit
def twice(X, Y, D: tl.constexpr):
    p = tl.program_id(0)
    x = tl.load(X + p * D + tl.arange(0, D))
    tl.store(Y + p * D + tl.arange(0, D), x * 2)


@triton.jit
def spmd(X, Y, NW: tl.constexpr, D: tl.constexpr, T: tl.constexpr, S: tl.constexpr):
    for wave in range(NW):
        air.herd(twice, (X, Y, D), 2, S)


@triton.jit
def bc_sink(Y, D: tl.constexpr, T: tl.constexpr):
    p = tl.program_id(0)
    for t in range(D // T):
        y = air.ChannelGet("toSink", [T], tl.bfloat16, indices=[p, 0])
        tl.store(Y + p * D + t * T + tl.arange(0, T), y + p.to(tl.bfloat16))


@triton.jit
def bc_src(X, D: tl.constexpr, T: tl.constexpr):
    for t in range(D // T):
        air.ChannelPut(
            "toSink", tl.load(X + t * T + tl.arange(0, T)) * 2, indices=[0, 0]
        )


@triton.jit
def bcast(X, Y, NW: tl.constexpr, D: tl.constexpr, T: tl.constexpr, S: tl.constexpr):
    air.Channel("toSink", size=[1, 1], broadcast_shape=[2, 1])
    for wave in range(NW):
        air.herd(bc_src, (X, D, T), 1, S)
        air.herd(bc_sink, (Y, D, T), 2, S)


@triton.jit
def m_src(X, D: tl.constexpr, T: tl.constexpr):
    p = tl.program_id(0)
    for t in range(D // T):
        x = tl.load(X + p * D + t * T + tl.arange(0, T))
        air.ChannelPut("parts", x * 2, indices=[p])


@triton.jit
def m_sink(Y, D: tl.constexpr, T: tl.constexpr):
    for t in range(D // T):
        a = air.ChannelGet("parts", [T], tl.bfloat16, indices=[0])
        b = air.ChannelGet("parts", [T], tl.bfloat16, indices=[1])
        tl.store(Y + t * T + tl.arange(0, T), a + b)


@triton.jit
def merge(X, Y, NW: tl.constexpr, D: tl.constexpr, T: tl.constexpr, S: tl.constexpr):
    air.Channel("parts", size=[2])
    for wave in range(NW):
        air.herd(m_src, (X, D, T), 2, S)
        air.herd(m_sink, (Y, D, T), 1, S)


@triton.jit
def g_src(X, D: tl.constexpr, T: tl.constexpr):
    i = tl.program_id(0)
    j = tl.program_id(1)
    for t in range(D // T):
        x = tl.load(X + (i * 2 + j) * D + t * T + tl.arange(0, T))
        air.ChannelPut("cells", x * 2, indices=[0, i, j])


@triton.jit
def g_sink(Y, D: tl.constexpr, T: tl.constexpr):
    i = tl.program_id(0)
    j = tl.program_id(1)
    for t in range(D // T):
        y = air.ChannelGet("cells", [T], tl.bfloat16, indices=[0, i, j])
        tl.store(Y + (i * 2 + j) * D + t * T + tl.arange(0, T), y + 1)


@triton.jit
def grid(X, Y, NW: tl.constexpr, D: tl.constexpr, T: tl.constexpr, S: tl.constexpr):
    air.Channel("cells", size=[1, 2, 2])
    for wave in range(NW):
        air.herd(g_src, (X, D, T), (2, 2), S)
        air.herd(g_sink, (Y, D, T), (2, 2), S)


# Superkernels that reading rejects.


@triton.jit
def unwritten(X, NW: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr):
    h = air.alloc([D], tl.bfloat16, memory_space=air.L2)
    for wave in range(NW):
        air.herd(add, (h, X, X, D, DP), 1, S)


@triton.jit
def overlap(X, NW: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr):
    for wave in range(NW):
        air.herd(add, (X, X, X, D, DP), 2, S, at=(2, 2))
        air.herd(add, (X, X, X, D, DP), 1, S, at=(3, 2))


@triton.jit
def two_shapes(X, NW: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr):
    for wave in range(NW):
        air.herd(add, (X, X, X, D, DP), 1, S, name="a")
        air.herd(add, (X, X, X, D, DP), 2, S, name="a")


@triton.jit
def wave_and_scalar(
    X, W, k, NW: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr
):
    for wave in range(NW):
        air.herd(scale, (X, W, X, wave + k, D, DP), 1, S)


@triton.jit
def read_two(X, h0, h1, o, D: tl.constexpr, DP: tl.constexpr):
    add(h0, X, o, D, DP)
    add(h1, X, o, D, DP)


@triton.jit
def phased_inputs(
    X, NW: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr
):
    h0 = air.alloc([D], tl.bfloat16, memory_space=air.L2)
    h1 = air.alloc([D], tl.bfloat16, memory_space=air.L2)
    o = air.alloc([D], tl.bfloat16, memory_space=air.L2)
    for wave in range(NW):
        air.herd(scale_twice, (X, X, h0, h1, wave, D, DP), 1, S)
        air.herd(read_two, (X, h0, h1, o, D, DP), 1, S)
        air.herd(add, (o, X, X, D, DP), 1, S)


@triton.jit
def interleaved(X, W, h0, h1, wave, D: tl.constexpr, DP: tl.constexpr):
    scale(X, W, h0, wave * 2, D, DP)
    add(X, h0, X, D, DP)
    scale(X, W, h1, wave * 2 + 1, D, DP)


@triton.jit
def phase_order(
    X, W, NW: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr
):
    h0 = air.alloc([D], tl.bfloat16, memory_space=air.L2)
    h1 = air.alloc([D], tl.bfloat16, memory_space=air.L2)
    for wave in range(NW):
        air.herd(interleaved, (X, W, h0, h1, wave, D, DP), 1, S)
        air.herd(add, (X, h1, X, D, DP), 1, S)


@triton.jit
def both_ways(X, Y, o0, o1, D: tl.constexpr, DP: tl.constexpr):
    sub(X, Y, o0, D, DP, False)
    sub(X, Y, o1, D, DP, True)


@triton.jit
def wiring(X, Y, NW: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr):
    o0 = air.alloc([D], tl.bfloat16, memory_space=air.L2)
    o1 = air.alloc([D], tl.bfloat16, memory_space=air.L2)
    for wave in range(NW):
        air.herd(both_ways, (X, Y, o0, o1, D, DP), 1, S)
        air.herd(add, (o0, o1, X, D, DP), 1, S)


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


def _bfx(x):
    return torch.from_numpy(bf(x)).to(torch.bfloat16)


# name: kernel, x and y lengths, y from x, what the module must show
XD, XT = 1024, 256
EXPLICIT = {
    "stream": (
        stream,
        XD,
        XD,
        lambda x: bf(bf(2 * x) + 1),
        ["air.channel @toSink [1]", "air.channel.put", "air.channel.get"],
    ),
    "spmd": (spmd, 2 * XD, 2 * XD, lambda x: bf(2 * x), ["in (%arg17=%c2"]),
    "bcast": (
        bcast,
        XD,
        2 * XD,
        lambda x: np.concatenate([bf(bf(2 * x) + p) for p in range(2)]),
        ["air.channel @toSink [1, 1] {broadcast_shape = [2, 1]}"],
    ),
    "merge": (
        merge,
        2 * XD,
        XD,
        lambda x: bf(bf(2 * x[:XD]) + bf(2 * x[XD:])),
        ["air.channel @parts [2]"],
    ),
    "grid": (
        grid,
        4 * XD,
        4 * XD,
        lambda x: bf(bf(2 * x) + 1),
        ["air.channel @cells [1, 2, 2]", "@cells[%c0"],
    ),
}


def run_explicit(name, script, has_npu, proj):
    kernel, nx, ny, ref, shows = EXPLICIT[name]
    rng = np.random.default_rng(1)
    x0 = bf(rng.standard_normal(nx))
    x = _bfx(x0)
    y = torch.zeros(ny, dtype=torch.bfloat16)
    kw = dict(NW=2, D=XD, T=XT, S=script)
    text = module_text(kernel, (x, y), kw, proj)
    for want in shows:
        check(want in text, f"{name}: the module shows {want}")
    if not has_npu:
        return
    from triton.backends.amd_triton_npu.driver_scope import driver_scope

    with driver_scope("npu"):
        kernel[(1,)](x, y, **kw)
    got = y.to(torch.float32).numpy()
    want = ref(x0)
    # The cores round bf16 sums toward zero: one ulp of the result.
    ulp = np.abs(want) * 2.0**-7 + 1e-6
    ok = bool(np.all(np.abs(got - want) <= ulp))
    check(ok, f"{name}: NPU result matches numpy to one ulp")


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
        direct = os.path.join(tmp, "direct.mlir")
        with open(direct, "w") as f:
            f.write(DIRECT)
        with config_context(bf16_emulation=False):
            error_checks(script, os.path.join(tmp, "bad"))
            for name in EXPLICIT:
                print(f"{name}:", flush=True)
                run_explicit(name, direct, has_npu, os.path.join(tmp, name))
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
