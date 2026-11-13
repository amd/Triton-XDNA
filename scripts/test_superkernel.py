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
  grid       a rank-3 channel between two 2x2 herds
  gather     segment code gathers two cores' halves in an L2 buffer and
             sends it three times to a third core
  feed       the host puts DDR blocks into a channel a core gets, and gets
             the core's results back into DDR
  neighbours two neighbouring cores share an L1 buffer of the segment
  q4gemv     a core gets q4 weight blocks into a buffer of its own and
             reads them in parts: 32 rows of a GEMV, in vectors
  proj16     a GEMV on 16 cores: x broadcast through a memtile, weight
             streams through the cores' memtiles, rows gathered back
             through them as packets whose header picks one of three
             consumers

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
# (channels, program ids): every copy a DMA of the core. Loops may carry
# buffers.
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
    %fbz = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fbz1 = transform.bufferization.one_shot_bufferize %fbz <{allow_return_allocs_from_loops = true}> : (!transform.any_op) -> !transform.any_op
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

# The same with the core's compute in vectors.
VECTOR = """

// DIRECT, with the core's compute in vectors: each 1-D op 32 elements at a
// time, each 2-D op a row at a time.
module attributes {transform.with_named_sequence} {
  // A 2-D linalg op, to be computed a row at a time.
  transform.named_sequence @match_rank2(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %op ["linalg.generic", "linalg.fill", "linalg.copy"] : !transform.any_op
    %r = transform.match.structured failures(propagate) %op : (!transform.any_op) -> !transform.param<i64> {
    ^bb0(%s: !transform.any_op):
      %rank = transform.match.structured.rank %s : (!transform.any_op) -> !transform.param<i64>
      transform.match.structured.yield %rank : !transform.param<i64>
    }
    %c2 = transform.param.constant 2 : i64 -> !transform.param<i64>
    transform.match.param.cmpi eq %r, %c2 : !transform.param<i64>
    transform.yield %op : !transform.any_op
  }
  // A 1-D linalg op, to be computed 32 elements at a time.
  transform.named_sequence @match_rank1(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %op ["linalg.generic", "linalg.fill", "linalg.copy"] : !transform.any_op
    %r = transform.match.structured failures(propagate) %op : (!transform.any_op) -> !transform.param<i64> {
    ^bb0(%s: !transform.any_op):
      %rank = transform.match.structured.rank %s : (!transform.any_op) -> !transform.param<i64>
      transform.match.structured.yield %rank : !transform.param<i64>
    }
    %c1 = transform.param.constant 1 : i64 -> !transform.param<i64>
    transform.match.param.cmpi eq %r, %c1 : !transform.param<i64>
    transform.yield %op : !transform.any_op
  }
  transform.named_sequence @tile_32(%op: !transform.any_op {transform.consumed}) {
    %t, %l = transform.structured.tile_using_for %op tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
  // Vectors where the op's shape allows; the rest stays scalar.
  transform.named_sequence @vectorize_one(%op: !transform.any_op {transform.consumed}) {
    transform.structured.vectorize %op : !transform.any_op
    transform.yield
  }
  // AIE2P's vector transcendentals and float compares are bf16 only.
  transform.named_sequence @match_transcendental(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %op ["math.exp", "math.tanh"] : !transform.any_op
    transform.yield %op : !transform.any_op
  }
  transform.named_sequence @tile_rows(%op: !transform.any_op {transform.consumed}) {
    %t, %l = transform.structured.tile_using_for %op tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
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
    %f1 = transform.air.fuse_elementwise_linalg %f0 no_recompute : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f1 { transform.apply_patterns.canonicalization } : !transform.any_op
    // reductions as generics, the elementwise ops before them fused in
    %rd = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %rdg = transform.structured.generalize %rd : (!transform.any_op) -> !transform.any_op
    %fz = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fz1 = transform.air.fuse_elementwise_linalg %fz no_recompute : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %fz1 { transform.apply_patterns.canonicalization } : !transform.any_op
    %te = transform.structured.match ops{["tensor.empty"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %te1 = transform.cast %te : !transform.any_op to !transform.op<"tensor.empty">
    %te2 = transform.bufferization.empty_tensor_to_alloc_tensor %te1 : (!transform.op<"tensor.empty">) -> !transform.op<"bufferization.alloc_tensor">
    %fbz = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fbz1 = transform.bufferization.one_shot_bufferize %fbz <{allow_return_allocs_from_loops = true}> : (!transform.any_op) -> !transform.any_op
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
    // copies between two of the core's buffers stay with the core, as
    // vectors
    %mcp = transform.structured.match ops{["memref.copy"]} in %h1 : (!transform.any_op) -> !transform.any_op
    %lcp = transform.air.convert_memref_copy_to_linalg_copy %mcp : (!transform.any_op) -> !transform.any_op
    %cp = transform.structured.match ops{["memref.copy", "linalg.copy"]} in %h1 : (!transform.any_op) -> !transform.any_op
    transform.foreach %cp : !transform.any_op {
    ^bb0(%c: !transform.any_op):
      %dm = transform.air.copy_to_dma %c : (!transform.any_op) -> !transform.any_op
    }
    %h1q = transform.foreach_match in %h1 @match_rank2 -> @tile_rows, @match_rank1 -> @tile_32 : (!transform.any_op) -> !transform.any_op
    %h1r = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %h1r {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    %vg = transform.structured.match ops{["linalg.generic", "linalg.fill", "linalg.copy"]} in %h1r : (!transform.any_op) -> !transform.any_op
    transform.foreach %vg : !transform.any_op {
    ^bb0(%v: !transform.any_op):
      transform.include @vectorize_one failures(suppress) (%v) : (!transform.any_op) -> ()
    }
    %hc = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.include @post_vectorize_reduce_cleanup failures(propagate) (%hc) : (!transform.any_op) -> ()
    // a reduction to a scalar as a vector.reduction
    %hm = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %hm {
      transform.apply_patterns.vector.multi_reduction_flattening lowering_strategy = "innerreduction"
      transform.apply_patterns.vector.multi_reduction_unrolling lowering_strategy = "innerreduction"
    } : !transform.any_op
    %hx = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %hx1 = transform.foreach_match in %hx @match_transcendental -> @action_cast_to_bf16, @match_cmpf -> @action_cast_cmpf_to_bf16 : (!transform.any_op) -> !transform.any_op
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


@triton.jit
def half(X, D: tl.constexpr, T: tl.constexpr):
    p = tl.program_id(0)
    air.ChannelPut("halves", tl.load(X + p * T + tl.arange(0, T)) * 2, indices=[p])


@triton.jit
def use3(Y, D: tl.constexpr, T: tl.constexpr):
    acc = air.ChannelGet("whole", [2 * T], tl.bfloat16)
    acc += air.ChannelGet("whole", [2 * T], tl.bfloat16)
    acc += air.ChannelGet("whole", [2 * T], tl.bfloat16)
    tl.store(Y + tl.arange(0, 2 * T), acc)


@triton.jit
def gather(X, Y, NW: tl.constexpr, D: tl.constexpr, T: tl.constexpr, S: tl.constexpr):
    air.Channel("halves", size=[2])
    air.Channel("whole")
    for wave in range(NW):
        air.herd(half, (X, D, T), 2, S, name="half")
        buf = air.alloc([2 * T], tl.bfloat16, memory_space=air.L2, memtile_col=3)
        for h in tl.static_range(2):
            air.ChannelGet("halves", buf, indices=[h], offsets=[h * T], sizes=[T])
        for r in range(3):
            air.ChannelPut("whole", buf, offsets=[0], sizes=[2 * T])
        air.herd(use3, (Y, D, T), 1, S, name="use")


@triton.jit
def gather_dynamic(
    X, Y, NW: tl.constexpr, D: tl.constexpr, T: tl.constexpr, S: tl.constexpr
):
    air.Channel("halves", size=[2])
    air.Channel("whole")
    for wave in range(NW):
        air.herd(half, (X, D, T), 2, S, name="half")
        buf = air.alloc([2 * T], tl.bfloat16, memory_space=air.L2)
        for h in range(2):
            air.ChannelGet("halves", buf, indices=[h], offsets=[h * T], sizes=[T])
        air.herd(use3, (Y, D, T), 1, S, name="use")


@triton.jit
def dbl(D: tl.constexpr, T: tl.constexpr):
    for t in range(D // T):
        x = air.ChannelGet("inp", [T], tl.bfloat16)
        air.ChannelPut("out", x * 2)


@triton.jit
def feed(X, Y, NW: tl.constexpr, D: tl.constexpr, T: tl.constexpr, S: tl.constexpr):
    air.Channel("inp")
    air.Channel("out")
    for wave in range(NW):
        for t in range(D // T):
            air.ChannelPut("inp", X, offsets=[t * T], sizes=[T])
        air.herd(dbl, (D, T), 1, S, name="dbl")
        for t in range(D // T):
            air.ChannelGet("out", Y, offsets=[t * T], sizes=[T])


@triton.jit
def wr(X, Sh, D: tl.constexpr):
    tl.store(Sh + tl.arange(0, D), tl.load(X + tl.arange(0, D)) * 2)


@triton.jit
def rd(Sh, Y, D: tl.constexpr):
    tl.store(Y + tl.arange(0, D), tl.load(Sh + tl.arange(0, D)) + 1)


@triton.jit
def neighbours(
    X, Y, NW: tl.constexpr, D: tl.constexpr, T: tl.constexpr, S: tl.constexpr
):
    sh = air.alloc([D], tl.bfloat16, memory_space=air.L1)
    for wave in range(NW):
        air.herd(wr, (X, sh, D), 1, S, name="wr", at=(2, 2))
        air.herd(rd, (sh, Y, D), 1, S, name="rd", at=(3, 2))


@triton.jit
def q4_group(xq, w, acc, t, sx, g):
    """Group g (32 columns) of a q4 block at w (2560 int16: scales [8][32]
    and mins [8][32] as bf16, then per group 8 words a row, word j of a row
    holding its columns 4 j .. 4 j + 3 at bits 0, 4, 8 and 12) times x, added
    to the 32 rows' sums at acc. xq [4][256] holds x times 2^(-4 s) for s =
    0 .. 3, then 32 elements in which a value of x is repeated; sx each
    group's sum of x; t is a buffer for the group's sums."""
    r = tl.arange(0, 32)
    xv = xq + 1024
    tl.store(t + r, tl.zeros([32], tl.float32))
    for j in range(8):
        v = tl.load(w + 512 + g * 256 + j * 32 + r)
        tj = tl.load(t + r)
        for s in tl.static_range(4):
            # Column 4 j + s times 2^(4 s), exactly, as bf16. The top nibble
            # is masked with bit 15 flipped, so that the masked word keeps
            # its sign as an int16, and the flip added back.
            if s == 3:
                q = ((v ^ -32768) & -4096).to(tl.bfloat16) + 32768.0
            else:
                q = (v & (15 << (4 * s))).to(tl.bfloat16)
            xs = tl.load(xq + s * 256 + g * 32 + j * 4 + s)
            tl.store(xv + r, tl.full([32], xs, tl.bfloat16))
            tj += q.to(tl.float32) * tl.load(xv + r).to(tl.float32)
        tl.store(t + r, tj)
    part = tl.load(t + r).to(tl.bfloat16).to(tl.float32)
    tl.store(xv + r, tl.full([32], tl.load(sx + g).to(tl.bfloat16), tl.bfloat16))
    red = tl.load(xv + r).to(tl.float32)
    scale = tl.load(w + g * 32 + r).to(tl.bfloat16, bitcast=True).to(tl.float32)
    mn = tl.load(w + 256 + g * 32 + r).to(tl.bfloat16, bitcast=True).to(tl.float32)
    tl.store(acc + r, tl.load(acc + r) + part * scale + mn * red)


@triton.jit
def q4_block(xb, w, acc, xq, t, sx):
    """A q4 block at w times the 256 x at xb, added to acc."""
    k = tl.arange(0, 256)
    xs = tl.reshape(tl.load(xb + k), [8, 32]).to(tl.float32)
    tl.store(sx + tl.arange(0, 8), tl.sum(xs, axis=1))
    for s in tl.static_range(4):
        xs4 = tl.load(xb + k).to(tl.float32) * (1.0 / (1 << (4 * s)))
        tl.store(xq + s * 256 + k, xs4.to(tl.bfloat16))
    for g in range(8):
        q4_group(xq, w, acc, t, sx, g)


@triton.jit
def gemv_core(NB: tl.constexpr):
    w = air.alloc([2560], tl.int16, memory_space=air.L1)
    xb = air.alloc([256], tl.bfloat16, memory_space=air.L1)
    acc = air.alloc([32], tl.float32, memory_space=air.L1)
    xq = air.alloc([1056], tl.bfloat16, memory_space=air.L1)
    t = air.alloc([32], tl.float32, memory_space=air.L1)
    sx = air.alloc([8], tl.float32, memory_space=air.L1)
    r = tl.arange(0, 32)
    tl.store(acc + r, tl.zeros([32], tl.float32))
    for j in range(NB):
        air.ChannelGet("inX", xb, sizes=[256])
        air.ChannelGet("inW", w, sizes=[2560])
        q4_block(xb, w, acc, xq, t, sx)
    air.ChannelPut("outY", tl.load(acc + r).to(tl.bfloat16))


@triton.jit
def q4gemv(
    X, Y, W, NW: tl.constexpr, D: tl.constexpr, T: tl.constexpr, S: tl.constexpr
):
    air.backend(
        use_lock_race_condition_fix_v2=True,
        merge_contiguous_feeds=False,
        stack_size=4096,
    )
    air.Channel("inX")
    air.Channel("inW")
    air.Channel("outY")
    for wave in range(NW):
        for j in range(D // T):
            air.ChannelPut("inX", X, offsets=[j * 256], sizes=[256])
            air.ChannelPut("inW", W, offsets=[j * 2560], sizes=[2560])
        air.herd(gemv_core, (D // T,), 1, S, name="gemv")
        air.ChannelGet("outY", Y, offsets=[0], sizes=[32])


@triton.jit
def proj_core(CX0: tl.constexpr, I2: tl.constexpr, J2: tl.constexpr):
    """Rows i 512 + cx 128 + cy 32 + [0, 32) of round i, over 2 J2 blocks of
    256 columns, sent to consumer i % 3; core (cx, cy) of the 4x4 cores of
    two herds."""
    cx = CX0 + tl.program_id(0)
    cy = tl.program_id(1)
    w = air.alloc([2560], tl.int16, memory_space=air.L1)
    xb = air.alloc([256], tl.bfloat16, memory_space=air.L1)
    acc = air.alloc([32], tl.float32, memory_space=air.L1)
    xq = air.alloc([1056], tl.bfloat16, memory_space=air.L1)
    t = air.alloc([32], tl.float32, memory_space=air.L1)
    sx = air.alloc([8], tl.float32, memory_space=air.L1)
    # The rows at [32, 64), where a vector store is aligned; the packet's
    # header in the word before them.
    y = air.alloc([64], tl.bfloat16, memory_space=air.L1)
    r = tl.arange(0, 32)
    for i in range(I2):
        tl.store(acc + r, tl.zeros([32], tl.float32))
        for j in range(2 * J2):
            air.ChannelGet("inX", xb, indices=[cx, cy], sizes=[256])
            air.ChannelGet("wL2ToL1", w, indices=[cx, cy], sizes=[2560])
            q4_block(xb, w, acc, xq, t, sx)
        tl.store(y + 32 + r, tl.load(acc + r).to(tl.bfloat16))
        air.ChannelPut(
            "outA", y, indices=[cx, cy], offsets=[30], sizes=[34], dest=i % 3
        )


@triton.jit
def consumer(D: tl.constexpr, N: tl.constexpr):
    """Consumer D's N rounds, doubled, plus 64 D, to the host."""
    for n in range(N):
        v = air.ChannelGet("outY", [512], tl.bfloat16, indices=[0, D])
        v = (v + v) + 64.0 * D
        if D == 0:
            air.ChannelPut("res0", v)
        if D == 1:
            air.ChannelPut("res1", v)
        if D == 2:
            air.ChannelPut("res2", v)


@triton.jit
def proj16(X, Y, W, I2: tl.constexpr, J2: tl.constexpr, S: tl.constexpr):
    """y = W x on 16 cores, 512 rows a round, round i to consumer i % 3: x
    lands in memtile 3 and is broadcast to every core; each of 8 weight
    streams fills a memtile's buffer with a block for each of two cores;
    each column's memtile gathers its 4 cores' rows behind the first one's
    packet header, memtile 2 the 4 columns', and the header picks the
    consumer the round goes to."""
    air.backend(
        use_lock_race_condition_fix_v2=True,
        merge_contiguous_feeds=False,
        stack_size=4096,
    )
    air.Channel("xnorm")
    air.Channel("inX", size=[1, 1], broadcast_shape=[4, 4])
    air.Channel("inW", size=[4, 2])
    air.Channel("wL2ToL1", size=[4, 4])
    air.Channel("outA", size=[4, 4], channel_type="npu_dma_packet")
    air.Channel("toMain", size=[4], channel_type="npu_dma_packet")
    air.Channel(
        "outY", size=[1, 1], broadcast_shape=[1, 3], channel_type="npu_dma_packet"
    )
    air.Channel("res0")
    air.Channel("res1")
    air.Channel("res2")
    NF: tl.constexpr = I2 * 2 * J2  # fills of each weight stream
    for i in range(I2):
        air.ChannelPut("xnorm", X, offsets=[0], sizes=[512 * J2])
    for x in tl.static_range(4):
        for k in tl.static_range(2):
            base = (x * 2 + k) * NF * 5120
            air.ChannelPut("inW", W, indices=[x, k], offsets=[base], sizes=[NF * 5120])
    for n in range(I2 * J2):
        xl = air.alloc([512], tl.bfloat16, memory_space=air.L2, memtile_col=3)
        air.ChannelGet("xnorm", xl, offsets=[0], sizes=[512])
        for h in tl.static_range(2):
            air.ChannelPut("inX", xl, indices=[0, 0], offsets=[h * 256], sizes=[256])
    for x in tl.static_range(4):
        for k in tl.static_range(2):
            for n in range(NF):
                # in the memtile under the cores: columns 0, 1, 6 and 7
                wf = air.alloc(
                    [5120], tl.int16, memory_space=air.L2, memtile_col=x + (x // 2) * 4
                )
                air.ChannelGet("inW", wf, indices=[x, k], offsets=[0], sizes=[5120])
                for c in tl.static_range(2):
                    air.ChannelPut(
                        "wL2ToL1",
                        wf,
                        indices=[x, 2 * k + c],
                        offsets=[c * 2560],
                        sizes=[2560],
                    )
    air.herd(proj_core, (0, I2, J2), (2, 4), S, name="proj_blk0", at=(0, 2))
    air.herd(proj_core, (2, I2, J2), (2, 4), S, name="proj_blk1", at=(6, 2))
    for i in range(I2):
        for x in tl.static_range(4):
            g = air.alloc(
                [130], tl.bfloat16, memory_space=air.L2, memtile_col=x + (x // 2) * 4
            )
            # the first row's packet with its header, the others' without
            air.ChannelGet("outA", g, indices=[x, 0], offsets=[0], sizes=[34])
            for c in tl.static_range(1, 4):
                air.ChannelGet(
                    "outA", g, indices=[x, c], offsets=[2 + c * 32], sizes=[32]
                )
            air.ChannelPut("toMain", g, indices=[x], offsets=[0], sizes=[130])
        m = air.alloc([514], tl.bfloat16, memory_space=air.L2, memtile_col=2)
        air.ChannelGet("toMain", m, indices=[0], offsets=[0], sizes=[130])
        for x in tl.static_range(1, 4):
            air.ChannelGet("toMain", m, indices=[x], offsets=[2 + x * 128], sizes=[128])
        air.ChannelPut("outY", m, indices=[0, 0], offsets=[0], sizes=[514])
    air.herd(consumer, (0, (I2 + 2) // 3), 1, S, name="rope", at=(2, 3))
    air.herd(consumer, (1, (I2 + 1) // 3), 1, S, name="rms", at=(2, 2))
    air.herd(consumer, (2, I2 // 3), 1, S, name="glu", at=(5, 3))
    # consumer d's rounds at Y[d NC 512 ...]
    NC: tl.constexpr = (I2 + 2) // 3
    for n in range((I2 + 2) // 3):
        air.ChannelGet("res0", Y, offsets=[n * 512], sizes=[512])
    for n in range((I2 + 1) // 3):
        air.ChannelGet("res1", Y, offsets=[(NC + n) * 512], sizes=[512])
    for n in range(I2 // 3):
        air.ChannelGet("res2", Y, offsets=[(2 * NC + n) * 512], sizes=[512])


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
    "gather": (
        gather,
        2 * XT,
        2 * XT,
        lambda x: bf(bf(bf(2 * x) * 2) + bf(2 * x)),
        [
            "air.channel.get  @halves[",
            "air.channel.put  @whole[] (%alloc",
            "{air.memtile_col = 3 : i32}",
        ],
    ),
    "neighbours": (
        neighbours,
        XD,
        XD,
        lambda x: bf(bf(2 * x) + 1),
        ["memref<1024xbf16, 2 : i32>, memref<*xbf16>"],
    ),
    "feed": (
        feed,
        XD,
        XD,
        lambda x: bf(2 * x),
        ["air.channel.put  @inp[] (%arg", "air.channel.get  @out[] (%arg"],
    ),
    "grid": (
        grid,
        4 * XD,
        4 * XD,
        lambda x: bf(bf(2 * x) + 1),
        ["air.channel @cells [1, 2, 2]", "@cells[%c0"],
    ),
}


def q4_pack(q, scale, mn):
    """q [32 rows, 256 columns] of 0..15 and scale, mn [8 groups, 32 rows]:
    the 2560 int16 of a q4 block, as q4_group reads it."""
    s16 = lambda a: (bf(a).view(np.uint32) >> 16).astype(np.uint16)  # noqa: E731
    out = np.zeros(2560, np.uint16)
    out[:256], out[256:512] = s16(scale).reshape(-1), s16(mn).reshape(-1)
    words = np.zeros((8, 8, 32), np.uint16)  # [group][j][row]
    for g in range(8):
        for j in range(8):
            for s in range(4):
                words[g, j] |= q[:, g * 32 + 4 * j + s].astype(np.uint16) << (4 * s)
    out[512:] = words.reshape(-1)
    return out.view(np.int16)


def q4_ref(x, q, scale, mn):
    """A q4nx block's 32 rows times x, rounded as the decode's cores do."""
    xg = x.reshape(8, 32)
    part = bf(np.einsum("gcr,gc->gr", q.T.reshape(8, 32, 32).astype(np.float32), xg))
    return (part * scale + mn * bf(xg.sum(1))[:, None]).sum(0)


def run_q4gemv(script, has_npu, proj):
    nb = 6
    rng = np.random.default_rng(2)
    x0 = bf(rng.standard_normal(nb * 256))
    blocks = [
        (
            rng.integers(0, 16, (32, 256)),
            bf(rng.uniform(0.01, 0.1, (8, 32))),
            bf(rng.uniform(-0.5, 0, (8, 32))),
        )
        for _ in range(nb)
    ]
    W = torch.from_numpy(np.concatenate([q4_pack(*b) for b in blocks]))
    want = bf(
        sum(q4_ref(x0[j * 256 : (j + 1) * 256], *b) for j, b in enumerate(blocks))
    )
    x = _bfx(x0)
    y = torch.zeros(32, dtype=torch.bfloat16)
    kw = dict(NW=1, D=nb * 256, T=256, S=script)
    text = module_text(q4gemv, (x, y, W), kw, proj)
    check(
        "air.channel.get  @inW[%c0" in text,
        "q4gemv: weights get into the core's buffer",
    )
    if not has_npu:
        return
    from triton.backends.amd_triton_npu.driver_scope import driver_scope

    with driver_scope("npu"):
        q4gemv[(1,)](x, y, W, **kw)
    got = y.to(torch.float32).numpy()
    ulp = np.abs(want) * 2.0**-7 + 1e-6
    check(
        bool(np.all(np.abs(got - want) <= ulp)),
        "q4gemv: NPU result matches numpy to one ulp",
    )


def run_proj16(script, has_npu, proj):
    i2, j2 = 6, 1
    rng = np.random.default_rng(3)
    x0 = bf(rng.standard_normal(512 * j2))
    nrb, ncb = i2 * 16, 2 * j2
    blk = [
        [
            (
                rng.integers(0, 16, (32, 256)),
                bf(rng.uniform(0.01, 0.1, (8, 32))),
                bf(rng.uniform(-0.5, 0, (8, 32))),
            )
            for _ in range(ncb)
        ]
        for _ in range(nrb)
    ]
    rows = np.concatenate(
        [
            bf(sum(q4_ref(x0[j * 256 : (j + 1) * 256], *blk[b][j]) for j in range(ncb)))
            for b in range(nrb)
        ]
    )
    nc = (i2 + 2) // 3
    want = np.zeros(3 * nc * 512, np.float32)
    for i in range(i2):
        d, n = i % 3, i // 3
        part = bf(bf(2 * rows[i * 512 : (i + 1) * 512]) + 64.0 * d)
        want[(d * nc + n) * 512 : (d * nc + n + 1) * 512] = part
    # Stream (x, k): per round i and column block j, the blocks of cores
    # (x, 2 k) and (x, 2 k + 1), whose rows are i 512 + x 128 + cy 32.
    order = [
        (i * 16 + x * 4 + 2 * k + c, j)
        for x in range(4)
        for k in range(2)
        for i in range(i2)
        for j in range(ncb)
        for c in range(2)
    ]
    W = torch.from_numpy(np.concatenate([q4_pack(*blk[b][j]) for b, j in order]))
    x = _bfx(x0)
    y = torch.zeros(3 * nc * 512, dtype=torch.bfloat16)
    kw = dict(I2=i2, J2=j2, S=script)
    text = module_text(proj16, (x, y, W), kw, proj)
    check(
        "air.channel @inX [1, 1] {broadcast_shape = [4, 4]}" in text
        and "air.herd @proj_blk1" in text,
        "proj16: x broadcast to two 2x4 herds",
    )
    check(text.count(" dest(") == 2, "proj16: the cores' puts pick the consumer")
    if not has_npu:
        return
    from triton.backends.amd_triton_npu.driver_scope import driver_scope

    with driver_scope("npu"):
        proj16[(1,)](x, y, W, **kw)
    got = y.to(torch.float32).numpy()
    ulp = np.abs(want) * 2.0**-7 + 1e-6
    check(
        bool(np.all(np.abs(got - want) <= ulp)),
        "proj16: each consumer gets its rounds, to one ulp",
    )


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


def error_checks(script, direct, proj):
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
    explicit = dict(NW=2, D=XD, T=XT)
    y = torch.empty(2 * XT, dtype=torch.bfloat16)
    try:
        module_text(
            gather_dynamic, (y, y), dict(explicit, S=direct), os.path.join(proj, "dyn")
        )
        check(False, "dynamic segment indices: no error")
    except ValueError as e:
        check("constant indices" in str(e), f"dynamic segment indices: {e}")
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
        vector = os.path.join(tmp, "vector.mlir")
        with open(vector, "w") as f:
            f.write(VECTOR)
        with config_context(bf16_emulation=False):
            error_checks(script, direct, os.path.join(tmp, "bad"))
            for name in EXPLICIT:
                print(f"{name}:", flush=True)
                run_explicit(name, direct, has_npu, os.path.join(tmp, name))
            print("q4gemv:", flush=True)
            run_q4gemv(vector, has_npu, os.path.join(tmp, "q4gemv"))
            print("proj16:", flush=True)
            run_proj16(vector, has_npu, os.path.join(tmp, "proj16"))
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
