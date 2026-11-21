// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// The schedule of the decode's cores (attention's aside): a kernel whole on
// one core, its body in a 1x1 herd, every buffer it allocates in the core's
// L1, every copy a DMA of the core.
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
    // Vectors for the compute only: the copies stay copies, so that the
    // ones into the kernel's own buffers can become its DMAs' targets.
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
