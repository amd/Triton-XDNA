// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// The attention cores' schedule: proj_aie2p.mlir's, and matrix products as
// vector contractions (the scores' and sums' 4x8x8 blocks).
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
  // A 2-D op wider than 8 (the products' 4x8 and 8x8 operands are a vector
  // each, and their copies go only whole).
  transform.named_sequence @match_rank2_wide(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %m = transform.include @match_rank2 failures(propagate) (%op) : (!transform.any_op) -> !transform.any_op
    %d = transform.match.structured failures(propagate) %op : (!transform.any_op) -> !transform.param<i64> {
    ^bb0(%s: !transform.any_op):
      %d1 = transform.match.structured.dim %s[1] : (!transform.any_op) -> !transform.param<i64>
      transform.match.structured.yield %d1 : !transform.param<i64>
    }
    %c8 = transform.param.constant 8 : i64 -> !transform.param<i64>
    transform.match.param.cmpi gt %d, %c8 : !transform.param<i64>
    transform.yield %op : !transform.any_op
  }
  // A matrix product of a transpose.
  transform.named_sequence @match_mm_trans_rhs(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %op ["linalg.matmul"] : !transform.any_op
    %p = transform.get_producer_of_operand %op[1] : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name %p ["linalg.transpose"] : !transform.any_op
    transform.yield %op : !transform.any_op
  }
  transform.named_sequence @read_rhs_transposed(%op: !transform.any_op {transform.consumed}) {
    %t = transform.structured.transpose_matmul %op <rhs> : (!transform.any_op) -> !transform.any_op
    transform.yield
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
    // a product of a transposed operand reads it transposed: the transpose
    // this adds cancels the kernel's
    %mt = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %mt1 = transform.foreach_match in %mt @match_mm_trans_rhs -> @read_rhs_transposed : (!transform.any_op) -> !transform.any_op
    %ft0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %ft0 { transform.apply_patterns.canonicalization } : !transform.any_op
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
    %h1q = transform.foreach_match in %h1 @match_rank2_wide -> @tile_rows, @match_rank1 -> @tile_32 : (!transform.any_op) -> !transform.any_op
    %h1r = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %h1r {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    // Vectors for the compute only: the copies stay copies, so that the
    // ones into the kernel's own buffers can become its DMAs' targets.
    %vg = transform.structured.match ops{["linalg.generic", "linalg.fill", "linalg.copy", "linalg.matmul", "linalg.matmul_transpose_b"]} in %h1r : (!transform.any_op) -> !transform.any_op
    transform.foreach %vg : !transform.any_op {
    ^bb0(%v: !transform.any_op):
      transform.include @vectorize_one failures(suppress) (%v) : (!transform.any_op) -> ()
    }
    // a matrix product's transposed operand read by the contraction itself
    %hk = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %hk {
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_patterns to %hk {
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    %hc = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.include @post_vectorize_reduce_cleanup failures(propagate) (%hc) : (!transform.any_op) -> ()
    // a reduction to a scalar as a vector.reduction
    %hm = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %hm {
      transform.apply_patterns.vector.multi_reduction_flattening lowering_strategy = "innerreduction"
      transform.apply_patterns.vector.multi_reduction_unrolling lowering_strategy = "innerreduction"
    } : !transform.any_op
    // In the loops of the products: stores read only by later loads become
    // the stored values and buffers left unread go (the copies of the
    // operands and sums), so a product adds into its sums itself, which then
    // stay in registers across the loop. (Not elsewhere: a transfer's block
    // copy must stay.)
    %ct = transform.structured.match ops{["vector.contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %lp = transform.get_parent_op %ct <{deduplicate, op_name = "scf.for"}> : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %lp {
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    transform.memref.erase_dead_alloc_and_stores %lp : (!transform.any_op) -> ()
    %fd = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %fd {
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %fd : !transform.any_op
    %fh = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fh2 = transform.structured.hoist_redundant_vector_transfers %fh : (!transform.any_op) -> !transform.any_op
    %hx = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %hx1 = transform.foreach_match in %hx @match_transcendental -> @action_cast_to_bf16, @match_cmpf -> @action_cast_cmpf_to_bf16 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
