// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// Weighted RMS Norm transform for AIE2P, following mlir-air xrt 43_triton_layernorm/transform_aie2p.mlir.
// Chain (after fuse_elementwise + transpose_reduce): generic_sq -> reduce -> output_generic (consumes W).
// Hybrid: bufferize_to_allocation for fills/generics/reduce; linalg_promote for W only (post-bufferize).

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {

    // PHASE 1: canonicalize + fold unit extent
    %func0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func0 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
    } : !transform.any_op
    transform.apply_cse to %func0 : !transform.any_op

    // PHASE 2: fuse elementwise + transpose reduce + canonicalize
    %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fused_func = transform.air.fuse_elementwise_linalg %func1 : (!transform.any_op) -> !transform.any_op
    %reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tr = transform.air.transpose_reduce %reduces : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %fused_func {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %fused_func : !transform.any_op

    // Data-flow navigation. Chain: generic_sq -> reduce -> output_generic (consumes W)
    %r = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %generic_sq = transform.get_producer_of_operand %r[0] : (!transform.any_op) -> !transform.any_op
    %materialize = transform.structured.match ops{["bufferization.materialize_in_destination"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %output_generic = transform.get_producer_of_operand %materialize[0] : (!transform.any_op) -> !transform.any_op
    %fill = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    // PHASE 3: L2 alloc for output, tile, fuse backward
    %ob, %on = transform.structured.bufferize_to_allocation %output_generic
        {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op
    %tiled_output, %forall = transform.structured.tile_using_forall %output_generic tile_sizes [1]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %fr, %fl_r = transform.structured.fuse_into_containing_op %r into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fg, %fl_g = transform.structured.fuse_into_containing_op %generic_sq into %fl_r : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %ff, %fl_f = transform.structured.fuse_into_containing_op %fill into %fl_g : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // PHASE 4: canonicalize
    %func2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func2 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func2 : !transform.any_op

    // PHASE 5: L1 alloc for fills + intermediate ops + X promotion
    %fills_2 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fb, %fn = transform.structured.bufferize_to_allocation %fills_2
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    %generics2 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tiled_generic1, %tiled_generic2 = transform.split_handle %generics2 : (!transform.any_op<"linalg.generic">) -> (!transform.any_op<"linalg.generic">, !transform.any_op<"linalg.generic">)
    %reduces2 = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    // Promote X (input of generic_sq, operand 0) to L1
    %op0 = transform.get_operand %tiled_generic1[0] : (!transform.any_op) -> !transform.any_value
    transform.structured.promote_tensor to 2 %op0 : !transform.any_value

    // L1 alloc for intermediate outputs (W stays at function input until post-bufferize linalg_promote)
    %g1b, %g1n = transform.structured.bufferize_to_allocation %tiled_generic1
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op
    %rb, %rn = transform.structured.bufferize_to_allocation %reduces2
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op
    %g2b, %g2n = transform.structured.bufferize_to_allocation %tiled_generic2
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // PHASE 6: canonicalize
    %func5 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func5 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func5 : !transform.any_op

    // PHASE 7: one_shot_bufferize
    transform.include @one_shot_bufferize failures(propagate) (%arg1) : (!transform.any_op) -> ()

    // PHASE 8: canonicalize + remove uninitialized copy
    %func6 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func6 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func6 : !transform.any_op
    transform.apply_patterns to %func6 {
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    %func_op_updated = transform.air.remove_uninitialized_copy %func6 : (!transform.any_op) -> !transform.any_op

    // PHASE 8.5: linalg_promote to add L1 staging for W (post-bufferize).
    // The output_generic now has memref operands ins(X_L1, reduced_L1, W_func_input).
    // Promote operand 2 (W) only - other operands are already L1.
    // Restrict match to the forall body so unrelated post-bufferize generics
    // (e.g. memcpy-like) can't poison the split.
    %forall_buf = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %generics_in_forall = transform.structured.match ops{["linalg.generic"]} in %forall_buf : (!transform.any_op) -> !transform.any_op
    %sq_buf, %out_buf = transform.split_handle %generics_in_forall : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %w_promoted = transform.air.linalg_promote %out_buf {memory_space = "L1", operands_to_promote = [2]} : (!transform.any_op) -> !transform.any_op

    // Canonicalize to fold any self-copies linalg_promote may have introduced.
    %fp = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %fp { transform.apply_patterns.canonicalization } : !transform.any_op
    transform.apply_cse to %fp : !transform.any_op

    // PHASE 9: generalize remaining linalg.reduce, tile for vectorization, divf-sqrt -> rsqrt
    %remaining_reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %generalized = transform.structured.generalize %remaining_reduces : (!transform.any_op) -> !transform.any_op

    %lg = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %inner, %vl:1 = transform.structured.tile_using_for %lg tile_sizes [0, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %fou1 = transform.air.convert_divf_sqrt_to_rsqrt %func_op_updated : (!transform.any_op) -> !transform.any_op

    // PHASE 10: par_to_herd, copy_to_dma, herd_vectorize, casts
    %fa = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %parallel = transform.loop.forall_to_parallel %fa : (!transform.any_op) -> !transform.any_op
    %herd = transform.air.par_to_herd %parallel : (!transform.any_op) -> !transform.any_op

    %copies_in_herd = transform.structured.match ops{["memref.copy", "linalg.copy"]} in %herd : (!transform.any_op) -> !transform.any_op
    %dmas = transform.air.copy_to_dma %copies_in_herd : (!transform.any_op) -> !transform.any_op

    %vh = transform.air.herd_vectorize %herd : (!transform.any_op) -> !transform.any_op

    %func4 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func4 {
        transform.apply_patterns.canonicalization
        transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op

    %vh2 = transform.air.broadcast_before_unary %func4 {op_name = "math.rsqrt"} : (!transform.any_op) -> !transform.any_op

    %vector_reductions = transform.structured.match ops{["vector.multi_reduction"]} in %vh2 : (!transform.any_op) -> !transform.any_op
    %r1 = transform.air.vector_type_cast %vector_reductions {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op

    %vector_muls = transform.structured.match ops{["arith.mulf"]} in %vh2 : (!transform.any_op) -> !transform.any_op
    %r2 = transform.air.vector_type_cast %vector_muls {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op

    %func7 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func7t = transform.air.convert_size1_vector_to_scalar %func7 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func7t {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
        transform.apply_patterns.vector.reorder_multi_reduction_dims lowering_strategy = "innerreduction"
        transform.apply_patterns.vector.multi_reduction_flattening lowering_strategy = "innerreduction"
        transform.apply_patterns.vector.multi_reduction_unrolling lowering_strategy = "innerreduction"
    } : !transform.any_op
    transform.apply_cse to %func7t : !transform.any_op

    transform.yield
  }
}
