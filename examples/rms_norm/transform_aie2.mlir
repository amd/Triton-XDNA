// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
// Triton RMS Norm Tiling Recipe Transform Script (AIE2P)
//===----------------------------------------------------------------------===//
// Computes: y = x * rsqrt(mean(x^2) + eps)
//
// Linalg IR structure (from triton-shared-opt):
//   1. linalg.generic (mulf x*x : bf16)          -- elementwise square
//   2. linalg.reduce (sum with extf : bf16->f32)  -- reduction
//   3. Scalar chain: extract -> truncf -> extf -> divf -> addf -> rsqrt
//   4. linalg.generic (extf x : bf16->f32)        -- broadcast input
//   5. linalg.fill (broadcast rsqrt to vector)
//   6. linalg.generic (mulf x*rstd : f32)         -- normalize
//   7. linalg.generic (truncf : f32->bf16)         -- output
//
// Single-row kernel (grid = (M,)), no batch tiling needed.
// Just tile for vectorization, handle reduction, map to herd.
//
// AIE2P type constraints:
//   bf16 ONLY: addf, subf, mulf, exp, reductions
//   f32 ONLY:  divf, rsqrt
//===----------------------------------------------------------------------===//

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {

    //===================================================================
    // PHASE 1: Canonicalization
    //===================================================================
    %func0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func0 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
    } : !transform.any_op
    transform.apply_cse to %func0 : !transform.any_op

    //===================================================================
    // PHASE 2: Fuse elementwise chain (post-reduce part)
    //===================================================================
    // Fuse the extf + mulf + truncf chain (ops 4,5,6,7) into a single
    // generic to reduce buffer count, similar to elementwise activation pattern.
    %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func1_fused = transform.air.fuse_elementwise_linalg %func1 : (!transform.any_op) -> !transform.any_op

    %func1a = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func1a {
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func1a : !transform.any_op

    //===================================================================
    // PHASE 3: Navigate IR and set up tiling
    //===================================================================
    // After fuse_elementwise_linalg, structure should be:
    //   linalg.generic (x*x bf16) -> linalg.reduce (f32) -> scalar chain ->
    //   linalg.generic (fused: extf+mulf+truncf, bf16->bf16)

    // Bufferize output to L2
    %generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %sq_op, %fused_output = transform.split_handle %generics
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %output_buf, %new_output = transform.structured.bufferize_to_allocation %fused_output
      {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Tile output with forall [256] for multi-core
    %tiled_output, %forall =
      transform.structured.tile_using_forall %fused_output tile_sizes [256] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fuse the square op and reduce into the forall
    %reduce_op = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tiled_reduce, %_1 = transform.structured.fuse_into_containing_op %reduce_op into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_sq, %_2 = transform.structured.fuse_into_containing_op %sq_op into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    //===================================================================
    // PHASE 4: Canonicalization
    //===================================================================
    %func2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func2 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func2 : !transform.any_op

    //===================================================================
    // PHASE 5: L1 Memory Allocation
    //===================================================================
    // Allocate intermediates to L1
    %generics2 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %gen_buf, %gen_new = transform.structured.bufferize_to_allocation %generics2
      {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    //===================================================================
    // PHASE 6: Canonicalization
    //===================================================================
    %func5 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func5 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func5 : !transform.any_op

    //===================================================================
    // PHASE 7: Bufferization
    //===================================================================
    %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func_bufferized = transform.bufferization.one_shot_bufferize %func_op : (!transform.any_op) -> !transform.any_op

    //===================================================================
    // PHASE 8: Post-Bufferization Cleanup
    //===================================================================
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

    //===================================================================
    // PHASE 9: Vectorization Tiling
    //===================================================================
    // Tile generics for 16-lane vectorization
    %linalg_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %inner_generics, %gen_loops:1 =
      transform.structured.tile_using_for %linalg_generics tile_sizes [16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Tile reduce for 16-lane vectorization
    %linalg_reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %inner_reduces, %red_loops:1 =
      transform.structured.tile_using_for %linalg_reduces tile_sizes [16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fill: scalar
    %linalg_fills = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fill_loops = transform.structured.convert_to_loops %linalg_fills : (!transform.any_op) -> !transform.any_op

    //===================================================================
    // PHASE 10: AIR Constructs Mapping
    //===================================================================
    %forall_as_herd = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %parallel = transform.loop.forall_to_parallel %forall_as_herd : (!transform.any_op) -> !transform.any_op
    %herd = transform.air.par_to_herd %parallel : (!transform.any_op) -> !transform.any_op

    %linalg_copies_in_herd = transform.structured.match ops{["linalg.copy"]} in %herd : (!transform.any_op) -> !transform.any_op
    %memref_copies_in_herd = transform.structured.match ops{["memref.copy"]} in %herd : (!transform.any_op) -> !transform.any_op
    %memref_copies_from_linalg = transform.structured.linalg_copy_to_memref %linalg_copies_in_herd : (!transform.any_op) -> !transform.any_op
    %all_copies = transform.merge_handles %memref_copies_in_herd, %memref_copies_from_linalg { deduplicate } : !transform.any_op
    %dmas = transform.air.copy_to_dma %all_copies : (!transform.any_op) -> !transform.any_op

    // Vectorize
    %vectorized_herd = transform.air.herd_vectorize %herd : (!transform.any_op) -> !transform.any_op

    // AIE2P bf16 type casts (same as elementwise pattern)
    // mulf (x*x and x*rstd) -> bf16
    %vector_muls = transform.structured.match ops{["arith.mulf"]} in %vectorized_herd : (!transform.any_op) -> !transform.any_op
    %mul_cast = transform.air.vector_type_cast %vector_muls {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op

    // Cast vector reductions to bf16
    %vector_reductions = transform.structured.match ops{["vector.multi_reduction"]} in %vectorized_herd : (!transform.any_op) -> !transform.any_op
    %red_cast = transform.air.vector_type_cast %vector_reductions {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op

    // divf and rsqrt stay f32 (AIE2P native f32)

    // Convert size-1 vectors to scalars and clean up
    %func7 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func7_t = transform.air.convert_size1_vector_to_scalar %func7 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func7_t {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
        transform.apply_patterns.vector.cast_away_vector_leading_one_dim
        transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerreduction"
    } : !transform.any_op
    transform.apply_cse to %func7_t : !transform.any_op

    transform.yield
  }
}
