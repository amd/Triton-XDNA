// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// RMS Norm transform for AIE2P. Requires mlir-air >= 4a3d9c7.
// y = x * rsqrt(mean(x^2) + eps)
//
// IR after triton-shared-opt (4 generics + 1 reduce + 1 fill):
//   sq(x*x bf16) -> reduce(sum bf16->f32) -> tensor.extract -> divf ->
//   addf -> rsqrt -> extf(x bf16->f32) -> fill(rsqrt) -> mulf(x*rstd) -> truncf
//
// Strategy:
//   1. fuse_elementwise_linalg to merge extf+mulf+truncf (post-reduce output)
//   2. Tile the fused output generic with forall[64] for herd creation
//   3. Fuse fill into forall (clone-and-fuse, uses rsqrt scalar from outside)
//   4. Leave reduce+sq+scalar chain outside forall
//   5. Promote alloc_tensor to L1, bufferize, convert forall->herd
//   6. Scalarize ops inside herd (no vectorization for now)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {

    // Phase 1: Canonicalize
    %func0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func0 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
    } : !transform.any_op
    transform.apply_cse to %func0 : !transform.any_op

    // Phase 2: Fuse post-reduce elementwise chain
    %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func1_fused = transform.air.fuse_elementwise_linalg %func1 : (!transform.any_op) -> !transform.any_op
    %func1a = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func1a { transform.apply_patterns.canonicalization } : !transform.any_op
    transform.apply_cse to %func1a : !transform.any_op

    // Phase 3: Tile the last generic to create a forall for herd.
    // After fuse_elementwise_linalg, match all generics and use the last one.
    %all_gens = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %other_gens, %last_gen = transform.split_handle %all_gens {overflow_result = 0}
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Phase 4: Tile last generic at [32] -> 2-iteration forall
    %tiled_output, %forall =
      transform.structured.tile_using_forall %last_gen tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fuse fill into forall
    %fill = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fused_fill, %_1 = transform.structured.fuse_into_containing_op %fill into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Phase 4b: Pad and promote tiled output to L1 (memory_space 2)
    // This ensures one_shot_bufferize uses L1 allocs inside the forall.
    %padded_op, %pad_op, %_pad = transform.structured.pad %tiled_output {
        padding_values=[0.0 : bf16, 0.0 : bf16],
        padding_dimensions=[0, 1],
        nofold_flags=[1, 1],
        copy_back_op="linalg.copy"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %pad_dps = transform.structured.rewrite_in_destination_passing_style %pad_op : (!transform.any_op) -> !transform.any_op
    %padded_input = transform.get_producer_of_operand %padded_op[0] : (!transform.any_op) -> (!transform.any_op)
    %pi_buf, %pi_new = transform.structured.bufferize_to_allocation %padded_input
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op
    %padded_result = transform.get_producer_of_operand %padded_op[1] : (!transform.any_op) -> (!transform.any_op)
    %pr_buf, %pr_new = transform.structured.bufferize_to_allocation %padded_result
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Phase 5: Canonicalize
    %func2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func2 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func2 : !transform.any_op

    // Phase 6: Promote ALL tensor allocations to L1
    // alloc_tensor (reduce init)
    %at = transform.structured.match ops{["bufferization.alloc_tensor"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %at_buf, %at_new = transform.structured.bufferize_to_allocation %at
        {memory_space = 1, emit_dealloc} : !transform.any_op

    // Promote generics INSIDE the forall to L1 (memory_space 2 -- herd-internal).
    // Use full bufferize_to_allocation (not just destination) to ensure
    // one_shot_bufferize doesn't create memory_space 1 allocs inside the forall.
    %gens_in_forall = transform.structured.match ops{["linalg.generic"]} in %forall : (!transform.any_op) -> !transform.any_op
    %gf_buf, %gf_new = transform.structured.bufferize_to_allocation %gens_in_forall
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Note: fills inside forall don't need separate promotion -- the fill's
    // output is consumed by the generic which was already promoted to L1.

    // Promote generics OUTSIDE the forall to L2 (memory_space 1)
    %remaining_gens = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %rg_buf, %rg_new = transform.structured.bufferize_to_allocation %remaining_gens
        {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Promote fills OUTSIDE the forall to L2
    %remaining_fills = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %rf_buf, %rf_new = transform.structured.bufferize_to_allocation %remaining_fills
        {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Promote reduces to L2
    %remaining_reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %rr_buf, %rr_new = transform.structured.bufferize_to_allocation %remaining_reduces
        {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    %func3 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func3 { transform.apply_patterns.canonicalization } : !transform.any_op
    transform.apply_cse to %func3 : !transform.any_op

    // Phase 7: Bufferize
    %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func_bufferized = transform.bufferization.one_shot_bufferize %func_op : (!transform.any_op) -> !transform.any_op

    // Phase 8: Post-bufferization cleanup
    %func6 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func6 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func6 : !transform.any_op
    %lc = transform.structured.match ops{["linalg.copy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %mc = transform.structured.linalg_copy_to_memref %lc : (!transform.any_op) -> !transform.any_op

    // Phase 9: Convert forall to herd FIRST (before scalarization)
    %forall_h = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %par = transform.loop.forall_to_parallel %forall_h : (!transform.any_op) -> !transform.any_op
    %herd = transform.air.par_to_herd %par : (!transform.any_op) -> !transform.any_op

    // Convert copies to DMA inside herd
    %lc2 = transform.structured.match ops{["linalg.copy"]} in %herd : (!transform.any_op) -> !transform.any_op
    %mc2 = transform.structured.match ops{["memref.copy"]} in %herd : (!transform.any_op) -> !transform.any_op
    %mc3 = transform.structured.linalg_copy_to_memref %lc2 : (!transform.any_op) -> !transform.any_op
    %all_c = transform.merge_handles %mc2, %mc3 { deduplicate } : !transform.any_op
    %dmas = transform.air.copy_to_dma %all_c : (!transform.any_op) -> !transform.any_op

    // Phase 10: Scalarize linalg ops (after herd creation)
    %all_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %gen_loops = transform.structured.convert_to_loops %all_generics : (!transform.any_op) -> !transform.any_op
    %all_reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %red_loops = transform.structured.convert_to_loops %all_reduces : (!transform.any_op) -> !transform.any_op
    %all_fills = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fill_loops = transform.structured.convert_to_loops %all_fills : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
