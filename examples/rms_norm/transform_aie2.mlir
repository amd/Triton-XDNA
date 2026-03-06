// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
// Triton RMS Norm Transform Script (AIE2P) - Minimal viable version
//===----------------------------------------------------------------------===//
// y = x * rsqrt(mean(x^2) + eps)
//
// Strategy: tile output with forall[64], fuse mulf/extf/fill into forall,
// leave reduce+sq+scalar chain outside. Bufferize, convert forall to herd.
// No vectorization for now (get correctness first).
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
    // PHASE 2: Tile output and fuse direct predecessors
    //===================================================================
    %generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %sq_op, %extf_op, %mulf_op, %truncf_op = transform.split_handle %generics
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %fill = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    // Bufferize output to L2
    %truncf_buf, %_ = transform.structured.bufferize_to_allocation %truncf_op
      {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Tile output with forall [64] -- single tile for 1x1 herd
    %tiled_truncf, %forall =
      transform.structured.tile_using_forall %truncf_op tile_sizes [64] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fuse tensor-valued predecessors that operate on the same-sized tensors.
    // reduce and sq stay outside -- they compute the scalar rsqrt from full input.
    %fused_mulf, %_1 = transform.structured.fuse_into_containing_op %mulf_op into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_extf, %_2 = transform.structured.fuse_into_containing_op %extf_op into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_fill, %_3 = transform.structured.fuse_into_containing_op %fill into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    //===================================================================
    // PHASE 3: Canonicalization
    //===================================================================
    %func2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func2 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func2 : !transform.any_op

    //===================================================================
    // PHASE 4: Promote alloc_tensors to L1 (prevent memory_space 0)
    //===================================================================
    %at = transform.structured.match ops{["bufferization.alloc_tensor"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %at_buf, %at_new = transform.structured.bufferize_to_allocation %at
        {memory_space = 1, emit_dealloc} : !transform.any_op

    //===================================================================
    // PHASE 5: Bufferization
    //===================================================================
    %func5 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func5 {
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func5 : !transform.any_op

    %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func_bufferized = transform.bufferization.one_shot_bufferize %func_op : (!transform.any_op) -> !transform.any_op

    //===================================================================
    // PHASE 6: Post-Bufferization Cleanup
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
    %linalg_copies = transform.structured.match ops{["linalg.copy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %memref_conv = transform.structured.linalg_copy_to_memref %linalg_copies : (!transform.any_op) -> !transform.any_op
    %func_upd = transform.air.remove_uninitialized_copy %func6 : (!transform.any_op) -> !transform.any_op

    //===================================================================
    // PHASE 7: Scalarize all linalg ops (no vectorization for now)
    //===================================================================
    %all_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %gen_loops = transform.structured.convert_to_loops %all_generics : (!transform.any_op) -> !transform.any_op

    %all_reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %red_loops = transform.structured.convert_to_loops %all_reduces : (!transform.any_op) -> !transform.any_op

    %all_fills = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fill_loops = transform.structured.convert_to_loops %all_fills : (!transform.any_op) -> !transform.any_op

    //===================================================================
    // PHASE 8: Convert forall to herd
    //===================================================================
    %forall_h = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %par = transform.loop.forall_to_parallel %forall_h : (!transform.any_op) -> !transform.any_op
    %herd = transform.air.par_to_herd %par : (!transform.any_op) -> !transform.any_op

    // Convert copies to DMA inside herd
    %lc = transform.structured.match ops{["linalg.copy"]} in %herd : (!transform.any_op) -> !transform.any_op
    %mc = transform.structured.match ops{["memref.copy"]} in %herd : (!transform.any_op) -> !transform.any_op
    %mc2 = transform.structured.linalg_copy_to_memref %lc : (!transform.any_op) -> !transform.any_op
    %all_c = transform.merge_handles %mc, %mc2 { deduplicate } : !transform.any_op
    %dmas = transform.air.copy_to_dma %all_c : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
