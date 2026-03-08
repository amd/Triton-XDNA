// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// RMS Norm transform for AIE2P. Requires mlir-air >= 1b0ae6e.
// y = x * rsqrt(mean(x^2) + eps)
//
// Uses L1 alloc approach (no pad) to avoid self-DMA from copy-back.
// The bufferize_to_allocation {memory_space=2} on the generic inside the
// forall creates explicit L1 allocs with copy-in/copy-out that become
// proper L2↔L1 DMAs.

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

    // Phase 2: Fuse post-reduce chain into single bf16→bf16 generic
    %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func1_fused = transform.air.fuse_elementwise_linalg %func1 : (!transform.any_op) -> !transform.any_op
    %func1a = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func1a { transform.apply_patterns.canonicalization } : !transform.any_op
    transform.apply_cse to %func1a : !transform.any_op

    // Phase 3: Tile output with forall [32] (2 tiles for 64 elements)
    %all_gens = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %other_gens, %last_gen = transform.split_handle %all_gens {overflow_result = 0}
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_op, %forall =
      transform.structured.tile_using_forall %last_gen tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Canonicalize
    %func_2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_2 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_2 : !transform.any_op

    // Phase 4: Promote generic inside forall to L1 (no pad needed)
    %gen_in_forall = transform.structured.match ops{["linalg.generic"]} in %forall : (!transform.any_op) -> !transform.any_op
    %gen_l1_buf, %gen_l1_new = transform.structured.bufferize_to_allocation %gen_in_forall
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    %func_3 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_3 { transform.apply_patterns.canonicalization } : !transform.any_op
    transform.apply_cse to %func_3 : !transform.any_op

    // Phase 5: Bufferize
    // Memory spaces for remaining allocs handled by post-transform overrides
    // in driver.py (scope=herd→2, scope=func→1 with exclusive scopes).
    %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func_bufferized = transform.bufferization.one_shot_bufferize %func_op : (!transform.any_op) -> !transform.any_op

    // Phase 6: Cleanup
    %func6 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func6 { transform.apply_patterns.canonicalization } : !transform.any_op
    transform.apply_cse to %func6 : !transform.any_op
    %lc = transform.structured.match ops{["linalg.copy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %mc = transform.structured.linalg_copy_to_memref %lc : (!transform.any_op) -> !transform.any_op
    %func_upd = transform.air.remove_uninitialized_copy %func6 : (!transform.any_op) -> !transform.any_op
    %func_upd2 = transform.air.eliminate_cascade_memcpy %func_upd : (!transform.any_op) -> !transform.any_op

    // Phase 7: Scalarize
    %all_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %gen_loops = transform.structured.convert_to_loops %all_generics : (!transform.any_op) -> !transform.any_op
    %all_reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %red_loops = transform.structured.convert_to_loops %all_reduces : (!transform.any_op) -> !transform.any_op
    %all_fills = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fill_loops = transform.structured.convert_to_loops %all_fills : (!transform.any_op) -> !transform.any_op

    // Phase 8: Forall → Herd
    %forall_h = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %par = transform.loop.forall_to_parallel %forall_h : (!transform.any_op) -> !transform.any_op
    %herd = transform.air.par_to_herd %par : (!transform.any_op) -> !transform.any_op

    // DMA inside herd
    %lc2 = transform.structured.match ops{["linalg.copy"]} in %herd : (!transform.any_op) -> !transform.any_op
    %mc2 = transform.structured.match ops{["memref.copy"]} in %herd : (!transform.any_op) -> !transform.any_op
    %mc3 = transform.structured.linalg_copy_to_memref %lc2 : (!transform.any_op) -> !transform.any_op
    %all_c = transform.merge_handles %mc2, %mc3 { deduplicate } : !transform.any_op
    %dmas = transform.air.copy_to_dma %all_c : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
