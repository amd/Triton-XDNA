// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// RMS Norm transform for AIE2P. Requires mlir-air >= 4a3d9c7.
// Same pattern as relu/sigmoid: fuse_elementwise → bufferize L2 → tile forall
// → pad → promote L1 → bufferize → herd. No fill fusion needed.

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

    // Phase 3: Get the fused output generic (last generic, 1 in + 1 out, bf16)
    %all_gens = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %other_gens, %last_gen = transform.split_handle %all_gens {overflow_result = 0}
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Phase 4: Bufferize output to L2, tile, pad, promote -- EXACT relu pattern
    // Step 4a: Bufferize result to L2
    %out_buf, %new_out = transform.structured.bufferize_to_allocation %last_gen
      {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Step 4b: Tile using forall [32] (2 tiles for 64 elements)
    %op_1 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // Match only the output generic (should be the one with bf16→bf16 and 2 operands)
    %sq_gen_2, %out_gen_2 = transform.split_handle %op_1 {overflow_result = 0}
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_op, %forall =
      transform.structured.tile_using_forall %out_gen_2 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Step 4c: Canonicalize
    %func_2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_2 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_2 : !transform.any_op

    // Step 4d: Pad the tiled generic (same as relu: 2 bf16 padding values)
    %op_2 = transform.structured.match ops{["linalg.generic"]} in %forall : (!transform.any_op) -> !transform.any_op
    %padded_op, %pad_op, %__ = transform.structured.pad %op_2 {
        padding_values=[0.0 : bf16, 0.0 : bf16],
        padding_dimensions=[0, 1],
        nofold_flags=[1, 1],
        copy_back_op="linalg.copy"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %pad_dps = transform.structured.rewrite_in_destination_passing_style %pad_op : (!transform.any_op) -> !transform.any_op

    // Step 4e: Promote input to L1 (memory_space 2)
    %padded_input = transform.get_producer_of_operand %padded_op[0] : (!transform.any_op) -> (!transform.any_op)
    %pi_buf, %pi_new = transform.structured.bufferize_to_allocation %padded_input
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Step 4f: Promote output to L1 (memory_space 2)
    %padded_result = transform.get_producer_of_operand %padded_op[1] : (!transform.any_op) -> (!transform.any_op)
    %pr_buf, %pr_new = transform.structured.bufferize_to_allocation %padded_result
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Phase 5: Promote ops outside forall to L2
    %func_3 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_3 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_3 : !transform.any_op

    // alloc_tensor (reduce init)
    %at = transform.structured.match ops{["bufferization.alloc_tensor"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %at_buf, %at_new = transform.structured.bufferize_to_allocation %at
        {memory_space = 1, emit_dealloc} : !transform.any_op

    // Promote sq generic output to L2 via promote_tensor (avoids hanging from bufferize_to_allocation)
    %sq_gens = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %sq_gen_only, %rest_gens = transform.split_handle %sq_gens {overflow_result = 1}
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %sq_result = transform.get_result %sq_gen_only[0]
        : (!transform.any_op) -> !transform.any_value
    transform.structured.promote_tensor to 1 %sq_result : !transform.any_value

    // Note: fills and reduces get memory_space 1 from the pre-pass
    // air-override-memref-memory-space{scope=func memory-space=1}.
    // No explicit promotion needed.

    // Phase 6: Bufferize
    %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func_bufferized = transform.bufferization.one_shot_bufferize %func_op : (!transform.any_op) -> !transform.any_op

    // Phase 7: Cleanup
    %func6 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func6 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func6 : !transform.any_op
    transform.apply_patterns to %func6 { transform.apply_patterns.canonicalization } : !transform.any_op
    %lc = transform.structured.match ops{["linalg.copy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %mc_conv = transform.structured.linalg_copy_to_memref %lc : (!transform.any_op) -> !transform.any_op
    %func_upd = transform.air.remove_uninitialized_copy %func6 : (!transform.any_op) -> !transform.any_op
    %func_upd2 = transform.air.eliminate_cascade_memcpy %func_upd : (!transform.any_op) -> !transform.any_op

    // Phase 8: Vectorize generics at 16-lane
    %linalg_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %inner_gen, %gen_lps:1 =
      transform.structured.tile_using_for %linalg_generics tile_sizes [16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Scalarize reduces and fills
    %all_reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %red_loops = transform.structured.convert_to_loops %all_reduces : (!transform.any_op) -> !transform.any_op
    %all_fills = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fill_loops = transform.structured.convert_to_loops %all_fills : (!transform.any_op) -> !transform.any_op

    // Phase 9: Forall → Herd
    %forall_h = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %par = transform.loop.forall_to_parallel %forall_h : (!transform.any_op) -> !transform.any_op
    %herd = transform.air.par_to_herd %par : (!transform.any_op) -> !transform.any_op

    // DMA inside herd
    %lc2 = transform.structured.match ops{["linalg.copy"]} in %herd : (!transform.any_op) -> !transform.any_op
    %mc2 = transform.structured.match ops{["memref.copy"]} in %herd : (!transform.any_op) -> !transform.any_op
    %mc3 = transform.structured.linalg_copy_to_memref %lc2 : (!transform.any_op) -> !transform.any_op
    %all_c = transform.merge_handles %mc2, %mc3 { deduplicate } : !transform.any_op
    %dmas = transform.air.copy_to_dma %all_c : (!transform.any_op) -> !transform.any_op

    // Vectorize herd
    %vh = transform.air.herd_vectorize %herd : (!transform.any_op) -> !transform.any_op

    // AIE2P bf16 casts
    %vm = transform.structured.match ops{["arith.mulf"]} in %vh : (!transform.any_op) -> !transform.any_op
    %mc4 = transform.air.vector_type_cast %vm {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
