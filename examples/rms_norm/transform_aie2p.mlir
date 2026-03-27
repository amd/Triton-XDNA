// RMS Norm transform for AIE2P.
// 2D kernel (BLOCK_M=2 x BLOCK_N=64).
//
// Strategy: bufferize FIRST (no L1 staging), then use linalg_promote
// on the linalg ops inside the forall to promote L2 subviews to L1 allocs.
// This creates memref.copy ops that par_to_herd + copy_to_dma convert to DMAs.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func0 { transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes } : !transform.any_op
    transform.apply_cse to %func0 : !transform.any_op
    %reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tr = transform.air.transpose_reduce %reduces : (!transform.any_op) -> !transform.any_op
    %func1a = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func1a { transform.apply_patterns.canonicalization } : !transform.any_op
    transform.apply_cse to %func1a : !transform.any_op
    %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %f = transform.air.fuse_elementwise_linalg %func1 : (!transform.any_op) -> !transform.any_op
    %fa = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %fa { transform.apply_patterns.canonicalization } : !transform.any_op
    transform.apply_cse to %fa : !transform.any_op

    %ag = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %sq, %out = transform.split_handle %ag : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %reduce = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fill = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    // L2 output alloc
    %ob, %nb = transform.structured.bufferize_to_allocation %out {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op
    // Tile at [1] on row dim
    %t, %fl = transform.structured.tile_using_forall %out tile_sizes [1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // Fuse all into forall
    %f1, %fl1 = transform.structured.fuse_into_containing_op %reduce into %fl : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %f2, %fl2 = transform.structured.fuse_into_containing_op %sq into %fl1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %f3, %fl3 = transform.structured.fuse_into_containing_op %fill into %fl2 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fuse sq into reduce
    %reduce3 = transform.structured.match ops{["linalg.reduce"]} in %fl3 : (!transform.any_op) -> !transform.any_op
    %sq3 = transform.structured.match ops{["linalg.generic"]} in %fl3 : (!transform.any_op) -> !transform.any_op
    %sq_only, %out_only = transform.split_handle %sq3 {overflow_result = 1} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_sr = transform.air.fuse_multi_op_linalg %sq_only, %reduce3 : (!transform.any_op, !transform.any_op) -> !transform.any_op

    // L1 for fills only (destination-only)
    %fills3 = transform.structured.match ops{["linalg.fill"]} in %fl3 : (!transform.any_op) -> !transform.any_op
    %fill_buf, %fill_new = transform.structured.bufferize_to_allocation %fills3
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Canonicalize + bufferize (no L1 staging for reduce/generic inputs)
    %f2c = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f2c { transform.apply_patterns.canonicalization } : !transform.any_op
    transform.apply_cse to %f2c : !transform.any_op
    transform.include @one_shot_bufferize failures(propagate) (%arg1) : (!transform.any_op) -> ()
    %f6 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f6 { transform.apply_patterns.canonicalization } : !transform.any_op
    transform.apply_cse to %f6 : !transform.any_op
    %lc = transform.structured.match ops{["linalg.copy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %mc = transform.structured.linalg_copy_to_memref %lc : (!transform.any_op) -> !transform.any_op
    %fu = transform.air.remove_uninitialized_copy %f6 : (!transform.any_op) -> (!transform.any_op)
    %fu2 = transform.air.eliminate_cascade_memcpy %fu : (!transform.any_op) -> (!transform.any_op)

    // NOW promote linalg ops inside forall to L1 (BEFORE herd creation)
    // This creates memref.copy from L2 subviews to L1 allocs
    %forall_op = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %gens_f = transform.structured.match ops{["linalg.generic"]} in %forall_op : (!transform.any_op) -> !transform.any_op
    %reds_f = transform.structured.match ops{["linalg.reduce"]} in %forall_op : (!transform.any_op) -> !transform.any_op
    %all_linalg_f = transform.merge_handles %reds_f, %gens_f { deduplicate } : !transform.any_op
    %promoted = transform.air.linalg_promote %all_linalg_f {memory_space = "L1"} : (!transform.any_op) -> !transform.any_op

    // Herd + DMA
    %fh = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %pa = transform.loop.forall_to_parallel %fh : (!transform.any_op) -> !transform.any_op
    %h = transform.air.par_to_herd %pa : (!transform.any_op) -> !transform.any_op
    %lc2 = transform.structured.match ops{["linalg.copy"]} in %h : (!transform.any_op) -> !transform.any_op
    %mc2 = transform.structured.match ops{["memref.copy"]} in %h : (!transform.any_op) -> !transform.any_op
    %mc3 = transform.structured.linalg_copy_to_memref %lc2 : (!transform.any_op) -> !transform.any_op
    %ac = transform.merge_handles %mc2, %mc3 { deduplicate } : !transform.any_op
    %dm = transform.air.copy_to_dma %ac : (!transform.any_op) -> !transform.any_op

    // Re-match the herd since handles may be stale after promote/dma
    %h2 = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // Inner vectorization tiling
    %gens_h = transform.structured.match ops{["linalg.generic"]} in %h2 : (!transform.any_op) -> !transform.any_op
    %inner_g, %inner_gl:1 = transform.structured.tile_using_for %gens_h tile_sizes [0, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %reds_h = transform.structured.match ops{["linalg.reduce"]} in %h2 : (!transform.any_op) -> !transform.any_op
    %inner_r, %inner_rl:1 = transform.structured.tile_using_for %reds_h tile_sizes [0, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fills_h = transform.structured.match ops{["linalg.fill"]} in %h2 : (!transform.any_op) -> !transform.any_op
    %fill_scl = transform.structured.convert_to_loops %fills_h : (!transform.any_op) -> !transform.any_op
    %vh = transform.air.herd_vectorize %h2 : (!transform.any_op) -> !transform.any_op

    // Lower vector reductions FIRST (creates arith.mulf/addf from vector.multi_reduction)
    %func_final = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_final {
        transform.apply_patterns.vector.reorder_multi_reduction_dims lowering_strategy = "innerreduction"
        transform.apply_patterns.vector.multi_reduction_flattening lowering_strategy = "innerreduction"
        transform.apply_patterns.vector.multi_reduction_unrolling lowering_strategy = "innerreduction"
        transform.apply_patterns.vector.lower_contraction
        transform.apply_patterns.vector.lower_transfer
    } : !transform.any_op
    transform.apply_cse to %func_final : !transform.any_op

    // AIE2P type casts AFTER lowering: mulf and addf are bf16-only, divf and rsqrt are f32-only
    %vh2 = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %vector_muls = transform.structured.match ops{["arith.mulf"]} in %vh2 : (!transform.any_op) -> !transform.any_op
    %mul_cast = transform.air.vector_type_cast %vector_muls {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op
    %vector_adds = transform.structured.match ops{["arith.addf"]} in %vh2 : (!transform.any_op) -> !transform.any_op
    %add_cast = transform.air.vector_type_cast %vector_adds {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op
    %func_s1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func_s1_done = transform.air.convert_size1_vector_to_scalar %func_s1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_s1_done {
        transform.apply_patterns.vector.cast_away_vector_leading_one_dim
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_s1_done : !transform.any_op
    transform.yield
  }
}
