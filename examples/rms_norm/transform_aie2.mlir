// RMS Norm transform for AIE2P. Requires local mlir-air build (ec7d2f0).
// 2D kernel (BLOCK_M=2 x BLOCK_N=64). After fuse_elementwise + transpose_reduce:
//   sq+extf(2D) → reduce(dim=1) → fill → fused_output(2D)
// Tile fused_output at [1] on rows, fuse all into forall.

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

    // Match ops
    %ag = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %sq_gen, %out_gen2 = transform.split_handle %ag
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %reduce = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fill = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    // Bufferize output to L2
    %ob, %nb = transform.structured.bufferize_to_allocation %out_gen2
        {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Tile output at [1] on row dim -- 2 iterations for BLOCK_M=2
    %tiled_out, %forall =
      transform.structured.tile_using_forall %out_gen2 tile_sizes [1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fuse predecessors one by one (reverse data-flow order)
    // sq_gen is the most distant producer (2D, feeds reduce)
    // reduce is the next (produces 1D result)
    // fill is the reduce init (1D)
    // Fuse in reverse data-flow order (closest to output first):
    // 1. reduce (direct producer of fused output's tensor input)
    // 2. sq (producer of reduce's input)
    // 3. fill (init value for reduce)
    %f1, %fl1 = transform.structured.fuse_into_containing_op %reduce into %forall
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %f2, %fl2 = transform.structured.fuse_into_containing_op %sq_gen into %fl1
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %f3, %fl3 = transform.structured.fuse_into_containing_op %fill into %fl2
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // L1 memory promotion (softmax pattern)
    // Fuse sq into reduce to eliminate intermediate buffer
    %fused_reduce = transform.structured.match ops{["linalg.reduce"]} in %fl3 : (!transform.any_op) -> !transform.any_op
    %fused_sq = transform.structured.match ops{["linalg.generic"]} in %fl3 : (!transform.any_op) -> !transform.any_op
    %fused_sq_only, %fused_out_only = transform.split_handle %fused_sq {overflow_result = 1}
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_sq_red = transform.air.fuse_multi_op_linalg %fused_sq_only, %fused_reduce : (!transform.any_op, !transform.any_op) -> !transform.any_op

    // Promote reduce input to L1 (creates L1 alloc + copy from L2)
    %reduce2 = transform.structured.match ops{["linalg.reduce"]} in %fl3 : (!transform.any_op) -> !transform.any_op
    %reduce_inp = transform.get_operand %reduce2[0]
        : (!transform.any_op) -> !transform.any_value
    transform.structured.promote_tensor to 2 %reduce_inp : !transform.any_value

    // Allocate fills to L1
    %fills2 = transform.structured.match ops{["linalg.fill"]} in %fl3 : (!transform.any_op) -> !transform.any_op
    %fill_buf, %fill_new = transform.structured.bufferize_to_allocation %fills2
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Allocate output generic to L1 (creates L1 alloc for output)
    %out_gens = transform.structured.match ops{["linalg.generic"]} in %fl3 : (!transform.any_op) -> !transform.any_op
    %gen_buf, %gen_new = transform.structured.bufferize_to_allocation %out_gens
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Canonicalize
    %f2c = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f2c { transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization } : !transform.any_op
    transform.apply_cse to %f2c : !transform.any_op

    // Bufferize
    %fop = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fb = transform.bufferization.one_shot_bufferize %fop : (!transform.any_op) -> !transform.any_op
    %f6 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f6 { transform.apply_patterns.canonicalization } : !transform.any_op
    transform.apply_cse to %f6 : !transform.any_op
    %lc = transform.structured.match ops{["linalg.copy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %mc = transform.structured.linalg_copy_to_memref %lc : (!transform.any_op) -> !transform.any_op
    %fu = transform.air.remove_uninitialized_copy %f6 : (!transform.any_op) -> (!transform.any_op)
    %fu2 = transform.air.eliminate_cascade_memcpy %fu : (!transform.any_op) -> (!transform.any_op)

    // Herd
    %fh = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %pa = transform.loop.forall_to_parallel %fh : (!transform.any_op) -> !transform.any_op
    %h = transform.air.par_to_herd %pa : (!transform.any_op) -> !transform.any_op
    %lc2 = transform.structured.match ops{["linalg.copy"]} in %h : (!transform.any_op) -> !transform.any_op
    %mc2 = transform.structured.match ops{["memref.copy"]} in %h : (!transform.any_op) -> !transform.any_op
    %mc3 = transform.structured.linalg_copy_to_memref %lc2 : (!transform.any_op) -> !transform.any_op
    %ac = transform.merge_handles %mc2, %mc3 { deduplicate } : !transform.any_op
    %dm = transform.air.copy_to_dma %ac : (!transform.any_op) -> !transform.any_op
    // Inner vectorization tiling at 16-lane width BEFORE vectorize
    %gens_h = transform.structured.match ops{["linalg.generic"]} in %h : (!transform.any_op) -> !transform.any_op
    %inner_g, %inner_gl:1 = transform.structured.tile_using_for %gens_h tile_sizes [16]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %reds_h = transform.structured.match ops{["linalg.reduce"]} in %h : (!transform.any_op) -> !transform.any_op
    %inner_r, %inner_rl:1 = transform.structured.tile_using_for %reds_h tile_sizes [16]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fills_h = transform.structured.match ops{["linalg.fill"]} in %h : (!transform.any_op) -> !transform.any_op
    %fill_scl = transform.structured.convert_to_loops %fills_h : (!transform.any_op) -> !transform.any_op

    %vh = transform.air.herd_vectorize %h : (!transform.any_op) -> !transform.any_op
    // Skip vector_type_cast for now -- let AIE backend handle type legalization

    // Lower vector reductions: multi_reduction → contract → lower contract
    %func_final = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_final {
        transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerreduction"
        transform.apply_patterns.vector.lower_contraction
        transform.apply_patterns.vector.lower_transfer
    } : !transform.any_op
    transform.apply_cse to %func_final : !transform.any_op

    // Convert size-1 vectors to scalars for scalar ops (divf, addf, rsqrt)
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
