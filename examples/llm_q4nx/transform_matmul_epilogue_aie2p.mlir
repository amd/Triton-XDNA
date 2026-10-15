// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// `transform_matmul_n128_aie2p.mlir` with an epilogue: the kernel's
// elementwise tail after its K loop runs on the herd that drains the
// accumulator, on the packed tile, and the GEMM writes its result. The K
// step is 32 and the reduction loop has no ping-pong, which leaves L1 room for
// the epilogue's output tile next to the 64x128 f32 accumulator. The
// epilogue's f32 math needs aiecc's bf16 emulation (products and tanh take
// bf16 inputs).

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {

    //==========================================================================
    // PHASE 0: FOLD A TRAILING ADD INTO THE CONTRACTION'S ACCUMULATOR
    // A `tt.dot` that carries a live accumulator arrives as a contraction into
    // a zero fill plus a separate add; every phase below assumes the folded
    // form. A no-op otherwise. See transform_library/fold_add.mlir.
    //==========================================================================

        transform.include @fold_add_into_matmul_init failures(propagate) (%arg1)
          : (!transform.any_op) -> ()

    //==========================================================================
    // PHASE 1: TILE L3->L2 MEMORY COPIES
    // Tile memref copies for streaming data from DDR (L3) to MemTile (L2).
    //==========================================================================


        %k_loops_in = transform.structured.match ops{["scf.for"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %k_loops_iv = transform.air.fold_iter_args_into_induction_var %k_loops_in : (!transform.any_op) -> !transform.any_op
        %func_k_loop = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func_k_loop {
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %k_loops = transform.structured.match ops{["scf.for"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.annotate %k_loops "kernel_k_loop" : !transform.any_op
        %func10 = transform.structured.match ops{["func.func"]} in %arg1  : (!transform.any_op) -> !transform.any_op
        %func10_updated = transform.air.convert_memref_copy_to_linalg_copy %func10 : (!transform.any_op) -> !transform.any_op
        %f0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %f0b = transform.air.fuse_elementwise_linalg %f0 : (!transform.any_op) -> !transform.any_op
        // The kernel's elementwise tail after its K loop, run on the epilogue herd.
        %epi_st = transform.structured.match ops{["bufferization.materialize_in_destination"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %epi_act = transform.get_producer_of_operand %epi_st[0] : (!transform.any_op) -> !transform.any_op
        transform.annotate %epi_act "epilogue_act" : !transform.any_op

    //==========================================================================
    // PHASE 2: PROMOTE OUTPUT TO L2
    // Allocate output buffer (C) in L2 for accumulation.
    //==========================================================================

        %result_l2 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %result_l2_buffer, %result_t2_new = transform.structured.bufferize_to_allocation %result_l2
            <{memory_space = 1, bufferize_destination_only, emit_dealloc}> : !transform.any_op

    //==========================================================================
    // PHASE 3: PACK MATMUL FOR VECTORIZED COMPUTATION
    // Pack [8, 8, 8], transpose A/B/C, promote C pack to L1.
    //==========================================================================

        %matmul_to_pack = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %packed = transform.structured.pack %matmul_to_pack packed_sizes = [8, 8, 8]
          : (!transform.any_op) -> (!transform.any_op)

        // Transpose A: outer_perm [1,0]
        %pack_producer_a = transform.get_producer_of_operand %packed[0]
          : (!transform.any_op) -> (!transform.any_op)
        %packed_a, %pack_a, %empty_unpack_a =
          transform.structured.pack_transpose %pack_producer_a with_compute_op(%packed)
          outer_perm = [1, 0] : (!transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op, !transform.any_op)

        // Transpose B: outer_perm [1,0] + inner_perm [1,0]
        %pack_producer_b = transform.get_producer_of_operand %packed_a[1]
          : (!transform.any_op) -> (!transform.any_op)
        %packed_b, %pack_b, %empty_unpack_b =
          transform.structured.pack_transpose %pack_producer_b with_compute_op(%packed_a)
          outer_perm = [1, 0] inner_perm = [1, 0] : (!transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op, !transform.any_op)

        // Transpose C: outer_perm [1,0]
        %unpack = transform.get_consumers_of_result %packed_b[0]
          : (!transform.any_op) -> (!transform.any_op)
        %packed_c, %pack_c, %unpack_c =
          transform.structured.pack_transpose %unpack with_compute_op(%packed_b)
          outer_perm = [1, 0] : (!transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op, !transform.any_op)

        // The K loop carries the packed accumulator: pack once before it and
        // unpack once after. A loop of one step has been folded away, and then
        // the pack is already outside any loop.
        transform.annotate %pack_c "acc_pack" : !transform.any_op
        %k_loop = transform.structured.match ops{["scf.for"]} attributes{kernel_k_loop} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %k_loop : !transform.any_op {
        ^bb0(%loop: !transform.any_op):
          %k_loop_packed, %hoisted_pack = transform.air.hoist_loop_carried_pack %loop : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        }
        %acc_pack = transform.structured.match ops{["linalg.pack"]} attributes{acc_pack} in %arg1 : (!transform.any_op) -> !transform.any_op

        // Promote C pack to L1
        %output_l1_pack_op_source_buffer, %output_l1_pack_op_new = transform.structured.bufferize_to_allocation %acc_pack
            <{memory_space = 2, bufferize_destination_only, memcpy_op = "linalg.copy", emit_dealloc}> : !transform.any_op

        // Annotate for robust matching after K-tiling
        transform.annotate %packed_c "packed_matmul" : !transform.any_op
        // The epilogue runs on the packed accumulator, ahead of the unpack.
        %f3 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f3 { transform.apply_patterns.linalg.data_layout_propagation } : !transform.any_op

    //==========================================================================
    // PHASE 4: TILE K REDUCTION AND FUSE PACK OPERATIONS
    // Tile packed K dim by 4 (= 32 raw K elements).
    //==========================================================================

        %tiled_reduction, %outer_for_loop =
          transform.structured.tile_using_for %packed_c tile_sizes [0, 0, 4]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.annotate %outer_for_loop "k_reduction_loop" : !transform.any_op

        %fused_lhs_l1_pack, %2 = transform.structured.fuse_into_containing_op %pack_a into %outer_for_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %fused_rhs_l1_pack, %3 = transform.structured.fuse_into_containing_op %pack_b into %outer_for_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    //==========================================================================
    // PHASE 5: TILE FOR MULTI-CORE PARALLELISM
    // Tile [8, 16, 0] for herd distribution.
    //==========================================================================

        %matmul_1 = transform.structured.match ops{["linalg.generic"]} attributes{packed_matmul} in %arg1 : (!transform.any_op) -> !transform.any_op
        %tiled_matmul_1, %inner_forall =
          transform.structured.tile_using_forall %matmul_1 tile_sizes [8, 16, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.annotate %inner_forall "compute_forall" : !transform.any_op
        transform.annotate %tiled_matmul_1 "matmul_compute" : !transform.any_op

        %fused_lhs_l1_pack2, %6 = transform.structured.fuse_into_containing_op %fused_lhs_l1_pack into %inner_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %fused_rhs_l1_pack2, %7 = transform.structured.fuse_into_containing_op %fused_rhs_l1_pack into %inner_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

        %func_2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func_2 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        transform.apply_cse to %func_2 : !transform.any_op

    //==========================================================================
    // PHASE 6: PROMOTE INPUTS TO L1 AND TILE PROLOGUE/EPILOGUE
    //==========================================================================

        // Promote A and B to L1
        %buffer_a, %new_a = transform.structured.bufferize_to_allocation %fused_lhs_l1_pack2
          <{memory_space = 2, bufferize_destination_only, emit_dealloc}> : !transform.any_op
        %buffer_b, %new_b = transform.structured.bufferize_to_allocation %fused_rhs_l1_pack2
          <{memory_space = 2, bufferize_destination_only, emit_dealloc}> : !transform.any_op

        // Prologue: fill -> generalize -> interchange -> tile for herd
        %fill_op = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %generic_fill_op = transform.structured.generalize %fill_op
            : (!transform.any_op) -> !transform.any_op
        transform.annotate %generic_fill_op "init_fill" : !transform.any_op
        %interchanged_fill_op = transform.structured.interchange %generic_fill_op
          iterator_interchange = [1, 0, 2, 3]
          : (!transform.any_op) -> !transform.any_op
        %prologue_tiled_fill, %prologue_forall =
          transform.structured.tile_using_forall %interchanged_fill_op tile_sizes [8, 16]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.annotate %prologue_forall "prologue_forall" : !transform.any_op

        // Epilogue: unpack -> tile for L2 write-back
        %unpack_op = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %unpack_l2_buf, %unpack_l2 = transform.structured.bufferize_to_allocation %unpack_op
            <{memory_space = 1, bufferize_destination_only, emit_dealloc}> : !transform.any_op
        %unpack_re = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %epilogue_tiled_unpack, %epilogue_forall =
          transform.structured.tile_using_forall %unpack_re tile_sizes [64, 128]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.annotate %epilogue_forall "epilogue_forall" : !transform.any_op
        %act_re = transform.structured.match ops{["linalg.generic"]} attributes{epilogue_act} in %arg1 : (!transform.any_op) -> !transform.any_op
        %fused_act0, %act_loop0 = transform.structured.fuse_into_containing_op %act_re into %epilogue_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %fused_act = transform.merge_handles %fused_act0 : !transform.any_op
        %act_l1_buf, %act_l1 = transform.structured.bufferize_to_allocation %fused_act
            <{memory_space = 2, bufferize_destination_only, emit_dealloc}> : !transform.any_op

        %func_3 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func_3 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        transform.apply_cse to %func_3 : !transform.any_op

    //==========================================================================
    // PHASE 7: BUFFERIZATION AND MEMORY OPTIMIZATION
    //==========================================================================

        %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %func_bufferized = transform.bufferization.one_shot_bufferize %func_op : (!transform.any_op) -> !transform.any_op

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
        %func_op_updated_1 = transform.air.eliminate_cascade_memcpy %func_op_updated : (!transform.any_op) -> !transform.any_op

        %func_hoist = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.air.hoist_static_alloc %func_hoist : (!transform.any_op) -> ()

    //==========================================================================
    // PHASE 9: TILE FOR VECTORIZATION
    //==========================================================================

        %generic1 = transform.structured.match ops{["linalg.generic"]} attributes{init_fill} in %arg1 : (!transform.any_op) -> !transform.any_op
        %generic2 = transform.structured.match ops{["linalg.generic"]} attributes{matmul_compute} in %arg1 : (!transform.any_op) -> !transform.any_op
        %inner_most_generics, %vec_loops:3 =
          transform.structured.tile_using_for %generic2 tile_sizes [2, 2, 1, 0, 0, 0]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
        transform.annotate %vec_loops#0 "air.disable_ping_pong" : !transform.any_op

        %inner_most_matmul_to_unroll, %vec_loops_to_unroll:2 =
          transform.structured.tile_using_for %inner_most_generics tile_sizes [1, 1, 0, 0, 0, 0]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
        transform.loop.unroll %vec_loops_to_unroll#1 factor = 2 : !transform.any_op
        transform.loop.unroll %vec_loops_to_unroll#0 factor = 2 : !transform.any_op

        %inner_most_fills, %vec_fill_loops:2 =
          transform.structured.tile_using_for %generic1 tile_sizes [1, 1, 0, 0]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

        %act_v = transform.structured.match ops{["linalg.generic"]} attributes{epilogue_act} in %arg1 : (!transform.any_op) -> !transform.any_op
        %act_tiled, %act_vloops:3 = transform.structured.tile_using_for %act_v tile_sizes [1, 1, 4, 0]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
        transform.annotate %act_vloops#0 "epi_loops" : !transform.any_op

    //==========================================================================
    // PHASE 10: CONVERT TO AIE HERDS AND VECTORIZE
    //==========================================================================

        %forall1 = transform.structured.match ops{["scf.forall"]} attributes{prologue_forall} in %arg1 : (!transform.any_op) -> !transform.any_op
        %forall2 = transform.structured.match ops{["scf.forall"]} attributes{compute_forall} in %arg1 : (!transform.any_op) -> !transform.any_op
        %forall3 = transform.structured.match ops{["scf.forall"]} attributes{epilogue_forall} in %arg1 : (!transform.any_op) -> !transform.any_op
        %parallel1 = transform.loop.forall_to_parallel %forall1  : (!transform.any_op) -> !transform.any_op
        %herd1 = transform.air.par_to_herd %parallel1 : (!transform.any_op) -> !transform.any_op
        transform.annotate %herd1 "prologue_herd" : !transform.any_op
        %parallel2 = transform.loop.forall_to_parallel %forall2  : (!transform.any_op) -> !transform.any_op
        %herd2 = transform.air.par_to_herd %parallel2 : (!transform.any_op) -> !transform.any_op
        transform.annotate %herd2 "compute_herd" : !transform.any_op
        %parallel3 = transform.loop.forall_to_parallel %forall3  : (!transform.any_op) -> !transform.any_op
        %herd3 = transform.air.par_to_herd %parallel3 : (!transform.any_op) -> !transform.any_op
        transform.annotate %herd3 "epilogue_herd" : !transform.any_op

        %herds = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %vectorized_herds = transform.air.herd_vectorize %herds : (!transform.any_op) -> !transform.any_op
        %ep_herd2 = transform.structured.match ops{["air.herd"]} attributes{epilogue_herd} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %ep_herd2 { transform.apply_patterns.memref.fold_memref_alias_ops } : !transform.any_op
        %epiloops = transform.structured.match ops{["scf.for"]} attributes{epi_loops} in %arg1 : (!transform.any_op) -> !transform.any_op
        %epilin = transform.air.linearize_vectors %epiloops arch = "aie2p" f32_lanes = 32 contract : (!transform.any_op) -> !transform.any_op

        %func7 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func7 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
            transform.apply_patterns.memref.fold_memref_alias_ops
        } : !transform.any_op
        %func_fold_1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %func_folded_1 = transform.air.fold_unit_extent_dims %func_fold_1 : (!transform.any_op) -> !transform.any_op

        %func7_rematch = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %func1_optimized = transform.air.eliminate_redundant_vector_transfers %func7_rematch : (!transform.any_op) -> !transform.any_op

    //==========================================================================
    // PHASE 11: HOIST LOOP-INVARIANT VECTOR TRANSFERS
    //==========================================================================

        %herd2_1 = transform.structured.match ops{["air.herd"]} attributes{compute_herd} in %arg1 : (!transform.any_op) -> !transform.any_op
        %scf_fors_1 = transform.structured.match ops{["scf.for"]} in %herd2_1 : (!transform.any_op) -> !transform.any_op
        %innermost_for, %outer_fors = transform.split_handle %scf_fors_1 overflow_result = 1 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        // Cast accumulator (input[2]) and output[0] to f32
        %vector_contracts = transform.structured.match ops{["vector.contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %result11 = transform.air.vector_type_cast %vector_contracts <{target_element_type = f32, input_indices = [2], output_indices = [0]}> : (!transform.any_op) -> !transform.any_op

        // Cast vector.contract inputs 0,1 to bf16
        // (matches hardware MAC unit native input type)
        %vector_contracts_2 = transform.structured.match ops{["vector.contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %result11b = transform.air.vector_type_cast %vector_contracts_2 <{target_element_type = bf16, input_indices = [0, 1], output_indices = []}> : (!transform.any_op) -> !transform.any_op

        %innermost_for_updated_3 = transform.air.hoist_loop_invariant_transfers %herd2_1, %innermost_for : (!transform.any_op, !transform.any_op) -> !transform.any_op

    //==========================================================================
    // PHASE 12: FINAL LOOP OPTIMIZATIONS
    //==========================================================================

        %innermost_for_updated_4 = transform.air.flatten_for_iter_args %innermost_for_updated_3 : (!transform.any_op) -> !transform.any_op
        %innermost_for_updated_5 = transform.air.hoist_vector_transfer_pointers %innermost_for_updated_4 : (!transform.any_op) -> !transform.any_op

        %func9 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func9 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
            transform.apply_patterns.memref.fold_memref_alias_ops
        } : !transform.any_op
        %func_fold_2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %func_folded_2 = transform.air.fold_unit_extent_dims %func_fold_2 : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
