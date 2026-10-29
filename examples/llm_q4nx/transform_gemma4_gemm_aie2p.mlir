// Schedule of gemma4_gemm.mm_g, the Gemma4 prefill GEMM on mlir-air fused
// prefill's weight packets: int4 or bf16, two K loops, one of which runs.
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {

    //==========================================================================
    // PHASE 0: FOLD A TRAILING ADD INTO THE CONTRACTION'S ACCUMULATOR
    // A `tt.dot` that carries a live accumulator arrives as a contraction into
    // a zero fill plus a separate add; every phase below assumes the folded
    // form. A no-op otherwise. See transform_library/fold_add.mlir.
    //==========================================================================

        // (per contraction: the library sequence takes one)
        %fold_mms = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %fold_mms : !transform.any_op {
        ^bb0(%fm: !transform.any_op):
          %fu0 = transform.get_consumers_of_result %fm[0] : (!transform.any_op) -> !transform.any_op
          %fu = transform.merge_handles deduplicate %fu0 : !transform.any_op
          transform.foreach %fu : !transform.any_op {
          ^bb1(%fx: !transform.any_op):
            transform.sequence %fx : !transform.any_op failures(suppress) {
            ^bb2(%fy: !transform.any_op):
              transform.structured.specialize %fy emit_category = true : (!transform.any_op) -> !transform.any_op
            }
          }
        }
        %fold_f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %fold_f { transform.apply_patterns.linalg.fold_add_into_dest } : !transform.any_op
        %fold_left = transform.structured.match ops{["linalg.elementwise"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %fold_left : !transform.any_op {
        ^bb0(%fe: !transform.any_op):
          transform.sequence %fe : !transform.any_op failures(suppress) {
          ^bb1(%fz: !transform.any_op):
            transform.structured.generalize %fz : (!transform.any_op) -> !transform.any_op
          }
        }

        // Masked loads as tensor values: the `other` fill with the in-mask box
        // inserted, so the slices the computation reads fold to the box.
        %func_raise = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %func_raised = transform.air.raise_load_buffer_to_tensor %func_raise : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func_raised { transform.apply_patterns.canonicalization } : !transform.any_op
        // The scale and min permutes to the dequant's dim order: generic, so
        // they fold into its indexing maps.
        %sm_perm = transform.structured.match ops{["linalg.transpose"]} filter_result_type = tensor<8x2x2x4x4x2x4xi16> in %arg1 : (!transform.any_op) -> !transform.any_op
        %sm_perm_g = transform.structured.generalize %sm_perm : (!transform.any_op) -> !transform.any_op
        // A's K pair comes as [rows, pk, 32, 128]: its permute to [M, K] as a
        // generic, which A's pack goes through.
        %a_perm = transform.structured.match ops{["linalg.transpose"]} filter_result_type = tensor<4x32x2x128xbf16> in %arg1 : (!transform.any_op) -> !transform.any_op
        %a_perm_g = transform.structured.generalize %a_perm : (!transform.any_op) -> !transform.any_op


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
        // One elementwise tail after the loop, so it splits as one op.
        %f0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %f0b = transform.air.fuse_elementwise_linalg %f0 fold_reshapes : (!transform.any_op) -> !transform.any_op
        // Columns past the mask are never stored: split them off the
        // contraction, carry the live slice alone, drop the rest.
        %mm_masked = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %mm_parts = transform.structured.split %mm_masked after 384 {dimension = 1} : !transform.any_op
        %loop_masked = transform.structured.match ops{["scf.for"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.loop.hoist_loop_invariant_subsets %loop_masked : !transform.any_op
        // The tail after the loop, split at the same column.
        %store_live = transform.structured.match ops{["bufferization.materialize_in_destination"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %live_slice = transform.get_producer_of_operand %store_live[0] : (!transform.any_op) -> !transform.any_op
        %tail = transform.get_producer_of_operand %live_slice[0] : (!transform.any_op) -> !transform.any_op
        %tail_parts = transform.structured.split %tail after 384 {dimension = 1} : !transform.any_op
        %func_live = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func_live {
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %func_live_air1 = transform.air.fold_extract_slice_of_insert_slice %func_live : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func_live_air1 {
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %func_live2 = transform.apply_registered_pass "remove-dead-values" to %func_live_air1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func_live2 { transform.apply_patterns.canonicalization } : !transform.any_op
        transform.apply_patterns to %func_live2 {
            transform.apply_patterns.linalg.swap_extract_slice_with_fill
            transform.apply_patterns.tensor.fold_tensor_empty
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %func_live2_air2 = transform.air.fold_extract_slice_of_insert_slice %func_live2 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func_live2_air2 {
            transform.apply_patterns.linalg.swap_extract_slice_with_fill
            transform.apply_patterns.tensor.fold_tensor_empty
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        // The weights' dequant feeds the contraction through a reshape;
        // split it at the same column and let the slices reach its inputs.
        // (the int4 loop's: the first in program order)
        %mm_q2 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %mm_q, %mm_b = transform.split_handle %mm_q2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        %b_slice = transform.get_producer_of_operand %mm_q[1] : (!transform.any_op) -> !transform.any_op
        %b_col = transform.get_producer_of_operand %b_slice[0] : (!transform.any_op) -> !transform.any_op
        // b_col is the transpose of the packet-order dequant to K x N: split it
        // and the dequant at the live column tiles, so the rest folds away
        %b_tr = transform.get_producer_of_operand %b_col[0] : (!transform.any_op) -> !transform.any_op
        %deq = transform.get_producer_of_operand %b_tr[0] : (!transform.any_op) -> !transform.any_op
        %b_trg = transform.structured.generalize %b_tr : (!transform.any_op) -> !transform.any_op
        %tr_parts = transform.structured.split %b_trg after 6 {dimension = 5} : !transform.any_op
        %f_trs = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f_trs {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %f_trs_air3 = transform.air.fold_extract_slice_of_insert_slice %f_trs : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f_trs_air3 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %deq_parts = transform.structured.split %deq after 6 {dimension = 0} : !transform.any_op
        %func_deq = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func_deq {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %func_deq_air4 = transform.air.fold_extract_slice_of_insert_slice %func_deq : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func_deq_air4 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %func_deq2 = transform.apply_registered_pass "remove-dead-values" to %func_deq_air4 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func_deq2 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.fold_tensor_empty
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %func_deq2_air5 = transform.air.fold_extract_slice_of_insert_slice %func_deq2 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func_deq2_air5 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.fold_tensor_empty
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        // bf16 B's live columns: the slice reaches each pair's packets.
        %b_trs_bf = transform.structured.match ops{["linalg.transpose"]} filter_result_type = tensor<2x2x2x4x2x4x8x8x8xi16> in %arg1 : (!transform.any_op) -> !transform.any_op
        %b_trg_bf = transform.structured.generalize %b_trs_bf : (!transform.any_op) -> !transform.any_op
        // the bitcast to bf16 joins it
        %f_bc_bf = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %f_bc2_bf = transform.air.fuse_elementwise_linalg %f_bc_bf fold_reshapes : (!transform.any_op) -> !transform.any_op
        %b_bc_bf = transform.structured.match ops{["linalg.generic"]} filter_result_type = tensor<2x2x2x4x2x4x8x8x8xbf16> in %arg1 : (!transform.any_op) -> !transform.any_op
        %b_trp_bf = transform.structured.split %b_bc_bf after 6 {dimension = 6} : !transform.any_op
        %f_bl_bf = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f_bl_bf {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.fold_tensor_empty
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %f_bl_bf_air6 = transform.air.bubble_slice_through_collapse %f_bl_bf : (!transform.any_op) -> !transform.any_op
        %f_bl_bf_air7 = transform.air.fold_extract_slice_of_insert_slice %f_bl_bf_air6 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f_bl_bf_air7 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.fold_tensor_empty
            transform.apply_patterns.canonicalization
        } : !transform.any_op

        // The K loop is the one the contraction sits in; an outer loop over
        // output tiles carries no accumulator.
        %mm_for_k = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %k_loops = transform.get_parent_op %mm_for_k <{deduplicate, op_name = "scf.for"}> : (!transform.any_op) -> !transform.any_op
        transform.annotate %k_loops "kernel_k_loop" : !transform.any_op
        %func10 = transform.structured.match ops{["func.func"]} in %arg1  : (!transform.any_op) -> !transform.any_op
        %func10_updated = transform.air.convert_memref_copy_to_linalg_copy %func10 : (!transform.any_op) -> !transform.any_op
        // The L2 transfers move a K step at a time, as air-FLM's a_l2[pp] and
        // b_l2[pp]: A a step per row, the weights a packet per column (int4
        // two per K pair, bf16 eight). Each its own slot, so the first step
        // starts once its own data is in, and a drained slot refills while the
        // next drains. (The copies into the loops' L2 buffers; their DDR
        // strides depend on K.)
        %l3a_a = transform.structured.match ops{["memref.alloc"]} filter_result_type = memref<4x2x32x128xbf16, 1 : i32> in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %l3a_a : !transform.any_op {
        ^bb0(%al: !transform.any_op):
          %l3u = transform.get_consumers_of_result %al[0] : (!transform.any_op) -> !transform.any_op
          transform.foreach %l3u : !transform.any_op {
          ^bb1(%u: !transform.any_op):
            %cc = transform.structured.match ops{["linalg.copy"]} in %u : (!transform.any_op) -> !transform.any_op
            transform.foreach %cc : !transform.any_op {
            ^bb2(%c: !transform.any_op):
              %l3t, %l3l = transform.structured.tile_using_for %c tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
            }
          }
        }
        %l3a_q = transform.structured.match ops{["memref.alloc"]} filter_result_type = memref<6x2x80x32xi16, 1 : i32> in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %l3a_q : !transform.any_op {
        ^bb0(%al: !transform.any_op):
          %l3u = transform.get_consumers_of_result %al[0] : (!transform.any_op) -> !transform.any_op
          transform.foreach %l3u : !transform.any_op {
          ^bb1(%u: !transform.any_op):
            %cc = transform.structured.match ops{["linalg.copy"]} in %u : (!transform.any_op) -> !transform.any_op
            transform.foreach %cc : !transform.any_op {
            ^bb2(%c: !transform.any_op):
              %l3t, %l3l = transform.structured.tile_using_for %c tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
            }
          }
        }
        %l3a_b = transform.structured.match ops{["memref.alloc"]} filter_result_type = memref<6x4x2x80x32xi16, 1 : i32> in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %l3a_b : !transform.any_op {
        ^bb0(%al: !transform.any_op):
          %l3u = transform.get_consumers_of_result %al[0] : (!transform.any_op) -> !transform.any_op
          transform.foreach %l3u : !transform.any_op {
          ^bb1(%u: !transform.any_op):
            %cc = transform.structured.match ops{["linalg.copy"]} in %u : (!transform.any_op) -> !transform.any_op
            transform.foreach %cc : !transform.any_op {
            ^bb2(%c: !transform.any_op):
              %l3t, %l3l:2 = transform.structured.tile_using_for %c tile_sizes [0, 1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
            }
          }
        }
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
    // Pack [8, 8, 8], transpose A/B/C, promote C pack to L1; the bf16 loop's
    // ops are named *_bf.
    //==========================================================================

        %mms3 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %mm3_q, %mm3_b = transform.split_handle %mms3 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        %packed = transform.structured.pack %mm3_q packed_sizes = [8, 8, 8] : (!transform.any_op) -> (!transform.any_op)
        %pa = transform.get_producer_of_operand %packed[0] : (!transform.any_op) -> (!transform.any_op)
        %packed_a, %pack_a, %ua = transform.structured.pack_transpose %pa with_compute_op(%packed)
          outer_perm = [1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
        %pb = transform.get_producer_of_operand %packed_a[1] : (!transform.any_op) -> (!transform.any_op)
        %packed_b, %pack_b, %ub = transform.structured.pack_transpose %pb with_compute_op(%packed_a)
          outer_perm = [1, 0] inner_perm = [1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
        %unpack = transform.get_consumers_of_result %packed_b[0] : (!transform.any_op) -> (!transform.any_op)
        %packed_c, %pack_c, %unpack_c = transform.structured.pack_transpose %unpack with_compute_op(%packed_b)
          outer_perm = [1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
        transform.annotate %packed_c "packed_matmul" : !transform.any_op
        transform.annotate %pack_a "a_pack" : !transform.any_op
        transform.annotate %pack_b "b_pack" : !transform.any_op
        transform.annotate %pack_c "acc_pack" : !transform.any_op
        %packed_bf = transform.structured.pack %mm3_b packed_sizes = [8, 8, 8] : (!transform.any_op) -> (!transform.any_op)
        %pa_bf = transform.get_producer_of_operand %packed_bf[0] : (!transform.any_op) -> (!transform.any_op)
        %packed_a_bf, %pack_a_bf, %ua_bf = transform.structured.pack_transpose %pa_bf with_compute_op(%packed_bf)
          outer_perm = [1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
        %pb_bf = transform.get_producer_of_operand %packed_a_bf[1] : (!transform.any_op) -> (!transform.any_op)
        %packed_b_bf, %pack_b_bf, %ub_bf = transform.structured.pack_transpose %pb_bf with_compute_op(%packed_a_bf)
          outer_perm = [1, 0] inner_perm = [1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
        %unpack_bf = transform.get_consumers_of_result %packed_b_bf[0] : (!transform.any_op) -> (!transform.any_op)
        %packed_c_bf, %pack_c_bf, %unpack_c_bf = transform.structured.pack_transpose %unpack_bf with_compute_op(%packed_b_bf)
          outer_perm = [1, 0] : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
        transform.annotate %packed_c_bf "packed_matmul_bf" : !transform.any_op
        transform.annotate %pack_a_bf "a_pack_bf" : !transform.any_op
        transform.annotate %pack_b_bf "b_pack_bf" : !transform.any_op
        transform.annotate %pack_c_bf "acc_pack" : !transform.any_op
        // The accumulator's unpack after one loop and pack before the other
        // fold; each loop carries the packed accumulator.
        %f3c = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f3c { transform.apply_patterns.canonicalization } : !transform.any_op
        transform.annotate %pack_c "acc_pack" : !transform.any_op
        %k_loop = transform.structured.match ops{["scf.for"]} attributes{kernel_k_loop} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %k_loop : !transform.any_op {
        ^bb0(%loop: !transform.any_op):
          %k_loop_packed, %hoisted_pack = transform.air.hoist_loop_carried_pack %loop : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        }
        %acc_pack = transform.structured.match ops{["linalg.pack"]} attributes{acc_pack} in %arg1 : (!transform.any_op) -> !transform.any_op
        // The epilogue runs on the packed accumulator, ahead of the unpack.
        %f3 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f3 { transform.apply_patterns.linalg.data_layout_propagation } : !transform.any_op
        // The hoist rewrites the loop, so the handles into it are matched again.
        %packed_c_k = transform.structured.match ops{["linalg.generic"]} attributes{packed_matmul} in %arg1 : (!transform.any_op) -> !transform.any_op
        %pack_a_k = transform.structured.match ops{["linalg.pack"]} attributes{a_pack} in %arg1 : (!transform.any_op) -> !transform.any_op

        // Promote C pack to L1
        %output_l1_pack_op_source_buffer, %output_l1_pack_op_new = transform.structured.bufferize_to_allocation %acc_pack
            <{memory_space = 2, bufferize_destination_only, memcpy_op = "linalg.copy", emit_dealloc}> : !transform.any_op


        // B's pack cannot pass the reshape to K x N (a tile of 8 columns
        // spans two of the dequant's dims, w and i): lower it, and fold its
        // reshape and transpose into the dequant, which then writes the
        // packed tiles itself.
        %b_pk = transform.structured.match ops{["linalg.pack"]} attributes{b_pack} in %arg1 : (!transform.any_op) -> !transform.any_op
        %b_pk_t = transform.cast %b_pk : !transform.any_op to !transform.op<"linalg.pack">
        %bp_pad, %bp_ex, %bp_tr = transform.structured.lower_pack %b_pk_t : (!transform.op<"linalg.pack">) -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
        transform.annotate %bp_tr "b_lowered" : !transform.op<"linalg.transpose">
        // bf16's B pack likewise, folded into the copy that builds B
        %b_pk_bf = transform.structured.match ops{["linalg.pack"]} attributes{b_pack_bf} in %arg1 : (!transform.any_op) -> !transform.any_op
        %b_pk_t_bf = transform.cast %b_pk_bf : !transform.any_op to !transform.op<"linalg.pack">
        %bp_pad_bf, %bp_ex_bf, %bp_tr_bf = transform.structured.lower_pack %b_pk_t_bf : (!transform.op<"linalg.pack">) -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
        transform.annotate %bp_tr_bf "b_lowered" : !transform.op<"linalg.transpose">
        %fb = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %fb1 = transform.apply_registered_pass "canonicalize" to %fb : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %fb1 {
            transform.apply_patterns.tensor.reassociative_reshape_folding
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %fb1_air8 = transform.air.swap_transpose_with_collapse %fb1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %fb1_air8 {
            transform.apply_patterns.tensor.reassociative_reshape_folding
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %bp_tr2 = transform.structured.match ops{["linalg.transpose"]} attributes{b_lowered} in %arg1 : (!transform.any_op) -> !transform.any_op
        %bp_g = transform.structured.generalize %bp_tr2 : (!transform.any_op) -> !transform.any_op
        %fb3 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %fb4 = transform.air.fuse_elementwise_linalg %fb3 fold_reshapes : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %fb4 {
            transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        // A's pack reaches its 3-D tile through the reshape.
        %fa = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %fa1 = transform.apply_registered_pass "canonicalize" to %fa : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %fa1 {
            transform.apply_patterns.linalg.data_layout_propagation
            transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %pm_q2 = transform.structured.match ops{["linalg.generic"]} attributes{packed_matmul} in %arg1 : (!transform.any_op) -> !transform.any_op
        %a_coll = transform.get_producer_of_operand %pm_q2[0] : (!transform.any_op) -> !transform.any_op
        %a_pk = transform.get_producer_of_operand %a_coll[0] : (!transform.any_op) -> !transform.any_op
        transform.annotate %a_pk "a_pack" : !transform.any_op
        %pm_dq = transform.structured.match ops{["linalg.generic"]} attributes{packed_matmul} in %arg1 : (!transform.any_op) -> !transform.any_op
        %b_dqc = transform.get_producer_of_operand %pm_dq[1] : (!transform.any_op) -> !transform.any_op
        %dq2 = transform.get_producer_of_operand %b_dqc[0] : (!transform.any_op) -> !transform.any_op
        transform.annotate %dq2 "b_dequant" : !transform.any_op
        %pm_bf3 = transform.structured.match ops{["linalg.generic"]} attributes{packed_matmul_bf} in %arg1 : (!transform.any_op) -> !transform.any_op
        %a_coll_bf3 = transform.get_producer_of_operand %pm_bf3[0] : (!transform.any_op) -> !transform.any_op
        %a_pk_bf3 = transform.get_producer_of_operand %a_coll_bf3[0] : (!transform.any_op) -> !transform.any_op
        transform.annotate %a_pk_bf3 "a_pack_bf" : !transform.any_op
        %b_dqc_bf3 = transform.get_producer_of_operand %pm_bf3[1] : (!transform.any_op) -> !transform.any_op
        %dq2_bf3 = transform.get_producer_of_operand %b_dqc_bf3[0] : (!transform.any_op) -> !transform.any_op
        transform.annotate %dq2_bf3 "b_dequant_bf" : !transform.any_op
    //==========================================================================
    // PHASE 4: TILE K REDUCTION AND FUSE PACK OPERATIONS
    // Tile packed K dim by 16 (= 128 raw K elements).
    //==========================================================================

        // A trip brings a K pair (32 blocks); the cores take it a packet
        // (16 blocks) at a time. The packet loop is unrolled later, so each
        // packet gets its own L1 buffers.
        %f4p = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f4p {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %f4p_air9 = transform.air.bubble_slice_through_collapse %f4p : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f4p_air9 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %pm_h = transform.structured.match ops{["linalg.generic"]} attributes{packed_matmul} in %arg1 : (!transform.any_op) -> !transform.any_op
        %pm_ht, %h_loop = transform.structured.tile_using_for %pm_h tile_sizes [0, 0, 16]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.annotate %h_loop "pkt_loop" : !transform.any_op
        %f4h = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f4h {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %f4h_air10 = transform.air.bubble_slice_through_collapse %f4h : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f4h_air10 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %pa_h = transform.structured.match ops{["linalg.pack"]} attributes{a_pack} in %arg1 : (!transform.any_op) -> !transform.any_op
        %pa_hf, %pa_hl = transform.structured.fuse_into_containing_op %pa_h into %h_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %dq_h = transform.structured.match ops{["linalg.generic"]} attributes{b_dequant} in %arg1 : (!transform.any_op) -> !transform.any_op
        %dq_hf, %dq_hl = transform.structured.fuse_into_containing_op %dq_h into %h_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %f4d = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f4d {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %f4d_air11 = transform.air.bubble_slice_through_collapse %f4d : (!transform.any_op) -> !transform.any_op
        %f4d_air12 = transform.air.fold_extract_slice_of_insert_slice %f4d_air11 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f4d_air12 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        // (the int4 loop's: the first K loop)
        %kls_q = transform.structured.match ops{["scf.for"]} attributes{kernel_k_loop} in %arg1 : (!transform.any_op) -> !transform.any_op
        %kl_q, %kl_b = transform.split_handle %kls_q : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        %hsplits = transform.structured.match ops{["linalg.transpose"]} in %kl_q : (!transform.any_op) -> !transform.any_op
        %hs0, %hs1, %hs2, %hs3 = transform.split_handle %hsplits : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
        %hs3f, %hs3l = transform.structured.fuse_into_containing_op %hs3 into %h_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %f4s3 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f4s3 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %f4s3_air13 = transform.air.bubble_slice_through_collapse %f4s3 : (!transform.any_op) -> !transform.any_op
        %f4s3_air14 = transform.air.fold_extract_slice_of_insert_slice %f4s3_air13 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f4s3_air14 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %hs2f, %hs2l = transform.structured.fuse_into_containing_op %hs2 into %h_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %f4s2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f4s2 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %f4s2_air15 = transform.air.bubble_slice_through_collapse %f4s2 : (!transform.any_op) -> !transform.any_op
        %f4s2_air16 = transform.air.fold_extract_slice_of_insert_slice %f4s2_air15 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f4s2_air16 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %hs1f, %hs1l = transform.structured.fuse_into_containing_op %hs1 into %h_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %f4s1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f4s1 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %f4s1_air17 = transform.air.bubble_slice_through_collapse %f4s1 : (!transform.any_op) -> !transform.any_op
        %f4s1_air18 = transform.air.fold_extract_slice_of_insert_slice %f4s1_air17 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f4s1_air18 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %hs0f, %hs0l = transform.structured.fuse_into_containing_op %hs0 into %h_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        // Each consumer got its own copy of a shared producer: one is enough.
        %f4c = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f4c { transform.apply_patterns.canonicalization } : !transform.any_op
        transform.apply_cse to %f4c : !transform.any_op
        %f4 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f4 {
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %f4_air19 = transform.air.bubble_slice_through_collapse %f4 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f4_air19 {
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op

        %matmul_1 = transform.structured.match ops{["linalg.generic"]} attributes{packed_matmul} in %arg1 : (!transform.any_op) -> !transform.any_op
        %tiled_matmul_1, %inner_forall =
          transform.structured.tile_using_forall %matmul_1 tile_sizes [4, 8, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.annotate %inner_forall "compute_forall" : !transform.any_op
        transform.annotate %tiled_matmul_1 "matmul_compute" : !transform.any_op
        %f5a = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f5a {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %pack_a_5 = transform.structured.match ops{["linalg.pack"]} attributes{a_pack} in %arg1 : (!transform.any_op) -> !transform.any_op
        %fused_lhs_l1_pack2, %6 = transform.structured.fuse_into_containing_op %pack_a_5 into %inner_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        // B is the dequant of the weights' tiles: its slice passes the
        // reshape, then the dequant and its input packs join the cores.
        %f5 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f5 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %dq5 = transform.structured.match ops{["linalg.generic"]} attributes{b_dequant} in %arg1 : (!transform.any_op) -> !transform.any_op
        %fused_dq, %7 = transform.structured.fuse_into_containing_op %dq5 into %inner_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %f5c = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f5c {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        // q, scale and min are parts of one packet load, split by transposes:
        // bring those into the cores too, last split first, so the dequant
        // reads the packet's tile.
        // The core tiles' offsets come as delinearized multiples of the tile
        // size; spelled out as affine.apply they fold to the zeros they are.
        %f_ix = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %f_ix2 = transform.apply_registered_pass "affine-expand-index-ops-as-affine" to %f_ix : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f_ix2 { transform.apply_patterns.canonicalization } : !transform.any_op
        %splits = transform.structured.match ops{["linalg.transpose"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %f_sp = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f_sp {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %f_sp_air20 = transform.air.bubble_slice_through_collapse %f_sp : (!transform.any_op) -> !transform.any_op
        %f_sp_air21 = transform.air.fold_extract_slice_of_insert_slice %f_sp_air20 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f_sp_air21 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        // q, scale and min: the packet's parts, split by transposes, some of
        // them consumers of others. Fuse each into the cores once its
        // consumers are there: a few rounds, one that is not ready yet is
        // left for the next.
        %spl0 = transform.structured.match ops{["linalg.transpose"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %spl0 : !transform.any_op {
        ^bb0(%t: !transform.any_op):
          transform.sequence %t : !transform.any_op failures(suppress) {
          ^bb1(%t1: !transform.any_op):
            %tf, %tc = transform.structured.fuse_into_containing_op %t1 into %inner_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
          }
        }
        %fspr0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %fspr0 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %fspr0_air22 = transform.air.bubble_slice_through_collapse %fspr0 : (!transform.any_op) -> !transform.any_op
        %fspr0_air23 = transform.air.fold_extract_slice_of_insert_slice %fspr0_air22 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %fspr0_air23 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %spl1 = transform.structured.match ops{["linalg.transpose"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %spl1 : !transform.any_op {
        ^bb0(%t: !transform.any_op):
          transform.sequence %t : !transform.any_op failures(suppress) {
          ^bb1(%t1: !transform.any_op):
            %tf, %tc = transform.structured.fuse_into_containing_op %t1 into %inner_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
          }
        }
        %fspr1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %fspr1 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %fspr1_air24 = transform.air.bubble_slice_through_collapse %fspr1 : (!transform.any_op) -> !transform.any_op
        %fspr1_air25 = transform.air.fold_extract_slice_of_insert_slice %fspr1_air24 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %fspr1_air25 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %spl2 = transform.structured.match ops{["linalg.transpose"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %spl2 : !transform.any_op {
        ^bb0(%t: !transform.any_op):
          transform.sequence %t : !transform.any_op failures(suppress) {
          ^bb1(%t1: !transform.any_op):
            %tf, %tc = transform.structured.fuse_into_containing_op %t1 into %inner_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
          }
        }
        %fspr2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %fspr2 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %fspr2_air26 = transform.air.bubble_slice_through_collapse %fspr2 : (!transform.any_op) -> !transform.any_op
        %fspr2_air27 = transform.air.fold_extract_slice_of_insert_slice %fspr2_air26 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %fspr2_air27 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %spl3 = transform.structured.match ops{["linalg.transpose"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %spl3 : !transform.any_op {
        ^bb0(%t: !transform.any_op):
          transform.sequence %t : !transform.any_op failures(suppress) {
          ^bb1(%t1: !transform.any_op):
            %tf, %tc = transform.structured.fuse_into_containing_op %t1 into %inner_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
          }
        }
        %fspr3 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %fspr3 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %fspr3_air28 = transform.air.bubble_slice_through_collapse %fspr3 : (!transform.any_op) -> !transform.any_op
        %fspr3_air29 = transform.air.fold_extract_slice_of_insert_slice %fspr3_air28 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %fspr3_air29 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %f_spz = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f_spz {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %f_spz_air30 = transform.air.fold_extract_slice_of_insert_slice %f_spz : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f_spz_air30 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        // On a core's tile the splits are unit-dim transposes, i.e. copies:
        // fold them, and read the packet's box directly.
        %tr_core = transform.structured.match ops{["linalg.transpose"]} in %inner_forall : (!transform.any_op) -> !transform.any_op
        %tr_core_g = transform.structured.generalize %tr_core : (!transform.any_op) -> !transform.any_op
        %f_spu = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f_spu {
            transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %f_spu_air31 = transform.air.fold_extract_slice_of_insert_slice %f_spu : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f_spu_air31 {
            transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        transform.apply_patterns to %f_spu_air31 {
            transform.apply_patterns.tensor.reassociative_reshape_folding
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op

        %func_2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func_2 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        transform.apply_cse to %func_2 : !transform.any_op
        // That rebuilt the dequant without its unit dims: name it again.
        %dq_r = transform.structured.match ops{["linalg.generic"]} filter_result_type = tensor<2x4x4x2x2x8x2x4xbf16> in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.annotate %dq_r "b_dequant" : !transform.any_op
        // The dequant's q, scale and min: one copy of the core's packet in
        // L1, as air-FLM moves it, and the parts read from it.
        %pkt_cp = transform.air.coalesce_slices %dq_r : (!transform.any_op) -> !transform.any_op
        %pkt_buf, %pkt_new = transform.structured.bufferize_to_allocation %pkt_cp
          <{memory_space = 2, bufferize_destination_only, emit_dealloc}> : !transform.any_op
    //==========================================================================
    // PHASE 4: TILE K BY THE A TILE (16 packed = 128 K), ONE CORE PASS PER TILE
    //==========================================================================

        // A trip brings a K pair; the cores take it a K step (128 K: one A
        // tile, four bf16 packets) at a time. The step loop is unrolled
        // here, so each step reads its two packet pairs by constant offsets.
        %f4p_bf = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f4p_bf {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %f4p_bf_air32 = transform.air.bubble_slice_through_collapse %f4p_bf : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f4p_bf_air32 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %pm_h_bf = transform.structured.match ops{["linalg.generic"]} attributes{packed_matmul_bf} in %arg1 : (!transform.any_op) -> !transform.any_op
        %pm_ht_bf, %h_loop_bf = transform.structured.tile_using_for %pm_h_bf tile_sizes [0, 0, 16]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        %f4h_bf = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f4h_bf {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %f4h_bf_air33 = transform.air.bubble_slice_through_collapse %f4h_bf : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f4h_bf_air33 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %pa_h_bf = transform.structured.match ops{["linalg.pack"]} attributes{a_pack_bf} in %arg1 : (!transform.any_op) -> !transform.any_op
        %pa_hf_bf, %pa_hl_bf = transform.structured.fuse_into_containing_op %pa_h_bf into %h_loop_bf : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %dq_h_bf = transform.structured.match ops{["linalg.generic"]} attributes{b_dequant_bf} in %arg1 : (!transform.any_op) -> !transform.any_op
        %dq_hf_bf, %dq_hl_bf = transform.structured.fuse_into_containing_op %dq_h_bf into %h_loop_bf : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.loop.unroll %h_loop_bf factor = 2 : !transform.any_op
        %f4d_bf = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f4d_bf {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %f4d_bf_air34 = transform.air.bubble_slice_through_collapse %f4d_bf : (!transform.any_op) -> !transform.any_op
        %f4d_bf_air35 = transform.air.fold_extract_slice_of_insert_slice %f4d_bf_air34 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f4d_bf_air35 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        // Each step on the cores: one forall per step.
        %pm_s_bf = transform.structured.match ops{["linalg.generic"]} attributes{packed_matmul_bf} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %pm_s_bf : !transform.any_op {
        ^bb0(%pm_bf: !transform.any_op):
          %tm_bf, %cf_bf = transform.structured.tile_using_forall %pm_bf tile_sizes [4, 8, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
          transform.annotate %cf_bf "compute_forall_bf" : !transform.any_op
          transform.annotate %tm_bf "matmul_compute" : !transform.any_op
        }
        %f5a_bf = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f5a_bf {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %cfs5_bf = transform.structured.match ops{["scf.forall"]} attributes{compute_forall_bf} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %cfs5_bf : !transform.any_op {
        ^bb0(%cf_bf: !transform.any_op):
          %mm5_bf = transform.structured.match ops{["linalg.generic"]} attributes{matmul_compute} in %cf_bf : (!transform.any_op) -> !transform.any_op
          %a5c_bf = transform.get_producer_of_operand %mm5_bf[0] : (!transform.any_op) -> !transform.any_op
          %a5s_bf = transform.get_producer_of_operand %a5c_bf[0] : (!transform.any_op) -> !transform.any_op
          %a5_bf = transform.get_producer_of_operand %a5s_bf[0] : (!transform.any_op) -> !transform.any_op
          %a5f_bf, %a5l_bf = transform.structured.fuse_into_containing_op %a5_bf into %cf_bf : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
          %b5c_bf = transform.get_producer_of_operand %mm5_bf[1] : (!transform.any_op) -> !transform.any_op
          %b5s_bf = transform.get_producer_of_operand %b5c_bf[0] : (!transform.any_op) -> !transform.any_op
          %b5_bf = transform.get_producer_of_operand %b5s_bf[0] : (!transform.any_op) -> !transform.any_op
          %b5f_bf, %b5l_bf = transform.structured.fuse_into_containing_op %b5_bf into %cf_bf : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        }
        %f5c_bf = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f5c_bf {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %f5c_bf_air36 = transform.air.fold_extract_slice_of_insert_slice %f5c_bf : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f5c_bf_air36 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        // A step's B is four packets: one loop trip per packet (pair jj0,
        // packet pk), so each trip reads one packet of one pair load.
        %cfs6_bf = transform.structured.match ops{["scf.forall"]} attributes{compute_forall_bf} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %cfs6_bf : !transform.any_op {
        ^bb0(%cf_bf: !transform.any_op):
          %mm6_bf = transform.structured.match ops{["linalg.generic"]} attributes{matmul_compute} in %cf_bf : (!transform.any_op) -> !transform.any_op
          %b6c_bf = transform.get_producer_of_operand %mm6_bf[1] : (!transform.any_op) -> !transform.any_op
          %b6_bf = transform.get_producer_of_operand %b6c_bf[0] : (!transform.any_op) -> !transform.any_op
          %b6t_bf, %pl_j_bf, %pl_k_bf = transform.structured.tile_using_for %b6_bf tile_sizes [0, 0, 0, 1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
          transform.annotate %pl_j_bf "pkt_loop_j" : !transform.any_op
          transform.annotate %pl_k_bf "pkt_loop_k" : !transform.any_op
          transform.annotate %b6t_bf "b_copy" : !transform.any_op
        }
        // Pair jj0 is a part of the join: unroll its loop, so each trip
        // reads its pair by a constant offset.
        %plj_bf = transform.structured.match ops{["scf.for"]} attributes{pkt_loop_j} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %plj_bf : !transform.any_op {
        ^bb0(%l_bf: !transform.any_op):
          transform.loop.unroll %l_bf factor = 2 : !transform.any_op
        }
        %plk_bf = transform.structured.match ops{["scf.for"]} attributes{pkt_loop_k} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %plk_bf : !transform.any_op {
        ^bb0(%l_bf: !transform.any_op):
          transform.loop.unroll %l_bf factor = 2 : !transform.any_op
        }
        %f6a_bf = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f6a_bf {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %f6a_bf_air37 = transform.air.fold_extract_slice_of_insert_slice %f6a_bf : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f6a_bf_air37 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        // The core tiles' offsets come as delinearized multiples of the tile
        // size; spelled out as affine.apply they fold to the zeros they are.
        %f_ix_bf = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %f_ix2_bf = transform.apply_registered_pass "affine-expand-index-ops-as-affine" to %f_ix_bf : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f_ix2_bf { transform.apply_patterns.canonicalization } : !transform.any_op
        transform.apply_patterns to %f_ix2_bf {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %f_ix2_bf_air38 = transform.air.bubble_slice_through_collapse %f_ix2_bf : (!transform.any_op) -> !transform.any_op
        %f_ix2_bf_air39 = transform.air.fold_extract_slice_of_insert_slice %f_ix2_bf_air38 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f_ix2_bf_air39 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        // round 0: bring each packet's producers (bitcast, the split's
        // transpose) into the cores; one not ready yet waits for the next round.
        %cfr0_bf = transform.structured.match ops{["scf.forall"]} attributes{compute_forall_bf} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %cfr0_bf : !transform.any_op {
        ^bb0(%cf_bf: !transform.any_op):
          %prods_b_bf = transform.structured.match ops{["linalg.generic"]} filter_result_type = tensor<8x2x64x32xbf16> in %arg1 : (!transform.any_op) -> !transform.any_op
          %prods_t_bf = transform.structured.match ops{["linalg.transpose"]} filter_result_type = tensor<8x4x2x64x32x2xi16> in %arg1 : (!transform.any_op) -> !transform.any_op
          %prods_bf = transform.merge_handles %prods_b_bf, %prods_t_bf : !transform.any_op
          transform.foreach %prods_bf : !transform.any_op {
          ^bb1(%t_bf: !transform.any_op):
            transform.sequence %t_bf : !transform.any_op failures(suppress) {
            ^bb2(%t1_bf: !transform.any_op):
              %tf_bf, %tc_bf = transform.structured.fuse_into_containing_op %t1_bf into %cf_bf : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
            }
          }
        }
        %fr0_bf = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %fr0_bf {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %fr0_bf_air40 = transform.air.bubble_slice_through_collapse %fr0_bf : (!transform.any_op) -> !transform.any_op
        %fr0_bf_air41 = transform.air.fold_extract_slice_of_insert_slice %fr0_bf_air40 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %fr0_bf_air41 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        // round 1: bring each packet's producers (bitcast, the split's
        // transpose) into the cores; one not ready yet waits for the next round.
        %cfr1_bf = transform.structured.match ops{["scf.forall"]} attributes{compute_forall_bf} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %cfr1_bf : !transform.any_op {
        ^bb0(%cf_bf: !transform.any_op):
          %prods_b_bf = transform.structured.match ops{["linalg.generic"]} filter_result_type = tensor<8x2x64x32xbf16> in %arg1 : (!transform.any_op) -> !transform.any_op
          %prods_t_bf = transform.structured.match ops{["linalg.transpose"]} filter_result_type = tensor<8x4x2x64x32x2xi16> in %arg1 : (!transform.any_op) -> !transform.any_op
          %prods_bf = transform.merge_handles %prods_b_bf, %prods_t_bf : !transform.any_op
          transform.foreach %prods_bf : !transform.any_op {
          ^bb1(%t_bf: !transform.any_op):
            transform.sequence %t_bf : !transform.any_op failures(suppress) {
            ^bb2(%t1_bf: !transform.any_op):
              %tf_bf, %tc_bf = transform.structured.fuse_into_containing_op %t1_bf into %cf_bf : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
            }
          }
        }
        %fr1_bf = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %fr1_bf {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %fr1_bf_air42 = transform.air.bubble_slice_through_collapse %fr1_bf : (!transform.any_op) -> !transform.any_op
        %fr1_bf_air43 = transform.air.fold_extract_slice_of_insert_slice %fr1_bf_air42 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %fr1_bf_air43 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        // round 2: bring each packet's producers (bitcast, the split's
        // transpose) into the cores; one not ready yet waits for the next round.
        %cfr2_bf = transform.structured.match ops{["scf.forall"]} attributes{compute_forall_bf} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %cfr2_bf : !transform.any_op {
        ^bb0(%cf_bf: !transform.any_op):
          %prods_b_bf = transform.structured.match ops{["linalg.generic"]} filter_result_type = tensor<8x2x64x32xbf16> in %arg1 : (!transform.any_op) -> !transform.any_op
          %prods_t_bf = transform.structured.match ops{["linalg.transpose"]} filter_result_type = tensor<8x4x2x64x32x2xi16> in %arg1 : (!transform.any_op) -> !transform.any_op
          %prods_bf = transform.merge_handles %prods_b_bf, %prods_t_bf : !transform.any_op
          transform.foreach %prods_bf : !transform.any_op {
          ^bb1(%t_bf: !transform.any_op):
            transform.sequence %t_bf : !transform.any_op failures(suppress) {
            ^bb2(%t1_bf: !transform.any_op):
              %tf_bf, %tc_bf = transform.structured.fuse_into_containing_op %t1_bf into %cf_bf : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
            }
          }
        }
        %fr2_bf = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %fr2_bf {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %fr2_bf_air44 = transform.air.bubble_slice_through_collapse %fr2_bf : (!transform.any_op) -> !transform.any_op
        %fr2_bf_air45 = transform.air.fold_extract_slice_of_insert_slice %fr2_bf_air44 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %fr2_bf_air45 {
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        // Each packet: one copy of all its 80 rows into L1 (the DMA int4's
        // packets take, as air-FLM moves both), its first 64 rows read.
        // Then the splits, unit-dim transposes on a core's tile, fold.
        %cft_bf = transform.structured.match ops{["scf.forall"]} attributes{compute_forall_bf} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %cft_bf : !transform.any_op {
        ^bb0(%cf_bf: !transform.any_op):
          %tr_core_bf = transform.structured.match ops{["linalg.transpose"]} in %cf_bf : (!transform.any_op) -> !transform.any_op
          transform.foreach %tr_core_bf : !transform.any_op {
          ^bb1(%tr1_bf: !transform.any_op):
            %pkt_cp_bf = transform.air.coalesce_slices %tr1_bf full_dims = [3] : (!transform.any_op) -> !transform.any_op
            transform.annotate %pkt_cp_bf "pkt_copy" : !transform.any_op
          }
          %tr_core_g_bf = transform.structured.generalize %tr_core_bf : (!transform.any_op) -> !transform.any_op
        }
        %f_spu_bf = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f_spu_bf {
            transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %f_spu_bf_air46 = transform.air.fold_extract_slice_of_insert_slice %f_spu_bf : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f_spu_bf_air46 {
            transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        transform.apply_patterns to %f_spu_bf_air46 {
            transform.apply_patterns.tensor.reassociative_reshape_folding
            transform.apply_patterns.tensor.bubble_up_extract_slice
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %f_fe_bf = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %f_fe2_bf = transform.air.fuse_elementwise_linalg %f_fe_bf fold_reshapes : (!transform.any_op) -> !transform.any_op
        %func_2_bf = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func_2_bf {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        transform.apply_cse to %func_2_bf : !transform.any_op
        %f_rr_bf = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f_rr_bf {
            transform.apply_patterns.tensor.reassociative_reshape_folding
            transform.apply_patterns.canonicalization
        } : !transform.any_op
    //==========================================================================
    // PHASE 6: PROMOTE INPUTS TO L1 AND TILE PROLOGUE/EPILOGUE
    //==========================================================================

        // Promote A and B to L1
        %buffer_a, %new_a = transform.structured.bufferize_to_allocation %fused_lhs_l1_pack2
          <{memory_space = 2, bufferize_destination_only, emit_dealloc}> : !transform.any_op
        // B: dequantized on the core into L1
        %buffer_dq, %new_dq = transform.structured.bufferize_to_allocation %dq_r
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
          transform.structured.tile_using_forall %interchanged_fill_op tile_sizes [4, 8]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.annotate %prologue_forall "prologue_forall" : !transform.any_op

        // Epilogue: unpack -> tile for L2 write-back
        %unpack_op = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %unpack_l2_buf, %unpack_l2 = transform.structured.bufferize_to_allocation %unpack_op
            <{memory_space = 1, bufferize_destination_only, emit_dealloc}> : !transform.any_op
        %unpack_re = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %epilogue_tiled_unpack, %epilogue_forall =
          transform.structured.tile_using_forall %unpack_re tile_sizes [32, 64]
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
        %f_em_bf = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f_em_bf {
            transform.apply_patterns.tensor.fold_tensor_empty
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %cfp_bf = transform.structured.match ops{["scf.forall"]} attributes{compute_forall_bf} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %cfp_bf : !transform.any_op {
        ^bb0(%cf_bf: !transform.any_op):
          // A's tile in L1
          %pa6_bf = transform.structured.match ops{["linalg.pack"]} attributes{a_pack_bf} in %cf_bf : (!transform.any_op) -> !transform.any_op
          %ba6_bf, %na6_bf = transform.structured.bufferize_to_allocation %pa6_bf
            <{memory_space = 2, bufferize_destination_only, emit_dealloc}> : !transform.any_op
          // each packet a DMA into L1, as air-FLM moves it
          %pc6_bf = transform.structured.match ops{["linalg.copy"]} attributes{pkt_copy} in %cf_bf : (!transform.any_op) -> !transform.any_op
          %bi6_bf, %ni6_bf = transform.structured.bufferize_to_allocation %pc6_bf
            <{memory_space = 2, bufferize_destination_only, emit_dealloc}> : !transform.any_op
          // the step's B tile in L1, written a packet at a time
          %bd6_bf = transform.structured.match ops{["tensor.empty"]} filter_result_type = tensor<1x8x1x2x2x4x2x4x8xbf16> in %cf_bf : (!transform.any_op) -> !transform.any_op
          %bd6t_bf = transform.cast %bd6_bf : !transform.any_op to !transform.op<"tensor.empty">
          %bd6a_bf = transform.bufferization.empty_tensor_to_alloc_tensor %bd6t_bf : (!transform.op<"tensor.empty">) -> !transform.op<"bufferization.alloc_tensor">
          %bb6_bf, %nb6_bf = transform.structured.bufferize_to_allocation %bd6a_bf
            <{memory_space = 2, emit_dealloc}> : !transform.op<"bufferization.alloc_tensor">
        }
    //==========================================================================
    // PHASE 7: BUFFERIZATION AND MEMORY OPTIMIZATION
    //==========================================================================

        %cps0 = transform.structured.match ops{["linalg.generic"]} filter_result_type = tensor<8x4x2x4x8xbf16> in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.annotate %cps0 "b_cp" : !transform.any_op
        // Each bf16 packet's copy writes its part of the B tile in place.
        %f_ee = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.bufferization.eliminate_empty_tensors %f_ee : !transform.any_op
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

        // The packet loop's two trips, each with its own buffers (ping-pong,
        // as air-FLM writes it): the DMA of one overlaps the other's compute.
        %pl = transform.structured.match ops{["scf.for"]} attributes{pkt_loop} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.loop.unroll %pl factor = 2 : !transform.any_op
        %fpl = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %fpl { transform.apply_patterns.canonicalization } : !transform.any_op
        %func_hoist = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.air.hoist_static_alloc %func_hoist : (!transform.any_op) -> ()
        // One set of L1 buffers for both loops, as air-FLM's GEMM herd: the
        // cores take the packets of both formats alternately into two
        // buffers (b_l1[pp]), one B tile, one A tile per K step. (Each loop
        // keeps its L2 buffers: aircc splits each per column and then merges
        // the two loops' channels, air-fuse-channels{sequential-loops}.)
        // hoist_static_alloc lists them last use first: the bf16 loop's second
        // step's packets, its first step's, then the int4 loop's second and
        // first packets.
        %pk_b = transform.structured.match ops{["memref.alloc"]} filter_result_type = memref<1x1x1x80x32xi16, 2> in %arg1 : (!transform.any_op) -> !transform.any_op
        %pb0, %pb1, %pb2, %pb3, %pb4, %pb5, %pb6, %pb7 = transform.split_handle %pk_b : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
        %pk_q = transform.structured.match ops{["memref.alloc"]} filter_result_type = memref<1x1x80x32xi16, 2> in %arg1 : (!transform.any_op) -> !transform.any_op
        %pq1, %pq0 = transform.split_handle %pk_q : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        %pkt_a = transform.merge_handles %pq0, %pb7, %pb5, %pb3, %pb1 : !transform.any_op
        %pkt_b = transform.merge_handles %pq1, %pb6, %pb4, %pb2, %pb0 : !transform.any_op
        %pkt_a1 = transform.air.share_alloc %pkt_a : (!transform.any_op) -> !transform.any_op
        %pkt_b1 = transform.air.share_alloc %pkt_b : (!transform.any_op) -> !transform.any_op
        %bdq_q = transform.structured.match ops{["memref.alloc"]} filter_result_type = memref<2x4x4x2x2x8x2x4xbf16, 2> in %arg1 : (!transform.any_op) -> !transform.any_op
        %bdq_b = transform.structured.match ops{["memref.alloc"]} filter_result_type = memref<1x8x1x2x2x4x2x4x8xbf16, 2> in %arg1 : (!transform.any_op) -> !transform.any_op
        %bdqs = transform.merge_handles %bdq_q, %bdq_b : !transform.any_op
        %bdq_one = transform.air.share_alloc %bdqs : (!transform.any_op) -> !transform.any_op
        %a1s = transform.structured.match ops{["memref.alloc"]} filter_result_type = memref<1x16x1x4x8x8xbf16, 2> in %arg1 : (!transform.any_op) -> !transform.any_op
        %ab1, %ab0, %aq1, %aq0 = transform.split_handle %a1s : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
        %a1_0 = transform.merge_handles %aq0, %ab0 : !transform.any_op
        %a1_1 = transform.merge_handles %aq1, %ab1 : !transform.any_op
        %a1_0s = transform.air.share_alloc %a1_0 : (!transform.any_op) -> !transform.any_op
        %a1_1s = transform.air.share_alloc %a1_1 : (!transform.any_op) -> !transform.any_op
        // The cores' packet and A copies are DMAs from L2.
        %cf7 = transform.structured.match ops{["scf.forall"]} attributes{compute_forall} in %arg1 : (!transform.any_op) -> !transform.any_op
        %cf7b = transform.structured.match ops{["scf.forall"]} attributes{compute_forall_bf} in %arg1 : (!transform.any_op) -> !transform.any_op
        %cf7s = transform.merge_handles %cf7, %cf7b : !transform.any_op
        transform.foreach %cf7s : !transform.any_op {
        ^bb0(%cf: !transform.any_op):
          %pkt_lcopies = transform.structured.match ops{["linalg.copy"]} in %cf : (!transform.any_op) -> !transform.any_op
          %pkt_mcopies = transform.structured.linalg_copy_to_memref %pkt_lcopies : (!transform.any_op) -> !transform.any_op
          %pkt_copies = transform.structured.match ops{["memref.copy"]} in %cf : (!transform.any_op) -> !transform.any_op
          %pkt_dmas = transform.air.copy_to_dma %pkt_copies : (!transform.any_op) -> !transform.any_op
          transform.annotate %cf "compute_forall" : !transform.any_op
        }

    //==========================================================================
    // PHASE 9: TILE FOR VECTORIZATION
    //==========================================================================

        %generic1 = transform.structured.match ops{["linalg.generic"]} attributes{init_fill} in %arg1 : (!transform.any_op) -> !transform.any_op
        %generic2 = transform.structured.match ops{["linalg.generic"]} attributes{matmul_compute} in %arg1 : (!transform.any_op) -> !transform.any_op
        %inner_most_generics, %vec_loops:3 =
          transform.structured.tile_using_for %generic2 tile_sizes [2, 2, 1, 0, 0, 0]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

        %inner_most_matmul_to_unroll, %vec_loops_to_unroll:2 =
          transform.structured.tile_using_for %inner_most_generics tile_sizes [1, 1, 0, 0, 0, 0]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
        transform.loop.unroll %vec_loops_to_unroll#1 factor = 2 : !transform.any_op
        transform.annotate %vec_loops#2 "mm_inner_loop" : !transform.any_op
        transform.annotate %vec_loops#0 "mm_outer_loop" : !transform.any_op
        transform.loop.unroll %vec_loops_to_unroll#0 factor = 2 : !transform.any_op

        %inner_most_fills, %vec_fill_loops:2 =
          transform.structured.tile_using_for %generic1 tile_sizes [1, 1, 0, 0]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

        %dq_v = transform.structured.match ops{["linalg.generic"]} attributes{b_dequant} in %arg1 : (!transform.any_op) -> !transform.any_op
        // (nh, nbl, g, kbm, kbl): one B tile, 64 lanes (k, n), per vector.
        // The group's four tiles unrolled and the (nh, nbl, g) loops made one,
        // so the dequant is one loop of 32 trips the core can pipeline.
        %dq_tiled, %dq_vloops:5 = transform.structured.tile_using_for %dq_v tile_sizes [1, 1, 1, 1, 1, 0, 0, 0]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
        transform.annotate %dq_vloops#0 "dq_loops" : !transform.any_op
        transform.loop.unroll %dq_vloops#4 factor = 2 : !transform.any_op
        transform.loop.unroll %dq_vloops#3 factor = 2 : !transform.any_op
        // The four K groups too: in the loop left, over the column blocks,
        // scale and min advance by a constant step.

        // A packet's copy into the B tile: per (column block, K row block)
        // 64 contiguous bf16, one vector.
        %cps_bf = transform.structured.match ops{["linalg.generic"]} attributes{b_cp} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %cps_bf : !transform.any_op {
        ^bb0(%cp_bf: !transform.any_op):
          %cpt_bf, %cpl_bf:2 = transform.structured.tile_using_for %cp_bf tile_sizes [1, 1, 0, 0, 0]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
          transform.annotate %cpl_bf#0 "cp_loops" : !transform.any_op
        }

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
        %herd1 = transform.air.par_to_herd %parallel1 <{first_dim = 1}> : (!transform.any_op) -> !transform.any_op
        transform.annotate %herd1 "prologue_herd" : !transform.any_op
        transform.foreach %forall2 : !transform.any_op {
        ^bb0(%f2: !transform.any_op):
          %parallel2 = transform.loop.forall_to_parallel %f2  : (!transform.any_op) -> !transform.any_op
          %herd2 = transform.air.par_to_herd %parallel2 <{first_dim = 1}> : (!transform.any_op) -> !transform.any_op
          transform.annotate %herd2 "compute_herd" : !transform.any_op
        }
        %parallel3 = transform.loop.forall_to_parallel %forall3  : (!transform.any_op) -> !transform.any_op
        %herd3 = transform.air.par_to_herd %parallel3 <{first_dim = 1}> : (!transform.any_op) -> !transform.any_op
        transform.annotate %herd3 "epilogue_herd" : !transform.any_op

        %herds = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %vectorized_herds = transform.air.herd_vectorize %herds : (!transform.any_op) -> !transform.any_op
        %ep_herd2 = transform.structured.match ops{["air.herd"]} attributes{epilogue_herd} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %ep_herd2 { transform.apply_patterns.memref.fold_memref_alias_ops } : !transform.any_op
        %epiloops = transform.structured.match ops{["scf.for"]} attributes{epi_loops} in %arg1 : (!transform.any_op) -> !transform.any_op
        %c_herd2 = transform.structured.match ops{["air.herd"]} attributes{compute_herd} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %c_herd2 { transform.apply_patterns.memref.fold_memref_alias_ops } : !transform.any_op
        %f_cse = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_cse to %f_cse : !transform.any_op
        // The packet copies: one loop of 32 vector moves each, a move a
        // contiguous 64-lane read and write.
        %c_herd_cp_bf = transform.structured.match ops{["scf.for"]} attributes{cp_loops} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %c_herd_cp_bf {
            transform.apply_patterns.vector.cast_away_vector_leading_one_dim
            transform.apply_patterns.vector.flatten_vector_transfer_ops
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        %cploops_bf = transform.structured.match ops{["scf.for"]} attributes{cp_loops} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %cploops_bf : !transform.any_op {
        ^bb0(%cl_bf: !transform.any_op):
          %cll_bf = transform.air.linearize_vectors %cl_bf arch = "aie2p" f32_lanes = 64 contract : (!transform.any_op) -> !transform.any_op
          %clf_bf = transform.loop.coalesce %cll_bf : (!transform.any_op) -> !transform.any_op
        }
        %f_cpx_bf = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %f_cpx2_bf = transform.apply_registered_pass "affine-expand-index-ops-as-affine" to %f_cpx_bf : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f_cpx2_bf { transform.apply_patterns.canonicalization } : !transform.any_op
        %dqloops = transform.structured.match ops{["scf.for"]} attributes{dq_loops} in %arg1 : (!transform.any_op) -> !transform.any_op
        %dqlin = transform.air.linearize_vectors %dqloops arch = "aie2p" f32_lanes = 64 contract : (!transform.any_op) -> !transform.any_op
        %dq_outer = transform.structured.match ops{["scf.for"]} attributes{dq_loops} in %arg1 : (!transform.any_op) -> !transform.any_op
        %dq_flat = transform.loop.coalesce %dq_outer : (!transform.any_op) -> !transform.any_op
        // q and the output advance by a constant per trip: walk them with
        // pointers instead of recomputing each address.
        %f_dqx = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %f_dqx2 = transform.apply_registered_pass "affine-expand-index-ops-as-affine" to %f_dqx : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f_dqx2 { transform.apply_patterns.canonicalization } : !transform.any_op
        %dq_flat_r = transform.structured.match ops{["scf.for"]} attributes{dq_loops} in %arg1 : (!transform.any_op) -> !transform.any_op
        %dq_ptr = transform.air.hoist_vector_transfer_pointers %dq_flat_r : (!transform.any_op) -> !transform.any_op
        // Scale and base depend only on the column block and the group:
        // their reads and arithmetic leave the inner loops.
        %f_hoist = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %f_hoisted = transform.structured.hoist_redundant_vector_transfers %f_hoist : (!transform.any_op) -> !transform.any_op
        %c_herd3 = transform.structured.match ops{["air.herd"]} attributes{compute_herd} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %c_herd3 : !transform.any_op {
        ^bb0(%ch: !transform.any_op):
          %c_loops = transform.structured.match ops{["scf.for"]} in %ch : (!transform.any_op) -> !transform.any_op
          transform.apply_licm to %c_loops : !transform.any_op
        }
        // The activation runs only when its flag is set: the epilogue loop
        // in one copy per value of the flag, the clear one just the cast.
        %epi_u = transform.air.unswitch_loop %epiloops : (!transform.any_op) -> !transform.any_op
        %f_epi = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f_epi { transform.apply_patterns.canonicalization } : !transform.any_op
        %epiloops2 = transform.structured.match ops{["scf.for"]} attributes{epi_loops} in %arg1 : (!transform.any_op) -> !transform.any_op
        %epilin = transform.air.linearize_vectors %epiloops2 arch = "aie2p" f32_lanes = 32 contract : (!transform.any_op) -> !transform.any_op

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


        // Cast accumulator (input[2]) and output[0] to f32
        %vector_contracts = transform.structured.match ops{["vector.contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %result11 = transform.air.vector_type_cast %vector_contracts <{target_element_type = f32, input_indices = [2], output_indices = [0]}> : (!transform.any_op) -> !transform.any_op

        // Cast vector.contract inputs 0,1 to bf16
        // (matches hardware MAC unit native input type)
        %vector_contracts_2 = transform.structured.match ops{["vector.contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %result11b = transform.air.vector_type_cast %vector_contracts_2 <{target_element_type = bf16, input_indices = [0, 1], output_indices = []}> : (!transform.any_op) -> !transform.any_op

        %herd2_1 = transform.structured.match ops{["air.herd"]} attributes{compute_herd} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %herd2_1 : !transform.any_op {
        ^bb0(%h: !transform.any_op):
          %innermost_for = transform.structured.match ops{["scf.for"]} attributes{mm_inner_loop} in %h : (!transform.any_op) -> !transform.any_op
          %innermost_for_updated_3 = transform.air.hoist_loop_invariant_transfers %h, %innermost_for : (!transform.any_op, !transform.any_op) -> !transform.any_op
          %innermost_for_updated_4 = transform.air.flatten_for_iter_args %innermost_for_updated_3 : (!transform.any_op) -> !transform.any_op
          %innermost_for_updated_5 = transform.air.hoist_vector_transfer_pointers %innermost_for_updated_4 : (!transform.any_op) -> !transform.any_op
        }

    //==========================================================================
    // PHASE 12: FINAL LOOP OPTIMIZATIONS
    //==========================================================================


        %func9 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func9 {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
            transform.apply_patterns.memref.fold_memref_alias_ops
        } : !transform.any_op
        %func_fold_2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %func_folded_2 = transform.air.fold_unit_extent_dims %func_fold_2 : (!transform.any_op) -> !transform.any_op
    //==========================================================================
    // PHASE 13: OUTLINE THE MATMUL AND THE DEQUANT (air-FLM's functions)
    //==========================================================================

        // As in air-FLM, the matmul and the dequant are functions of their
        // own: values one keeps live do not take registers from the other's
        // pipelined loop.
        %dq_outs = transform.structured.match ops{["scf.for"]} attributes{dq_loops} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %dq_outs : !transform.any_op {
        ^bb0(%l: !transform.any_op):
          %fn, %call = transform.loop.outline %l func_name = "dq4_bf16" : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
          transform.annotate %fn "no_inline" : !transform.any_op
          %priv = transform.param.constant "private" -> !transform.any_param
          transform.annotate %fn "sym_visibility" = %priv : !transform.any_op, !transform.any_param
        }
        // The functions are private and called once: the constants the
        // outlining passed in fold into them, and the arguments left unused go.
        %mod = transform.structured.match ops{["builtin.module"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %m_sccp = transform.apply_registered_pass "sccp" to %mod : (!transform.any_op) -> !transform.any_op
        %m_rdv = transform.apply_registered_pass "remove-dead-values" to %m_sccp : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
