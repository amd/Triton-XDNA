// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
// Triton RMSNorm Tiling Recipe Transform Script (AIE2P)
//===----------------------------------------------------------------------===//
// RMSNorm: out = x * rsqrt(mean(x^2) + eps)
// Unlike LayerNorm there is a SINGLE reduction (sum of squares) and no mean
// subtraction / no bias. The linalg form is:
//   generic1 (x*x) -> reduce1 (sum) -> output_generic (rsqrt + mul)
// This script mirrors transform_layernorm_aie2p.mlir but navigates one
// reduce anchor instead of two.
//===----------------------------------------------------------------------===//

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {

        //===================================================================
        // PHASE 1: Initial Canonicalization and Cleanup
        //===================================================================
        transform.include @canonicalize_with_fold_dims failures(propagate) (%arg1) : (!transform.any_op) -> ()

        //===================================================================
        // PHASE 2: Elementwise Fusion and Reduction Transformation
        //===================================================================
        %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %fused_func = transform.air.fuse_elementwise_linalg %func1 : (!transform.any_op) -> !transform.any_op

        %reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1  : (!transform.any_op) -> !transform.any_op
        %transformed_reduces = transform.air.transpose_reduce %reduces : (!transform.any_op) -> !transform.any_op

        transform.include @canonicalize_with_cse failures(propagate) (%arg1) : (!transform.any_op) -> ()

        // Single reduce anchor. Chain: generic1 -> reduce1 -> output_generic
        %reduce1 = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op

        %generic1 = transform.get_producer_of_operand %reduce1[0]
            : (!transform.any_op) -> !transform.any_op

        %materialize = transform.structured.match ops{["bufferization.materialize_in_destination"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %output_generic = transform.get_producer_of_operand %materialize[0]
            : (!transform.any_op) -> !transform.any_op

        %fill = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!transform.any_op) -> !transform.any_op

        //===================================================================
        // PHASE 3: Batch-Level Tiling and Producer-Consumer Fusion
        //===================================================================
        %output_buf, %new_output = transform.structured.bufferize_to_allocation %output_generic
          {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

        %tiled_output, %forall_4 =
        transform.structured.tile_using_forall %output_generic tile_sizes [1]  : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        // Backward fusion in reverse data-flow order
        %fused_reduce1, %4 = transform.structured.fuse_into_containing_op %reduce1 into %forall_4 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %fused_generic1, %6 = transform.structured.fuse_into_containing_op %generic1 into %forall_4 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %fused_fill, %7 = transform.structured.fuse_into_containing_op %fill into %forall_4 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

        //===================================================================
        // PHASE 4: Post-Fusion Canonicalization
        //===================================================================
        transform.include @canonicalize_with_cse failures(propagate) (%arg1) : (!transform.any_op) -> ()

        //===================================================================
        // PHASE 5: L2 Memory Allocation for Intermediate Buffers
        //===================================================================
        %fills_2 = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!transform.any_op) -> !transform.any_op
        %fill1_buffer, %fill1_new = transform.structured.bufferize_to_allocation %fills_2
          {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

        // After fusion there are 2 generics (x*x feeding reduce, and the final
        // rsqrt*x output) plus 1 reduce. Split the two generics; reduce is single.
        %generics2 = transform.structured.match ops{["linalg.generic"]} in %arg1  : (!transform.any_op) -> !transform.any_op
        %tiled_generic1, %tiled_generic2 = transform.split_handle %generics2 : (!transform.any_op<"linalg.generic">) -> (!transform.any_op<"linalg.generic">, !transform.any_op<"linalg.generic">)
        %reduces2 = transform.structured.match ops{["linalg.reduce"]} in %arg1  : (!transform.any_op) -> !transform.any_op

        // Promote input tensor to L2 memory
        %op0 = transform.get_operand %tiled_generic1[0]
            : (!transform.any_op) -> !transform.any_value
        transform.structured.promote_tensor to 2 %op0 : !transform.any_value

        %gen1_in_buffer, %gen1_in_new = transform.structured.bufferize_to_allocation %tiled_generic1
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

        %red1_in_buffer, %red1_in_new = transform.structured.bufferize_to_allocation %reduces2
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

        %gen2_in_buffer, %gen2_in_new = transform.structured.bufferize_to_allocation %tiled_generic2
            {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

        //===================================================================
        // PHASE 6: Final Canonicalization
        //===================================================================
        transform.include @canonicalize_with_cse failures(propagate) (%arg1) : (!transform.any_op) -> ()

        //===================================================================
        // PHASE 7: Complete Bufferization
        //===================================================================
        transform.include @one_shot_bufferize failures(propagate) (%arg1) : (!transform.any_op) -> ()

        //===================================================================
        // PHASE 8: Post-Bufferization Cleanup
        //===================================================================
        transform.include @canonicalize_with_cse failures(propagate) (%arg1) : (!transform.any_op) -> ()
        %func6 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func6 {
            transform.apply_patterns.canonicalization
        } : !transform.any_op

        %func_op_updated = transform.air.remove_uninitialized_copy %func6 : (!transform.any_op) -> !transform.any_op

        //===================================================================
        // PHASE 9: Vectorization Preparation
        //===================================================================
        %remaining_reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %generalized_reduces = transform.structured.generalize %remaining_reduces : (!transform.any_op) -> !transform.any_op

        %linalg_generics = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %inner_most_generics, %vec_loops:1 =
          transform.structured.tile_using_for %linalg_generics tile_sizes [0, 16]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        %func_op_updated_1 = transform.air.convert_divf_sqrt_to_rsqrt %func_op_updated : (!transform.any_op) -> !transform.any_op

        //===================================================================
        // PHASE 10: AIE Hardware Mapping and Vectorization
        //===================================================================
        %forall_as_herd = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %parallel = transform.loop.forall_to_parallel %forall_as_herd  : (!transform.any_op) -> !transform.any_op
        %herd = transform.air.par_to_herd %parallel : (!transform.any_op) -> !transform.any_op

        %copies_in_herd = transform.structured.match ops{["memref.copy", "linalg.copy"]} in %herd : (!transform.any_op) -> !transform.any_op
        %dmas_from_copies = transform.air.copy_to_dma %copies_in_herd : (!transform.any_op) -> !transform.any_op

        %vectorized_herd = transform.air.herd_vectorize %herd : (!transform.any_op) -> !transform.any_op

        %func4 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func4 {
            transform.apply_patterns.canonicalization
            transform.apply_patterns.vector.cast_away_vector_leading_one_dim
        } : !transform.any_op

        %vectorized_herd_updated = transform.air.broadcast_before_unary %func4 {op_name = "math.rsqrt"} : (!transform.any_op) -> !transform.any_op

        %vector_reductions_in_herd = transform.structured.match ops{["vector.multi_reduction"]} in %vectorized_herd_updated : (!transform.any_op) -> !transform.any_op
        %result10 = transform.air.vector_type_cast %vector_reductions_in_herd {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op

        %vector_muls_in_herd = transform.structured.match ops{["arith.mulf"]} in %vectorized_herd_updated : (!transform.any_op) -> !transform.any_op
        %result11 = transform.air.vector_type_cast %vector_muls_in_herd {target_element_type = bf16} : (!transform.any_op) -> !transform.any_op

        %func7 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %func7_transformed = transform.air.convert_size1_vector_to_scalar %func7 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %func7_transformed {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.scf.for_loop_canonicalization
            transform.apply_patterns.canonicalization
            transform.apply_patterns.vector.reorder_multi_reduction_dims lowering_strategy = "innerreduction"
            transform.apply_patterns.vector.multi_reduction_flattening lowering_strategy = "innerreduction"
            transform.apply_patterns.vector.multi_reduction_unrolling lowering_strategy = "innerreduction"
        } : !transform.any_op
        transform.apply_cse to %func7_transformed : !transform.any_op
    transform.yield
  }
}
