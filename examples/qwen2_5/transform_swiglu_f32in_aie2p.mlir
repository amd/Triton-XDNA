// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
////////////////////////////////////////////////////////////////////////////////
// Transform Script for fused SwiGLU merge with f32 inputs, bf16 output (AIE2P).
// Variant for load_pdi fused chains where the SwiGLU merge (silu(gate) * up)
// consumes two f32 buffers produced on-device by the gate/up matmuls and emits a
// bf16 buffer for the following down matmul. Binary analog of gpt2's
// transform_gelu_f32in_aie2p.mlir: uses @pad_and_promote_binary_f32in_bf16out so
// the two input pad values match the f32 operands and the result pad value is
// bf16.
////////////////////////////////////////////////////////////////////////////////

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {

    transform.include @canonicalize_with_fold_dims failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @fuse_elementwise_and_canonicalize failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @flatten_tile_forall failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @canonicalize_with_cse failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @pad_and_promote_binary_f32in_bf16out failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @canonicalize_with_cse failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @one_shot_bufferize failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @post_bufferize_cleanup failures(propagate)
        (%arg1) : (!transform.any_op) -> ()

    transform.include @vectorize_generics_at_16 failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    %vh = transform.include @air_herd_mapping_and_vectorize
        failures(propagate) (%arg1) : (!transform.any_op) -> !transform.any_op
    transform.include @cast_bf16_only_ops failures(propagate)
        (%vh) : (!transform.any_op) -> ()

    transform.yield
  }
}
