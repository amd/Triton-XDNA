// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

////////////////////////////////////////////////////////////////////////////////
// Transform Script for Leaky ReLU (AIE2P)
// leaky_relu(x) = x if x > 0, else alpha * x
// Cast cmpf inputs and select values to bf16.
// Uses shared library sequences from transform_library.mlir (auto-injected).
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
    transform.include @pad_and_promote_unary_bf16 failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @canonicalize_with_cse failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @one_shot_bufferize failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @post_bufferize_cleanup failures(propagate)
        (%arg1) : (!transform.any_op) -> ()

    %generics = transform.structured.match ops{["linalg.generic"]}
        in %arg1 : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:1 = transform.structured.tile_using_for %generics
        tile_sizes [16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %vh = transform.include @air_herd_mapping_and_vectorize
        failures(propagate) (%arg1) : (!transform.any_op) -> !transform.any_op

    // Cast cmpf inputs and select values to bf16
    transform.foreach_match in %vh
        @match_cmpf -> @action_cast_cmpf_to_bf16,
        @match_select -> @action_cast_select_to_bf16
        : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
