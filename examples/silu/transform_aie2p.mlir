// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

////////////////////////////////////////////////////////////////////////////////
// Transform Script for SiLU (AIE2P)
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Strategy: fuse_elementwise_linalg -> unary pad+promote -> vectorize at 16
// -> cast exp, subf, addf, mulf to bf16; divf stays f32.
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

    // Cast bf16-only ops; divf stays f32
    transform.foreach_match in %vh
        @match_bf16_only_op -> @action_cast_to_bf16
        : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
