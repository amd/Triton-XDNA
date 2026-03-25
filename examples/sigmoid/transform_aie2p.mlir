// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

////////////////////////////////////////////////////////////////////////////////
// Transform Script for Sigmoid (AIE2P)
//
// sigmoid(x) = 1 / (1 + exp(-x))
//
// Strategy: fuse_elementwise_linalg -> unary pad+promote -> vectorize at 16
// -> cast exp, subf, addf, mulf to bf16; divf stays f32.
//
// Uses shared library sequences from transform_library.mlir (auto-injected).
////////////////////////////////////////////////////////////////////////////////

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {

    // Phase 1: Initial canonicalization
    transform.include @canonicalize_with_fold_dims failures(propagate)
        (%arg1) : (!transform.any_op) -> ()

    // Phase 2: Fuse elementwise chain (extf + subf + exp + addf + divf + truncf)
    transform.include @fuse_elementwise_and_canonicalize failures(propagate)
        (%arg1) : (!transform.any_op) -> ()

    // Phase 3: Flatten + tile forall [256]
    transform.include @flatten_tile_forall failures(propagate)
        (%arg1) : (!transform.any_op) -> ()

    // Phase 4: Canonicalization
    transform.include @canonicalize_with_cse failures(propagate)
        (%arg1) : (!transform.any_op) -> ()

    // Phase 5: Pad and promote to L1 (unary: 1 input + 1 output)
    transform.include @pad_and_promote_unary_bf16 failures(propagate)
        (%arg1) : (!transform.any_op) -> ()

    // Phase 6: Canonicalization
    transform.include @canonicalize_with_cse failures(propagate)
        (%arg1) : (!transform.any_op) -> ()

    // Phase 7: Bufferization
    transform.include @one_shot_bufferize failures(propagate)
        (%arg1) : (!transform.any_op) -> ()

    // Phase 8: Post-bufferization cleanup
    transform.include @post_bufferize_cleanup failures(propagate)
        (%arg1) : (!transform.any_op) -> ()

    // Phase 9: Vectorization tiling (16-lane for bf16)
    %linalg_generics = transform.structured.match ops{["linalg.generic"]}
        in %arg1 : (!transform.any_op) -> !transform.any_op
    %inner_most_generics, %vec_loops:1 =
        transform.structured.tile_using_for %linalg_generics tile_sizes [16]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Phase 10: AIR herd mapping + vectorization
    %vectorized_herd = transform.include @air_herd_mapping_and_vectorize
        failures(propagate) (%arg1)
        : (!transform.any_op) -> !transform.any_op

    // Cast all bf16-only ops (exp, subf, addf, mulf) to bf16; divf stays f32
    transform.foreach_match in %vectorized_herd
        @match_bf16_only_op -> @action_cast_to_bf16
        : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
