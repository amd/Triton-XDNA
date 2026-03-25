// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

////////////////////////////////////////////////////////////////////////////////
// Transform Script for Vector Addition (AIE2)
// Simple elementwise add: out = a + b
// Binary op (2 inputs + 1 output). No fusion needed. Vec tile = 16 (AIE2).
// No type casts needed (bf16 add is native).
// Uses shared library sequences from transform_library.mlir (auto-injected).
////////////////////////////////////////////////////////////////////////////////

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {

    // No Phase 1/2 for vec-add (no elementwise fusion needed)
    transform.include @flatten_tile_forall failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @canonicalize_with_cse failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @pad_and_promote_binary_bf16 failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @canonicalize_with_cse failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @one_shot_bufferize failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @post_bufferize_cleanup failures(propagate)
        (%arg1) : (!transform.any_op) -> ()

    // Vectorization tiling (16-lane for AIE2)
    %generics = transform.structured.match ops{["linalg.generic"]}
        in %arg1 : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:1 = transform.structured.tile_using_for %generics
        tile_sizes [16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // AIR mapping (no type casts needed for vec-add)
    %vh = transform.include @air_herd_mapping_and_vectorize
        failures(propagate) (%arg1) : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
