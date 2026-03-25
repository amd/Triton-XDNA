// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Vectorization tiling and post-vectorization cleanup sequences.

// Tile all linalg.generic ops for vectorization at 16 vector lanes.
// Standard width for most bf16 operations on AIE2 and AIE2P.
transform.named_sequence @vectorize_generics_at_16(
    %module: !transform.any_op {transform.readonly}) {
  %generics = transform.structured.match ops{["linalg.generic"]}
      in %module : (!transform.any_op) -> !transform.any_op
  %tiled, %loops:1 = transform.structured.tile_using_for %generics
      tile_sizes [16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.yield
}

// Tile all linalg.generic ops for vectorization at 32 vector lanes.
// Used on AIE2P for simple ops (e.g. add) that support wider vectors.
transform.named_sequence @vectorize_generics_at_32(
    %module: !transform.any_op {transform.readonly}) {
  %generics = transform.structured.match ops{["linalg.generic"]}
      in %module : (!transform.any_op) -> !transform.any_op
  %tiled, %loops:1 = transform.structured.tile_using_for %generics
      tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.yield
}

// Post-vectorization cleanup for reduction kernels (softmax, layernorm,
// rms_norm): convert size-1 vectors to scalars (downstream compiler
// cannot handle them), then lower multi_reduction and cast away
// leading-one vector dims.
transform.named_sequence @post_vectorize_reduce_cleanup(
    %module: !transform.any_op {transform.readonly}) {
  %func = transform.structured.match ops{["func.func"]} in %module
      : (!transform.any_op) -> !transform.any_op
  %func_done = transform.air.convert_size1_vector_to_scalar %func
      : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %func_done {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
      transform.apply_patterns.vector.lower_multi_reduction
          lowering_strategy = "innerreduction"
  } : !transform.any_op
  transform.apply_cse to %func_done : !transform.any_op
  transform.yield
}
