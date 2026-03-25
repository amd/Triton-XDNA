// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Vector type casting sequences for AIE2P bf16/f32 constraints.
//
// AIE2P constraint: most arithmetic/transcendental ops are bf16-only,
// while divf, rsqrt, reciprocal are f32-only.

// Matcher: succeeds on ops that must be cast to bf16 on AIE2P.
transform.named_sequence @match_bf16_only_op(
    %op: !transform.any_op {transform.readonly}) -> !transform.any_op {
  transform.match.operation_name %op
      ["math.exp", "arith.addf", "arith.subf", "arith.mulf",
       "arith.maxnumf", "vector.multi_reduction"] : !transform.any_op
  transform.yield %op : !transform.any_op
}

// Action: cast a matched op's vector types to bf16.
transform.named_sequence @action_cast_to_bf16(
    %op: !transform.any_op {transform.consumed}) {
  %cast = transform.air.vector_type_cast %op {target_element_type = bf16}
      : (!transform.any_op) -> !transform.any_op
  transform.yield
}

// Matcher: succeeds on arith.cmpf (needs input-only cast).
transform.named_sequence @match_cmpf(
    %op: !transform.any_op {transform.readonly}) -> !transform.any_op {
  transform.match.operation_name %op ["arith.cmpf"] : !transform.any_op
  transform.yield %op : !transform.any_op
}

// Action: cast cmpf inputs [0,1] to bf16, leave i1 result alone.
transform.named_sequence @action_cast_cmpf_to_bf16(
    %op: !transform.any_op {transform.consumed}) {
  %cast = transform.air.vector_type_cast %op
      {target_element_type = bf16, input_indices = [0, 1]}
      : (!transform.any_op) -> !transform.any_op
  transform.yield
}

// Matcher: succeeds on arith.select (needs value-only cast).
transform.named_sequence @match_select(
    %op: !transform.any_op {transform.readonly}) -> !transform.any_op {
  transform.match.operation_name %op ["arith.select"] : !transform.any_op
  transform.yield %op : !transform.any_op
}

// Action: cast select values [1,2] and output [0] to bf16, leave i1 condition alone.
transform.named_sequence @action_cast_select_to_bf16(
    %op: !transform.any_op {transform.consumed}) {
  %cast = transform.air.vector_type_cast %op
      {target_element_type = bf16, input_indices = [1, 2], output_indices = [0]}
      : (!transform.any_op) -> !transform.any_op
  transform.yield
}

// Cast all bf16-only vector ops (exp, addf, subf, mulf, maxnumf,
// multi_reduction) from f32 to bf16 in the given vectorized herd.
// f32-only ops (divf, rsqrt) are left unchanged.
// Used by: sigmoid, silu, gelu, relu, axpy, swiglu.
transform.named_sequence @cast_bf16_only_ops(
    %herd: !transform.any_op {transform.consumed}) {
  transform.foreach_match in %herd
      @match_bf16_only_op -> @action_cast_to_bf16
      : (!transform.any_op) -> !transform.any_op
  transform.yield
}

// Cast arith.cmpf inputs and arith.select values to bf16 in the given
// vectorized herd. Leaves i1 condition/result types unchanged.
// Used by: leaky_relu.
transform.named_sequence @cast_cmpf_and_select_ops(
    %herd: !transform.any_op {transform.consumed}) {
  transform.foreach_match in %herd
      @match_cmpf -> @action_cast_cmpf_to_bf16,
      @match_select -> @action_cast_select_to_bf16
      : (!transform.any_op) -> !transform.any_op
  transform.yield
}
