// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Elementwise operation sequences: fusion, flattening, padding, and promotion.

// Fuse elementwise linalg chain (extf + compute + truncf) then canonicalize.
transform.named_sequence @fuse_elementwise_and_canonicalize(
    %module: !transform.any_op {transform.readonly}) {
  %func1 = transform.structured.match ops{["func.func"]} in %module
      : (!transform.any_op) -> !transform.any_op
  %func1_fused = transform.air.fuse_elementwise_linalg %func1
      : (!transform.any_op) -> !transform.any_op
  %func1a = transform.structured.match ops{["func.func"]} in %module
      : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %func1a {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
  } : !transform.any_op
  transform.apply_cse to %func1a : !transform.any_op
  transform.yield
}

// Flatten to 1D, allocate result in L2, tile forall [256] for multi-core.
transform.named_sequence @flatten_tile_forall(
    %module: !transform.any_op {transform.readonly}) {
  %op = transform.structured.match ops{["linalg.generic"]} in %module
      : (!transform.any_op) -> !transform.any_op
  %op_flattened = transform.structured.flatten_elementwise %op
      : (!transform.any_op) -> !transform.any_op
  %op_res_shared, %new_op = transform.structured.bufferize_to_allocation
      %op_flattened
      {memory_space = 1, bufferize_destination_only, emit_dealloc}
      : !transform.any_op
  %op_1 = transform.structured.match ops{["linalg.generic"]} in %module
      : (!transform.any_op) -> !transform.any_op
  %tiled_op_1, %forall_op_1 =
      transform.structured.tile_using_forall %op_1 tile_sizes [256]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.yield
}

// Unary variant: 1 input + 1 output = 2 operands (relu, sigmoid, silu, gelu).
transform.named_sequence @pad_and_promote_unary_bf16(
    %module: !transform.any_op {transform.readonly}) {
  %op = transform.structured.match ops{["linalg.generic"]} in %module
      : (!transform.any_op) -> !transform.any_op
  %padded_op, %pad_op, %__ = transform.structured.pad %op {
      padding_values=[0.0 : bf16, 0.0 : bf16],
      padding_dimensions=[0, 1],
      nofold_flags=[1, 1],
      copy_back_op="linalg.copy"
  } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
  %pad_dps = transform.structured.rewrite_in_destination_passing_style %pad_op
      : (!transform.any_op) -> !transform.any_op
  %padded_input = transform.get_producer_of_operand %padded_op[0]
      : (!transform.any_op) -> (!transform.any_op)
  %padded_input_buffer, %padded_input_new =
      transform.structured.bufferize_to_allocation %padded_input
      {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op
  %padded_result = transform.get_producer_of_operand %padded_op[1]
      : (!transform.any_op) -> (!transform.any_op)
  %padded_result_buffer, %padded_result_new =
      transform.structured.bufferize_to_allocation %padded_result
      {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op
  transform.yield
}

// Binary variant: 2 inputs + 1 output = 3 operands (vec-add, axpy, swiglu).
transform.named_sequence @pad_and_promote_binary_bf16(
    %module: !transform.any_op {transform.readonly}) {
  %op = transform.structured.match ops{["linalg.generic"]} in %module
      : (!transform.any_op) -> !transform.any_op
  %padded_op, %pad_op, %__ = transform.structured.pad %op {
      padding_values=[0.0 : bf16, 0.0 : bf16, 0.0 : bf16],
      padding_dimensions=[0, 1, 2],
      nofold_flags=[1, 1, 1],
      copy_back_op="linalg.copy"
  } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
  %pad_dps = transform.structured.rewrite_in_destination_passing_style %pad_op
      : (!transform.any_op) -> !transform.any_op
  %padded_lhs = transform.get_producer_of_operand %padded_op[0]
      : (!transform.any_op) -> (!transform.any_op)
  %padded_lhs_buffer, %padded_lhs_new =
      transform.structured.bufferize_to_allocation %padded_lhs
      {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op
  %padded_rhs = transform.get_producer_of_operand %padded_op[1]
      : (!transform.any_op) -> (!transform.any_op)
  %padded_rhs_buffer, %padded_rhs_new =
      transform.structured.bufferize_to_allocation %padded_rhs
      {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op
  %padded_result = transform.get_producer_of_operand %padded_op[2]
      : (!transform.any_op) -> (!transform.any_op)
  %padded_result_buffer, %padded_result_new =
      transform.structured.bufferize_to_allocation %padded_result
      {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op
  transform.yield
}
