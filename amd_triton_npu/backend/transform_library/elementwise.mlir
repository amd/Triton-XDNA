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

// Flatten to 1D, allocate result in L2, split across a fixed number of cores.
// num_threads (not tile_sizes) keeps the herd width independent of block size.
// With tile_sizes the width was ceildiv(block, tile): a single trip when the
// block fits one tile (the forall is then folded away, leaving no herd) and
// wider than the target's column count for large blocks (placement fails). A
// fixed thread count avoids both.
//
// The count is hardcoded to 4 for the npu1 4-column array. AIE2P (npu2)
// elementwise scripts include @flatten_tile_forall_aie2p below instead, which
// tiles into 8 threads to fill the 8-column Strix array.
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
      // 4 = npu1 column count (hardcoded; AIE2P uses the _aie2p variant below).
      transform.structured.tile_using_forall %op_1 num_threads [4]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.yield
}

// AIE2P (npu2) variant of @flatten_tile_forall: 8 threads for the 8-column
// Strix array instead of npu1's 4. Identical otherwise.
//
// DEPENDS ON mlir-air PR #1696 (Xilinx/mlir-air): "Preserve launch base offset
// when splitting L2 memref". The 8-way split this triggers exposed a bug in
// air-split-l2-memref where the per-iteration air.launch base offset was
// dropped, so a multi-program (grid > 1) elementwise kernel silently moved
// only the first program's data on npu2. Without an mlir-air build that
// contains that fix, grid > 1 produces wrong results here; grid == 1 (a single
// large block split across the herd) is correct regardless.
transform.named_sequence @flatten_tile_forall_aie2p(
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
      // 8 = npu2/AIE2P column count (Strix). See dependency note above.
      transform.structured.tile_using_forall %op_1 num_threads [8]
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

// Unary variant with f32 input, bf16 output (1 input + 1 output = 2 operands).
// For fused chains where a producing op (e.g. an f32-accumulate matmul) hands a
// non-padded f32 buffer to a unary elementwise that emits bf16 -- the input pad
// value must match the f32 operand element type, the output the bf16 one.
transform.named_sequence @pad_and_promote_unary_f32in_bf16out(
    %module: !transform.any_op {transform.readonly}) {
  %op = transform.structured.match ops{["linalg.generic"]} in %module
      : (!transform.any_op) -> !transform.any_op
  %padded_op, %pad_op, %__ = transform.structured.pad %op {
      padding_values=[0.0 : f32, 0.0 : bf16],
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

// Binary variant with f32 inputs and bf16 output (2 inputs + 1 output).
// For fused chains where a binary elementwise consumes two non-padded f32
// buffers produced on-device (e.g. two f32-accumulate matmuls) and produces a
// bf16 buffer for a following op -- e.g. the SwiGLU merge silu(gate)*up at the
// middle of the load_pdi MLP chain (gate_f32, up_f32 -> H_bf16 feeding the down
// matmul). Input pad values are f32; the result pad value is bf16. Mirrors the
// unary @pad_and_promote_unary_f32in_bf16out.
transform.named_sequence @pad_and_promote_binary_f32in_bf16out(
    %module: !transform.any_op {transform.readonly}) {
  %op = transform.structured.match ops{["linalg.generic"]} in %module
      : (!transform.any_op) -> !transform.any_op
  %padded_op, %pad_op, %__ = transform.structured.pad %op {
      padding_values=[0.0 : f32, 0.0 : f32, 0.0 : bf16],
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

// Binary variant with all-f32 operands (2 inputs + 1 output = 3 operands).
// For fused chains where a producing op (e.g. an f32-accumulate matmul) hands a
// non-padded f32 buffer to a binary elementwise that stays in f32 -- e.g. the
// residual add at the tail of the load_pdi MLP chain (out = mlp_proj_f32 +
// residual_f32). All three pad values must match the f32 operand element type.
transform.named_sequence @pad_and_promote_binary_f32(
    %module: !transform.any_op {transform.readonly}) {
  %op = transform.structured.match ops{["linalg.generic"]} in %module
      : (!transform.any_op) -> !transform.any_op
  %padded_op, %pad_op, %__ = transform.structured.pad %op {
      padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32],
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
