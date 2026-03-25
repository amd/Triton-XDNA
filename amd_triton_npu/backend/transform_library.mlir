// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Shared Transform Dialect Library for Triton-XDNA
//
// This file contains reusable named sequences that capture common patterns
// across transform scripts. The driver.py tool parses this library and
// inlines referenced sequences at each `transform.include` site in user scripts.
//
// Usage in per-example scripts:
//   transform.include @canonicalize_with_cse failures(propagate) (%arg1)
//       : (!transform.any_op) -> ()
//
// NOTE: This file contains bare named sequences without a module wrapper.
// The driver does not inject these definitions into the user's module; instead,
// it inlines the bodies of referenced sequences at each `transform.include` site.

//===----------------------------------------------------------------------===//
// Canonicalization Sequences
//===----------------------------------------------------------------------===//

// Standard canonicalization + CSE (the most frequently repeated block).
transform.named_sequence @canonicalize_with_cse(
    %module: !transform.any_op {transform.readonly}) {
  %func = transform.structured.match ops{["func.func"]} in %module
      : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %func {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
  } : !transform.any_op
  transform.apply_cse to %func : !transform.any_op
  transform.yield
}

// Initial canonicalization with fold_unit_extent_dims (Phase 1 of elementwise).
transform.named_sequence @canonicalize_with_fold_dims(
    %module: !transform.any_op {transform.readonly}) {
  %func = transform.structured.match ops{["func.func"]} in %module
      : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %func {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
  } : !transform.any_op
  transform.apply_cse to %func : !transform.any_op
  transform.yield
}

//===----------------------------------------------------------------------===//
// Elementwise Fusion
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Elementwise Flatten and Distribute
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Pad and Promote to L1
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Bufferization
//===----------------------------------------------------------------------===//

// One-shot bufferization of the function.
transform.named_sequence @one_shot_bufferize(
    %module: !transform.any_op {transform.readonly}) {
  %func_op = transform.structured.match ops{["func.func"]} in %module
      : (!transform.any_op) -> !transform.any_op
  %func_bufferized = transform.bufferization.one_shot_bufferize %func_op
      : (!transform.any_op) -> !transform.any_op
  transform.yield
}

// Post-bufferization cleanup: canonicalize, convert copies, remove uninit,
// eliminate cascade memcpy.
transform.named_sequence @post_bufferize_cleanup(
    %module: !transform.any_op {transform.readonly}) {
  %func6 = transform.structured.match ops{["func.func"]} in %module
      : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %func6 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
  } : !transform.any_op
  transform.apply_cse to %func6 : !transform.any_op
  transform.apply_patterns to %func6 {
      transform.apply_patterns.canonicalization
  } : !transform.any_op
  %linalg_copies = transform.structured.match ops{["linalg.copy"]} in %module
      : (!transform.any_op) -> !transform.any_op
  %memref_copies = transform.structured.linalg_copy_to_memref %linalg_copies
      : (!transform.any_op) -> !transform.any_op
  %func_op_updated = transform.air.remove_uninitialized_copy %func6
      : (!transform.any_op) -> !transform.any_op
  %func_op_updated_1 = transform.air.eliminate_cascade_memcpy %func_op_updated
      : (!transform.any_op) -> !transform.any_op
  transform.yield
}

//===----------------------------------------------------------------------===//
// AIR Herd Mapping and Vectorization
//===----------------------------------------------------------------------===//

// Convert forall -> parallel -> herd -> copy_to_dma -> vectorize.
// Returns the vectorized herd handle for subsequent type casts.
transform.named_sequence @air_herd_mapping_and_vectorize(
    %module: !transform.any_op {transform.readonly})
    -> !transform.any_op {
  %forall_as_herd = transform.structured.match ops{["scf.forall"]} in %module
      : (!transform.any_op) -> !transform.any_op
  %parallel = transform.loop.forall_to_parallel %forall_as_herd
      : (!transform.any_op) -> !transform.any_op
  %herd = transform.air.par_to_herd %parallel
      : (!transform.any_op) -> !transform.any_op
  %copies_in_herd = transform.structured.match
      ops{["memref.copy", "linalg.copy"]} in %herd
      : (!transform.any_op) -> !transform.any_op
  %dmas_from_copies = transform.air.copy_to_dma %copies_in_herd
      : (!transform.any_op) -> !transform.any_op
  %vectorized_herd = transform.air.herd_vectorize %herd
      : (!transform.any_op) -> !transform.any_op
  transform.yield %vectorized_herd : !transform.any_op
}

// Variant with extern_func.o linking (for AIE2 kernels using math ops like exp).
transform.named_sequence @air_herd_mapping_with_extern_and_vectorize(
    %module: !transform.any_op {transform.readonly})
    -> !transform.any_op {
  %forall_as_herd = transform.structured.match ops{["scf.forall"]} in %module
      : (!transform.any_op) -> !transform.any_op
  %parallel = transform.loop.forall_to_parallel %forall_as_herd
      : (!transform.any_op) -> !transform.any_op
  %herd = transform.air.par_to_herd %parallel
      : (!transform.any_op) -> !transform.any_op
  %extern_func_param = transform.param.constant "extern_func.o"
      -> !transform.any_param
  transform.annotate %herd "link_with" = %extern_func_param
      : !transform.any_op, !transform.any_param
  %copies_in_herd = transform.structured.match
      ops{["memref.copy", "linalg.copy"]} in %herd
      : (!transform.any_op) -> !transform.any_op
  %dmas_from_copies = transform.air.copy_to_dma %copies_in_herd
      : (!transform.any_op) -> !transform.any_op
  %vectorized_herd = transform.air.herd_vectorize %herd
      : (!transform.any_op) -> !transform.any_op
  transform.yield %vectorized_herd : !transform.any_op
}

//===----------------------------------------------------------------------===//
// Vector Type Casting via foreach_match
//
// AIE2P constraint: most arithmetic/transcendental ops are bf16-only,
// while divf, rsqrt, reciprocal are f32-only.
// These matchers + actions use foreach_match to cast all bf16-only ops
// in a single declarative pass over the vectorized herd.
//===----------------------------------------------------------------------===//

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
