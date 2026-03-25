// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Bufferization sequences: tensor-to-memref conversion and cleanup.

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
