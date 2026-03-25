// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Canonicalization sequences for Triton-XDNA transform scripts.

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
