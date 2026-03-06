// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// RMS Norm: absolute minimum transform. Requires mlir-air >= 2ed2d26.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func0 {
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
    } : !transform.any_op
    transform.apply_cse to %func0 : !transform.any_op

    // Promote reduce init tensor to L1
    %at = transform.structured.match ops{["bufferization.alloc_tensor"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %at_buf, %at_new = transform.structured.bufferize_to_allocation %at
        {memory_space = 1, emit_dealloc} : !transform.any_op

    %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func1 { transform.apply_patterns.canonicalization } : !transform.any_op
    transform.apply_cse to %func1 : !transform.any_op

    transform.yield
  }
}
