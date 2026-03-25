// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// AIR herd mapping and vectorization sequences.

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
