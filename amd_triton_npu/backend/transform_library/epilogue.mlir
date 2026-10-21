// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Steps for running a matmul's elementwise tail (a scale, an activation, a
// cast) on the epilogue herd, on the packed accumulator before it is
// unpacked. The tail may read the accumulator, scalars and constants only.
//
// A matmul schedule includes them in this order:
//   @epilogue_fuse_tail        before anything matches linalg.fill
//   @epilogue_unpack_after_tail after C is packed
//   @epilogue_unpack_to_l2     before the unpack is tiled for the herd
//   ... tile the unpack, fuse the `epilogue_op` generic into its loops ...
//   @epilogue_vectorize        after herd_vectorize

// Fuse the tail into one generic tagged `epilogue_op`. Scalar operands then
// sit in its body rather than in broadcast fills, so the accumulator init is
// the only linalg.fill left.
transform.named_sequence @epilogue_fuse_tail(
    %module: !transform.any_op {transform.readonly}) {
  %ep_f = transform.structured.match ops{["func.func"]} in %module
      : (!transform.any_op) -> !transform.any_op
  %ep_f_fused = transform.air.fuse_elementwise_linalg %ep_f
      : (!transform.any_op) -> !transform.any_op
  %ep_tail = transform.structured.match ops{["linalg.generic"]} in %module
      : (!transform.any_op) -> !transform.any_op
  transform.annotate %ep_tail "epilogue_op" : !transform.any_op
  transform.yield
}

// Move C's unpack after the tail, so the tail works on the packed accumulator
// and can be fused into the epilogue loops as the unpack's producer.
transform.named_sequence @epilogue_unpack_after_tail(
    %module: !transform.any_op {transform.readonly}) {
  %ep_dlp_f = transform.structured.match ops{["func.func"]} in %module
      : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %ep_dlp_f {
    transform.apply_patterns.linalg.data_layout_propagation
  } : !transform.any_op
  transform.apply_cse to %ep_dlp_f : !transform.any_op
  transform.yield
}

// The unpack now writes the tail's result, which has no L2 buffer yet. Match
// the unpack again afterwards: bufferize_to_allocation replaces it.
transform.named_sequence @epilogue_unpack_to_l2(
    %module: !transform.any_op {transform.readonly}) {
  %ep_unpack = transform.structured.match ops{["linalg.unpack"]} in %module
      : (!transform.any_op) -> !transform.any_op
  %ep_unpack_buf, %ep_unpack_new = transform.structured.bufferize_to_allocation
      %ep_unpack
      <{memory_space = 1, bufferize_destination_only, emit_dealloc}>
      : !transform.any_op
  transform.yield
}

// Flatten the tail's vectors to rank 1, then apply the same bf16 casts as the
// standalone elementwise schedules. Both are limited to the epilogue herd.
transform.named_sequence @epilogue_vectorize(
    %module: !transform.any_op {transform.readonly}) {
  %ep_herd = transform.structured.match ops{["air.herd"]}
      attributes{epilogue_herd} in %module
      : (!transform.any_op) -> !transform.any_op
  %ep_herd_lin = transform.air.linearize_vectors %ep_herd
      : (!transform.any_op) -> !transform.any_op
  transform.include @cast_bf16_only_ops failures(propagate) (%ep_herd_lin)
      : (!transform.any_op) -> ()
  transform.yield
}
