// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// Weighted RMS Norm transform for AIE2P.
// y = x * rsqrt(mean(x^2) + eps) * w
//
// No fuse_multi_op_linalg (creates L2 refs). CSE merges duplicate X
// extract_slices so linalg_promote shares one L1 X buffer across ops.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func0 { transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes } : !transform.any_op
    transform.apply_cse to %func0 : !transform.any_op
    %reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tr = transform.air.transpose_reduce %reduces : (!transform.any_op) -> !transform.any_op
    %func1a = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func1a { transform.apply_patterns.canonicalization } : !transform.any_op
    transform.apply_cse to %func1a : !transform.any_op
    %func1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %f = transform.air.fuse_elementwise_linalg %func1 : (!transform.any_op) -> !transform.any_op
    %fa = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %fa { transform.apply_patterns.canonicalization } : !transform.any_op
    transform.apply_cse to %fa : !transform.any_op

    %ag = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %sq, %out = transform.split_handle %ag : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %reduce = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fill = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    %ob, %nb = transform.structured.bufferize_to_allocation %out {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op
    %t, %fl = transform.structured.tile_using_forall %out tile_sizes [1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %f1, %fl1 = transform.structured.fuse_into_containing_op %reduce into %fl : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %f2, %fl2 = transform.structured.fuse_into_containing_op %sq into %fl1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %f3, %fl3 = transform.structured.fuse_into_containing_op %fill into %fl2 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fill dest → L1
    %fills3 = transform.structured.match ops{["linalg.fill"]} in %fl3 : (!transform.any_op) -> !transform.any_op
    %fill_buf, %fill_new = transform.structured.bufferize_to_allocation %fills3
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // CSE to merge duplicate X extract_slices from fuse_into_containing_op.
    // sq and out both slice from the same X tensor -- CSE merges them so
    // linalg_promote's promotedValueMap shares one L1 buffer.
    %func_cse = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_cse {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_cse : !transform.any_op

    // Bufferize
    %fop = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fb = transform.bufferization.one_shot_bufferize %fop : (!transform.any_op) -> !transform.any_op
    %f6 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f6 { transform.apply_patterns.canonicalization } : !transform.any_op
    transform.apply_cse to %f6 : !transform.any_op
    %lc = transform.structured.match ops{["linalg.copy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %mc = transform.structured.linalg_copy_to_memref %lc : (!transform.any_op) -> !transform.any_op
    %fu = transform.air.remove_uninitialized_copy %f6 : (!transform.any_op) -> (!transform.any_op)
    %fu2 = transform.air.eliminate_cascade_memcpy %fu : (!transform.any_op) -> (!transform.any_op)

    // linalg_promote for ALL operands (X, weight, output) with consistent
    // memory space attributes. No tensor-domain promotion needed.
    %forall_op = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %gens_f = transform.structured.match ops{["linalg.generic"]} in %forall_op : (!transform.any_op) -> !transform.any_op
    %reds_f = transform.structured.match ops{["linalg.reduce"]} in %forall_op : (!transform.any_op) -> !transform.any_op
    %all_linalg_f = transform.merge_handles %reds_f, %gens_f { deduplicate } : !transform.any_op
    %promoted = transform.air.linalg_promote %all_linalg_f {memory_space = "L1"} : (!transform.any_op) -> !transform.any_op

    // No post-promote canonicalization (it hoists allocs out of forall region)

    // Herd + DMA
    %fh = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %pa = transform.loop.forall_to_parallel %fh : (!transform.any_op) -> !transform.any_op
    %h = transform.air.par_to_herd %pa : (!transform.any_op) -> !transform.any_op

    // Override herd-internal allocs to L1
    %herd_ms = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %herd_ms_done = transform.air.override_memref_memory_space %herd_ms {memory_space = 2 : i32} : (!transform.any_op) -> !transform.any_op

    %h_fresh = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    // Skip copy_to_dma here -- driver's step 3 runs air-copy-to-dma pass
    // which handles self-copy elimination (PR #1390).

    // No post-herd cleanup (canonicalization hoists allocs across regions)

    // STOP HERE to check domination
    transform.yield
  }
}
