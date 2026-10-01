// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Fold a trailing add back into a contraction's accumulator.

// Rewrite
//
//     %z   = linalg.fill ins(0) outs(tensor.empty)
//     %mm  = linalg.matmul ins(%a, %b) outs(%z)
//     %res = <generic: addf %c, %mm>
//
// into `linalg.matmul ins(%a, %b) outs(%c)`. This is how triton-shared lowers
// a `tt.dot` whose accumulator is not a zero tensor; `linalg.matmul` already
// computes `outs + A*B`, so the separate add costs a second full-size
// accumulator for nothing, and the schedules here expect the folded shape.
//
// Two steps rather than one: upstream's `FoldAddIntoDest` matches
// `linalg.elementwise`, and triton-shared leaves a `linalg.generic`, so the
// generic is specialized first. Keep `emit_category`; without it `specialize`
// looks for a named op and fails.
//
// The specialize is scoped to the contraction's consumer. Applying
// `linalg-morph-ops generic-to-category` to the function rewrites every
// eligible generic, and morphing back with `category-to-generic` also
// generalizes `linalg.fill` and `linalg.matmul`, which leaves the schedules
// with no matmul to match.
//
// A no-op on IR already in the folded form.
//
// The `fold_` prefixes matter: `_inject_transform_library` renames SSA values
// when it inlines a sequence, but not block labels, so two sequences sharing a
// `^bb0` would collide in one script.
transform.named_sequence @fold_add_into_matmul_init(
    %module: !transform.any_op {transform.readonly}) {
  %fold_f = transform.structured.match ops{["func.func"]} in %module
      : (!transform.any_op) -> !transform.any_op

  %fold_mm = transform.structured.match
      ops{["linalg.matmul", "linalg.batch_matmul"]} in %fold_f
      : (!transform.any_op) -> !transform.any_op

  // Suppressed per op: a contraction whose consumer is a store or a cast has
  // nothing to fold.
  %fold_raw = transform.get_consumers_of_result %fold_mm[0]
      : (!transform.any_op) -> !transform.any_op
  // The add reads the matmul in both `ins` and `outs`, so it comes back twice,
  // and `foreach` rejects a handle naming one payload op twice.
  %fold_use = transform.merge_handles deduplicate %fold_raw : !transform.any_op
  transform.foreach %fold_use : !transform.any_op {
  ^fold_bb1(%fold_one: !transform.any_op):
    transform.sequence %fold_one : !transform.any_op failures(suppress) {
    ^fold_bb2(%fold_x: !transform.any_op):
      transform.structured.specialize %fold_x emit_category = true
          : (!transform.any_op) -> !transform.any_op
    }
  }

  transform.apply_patterns to %fold_f {
    transform.apply_patterns.linalg.fold_add_into_dest
  } : !transform.any_op

  // Restore anything specialized above that the fold did not consume, since
  // the schedules match `linalg.generic`. `generalize` touches only the
  // handles given to it, leaving `linalg.fill` and `linalg.matmul` alone.
  %fold_left = transform.structured.match ops{["linalg.elementwise"]} in %fold_f
      : (!transform.any_op) -> !transform.any_op
  transform.foreach %fold_left : !transform.any_op {
  ^fold_bb3(%fold_e: !transform.any_op):
    transform.sequence %fold_e : !transform.any_op failures(suppress) {
    ^fold_bb4(%fold_y: !transform.any_op):
      transform.structured.generalize %fold_y
          : (!transform.any_op) -> !transform.any_op
    }
  }

  transform.yield
}
