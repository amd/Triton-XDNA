#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Checks for the `@fold_add_into_matmul_init` transform library sequence.

triton-shared lowers a `tt.dot` whose accumulator is not a zero tensor to a
contraction into a fresh zero fill plus a separate add. The sequence folds
that back into the contraction's `outs`, which is the shape the schedules in
this backend match.

Every failure here is silent at runtime: a fold that stops firing produces
slower code, and a fold that fires too eagerly produces wrong code or leaves
the schedules with nothing to match. Nothing else in the suite exercises it,
because no kernel in the tree passes an accumulator to `tl.dot`.

Runs `air-opt` on fixtures; no NPU, no weights. Exits 77 when air is not
installed.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))

# Captured ttsharedir for `tl.store(Out, tl.dot(a, b) + c)`. Triton's own
# `triton-combine` folds the add into `tt.dot`, so the accumulator is live by
# the time triton-shared sees it -- the same thing `acc += tl.dot(a, b)` in a
# K loop becomes.
_PROLOGUE = """\
  func.func @mm_plus_c(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>,
                       %arg2: memref<*xf32>, %arg3: memref<*xf32>) {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<*xbf16> to memref<64x64xbf16, strided<[64, 1]>>
    %alloc = memref.alloc() : memref<64x64xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<64x64xbf16, strided<[64, 1]>> to memref<64x64xbf16>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<64x64xbf16> to tensor<64x64xbf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<*xbf16> to memref<64x64xbf16, strided<[64, 1]>>
    %alloc_1 = memref.alloc() : memref<64x64xbf16>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<64x64xbf16, strided<[64, 1]>> to memref<64x64xbf16>
    %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<64x64xbf16> to tensor<64x64xbf16>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<*xf32> to memref<64x64xf32, strided<[64, 1]>>
    %alloc_3 = memref.alloc() : memref<64x64xf32>
    memref.copy %reinterpret_cast_2, %alloc_3 : memref<64x64xf32, strided<[64, 1]>> to memref<64x64xf32>
    %2 = bufferization.to_tensor %alloc_3 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<*xf32> to memref<64x64xf32, strided<[64, 1]>>
"""

_EPILOGUE = """\
    bufferization.materialize_in_destination %3 in writable %reinterpret_cast_4 : (tensor<64x64xf32>, memref<64x64xf32, strided<[64, 1]>>) -> ()
    return
  }
}
"""

FOLDED = (
    "module {\n"
    + _PROLOGUE
    + "    %3 = linalg.matmul ins(%0, %1 : tensor<64x64xbf16>, tensor<64x64xbf16>)"
    " outs(%2 : tensor<64x64xf32>) -> tensor<64x64xf32>\n" + _EPILOGUE
)

UNFOLDED = "#map = affine_map<(d0, d1) -> (d0, d1)>\n" "module {\n" + _PROLOGUE + """\
    %cst = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : tensor<64x64xf32>
    %z = linalg.fill ins(%cst : f32) outs(%e : tensor<64x64xf32>) -> tensor<64x64xf32>
    %mm = linalg.matmul ins(%0, %1 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%z : tensor<64x64xf32>) -> tensor<64x64xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %mm : tensor<64x64xf32>, tensor<64x64xf32>) outs(%2 : tensor<64x64xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %s = arith.addf %x, %y : f32
      linalg.yield %s : f32
    } -> tensor<64x64xf32>
""" + _EPILOGUE

# tl.dot on a 3-D block lowers to linalg.batch_matmul.
BATCH_UNFOLDED = """\
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @bmm(%a: tensor<2x64x64xbf16>, %b: tensor<2x64x64xbf16>,
               %acc: tensor<2x64x64xf32>) -> tensor<2x64x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<2x64x64xf32>
  %z = linalg.fill ins(%cst : f32) outs(%e : tensor<2x64x64xf32>) -> tensor<2x64x64xf32>
  %mm = linalg.batch_matmul ins(%a, %b : tensor<2x64x64xbf16>, tensor<2x64x64xbf16>) outs(%z : tensor<2x64x64xf32>) -> tensor<2x64x64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%acc, %mm : tensor<2x64x64xf32>, tensor<2x64x64xf32>) outs(%acc : tensor<2x64x64xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %s = arith.addf %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<2x64x64xf32>
  return %r : tensor<2x64x64xf32>
}
"""

BATCH_FOLDED = """\
func.func @bmm(%a: tensor<2x64x64xbf16>, %b: tensor<2x64x64xbf16>,
               %acc: tensor<2x64x64xf32>) -> tensor<2x64x64xf32> {
  %mm = linalg.batch_matmul ins(%a, %b : tensor<2x64x64xbf16>, tensor<2x64x64xbf16>) outs(%acc : tensor<2x64x64xf32>) -> tensor<2x64x64xf32>
  return %mm : tensor<2x64x64xf32>
}
"""

# The i8 path, where the add is arith.addi and the zero is an integer.
INT_UNFOLDED = """\
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @mmi(%a: tensor<64x64xi8>, %b: tensor<64x64xi8>,
               %acc: tensor<64x64xi32>) -> tensor<64x64xi32> {
  %c0 = arith.constant 0 : i32
  %e = tensor.empty() : tensor<64x64xi32>
  %z = linalg.fill ins(%c0 : i32) outs(%e : tensor<64x64xi32>) -> tensor<64x64xi32>
  %mm = linalg.matmul ins(%a, %b : tensor<64x64xi8>, tensor<64x64xi8>) outs(%z : tensor<64x64xi32>) -> tensor<64x64xi32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%acc, %mm : tensor<64x64xi32>, tensor<64x64xi32>) outs(%acc : tensor<64x64xi32>) {
  ^bb0(%x: i32, %y: i32, %o: i32):
    %s = arith.addi %x, %y : i32
    linalg.yield %s : i32
  } -> tensor<64x64xi32>
  return %r : tensor<64x64xi32>
}
"""

INT_FOLDED = """\
func.func @mmi(%a: tensor<64x64xi8>, %b: tensor<64x64xi8>,
               %acc: tensor<64x64xi32>) -> tensor<64x64xi32> {
  %mm = linalg.matmul ins(%a, %b : tensor<64x64xi8>, tensor<64x64xi8>) outs(%acc : tensor<64x64xi32>) -> tensor<64x64xi32>
  return %mm : tensor<64x64xi32>
}
"""

# A contraction into a destination that is not the additive identity. Folding
# the add in here would drop %init from the sum.
NONZERO_DEST = """\
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @keep(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>,
                %init: tensor<64x64xf32>, %acc: tensor<64x64xf32>)
    -> tensor<64x64xf32> {
  %mm = linalg.matmul ins(%a, %b : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%acc, %mm : tensor<64x64xf32>, tensor<64x64xf32>) outs(%acc : tensor<64x64xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %s = arith.addf %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<64x64xf32>
  return %r : tensor<64x64xf32>
}
"""

# An epilogue whose add does not read the contraction. The sequence specializes
# the contraction's consumer to reach the fold, so both generics have to come
# back as generics -- that is the form the schedules match.
EPILOGUE = """\
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @epi(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>,
               %s: tensor<64x64xf32>, %t: tensor<64x64xf32>)
    -> tensor<64x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<64x64xf32>
  %z = linalg.fill ins(%cst : f32) outs(%e : tensor<64x64xf32>) -> tensor<64x64xf32>
  %mm = linalg.matmul ins(%a, %b : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%z : tensor<64x64xf32>) -> tensor<64x64xf32>
  %scaled = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%mm, %s : tensor<64x64xf32>, tensor<64x64xf32>) outs(%mm : tensor<64x64xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %p = arith.mulf %x, %y : f32
    linalg.yield %p : f32
  } -> tensor<64x64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%scaled, %t : tensor<64x64xf32>, tensor<64x64xf32>) outs(%scaled : tensor<64x64xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %q = arith.addf %x, %y : f32
    linalg.yield %q : f32
  } -> tensor<64x64xf32>
  return %r : tensor<64x64xf32>
}
"""

CALLER = """\
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    transform.include @fold_add_into_matmul_init failures(propagate) (%arg1)
      : (!transform.any_op) -> ()
    transform.yield
  }
}
"""

#: (name, input, expected, why it matters if it breaks)
CASES = [
    (
        "accumulator folds",
        UNFOLDED,
        FOLDED,
        "a K-looping kernel keeps a second full-size accumulator",
    ),
    (
        "already folded is left alone",
        FOLDED,
        FOLDED,
        "every kernel we dispatch today goes through here",
    ),
    (
        "batch_matmul folds",
        BATCH_UNFOLDED,
        BATCH_FOLDED,
        "a 3-D tl.dot keeps a second accumulator",
    ),
    (
        "integer folds",
        INT_UNFOLDED,
        INT_FOLDED,
        "the i8 path keeps a second accumulator",
    ),
    (
        "non-zero contraction dest is left alone",
        NONZERO_DEST,
        NONZERO_DEST,
        "WRONG RESULTS: the contraction's own init would be dropped",
    ),
    (
        "epilogue generics survive as generics",
        EPILOGUE,
        EPILOGUE,
        "the schedules match linalg.generic and would find none",
    ),
]


def skip(msg):
    print(f"SKIP: {msg}")
    sys.exit(77)


def air_opt():
    exe = shutil.which("air-opt")
    if exe:
        return exe
    # Found, not imported, for the reason `_load` gives.
    spec = importlib.util.find_spec("mlir_air")
    if spec is None or not spec.submodule_search_locations:
        return None
    cand = os.path.join(list(spec.submodule_search_locations)[0], "bin", "air-opt")
    return cand if os.path.isfile(cand) else None


def _load(name):
    """Load a stdlib-only module of the source tree's backend off disk.

    Not `import amd_triton_npu.backend...`: the package loads air's LLVM, and
    in CI's environment a second LLVM in the process aborts it before any check
    runs (`LLVM ERROR: Option 'fast' already exists!`). This test needs no
    native code, so it loads none.
    """
    path = os.path.join(
        os.path.dirname(_HERE), "amd_triton_npu", "backend", name + ".py"
    )
    spec = importlib.util.spec_from_file_location(f"_fold_add_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize(ir):
    """Drop locations and blank lines, so the comparison is about structure."""
    out = []
    for line in ir.splitlines():
        if line.startswith("#loc"):
            continue
        line = re.sub(r" loc\(#loc\d*\)", "", line).rstrip()
        if line:
            out.append(line)
    return "\n".join(out)


def run(exe, ir, script=None):
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in.mlir")
        with open(src, "w") as f:
            f.write(ir)
        cmd = [exe, src]
        if script is not None:
            lib = os.path.join(d, "lib.mlir")
            with open(lib, "w") as f:
                f.write(script)
            cmd += [
                f"--transform-preload-library=transform-library-paths={lib}",
                "--transform-interpreter=entry-point=__transform_main",
            ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"air-opt failed:\n{p.stderr[:2000]}")
        return normalize(p.stdout)


def main():
    exe = air_opt()
    if exe is None:
        skip("air-opt not found (needs mlir_air installed or on PATH)")

    _inject_transform_library = _load("transform_inject")._inject_transform_library
    generate_matmul_transform = _load("matmul_transform").generate_matmul_transform

    script = _inject_transform_library(CALLER)
    if "fold_add_into_dest" not in script:
        print(
            "FAIL: transform.include @fold_add_into_matmul_init did not resolve.\n"
            "      transform_library/fold_add.mlir is missing, or the inliner in\n"
            "      transform_inject._inject_transform_library no longer matches\n"
            "      its shape."
        )
        return 1

    failures = 0
    for name, src, want_src, stake in CASES:
        want = run(exe, want_src)
        try:
            got = run(exe, src, script)
        except RuntimeError as e:
            print(f"FAIL: {name} -- {e}\n      {stake}")
            failures += 1
            continue
        if got == want:
            print(f"  ok   {name}")
        else:
            print(f"FAIL: {name}\n      {stake}\n--- want\n{want}\n--- got\n{got}")
            failures += 1

    if "@fold_add_into_matmul_init" not in generate_matmul_transform(l1_m=64, l1_n=64):
        print(
            "FAIL: generate_matmul_transform no longer includes the fold, so a "
            "kernel needing it would reach PHASE 1 unfolded."
        )
        failures += 1
    else:
        print("  ok   generated matmul schedule includes it")

    if failures:
        print(f"{failures} failed")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
