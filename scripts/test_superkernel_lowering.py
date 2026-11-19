#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Rewrites of a superkernel's AIR module, on small modules written by hand:

  in place      a core's copy between a channel's buffer and an L1 buffer is
                removed only when it copies the whole of both, in order
  forwarding    a vector a core writes to its own buffer replaces only reads
                of the whole buffer, in the same order
  index casts   a constant cast to index becomes an index constant; a cast
                from index keeps its type
  herd depth    whether code is a herd's is kept per thread of compilation

Needs no NPU.

    python3 scripts/test_superkernel_lowering.py
"""

import sys
import threading

try:
    from air.ir import Context, Location, Module
    from triton.backends.amd_triton_npu import superkernel as S
    from triton.language.extra.air import superkernel as L
except ImportError as e:
    print(f"SKIP: {e}")
    sys.exit(77)

failures = 0


def check(ok, what):
    global failures
    print(f"  {'ok  ' if ok else 'FAIL'} {what}")
    failures += not ok


def herd_module(body):
    return f"""
module {{
  air.channel @c [1]
  func.func @f() {{
    %c1 = arith.constant 1 : index
    air.herd tile (%tx, %ty) in (%sx=%c1, %sy=%c1) {{
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %pad = arith.constant 0.0 : bf16
{body}
    }}
    return
  }}
}}
"""


def herd_of(module):
    return S._ops_named(module.operation, "air.herd")[0]


def alloc(module, n):
    """The `n`th memref.alloc of the herd."""
    return S._ops_named(herd_of(module), "memref.alloc")[n].results[0]


def in_place():
    whole = """
      %tmp = memref.alloc() : memref<32xbf16, 2>
      air.channel.get @c[] (%tmp[] [] []) : (memref<32xbf16, 2>)
      %home = memref.alloc() : memref<32xbf16, 2>
      %v = vector.transfer_read %tmp[%c0], %pad {in_bounds = [true]} : memref<32xbf16, 2>, vector<32xbf16>
      vector.transfer_write %v, %home[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, 2>
"""
    part = """
      %tmp = memref.alloc() : memref<256xbf16, 2>
      air.channel.get @c[] (%tmp[] [] []) : (memref<256xbf16, 2>)
      %home = memref.alloc() : memref<32xbf16, 2>
      %v = vector.transfer_read %tmp[%c32], %pad {in_bounds = [true]} : memref<256xbf16, 2>, vector<32xbf16>
      vector.transfer_write %v, %home[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, 2>
"""
    for body, want, what in (
        (whole, 1, "a whole copy is removed"),
        (part, 0, "a copy of 32 elements at 32 of 256 is kept"),
    ):
        m = Module.parse(herd_module(body))
        got = S._l1_copies(herd_of(m), [alloc(m, 1)])
        check(len(got) == want, f"in place: {what}")


def forwarding():
    body = """
      %buf = memref.alloc() : memref<8x8xf32, 2>
      %cst = arith.constant dense<1.0> : vector<8x8xf32>
      vector.transfer_write %cst, %buf[%c0, %c0] {{in_bounds = [true, true]}} : vector<8x8xf32>, memref<8x8xf32, 2>
      %r = vector.transfer_read %buf[%c0, %c0], %f0 {{in_bounds = [true, true]{map}}} : memref<8x8xf32, 2>, vector<8x8xf32>
      %out = memref.alloc() : memref<8x8xf32, 2>
      vector.transfer_write %r, %out[%c0, %c0] {{in_bounds = [true, true]}} : vector<8x8xf32>, memref<8x8xf32, 2>
"""
    for perm, want, what in (
        ("", True, "a read of the whole buffer takes the written vector"),
        (
            ", permutation_map = affine_map<(d0, d1) -> (d1, d0)>",
            False,
            "a transposed read keeps reading the buffer",
        ),
    ):
        text = herd_module(
            "      %f0 = arith.constant 0.0 : f32\n" + body.format(map=perm)
        )
        m = Module.parse(text)
        S._forward_private_vectors(m.operation)
        reads = S._ops_named(herd_of(m), "vector.transfer_read")
        check((not reads) == want, f"forwarding: {what}")


def index_casts():
    text = """
module {
  func.func @f() -> (index, i32) {
    %a = arith.constant 7 : i32
    %b = arith.constant 9 : index
    %ai = arith.index_cast %a : i32 to index
    %bi = arith.index_cast %b : index to i32
    return %ai, %bi : index, i32
  }
}
"""
    m = Module.parse(text)
    S._fold_constant_casts(m.operation, m.operation)
    casts = S._ops_named(m.operation, "arith.index_cast")
    check(
        len(casts) == 1 and str(casts[0].results[0].type) == "i32",
        "index casts: only the cast to index is folded",
    )
    check(m.operation.verify(), "index casts: the module verifies")


def herd_depth():
    entered, checked = threading.Event(), threading.Event()
    seen = []

    def in_herd():
        token = L._HERD.set(L._HERD.get() + 1)
        entered.set()
        checked.wait()
        L._HERD.reset(token)

    def outside():
        entered.wait()
        seen.append(L._in_herd())
        checked.set()

    threads = [threading.Thread(target=f) for f in (in_herd, outside)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check(seen == [False], "herd depth: another thread is not in the herd")


def main():
    with Context() as ctx, Location.unknown():
        ctx.allow_unregistered_dialects = True
        in_place()
        forwarding()
        index_casts()
    herd_depth()
    if failures:
        print(f"{failures} check(s) failed")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
