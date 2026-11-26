#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Rewrites of a superkernel's AIR module, on small modules written by hand:

  forward copies  a block loaded from a channel and stored unchanged into
                  the kernel's own buffer is one copy
  forwarding      a vector a core writes to a buffer replaces reads of it in
                  the same order; a buffer sent on a channel stays
  placement       herds without `at` go where AIR's placer puts them, around
                  the pinned ones; herds that do not fit are an error
  index casts     a constant cast to index becomes an index constant; a cast
                  from index keeps its type
  herd depth      whether code is a herd's is kept per thread of compilation
  tool args       air.backend's opt_args and llc_args reach aircc as one -Xopt
                  or -Xllc each, commas kept
  call buffers    a buffer only a call reads stays, with the call, through the
                  herds' rewrites; one nothing reads goes

Needs no NPU.

    python3 scripts/test_superkernel_lowering.py
"""

import sys
import threading

try:
    from air.ir import Context, IntegerAttr, Location, Module
    from triton.backends.amd_triton_npu import superkernel as S
    from triton.backends.amd_triton_npu import superkernel_jit as SJ
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


def forward_copies():
    # A block loaded from a channel's address and stored into the kernel's own
    # buffer unchanged, as triton-shared lowers it.
    text = """
func.func @k(%chan: memref<*xbf16>, %own: memref<*xbf16>) {
  %c0 = arith.constant 0 : index
  %src = memref.reinterpret_cast %chan to offset: [%c0], sizes: [256], strides: [1] : memref<*xbf16> to memref<256xbf16, strided<[1], offset: ?>>
  %dst = memref.reinterpret_cast %own to offset: [0], sizes: [256], strides: [1] : memref<*xbf16> to memref<256xbf16, strided<[1]>>
  %buf = memref.alloc() : memref<256xbf16>
  memref.copy %src, %buf : memref<256xbf16, strided<[1], offset: ?>> to memref<256xbf16>
  %t = bufferization.to_tensor %buf restrict writable : memref<256xbf16> to tensor<256xbf16>
  bufferization.materialize_in_destination %t in writable %dst : (tensor<256xbf16>, memref<256xbf16, strided<[1]>>) -> ()
  return
}
"""
    own = S._Local(256, "bf16", "air.local.0")
    got = S._forward_copies(text, ["chan", own])
    check(
        got.count("memref.copy") == 1
        and "materialize_in_destination" not in got
        and "memref.alloc" not in got,
        "forward copies: a channel load stored into the kernel's buffer is one copy",
    )
    check(
        S._forward_copies(text, ["chan", "ddr"]) == text,
        "forward copies: a kernel without buffers of its own is left as it is",
    )


def forwarding():
    from air.compiler.util import run_transform

    def herds(body):
        m = Module.parse(herd_module("      %f0 = arith.constant 0.0 : f32\n" + body))
        run_transform(Module.parse(S._HERDS), m)
        return m

    write = """
      %buf = memref.alloc() : memref<8x8xf32, 2>
      %cst = arith.constant dense<1.0> : vector<8x8xf32>
      vector.transfer_write %cst, %buf[%c0, %c0] {{in_bounds = [true, true]}} : vector<8x8xf32>, memref<8x8xf32, 2>
"""
    read = """
      %r = vector.transfer_read %buf[%c0, %c0], %f0 {{in_bounds = [true, true]{map}}} : memref<8x8xf32, 2>, vector<8x8xf32>
      %out = memref.alloc() : memref<8x8xf32, 2>
      vector.transfer_write %r, %out[%c0, %c0] {{in_bounds = [true, true]}} : vector<8x8xf32>, memref<8x8xf32, 2>
      air.channel.put @c[%c0] (%out[] [] []) : (memref<8x8xf32, 2>)
"""
    for perm, want, what in (
        ("", True, "a read of the whole buffer takes the written vector"),
        (
            ", permutation_map = affine_map<(d0, d1) -> (d1, d0)>",
            False,
            "a transposed read keeps reading the buffer",
        ),
    ):
        m = herds((write + read).format(map=perm))
        h = herd_of(m)
        reads = S._ops_named(h, "vector.transfer_read")
        check((not reads) == want, f"forwarding: {what}")
        check(
            len(S._ops_named(h, "air.channel.put")) == 1,
            f"forwarding: the sent buffer stays ({what})",
        )
    send = "      air.channel.put @c[%c0] (%buf[] [] []) : (memref<8x8xf32, 2>)\n"
    m = herds(write.format() + send)
    h = herd_of(m)
    check(
        len(S._ops_named(h, "vector.transfer_write")) == 1
        and len(S._ops_named(h, "air.channel.put")) == 1,
        "forwarding: a buffer the core fills and sends stays, with its write",
    )


def placement():
    def seg_module(herds):
        body = "".join(f"""
        air.herd @{name} tile (%tx, %ty) in (%sx=%c{w}, %sy=%c1){attrs} {{
        }}""" for name, w, attrs in herds)
        return f"""
module {{
  func.func @f() {{
    air.launch () in () {{
      air.segment @seg attributes {{x_loc = 0 : i64, y_loc = 2 : i64}} {{
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c8 = arith.constant 8 : index{body}
      }}
    }}
    return
  }}
}}
"""

    def places(m):
        return {
            str(h.attributes["sym_name"]).strip('"'): (
                IntegerAttr(h.attributes["x_loc"]).value,
                IntegerAttr(h.attributes["y_loc"]).value,
            )
            for h in S._ops_named(m.operation, "air.herd")
        }

    pinned = " attributes {x_loc = 0 : i64, y_loc = 2 : i64}"
    m = Module.parse(seg_module([("p", 2, pinned), ("q", 2, "")]))
    seg = S._ops_named(m.operation, "air.segment")[0]
    S._place_herds(m, seg, "npu2")
    got = places(m)
    check(
        got["p"] == (0, 2)
        and got["q"][1] in range(2, 6)
        and got["q"][0] + 2 <= 8
        and got["q"] not in ((0, 2), (1, 2)),
        f"placement: a herd without `at` goes beside the pinned one ({got})",
    )
    m = Module.parse(seg_module([("p", 2, pinned)]))
    seg = S._ops_named(m.operation, "air.segment")[0]
    S._place_herds(m, seg, "npu1")
    check(places(m) == {"p": (0, 2)}, "placement: pinned herds need no array")
    m = Module.parse(seg_module([(f"h{i}", 8, "") for i in range(5)]))
    seg = S._ops_named(m.operation, "air.segment")[0]
    try:
        S._place_herds(m, seg, "npu2")
        failed = False
    except Exception:
        failed = True
    check(failed, "placement: herds that do not fit are an error")


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


def tool_args():
    gep = (
        "--passes-ep-optimizer-last=function(separate-const-offset-from-gep,early-cse)"
    )
    opts = {
        k: L._backend_text(k, v)
        for k, v in dict(
            opt_args=[gep, "-b"],
            llc_args="--c",
            stack_size=4096,
            channel_multiplexing=["L1", "L2"],
        ).items()
    }
    check(
        opts["channel_multiplexing"] == "L1,L2",
        "tool args: other lists still joined by commas",
    )
    check(
        L._backend_text("opt_args", None) is None,
        "tool args: an option not given has no text",
    )
    args = SJ._aircc_options(opts)["extra_args"]
    check(
        [a for a in args if a.startswith("-X")]
        == [f"-Xopt={gep}", "-Xopt=-b", "-Xllc=--c"],
        "tool args: one -Xopt/-Xllc each, in order, commas kept",
    )


def call_buffers():
    from air.compiler.util import run_transform

    def module(body):
        return Module.parse(
            """
module {
  func.func private @use(memref<8xf32, 2>)
  func.func private @use2(memref<8xf32, 2>, memref<4xf32, strided<[1], offset: 2>, 2>)
  func.func private @get(memref<8xf32, 2>) -> f32
  func.func @f() {
    %c1 = arith.constant 1 : index
    air.herd tile (%tx, %ty) in (%sx=%c1, %sy=%c1) {
      %c0 = arith.constant 0 : index
      %x = arith.constant 1.0 : f32
      %buf = memref.alloc() : memref<8xf32, 2>
      memref.store %x, %buf[%c0] : memref<8xf32, 2>
"""
            + body
            + """
    }
    return
  }
}
"""
        )

    def herds(m):
        S._hide_call_buffers(m.operation)
        run_transform(Module.parse(S._HERDS), m)
        S._unhide_call_buffers(m.operation)
        h = herd_of(m)
        casts = S._ops_named(h, "builtin.unrealized_conversion_cast")
        return h, not casts and m.operation.verify()

    def owners(op):
        return [v.owner.name for v in op.operands]

    m = module("""
      func.call @use(%buf) : (memref<8xf32, 2>) -> ()
      %dead = memref.alloc() : memref<8xf32, 2>
      memref.store %x, %dead[%c0] : memref<8xf32, 2>
""")
    h, clean = herds(m)
    calls = S._ops_named(h, "func.call")
    check(
        len(calls) == 1
        and owners(calls[0]) == ["memref.alloc"]
        and len(S._ops_named(h, "memref.alloc")) == 1,
        "call buffers: a buffer only a call reads stays, and the call takes it;"
        " one nothing reads goes",
    )
    check(clean, "call buffers: no cast is left, and the module verifies")

    # The buffer twice in one call, once through a view.
    m = module("""
      %v = memref.subview %buf[2] [4] [1] : memref<8xf32, 2> to memref<4xf32, strided<[1], offset: 2>, 2>
      func.call @use2(%buf, %v) : (memref<8xf32, 2>, memref<4xf32, strided<[1], offset: 2>, 2>) -> ()
""")
    h, clean = herds(m)
    calls = S._ops_named(h, "func.call")
    check(
        len(calls) == 1 and owners(calls[0]) == ["memref.alloc", "memref.subview"],
        "call buffers: a buffer a call takes twice, once through a view, stays",
    )
    check(clean, "call buffers: no cast is left after a call taking two")

    # A call with a result is one the erasure keeps already: left as it is.
    m = module("""
      %r = func.call @get(%buf) : (memref<8xf32, 2>) -> f32
      memref.store %r, %buf[%c0] : memref<8xf32, 2>
""")
    S._hide_call_buffers(m.operation)
    call = S._ops_named(herd_of(m), "func.call")[0]
    check(
        owners(call) == ["memref.alloc"],
        "call buffers: a call with a result is not given casts",
    )

    # Casts not made here are not undone.
    m = module("""
      %c = builtin.unrealized_conversion_cast %buf : memref<8xf32, 2> to memref<8xf32, 2>
      func.call @use(%c) : (memref<8xf32, 2>) -> ()
""")
    S._hide_call_buffers(m.operation)
    S._unhide_call_buffers(m.operation)
    call = S._ops_named(herd_of(m), "func.call")[0]
    check(
        owners(call) == ["builtin.unrealized_conversion_cast"],
        "call buffers: a herd's own cast stays",
    )


def main():
    with Context() as ctx, Location.unknown():
        ctx.allow_unregistered_dialects = True
        forward_copies()
        forwarding()
        placement()
        index_casts()
        call_buffers()
    herd_depth()
    tool_args()
    if failures:
        print(f"{failures} check(s) failed")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
