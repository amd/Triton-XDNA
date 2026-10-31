# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Which pointer arguments a compiled Triton kernel writes.

Read from the kernel's TTIR, where helper functions are inlined: each store
and atomic is traced back from its address through pointer-typed values
(`tt.addptr`'s base, `tt.splat`, `tt.broadcast`, ...) to the kernel's
arguments. An address that cannot be traced that way, such as a pointer
carried through a loop or chosen by a branch (the result of an op with
regions), makes the answer "every pointer argument".
"""

import os
import tempfile

#: ops whose operand 0 is the address written
_WRITES = ("tt.store", "tt.atomic_rmw", "tt.atomic_cas", "tt.descriptor_store")


def _is_ptr(value):
    return "!tt.ptr" in str(value.get_type())


def written_args(ttir, names):
    """The kernel arguments it writes through, by name, or None if some
    write's address could not be traced. `names` are the TTIR entry
    function's arguments: the kernel's non-constexpr parameters, in order."""
    from triton._C.libtriton import ir

    ctx = ir.context()
    ir.load_dialects(ctx)
    # parse_mlir_module reads a file
    with tempfile.NamedTemporaryFile("w", suffix=".ttir", delete=False) as f:
        f.write(ttir)
    try:
        mod = ir.parse_mlir_module(f.name, ctx)
    finally:
        os.unlink(f.name)
    fn = mod.get_function(mod.get_entry_func_name())
    if fn.get_num_args() != len(names):
        return None
    args = {fn.args(i).id(): names[i] for i in range(fn.get_num_args())}
    producer, writes = {}, []

    def visit(op):
        for i in range(op.get_num_results()):
            producer[op.get_result(i).id()] = op
        if op.get_name() in _WRITES:
            writes.append(op.get_operand(0))

    mod.walk(visit)
    out = set()
    for addr in writes:
        todo, seen = [addr], set()
        while todo:
            v = todo.pop()
            if v.id() in seen:
                continue
            seen.add(v.id())
            if v.id() in args:
                out.add(args[v.id()])
                continue
            op = producer.get(v.id())
            # a loop or branch result is whatever its regions yield
            if op is None or op.get_num_regions():
                return None
            ptrs = [
                op.get_operand(i)
                for i in range(op.get_num_operands())
                if _is_ptr(op.get_operand(i))
            ]
            if not ptrs:
                return None
            todo += ptrs
    return out
