# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""A core kernel's ``tl.extra.air.aie`` code after its schedule: its
functions made, as a C kernel's are. Only a kernel ``aie_ops.lower`` lowered
is touched; it marks the module (``MARK``) and the loops to outline.

The intrinsics the ops call, declared and called as llvm ops so that the
schedule's matches of func.func see the kernel alone, become func ops, and
each herd sets the rounding mode first when any op sets it. ``_POST`` then
outlines each loop or region marked ``aie.fnloop`` (``function``,
``begin_function``), one function for identical ones, and in each
``aie.k_loop`` carries what every trip reads and writes back before
unrolling it by 4.

Each outlined function is named as its ``function`` or ``begin_function``
says, takes the constants its calls all pass as its own, starts a stream it
is only handed itself, and is never inlined. With ``RESTRICT``, a buffer
argument that no other argument reaches at any call is ``llvm.noalias``.
Functions whose calls return values are left as they are.
"""

_POST = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %hl = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %hl1 = transform.air.linearize_vectors %hl arch = "aie2p" f32_lanes = 64 contract : (!transform.any_op) -> !transform.any_op
    %h0 = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %h0 {
      transform.apply_patterns.memref.fold_memref_alias_ops
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %f2 = transform.apply_registered_pass "affine-expand-index-ops-as-affine" to %f : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f2 { transform.apply_patterns.canonicalization } : !transform.any_op
    %h = transform.structured.match ops{["air.herd"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %ks = transform.structured.match ops{["scf.for"]} attributes{aie.k_loop} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.foreach %ks : !transform.any_op {
    ^bb0(%k: !transform.any_op):
      %k1 = transform.air.hoist_loop_invariant_transfers %h, %k : (!transform.any_op, !transform.any_op) -> !transform.any_op
    }
    %ks2 = transform.structured.match ops{["scf.for"]} attributes{aie.k_loop} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.foreach %ks2 : !transform.any_op {
    ^bb0(%k1: !transform.any_op):
      %k2 = transform.air.flatten_for_iter_args %k1 : (!transform.any_op) -> !transform.any_op
      %k3 = transform.air.hoist_vector_transfer_pointers %k2 : (!transform.any_op) -> !transform.any_op
      %unroll4 = transform.param.constant #llvm.loop_annotation<unroll = <count = 4 : i32>, mustProgress = true> -> !transform.any_param
      transform.annotate %k3 "loop_annotation" = %unroll4 : !transform.any_op, !transform.any_param
    }
    %lin = transform.structured.match ops{["linalg.generic", "linalg.copy", "linalg.transpose", "linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %lin_l = transform.structured.convert_to_loops %lin : (!transform.any_op) -> !transform.any_op
    %outf = transform.structured.match ops{["scf.for", "scf.execute_region"]} attributes{aie.fnloop} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.foreach %outf : !transform.any_op {
    ^bb0(%l: !transform.any_op):
      %fn, %call = transform.loop.outline %l func_name = "aie_fn" : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      %priv = transform.param.constant "private" -> !transform.any_param
      transform.annotate %fn "sym_visibility" = %priv : !transform.any_op, !transform.any_param
    }
    %mod = transform.structured.match ops{["builtin.module"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %m_sccp = transform.apply_registered_pass "sccp" to %mod : (!transform.any_op) -> !transform.any_op
    %m_rdv = transform.apply_registered_pass "remove-dead-values" to %m_sccp : (!transform.any_op) -> !transform.any_op
    %m_dfe = transform.apply_registered_pass "duplicate-function-elimination" to %m_rdv : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
"""

#: the module attribute of a kernel whose aie ops `aie_ops.lower` lowered
MARK = "aie.code"
_INTRINSIC = "llvm.aie2p."
_SET_CTRL = "llvm.aie2p.set.ctrl.reg"
_CONSTANTS = (
    "arith.constant",
    "llvm.mlir.constant",
    "llvm.mlir.poison",
    "llvm.mlir.undef",
    "ub.poison",
)
_VIEWS = (
    "memref.reinterpret_cast",
    "memref.subview",
    "memref.cast",
    "memref.collapse_shape",
    "memref.expand_shape",
    "memref.view",
    "aievec.fifo_start",
)


def finish(module):
    """`module` (an air.ir Module, after its schedule) with its aie code's
    functions made, if `aie_ops.lower` marked it (`MARK`); unchanged
    otherwise, whatever aievec ops its own schedule made."""
    from air.compiler.util import run_transform
    from air.ir import Location, Module

    if MARK not in module.operation.attributes:
        return
    del module.operation.attributes[MARK]
    with module.context, Location.unknown(module.context):
        _intrinsics_as_func(module)
        _rounding_at_herd_start(module)
        run_transform(Module.parse(_POST), module)
        restrict = _name_functions(module)
        _unwrap_calls(module)
        _unwrap_bodies(module)
        _sink_constant_arguments(module)
        _sink_stream_starts(module)
        _restrict(module, restrict)


def _walk(op, out=None):
    out = [] if out is None else out
    out.append(op)
    for r in op.regions:
        for b in r.blocks:
            for o in b.operations:
                _walk(o.operation, out)
    return out


def _all_ops(module):
    out = []
    for o in module.body.operations:
        _walk(o.operation, out)
    return out


def _callee(call):
    return call.attributes["callee"].value


def _def(v):
    from air.ir import Block

    o = v.owner
    return None if isinstance(o, Block) else o.operation


def _functions(module):
    return [
        o.operation for o in module.body.operations if o.operation.name == "func.func"
    ]


def _name(f):
    return f.attributes["sym_name"].value


def _has_body(f):
    return len(f.regions[0].blocks) > 0


def _calls_of(module, name):
    return [
        op for op in _all_ops(module) if op.name == "func.call" and _callee(op) == name
    ]


def _intrinsics_as_func(module):
    """llvm.func declarations and llvm.call ops of the intrinsics as
    func.func and func.call."""
    from air.dialects import func
    from air.ir import FunctionType, InsertionPoint

    decls = [
        o.operation
        for o in module.body.operations
        if o.operation.name == "llvm.func" and _name(o.operation).startswith(_INTRINSIC)
    ]
    calls = [
        op
        for op in _all_ops(module)
        if op.name == "llvm.call" and _callee(op).startswith(_INTRINSIC)
    ]
    types = {}
    for c in calls:
        types.setdefault(
            _callee(c), ([v.type for v in c.operands], [r.type for r in c.results])
        )
    for d in decls:
        sig = types.get(_name(d))
        if sig is not None:
            func.FuncOp(
                _name(d),
                FunctionType.get(*sig),
                visibility="private",
                ip=InsertionPoint(d),
            )
        d.erase()
    for c in calls:
        new = func.CallOp(
            [r.type for r in c.results],
            _callee(c),
            list(c.operands),
            ip=InsertionPoint(c),
        )
        for a, b in zip(c.results, new.results):
            a.replace_all_uses_with(b)
        c.erase()


def _rounding_at_herd_start(module):
    """set.ctrl.reg(1, 12), conv_even rounding, first in each herd when any
    op sets it: airrt-to-npu's cleanup of host code may erase the call where
    a function the core calls makes it."""
    from air.dialects import arith, func
    from air.ir import InsertionPoint, IntegerType

    ops = _all_ops(module)
    if not any(op.name == "func.call" and _callee(op) == _SET_CTRL for op in ops):
        return
    i32 = IntegerType.get_signless(32)
    for h in (op for op in ops if op.name == "air.herd"):
        ip = InsertionPoint.at_block_begin(h.regions[0].blocks[0])
        mode = arith.ConstantOp(i32, 1, ip=ip)
        value = arith.ConstantOp(i32, 12, ip=ip)
        func.CallOp([], _SET_CTRL, [mode.result, value.result], ip=ip)


def _name_functions(module):
    """The outlined functions (aie_fn, aie_fn_0, ...) named as the loop
    each was made of says, and every private function made llvm.no_inline.
    Returns the names of the ones marked restrict."""
    from air.ir import StringAttr, SymbolTable, UnitAttr

    restrict = set()
    taken = {_name(f) for f in _functions(module)}
    for f in _functions(module):
        name = _name(f)
        if not name.startswith("aie_fn") or not _has_body(f):
            continue
        marked = [op for op in _walk(f) if "aie.fn" in op.attributes]
        assert marked, f"{name}: no aie.fn loop"
        base = marked[0].attributes["aie.fn"].value
        # (functions that differ keep a name each)
        new, k = base, 0
        while new in taken:
            new, k = f"{base}_{k}", k + 1
        taken.add(new)
        if "aie.restrict" in marked[0].attributes:
            restrict.add(new)
        SymbolTable.replace_all_symbol_uses(name, new, module.operation)
        SymbolTable.set_symbol_name(f, new)
    for f in _functions(module):
        private = (
            "sym_visibility" in f.attributes
            and f.attributes["sym_visibility"].value == "private"
        )
        if (
            private
            and not _name(f).startswith(_INTRINSIC)
            and _has_body(f)
            and "no_inline" not in f.attributes
        ):
            f.attributes["llvm.no_inline"] = UnitAttr.get()
    return restrict


def _unwrap_calls(module):
    """An scf.execute_region the outlining left around a call (no results,
    not no_inline, only the call and a yield): the call."""
    for op in _all_ops(module):
        if (
            op.name != "scf.execute_region"
            or len(op.results)
            or "no_inline" in op.attributes
        ):
            continue
        body = [o.operation.name for o in op.regions[0].blocks[0].operations]
        if body == ["func.call", "scf.yield"]:
            body = list(op.regions[0].blocks[0].operations)
            body[0].operation.move_before(op)
            op.erase()


def _unwrap_bodies(module):
    """A function whose body holds a begin_function region (an
    scf.execute_region no_inline marked aie.fnloop): the region's ops in
    its place."""
    for f in _functions(module):
        if not _has_body(f):
            continue
        for o in list(f.regions[0].blocks[0].operations):
            region = o.operation
            if (
                region.name != "scf.execute_region"
                or "aie.fnloop" not in region.attributes
            ):
                continue
            inner = list(region.regions[0].blocks[0].operations)
            for x in inner[:-1]:
                x.operation.move_before(region)
            region.erase()


def _constant_key(op):
    if op is None or op.name not in _CONSTANTS:
        return None
    attrs = tuple(sorted((n, str(op.attributes[n])) for n in op.attributes))
    return op.name, attrs, tuple(str(r.type) for r in op.results)


def _drop_arguments(f, calls, drop):
    """f and its calls without the arguments `drop`."""
    from air.dialects import func
    from air.ir import FunctionType, InsertionPoint, TypeAttr

    block = f.regions[0].blocks[0]
    ty = f.attributes["function_type"].value
    keep = [i for i in range(len(block.arguments)) if i not in drop]
    ins = [block.arguments[i].type for i in keep]
    for i in sorted(drop, reverse=True):
        block.erase_argument(i)
    f.attributes["function_type"] = TypeAttr.get(
        FunctionType.get(ins, list(ty.results))
    )
    if "arg_attrs" in f.attributes:
        from air.ir import ArrayAttr

        aa = f.attributes["arg_attrs"]
        f.attributes["arg_attrs"] = ArrayAttr.get([aa[i] for i in keep])
    for c in calls:
        func.CallOp([], _callee(c), [c.operands[i] for i in keep], ip=InsertionPoint(c))
        c.erase()


def _sink_constant_arguments(module):
    """An outlined function's argument that every call passes the same
    constant: the constant made in the function instead."""
    from air.ir import InsertionPoint

    for f in _functions(module):
        if _name(f).startswith(_INTRINSIC) or not _has_body(f):
            continue
        calls = _calls_of(module, _name(f))
        if not calls or any(len(c.results) for c in calls):
            continue
        block = f.regions[0].blocks[0]
        consts = {}
        for i in range(len(block.arguments)):
            defs = [_def(c.operands[i]) for c in calls]
            keys = [_constant_key(d) for d in defs]
            if keys[0] is not None and all(k == keys[0] for k in keys):
                consts[i] = defs[0]
        if not consts:
            continue
        first = list(block.operations)[0]
        for i, d in consts.items():
            c = d.clone(ip=InsertionPoint(first))
            block.arguments[i].replace_all_uses_with(c.results[0])
        _drop_arguments(f, calls, set(consts))


def _sink_stream_starts(module):
    """Three arguments (a stream's state) that every call passes from one
    aievec.fifo_start, of one buffer type and one constant offset: the
    buffer passed, the stream started in the function."""
    from air.dialects import func
    from air.ir import (
        ArrayAttr,
        DictAttr,
        FunctionType,
        InsertionPoint,
        IntegerAttr,
        TypeAttr,
    )

    state = ["!llvm.ptr", "vector<32xi32>", "i32"]
    for f in _functions(module):
        if _name(f).startswith(_INTRINSIC) or not _has_body(f):
            continue
        calls = _calls_of(module, _name(f))
        if not calls or any(len(c.results) for c in calls):
            continue
        block = f.regions[0].blocks[0]
        tys = [str(a.type) for a in block.arguments]
        for i in range(len(tys) - 2):
            if tys[i : i + 3] != state:
                continue
            starts = []
            for c in calls:
                d = [_def(c.operands[i + j]) for j in range(3)]
                if (
                    d[0] is None
                    or d[0].name != "aievec.fifo_start"
                    or any(x != d[0] for x in d)
                ):
                    starts = None
                    break
                if any(c.operands[i + j] != d[0].results[j] for j in range(3)):
                    starts = None
                    break
                starts.append(d[0])
            if not starts:
                continue
            offs = set()
            for s in starts:
                o = _def(s.operands[1])
                offs.add(
                    IntegerAttr(o.attributes["value"]).value
                    if o is not None and o.name == "arith.constant"
                    else None
                )
            mtys = {str(s.operands[0].type) for s in starts}
            if len(offs) != 1 or None in offs or len(mtys) != 1:
                continue
            # the callee: the buffer in place of the three (the arguments
            # from i on made again after it), the start first
            n = len(block.arguments)
            buf = block.add_argument(starts[0].operands[0].type, starts[0].location)
            for k in range(i + 3, n):
                a = block.arguments[k]
                a.replace_all_uses_with(block.add_argument(a.type, a.location))
            first = list(block.operations)[0]
            off = _def(starts[0].operands[1]).clone(ip=InsertionPoint(first))
            st = starts[0].clone(ip=InsertionPoint(first))
            st.operands[0] = buf
            st.operands[1] = off.results[0]
            for j in range(3):
                block.arguments[i + j].replace_all_uses_with(st.results[j])
            for k in reversed(range(i, n)):
                block.erase_argument(k)
            if "arg_attrs" in f.attributes:
                aa = list(f.attributes["arg_attrs"])
                f.attributes["arg_attrs"] = ArrayAttr.get(
                    aa[:i] + [DictAttr.get({})] + aa[i + 3 :]
                )
            ty = f.attributes["function_type"].value
            f.attributes["function_type"] = TypeAttr.get(
                FunctionType.get([a.type for a in block.arguments], list(ty.results))
            )
            for c, s in zip(calls, starts):
                ops = list(c.operands)
                func.CallOp(
                    [],
                    _callee(c),
                    ops[:i] + [s.operands[0]] + ops[i + 3 :],
                    ip=InsertionPoint(c),
                )
                c.erase()
            break


def _base(v):
    """The buffer the memref or pointer v views: an allocation or a block
    argument (the value), None if not known."""
    for _ in range(16):
        d = _def(v)
        if d is None:
            return v
        if d.name in ("memref.alloc", "memref.alloca"):
            return v
        if d.name not in _VIEWS:
            return None
        v = d.operands[0]
    return None


def _restrict(module, names):
    """llvm.noalias on a restrict function's buffer argument when at every
    call no other pointer argument reaches the same buffer."""
    from air.ir import ArrayAttr, DictAttr, UnitAttr

    for f in _functions(module):
        name = _name(f)
        if name not in names or not _has_body(f):
            continue
        block = f.regions[0].blocks[0]
        tys = [str(a.type) for a in block.arguments]
        ptrs = [i for i, t in enumerate(tys) if t.startswith(("memref<", "!llvm.ptr"))]
        mem = [i for i in ptrs if tys[i].startswith("memref<")]
        calls = _calls_of(module, name)
        if not mem or not calls:
            continue
        keep = set(mem)
        for c in calls:
            bases = {i: _base(c.operands[i]) for i in ptrs}
            for i in mem:
                if bases[i] is None or any(
                    bases[j] is None or bases[j] == bases[i] for j in ptrs if j != i
                ):
                    keep.discard(i)
        if not keep:
            continue
        old = f.attributes["arg_attrs"] if "arg_attrs" in f.attributes else None
        attrs = []
        for i in range(len(tys)):
            d = {}
            if old is not None:
                od = DictAttr(old[i])
                for k in range(len(od)):
                    na = od[k]
                    d[na.name] = na.attr
            if i in keep:
                d["llvm.noalias"] = UnitAttr.get()
            attrs.append(DictAttr.get(d))
        f.attributes["arg_attrs"] = ArrayAttr.get(attrs)
