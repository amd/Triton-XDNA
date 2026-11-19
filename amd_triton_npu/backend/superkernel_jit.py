# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Superkernels (see `tl.extra.air`) from their TTIR.

The TTIR of a superkernel is its function with the loop over the waves, the
functions of its herds, the kernels they call, and one declaration per
L2 buffer. `read` turns it into a `Superkernel`: each kernel call becomes a
call compiled on its own, with every argument traced back to a DDR buffer, an
L2 buffer, an int argument of the function, a constant, or a value of the
wave's index that is evaluated for every wave. `Launcher` builds the dispatch
on its first launch and runs it.
"""

import hashlib
import os
import subprocess
import tempfile

from .superkernel import (
    _TILE,
    Superkernel,
    _Segment,
    Tensor,
    _Channel,
    _Local,
    _Scalar,
    _Table,
    _TileId,
    _Wave,
)

#: the core array npu2 places herds on: columns, then the core rows
_ARRAY = {"npu2": (8, range(2, 6))}


def _generic(ttir):
    """`ttir` in MLIR's generic form, which parses without Triton's dialects."""
    from .compiler import _get_amd_triton_npu_opt_path

    with tempfile.NamedTemporaryFile("w", suffix=".ttir", delete=False) as f:
        f.write(ttir)
    try:
        return subprocess.run(
            [_get_amd_triton_npu_opt_path(), f.name, "--mlir-print-op-generic"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    finally:
        os.unlink(f.name)


def _attr(op, name, default=None):
    from air.ir import IntegerAttr, StringAttr

    if name not in op.attributes:
        return default
    v = op.attributes[name]
    if isinstance(v, IntegerAttr):
        return v.value
    if isinstance(v, StringAttr):
        return v.value
    return v


def _callee(op):
    from air.ir import FlatSymbolRefAttr

    return FlatSymbolRefAttr(op.attributes["callee"]).value


def _kernel_name(symbol):
    """The Python name of the jit function a TTIR function was made from."""
    return symbol.rsplit(".", 1)[-1].split("__")[0]


class _PerWave:
    """An int whose value depends on the wave: `values[k]` at wave k."""

    def __init__(self, values):
        self.values = list(values)

    def lowered(self):
        """As an argument of a call: an int, a `_Wave` or a `_Table`."""
        v = self.values
        if len(set(v)) == 1:
            return v[0]
        a = v[1] - v[0] if len(v) > 1 else 0
        if all(x == a * k + v[0] for k, x in enumerate(v)):
            return _Wave(a, v[0])
        return _Table(v)


_CMP = {
    0: lambda x, y: x == y,
    1: lambda x, y: x != y,
    2: lambda x, y: x < y,
    3: lambda x, y: x <= y,
    4: lambda x, y: x > y,
    5: lambda x, y: x >= y,
}


def _divsi(x, y):
    """Division rounding toward zero, as arith.divsi."""
    q = abs(x) // abs(y)
    return q if (x < 0) == (y < 0) else -q


_BINARY = {
    "arith.addi": lambda x, y: x + y,
    "arith.subi": lambda x, y: x - y,
    "arith.muli": lambda x, y: x * y,
    "arith.divsi": _divsi,
    "arith.remsi": lambda x, y: x - y * _divsi(x, y),
    "arith.maxsi": max,
    "arith.minsi": min,
}

_CASTS = (
    "arith.extsi",
    "arith.extui",
    "arith.trunci",
    "arith.index_cast",
    "arith.index_castui",
)


def _evaluate(op, env, what):
    """The value of the one result of scalar op `op`, given its operands'
    values in `env`: a constant, a _PerWave, or the operand's own value.
    Only ints are computed with."""
    from air.ir import FloatAttr, IntegerAttr

    name = op.operation.name
    if name == "arith.constant":
        v = op.attributes["value"]
        if isinstance(v, FloatAttr):
            return v.value
        if str(v.type) == "i1":
            return bool(IntegerAttr(v).value)
        return IntegerAttr(v).value
    args = [env.get(v) for v in op.operands]
    for a in args:
        if not isinstance(a, (int, _PerWave)):
            raise ValueError(
                f"{what}: {name} on {a!r}; only ints from constants and the "
                "wave's index can be computed"
            )
    if name in _CASTS:
        return args[0]
    if name == "arith.cmpi":
        f = _CMP.get(_attr(op, "predicate"))
        if f is None:
            raise ValueError(f"{what}: unsupported comparison in {op}")
    elif name == "arith.select":
        f = lambda c, x, y: x if c else y
    elif name in _BINARY:
        f = _BINARY[name]
    else:
        raise ValueError(f"{what}: {name} on the wave's index is not supported")
    n = max((len(a.values) for a in args if isinstance(a, _PerWave)), default=0)
    if not n:
        return int(f(*args))
    per = [a.values if isinstance(a, _PerWave) else [a] * n for a in args]
    return _PerWave(int(f(*xs)) for xs in zip(*per))


def _array(npu):
    if npu not in _ARRAY:
        raise ValueError(
            f"superkernels place herds on npu2 only; give `at` for every "
            f"herd on {npu}"
        )
    return _ARRAY[npu]


def _place(parts, npu):
    """The corner of each herd: its own `at`, or the first free spot
    for its cores, row by row from the bottom."""
    cols, rows = None, None
    taken = set()

    def cells(at, cores):
        return {
            (at[0] + c, at[1] + r) for c in range(cores[0]) for r in range(cores[1])
        }

    corners = []
    for p in parts:
        if p["at"] is not None:
            got = cells(p["at"], p["cores"])
            if got & taken:
                raise ValueError(f"herd {p['name']} at {p['at']} overlaps another")
            taken |= got
    for p in parts:
        if p["at"] is not None:
            corners.append(p["at"])
            continue
        if cols is None:
            cols, rows = _array(npu)
        spot = next(
            (
                (c, r)
                for r in rows
                for c in range(cols)
                if c + p["cores"][0] <= cols
                and r + p["cores"][1] <= rows[-1] + 1
                and not cells((c, r), p["cores"]) & taken
            ),
            None,
        )
        if spot is None:
            raise ValueError(f"no room on the array for herd {p['name']}")
        taken |= cells(spot, p["cores"])
        corners.append(spot)
    return corners


def _program_ids_as_args(f):
    """Replace kernel function `f`'s program ids by i32 arguments appended
    to it. Returns the axes, in the order of the new arguments."""
    from air.ir import FunctionType, IntegerType, Location, TypeAttr

    block = f.regions[0].blocks[0]
    pids = [o for o in _walk(f) if o.operation.name == "tt.get_program_id"]
    axes = sorted({_attr(o, "axis") for o in pids})
    if not axes:
        return []
    i32 = IntegerType.get_signless(32)
    args = {a: block.add_argument(i32, Location.unknown()) for a in axes}
    for o in pids:
        o.results[0].replace_all_uses_with(args[_attr(o, "axis")])
        o.operation.erase()
    ft = FunctionType(TypeAttr(f.attributes["function_type"]).value)
    f.attributes["function_type"] = TypeAttr.get(
        FunctionType.get(list(ft.inputs) + [i32] * len(axes), list(ft.results))
    )
    return axes


def _channels_as_args(f):
    """Replace kernel function `f`'s uses of channels by pointer arguments
    appended to it, one per channel and element type. Returns the channels'
    names, in the order of the new arguments."""
    from air.ir import FunctionType, Location, TypeAttr

    block = f.regions[0].blocks[0]
    prefix = "air.channel."
    uses = []  # (name, value standing for the channel, ops to erase)
    for o in _walk(f):
        if o.operation.name != "tt.call" or not _callee(o).startswith(prefix):
            continue
        name = _callee(o)[len(prefix) :]
        users = [u.owner for u in o.results[0].uses]
        # The element type comes from the cast of the channel's i8 pointer.
        if len(users) == 1 and users[0].name == "tt.bitcast":
            uses.append((name, users[0].results[0], [users[0], o.operation]))
        else:
            uses.append((name, o.results[0], [o.operation]))
    args, names, types = {}, [], []
    for name, v, _ in uses:
        key = (name, str(v.type))
        if key not in args:
            args[key] = block.add_argument(v.type, Location.unknown())
            names.append(name)
            types.append(v.type)
    for name, v, ops in uses:
        v.replace_all_uses_with(args[(name, str(v.type))])
        for op in ops:
            op.erase()
    if names:
        ft = FunctionType(TypeAttr(f.attributes["function_type"]).value)
        f.attributes["function_type"] = TypeAttr.get(
            FunctionType.get(list(ft.inputs) + types, list(ft.results))
        )
    return names


def _locals_as_args(f, funcs):
    """Replace kernel function `f`'s own L1 buffers by pointer arguments
    appended to it. Returns each buffer's (symbol, elements, element type),
    in the order of the new arguments."""
    from air.ir import FunctionType, Location, TypeAttr

    block = f.regions[0].blocks[0]
    calls = [
        o
        for o in _walk(f)
        if o.operation.name == "tt.call" and _callee(o).startswith("air.local.")
    ]
    out, types = [], []
    for o in calls:
        decl = funcs[_callee(o)]
        t = o.results[0].type
        a = block.add_argument(t, Location.unknown())
        o.results[0].replace_all_uses_with(a)
        out.append(
            (
                _callee(o),
                _attr(decl, "air.alloc_elements"),
                str(t).split("<")[1].rstrip(">"),
            )
        )
        o.operation.erase()
        types.append(t)
    if out:
        ft = FunctionType(TypeAttr(f.attributes["function_type"]).value)
        f.attributes["function_type"] = TypeAttr.get(
            FunctionType.get(list(ft.inputs) + types, list(ft.results))
        )
    return out


def _leaf_module(module, symbol, name):
    """The TTIR of the kernel `symbol` of `module` as a module of its own: a
    public function `name` with the functions it calls inlined, run through
    the backend's TTIR passes like a kernel compiled alone; and the axes of
    the program ids it reads, which become arguments after its own."""
    import re

    from triton._C.libtriton import ir

    from .compiler import NPUBackend

    funcs = {_attr(f, "sym_name"): f for f in module.body.operations}
    order, todo = [], [symbol]
    while todo:
        s = todo.pop()
        if s in order:
            continue
        order.append(s)
        for o in _walk(funcs[s]):
            if o.operation.name == "tt.call":
                todo.append(_callee(o))
    for s in order[1:]:
        for o in _walk(funcs[s]):
            if o.operation.name == "tt.get_program_id" or (
                o.operation.name == "tt.call" and _callee(o).startswith("air.channel.")
            ):
                raise ValueError(
                    f"{name}: a function it calls reads a program id or a channel"
                )
    axes = _program_ids_as_args(funcs[symbol])
    chans = _channels_as_args(funcs[symbol])
    own = _locals_as_args(funcs[symbol], funcs)
    texts = []
    for s in order:
        t = funcs[s].operation.get_asm(print_generic_op_form=True)
        t = t.replace("noinline = true", "noinline = false")
        if s == symbol:
            t = t.replace(f'sym_name = "{s}"', f'sym_name = "{name}"')
            t = t.replace('sym_visibility = "private"', 'sym_visibility = "public"')
            # Kernel pointers are 16-byte aligned, which a kernel compiled
            # alone is told through its arguments' attributes.
            block = funcs[s].regions[0].blocks[0]
            attrs = ", ".join(
                "{tt.divisibility = 16 : i32}" if "!tt.ptr" in str(a.type) else "{}"
                for a in block.arguments
            )
            t = re.sub(
                r"<\{(arg_attrs = \[[^\]]*\], )?", f"<{{arg_attrs = [{attrs}], ", t, 1
            )
        texts.append(t)
    text = "module {\n" + "\n".join(texts) + "\n}\n"
    ctx = ir.context()
    ir.load_dialects(ctx)
    with tempfile.NamedTemporaryFile("w", suffix=".ttir", delete=False) as f:
        f.write(text)
    try:
        mod = ir.parse_mlir_module(f.name, ctx)
    finally:
        os.unlink(f.name)
    mod.context = ctx  # make_ttir's pass manager needs it, as in Triton's compile
    NPUBackend.make_ttir(mod, {}, None)
    return str(mod), axes, chans, [_Local(n, t, sym) for sym, n, t in own]


def _walk(op):
    """The ops nested in `op`, in order."""
    out = []
    for r in op.regions:
        for b in r.blocks:
            for o in b.operations:
                out.append(o)
                out += _walk(o)
    return out


def read(ttir, air_project_path, fp_fusion=False):
    """The `Superkernel` of the superkernel whose TTIR is `ttir`, and the
    kind of each argument of its function: "buffer" or "scalar". Its kernels
    are compiled with contraction if `fp_fusion`."""
    from air.ir import Context, Location, Module

    from .driver import detect_npu_version

    with Context() as ctx, Location.unknown():
        ctx.allow_unregistered_dialects = True
        module = Module.parse(_generic(ttir))
        funcs = {_attr(f, "sym_name"): f for f in module.body.operations}
        top = next(f for f in funcs.values() if _attr(f, "sym_visibility") == "public")
        name = _attr(top, "sym_name")
        body = top.regions[0].blocks[0]

        def runs_herds(o):
            return any(
                x.operation.name == "tt.call"
                and _attr(funcs[_callee(x)], "air.herd") is not None
                for x in o.regions[0].blocks[0].operations
            )

        # The loop over the waves is the one that runs herds; with one wave
        # it folds away and the function's body is the wave's.
        loops = [
            o
            for o in body.operations
            if o.operation.name == "scf.for" and runs_herds(o)
        ]
        if len(loops) > 1:
            raise ValueError(f"{name}: a superkernel loops over its waves once")
        loop = loops[0] if loops else None
        env = {}
        kinds = [
            "buffer" if "!tt.ptr" in str(a.type) else "scalar" for a in body.arguments
        ]
        # Arithmetic hoisted out of the wave loop that needs the run-time
        # scalars: the waves' segment code computes it.
        hoisted = []
        for o in body.operations:
            n = o.operation.name
            if loop is None or o == loop or n == "tt.return":
                continue
            decl = funcs[_callee(o)] if n == "tt.call" else None
            if decl is not None and (
                _attr(decl, "air.alloc_elements")
                or _attr(decl, "air.channel_size")
                or _callee(o) == "air.backend"
            ):
                continue
            if n.startswith("arith."):
                try:
                    env[o.result] = _evaluate(o, env, name)
                except ValueError as e:
                    env[o.result] = e
                    hoisted.append(o)
                continue
            raise ValueError(f"{name}: {n} outside the loop over the waves")
        if loop is None:
            waves = [0]
        else:
            lb, ub, step = (env.get(v) for v in loop.operands[:3])
            if not all(isinstance(x, int) for x in (lb, ub, step)):
                raise ValueError(
                    f"{name}: the loop over the waves needs a constexpr trip count"
                )
            waves = list(range(lb, ub, step))
        sk = Superkernel(name, air_project_path, len(waves), fp_fusion)
        if "air.backend" in funcs:
            f = funcs["air.backend"]
            prefix = "air.backend."
            sk.backend = {
                k[len(prefix) :]: _attr(f, k)
                for k in (
                    str(a if isinstance(a, str) else a.name) for a in f.attributes
                )
                if k.startswith(prefix)
            }
        for i, a in enumerate(body.arguments):
            if kinds[i] == "buffer":
                elt = str(a.type).split("<")[1].rstrip(">")
                env[a] = sk.ddr(f"a{i}", 0, elt)
            else:
                env[a] = sk.scalar(f"r{i}")
        chans = {}
        for f in funcs.values():
            size = _attr(f, "air.channel_size")
            if size is None:
                continue
            ints = lambda t: tuple(int(x) for x in t.split(","))  # noqa: E731
            bshape = _attr(f, "air.channel_broadcast_shape")
            cname = _attr(f, "sym_name")[len("air.channel.") :]
            if cname == name:
                raise ValueError(f"channel {cname} has the superkernel's name")
            chans[cname] = sk.channel(
                cname,
                ints(size),
                ints(bshape) if bshape else None,
                _attr(f, "air.channel_type"),
                _attr(f, "air.channel_buffer_resources"),
                _attr(f, "air.channel_tile_dma"),
            )
        for o in body.operations:
            if o.operation.name == "tt.call":
                decl = funcs[_callee(o)]
                n = _attr(decl, "air.alloc_elements")
                if n:
                    elt = str(o.results[0].type).split("<")[1].rstrip(">")
                    space = _attr(decl, "air.memory_space")
                    col = _attr(decl, "air.memtile_col")
                    env[o.results[0]] = sk.onchip(_callee(o), n, elt, space, col)
        if loop is None:
            lbody = body
        else:
            lbody = loop.regions[0].blocks[0]
            env[lbody.arguments[0]] = _PerWave(waves)

        # The wave's body: herds, and the segment code between them.
        parts, order, pending = [], [], list(hoisted)
        for o in lbody.operations:
            n = o.operation.name
            if n in ("scf.yield", "tt.return"):
                continue
            if loop is None and len(o.results) and o.results[0] in env:
                continue  # a buffer of the function's body, made above
            if n == "tt.call" and _callee(o).startswith("air.alloc."):
                # A buffer of the wave's body: one for all waves, as those of
                # the function's body, which herds and segment code share.
                decl = funcs[_callee(o)]
                elt = str(o.results[0].type).split("<")[1].rstrip(">")
                env[o.results[0]] = sk.onchip(
                    _callee(o),
                    _attr(decl, "air.alloc_elements"),
                    elt,
                    _attr(decl, "air.memory_space"),
                    _attr(decl, "air.memtile_col"),
                )
                continue
            if n == "tt.call" and (
                _callee(o).startswith("air.channel.") or _callee(o) == "air.backend"
            ):
                continue  # where a channel is declared, or the options given
            if n.startswith("arith."):
                try:
                    env[o.result] = _evaluate(o, env, name)
                except ValueError as e:
                    # Segment code may compute it; a herd may not take it.
                    env[o.result] = e
                    pending.append(o)
                continue
            if n == "tt.call" and _attr(funcs[_callee(o)], "air.herd") is not None:
                if pending:
                    order.append(("segment", pending))
                    pending = []
                order.append(("herd", len(parts)))
                parts.append((o, funcs[_callee(o)]))
                continue
            pending.append(o)
        if pending:
            order.append(("segment", pending))
        if not parts:
            raise ValueError(f"{name}: the wave loop runs no air.herd")

        # Herds of one name run in turn
        # on the same cores.
        settings = [_settings(f) for _, f in parts]
        herds = {}
        for p in settings:
            h = herds.setdefault(p["name"], dict(p))
            if p["cores"] != h["cores"]:
                raise ValueError(
                    f"herd {p['name']} is given {h['cores']} and {p['cores']} cores"
                )
            if p["at"] is not None and h["at"] not in (None, p["at"]):
                raise ValueError(
                    f"herd {p['name']} is placed at {h['at']} and {p['at']}"
                )
            h["at"] = h["at"] or p["at"]
        places = _place(
            list(herds.values()),
            (
                detect_npu_version()
                if any(h["at"] is None for h in herds.values())
                else None
            ),
        )
        corner = dict(zip(herds, places))
        corners = [corner[p["name"]] for p in settings]

        sched_dir = os.path.join(sk.proj, "schedules")
        os.makedirs(sched_dir, exist_ok=True)
        leaves, exported = {}, {}
        for kind, item in order:
            if kind == "segment":
                for where in ("launch", "segment"):
                    sg = _SegmentWriter(env, funcs, chans, exported, where).write(item)
                    if not sg.empty:
                        sk.segment(sg)
                continue
            (call, f), p, at = parts[item], settings[item], corners[item]
            schedule = _bound_schedule(p, sched_dir)
            for sym, args in _kernel_calls(call, f, funcs, env, p["name"]):
                kname = _kernel_name(sym)
                if sym not in leaves:
                    leaves[sym] = _leaf_module(module, sym, kname)
                leaf, axes, used, own = leaves[sym]
                missing = [n for n in used if n not in chans]
                if missing:
                    raise ValueError(f"{kname}: channel {missing[0]} is not declared")
                args = (
                    list(args)
                    + [_TILE[a] for a in axes]
                    + [chans[n] for n in used]
                    + own
                )
                vals = []
                for v in args:
                    if isinstance(v, ValueError):
                        raise v
                    if isinstance(v, _PerWave):
                        v = v.lowered()
                    if not isinstance(
                        v,
                        (Tensor, _Channel, _Local, _Scalar, _TileId, _Wave, int, float),
                    ):
                        raise ValueError(f"{p['name']}: {kname} takes {v!r}")
                    vals.append(v)
                sk.call(kname, leaf, vals, at, p["name"], schedule, p["cores"])
        return sk, kinds


_PREDICATES = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]
_BINARY_TEXT = (
    "arith.addi",
    "arith.subi",
    "arith.muli",
    "arith.divsi",
    "arith.divui",
    "arith.remsi",
    "arith.remui",
    "arith.maxsi",
    "arith.minsi",
    "arith.andi",
    "arith.ori",
    "arith.xori",
    "arith.shli",
    "arith.shrsi",
    "arith.shrui",
)


class _SegmentWriter:
    """Writes segment code, ops of the wave's body between herds, as an
    AIR `_Segment`: loops and branches as they are (on index), L2 buffers
    as allocs, and channel ops on buffers as AIR's. Values from outside are
    the func's arguments; the buffers allocated at the wave's level are
    exported, for later segment code and herds, and freed at the wave's
    end."""

    def __init__(self, env, funcs, chans, exported, where="segment"):
        self.env, self.funcs, self.chans = env, funcs, chans
        # "segment": the code on L2 buffers; "launch": the transfers to and
        # from DDR buffers, which the host issues, outside the segment.
        self.where = where
        self.exported = exported  # TTIR value -> (name, memref type)
        self.names, self.types = {}, {}
        self.args, self.binds, self.lines = [], [], []
        self.exports = []  # (name, local name, type)
        self.writes = []  # the tensors its gets fill
        self.n = 0

    def fresh(self):
        self.n += 1
        return f"%v{self.n}"

    def ref(self, v):
        """The local name of TTIR value `v`, an argument if from outside."""
        from air.ir import OpResult

        if v in self.names:
            return self.names[v]
        if isinstance(self.env.get(v), ValueError):
            # Arithmetic on run-time scalars from earlier in the wave (or
            # hoisted before it): computed again here.
            if isinstance(v, OpResult) and v.owner.name.startswith("arith."):
                self.block([v.owner])
                return self.names[v]
            raise self.env[v]
        name = f"%a{len(self.args)}"
        if v in self.exported:
            bind, typ = self.exported[v]
            self.types[name] = typ
        else:
            bind = self.env.get(v)
            if isinstance(bind, _PerWave):
                bind = bind.lowered()
            if isinstance(bind, Tensor) and bind.onchip:
                typ = f"memref<{bind.n}x{bind.dtype}, {bind.space} : i32>"
                self.types[name] = typ
            elif isinstance(bind, Tensor) and self.where == "launch":
                typ = f"memref<*x{bind.dtype}>"
                self.types[name] = typ
            elif isinstance(bind, (bool, int, _Wave, _Scalar)):
                typ = str(v.type)
            else:
                raise ValueError(f"segment code uses {bind!r}, from outside it")
        self.args.append(f"{name}: {typ}")
        self.binds.append(bind)
        self.names[v] = name
        return name

    def constant(self, v):
        """Whether TTIR value `v` is a constant."""
        from air.ir import OpResult

        if isinstance(self.env.get(v), int):
            return True
        if isinstance(v, OpResult) and v.owner.name == "arith.constant":
            return True
        if isinstance(v, OpResult) and v.owner.name in _CASTS:
            return self.constant(v.owner.operands[0])
        return False

    def index(self, v):
        """An index-typed name for TTIR int value `v`."""
        a = self.ref(v)
        if str(v.type) == "index":
            return a
        r = self.fresh()
        self.lines.append(f"{r} = arith.index_cast {a} : {v.type} to index")
        return r

    def write(self, ops):
        self.block(ops, top=True)
        rets = [loc for _, loc, _ in self.exports]
        types = [t for _, _, t in self.exports]
        body = "\n".join(self.lines)
        ret = f"return {', '.join(rets)} : {', '.join(types)}" if rets else "return"
        text = (
            f"func.func @segment({', '.join(self.args)}) -> ({', '.join(types)}) {{\n"
            f"{body}\n{ret}\n}}\n"
        )
        for (name, _, typ), v in zip(self.exports, self._exported_values):
            self.exported[v] = (name, typ)
        sg = _Segment(text, self.binds, [n for n, _, _ in self.exports], self.where)
        sg.writes = self.writes
        sg.empty = not any(
            "air.channel" in ln or "memref.alloc" in ln for ln in self.lines
        )
        return sg

    _exported_values = ()

    def side(self, o):
        """Where top-level op `o` of the segment code goes: "launch" if it
        moves DDR data, "segment" if L2 data, None if neither."""
        sides = set()
        for x in [o] + _walk(o):
            if x.operation.name != "tt.call":
                continue
            c = _callee(x)
            if c.startswith("air.alloc."):
                sides.add("segment")
            elif c.startswith("air.dma."):
                b = self.env.get(x.operands[0])
                ddr = isinstance(b, Tensor) and not b.onchip
                sides.add("launch" if ddr else "segment")
        if len(sides) > 1:
            raise ValueError(
                "segment code: one loop or branch moves both DDR and L2 data"
            )
        return sides.pop() if sides else None

    def block(self, ops, top=False):
        frees = []
        for o in ops:
            n = o.operation.name
            if top and self.side(o) not in (None, self.where):
                continue
            if n in ("scf.yield", "tt.return"):
                if n == "scf.yield" and len(o.operands):
                    raise ValueError("segment code: loops and branches yield nothing")
                continue
            if n == "tt.call" and _callee(o).startswith("air.alloc."):
                decl = self.funcs[_callee(o)]
                elt = str(o.results[0].type).split("<")[1].rstrip(">")
                space = _attr(decl, "air.memory_space")
                n_ = _attr(decl, "air.alloc_elements")
                typ = f"memref<{n_}x{elt}, {space} : i32>"
                r = self.fresh()
                col = _attr(decl, "air.memtile_col")
                at = "" if col is None else f" {{air.memtile_col = {col} : i32}}"
                self.lines.append(f"{r} = memref.alloc(){at} : {typ}")
                self.names[o.results[0]], self.types[r] = r, typ
                if top:
                    self.exports.append((f"{_callee(o)}", r, typ))
                    self._exported_values = tuple(self._exported_values) + (
                        o.results[0],
                    )
                else:
                    frees.append((r, typ))
            elif n == "tt.call" and _callee(o).startswith("air.dma."):
                self.dma(o)
            elif n == "scf.for":
                if len(o.results):
                    raise ValueError("segment code: a loop carries no values")
                lb, ub, st = (self.index(v) for v in o.operands[:3])
                body = o.regions[0].blocks[0]
                iv, ivx = self.fresh(), self.fresh()
                self.lines.append(f"scf.for {ivx} = {lb} to {ub} step {st} {{")
                self.lines.append(
                    f"{iv} = arith.index_cast {ivx} : index to {body.arguments[0].type}"
                )
                self.names[body.arguments[0]] = iv
                self.block(list(body.operations))
                self.lines.append("}")
            elif n == "scf.if":
                if len(o.results):
                    raise ValueError("segment code: a branch yields no values")
                c = self.ref(o.operands[0])
                self.lines.append(f"scf.if {c} {{")
                self.block(list(o.regions[0].blocks[0].operations))
                if len(o.regions[1].blocks):
                    self.lines.append("} else {")
                    self.block(list(o.regions[1].blocks[0].operations))
                self.lines.append("}")
            elif n == "arith.constant":
                r = self.fresh()
                v = o.attributes["value"]
                self.lines.append(f"{r} = arith.constant {v}")
                self.names[o.result] = r
            elif n in _BINARY_TEXT:
                a, b = (self.ref(v) for v in o.operands)
                r = self.fresh()
                self.lines.append(f"{r} = {n} {a}, {b} : {o.result.type}")
                self.names[o.result] = r
            elif n == "arith.cmpi":
                a, b = (self.ref(v) for v in o.operands)
                r = self.fresh()
                pred = _PREDICATES[_attr(o, "predicate")]
                self.lines.append(
                    f"{r} = arith.cmpi {pred}, {a}, {b} : {o.operands[0].type}"
                )
                self.names[o.result] = r
            elif n == "arith.select":
                c, a, b = (self.ref(v) for v in o.operands)
                r = self.fresh()
                self.lines.append(f"{r} = arith.select {c}, {a}, {b} : {o.result.type}")
                self.names[o.result] = r
            elif n in _CASTS:
                a = self.ref(o.operands[0])
                r = self.fresh()
                self.lines.append(
                    f"{r} = {n} {a} : {o.operands[0].type} to {o.result.type}"
                )
                self.names[o.result] = r
            else:
                raise ValueError(f"segment code: {n} is not supported")
        for r, typ in frees:
            self.lines.append(f"memref.dealloc {r} : {typ}")

    def dma(self, o):
        decl = self.funcs[_callee(o)]
        ch = _attr(decl, "air.dma_channel")
        if ch not in self.chans:
            raise ValueError(f"channel {ch} is not declared")
        sizes = _attr(decl, "air.dma_sizes").split(",")
        strides = [int(x) for x in _attr(decl, "air.dma_strides").split(",")]
        k = _attr(decl, "air.dma_indices")
        ops = list(o.operands)
        nd = sizes.count("?")
        buf, offs = ops[0], ops[1 : 1 + len(sizes)]
        dyn, idx = (
            iter(ops[1 + len(sizes) : 1 + len(sizes) + nd]),
            ops[1 + len(sizes) + nd :],
        )
        # a size known at run time: the transfer's, as an index value
        sizes = [self.index(next(dyn)) if x == "?" else x for x in sizes]
        if len(idx) != k:
            raise ValueError(f"channel {ch}: {len(idx)} indices, not {k}")
        b = self.ref(buf)
        if b not in self.types:
            raise ValueError(
                f"channel {ch}: a transfer of a buffer the segment has not"
            )
        for v in idx:
            if not self.constant(v):
                raise ValueError(
                    f"channel {ch}: the segment's transfers take constant indices, "
                    "the routes of the memtiles being fixed (unroll the loop with "
                    "tl.static_range)"
                )
        o_ = [self.index(v) for v in offs]
        i_ = [self.index(v) for v in idx]
        op = "air.channel.put" if _attr(decl, "air.dma_put") else "air.channel.get"
        target = self.env.get(buf)
        if op == "air.channel.get" and isinstance(target, Tensor):
            self.writes.append(target)
        self.lines.append(
            f"{op} @{ch}[{', '.join(i_)}] ({b}[{', '.join(o_)}] "
            f"[{', '.join(map(str, sizes))}] [{', '.join(map(str, strides))}]) "
            f": ({self.types[b]})"
        )


def _aircc_options(opts):
    """`Superkernel.compile`'s aircc arguments for `air.backend`'s options
    `opts` (strings, by name), as AIR's XRTBackend passes them."""
    if opts is None:
        return dict(
            extra_args=(
                "--air-channel-multiplexing=L1",
                "--omit-ping-pong-transform=L1",
            )
        )
    yes = lambda k: opts.get(k) == "True"  # noqa: E731
    args = []
    if opts.get("omit_pingpong"):
        args.append(f"--omit-ping-pong-transform={opts['omit_pingpong']}")
    for level in filter(None, opts.get("channel_multiplexing", "").split(",")):
        args.append(f"--air-channel-multiplexing={level}")
    if yes("use_lock_race_condition_fix"):
        args.append("--use-lock-race-condition-fix")
    if yes("use_lock_race_condition_fix_v2"):
        args.append("--use-lock-race-condition-fix-v2")
    if yes("allow_deadlock_prone_routing"):
        args.append("--allow-deadlock-prone-routing")
    feeds = []
    if opts.get("unroll_feed_loops", "True") == "True":
        feeds.append("--unroll-feed-loops")
    if opts.get("merge_contiguous_feeds", "True") == "True":
        feeds.append("--merge-contiguous-feeds")
    if int(opts.get("trace_size") or 0) > 0:
        args.append(f"--trace-size={opts['trace_size']}")
        args.append(f"--trace-offset={opts.get('trace_offset', 0)}")
    out = dict(extra_args=tuple(args), feed_args=tuple(feeds))
    if opts.get("stack_size"):
        out["stack_size"] = int(opts["stack_size"])
    return out


def _settings(f):
    """The settings `herd` gave herd function `f`."""
    at = None
    if _attr(f, "air.at_col") is not None:
        at = (_attr(f, "air.at_col"), _attr(f, "air.at_row"))
    return {
        "name": _attr(f, "air.name"),
        "cores": (_attr(f, "air.cores_x"), _attr(f, "air.cores_y")),
        "at": at,
        "schedule": _attr(f, "air.schedule"),
    }


def _bound_schedule(p, sched_dir):
    """The path of herd `p`'s schedule with its core counts bound."""
    from .transform_inject import _bind_core_counts

    with open(p["schedule"]) as f:
        text = f.read()
    try:
        text = _bind_core_counts(text, p["cores"])
    except ValueError as e:
        raise ValueError(f"herd {p['name']}: {e}") from None
    path = os.path.join(sched_dir, f"{p['name']}.mlir")
    with open(path, "w") as f:
        f.write(text)
    return path


def _kernel_calls(call, f, funcs, env, what):
    """The kernels herd function `f`, called by `call`, runs, in order:
    (function symbol, argument values). A herd function that is a kernel runs
    itself; otherwise it holds only calls of kernels and int arithmetic."""
    args = [env.get(v) for v in call.operands]
    body = f.regions[0].blocks[0]
    ops = [o for o in body.operations if o.operation.name != "tt.return"]
    if not all(
        o.operation.name == "tt.call" or o.operation.name.startswith("arith.")
        for o in ops
    ) or not any(o.operation.name == "tt.call" for o in ops):
        return [(_attr(f, "sym_name"), args)]
    local = dict(zip(body.arguments, args))
    out = []
    for o in ops:
        if o.operation.name == "tt.call":
            out.append((_callee(o), [local.get(v) for v in o.operands]))
        else:
            local[o.result] = _evaluate(o, local, what)
    return out


class Launcher:
    """Runs a superkernel launched as a Triton kernel. The dispatch is built
    on its first launch, in a directory under Triton's cache keyed by the
    TTIR, the schedules' text and the device."""

    def __init__(self, src, metadata):
        self.signature = list(src.signature.values())
        self.fp_fusion = bool(getattr(metadata, "enable_fp_fusion", False))
        self.built = None
        self.device = self._run = None

    def _build(self, ttir):
        import json
        import re

        from triton import knobs

        from .config import npu_config
        from .driver import detect_npu_version

        key = hashlib.sha256(ttir.encode())
        for path in sorted(set(re.findall(r'air\.schedule = "([^"]*)"', ttir))):
            with open(path, "rb") as f:
                key.update(f.read())
        key.update(
            f"{detect_npu_version()} {npu_config.bf16_emulation} {self.fp_fusion}".encode()
        )
        # The code that lowers it, so that a change to it rebuilds.
        here = os.path.dirname(__file__)
        for f in ("superkernel.py", "superkernel_jit.py", "transform_inject.py"):
            with open(os.path.join(here, f), "rb") as src:
                key.update(src.read())
        proj = os.path.join(knobs.cache.dir, "npu_superkernel", key.hexdigest()[:32])
        sk, kinds = read(ttir, proj, self.fp_fusion)
        done = os.path.join(proj, "built.json")
        if os.path.exists(done):
            with open(done) as f:
                xclbin, header = json.load(f)
        else:
            xclbin, header = sk.compile(
                runtime_loop_tiling_sizes=(),
                bf16_emulation=npu_config.bf16_emulation,
                **_aircc_options(sk.backend),
            )
            with open(done, "w") as f:
                json.dump([xclbin, header], f)
        segs = [g for k, g in sk.items if k == "segment"]
        written = [
            any(t in c.written for c in sk.calls) or any(t in g.writes for g in segs)
            for t in sk.buffers
        ]
        if (
            sk.backend
            and int(sk.backend.get("trace_size") or 0) > 0
            and len(written) > 2
        ):
            written[2] = True  # where AIR writes the traces
        return sk.name, kinds, xclbin, header, written

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):
        from . import driver
        from .config import npu_config

        if self.built is None:
            ttir = function.decode() if isinstance(function, bytes) else function
            self.built = self._build(ttir)
        if npu_config.compile_only:
            return
        # Triton passes the launch metadata and hooks, then every argument.
        values = [a for a, t in zip(args[4:], self.signature) if t != "constexpr"]
        # The hw_context is held between launches, in the same rationed pool
        # as the ELF launchers' sessions.
        driver._launch_with_session(self._key(), self, values)

    def _key(self):
        return ("superkernel", id(self))

    def launch(self, values):
        import numpy as np
        import torch

        from .npu_runner import sync_from, sync_to

        name, kinds, xclbin, header, written = self.built
        bufs = [v for v, k in zip(values, kinds) if k == "buffer"]
        rt = [int(v) for v, k in zip(values, kinds) if k == "scalar"]
        if self.device is None:
            self.device = _open_device(xclbin, len(bufs), header)
        bos = {}
        for i, t in enumerate(bufs):
            if not t.is_contiguous():
                raise ValueError(f"{name}: buffer argument {i} is not contiguous")
            raw = t.view(-1).view(torch.uint8).numpy()
            bos[i] = self.device.bo(raw.nbytes, i)
            np.frombuffer(bos[i].map(), np.uint8, raw.nbytes)[:] = raw
            sync_to(bos[i])
        run = self._run
        if run is not None:
            for i, b in bos.items():
                run.set_bo(i, b)
            try:
                run.set_rt(*rt)
            except ValueError:  # these scalars give a stream of another length
                self.device.launches.remove(run)
                run = None
        if run is None:
            run = self._run = self.device.launch(name, bos, rt=rt)
        run.start()
        run.wait()
        for i, t in enumerate(bufs):
            if written[i]:
                sync_from(bos[i])
                raw = t.view(-1).view(torch.uint8).numpy()
                raw[:] = np.frombuffer(bos[i].map(), np.uint8, raw.nbytes)

    def release_session(self):
        """Give the hw_context back; the next launch opens another."""
        if self.device is not None:
            self.device.close()
        self.device = self._run = None

    def resident(self, *values):
        """The launch with its buffers kept on the device, for repeated runs.
        `values` are the kernel's non-constexpr arguments, as at a launch that
        has built it. Returns a `Resident`, which takes over this launcher's
        hw_context and keeps it until it is dropped."""
        from . import driver

        if self.built is None:
            raise RuntimeError("resident: launch the kernel once first")
        driver._live_sessions.pop(self._key(), None)
        device = self.device
        if device is not None and self._run is not None:
            device.launches.remove(self._run)
        self.device = self._run = None
        return Resident(self.built, values, device)


def _open_device(xclbin, nargs, header):
    """A `Device` on `xclbin`. If the device has no hw_context to give, the
    ones this process can reopen later are given back and it is tried once
    more."""
    from . import driver
    from .npu_runner import Device

    try:
        return Device(xclbin, nargs, header)
    except RuntimeError as e:
        if driver._NO_HWCTX not in str(e):
            raise
    driver.release_external_contexts()
    driver._release_all_sessions()
    return Device(xclbin, nargs, header)


class Resident:
    """A superkernel's dispatch on buffers that stay on the device.
    `tensors[i]` is the i-th buffer argument as a tensor on its buffer
    object's memory (written in place by the host and the device).
    `run(*scalars, to=..., back=...)` syncs the buffers `to` to the device,
    runs with the scalar arguments, then syncs `back` (default: every buffer
    the kernel writes) to the host."""

    def __init__(self, built, values, device=None):
        import torch

        from .npu_runner import sync_to

        name, kinds, xclbin, header, written = built
        bufs = [v for v, k in zip(values, kinds) if k == "buffer"]
        rt = [int(v) for v, k in zip(values, kinds) if k == "scalar"]
        dev = device or _open_device(xclbin, len(bufs), header)
        self.device = dev
        self.written = [i for i, w in enumerate(written) if w]
        self.bos, self.tensors = {}, []
        for i, t in enumerate(bufs):
            if not t.is_contiguous():
                raise ValueError(f"{name}: buffer argument {i} is not contiguous")
            nbytes = t.numel() * t.element_size()
            bo = dev.bo(nbytes, i)
            v = torch.frombuffer(bo.map(), dtype=t.dtype, count=t.numel())
            v = v.view(t.shape)
            v.copy_(t)
            sync_to(bo)
            self.bos[i] = bo
            self.tensors.append(v)
        self.launch = dev.launch(name, self.bos, rt=rt)

    def run(self, *rt, to=(), back=None):
        from .npu_runner import sync_from, sync_to

        for i in to:
            sync_to(self.bos[i])
        self.launch.set_rt(*(int(v) for v in rt))
        self.launch.start()
        self.launch.wait()
        for i in self.written if back is None else back:
            sync_from(self.bos[i])
