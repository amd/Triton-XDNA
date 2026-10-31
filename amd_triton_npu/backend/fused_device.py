# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Several Triton kernels and prebuilt AIR modules on one NPU device.

Each op is one func whose segment is named like every other op's, so aircc
builds one device with a runtime sequence per op (an xclbin and a TXN builder
header). Each kernel's herds get columns of their own, next to the columns the
prebuilt modules' herds already use.
"""

import json
import os

from .arg_access import written_args
from .config import config_context
from .driver import (
    _aircc_compile,
    _ttshared_to_air,
    detect_npu_version,
)
from .driver_scope import driver_scope

SEG = "seg"


class FusedDevice:
    """Ops added with `add` (a Triton kernel at one set of arguments) or
    `add_air` (AIR built elsewhere), then compiled together by `compile`."""

    def __init__(self, name, air_project_path):
        self.name, self.proj = name, os.path.abspath(air_project_path)
        self.ops = []  # (op name, kernel key, air text)
        self.cols = {}  # kernel key -> first column
        self.width = {}  # kernel key -> columns
        self.pinned = set()  # kernel keys whose L2 buffers sit in their column
        self.signatures = {}  # op name -> how a call of its kernel maps to it

    def add(
        self,
        name,
        kernel,
        grid,
        args,
        constexprs=None,
        transform_script=None,
        slot=0,
        runtime=(),
        key=None,
        pin_l2=False,
        multiplex=(),
    ):
        """Add `kernel` at `args` as op `name`.

        slot: index of the device buffer the op's first buffer is bound to;
            the device buffers before it are passed but unused.
        runtime: int parameters kept as arguments of the op's sequence; the
            other ints are bound to their values in `args`.
        key: ops with the same key share herds and columns, so their herd
            bodies must match; defaults to the kernel's name.
        pin_l2: keep the op's L2 buffers whole in the memtile of its column
            (one-column kernels only), away from other ops' memtiles.
        multiplex: memory spaces ("L1", "L2", "L3") whose channels aircc
            time-multiplexes for this op.
        """
        g = tuple(grid) + (1,) * (3 - len(grid))
        with driver_scope("npu"), config_context(compile_only=True):
            c = kernel.warmup(*args, grid=g, **(constexprs or {}))
        with config_context(transform_tiling_script=transform_script):
            m = _ttshared_to_air(c.asm["ttsharedir"], *g)
        # The op's ints, bound in the launch: the herds take them as RTPs, so
        # every shape of a kernel has the same segment body.
        _bind_ints(m, kernel, args, keep=runtime)
        kept = _drop_unused_scalars(_entry(m))
        self.signatures[name] = _signature(
            kernel, g, args, constexprs, slot, runtime, kept, c
        )
        if slot:
            _shift_buffers(m, slot)
        if multiplex:
            _set_multiplexing(m, multiplex)
        key = key or kernel.fn.__name__
        if pin_l2:
            self.pinned.add(key)
        self.ops.append((name, key, str(m)))

    def add_air(self, name, air_text):
        """An op whose AIR is built elsewhere; its herds keep their columns."""
        self.ops.append((name, None, air_text))

    def _taken(self):
        """Columns the prebuilt ops' herds occupy."""
        from air.ir import Context, Module

        cols = set()
        with Context():
            for _, key, air in self.ops:
                if key is not None:
                    continue
                m = Module.parse(air)  # keep alive while its ops are read
                for h in _ops_named(m.operation, "air.herd"):
                    x = int(str(h.attributes["x_loc"]).split(":")[0])
                    w = _herd_width(h)
                    cols |= set(range(x, x + w))
        return cols

    def _place(self, key, module):
        """Give the kernel's herds columns of their own, side by side."""
        from air.ir import IntegerAttr, IntegerType, StringAttr

        i64 = IntegerType.get_signless(64)
        herds = []

        def walk(op):
            for region in op.regions:
                for block in region:
                    for o in block.operations:
                        if o.operation.name == "air.herd":
                            herds.append(o)
                        walk(o.operation)

        walk(module.operation)

        def size(h):
            return int(
                str(h.operation.operands[0].owner.attributes["value"]).split(":")[0]
            )

        # one entry per herd name: a herd split in prologue/body/epilogue
        # parts is one herd on one set of columns
        widths = {}
        for h in herds:
            widths.setdefault(str(h.operation.attributes["sym_name"]), size(h))
        if key not in self.cols:
            width = sum(widths.values())
            x = 0
            while self._used & set(range(x, x + width)):
                x += 1
            self.cols[key], self.width[key] = x, width
            self._used |= set(range(x, x + width))
        at, x = {}, self.cols[key]
        for n, w in widths.items():
            at[n], x = x, x + w
        for h in herds:
            n = str(h.operation.attributes["sym_name"])
            h.operation.attributes["x_loc"] = IntegerAttr.get(i64, at[n])
            h.operation.attributes["y_loc"] = IntegerAttr.get(i64, 2)
            h.operation.attributes["sym_name"] = StringAttr.get(f"{key}_{n[1:-1]}")
        if key in self.pinned:
            if any(w != 1 for w in widths.values()):
                raise ValueError(f"{key}: pin_l2 needs herds one column wide")
            _pin_l2(herds)

    def module_text(self):
        from air.ir import (
            Context,
            IntegerAttr,
            IntegerType,
            Location,
            Module,
            StringAttr,
        )

        texts = []
        self._used = self._taken()
        self.cols, self.width = {}, {}
        with Context(), Location.unknown():
            i64 = IntegerType.get_signless(64)
            # The device's kernel takes the buffers of the op whose sequence
            # comes first, so every op's func takes all of them.
            mods = [Module.parse(air) for _, _, air in self.ops]
            nbuf = self.nargs = max(_buffer_count(m) for m in mods)
            for (name, key, air), m in zip(self.ops, mods):
                if key is None:
                    if _buffer_count(m) != nbuf:
                        raise ValueError(f"{name} takes other buffers than the device")
                    texts.append(air)
                    continue
                if _buffer_count(m) < nbuf:
                    _shift_buffers(m, 0, after=nbuf - _buffer_count(m))
                for f in m.body.operations:
                    if (
                        f.operation.name == "func.func"
                        and "sym_visibility" not in f.attributes
                    ):
                        f.attributes["sym_name"] = StringAttr.get(name)
                self._place(key, m)
                for s in _ops_named(m.operation, "air.segment"):
                    s.attributes["sym_name"] = StringAttr.get(SEG)
                    s.attributes["x_loc"] = IntegerAttr.get(i64, 0)
                    s.attributes["y_loc"] = IntegerAttr.get(i64, 2)
                texts.append(str(m))
        return _link(texts)

    def compile(
        self,
        objects=(),
        runtime_loop_tiling_sizes=(4, 4),
        bf16_emulation=None,
        extra_args=(),
        stack_size=2048,
    ):
        npu = detect_npu_version()
        text = self.module_text()
        os.makedirs(os.path.join(self.proj, "air_project"), exist_ok=True)
        for o in map(os.path.abspath, objects):
            # link_with objects, found next to the cores' code
            for d in (self.proj, os.path.join(self.proj, "air_project")):
                dst = os.path.join(d, os.path.basename(o))
                if not os.path.exists(dst):
                    os.symlink(o, dst)
        path = os.path.join(self.proj, f"{self.name}.air.mlir")
        with open(path, "w") as f:
            f.write(text)
        cwd = os.getcwd()
        os.chdir(self.proj)
        try:
            out = _aircc_compile(
                path,
                "xclbin",
                npu,
                self.proj,
                bf16_emulation=bf16_emulation,
                extra_args=["--emit-txn-cpp", *extra_args],
                stack_size=stack_size,
                runtime_loop_tiling_sizes=runtime_loop_tiling_sizes,
            )
        finally:
            os.chdir(cwd)
        header = os.path.join(self.proj, "air_project", f"npu.{self.name}.air.txn.h")
        self._built = out["bin_path"], header
        return self._built

    def save(self, out_dir):
        """Put the compiled device into `out_dir` as `FusedProgram` loads it:
        `<name>.xclbin`, `<name>.txn.h` and `<name>.ops.json`, how a call of
        each op's kernel maps to the op."""
        import shutil

        xclbin, header = self._built
        shutil.copy(xclbin, os.path.join(out_dir, f"{self.name}.xclbin"))
        shutil.copy(header, os.path.join(out_dir, f"{self.name}.txn.h"))
        with open(os.path.join(out_dir, f"{self.name}.ops.json"), "w") as f:
            json.dump(dict(nargs=self.nargs, ops=self.signatures), f, indent=1)


def _herd_width(h):
    """The herd's x size, from the constant it is given."""
    owner = h.operands[0].owner
    if hasattr(owner, "attributes") and "value" in owner.attributes:
        return int(str(owner.attributes["value"]).split(":")[0])
    raise ValueError(f"herd {h.attributes['sym_name']}: size is not a constant")


def _entry(m):
    """The module's public func with a body: the kernel."""
    return next(
        op
        for op in m.body.operations
        if op.operation.name == "func.func"
        and len(op.regions[0].blocks)
        and "sym_visibility" not in op.attributes
    )


def _pin_l2(herds):
    """Keep each L2 buffer whole in the memtile of the column of the herds it
    is passed to."""
    from air.ir import IntegerAttr, IntegerType, MemRefType, OpResult, UnitAttr

    i32 = IntegerType.get_signless(32)
    for h in herds:
        x = IntegerAttr(h.operation.attributes["x_loc"]).value
        for v in h.operation.operands:
            if not isinstance(v, OpResult) or v.owner.name != "memref.alloc":
                continue
            space = MemRefType(v.type).memory_space
            if space is None or IntegerAttr(space).value != 1:  # L2
                continue
            attrs = v.owner.attributes
            if (
                "air.memtile_col" in attrs
                and IntegerAttr(attrs["air.memtile_col"]).value != x
            ):
                raise ValueError("an L2 buffer is passed to herds of two columns")
            attrs["air.memtile_col"] = IntegerAttr.get(i32, x)
            v.owner.attributes["air.no_split"] = UnitAttr.get()


def _drop_unused_scalars(fn):
    """Erase the func's scalar arguments nothing reads, so its runtime
    sequence takes only buffers and the scalars it still uses. Returns which
    arguments were kept."""
    from air.ir import ArrayAttr, FunctionType, MemRefType, TypeAttr

    block = fn.regions[0].blocks[0]
    keep = [
        isinstance(a.type, MemRefType) or bool(list(a.uses)) for a in block.arguments
    ]
    for i in reversed(range(len(keep))):
        if not keep[i]:
            block.erase_argument(i)
    old = FunctionType(TypeAttr(fn.attributes["function_type"]).value)
    with fn.context:
        fn.attributes["function_type"] = TypeAttr.get(
            FunctionType.get([a.type for a in block.arguments], old.results)
        )
        if "arg_attrs" in fn.attributes:
            attrs = ArrayAttr(fn.attributes["arg_attrs"])
            fn.attributes["arg_attrs"] = ArrayAttr.get(
                [a for a, k in zip(attrs, keep) if k]
            )
    return keep


def kernel_id(kernel):
    """A Triton kernel's name across processes: its module and name."""
    return f"{kernel.fn.__module__}.{kernel.fn.__qualname__}"


def canonical(v):
    """A bound argument as it is recorded and matched: JSON's types."""
    return v if isinstance(v, (bool, int, float, str)) else str(v)


def _signature(kernel, grid, args, constexprs, slot, runtime, kept, compiled):
    """How a call of `kernel` reaches the op: its pointer parameters' device
    buffers, the values bound into the op, its run-time parameters in the
    sequence's order, and the pointers the kernel writes."""
    values = dict(zip((p.name for p in kernel.params), args))
    values.update(constexprs or {})
    plain = [p.name for p in kernel.params if not p.is_constexpr]
    pointers = [n for n in plain if hasattr(values[n], "data_ptr")]
    # the func's arguments are the kernel's non-constexpr parameters in order,
    # then Triton's grid ints; `kept` says which survived
    named = plain + [None] * (len(kept) - len(plain))
    rt = [n for n, k in zip(named, kept) if k and n in runtime]
    written = written_args(compiled.asm["ttir"], plain)
    return dict(
        kernel=kernel_id(kernel),
        grid=list(grid),
        slots={n: slot + i for i, n in enumerate(pointers)},
        bound={
            n: canonical(v)
            for n, v in values.items()
            if n not in pointers and n not in runtime
        },
        runtime=rt,
        written=None if written is None else sorted(written),
    )


def _bind_ints(m, kernel, args, keep=()):
    """Replace the func's non-constexpr int and float arguments, except those
    named in `keep`, by this op's values."""
    from air.dialects import arith
    from air.ir import FloatAttr, InsertionPoint, IntegerAttr, IntegerType, Location

    unknown = set(keep) - {p.name for p in kernel.params}
    if unknown:
        raise ValueError(f"no parameters named {sorted(unknown)}")
    block = _entry(m).regions[0].blocks[0]
    with m.context, Location.unknown(m.context):
        ip = InsertionPoint.at_block_begin(block)
        i32 = IntegerType.get_signless(32)
        i = 0
        for p, v in zip(kernel.params, args):
            if p.is_constexpr:
                continue
            arg = block.arguments[i]
            i += 1
            if p.name in keep:
                continue
            if isinstance(v, int) and not isinstance(v, bool):
                c = arith.ConstantOp(i32, IntegerAttr.get(i32, v), ip=ip)
                arg.replace_all_uses_with(c.result)
            elif isinstance(v, float):
                c = arith.ConstantOp(arg.type, FloatAttr.get(arg.type, v), ip=ip)
                arg.replace_all_uses_with(c.result)


def _set_multiplexing(m, spaces):
    """Have aircc time-multiplex the func's channels in `spaces` too."""
    from air.ir import ArrayAttr, StringAttr

    with m.context:
        _entry(m).attributes["air.channel_multiplexing"] = ArrayAttr.get(
            [StringAttr.get(s) for s in spaces]
        )


def _buffer_count(m):
    from air.ir import UnrankedMemRefType, MemRefType

    block = _entry(m).regions[0].blocks[0]
    return sum(
        isinstance(a.type, (UnrankedMemRefType, MemRefType)) for a in block.arguments
    )


def _shift_buffers(m, n, after=0):
    """Put `n` unused buffer arguments ahead of the func's own buffers, and
    `after` more behind them, ahead of its scalars."""
    from air.ir import (
        ArrayAttr,
        DictAttr,
        FunctionType,
        Location,
        MemRefType,
        TypeAttr,
        UnrankedMemRefType,
    )

    fn = _entry(m)
    block = fn.regions[0].blocks[0]
    with m.context, Location.unknown(m.context):
        old = list(block.arguments)
        is_buf = [isinstance(a.type, (UnrankedMemRefType, MemRefType)) for a in old]
        nbuf = sum(is_buf)
        if nbuf == 0 or any(is_buf[nbuf:]):
            raise ValueError("expected buffer arguments ahead of the scalars")
        filler = old[0].type
        loc = Location.unknown(m.context)
        for _ in range(n):
            block.add_argument(filler, loc)
        for a in old[:nbuf]:
            a.replace_all_uses_with(block.add_argument(a.type, a.location))
        for _ in range(after):
            block.add_argument(filler, loc)
        for a in old[nbuf:]:
            a.replace_all_uses_with(block.add_argument(a.type, a.location))
        for _ in old:
            block.erase_argument(0)
        ftype = FunctionType(fn.attributes["function_type"].value)
        fn.attributes["function_type"] = TypeAttr.get(
            FunctionType.get([a.type for a in block.arguments], ftype.results)
        )
        if "arg_attrs" in fn.attributes:
            attrs = list(ArrayAttr(fn.attributes["arg_attrs"]))
            none = [DictAttr.get()]
            fn.attributes["arg_attrs"] = ArrayAttr.get(
                none * n + attrs[:nbuf] + none * after + attrs[nbuf:]
            )


def _ops_named(op, name):
    out = []
    for region in op.regions:
        for block in region:
            for o in block.operations:
                if o.operation.name == name:
                    out.append(o.operation)
                out += _ops_named(o.operation, name)
    return out


def _link(texts):
    """Merge modules into one. A symbol several modules declare (a kernel
    function, a shared private func) is kept once, and must be declared the
    same way in each."""
    from air.ir import Context, Location, Module

    with Context(), Location.unknown():
        mods = [Module.parse(t) for t in texts]
        base, seen = mods[0], {}

        def key(op):
            if "sym_name" in op.attributes:
                return str(op.attributes["sym_name"])

        for op in base.body.operations:
            seen[key(op)] = op.get_asm(print_generic_op_form=True)
        for m in mods[1:]:
            for op in list(m.body.operations):
                k, text = key(op), op.get_asm(print_generic_op_form=True)
                if k in seen:
                    if seen[k] != text:
                        raise ValueError(f"modules declare {k} differently")
                    continue
                seen[k] = text
                base.body.append(op)
        return str(base)
