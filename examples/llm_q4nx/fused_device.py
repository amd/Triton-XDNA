# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Several Triton kernels and prebuilt AIR modules on one NPU device.

Each op is one func whose segment is named like every other op's, so aircc
builds one device with a runtime sequence per op (an xclbin and a TXN builder
header). Each kernel's herds get columns of their own, next to the columns the
prebuilt modules' herds already use.
"""

import os

from triton.backends.amd_triton_npu.config import config_context
from triton.backends.amd_triton_npu.driver import (
    _aircc_compile,
    _ttshared_to_air,
    detect_npu_version,
)
from triton.backends.amd_triton_npu.driver_scope import driver_scope

SEG = "seg"


class FusedDevice:
    """Ops added with `add` (a Triton kernel at one set of arguments) or
    `add_air` (AIR built elsewhere), then compiled together by `compile`."""

    def __init__(self, name, air_project_path):
        self.name, self.proj = name, os.path.abspath(air_project_path)
        self.ops = []  # (op name, kernel key, air text)
        self.cols = {}  # kernel key -> first column
        self.width = {}  # kernel key -> columns

    def add(self, name, kernel, grid, args, constexprs=None, transform_script=None):
        g = tuple(grid) + (1,) * (3 - len(grid))
        with driver_scope("npu"), config_context(compile_only=True):
            c = kernel.warmup(*args, grid=g, **(constexprs or {}))
        with config_context(transform_tiling_script=transform_script):
            m = _ttshared_to_air(c.asm["ttsharedir"], *g)
        # The op's run-time ints, bound in the launch: the herds take them as
        # RTPs, so every shape of a kernel has the same segment body.
        _bind_ints(m, kernel, args)
        _drop_unused_scalars(_entry(m))
        self.ops.append((name, kernel.fn.__name__, str(m)))

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
            for name, key, air in self.ops:
                if key is None:
                    texts.append(air)
                    continue
                m = Module.parse(air)
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
        return out["bin_path"], header


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


def _drop_unused_scalars(fn):
    """Erase the func's scalar arguments nothing reads, so its runtime
    sequence takes only buffers and the scalars it still uses."""
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


def _bind_ints(m, kernel, args):
    """Replace the func's non-constexpr int arguments by this op's values."""
    from air.dialects import arith
    from air.ir import FloatAttr, InsertionPoint, IntegerAttr, IntegerType, Location

    block = _entry(m).regions[0].blocks[0]
    with m.context, Location.unknown(m.context):
        ip = InsertionPoint.at_block_begin(block)
        i32 = IntegerType.get_signless(32)
        i = 0
        for p, v in zip(kernel.params, args):
            if p.is_constexpr:
                continue
            arg = block.arguments[i]
            if isinstance(v, int) and not isinstance(v, bool):
                c = arith.ConstantOp(i32, IntegerAttr.get(i32, v), ip=ip)
                arg.replace_all_uses_with(c.result)
            elif isinstance(v, float):
                c = arith.ConstantOp(arg.type, FloatAttr.get(arg.type, v), ip=ip)
                arg.replace_all_uses_with(c.result)
            i += 1


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
    import airsrc

    airsrc.register_air_examples()
    from air_examples.llms.shared.fused_prefill.build import _link

    return _link(texts)
