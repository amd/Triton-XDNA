# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Superkernels (see `tl.extra.npu.spatial`) from their TTIR.

The TTIR of a superkernel is its function with the loop over the waves, the
functions of its partitions, the kernels they call, and one declaration per
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

from .superkernel import Superkernel, Tensor, _Scalar, _Table, _Wave

#: the core array npu2 places partitions on: columns, then the core rows
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
    return str(op.attributes["callee"]).lstrip("@")


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

_CASTS = ("arith.extsi", "arith.extui", "arith.trunci", "arith.index_cast")


def _evaluate(op, env, what):
    """The value of the one result of scalar int op `op`, given its operands'
    values in `env`: an int, a _PerWave, or the operand's own value."""
    from air.ir import IntegerAttr

    name = op.operation.name
    if name == "arith.constant":
        return IntegerAttr(op.attributes["value"]).value
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
            f"superkernels place partitions on npu2 only; give `at` for every "
            f"partition on {npu}"
        )
    return _ARRAY[npu]


def _place(parts, npu):
    """The corner of each partition: its own `at`, or the first free spot
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
                raise ValueError(f"partition {p['name']} at {p['at']} overlaps another")
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
            raise ValueError(f"no room on the array for partition {p['name']}")
        taken |= cells(spot, p["cores"])
        corners.append(spot)
    return corners


def _leaf_module(module, symbol, name):
    """The TTIR of the kernel `symbol` of `module` as a module of its own: a
    public function `name` with the functions it calls inlined, run through
    the backend's TTIR passes like a kernel compiled alone."""
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
    return str(mod)


def _walk(op):
    """The ops nested in `op`, in order."""
    out = []
    for r in op.regions:
        for b in r.blocks:
            for o in b.operations:
                out.append(o)
                out += _walk(o)
    return out


def read(ttir, air_project_path):
    """The `Superkernel` of the superkernel whose TTIR is `ttir`, and the
    kind of each argument of its function: "buffer" or "scalar"."""
    from air.ir import Context, Location, Module

    from .driver import detect_npu_version

    with Context() as ctx, Location.unknown():
        ctx.allow_unregistered_dialects = True
        module = Module.parse(_generic(ttir))
        funcs = {_attr(f, "sym_name"): f for f in module.body.operations}
        top = next(f for f in funcs.values() if _attr(f, "sym_visibility") == "public")
        name = _attr(top, "sym_name")
        body = top.regions[0].blocks[0]
        loops = [o for o in body.operations if o.operation.name == "scf.for"]
        if len(loops) != 1:
            raise ValueError(f"{name}: a superkernel loops over its waves once")
        loop = loops[0]
        env = {}
        kinds = [
            "buffer" if "!tt.ptr" in str(a.type) else "scalar" for a in body.arguments
        ]
        for o in body.operations:
            n = o.operation.name
            if o == loop or n == "tt.return":
                continue
            if n == "tt.call" and _attr(funcs[_callee(o)], "npu.l2_elements"):
                continue
            if n.startswith("arith."):
                env[o.result] = _evaluate(o, env, name)
                continue
            raise ValueError(f"{name}: {n} outside the loop over the waves")
        lb, ub, step = (env.get(v) for v in loop.operands[:3])
        if not all(isinstance(x, int) for x in (lb, ub, step)):
            raise ValueError(
                f"{name}: the loop over the waves needs a constexpr trip count"
            )
        waves = list(range(lb, ub, step))
        sk = Superkernel(name, air_project_path, len(waves))
        for i, a in enumerate(body.arguments):
            if kinds[i] == "buffer":
                elt = str(a.type).split("<")[1].rstrip(">")
                env[a] = sk.ddr(f"a{i}", 0, elt)
            else:
                env[a] = sk.scalar(f"r{i}")
        for o in body.operations:
            if o.operation.name == "tt.call":
                decl = funcs[_callee(o)]
                n = _attr(decl, "npu.l2_elements")
                if n:
                    elt = str(o.results[0].type).split("<")[1].rstrip(">")
                    env[o.results[0]] = sk.onchip(_callee(o), n, elt)
        lbody = loop.regions[0].blocks[0]
        env[lbody.arguments[0]] = _PerWave(waves)

        parts = []
        for o in lbody.operations:
            n = o.operation.name
            if n == "scf.yield":
                continue
            if n.startswith("arith."):
                env[o.result] = _evaluate(o, env, name)
                continue
            if n != "tt.call" or _attr(funcs[_callee(o)], "npu.partition") is None:
                raise ValueError(f"{name}: {n} in the wave loop outside npu.spatial")
            parts.append((o, funcs[_callee(o)]))
        if len({_attr(f, "npu.step") for _, f in parts}) > 1:
            raise ValueError(f"{name}: a wave runs one npu.spatial")
        if not parts:
            raise ValueError(f"{name}: the wave loop runs no npu.spatial")

        settings = [_settings(f) for _, f in parts]
        corners = _place(
            settings,
            detect_npu_version() if any(p["at"] is None for p in settings) else None,
        )

        sched_dir = os.path.join(sk.proj, "schedules")
        os.makedirs(sched_dir, exist_ok=True)
        leaves = {}
        for (call, f), p, at in zip(parts, settings, corners):
            schedule = _bound_schedule(p, sched_dir)
            for sym, args in _kernel_calls(call, f, funcs, env, p["name"]):
                kname = _kernel_name(sym)
                if sym not in leaves:
                    leaves[sym] = _leaf_module(module, sym, kname)
                vals = []
                for v in args:
                    if isinstance(v, _PerWave):
                        v = v.lowered()
                    if not isinstance(v, (Tensor, _Scalar, _Wave, int)):
                        raise ValueError(f"{p['name']}: {kname} takes {v!r}")
                    vals.append(v)
                sk.call(kname, leaves[sym], vals, at, p["name"], schedule, p["cores"])
        return sk, kinds


def _settings(f):
    """The settings `spatial` gave partition function `f`."""
    at = None
    if _attr(f, "npu.at_col") is not None:
        at = (_attr(f, "npu.at_col"), _attr(f, "npu.at_row"))
    return {
        "name": _attr(f, "npu.name"),
        "cores": (_attr(f, "npu.cores_x"), _attr(f, "npu.cores_y")),
        "at": at,
        "schedule": _attr(f, "npu.schedule"),
    }


def _bound_schedule(p, sched_dir):
    """The path of partition `p`'s schedule with its core counts bound."""
    from .transform_inject import _bind_core_counts

    with open(p["schedule"]) as f:
        text = f.read()
    try:
        text = _bind_core_counts(text, p["cores"])
    except ValueError as e:
        raise ValueError(f"partition {p['name']}: {e}") from None
    path = os.path.join(sched_dir, f"{p['name']}.mlir")
    with open(path, "w") as f:
        f.write(text)
    return path


def _kernel_calls(call, f, funcs, env, what):
    """The kernels partition function `f`, called by `call`, runs, in order:
    (function symbol, argument values). A partition that is a kernel runs
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
        self.built = None
        self.device = None

    def _build(self, ttir):
        import json
        import re

        from triton import knobs

        from .config import npu_config
        from .driver import detect_npu_version

        key = hashlib.sha256(ttir.encode())
        for path in sorted(set(re.findall(r'npu\.schedule = "([^"]*)"', ttir))):
            with open(path, "rb") as f:
                key.update(f.read())
        key.update(f"{detect_npu_version()} {npu_config.bf16_emulation}".encode())
        proj = os.path.join(knobs.cache.dir, "npu_superkernel", key.hexdigest()[:32])
        sk, kinds = read(ttir, proj)
        done = os.path.join(proj, "built.json")
        if os.path.exists(done):
            with open(done) as f:
                xclbin, header = json.load(f)
        else:
            xclbin, header = sk.compile(
                runtime_loop_tiling_sizes=(),
                bf16_emulation=npu_config.bf16_emulation,
                extra_args=(
                    "--air-channel-multiplexing=L1",
                    "--omit-ping-pong-transform=L1",
                ),
            )
            with open(done, "w") as f:
                json.dump([xclbin, header], f)
        written = [any(t in c.written for c in sk.calls) for t in sk.buffers]
        return sk.name, kinds, xclbin, header, written

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):
        import numpy as np
        import torch

        from .config import npu_config
        from .npu_runner import Device, sync_from, sync_to

        if self.built is None:
            ttir = function.decode() if isinstance(function, bytes) else function
            self.built = self._build(ttir)
        name, kinds, xclbin, header, written = self.built
        if npu_config.compile_only:
            return
        # Triton passes the launch metadata and hooks, then every argument.
        values = [a for a, t in zip(args[4:], self.signature) if t != "constexpr"]
        bufs = [v for v, k in zip(values, kinds) if k == "buffer"]
        rt = [int(v) for v, k in zip(values, kinds) if k == "scalar"]
        if self.device is None:
            self.device = Device(xclbin, len(bufs), header)
        bos = {}
        for i, t in enumerate(bufs):
            if not t.is_contiguous():
                raise ValueError(f"{name}: buffer argument {i} is not contiguous")
            raw = t.view(-1).view(torch.uint8).numpy()
            bos[i] = self.device.bo(raw.nbytes, i)
            np.frombuffer(bos[i].map(), np.uint8, raw.nbytes)[:] = raw
            sync_to(bos[i])
        run = self.device.launch(name, bos, rt=rt)
        run.start()
        run.wait()
        for i, t in enumerate(bufs):
            if written[i]:
                sync_from(bos[i])
                raw = t.view(-1).view(torch.uint8).numpy()
                raw[:] = np.frombuffer(bos[i].map(), np.uint8, raw.nbytes)
