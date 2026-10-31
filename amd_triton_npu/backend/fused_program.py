# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Calling the ops of a fused device as Triton kernels.

A `FusedDevice` build holds several Triton kernels, each at one set of bound
arguments, as runtime sequences of one xclbin, and records how a call of each
kernel maps to its op (`<name>.ops.json`). `FusedProgram` loads such a build.
Inside `with program:` a call

    mm_g[grid](a, w, c, n, k, ...)

finds the op built for those bound arguments, binds each tensor argument's
buffer to the op's device buffer, puts the op's run-time arguments into its
instruction stream and starts it, without waiting. Its tensors are `Buffer`s
from `program.empty` / `program.zeros`.

The host and the device see a buffer through caches that do not snoop each
other, so a buffer tracks who wrote it last:
- a device run waits for the runs it depends on and first syncs to the device
  whatever the host wrote;
- before the host reads a buffer the device writes, it waits for the run and
  syncs the run's output back. Host Triton kernels do so on their own when
  launched under `driver_scope("cpu")`; other host code asks with
  `Buffer.host()`.
What each kernel writes comes from its TTIR (`arg_access`).
"""

import functools
import json
from pathlib import Path

import numpy as np

from .arg_access import written_args
from .npu_runner import Device, sub_bo, sync_from, sync_to

#: Arguments Triton passes a launcher ahead of the kernel's own.
_LAUNCH_PREFIX = 9


class _Storage:
    """One device buffer and what is pending on it."""

    def __init__(self, device, nbytes):
        self.bo = device.bo(nbytes, 0)
        self.nbytes = nbytes
        self.addr = np.frombuffer(self.bo.map(), np.uint8).ctypes.data
        self.subs = {}  # offset -> sub-buffer starting there
        self.host_dirty = []  # byte ranges the host wrote since the last sync
        self.dev_dirty = []  # byte ranges a device run wrote since
        self.runs = []  # (run, writes, lo, hi) in flight on it

    def sub(self, offset):
        if offset == 0:
            return self.bo
        if offset not in self.subs:
            self.subs[offset] = sub_bo(self.bo, offset)
        return self.subs[offset]

    def wait(self, lo, hi, writers_only):
        """Wait for the runs on bytes [lo, hi) of the buffer: the ones writing
        them, or all."""
        keep = []
        for r in self.runs:
            run, writes, a, b = r
            if a < hi and lo < b and (writes or not writers_only):
                run.wait()
            elif not run.done:
                keep.append(r)
        self.runs = keep

    def add_run(self, run, writes, lo, hi):
        self.runs.append((run, writes, lo, hi))
        if len(self.runs) > 64:
            # a buffer the host never touches is never waited on itself;
            # its runs are done once waited for through other buffers
            self.runs = [r for r in self.runs if not r[0].done]


class Buffer:
    """A device buffer as a Triton argument: a contiguous array mapped on the
    host. Indexing the first dimension with a slice gives a view, which a
    kernel sees as a buffer starting at the view."""

    def __init__(self, storage, offset, shape, dtype):
        self.storage, self.offset = storage, offset
        self.shape, self.np_dtype = tuple(shape), _np_dtype(dtype)
        self.itemsize = self.np_dtype.itemsize
        strides, n = [], self.itemsize
        for d in reversed(self.shape):
            strides.append(n)
            n *= d
        self.strides = tuple(reversed(strides))
        self.nbytes = n
        self.size = n // self.itemsize
        self.lo, self.hi = offset, offset + n
        # what Triton specializes a pointer on
        self.dtype = self.np_dtype.name
        self._views = {}

    def data_ptr(self):
        return self.storage.addr + self.offset

    def __getitem__(self, rows):
        if not isinstance(rows, slice) or rows.step not in (None, 1):
            raise TypeError("a Buffer is sliced along its first dimension only")
        r0, r1, _ = rows.indices(self.shape[0])
        row = self.strides[0] if len(self.shape) > 1 else self.itemsize
        shape = (max(r1 - r0, 0), *self.shape[1:])
        return Buffer(self.storage, self.offset + r0 * row, shape, self.np_dtype)

    def view(self, dtype):
        """The same bytes as `dtype` (numpy's or torch's), the last
        dimension rescaled."""
        v = self._views.get(dtype)
        if v is None:
            dt = _np_dtype(dtype)
            last = self.shape[-1] * self.itemsize // dt.itemsize
            v = Buffer(self.storage, self.offset, (*self.shape[:-1], last), dt)
            self._views[dtype] = v
        return v

    def reshape(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        if -1 in shape:
            known = int(np.prod([d for d in shape if d != -1]))
            shape = tuple(self.size // known if d == -1 else d for d in shape)
        if int(np.prod(shape)) != self.size:
            raise ValueError(f"cannot reshape {self.shape} to {shape}")
        return Buffer(self.storage, self.offset, shape, self.np_dtype)

    def host(self, write=False):
        """The buffer as a numpy array the host can read, and write if
        `write`: waits for the device runs on it and syncs back what they
        wrote."""
        _acquire(self, write)
        a = np.frombuffer(
            self.storage.bo.map(), self.np_dtype, self.size, self.offset
        ).reshape(self.shape)
        if write:
            self.storage.host_dirty.append((self.lo, self.hi))
        return a


_DTYPES = {}


def _np_dtype(dtype):
    """numpy's dtype for a numpy or torch dtype."""
    dt = _DTYPES.get(dtype)
    if dt is None:
        try:
            dt = np.dtype(dtype)
        except TypeError:
            import ml_dtypes  # noqa: F401  registers bfloat16 with numpy

            dt = np.dtype(str(dtype).split(".")[-1])
        _DTYPES[dtype] = dt
    return dt


def _acquire(buf, write):
    """Make `buf` safe for the host to read (and write, if `write`)."""
    s = buf.storage
    if s.runs:
        s.wait(buf.lo, buf.hi, writers_only=not write)
    if s.dev_dirty:
        keep = []
        for lo, hi in s.dev_dirty:
            if lo < buf.hi and buf.lo < hi:
                sync_from(s.bo, hi - lo, lo)
            else:
                keep.append((lo, hi))
        s.dev_dirty = keep


class _Run:
    """One start of an op's sequence."""

    def __init__(self, launch):
        self.launch, self.done = launch, False

    def wait(self):
        if not self.done:
            self.launch.wait()
            self.done = True


class _Op:
    def __init__(self, name, sig):
        self.name = name
        self.slots = sig["slots"]
        self.runtime = sig["runtime"]
        self.written = sig["written"]
        self.idle = []  # launches whose last run has been waited for
        self.busy = []  # (launch, run)

    def launch(self, program, rt):
        """A launch of this op whose last run has been waited for, else a new
        one, which starts with the run-time values `rt`. Runs are known to be
        done only once waited for, which the host does before it reads their
        output, so the launches grow to as many as are in flight at once."""
        if not self.idle and self.busy:
            still = []
            for la, run in self.busy:
                if run.done:
                    self.idle.append(la)
                else:
                    still.append((la, run))
            self.busy = still
        if self.idle:
            return self.idle.pop()
        return program.device.launch(self.name, {}, rt=rt)


class FusedProgram:
    """A `FusedDevice` build (`<dir>/<name>.xclbin`, `.txn.h`, `.ops.json`)
    on its own hardware context."""

    def __init__(self, build_dir, name):
        d = Path(build_dir)
        spec = json.loads((d / f"{name}.ops.json").read_text())
        self.device = Device(
            d / f"{name}.xclbin", spec["nargs"], txn_h=d / f"{name}.txn.h"
        )
        self.ops = {}  # kernel id -> {bound names: {bound values: _Op}}
        for op, sig in spec["ops"].items():
            names = tuple(sorted(sig["bound"]))
            key = tuple(sig["bound"][n] for n in names)
            self.ops.setdefault(sig["kernel"], {}).setdefault(names, {})[key] = _Op(
                op, sig
            )
        self.driver = _driver(self)

    def empty(self, shape, dtype):
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        nbytes = int(np.prod(shape)) * _np_dtype(dtype).itemsize
        return Buffer(_Storage(self.device, nbytes), 0, shape, dtype)

    def zeros(self, shape, dtype):
        b = self.empty(shape, dtype)
        b.host(write=True)[...] = 0
        return b

    def contexts(self):
        return [self.device]

    def __enter__(self):
        import triton

        cfg = triton.runtime.driver
        self._prev = getattr(cfg, "_active", None)
        cfg.set_active(self.driver)
        return self

    def __exit__(self, *exc):
        import triton

        triton.runtime.driver._active = self._prev

    def find(self, kernel, values):
        """The op for a call of `kernel` with `values` ({name: value})."""
        from .fused_device import canonical, kernel_id

        table = self.ops.get(kernel_id(kernel))
        if table is None:
            raise KeyError(f"{kernel_id(kernel)}: no op of this device runs it")
        for names, ops in table.items():
            key = tuple(canonical(values[n]) for n in names)
            if key in ops:
                return ops[key]
        raise KeyError(
            f"{kernel_id(kernel)}: no op of this device is built for "
            f"{ {n: values[n] for n in next(iter(table))} }"
        )

    def plan(self, kernel, args):
        """How a call of `kernel` with `args` (all its parameters, in order)
        runs: its op, then per device buffer the argument's index, the slot
        and whether the op writes it, then the run-time arguments' indices."""
        names = kernel.arg_names
        op = self.find(kernel, dict(zip(names, args)))
        bufs = []
        for name, slot in op.slots.items():
            i = names.index(name)
            if not isinstance(args[i], Buffer):
                raise TypeError(f"{op.name}: {name} is not a Buffer of this program")
            bufs.append((i, slot, op.written is None or name in op.written))
        return op, bufs, [names.index(n) for n in op.runtime]

    def run(self, plan, args):
        """Start a planned call on its buffers and run-time values."""
        op, slots, rts = plan
        rt = tuple(int(args[i]) for i in rts)
        bufs = [args[i] for i, _, _ in slots]
        la = op.launch(self, rt)
        for buf, (_, slot, writes) in zip(bufs, slots):
            s = buf.storage
            if s.runs:
                s.wait(buf.lo, buf.hi, writers_only=not writes)
            if s.host_dirty:
                for lo, hi in s.host_dirty:
                    sync_to(s.bo, hi - lo, lo)
                s.host_dirty = []
            bo = s.sub(buf.offset) if buf.offset else s.bo
            if la.bos[slot] is not bo:
                la.set_bo(slot, bo)
        if rt:
            la.set_rt(*rt)
        la.start()
        run = _Run(la)
        op.busy.append((la, run))
        for buf, (_, _, writes) in zip(bufs, slots):
            s = buf.storage
            s.add_run(run, writes, buf.lo, buf.hi)
            if writes:
                s.dev_dirty.append((buf.lo, buf.hi))
        return run


class _Launcher:
    def __init__(self, program, src, metadata):
        self.program, self.kernel = program, src.fn
        # constexprs are fixed per compiled kernel, so they need no matching
        self.plain = [i for i, p in enumerate(src.fn.params) if not p.is_constexpr]
        self.plans = {}  # the call's scalars -> its plan

    def __call__(self, *args):
        args = args[_LAUNCH_PREFIX:]
        key = tuple(
            a for a in (args[i] for i in self.plain) if not isinstance(a, Buffer)
        )
        plan = self.plans.get(key)
        if plan is None:
            plan = self.plans[key] = self.program.plan(self.kernel, args)
        self.program.run(plan, args)


class _Utils:
    """The NPU driver's utils, with a module for each loaded kernel: Triton
    sets a kernel's launcher up again on every call while its module is None,
    and a launcher keeps the plans of the calls it has seen."""

    def __init__(self, utils):
        self._utils = utils

    def __getattr__(self, name):
        return getattr(self._utils, name)

    def load_binary(self, *args):
        _, *rest = self._utils.load_binary(*args)
        return (True, *rest)


def _driver(program):
    """The NPU driver, launching on `program`'s ops."""
    from .driver import NPUDriver

    d = NPUDriver()
    d.launcher_cls = functools.partial(_Launcher, program)
    d.utils = _Utils(d.utils)
    # Triton caches compiled kernels, and their launchers, per device; a
    # device of its own keeps a call from reaching another program's ops
    device = f"npu:fused:{id(program)}"
    d.get_current_device = lambda: device
    return d


# ---- host kernels ----------------------------------------------------------


def _host_written(metadata, names):
    """The arguments a host kernel writes, from its TTIR in Triton's cache."""
    from triton.runtime.cache import get_cache_manager

    path = get_cache_manager(metadata.hash).get_file(f"{metadata.name}.ttir")
    if path is None:
        return None
    return written_args(Path(path).read_text(), names)


def host_launcher(inner_cls):
    """A launcher class around a CPU backend's: a launch with `Buffer`
    arguments first makes them safe for the host (`Buffer.host`), and marks
    the ones the kernel writes as written by the host."""

    class _HostLauncher:
        def __init__(self, src, metadata):
            self.inner = inner_cls(src, metadata)
            plain = [p.name for p in src.fn.params if not p.is_constexpr]
            written = _host_written(metadata, plain)
            # where an argument can be a buffer, and whether the kernel
            # writes through it
            self.ptrs = [
                (_LAUNCH_PREFIX + i, written is None or p.name in written)
                for i, p in enumerate(src.fn.params)
                if not p.is_constexpr
            ]

        def __call__(self, *args):
            bufs = [(args[i], w) for i, w in self.ptrs if type(args[i]) is Buffer]
            for b, w in bufs:
                _acquire(b, w)
            self.inner(*args)
            for b, w in bufs:
                if w:
                    b.storage.host_dirty.append((b.lo, b.hi))

    return _HostLauncher
