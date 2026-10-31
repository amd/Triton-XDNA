# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dispatch of a device built as an xclbin with several runtime sequences.

A `Device` holds one hw_context on the xclbin. Each runtime sequence is
dispatched as a `Launch`: its instruction stream, from the compiler's TXN
builder header, and the kernel's buffer arguments. A sequence's run-time
arguments are patched into its stream when they change. `FusedProgram` is
built on these.
"""

import numpy as np


def _xrt():
    import pyxrt

    return pyxrt


def sync_to(bo, nbytes=None, offset=0):
    x = _xrt()
    d = x.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    bo.sync(d) if nbytes is None else bo.sync(d, nbytes, offset)


def sync_from(bo, nbytes=None, offset=0):
    x = _xrt()
    d = x.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    bo.sync(d) if nbytes is None else bo.sync(d, nbytes, offset)


def view(bo, dtype, shape):
    return np.frombuffer(bo.map(), dtype, count=int(np.prod(shape))).reshape(shape)


class Launch:
    """One runtime sequence of a device on a fixed set of buffers."""

    def __init__(self, device, seq, bos, rt=()):
        self.device, self.seq, self.bos = device, seq, list(bos)
        self.rt = None
        words = device.words(seq, *rt)
        x = _xrt()
        self.ib = x.bo(
            device.dev, words.nbytes, x.bo.cacheable, device.kern.group_id(1)
        )
        self.iv = view(self.ib, np.uint32, words.shape)
        self._set_words(words, tuple(rt))
        self.make()

    def _set_words(self, words, rt):
        if words.shape != self.iv.shape:
            raise ValueError(f"{self.seq}: the stream's length changed with {rt}")
        self.iv[:] = words
        sync_to(self.ib)
        self.rt = rt

    def make(self):
        x = _xrt()
        self.run = x.run(self.device.kern)
        for i, v in enumerate([3, self.ib, self.iv.size, *self.bos]):
            self.run.set_arg(i, v)

    def set_bo(self, arg, bo):
        self.bos[arg] = bo
        self.run.set_arg(3 + arg, bo)

    def set_rt(self, *rt):
        if tuple(rt) != self.rt:
            self._set_words(self.device.words(self.seq, *rt), tuple(rt))

    def start(self):
        self.run.start()

    def wait(self):
        st = self.run.wait()
        if "COMPLETED" not in str(st):
            raise RuntimeError(f"{self.seq}: {st}")


class Device:
    """An xclbin on its own hw_context, kernel MLIR_AIE with `nargs` buffer
    arguments. `txn_h` is the TXN builder header of its sequences."""

    def __init__(self, xclbin, nargs, txn_h):
        from air.backend.txn_builder import TxnBuilder

        x = _xrt()
        self.dev = x.device(0)
        self.xb = x.xclbin(str(xclbin))
        self.dev.register_xclbin(self.xb)
        self.nargs = nargs
        self.txn = TxnBuilder(str(txn_h))
        self.launches = []
        self.ctx = self.kern = None
        self.open()
        self.dummy = [self.bo(4096, i) for i in range(nargs)]

    def open(self):
        x = _xrt()
        self.ctx = x.hw_context(self.dev, self.xb.get_uuid())
        self.kern = x.kernel(self.ctx, "MLIR_AIE")
        for la in self.launches:
            la.make()

    def close(self):
        for la in self.launches:
            la.run = None
        self.kern = self.ctx = None

    def bo(self, nbytes, arg):
        """A host-only buffer for kernel argument `arg`."""
        x = _xrt()
        return x.bo(
            self.dev,
            max(int(nbytes), 4096),
            x.bo.host_only,
            self.kern.group_id(3 + arg),
        )

    def words(self, seq, *rt):
        return np.asarray(self.txn(f"seg_{seq}", *rt), np.uint32)

    def launch(self, seq, bos, rt=()):
        """Sequence `seq` on `bos` ({arg: bo}; the others get dummies)."""
        args = list(self.dummy)
        for i, b in bos.items():
            args[i] = b
        la = Launch(self, seq, args, rt)
        self.launches.append(la)
        return la


def sub_bo(bo, offset):
    """The part of `bo` from byte `offset` on, as a buffer of its own."""
    return _xrt().bo(bo, bo.size() - offset, offset)
