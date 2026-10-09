# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""The subset of pyxrt that mlir-air's fused prefill engine uses, on HSA.

That engine (`llms/shared/fused_prefill/engine.py`) is written against pyxrt:
one xclbin on a hardware context, host-only BOs, an instruction-stream BO per
op, and runs started and waited on. Assigning this module to the engine's
`xrt` global runs the same code on the HSA runtime instead:

    from air_examples.llms.shared.fused_prefill import engine
    from triton.backends.amd_triton_npu import hsa_xrt
    engine.xrt = hsa_xrt

How each piece maps:

* `bo` is a region from `triton_npu_hsa_shared_alloc`, so a dispatch runs on it
  in place, and `map()` is its host view. A dispatch flushes the regions it is
  given, which makes `sync()` unnecessary for correctness; it is used to decide
  residency instead. A BO synced to the device whole, and never synced back,
  is an input -- weights, mostly -- and is made resident, so later dispatches
  do not flush it again; a later `sync(TO)` marks it dirty for one flush. A
  BO that is ever synced back stays an ordinary region. A sub-BO is a range of
  its parent.
* An instruction-stream BO becomes an HSA program: the xclbin's PDI with that
  stream, prepared on first use and patched whenever the host has changed the
  stream since the last dispatch.
* `run.start()` queues the dispatch on one worker thread and `wait()` joins it,
  so host work between the two still overlaps the device. HSA dispatches are
  synchronous and serialised by the runtime.
* The PDI is extracted from the xclbin with `xclbinutil` and kept beside it.

Only what that engine calls is implemented.
"""

import concurrent.futures
import ctypes
import fcntl
import os
import subprocess
import tempfile
import weakref

import numpy as np


class xclBOSyncDirection:
    XCL_BO_SYNC_BO_TO_DEVICE = "XCL_BO_SYNC_BO_TO_DEVICE"
    XCL_BO_SYNC_BO_FROM_DEVICE = "XCL_BO_SYNC_BO_FROM_DEVICE"


class HsaXrtError(RuntimeError):
    pass


def _errbuf(n=1024):
    return ctypes.create_string_buffer(n), ctypes.c_size_t(n)


_lib = None


def _runtime():
    global _lib
    if _lib is None:
        from .driver import load_hsa_runtime

        _lib = load_hsa_runtime()
        _lib.triton_npu_hsa_shared_alloc.restype = ctypes.c_void_p
        _lib.triton_npu_hsa_prepare.restype = ctypes.c_void_p
    return _lib


def _check(rc, buf):
    if rc != 0:
        raise HsaXrtError(buf.value.decode())


def _xclbinutil():
    xrt = os.environ.get("XILINX_XRT", "/opt/xilinx/xrt")
    return os.path.join(xrt, "bin", "xclbinutil")


def extract_pdi(xclbin_path):
    """The PDI inside `xclbin_path`, extracted to `<xclbin>.pdi`.

    Extracted again whenever the xclbin is newer than the extracted PDI, so a
    rebuild in place is not paired with the old design.
    """
    out = xclbin_path + ".pdi"

    def current():
        return os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(
            xclbin_path
        )

    if current():
        return out
    with open(xclbin_path + ".pdi.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if current():
            return out
        # Beside the xclbin, so the result can be renamed into place.
        with tempfile.TemporaryDirectory(dir=os.path.dirname(xclbin_path)) as tmp:
            subprocess.run(
                [
                    _xclbinutil(),
                    "-i",
                    xclbin_path,
                    "--dump-section",
                    "AIE_PARTITION:JSON:" + os.path.join(tmp, "part.json"),
                ],
                check=True,
                capture_output=True,
            )
            pdis = [f for f in os.listdir(tmp) if f.endswith(".pdi")]
            if len(pdis) != 1:
                raise HsaXrtError(f"{xclbin_path}: expected one PDI, found {pdis}")
            os.replace(os.path.join(tmp, pdis[0]), out)
    return out


class device:
    def __init__(self, index=0):
        if index != 0:
            raise HsaXrtError("the HSA runtime has one AIE agent, index 0")
        _runtime()

    def register_xclbin(self, xb):
        pass


class xclbin:
    def __init__(self, path):
        self.path = str(path)

    def get_uuid(self):
        return self

    def pdi(self):
        return extract_pdi(self.path)


class hw_context:
    def __init__(self, dev, uuid):
        self.xclbin = uuid


class kernel:
    def __init__(self, ctx, name):
        self.ctx = ctx

    def group_id(self, i):
        return i


class _Region:
    """One shared allocation; freed when the last BO over it is collected."""

    def __init__(self, nbytes):
        lib = _runtime()
        buf, n = _errbuf()
        va = lib.triton_npu_hsa_shared_alloc(ctypes.c_uint64(nbytes), buf, n)
        if not va:
            raise HsaXrtError(buf.value.decode())
        self.va = va
        self.host = np.ctypeslib.as_array(
            ctypes.cast(va, ctypes.POINTER(ctypes.c_uint8)), shape=(nbytes,)
        )

        self.nbytes = nbytes
        self.resident = False
        self.read_back = False

        def free(_va=va, _lib=lib):
            b, m = _errbuf()
            _lib.triton_npu_hsa_shared_free(ctypes.c_void_p(_va), b, m)

        weakref.finalize(self, free)

    def make_resident(self):
        buf, n = _errbuf()
        _check(
            _runtime().triton_npu_hsa_shared_set_resident(
                ctypes.c_void_p(self.va), ctypes.c_int(1), buf, n
            ),
            buf,
        )
        self.resident = True

    def mark_dirty(self):
        buf, n = _errbuf()
        _check(
            _runtime().triton_npu_hsa_shared_mark_dirty(
                ctypes.c_void_p(self.va), buf, n
            ),
            buf,
        )


class bo:
    host_only = 0
    cacheable = 1
    normal = 2

    def __init__(self, *args):
        if isinstance(args[0], bo):
            parent, size, offset = args
            self._region = parent._region
            self._off = parent._off + int(offset)
            self._size = int(size)
            if self._off + self._size > parent._off + parent._size:
                raise HsaXrtError("sub-BO extends past its parent")
        else:
            size = int(args[1])
            self._region = _Region(size)
            self._off, self._size = 0, size
        self._program = None

    def size(self):
        return self._size

    def address(self):
        return self._region.va + self._off

    def map(self):
        return self._region.host[self._off : self._off + self._size]

    def write(self, data, offset=0):
        src = np.frombuffer(data, np.uint8)
        self.map()[offset : offset + src.size] = src

    def read(self, size, offset=0):
        return self.map()[offset : offset + size].copy()

    def sync(self, direction, size=None, offset=0):
        r = self._region
        if "FROM" in str(direction):
            if r.resident:
                raise HsaXrtError(
                    "a BO synced to the device whole was later synced back; it "
                    "was made resident, so the host may read stale lines"
                )
            r.read_back = True
            return
        whole = self._off == 0 and self._size == r.nbytes
        whole = whole and (size is None or (offset == 0 and size >= self._size))
        if r.resident:
            r.mark_dirty()
        elif whole and not r.read_back:
            r.make_resident()

    def _prepared(self, pdi, nwords):
        """This BO as the instruction stream of an HSA program over `pdi`."""
        words = np.frombuffer(self.map(), np.uint32, count=nwords)
        if self._program is None or self._program[0] != pdi:
            # The runtime reads the stream from a file once, at prepare.
            fd, path = tempfile.mkstemp(suffix=".insts.bin")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(words.tobytes())
                buf, n = _errbuf()
                handle = _runtime().triton_npu_hsa_prepare(
                    pdi.encode(), path.encode(), buf, n
                )
            finally:
                os.unlink(path)
            if not handle:
                raise HsaXrtError(buf.value.decode())
            self._program = (pdi, handle, words.copy())
            return handle
        _, handle, sent = self._program
        if not np.array_equal(sent, words):
            buf, n = _errbuf()
            _check(
                _runtime().triton_npu_hsa_patch_insts(
                    ctypes.c_void_p(handle),
                    ctypes.c_uint64(0),
                    words.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_uint64(words.nbytes),
                    buf,
                    n,
                ),
                buf,
            )
            sent[:] = words
        return handle


_worker = concurrent.futures.ThreadPoolExecutor(max_workers=1)


class run:
    def __init__(self, kern):
        self.kern = kern
        self.args = {}
        self._pending = None

    def set_arg(self, i, value):
        self.args[i] = value

    def _dispatch(self):
        ib, nwords = self.args[1], int(self.args[2])
        bos = [self.args[i] for i in sorted(self.args) if i >= 3]
        handle = ib._prepared(self.kern.ctx.xclbin.pdi(), nwords)
        n = len(bos)
        ptrs = (ctypes.c_void_p * n)(*[b.address() for b in bos])
        sizes = (ctypes.c_uint64 * n)(*[b.size() for b in bos])
        buf, nb = _errbuf()
        _check(
            _runtime().triton_npu_hsa_dispatch(
                ctypes.c_void_p(handle), ctypes.c_uint32(n), ptrs, sizes, buf, nb
            ),
            buf,
        )

    def start(self):
        self._pending = _worker.submit(self._dispatch)

    def wait(self, timeout_ms=None):
        if self._pending is not None:
            self._pending.result()
            self._pending = None
        return "ert_cmd_state.ERT_CMD_STATE_COMPLETED"
