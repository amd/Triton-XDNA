#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""End-to-end check of the multi-device shared buffers in ``shared``.

Covers every combination the API admits:

  * each device kind as primary (the NPU and the iGPU), with an empty secondary
    list and with the other kind as secondary;
  * ``share_with`` / ``is_shared_with`` after construction, including the
    idempotent and closed-buffer cases;
  * device-spec parsing -- device strings, the ``(kind, handle)`` pair, case
    folding, and what a typo does;
  * the ``shared`` module's torch-shaped factories, including that ``device``
    has no default;
  * the DLPack protocol on a buffer that *has* an iGPU mapping and on one that
    does not, since those describe themselves as different kinds of memory;
  * a real Triton-compiled NPU kernel dispatched on shared buffers, once with
    the NPU owning the pages and once with the iGPU owning them -- the second
    is the import direction, which is the part that is easy to get subtly
    wrong;
  * a clean interpreter exit -- an early ctypes-callback DLPack deleter
    segfaulted at shutdown, which is why the producer is compiled into the
    plugin and its deleter is nanobind's rather than a Python callback.

Which NPU runtime, and why it is one per run
--------------------------------------------
The NPU can be reached through XRT or through HSA/ROCR, and a buffer names one
or the other: they are the same device, and neither runtime can map the other's
pages. So this file runs against whichever one is selected --
``AMD_TRITON_NPU_RUNTIME``, defaulting to ``xrt`` -- and reports the other
runtime's sections as SKIP. Run it twice to cover both::

    source scripts/dev-env.sh
    python examples/npu_gpu_dlpack/shared_buffer_test.py
    AMD_TRITON_NPU_RUNTIME=hsa python examples/npu_gpu_dlpack/shared_buffer_test.py

One process *can* drive both -- an XRT dispatch and an HSA one, in either
order, work in the same process, on separate buffers. They are kept apart here
so that a failure names a runtime rather than a combination.

Exit status is the result, so this doubles as a regression test. A selected
runtime that is unavailable is a failure, not a skip -- asking for it is what
selects it.
"""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Callable, Sequence
from typing import Any

import torch
import triton
import triton.language as tl

from triton.backends.amd_triton_npu.config import npu_config
from triton.backends.amd_triton_npu.driver import NPUDriver, detect_npu_version
from triton.backends.amd_triton_npu.shared import SharedBuffer, SharedBufferError

try:
    import pyxrt
except ImportError:  # an HSA-only host needs no XRT at all
    pyxrt = None

M, D = 128, 768
N = M * D

#: Which runtime this run drives the NPU through, and so which NPU device the
#: buffers below name. The canonical spelling, which is also what devices/device
#: report back, so these double as expected values.
RUNTIME = npu_config.runtime
NPU = f"{RUNTIME}:0"
OTHER_NPU = "hsa:0" if RUNTIME == "xrt" else "xrt:0"
HIP = "hip:0"

# The dispatch section compiles a kernel, which needs the driver bound to the
# runtime we are testing. Done here rather than inside that section so every
# section runs against the same driver, as a real program would.
triton.runtime.driver.set_active(NPUDriver(RUNTIME))

_failures: list[str] = []
_skipped: list[str] = []


class _Buffers:
    """Tracks the buffers a test allocates so they can be released together.

    Not a general-purpose cache: the test just needs every buffer closed before
    the process exits, since a clean shutdown is one of the things checked.
    """

    def __init__(self) -> None:
        self._all: list[SharedBuffer] = []

    def new(
        self,
        shape: Sequence[int],
        dtype: torch.dtype,
        device: Any,
        share: Any = (),
    ) -> SharedBuffer:
        """Allocate a shared buffer and keep it for close_all()."""
        return self.adopt(SharedBuffer(shape, dtype, device, share))

    def adopt(self, buf: SharedBuffer) -> SharedBuffer:
        """Take responsibility for a buffer allocated elsewhere (a factory)."""
        self._all.append(buf)
        return buf

    def close_all(self) -> None:
        """Release every buffer handed out, newest first."""
        for buf in reversed(self._all):
            buf.close()
        self._all.clear()


def check(name: str, ok: bool, detail: str = "") -> bool:
    """Record and print one assertion; returns whether it passed."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        _failures.append(name)
    return ok


def skip(name: str, reason: str) -> None:
    """Record a section this run cannot cover, without failing it.

    Kept out of ``_failures`` on purpose: the exit status is what the test
    harness reads, and "this runtime was not the one selected" is not a defect.
    Printed and summarised so a skipped run is never mistaken for a full one.
    """
    print(f"  [SKIP] {name} -- {reason}")
    _skipped.append(name)


def check_raises(
    name: str,
    exc: type[BaseException],
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> bool:
    """Assert that ``fn`` raises ``exc``; reports the message when it does."""
    try:
        fn(*args, **kwargs)
    except exc as e:
        return check(name, True, str(e))
    except Exception as e:  # wrong type is as much a failure as no raise
        return check(name, False, f"raised {type(e).__name__}: {e}")
    return check(name, False, "nothing raised")


# ---------------------------------------------------------------------------
# Device specs
# ---------------------------------------------------------------------------
def test_device_specs(buffers: _Buffers) -> None:
    """How a device is named: strings, pair form, case, and typos."""
    print("\n-- device specs --")

    # Every spelling of "NPU device 0" must resolve to the same device, so a
    # buffer named one way is recognised when named another.
    kind = RUNTIME.upper()
    same: list[tuple[str, Any]] = [
        ("device string", NPU),
        ("bare kind string", kind),
        ("case-folded string", RUNTIME),
        ("(kind, index) pair", (kind, 0)),
    ]
    if pyxrt is not None and RUNTIME == "xrt":
        # The last entry opens its own handle: distinct wrappers, same device.
        # Only XRT names a device by a handle -- an HSA agent is an index.
        same.append(("(kind, handle) pair", ("XRT", pyxrt.device(0))))
    for label, spec in same:
        buf = buffers.new((4,), torch.float32, spec)
        check(
            f"{label} resolves to the same {kind} device",
            buf.is_shared_with(NPU),
        )

    buf = buffers.new((4,), torch.float32, NPU, [HIP])
    check("device-string secondary resolves to HIP 0", buf.is_shared_with(HIP))
    check("bare 'hip' means index 0", buf.is_shared_with("hip"))
    check("a different index is a different device", not buf.is_shared_with("hip:1"))

    # What a buffer reports is a spec you can hand straight back to it. That is
    # what lets the device cache stay private: reading a buffer back never
    # requires being able to construct a runtime handle.
    check("devices report as canonical strings", buf.devices == (NPU, HIP))
    check("device round-trips through is_shared_with", buf.is_shared_with(buf.device))

    if RUNTIME == "xrt" and pyxrt is not None:
        # The exception, and the reason the return type is a union: a handle the
        # caller opened has no index to report, so it comes back as it went in.
        own_handle = pyxrt.device(0)
        by_handle = buffers.new((4,), torch.float32, ("XRT", own_handle))
        check(
            "a caller-supplied handle reports as a pair",
            by_handle.device == ("XRT", own_handle),
        )
        check("and round-trips too", by_handle.is_shared_with(by_handle.device))
        # ...and is the same device as "xrt:0", even though the wrapper differs.
        # Keying on the wrapper object would make these look like two devices.
        check(
            "a separate handle is not a separate device",
            by_handle.is_shared_with(NPU),
        )
    else:
        skip("caller-supplied handles", "only XRT names a device by a handle")

    if RUNTIME == "hsa":
        # There is exactly one AIE agent, so any other index is a mistake
        # rather than a device nobody has plugged in yet.
        check_raises(
            "a second HSA agent is rejected rather than aliased",
            SharedBufferError,
            lambda: buffers.new((4,), torch.float32, "hsa:1"),
        )

    # secondary=("HIP", 0) is a single device, not the two-element sequence it
    # looks like; getting this wrong would try to open a device named 0.
    one = buffers.new((4,), torch.float32, NPU, ("HIP", 0))
    check("a bare pair is one secondary, not two", one.devices == (NPU, HIP))

    two = buffers.new((4,), torch.float32, NPU, [HIP, HIP])
    check("duplicate secondaries collapse", len(two.devices) == 2)

    new = buffers.new
    bad: tuple[tuple[str, Any], ...] = (
        ("unknown kind", ("VULKAN", 0)),
        ("unknown kind as a string", "cuda:0"),
        ("non-numeric index", "hip:zero"),
        ("malformed spec", 17),
    )
    for label, spec in bad:
        check_raises(
            f"{label} rejected",
            SharedBufferError,
            lambda s=spec: new((4,), torch.float32, s),
        )
    check_raises(
        "unsupported dtype rejected",
        SharedBufferError,
        lambda: new((4,), torch.complex64, NPU),
    )


def test_degenerate_shapes(buffers: _Buffers) -> None:
    """Shapes no device can serve, refused in the module's own terms.

    Left to the runtimes, one shape produced five different errors between them
    -- ``mmap_range(len=0)`` from XRT, a pybind11 ``TypeError``, a null pointer
    from HIP -- and two were not ``SharedBufferError``, so the fallback every
    caller is told to write (``except SharedBufferError``) would not catch them.
    """
    print("\n-- degenerate shapes --")
    for label, shape in (
        ("an empty buffer", (0,)),
        ("a zero-length dimension", (4, 0, 8)),
        ("a negative dimension", (-4,)),
    ):
        for where, spec in (("alone", (NPU, ())), ("shared", (NPU, [HIP]))):
            check_raises(
                f"{label} rejected ({where})",
                SharedBufferError,
                lambda s=shape, d=spec: buffers.new((s), torch.float32, d[0], d[1]),
            )

    # A scalar is not degenerate: one element is something to map.
    scalar = buffers.new((), torch.float32, NPU)
    scalar.torch().fill_(3.0)
    check(
        "a scalar shape is a buffer of one element",
        scalar.shape == () and float(scalar.numpy()) == 3.0,
    )


# ---------------------------------------------------------------------------
# Torch-shaped factories
# ---------------------------------------------------------------------------
def test_factories(buffers: _Buffers) -> None:
    """shared.empty/zeros/ones/empty_like/from_tensor, spelled like torch."""
    print("\n-- factories --")
    from triton.backends.amd_triton_npu import shared

    def new(fn, *args, **kwargs):
        """Call a factory and keep the buffer for close_all()."""
        return buffers.adopt(fn(*args, **kwargs))

    on = dict(device=NPU, share=HIP)

    # Varargs shape and a shape tuple must agree, since torch accepts both and
    # empty(t.shape) is how you usually have one.
    a = new(shared.empty, M, D, dtype=torch.float32, **on)
    b = new(shared.empty, (M, D), dtype=torch.float32, **on)
    check("varargs and tuple shapes agree", a.shape == b.shape == (M, D))
    check("factory buffer is shared with both devices", a.devices == (NPU, HIP))

    default = new(shared.empty, 4, **on)
    check(
        "dtype defaults to torch's",
        default.dtype == torch.get_default_dtype(),
        str(default.dtype),
    )

    z = new(shared.zeros, 64, dtype=torch.float32, **on)
    o = new(shared.ones, 64, dtype=torch.float32, **on)
    torch.cuda.synchronize()
    check("zeros is zeroed", float(z.torch().abs().max()) == 0.0)
    check("ones is filled", float(o.torch().min()) == 1.0)

    # empty_like inherits the whole device set, so the result pairs with its
    # model everywhere the model is reachable.
    like = new(shared.empty_like, z)
    check(
        "empty_like inherits shape, dtype and devices",
        (like.shape, like.dtype, like.devices) == (z.shape, z.dtype, z.devices),
    )
    narrow = new(shared.empty_like, z, share=())
    check("empty_like overrides are independent", narrow.devices == (NPU,))

    # A plain tensor carries no device set, so device= is required rather than
    # guessed -- the same rule as empty(), for the same reason.
    check_raises(
        "empty_like(tensor) demands a device",
        SharedBufferError,
        lambda: shared.empty_like(torch.zeros(4)),
    )
    from_t = new(shared.empty_like, torch.zeros(4, 5, dtype=torch.float16), **on)
    check(
        "empty_like(tensor) takes shape and dtype",
        (from_t.shape, from_t.dtype) == ((4, 5), torch.float16),
    )

    # from_tensor is the one copy in the module: torch does the crossing.
    src = torch.arange(32, dtype=torch.float32, device="cuda") * 3.0
    copied = new(shared.from_tensor, src, **on)
    torch.cuda.synchronize()
    check(
        "from_tensor copies an iGPU tensor in",
        bool(torch.equal(copied.torch(), src)),
    )
    host_src = torch.arange(32, dtype=torch.float32)
    host_copy = new(shared.from_tensor, host_src, **on)
    torch.cuda.synchronize()
    check(
        "from_tensor copies a CPU tensor in",
        bool(torch.equal(host_copy.torch().cpu(), host_src)),
    )

    # device= has no default, on purpose: which runtime owns the pages decides
    # how they are allocated, and a silent default would hide that.
    check_raises("empty demands a device", TypeError, lambda: shared.empty(4))


# ---------------------------------------------------------------------------
# Each primary, with and without a secondary
# ---------------------------------------------------------------------------
def test_npu_primary_alone(buffers: _Buffers) -> None:
    """NPU-only: memory the NPU can reach, and nothing the iGPU can."""
    print(f"\n-- {RUNTIME.upper()} primary, no secondary --")
    buf = buffers.new((M, D), torch.float32, NPU)

    check("device reported", buf.device == NPU)
    check("shared with the NPU", buf.is_shared_with(NPU))
    check("not shared with HIP", not buf.is_shared_with(HIP))
    if RUNTIME == "xrt":
        check("BO available", buf.bo is not None)
    else:
        check("AIE address available", buf.aie_ptr() is not None, _hex(buf.aie_ptr()))

    # No iGPU mapping, so the buffer describes itself as host memory rather
    # than refusing -- that is the honest answer and keeps from_dlpack working.
    from triton.backends.amd_triton_npu.shared import kDLCPU, dlpack_device

    check("describes itself as CPU memory", dlpack_device(buf) == (kDLCPU, 0))
    t = buf.torch()
    check("torch view is a CPU tensor", not t.is_cuda and t.shape == (M, D))

    t.fill_(2.5)
    check("host view agrees with the torch view", bool((buf.numpy() == 2.5).all()))
    check_raises("device_ptr rejected without HIP", SharedBufferError, buf.device_ptr)


def test_npu_primary_hip_secondary(buffers: _Buffers) -> None:
    """The main path: the NPU owns the pages, the iGPU borrows them."""
    print(f"\n-- {RUNTIME.upper()} primary, HIP secondary --")
    buf = buffers.new((M, D), torch.float32, NPU, [HIP])

    check("shared with both", buf.is_shared_with(NPU) and buf.is_shared_with(HIP))
    check("device is still the NPU", buf.device == NPU)

    t = buf.torch()
    check(
        "torch view is an iGPU tensor",
        t.is_cuda and t.shape == (M, D),
        f"{tuple(t.shape)} {t.dtype} {t.device}",
    )

    t.fill_(3.5)
    torch.cuda.synchronize()
    host = buf.numpy()
    check(
        "iGPU write visible on the host mapping",
        bool((host == 3.5).all()),
        f"host[0,0]={host[0, 0]}",
    )

    host[:] = -1.25
    torch.cuda.synchronize()
    check(
        "host write visible on the iGPU",
        float(t.min()) == -1.25 and float(t.max()) == -1.25,
        f"gpu min/max={float(t.min())}/{float(t.max())}",
    )

    # A second torch() must alias, not copy.
    check("torch view is stable", buf.torch().data_ptr() == t.data_ptr())

    # Indexing goes straight to the torch view, in both directions.
    buf[0, :4] = 8.0
    torch.cuda.synchronize()
    check("__setitem__ writes through", bool((host[0, :4] == 8.0).all()))
    check("__getitem__ reads the same memory", float(buf[0, 0]) == 8.0)
    check(
        "a slice aliases rather than copies",
        buf[1:3].data_ptr() == t[1:3].data_ptr(),
    )

    text = repr(buf)
    check(
        "repr names shape, dtype and devices",
        all(p in text for p in ("(128, 768)", "float32", NPU, HIP)),
        text,
    )
    check("device_ptr is the iGPU alias", buf.device_ptr() == t.data_ptr())


def test_hip_primary_alone(buffers: _Buffers) -> None:
    """HIP-only: pinned pages the iGPU sees, with nothing to dispatch on."""
    print("\n-- HIP primary, no secondary --")
    buf = buffers.new((M, D), torch.float32, HIP)

    check("device reported", buf.device == HIP)
    check("not shared with the NPU", not buf.is_shared_with(NPU))

    t = buf.torch()
    check("torch view is an iGPU tensor", t.is_cuda, f"{t.device}")
    t.fill_(9.0)
    torch.cuda.synchronize()
    check("iGPU write visible on the host mapping", bool((buf.numpy() == 9.0).all()))

    # Returning None here would defer the failure into the launch, where it
    # reads as an XRT error rather than a missing share_with.
    check_raises("bo rejected without XRT", SharedBufferError, lambda: buf.bo)
    check_raises("aie_ptr rejected without HSA", SharedBufferError, buf.aie_ptr)


def test_hip_primary_npu_secondary(buffers: _Buffers) -> None:
    """The reverse import: the iGPU owns the pages, the NPU maps them."""
    print(f"\n-- HIP primary, {RUNTIME.upper()} secondary --")
    buf = buffers.new((M, D), torch.float32, HIP, [NPU])

    check("shared with both", buf.is_shared_with(HIP) and buf.is_shared_with(NPU))
    check("device is HIP", buf.device == HIP)

    t = buf.torch()
    t.fill_(4.25)
    torch.cuda.synchronize()
    check("iGPU write visible on the host mapping", bool((buf.numpy() == 4.25).all()))

    if RUNTIME == "xrt":
        check("BO available", buf.bo is not None)
        # XRT mapped the same pages rather than making its own copy; if the
        # userptr constructor had allocated, this would differ.
        check(
            "the BO maps the very pages HIP allocated",
            _bo_address(buf.bo) == buf.host_ptr,
            f"bo={hex(_bo_address(buf.bo))} host={hex(buf.host_ptr)}",
        )
    else:
        # The NPU's own address for the pages, which is a mapping of the iGPU
        # allocation rather than a copy of it -- nothing readable from here
        # (the import is granted to the AIE agent alone), so what this checks
        # is that there is one at all. That it names the same memory is what
        # the dispatch section establishes.
        check("AIE address available", buf.aie_ptr() is not None, _hex(buf.aie_ptr()))
        check(
            "the iGPU keeps its own address for those pages",
            buf.device_ptr() == buf.host_ptr,
            f"device={hex(buf.device_ptr())} host={hex(buf.host_ptr)}",
        )


def _hex(address: int | None) -> str:
    """An address for a check's detail line, printable when there is none.

    ``hex(None)`` raises, and a check's detail is evaluated whether or not the
    check passes -- so the naive spelling dies on exactly the failure it was
    written to report.
    """
    return "none" if address is None else hex(address)


def _bo_address(bo: Any) -> int:
    """Host address of a BO's mapping, without copying it."""
    return ctypes.addressof(ctypes.c_char.from_buffer(bo.map()))


def test_one_npu_runtime(buffers: _Buffers) -> None:
    """The two NPU runtimes are alternatives, and naming both is refused."""
    print("\n-- one NPU runtime at a time --")
    check_raises(
        "the other runtime rejected as a secondary",
        SharedBufferError,
        lambda: buffers.new((4,), torch.float32, NPU, [OTHER_NPU]),
    )
    check_raises(
        "...and alongside an iGPU primary",
        SharedBufferError,
        lambda: buffers.new((4,), torch.float32, HIP, [NPU, OTHER_NPU]),
    )
    # Refused from the device set, before the primary allocates -- so there is
    # nothing to unwind, which is the point of checking it there.
    buf = SharedBuffer((4,), torch.float32, NPU)
    check(
        "the other runtime is a known kind, not a typo",
        not buf.is_shared_with(OTHER_NPU),
    )
    buf.close()


def test_hip_pages_need_naming_the_npu_early(buffers: _Buffers) -> None:
    """Under HSA, iGPU pages must be allocated knowing the NPU will map them."""
    print("\n-- late share_with on iGPU pages --")
    if RUNTIME != "hsa":
        skip("late share_with", "XRT pins host pages, so it has no such rule")
        return
    # Pinned host memory cannot be exported, so an iGPU buffer that did not
    # name the NPU up front cannot be handed to it afterwards. The message has
    # to say that, since the fix is at construction, not here.
    buf = buffers.new((4,), torch.float32, HIP)
    check_raises(
        "share_with(hsa) on pinned pages is refused with the fix named",
        SharedBufferError,
        lambda: buf.share_with(NPU),
    )
    check(
        "the buffer is unharmed",
        buf.is_shared_with(HIP) and not buf.is_shared_with(NPU),
    )


# ---------------------------------------------------------------------------
# Sharing after construction
# ---------------------------------------------------------------------------
def test_share_with(buffers: _Buffers) -> None:
    """share_with adds a device to a live buffer, in both directions."""
    print("\n-- share_with --")
    from triton.backends.amd_triton_npu.shared import kDLCPU, kDLROCM

    buf = buffers.new((16, 8), torch.float32, NPU)
    check("starts CPU-flavoured", buf.__dlpack_device__() == (kDLCPU, 0))
    cpu_view = buf.torch()
    cpu_view.fill_(1.5)

    check("share_with returns self", buf.share_with(HIP) is buf)
    check("now shared with HIP", buf.is_shared_with(HIP))
    check("now iGPU-flavoured", buf.__dlpack_device__() == (kDLROCM, 0))

    # The cached view had to be dropped: it described the same pages as host
    # memory, and handing it back would send iGPU work to a CPU tensor.
    gpu_view = buf.torch()
    check("torch view was re-derived", gpu_view.is_cuda)
    check("and still sees what the CPU view wrote", float(gpu_view.max()) == 1.5)

    check("share_with is idempotent", buf.share_with(HIP).is_shared_with(HIP))
    check("no duplicate attachment", len(buf.devices) == 2)

    # ...and the other way round, for the runtime that can take pages it did
    # not allocate (see test_hip_pages_need_naming_the_npu_early for the one
    # that cannot).
    if RUNTIME == "xrt":
        rev = buffers.new((16, 8), torch.float32, HIP)
        rev.share_with(NPU)
        check("HIP buffer can gain an XRT device", rev.bo is not None)
    else:
        skip("late share_with(hsa)", "covered as a refusal instead")


def test_is_shared_with(buffers: _Buffers) -> None:
    """What is_shared_with answers for present, absent and bogus devices."""
    print("\n-- is_shared_with --")
    buf = buffers.new((4,), torch.float32, NPU, [HIP])

    check("primary counts as shared", buf.is_shared_with(NPU))
    check("secondary counts as shared", buf.is_shared_with(HIP))
    check("another HIP index is not shared", not buf.is_shared_with(("HIP", 7)))
    # A typo must not pass as a legitimate "no".
    check_raises(
        "unknown kind raises rather than answering False",
        SharedBufferError,
        lambda: buf.is_shared_with(("CUDA", 0)),
    )


def test_lifetime(buffers: _Buffers) -> None:
    """close() is idempotent and leaves the object in a stated state."""
    print("\n-- lifetime --")
    buf = SharedBuffer((8,), torch.float32, NPU, [HIP])
    check("shared before close", buf.is_shared_with(HIP))
    buf.close()
    buf.close()  # must not raise
    check("close is idempotent", True)
    check("repr says so once closed", repr(buf) == "SharedBuffer(closed)", repr(buf))
    check("nothing shared after close", buf.devices == ())
    check_raises(
        "share_with rejected after close",
        SharedBufferError,
        lambda: buf.share_with(HIP),
    )
    check_raises("device rejected after close", SharedBufferError, lambda: buf.device)


# ---------------------------------------------------------------------------
# DLPack protocol
# ---------------------------------------------------------------------------
def _capsule_name(cap: Any) -> str:
    """The capsule's tag, which is what identifies the DLPack flavour."""
    ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
    ctypes.pythonapi.PyCapsule_GetName.argtypes = [ctypes.py_object]
    return ctypes.pythonapi.PyCapsule_GetName(cap).decode()


def test_dlpack_protocol(buffers: _Buffers) -> None:
    """Version negotiation, device reporting, and refusing what we can't do."""
    print("\n-- DLPack protocol --")
    from triton.backends.amd_triton_npu.shared import (
        as_torch,
        dlpack_device,
        is_on_device,
        kDLCPU,
        kDLROCM,
    )

    buf = buffers.new((16, 8), torch.float32, NPU, [HIP])

    # torch asks for (1, 0) and only falls back on TypeError, so answering the
    # negotiation properly is what keeps this off an accidental code path.
    check(
        "legacy capsule for max_version=None",
        _capsule_name(buf.__dlpack__(max_version=None)) == "dltensor",
    )
    check(
        "versioned capsule for max_version=(1,0)",
        _capsule_name(buf.__dlpack__(max_version=(1, 0))) == "dltensor_versioned",
    )

    check("__dlpack_device__ reports ROCm:0", dlpack_device(buf) == (kDLROCM, 0))
    check("cpu tensor reports CPU", dlpack_device(torch.zeros(2)) == (kDLCPU, 0))

    t = buf.torch()
    check("is_on_device accepts its own device", is_on_device(t, HIP))
    # .is_cuda would say yes here; the device index is the whole point.
    check("is_on_device rejects another index", not is_on_device(t, "hip:1"))
    check_raises(
        "is_on_device rejects a device no tensor can live on",
        SharedBufferError,
        lambda: is_on_device(t, NPU),
    )

    # A foreign producer (numpy implements __dlpack__) must be consumable.
    import numpy as np

    check(
        "as_torch consumes a numpy array",
        as_torch(np.ones(4, dtype=np.float32)).sum().item() == 4.0,
    )

    for label, kwargs in (
        ("copy=True", {"copy": True}),
        ("dl_device mismatch", {"dl_device": (kDLCPU, 0)}),
    ):
        check_raises(f"rejects {label}", BufferError, buf.__dlpack__, **kwargs)

    # The same refusal, inverted: an NPU-only buffer *is* CPU memory, so the
    # mismatch it rejects is the ROCm one.
    host_only = buffers.new((16, 8), torch.float32, NPU)
    check_raises(
        "NPU-only buffer rejects a ROCm dl_device",
        BufferError,
        host_only.__dlpack__,
        dl_device=(kDLROCM, 0),
    )


# ---------------------------------------------------------------------------
# Real NPU dispatch
# ---------------------------------------------------------------------------
def test_npu_dispatch_xrt(buffers: _Buffers) -> None:
    """A real NPU kernel over shared buffers, with no host staging.

    Run once per ownership direction. The XRT-primary case is the one the fused
    MLP uses; the HIP-primary case dispatches on userptr BOs, which is the part
    that could plausibly be accepted by XRT and then read the wrong pages.

    Both are also cross-handle by construction, which is the point: the chain
    opens its own pyxrt.device and every buffer opens another, so a passing
    dispatch here is what licenses the module to stop caching one shared
    handle. If XRT ever did require the BO and the dispatch to share a wrapper,
    this is where it would show up.
    """
    print("\n-- NPU dispatch on shared buffers (XRT) --")
    import numpy as np

    import add_chain

    chain = add_chain.build("shared_buffer_add", N)
    # Host-side operands: the intermediate never leaves the NPU and the zero
    # addend is static, so neither is worth a shared buffer.
    tmp_host: np.ndarray = np.zeros(N, dtype=np.float32)
    zero_host: np.ndarray = np.zeros(N, dtype=np.float32)
    try:
        for label, primary, secondary in (
            ("XRT-owned pages", NPU, [HIP]),
            ("HIP-owned pages", HIP, [NPU]),
        ):
            a = buffers.new((N,), torch.float32, primary, secondary)
            b = buffers.new((N,), torch.float32, primary, secondary)
            c = buffers.new((N,), torch.float32, primary, secondary)

            # Operands are produced on the iGPU and never leave it.
            ta, tb, tc = a.torch(), b.torch(), c.torch()
            ta.copy_(torch.arange(N, dtype=torch.float32, device="cuda") * 0.25)
            tb.fill_(7.0)
            tc.zero_()
            torch.cuda.synchronize()
            expected = (ta + tb).clone()
            torch.cuda.synchronize()

            chain.run(
                [a.numpy(), b.numpy(), tmp_host, zero_host, c.numpy()],
                bo_key=f"shared_buffer_add_{primary}",
                static_indices={add_chain.I_ADDEND},
                intermediate_indices=add_chain.INTERMEDIATE,
                output_indices=add_chain.OUTPUT,
                bound_buffers={
                    add_chain.I_A: a.bo,
                    add_chain.I_B: b.bo,
                    add_chain.I_OUT: c.bo,
                },
            )

            torch.cuda.synchronize()
            err = float((tc - expected).abs().max())
            check(
                f"NPU result read straight off the iGPU ({label})",
                err == 0.0,
                f"max abs err={err}",
            )
    finally:
        chain.close()


@triton.jit
def _matmul_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """One tile of C = A @ B; the kernel examples/hsa_matmul dispatches."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b = tl.load(B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    tl.store(
        C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, tl.dot(a, b)
    )


def test_npu_dispatch_hsa(buffers: _Buffers) -> None:
    """A real NPU kernel over shared buffers, dispatched through ROCR.

    Run once per ownership direction, and the only thing here that proves the
    AIE agent really reaches those pages -- every check above establishes that
    the mapping exists, not that the device can use it.

    A matmul rather than the add chain the XRT path uses: the multi-launch ELF
    machinery is XRT-only, so this goes through the ordinary Triton launch,
    which under ROCR wants a tiling script (the scriptless elementwise path has
    an unrelated npu2 legalization limitation). The script is the one
    examples/matmul_bf16_m64_n64_k64 tunes for this shape.

    The dispatch counters are checked as well as the result, because a result
    alone cannot tell the two paths apart: a staged buffer computes the same
    answer, having copied itself to the device and back to prove it.
    """
    print("\n-- NPU dispatch on shared buffers (HSA) --")
    from triton.backends.amd_triton_npu.shared import hsa_dispatch_counts

    size = 256  # square, and a whole number of 256-wide tiles
    script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "matmul_bf16_m64_n64_k64",
        f"transform_{'aie2' if detect_npu_version() == 'npu1' else 'aie2p'}.mlir",
    )
    npu_config.transform_tiling_script = script
    grid = lambda meta: (  # noqa: E731 - Triton's own idiom for a grid
        triton.cdiv(size, meta["BLOCK_SIZE_M"]),
        triton.cdiv(size, meta["BLOCK_SIZE_N"]),
    )

    def matmul(a_arg: Any, b_arg: Any, c_arg: Any) -> tuple[int, int]:
        """Run C = A @ B on the NPU; returns what the dispatch did with the
        operands, as ``(in place, staged)`` deltas."""
        before = hsa_dispatch_counts()
        _matmul_kernel[grid](
            a_arg,
            b_arg,
            c_arg,
            size,
            size,
            size,
            size,
            1,
            size,
            1,
            size,
            1,
            BLOCK_SIZE_M=256,
            BLOCK_SIZE_N=256,
            BLOCK_SIZE_K=size,
        )
        after = hsa_dispatch_counts()
        return after[0] - before[0], after[1] - before[1]

    for label, primary, secondary in (
        ("NPU-owned pages", NPU, [HIP]),
        ("iGPU-owned pages", HIP, [NPU]),
    ):
        a = buffers.new((size, size), torch.bfloat16, primary, secondary)
        b = buffers.new((size, size), torch.bfloat16, primary, secondary)
        c = buffers.new((size, size), torch.float32, primary, secondary)

        # Operands are produced on the iGPU and never leave it.
        ta, tb, tc = a.torch(), b.torch(), c.torch()
        ta.copy_(torch.randn(size, size, dtype=torch.bfloat16, device="cuda"))
        tb.copy_(torch.randn(size, size, dtype=torch.bfloat16, device="cuda"))
        tc.zero_()
        torch.cuda.synchronize()
        expected = torch.matmul(ta.float(), tb.float())
        torch.cuda.synchronize()

        moved = matmul(ta, tb, tc)
        torch.cuda.synchronize()

        # bf16 inputs accumulated in f32, against a torch reference that rounds
        # differently, so this is a "did the NPU compute this" tolerance rather
        # than a numerical claim.
        err = float((tc - expected).abs().max())
        check(
            f"NPU result read straight off the iGPU ({label})",
            err < 20.0,
            f"max abs err={err:.3f}",
        )
        check(
            f"all three operands dispatched in place ({label})",
            moved == (3, 0),
            f"in place {moved[0]}, staged {moved[1]}",
        )

        # A staged operand for contrast: an ordinary tensor is copied in and
        # out, which is what a shared buffer exists to avoid.
        if label.startswith("NPU"):
            plain = torch.zeros(size, size, dtype=torch.float32)
            moved = matmul(ta, tb, plain)
            check(
                "an ordinary tensor is still staged",
                moved == (2, 1),
                f"in place {moved[0]}, staged {moved[1]}",
            )
            check(
                "and computes the same thing",
                float((plain - expected.cpu()).abs().max()) < 20.0,
            )


def main() -> int:
    """Run every check and return a shell exit status."""
    print("=" * 70)
    print("  shared multi-device buffer test")
    print("=" * 70)

    buffers = _Buffers()
    # Allocating is the availability check; the error names what is missing.
    # Scoped to this one call so a SharedBufferError raised by a *test* is
    # reported as a failure rather than as a missing environment. A failure
    # here is a failure: this run was asked for this runtime by name.
    try:
        buffers.new((1,), torch.float32, NPU, [HIP])
    except SharedBufferError as e:
        sys.exit(f"shared NPU/iGPU buffers unavailable for {RUNTIME}: {e}")

    try:
        print(f"  runtime: {RUNTIME}")
        print(f"  device : {torch.cuda.get_device_name(0)}")
        test_device_specs(buffers)
        test_degenerate_shapes(buffers)
        test_factories(buffers)
        test_npu_primary_alone(buffers)
        test_npu_primary_hip_secondary(buffers)
        test_hip_primary_alone(buffers)
        test_hip_primary_npu_secondary(buffers)
        test_one_npu_runtime(buffers)
        test_hip_pages_need_naming_the_npu_early(buffers)
        test_share_with(buffers)
        test_is_shared_with(buffers)
        test_lifetime(buffers)
        test_dlpack_protocol(buffers)
        if RUNTIME == "xrt":
            test_npu_dispatch_xrt(buffers)
            skip("NPU dispatch (HSA)", "this run drives the NPU through XRT")
        else:
            test_npu_dispatch_hsa(buffers)
            skip("NPU dispatch (XRT)", "this run drives the NPU through HSA")
    finally:
        buffers.close_all()

    print("\n" + "=" * 70)
    if _skipped:
        print(f"  SKIPPED: {', '.join(_skipped)}")
    if _failures:
        print(f"  FAILED: {', '.join(_failures)}")
    else:
        print("  ALL PASS")
    print("=" * 70)
    # Deliberately NOT os._exit(0): a clean interpreter shutdown is part of what
    # this test checks (the ctypes-callback deleter used to segfault here).
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
