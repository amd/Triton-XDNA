#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""End-to-end check of the multi-device shared buffers in ``shared``.

Covers every combination the API admits:

  * each device kind as primary (XRT, HIP, and HSA's not-implemented stub),
    with an empty secondary list and with each other kind as secondary;
  * ``share_with`` / ``is_shared_with`` after construction, including the
    idempotent and closed-buffer cases;
  * device-spec parsing -- device strings, the ``(kind, handle)`` pair, case
    folding, and what a typo does;
  * the ``shared`` module's torch-shaped factories, including that ``device``
    has no default;
  * the DLPack protocol on a buffer that *has* an iGPU mapping and on one that
    does not, since those describe themselves as different kinds of memory;
  * a real Triton-compiled NPU kernel dispatched on shared BOs, once with XRT
    owning the pages and once with HIP owning them -- the second is the userptr
    import, which is the part that is easy to get subtly wrong;
  * a clean interpreter exit -- the original ctypes-callback DLPack deleter
    segfaulted at shutdown, which the C deleter is supposed to fix.

Exit status is the result, so this doubles as a regression test.

Run it::

    source scripts/dev-env.sh
    python examples/npu_gpu_dlpack/shared_buffer_test.py
"""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Callable, Sequence
from typing import Any

import torch

import add_chain
import pyxrt

from triton.backends.amd_triton_npu.shared import SharedBuffer, SharedBufferError

M, D = 128, 768
N = M * D

# The two devices used throughout, in the canonical spelling -- which is also
# what devices/device report back, so these double as expected values.
XRT = "xrt:0"
HIP = "hip:0"

_failures: list[str] = []


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

    # Every spelling of "XRT device 0" must resolve to the same device, so a
    # buffer named one way is recognised when named another. Note the last
    # entry opens its own handle: distinct wrappers, same device.
    same: list[tuple[str, Any]] = [
        ("device string", "xrt:0"),
        ("bare kind string", "XRT"),
        ("case-folded string", "xrt"),
        ("(kind, index) pair", ("XRT", 0)),
        ("(kind, handle) pair", ("XRT", pyxrt.device(0))),
    ]
    for label, spec in same:
        buf = buffers.new((4,), torch.float32, spec)
        check(
            f"{label} resolves to the same XRT device",
            buf.is_shared_with(XRT),
        )

    buf = buffers.new((4,), torch.float32, "xrt:0", ["hip:0"])
    check("device-string secondary resolves to HIP 0", buf.is_shared_with(HIP))
    check("bare 'hip' means index 0", buf.is_shared_with("hip"))
    check("a different index is a different device", not buf.is_shared_with("hip:1"))

    # What a buffer reports is a spec you can hand straight back to it. That is
    # what lets the device cache stay private: reading a buffer back never
    # requires being able to construct a runtime handle.
    check("devices report as canonical strings", buf.devices == (XRT, HIP))
    check("device round-trips through is_shared_with", buf.is_shared_with(buf.device))

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
    check("a separate handle is not a separate device", by_handle.is_shared_with(XRT))

    # secondary=("HIP", 0) is a single device, not the two-element sequence it
    # looks like; getting this wrong would try to open a device named 0.
    one = buffers.new((4,), torch.float32, XRT, ("HIP", 0))
    check("a bare pair is one secondary, not two", one.devices == (XRT, HIP))

    two = buffers.new((4,), torch.float32, XRT, ["hip:0", "hip:0"])
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
        lambda: new((4,), torch.complex64, XRT),
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

    on = dict(device="xrt:0", share="hip:0")

    # Varargs shape and a shape tuple must agree, since torch accepts both and
    # empty(t.shape) is how you usually have one.
    a = new(shared.empty, M, D, dtype=torch.float32, **on)
    b = new(shared.empty, (M, D), dtype=torch.float32, **on)
    check("varargs and tuple shapes agree", a.shape == b.shape == (M, D))
    check("factory buffer is shared with both devices", a.devices == (XRT, HIP))

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
    check("empty_like overrides are independent", narrow.devices == (XRT,))

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
def test_xrt_primary_alone(buffers: _Buffers) -> None:
    """XRT-only: a BO and host pages, but nothing the iGPU can reach."""
    print("\n-- XRT primary, no secondary --")
    buf = buffers.new((M, D), torch.float32, XRT)

    check("device reported", buf.device == XRT)
    check("shared with XRT", buf.is_shared_with(XRT))
    check("not shared with HIP", not buf.is_shared_with(HIP))
    check("BO available", buf.bo is not None)

    # No iGPU mapping, so the buffer describes itself as host memory rather
    # than refusing -- that is the honest answer and keeps from_dlpack working.
    from triton.backends.amd_triton_npu.shared import kDLCPU, dlpack_device

    check("describes itself as CPU memory", dlpack_device(buf) == (kDLCPU, 0))
    t = buf.torch()
    check("torch view is a CPU tensor", not t.is_cuda and t.shape == (M, D))

    t.fill_(2.5)
    check("host view agrees with the torch view", bool((buf.numpy() == 2.5).all()))
    check_raises("device_ptr rejected without HIP", SharedBufferError, buf.device_ptr)


def test_xrt_primary_hip_secondary(buffers: _Buffers) -> None:
    """The main path: XRT owns the pages, the iGPU borrows them."""
    print("\n-- XRT primary, HIP secondary --")
    buf = buffers.new((M, D), torch.float32, XRT, [HIP])

    check("shared with both", buf.is_shared_with(XRT) and buf.is_shared_with(HIP))
    check("device is still XRT", buf.device == XRT)

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
        all(p in text for p in ("(128, 768)", "float32", "xrt:0", "hip:0")),
        text,
    )
    check("device_ptr is the iGPU alias", buf.device_ptr() == t.data_ptr())


def test_hip_primary_alone(buffers: _Buffers) -> None:
    """HIP-only: pinned pages the iGPU sees, with no BO to dispatch on."""
    print("\n-- HIP primary, no secondary --")
    buf = buffers.new((M, D), torch.float32, HIP)

    check("device reported", buf.device == HIP)
    check("not shared with XRT", not buf.is_shared_with(XRT))

    t = buf.torch()
    check("torch view is an iGPU tensor", t.is_cuda, f"{t.device}")
    t.fill_(9.0)
    torch.cuda.synchronize()
    check("iGPU write visible on the host mapping", bool((buf.numpy() == 9.0).all()))

    # Returning None here would defer the failure into the launch, where it
    # reads as an XRT error rather than a missing share_with.
    check_raises("bo rejected without XRT", SharedBufferError, lambda: buf.bo)


def test_hip_primary_xrt_secondary(buffers: _Buffers) -> None:
    """The reverse import: HIP owns the pages, XRT wraps them as a userptr BO."""
    print("\n-- HIP primary, XRT secondary --")
    buf = buffers.new((M, D), torch.float32, HIP, [XRT])

    check("shared with both", buf.is_shared_with(HIP) and buf.is_shared_with(XRT))
    check("device is HIP", buf.device == HIP)
    check("BO available", buf.bo is not None)

    t = buf.torch()
    t.fill_(4.25)
    torch.cuda.synchronize()
    check("iGPU write visible on the host mapping", bool((buf.numpy() == 4.25).all()))
    # XRT mapped the same pages rather than making its own copy; if the userptr
    # constructor had allocated, this would differ.
    check(
        "the BO maps the very pages HIP allocated",
        _bo_address(buf.bo) == buf.host_ptr,
        f"bo={hex(_bo_address(buf.bo))} host={hex(buf.host_ptr)}",
    )


def _bo_address(bo: Any) -> int:
    """Host address of a BO's mapping, without copying it."""
    return ctypes.addressof(ctypes.c_char.from_buffer(bo.map()))


def test_hsa_stub(buffers: _Buffers) -> None:
    """HSA is recognised as a kind but not implemented, in either role."""
    print("\n-- HSA stub --")
    new = buffers.new
    check_raises(
        "HSA primary rejected",
        SharedBufferError,
        lambda: new((4,), torch.float32, ("HSA", 0)),
    )
    check_raises(
        "HSA secondary rejected",
        SharedBufferError,
        lambda: new((4,), torch.float32, XRT, [("HSA", 0)]),
    )
    # The failure above happened after the primary had already allocated. If
    # the constructor did not unwind, this buffer would have leaked a BO -- and
    # the caller would hold no reference with which to release it.
    buf = SharedBuffer((4,), torch.float32, XRT)
    check("HSA is a known kind, not a typo", not buf.is_shared_with(("HSA", 0)))
    buf.close()


# ---------------------------------------------------------------------------
# Sharing after construction
# ---------------------------------------------------------------------------
def test_share_with(buffers: _Buffers) -> None:
    """share_with adds a device to a live buffer, in both directions."""
    print("\n-- share_with --")
    from triton.backends.amd_triton_npu.shared import kDLCPU, kDLROCM

    buf = buffers.new((16, 8), torch.float32, XRT)
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

    # ...and the other way round.
    rev = buffers.new((16, 8), torch.float32, HIP)
    rev.share_with(XRT)
    check("HIP buffer can gain an XRT device", rev.bo is not None)


def test_is_shared_with(buffers: _Buffers) -> None:
    """What is_shared_with answers for present, absent and bogus devices."""
    print("\n-- is_shared_with --")
    buf = buffers.new((4,), torch.float32, XRT, [HIP])

    check("primary counts as shared", buf.is_shared_with(XRT))
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
    buf = SharedBuffer((8,), torch.float32, XRT, [HIP])
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

    buf = buffers.new((16, 8), torch.float32, XRT, [HIP])

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
        lambda: is_on_device(t, XRT),
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

    # The same refusal, inverted: an XRT-only buffer *is* CPU memory, so the
    # mismatch it rejects is the ROCm one.
    host_only = buffers.new((16, 8), torch.float32, XRT)
    check_raises(
        "XRT-only buffer rejects a ROCm dl_device",
        BufferError,
        host_only.__dlpack__,
        dl_device=(kDLROCM, 0),
    )


# ---------------------------------------------------------------------------
# Real NPU dispatch
# ---------------------------------------------------------------------------
def test_npu_dispatch(buffers: _Buffers) -> None:
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
    print("\n-- NPU dispatch on shared buffers --")
    import numpy as np

    chain = add_chain.build("shared_buffer_add", N)
    # Host-side operands: the intermediate never leaves the NPU and the zero
    # addend is static, so neither is worth a shared buffer.
    tmp_host: np.ndarray = np.zeros(N, dtype=np.float32)
    zero_host: np.ndarray = np.zeros(N, dtype=np.float32)
    try:
        for label, primary, secondary in (
            ("XRT-owned pages", XRT, [HIP]),
            ("HIP-owned pages", HIP, [XRT]),
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


def main() -> int:
    """Run every check and return a shell exit status."""
    print("=" * 70)
    print("  shared multi-device buffer test")
    print("=" * 70)

    buffers = _Buffers()
    # Allocating is the availability check; the error names what is missing.
    # Scoped to this one call so a SharedBufferError raised by a *test* is
    # reported as a failure rather than as a missing environment.
    try:
        buffers.new((1,), torch.float32, XRT, [HIP])
    except SharedBufferError as e:
        sys.exit(f"shared NPU/iGPU buffers unavailable: {e}")

    try:
        print(f"  device : {torch.cuda.get_device_name(0)}")
        test_device_specs(buffers)
        test_factories(buffers)
        test_xrt_primary_alone(buffers)
        test_xrt_primary_hip_secondary(buffers)
        test_hip_primary_alone(buffers)
        test_hip_primary_xrt_secondary(buffers)
        test_hsa_stub(buffers)
        test_share_with(buffers)
        test_is_shared_with(buffers)
        test_lifetime(buffers)
        test_dlpack_protocol(buffers)
        test_npu_dispatch(buffers)
    finally:
        buffers.close_all()

    print("\n" + "=" * 70)
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
