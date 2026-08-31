# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Buffers several runtimes can address at once.

On an APU the iGPU and the NPU sit on the same physical DRAM, so a tensor handed
from one to the other never has to move -- yet the default path copies it twice
(host round trip, then a host->BO ``memcpy`` inside the launch). This module
removes both by allocating the pages once, on a *primary* device, and mapping
them into every *secondary* device that also needs to reach them.

Allocate through the torch-shaped factories at the bottom of the file::

    from triton.backends.amd_triton_npu import shared

    c = shared.empty(128, 768, dtype=torch.float32,
                     device="xrt:0", share="hip:0")

    torch.matmul(a, b, out=c.torch())          # iGPU writes where the NPU reads
    chain.run(..., bound_buffers={0: c.bo})    # NPU names the same pages
    c.close()                                  # release before the dispatcher

``SharedBuffer(shape, dtype, primary, secondary)`` is the same thing with a
shape tuple and no defaults; the factories are what most code should reach for.

Naming a device
---------------
A device string, spelled the way ``torch.device`` spells one, with a bare
``"hip"`` meaning index 0:

``xrt:N``
    The Nth XRT device. The buffer is an ``xrt::bo`` there, which is what an
    NPU dispatch can name as an argument.
``hip:N``
    The Nth HIP device. The buffer has a device pointer there, so torch sees
    an iGPU tensor.
``hsa:N``
    The Nth HSA AIE agent -- the same NPU as ``xrt:N``, reached through ROCR
    instead. The buffer is memory the AIE agent can address, which is what an
    NPU dispatch under the HSA runtime (``NPUDriver("hsa")``) runs on directly,
    with no staging copy. Only agent 0 exists today.

A ``(kind, handle)`` pair is also accepted, and is the only way to name a
runtime object the caller already holds -- ``("XRT", pyxrt.device(0))`` when it
has to be *that* handle -- though for XRT any handle to the same device
will do, since they are interchangeable.

A buffer names one NPU runtime or the other, never both: XRT and HSA have no
way to map each other's pages, and asking for both is refused rather than
half-honoured.

What lives where
----------------
Either ownership direction works. Each runtime's half -- how it allocates pages,
how it imports someone else's, and why by that particular mechanism -- lives on
its ``_Attachment`` subclass, which is the only place that has to change when a
runtime's import story does.

DLPack appears here only as the interchange format: it is how a buffer describes
itself to torch, CuPy or JAX without a copy. DLPack is the protocol; nanobind is
the implementation. The plugin returns a nanobind ndarray and the protocol
methods delegate to it, so nothing here hand-rolls the ABI -- but the names stay
DLPack's, because that is what the protocol is called and what a reader looking
for ``__dlpack__`` will search for.

The helpers for consuming *other* producers (``as_torch``, ``dlpack_device``,
``is_on_device``) are the one part of this file that has nothing to do with
shared buffers, and they are grouped together for that reason.

Sharing is not free, but it is paid once per buffer rather than once per
dispatch, and the design assumes buffers outlive many dispatches. Reads and
writes through a shared mapping are somewhat slower than through a native
device allocation; removing the per-dispatch copies more than covers it for the
access patterns this exists to serve.
"""

# Annotations are strings at runtime, so the type names below can refer to torch,
# numpy and pyxrt without importing any of them at module load. That matters:
# this module is imported by code that may have only one of the three, and each
# is deliberately imported inside the function that needs it.
from __future__ import annotations

import contextlib
import ctypes
import functools
import glob
import math
import os
import weakref
from typing import TYPE_CHECKING, Any, Callable, ClassVar, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from typing import TypeAlias

    import numpy as np
    import pyxrt
    import torch

    # SharedBuffer has a method named `torch`, which shadows the module for any
    # annotation written after it in the class body. Annotate tensors with this
    # alias inside the class; `torch.Tensor` there resolves to the method and
    # fails to type-check.
    from torch import Tensor

    #: What a runtime calls one of its devices: an index for HIP and HSA, an
    #: already-open handle for XRT, or nothing at all to mean index 0.
    DeviceHandle: TypeAlias = int | pyxrt.device | None

    #: How a caller names a device -- ``"hip:0"`` or ``("HIP", 0)``. See
    #: ``_make_attachment`` for the full grammar.
    DeviceSpec: TypeAlias = str | tuple[str, DeviceHandle]

    #: A spec resolved to its canonical, hashable form: the kind plus
    #: whatever uniquely names that device to its runtime.
    DeviceKey: TypeAlias = tuple[str, int | str | None]

    #: The ``share`` argument of the factories: any number of devices, or
    #: one written bare.
    Shared: TypeAlias = DeviceSpec | Iterable[DeviceSpec]

__all__ = [
    "SharedBuffer",
    "SharedBufferError",
    "as_torch",
    "dlpack_device",
    "empty",
    "empty_like",
    "from_tensor",
    "hsa_dispatch_counts",
    "is_on_device",
    "ones",
    "zeros",
    "zeros_like",
]

# DLPack device types, as the spec numbers them. A buffer shared with a HIP
# device is described as a ROCm tensor: the pointer handed out is the one the
# iGPU addresses -- an alias of host pages, or a real device allocation,
# depending on which runtimes hold the buffer -- so the consumer must treat it
# as device memory either way. With no HIP device attached the buffer is plain
# host memory and is described as such.
kDLCPU = 1
kDLCUDA = 2
kDLROCM = 10

# DLPack dtype codes. Same values as nanobind's dlpack::dtype_code, which is
# what these are ultimately handed to.
_kDLInt, _kDLUInt, _kDLFloat, _kDLBfloat = 0, 1, 2, 4

# hipHostRegisterMapped / hipHostMallocMapped: map the pinned range into the
# device address space, which is what makes hipHostGetDevicePointer work.
_HIP_HOST_REGISTER_MAPPED = 0x2
_HIP_HOST_MALLOC_MAPPED = 0x2


class SharedBufferError(RuntimeError):
    """Raised when a buffer cannot be allocated, shared, or reached."""


# ---------------------------------------------------------------------------
# Native pieces: the HIP runtime and the DLPack producer
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _hip() -> ctypes.CDLL:
    """The HIP runtime torch is already using.

    Deliberately not the system ROCm: the device pointer we register has to be
    valid in torch's HIP context, so it must come from torch's own runtime.
    """
    try:
        import torch
    except ImportError as e:  # pragma: no cover - torch is required to consume
        raise SharedBufferError(f"torch is required for shared buffers: {e}")
    root = os.path.join(os.path.dirname(torch.__file__), "lib")
    hits = glob.glob(os.path.join(root, "libamdhip64.so*"))
    if not hits:
        raise SharedBufferError(
            f"no libamdhip64.so under {root}; this needs a ROCm build of torch"
        )
    if not torch.cuda.is_available():
        # Checked here rather than left to hipHostRegister: the library loads
        # fine without a device and the failure would surface later as an
        # opaque HIP error code.
        raise SharedBufferError("no ROCm device visible to torch")
    lib = ctypes.CDLL(hits[0])
    lib.hipGetErrorName.restype = ctypes.c_char_p
    return lib


def _hip_check(rc: int, what: str) -> None:
    """Raise if a HIP call failed, naming the call and the error symbol."""
    if rc != 0:
        # hipGetErrorName returns NULL for codes it does not know, so decoding
        # unconditionally would replace the real error with an AttributeError.
        raw = _hip().hipGetErrorName(rc)
        name = raw.decode() if raw else "unknown"
        raise SharedBufferError(f"{what} failed: {rc} ({name})")


@functools.lru_cache(maxsize=1)
def _hsa() -> ctypes.CDLL:
    """The backend's HSA runtime library.

    The very same one kernel dispatches go through, deliberately: a shared
    region is only shared because *that* runtime knows about it, and the AIE
    agent permits a single queue, so a second ``hsa_init`` here would be a
    second runtime that neither sees our regions nor can dispatch on them.
    """
    try:
        from .driver import load_hsa_runtime
    except ImportError as e:  # pragma: no cover - the driver is always present
        raise SharedBufferError(f"the NPU backend driver is unavailable: {e}")
    try:
        return load_hsa_runtime()
    except Exception as e:
        # Everything that can go wrong here -- no AIE-capable ROCR, no NPU, a
        # ROCR already bound by something else -- arrives as some other
        # runtime's exception type, and the message is the useful part.
        raise SharedBufferError(f"the NPU's HSA runtime is unavailable: {e}") from None


def _hsa_call(name: str, *args: Any, restype: Any = ctypes.c_int) -> Any:
    """Call one of the runtime's shared-region entry points, or raise.

    They share a convention worth writing once: every one takes an error buffer
    as its last two arguments, and reports failure by returning NULL (pointers)
    or a non-zero value (ints), with the reason written into that buffer.

    The arguments are ctypes instances, so they describe the signature as well
    as carry the values -- declaring ``argtypes`` alongside them would be the
    same list written twice, with nothing to catch the two drifting apart.
    """
    fn = getattr(_hsa(), name)
    fn.restype = restype
    fn.argtypes = [*(type(a) for a in args), ctypes.c_char_p, ctypes.c_size_t]
    err = ctypes.create_string_buffer(512)
    result = fn(*args, err, len(err))
    # A NULL pointer comes back from ctypes as None, not as 0.
    ok = result is not None if restype is ctypes.c_void_p else result == 0
    if not ok:
        raise SharedBufferError(
            err.value.decode("utf-8", errors="replace") or f"{name} failed"
        )
    return result


def hsa_dispatch_counts() -> tuple[int, int]:
    """``(in_place, staged)`` tensor arguments dispatched through the NPU's
    HSA runtime since this process started.

    Sharing is invisible in results -- a staged buffer produces the same answer
    as a shared one, only slower -- so this is how a caller confirms that a
    buffer really is being dispatched on where it lives. ``staged`` growing
    across a launch means something about that operand was not recognised: not
    a shared buffer, a device the runtime does not know, or a view reaching
    past the region.

    Counts dispatches through the *HSA* runtime only, and does not start it:
    ``(0, 0)`` from a process driving the NPU through XRT means there have been
    none, not that sharing failed.
    """
    in_place = ctypes.c_uint64()
    staged = ctypes.c_uint64()
    fn = _hsa().triton_npu_hsa_dispatch_counts
    fn.restype = None
    fn.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
    fn(ctypes.byref(in_place), ctypes.byref(staged))
    return in_place.value, staged.value


@functools.lru_cache(maxsize=1)
def _dlpack_ndarray() -> Callable[..., Any]:
    """The DLPack producer, compiled into the backend plugin.

    Returns a ``nanobind.nb_ndarray`` over a raw pointer. nanobind owns the
    DLPack ABI and the consumer-facing protocol -- see the comment in
    ``amd_triton_npu/amd_triton_npu.cc`` for what that covers and why it is
    compiled in rather than built at import time.
    """
    from triton._C.libtriton import amd_triton_npu as _plugin

    return _plugin.dlpack_ndarray


def _void_capsule(ptr: int) -> Any:
    """Wrap a raw address in an unnamed ``PyCapsule``.

    Unrelated to DLPack, despite the type: this one is an argument to pyxrt.
    pyxrt binds the ``void* userptr`` constructor of ``xrt::ext::bo``, and
    pybind11 marshals a bare ``void*`` as a capsule -- there is no overload that
    takes an integer address, so this is the only way to hand XRT a pointer that
    came from another runtime.
    """
    new = ctypes.pythonapi.PyCapsule_New
    new.restype = ctypes.py_object
    new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    return new(ctypes.c_void_p(ptr), None, None)


# ---------------------------------------------------------------------------
# Consuming other producers
# ---------------------------------------------------------------------------
def dlpack_device(obj: Any) -> tuple[int, int]:
    """``(device_type, device_id)`` for any DLPack producer.

    Preferred over torch's ``.is_cuda``, which answers a coarser question: it is
    true for any GPU, so a tensor on a *different* GPU than the shared buffer
    would pass the check and then be copied across devices (or fail) further
    down. The DLPack tuple carries the device index, so the caller can compare
    it against the one the buffer is registered on.

    Falls back to inspecting torch tensors directly for producers that predate
    the protocol.
    """
    fn = getattr(obj, "__dlpack_device__", None)
    if fn is not None:
        dev_type, dev_id = fn()
        return int(dev_type), int(dev_id)
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            if obj.device.type == "cuda":
                return kDLROCM, obj.device.index or 0
            return kDLCPU, 0
    except ImportError:
        pass
    return kDLCPU, 0


def as_torch(obj: Any) -> torch.Tensor:
    """A torch tensor over ``obj``, without copying when it can be avoided.

    The interchange happens here and only here: anything implementing
    ``__dlpack__`` is adopted zero-copy, and everything downstream is plain
    torch. Compute stays torch's job -- DLPack describes memory, it does not
    provide operations.
    """
    import torch

    if isinstance(obj, torch.Tensor):
        return obj
    if hasattr(obj, "__dlpack__"):
        return torch.from_dlpack(obj)
    raise SharedBufferError(
        f"{type(obj).__name__} is neither a torch tensor nor a DLPack producer"
    )


def is_on_device(obj: Any, device: DeviceSpec) -> bool:
    """Whether ``obj`` lives on ``device``, named as everywhere else in this module.

    Only runtimes a framework can hold a tensor on can answer this, so the spec
    has to be a HIP one -- asking whether a torch tensor is "on xrt:0" has no
    meaning, and returning False would read like a legitimate no.

    ROCm and CUDA both count: torch spells its device ``cuda`` on a ROCm build,
    and producers differ on which DLPack code they report for it.
    """
    want = _make_attachment(device).dlpack_device()
    if want is None:
        raise SharedBufferError(f"{device!r} cannot hold a framework tensor")
    dev_type, dev_index = dlpack_device(obj)
    return dev_type in (kDLROCM, kDLCUDA) and dev_index == want[1]


# ---------------------------------------------------------------------------
# dtype plumbing
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _dtype_table() -> dict[torch.dtype, tuple[int, int, np.dtype]]:
    """torch dtype -> (dlpack code, bits, numpy dtype).

    One table rather than three: the DLPack code, the element size and the
    host-side numpy dtype all answer the same question about the same dtype,
    and keeping them apart lets the supported sets drift.

    bfloat16 has no numpy equivalent, so ml_dtypes supplies it -- the same one
    the multi-launch path already stages bf16 through.
    """
    import numpy as np
    import torch
    from ml_dtypes import bfloat16 as _bf16

    return {
        torch.float64: (_kDLFloat, 64, np.dtype(np.float64)),
        torch.float32: (_kDLFloat, 32, np.dtype(np.float32)),
        torch.float16: (_kDLFloat, 16, np.dtype(np.float16)),
        torch.bfloat16: (_kDLBfloat, 16, np.dtype(_bf16)),
        torch.int64: (_kDLInt, 64, np.dtype(np.int64)),
        torch.int32: (_kDLInt, 32, np.dtype(np.int32)),
        torch.int16: (_kDLInt, 16, np.dtype(np.int16)),
        torch.int8: (_kDLInt, 8, np.dtype(np.int8)),
        torch.uint8: (_kDLUInt, 8, np.dtype(np.uint8)),
    }


def _dtype_info(torch_dtype: torch.dtype) -> tuple[int, int, np.dtype]:
    """(dlpack code, bits, numpy dtype) for a torch dtype.

    Raises rather than returning a default: silently guessing a dtype would
    hand the consumer a tensor that reinterprets the buffer's bytes.
    """
    try:
        return _dtype_table()[torch_dtype]
    except KeyError:
        raise SharedBufferError(f"no DLPack mapping for {torch_dtype}") from None


# ---------------------------------------------------------------------------
# Per-runtime attachments
# ---------------------------------------------------------------------------
class _Attachment:
    """One runtime's hold on a shared buffer's pages.

    Each instance wraps a single device and plays exactly one of two roles for
    the buffer it belongs to:

    * *primary* -- ``allocate()`` obtains the pages and returns their host
      address. One attachment per buffer plays this role.
    * *secondary* -- ``attach()`` maps pages some other runtime allocated.

    ``release()`` undoes whichever one ran, and is idempotent so ``close()`` and
    ``__del__`` can both call it. Secondaries are always released before the
    primary, since they only borrow its pages.

    Subclasses that can back a DLPack tensor override ``dlpack_device()`` and
    ``data_ptr()``; the rest inherit the "not a DLPack device" answer.
    """

    #: Uppercase name this runtime is selected by in a device spec. Declared
    #: but not defined: every concrete subclass sets it, and a default would
    #: let one forget to.
    kind: ClassVar[str]

    def __init__(self, handle: DeviceHandle) -> None:
        self.handle = handle

    @property
    def key(self) -> DeviceKey:
        """Hashable identity, used to tell devices apart within one buffer.

        HIP and HSA identify a device by index, so the index is the identity.
        XRT overrides this: its handle is an opaque wrapper, and two wrappers
        for one device must not look like two devices.
        """
        return (self.kind, self._spec_index)

    @property
    def _spec_index(self) -> int | None:
        """The index this device was named by, if it was named by one.

        HIP and HSA identify devices by index and store it as the handle, so
        the default reads it straight off. XRT resolves an index to an opened
        handle and has to remember it separately.
        """
        return self.handle if isinstance(self.handle, int) else None

    @property
    def spec(self) -> DeviceSpec:
        """This device, written the way a caller would name it.

        A device string whenever the index is known, so that
        ``is_shared_with(buf.device)`` round-trips through the grammar without
        the caller holding a runtime handle -- which is the whole reason the
        device cache stays private. A handle the caller supplied has no index
        to report, so it comes back as the pair it went in as.
        """
        index = self._spec_index
        if index is None:
            return (self.kind, self.handle)
        return f"{self.kind.lower()}:{index}"

    def allocate(self, nbytes: int, peers: frozenset[str]) -> int:
        """Obtain ``nbytes`` of shareable pages; returns their host address.

        ``peers`` names the kinds this buffer will be shared with, because for
        HIP that decides *what to allocate*: pages the AIE agent can be given
        are a different kind of memory from pinned host pages, and only one of
        the two can be exported (see ``_HipAttachment``). Runtimes that
        allocate the same way regardless ignore it.
        """
        raise NotImplementedError

    def attach(self, host_ptr: int, nbytes: int, primary: _Attachment) -> None:
        """Map a range the ``primary`` attachment allocated into this device.

        ``primary`` is passed rather than just its address because importing is
        not always a matter of naming pages: the HSA path needs a dma-buf that
        only the owning runtime can mint. It is also what lets each runtime keep
        its own refusals -- which pairings it cannot serve -- on its own class.
        """
        raise NotImplementedError

    def on_peer_attached(self, peer: _Attachment) -> None:
        """Note that ``peer`` also holds this buffer.

        Called both ways once a new attachment maps the pages -- each existing
        one hears about the newcomer and the newcomer hears about each of them
        -- so an attachment learns its peers whatever order they arrived in.
        Only the HSA attachment does anything with it: it has to be told the
        address a peer will name those pages by, since a dispatch that sees an
        unfamiliar address has no way back to the region.
        """

    def on_peer_released(self, peer: _Attachment) -> None:
        """Note that ``peer`` is about to drop its hold; the inverse of above."""

    def release(self) -> None:
        """Undo allocate()/attach(). Idempotent; never raises."""

    def dlpack_device(self) -> tuple[int, int] | None:
        """``(device_type, device_id)`` if this device can back a DLPack tensor."""
        return None

    def data_ptr(self) -> int | None:
        """The buffer's address as seen by this device, if it has one."""
        return None


class _XrtAttachment(_Attachment):
    """XRT's hold: an ``xrt::bo``, which is what an NPU dispatch can name.

    As primary it allocates the BO and the pages are XRT's own mapping. As
    secondary it wraps someone else's pages in a userptr BO -- XRT accepts any
    page-aligned host range, and every allocation this module hands out is
    page-aligned by construction (both XRT's mapping and ``hipHostMalloc``).
    """

    kind = "XRT"

    def __init__(self, handle: DeviceHandle) -> None:
        try:
            import pyxrt
        except ImportError as e:
            raise SharedBufferError(f"pyxrt is required for XRT buffers: {e}") from None

        # Opening is cheap after the first -- XRT refcounts the device beneath
        # the wrapper -- so an index is resolved here rather than through a
        # shared cache. Handles are interchangeable: a BO allocated on one
        # dispatches correctly against a kernel built on another, which
        # shared_buffer_test.py pins down.
        if handle is None or isinstance(handle, int):
            self._index: int | None = 0 if handle is None else handle
            device = pyxrt.device(self._index)
        else:
            self._index = None
            device = handle
        super().__init__(device)
        # Identity is the device's bus address, not the wrapper object and not
        # the index: it is the one spelling every caller agrees on, so a handle
        # the caller opened compares equal to "xrt:0" for the same device.
        self._bdf = device.get_info(pyxrt.xrt_info_device.bdf)
        self._bo: pyxrt.ext.bo | None = None

    @property
    def key(self) -> DeviceKey:
        return (self.kind, self._bdf)

    @property
    def _spec_index(self) -> int | None:
        """The handle is an opened device, so the index is kept alongside it."""
        return self._index

    def allocate(self, nbytes: int, peers: frozenset[str]) -> int:
        import pyxrt

        self._bo = pyxrt.ext.bo(self.handle, nbytes)
        # pyxrt hands back a writable memoryview over the BO's host mapping;
        # ctypes gives us its address without copying.
        #
        # Both the memoryview and the ctypes view are dropped on return, which
        # reads like a dangling pointer but is not: the mapping belongs to the
        # BO, not to either wrapper, and self._bo keeps it alive. That is also
        # why release() drops the BO only after every borrower is gone.
        mv = self._bo.map()
        return ctypes.addressof(ctypes.c_char.from_buffer(mv))

    def attach(self, host_ptr: int, nbytes: int, primary: _Attachment) -> None:
        import pyxrt

        # Whatever XRT is asked to map is a host range: the only pages it could
        # not pin are the iGPU-allocated ones, and those only exist on a buffer
        # that named HSA, which _check_device_mix refuses before we get here.
        self._bo = pyxrt.ext.bo(self.handle, _void_capsule(host_ptr), nbytes)

    def release(self) -> None:
        self._bo = None

    @property
    def bo(self) -> pyxrt.ext.bo | None:
        """The BO, for use as a kernel argument."""
        return self._bo


class _HipAttachment(_Attachment):
    """HIP's hold: an address the iGPU can reach.

    As secondary it pins pages another runtime owns with ``hipHostRegister``
    and asks ``hipHostGetDevicePointer`` for the iGPU-side alias. As primary it
    allocates, and *what* it allocates depends on who else will hold the buffer:

    * with an XRT device, or alone -- ``hipHostMalloc``. Pinned host pages,
      which is what XRT's userptr BO can wrap.
    * with an HSA device -- ``hipMalloc``. An iGPU allocation, which is the only
      kind that can be exported as a dma-buf, and a dma-buf is the only way
      pages the iGPU owns can be handed to the AIE agent. On an APU that memory
      is host-visible as well, which is what keeps the host views working.

    The two are mutually exclusive, which is why the choice is made from the
    device set at construction rather than adapted to later: XRT cannot pin
    device memory (the userptr ioctl returns ``ENOMEM``) and pinned pages
    cannot be exported at all, so there is no allocation that serves both and
    no way to convert one into the other in place.

    Why pinning and not an external-memory import
    ---------------------------------------------
    In the secondary role the obvious route -- export the other runtime's
    buffer as a dma-buf and import it into ROCm -- does not work for an
    *XRT*-owned BO. ``AMDKFD_IOC_GET_DMABUF_INFO`` returns ``EINVAL`` for an
    ``amdxdna``-exported dma-buf (a ``drm``-exported one succeeds), because KFD
    can only describe buffers it can resolve back to an amdgpu object. That
    surfaces as ``hipErrorOutOfMemory`` from ``hipImportExternalMemory``, which
    is misleading, and the vmem import paths fail on the same fd for the same
    reason.

    ``hipHostRegister`` sidesteps all of it by never touching the fd -- it pins
    an existing host mapping and hands back a device pointer for the same pages.
    It works on an HSA-allocated range too, so both NPU runtimes are served by
    one secondary path. (The reverse direction, HIP's own pages into the AIE
    agent, *does* go through a dma-buf; see ``_HsaAttachment``.)
    """

    kind = "HIP"

    def __init__(self, handle: DeviceHandle) -> None:
        index = 0 if handle is None else int(handle)
        super().__init__(index)
        #: The same value as ``handle``, but typed: HIP names devices by index,
        #: where the base class has to stay open to whatever a runtime uses.
        self.index = index
        self._host_ptr: int | None = None
        self._device_ptr: int | None = None
        #: Whether the pages are an iGPU allocation rather than pinned host
        #: memory -- which is what decides whether the AIE agent can be given
        #: them, so the HSA attachment reads it before trying.
        self.device_memory = False
        # What release() owes, set when the memory is obtained: free the host
        # pages we pinned, free the iGPU allocation we made, or unregister
        # pages another runtime owns. A name rather than a bound callable: the
        # callable would be a method with extra steps, and it would hold a
        # reference back to the attachment for as long as the attachment holds
        # it.
        self._held: str | None = None

    @contextlib.contextmanager
    def _selected(self) -> Iterator[None]:
        """Make this attachment's device current for the enclosing block.

        The host-memory calls below act on whatever device HIP considers
        current, not on one passed in, so a buffer asked for HIP device 1 would
        otherwise silently land on device 0.

        The previous device is restored on the way out, which is not
        housekeeping: torch caches its own idea of the current device and skips
        hipSetDevice when it believes nothing changed. Leaving ours set would
        make the next ``torch.empty(device="cuda")`` allocate on this device
        instead of the one torch thinks it is on.
        """
        hip = _hip()
        previous = ctypes.c_int()
        _hip_check(hip.hipGetDevice(ctypes.byref(previous)), "hipGetDevice")
        _hip_check(hip.hipSetDevice(ctypes.c_int(self.index)), "hipSetDevice")
        try:
            yield
        finally:
            hip.hipSetDevice(previous)

    def _release_on_error(self, step: Callable[[], None]) -> None:
        """Run ``step``, undoing this attachment's pin/allocation if it raises."""
        try:
            step()
        except Exception:
            self.release()
            raise

    def _resolve_device_ptr(self) -> None:
        """Fill in the iGPU-side alias of ``self._host_ptr``."""
        dev = ctypes.c_void_p()
        _hip_check(
            _hip().hipHostGetDevicePointer(
                ctypes.byref(dev), ctypes.c_void_p(self._host_ptr), ctypes.c_uint(0)
            ),
            "hipHostGetDevicePointer",
        )
        if dev.value is None:
            raise SharedBufferError("hipHostGetDevicePointer returned null")
        self._device_ptr = dev.value

    def allocate(self, nbytes: int, peers: frozenset[str]) -> int:
        if _HsaAttachment.kind in peers:
            return self._allocate_device(nbytes)
        return self._allocate_pinned(nbytes)

    def _allocate_pinned(self, nbytes: int) -> int:
        """Pinned host pages, with an iGPU-side alias."""
        ptr = ctypes.c_void_p()
        with self._selected():
            _hip_check(
                _hip().hipHostMalloc(
                    ctypes.byref(ptr),
                    ctypes.c_size_t(nbytes),
                    ctypes.c_uint(_HIP_HOST_MALLOC_MAPPED),
                ),
                "hipHostMalloc",
            )
        if ptr.value is None:
            raise SharedBufferError("hipHostMalloc returned a null pointer")
        self._host_ptr = ptr.value
        self._held = "pinned"
        # Past this point the pages are pinned, so a failure has to undo it
        # here: the caller has no handle on this attachment yet -- SharedBuffer
        # only records it once allocate()/attach() returns -- so nothing else
        # would ever call release().
        self._release_on_error(self._resolve_device_ptr)
        return self._host_ptr

    def _allocate_device(self, nbytes: int) -> int:
        """An iGPU allocation, which is the flavour HIP will export.

        There is no separate host address to resolve: the allocation *is* the
        device address, and on the APUs this module exists for it is
        host-visible at the same value, which is what lets the buffer keep
        offering host views of it.
        """
        ptr = ctypes.c_void_p()
        with self._selected():
            _hip_check(
                _hip().hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)),
                "hipMalloc",
            )
        if ptr.value is None:
            raise SharedBufferError("hipMalloc returned a null pointer")
        self._host_ptr = ptr.value
        self._device_ptr = ptr.value
        self.device_memory = True
        self._held = "device"
        return self._host_ptr

    def attach(self, host_ptr: int, nbytes: int, primary: _Attachment) -> None:
        with self._selected():
            _hip_check(
                _hip().hipHostRegister(
                    ctypes.c_void_p(host_ptr),
                    ctypes.c_size_t(nbytes),
                    ctypes.c_uint(_HIP_HOST_REGISTER_MAPPED),
                ),
                "hipHostRegister",
            )
        self._host_ptr = host_ptr
        self._held = "registered"
        self._release_on_error(self._resolve_device_ptr)

    def release(self) -> None:
        # Swallowed: release() runs from close() and from __del__, and __del__
        # can run during interpreter teardown, where raising is reported as an
        # unraisable and can mask the real cause of a shutdown failure.
        try:
            if self._held == "pinned":
                _hip().hipHostFree(ctypes.c_void_p(self._host_ptr))
            elif self._held == "device":
                _hip().hipFree(ctypes.c_void_p(self._device_ptr))
            elif self._held == "registered":
                _hip().hipHostUnregister(ctypes.c_void_p(self._host_ptr))
        except Exception:
            pass
        self._held = None
        self._device_ptr = None

    def dlpack_device(self) -> tuple[int, int]:
        return (kDLROCM, self.index)

    def data_ptr(self) -> int | None:
        return self._device_ptr


class _HsaAttachment(_Attachment):
    """The NPU's hold when it is driven through ROCR rather than XRT.

    A *shared region* in the backend's HSA runtime: memory the AIE agent can
    address, registered there so a dispatch naming it runs on it in place. That
    registration is the whole point -- without it the runtime would stage the
    tensor through a pooled buffer and copy it twice, which is exactly what a
    shared buffer exists to avoid.

    Both roles go through the vmem API, in the two ways it offers:

    * primary -- ``hsa_amd_vmem_handle_create`` on the AIE agent's data pool,
      mapped and granted to the CPU and the AIE agent. The result is ordinary
      addressable memory, so a HIP secondary can pin it with
      ``hipHostRegister`` like any other host range.
    * secondary -- ``hsa_amd_vmem_import_shareable_handle`` on the iGPU
      allocation the HIP primary owns, which the runtime reaches through a
      dma-buf it exports for it, then maps and grants to the AIE agent. ROCR
      refuses to grant an imported range to the CPU, so the host keeps reaching
      those pages by HIP's address for them, not this one.

    Only an XRT primary has no route: KFD cannot describe an ``amdxdna``-backed
    dma-buf, and there is no reason to try -- both name the same NPU, so a
    buffer wanting the NPU should name whichever runtime it will dispatch on.

    Nothing here touches ROCR until the buffer actually asks for memory:
    ``_make_attachment`` runs on every ``is_shared_with`` query, including from
    processes driving the NPU through XRT, and initialising ROCR there would
    have it open a device XRT is already using -- on an agent that permits a
    single queue.
    """

    kind = "HSA"

    def __init__(self, handle: DeviceHandle) -> None:
        index = 0 if handle is None else int(handle)
        if index != 0:
            # The runtime binds the first AIE agent it finds, so any other
            # index would silently land on agent 0.
            raise SharedBufferError(
                f"hsa:{index} does not exist; the NPU is agent 0 (hsa:0)"
            )
        super().__init__(index)
        #: Address the AIE agent reaches the region by, and the key the runtime
        #: knows it by. Also the CPU's address when we allocated it ourselves.
        self._va: int | None = None
        self._nbytes = 0
        #: Peer addresses registered as aliases of this region, so they can be
        #: retired before the mappings behind them go away.
        self._aliases: list[int] = []

    def allocate(self, nbytes: int, peers: frozenset[str]) -> int:
        va = _hsa_call(
            "triton_npu_hsa_shared_alloc",
            ctypes.c_uint64(nbytes),
            restype=ctypes.c_void_p,
        )
        self._va = va
        self._nbytes = nbytes
        return va

    def attach(self, host_ptr: int, nbytes: int, primary: _Attachment) -> None:
        # An XRT primary never reaches here (_check_device_mix refuses that
        # buffer), so what is left is an iGPU one -- and only its exportable
        # flavour will do. Pinned host pages cannot be exported at all, which
        # is worth saying here rather than letting the export fail: the
        # allocation that would have worked is one the buffer can no longer go
        # back and make.
        if not getattr(primary, "device_memory", False):
            raise SharedBufferError(
                "the AIE agent can only be given iGPU memory allocated for it; "
                f"this buffer's pages came from {primary.kind}. Name the HSA "
                "device in share= when the buffer is created, so its pages are "
                "allocated as iGPU memory in the first place"
            )
        self._va = _hsa_call(
            "triton_npu_hsa_shared_import",
            ctypes.c_void_p(host_ptr),
            ctypes.c_uint64(nbytes),
            restype=ctypes.c_void_p,
        )
        self._nbytes = nbytes
        # The import registered the iGPU's address as well, since that is what
        # a torch tensor over these pages carries into a dispatch.
        self._aliases.append(host_ptr)

    def on_peer_attached(self, peer: _Attachment) -> None:
        """Register the address ``peer`` names these pages by.

        A HIP secondary over an HSA-owned region gets its own address for the
        pages -- ``hipHostGetDevicePointer`` returns an alias, not the range it
        was handed -- and that alias is what a torch tensor carries into a
        dispatch. Without it the runtime would not recognise the pointer and
        would stage a copy of memory it already had.
        """
        if self._va is None:
            return
        alias = peer.data_ptr()
        if alias is None or alias == self._va or alias in self._aliases:
            return
        _hsa_call(
            "triton_npu_hsa_shared_alias",
            ctypes.c_void_p(alias),
            ctypes.c_void_p(self._va),
            ctypes.c_uint64(self._nbytes),
        )
        self._aliases.append(alias)

    def on_peer_released(self, peer: _Attachment) -> None:
        """Retire a peer's alias before the mapping behind it goes away.

        Otherwise a dispatch could resolve an address the peer has handed back
        to its own runtime, which may by then belong to something else.
        """
        alias = peer.data_ptr()
        if alias is None or alias not in self._aliases:
            return
        try:
            _hsa_call("triton_npu_hsa_shared_unalias", ctypes.c_void_p(alias))
        except SharedBufferError:
            pass  # teardown path; see release()
        self._aliases.remove(alias)

    def release(self) -> None:
        # Swallowed for the same reason as _HipAttachment.release: this runs
        # from __del__ too, where raising only obscures a shutdown failure.
        try:
            if self._va is not None:
                _hsa_call("triton_npu_hsa_shared_free", ctypes.c_void_p(self._va))
        except Exception:
            pass
        self._va = None
        self._aliases.clear()

    def data_ptr(self) -> int | None:
        """The AIE agent's address for the region.

        Not a host address: in the secondary role it is granted to the AIE
        agent alone, and dereferencing it faults. ``dlpack_device`` stays None
        so nothing hands it to a framework -- it is here to be inspected, and
        for the buffer's ``aie_ptr``.
        """
        return self._va


#: Binds the attachment selectors below to the runtime-specific subtype the
#: caller asked for, so ``_one_of_kind(_XrtAttachment).bo`` needs no re-check.
_A = TypeVar("_A", bound=_Attachment)

#: Device kind -> attachment class. Also the set of names a spec may use.
_BACKENDS = {cls.kind: cls for cls in (_XrtAttachment, _HipAttachment, _HsaAttachment)}


def _split_device_string(text: str) -> tuple[str, int | None]:
    """``"hip:0"`` -> ``("hip", 0)``; ``"hip"`` -> ``("hip", None)``.

    Spelled like ``torch.device``, because that is what a reader coming from
    torch will try first. A bare kind means index 0.
    """
    kind, sep, index = text.partition(":")
    if not sep:
        return kind, None
    try:
        return kind, int(index)
    except ValueError:
        raise SharedBufferError(
            f"device {text!r} has a non-numeric index; expected e.g. 'hip:0'"
        ) from None


def _parse_spec(device: DeviceSpec) -> tuple[str, DeviceHandle]:
    """The kind and handle a device spec names, without building anything.

    Separate from ``_make_attachment`` because a buffer has to know which kinds
    it will be shared with *before* its primary allocates -- HIP allocates a
    different kind of memory depending on whether the NPU will reach it through
    HSA -- and constructing the attachments to ask them is not free: opening an
    XRT device is a real device operation, and building an HSA one would
    initialise ROCR.

    Three spellings, all naming the same thing:

    * ``"hip:0"`` -- a device string, as ``torch.device`` writes them, and the
      form to prefer. A bare ``"hip"`` means index 0.
    * ``("HIP", 0)`` -- a ``(kind, index)`` pair.
    * ``("XRT", pyxrt.device(0))`` -- a pair carrying a handle the caller
      already owns, for when it has one to hand.

    The kind is matched case-insensitively. All three resolve to the same
    device identity, so a buffer named one way is recognised when named
    another.
    """
    if isinstance(device, str):
        kind, handle = _split_device_string(device)
    else:
        try:
            kind, handle = device
        except (TypeError, ValueError):
            raise SharedBufferError(
                f"device must be a device string like 'hip:0' or a "
                f"(kind, handle) pair, got {device!r}"
            ) from None
    name = str(kind).upper()
    if name not in _BACKENDS:
        raise SharedBufferError(
            f"unknown device kind {kind!r}; known kinds are "
            f"{', '.join(sorted(_BACKENDS))}"
        )
    return name, handle


def _make_attachment(device: DeviceSpec) -> _Attachment:
    """Build the attachment named by a device spec; see ``_parse_spec``."""
    kind, handle = _parse_spec(device)
    return _BACKENDS[kind](handle)


def _check_device_mix(kinds: set[str]) -> None:
    """Refuse a device set naming both NPU runtimes.

    They are two ways to reach the same device, and neither can map the other's
    pages: XRT cannot pin an HSA allocation's device memory, and KFD cannot
    describe an XRT-exported dma-buf. Caught here, from the whole device set,
    because the pair is not always adjacent -- a HIP buffer shared with both is
    refused by neither runtime's own check until it is too late to allocate
    differently.
    """
    if {_XrtAttachment.kind, _HsaAttachment.kind} <= kinds:
        raise SharedBufferError(
            "a buffer cannot be shared with both XRT and HSA devices: they are "
            "the same NPU reached two ways, and neither can map the other's "
            "pages -- XRT cannot pin the iGPU memory that an HSA-shared buffer "
            "has to be allocated as, and the AIE agent cannot be given an "
            "XRT-exported buffer. Name the runtime you will dispatch on"
        )


def _as_device_list(
    devices: Shared | None,
) -> list[DeviceSpec]:
    """Coerce the ``secondary`` argument into a list of device specs.

    A single spec is accepted as shorthand for a one-element list, because
    ``secondary=("HIP", 0)`` is what people write and iterating it as a
    sequence would read ``"HIP"`` and ``0`` as two separate devices. The
    handle-is-not-a-string test is what keeps ``["xrt:0", "hip:0"]`` -- two
    devices -- from being mistaken for one. Device strings make this mostly
    moot: ``secondary="hip:0"`` is unambiguous on its own.

    The kind test is a bare lookup rather than a parse: in the pair form the
    first element is always a plain kind, never ``"hip:0"``.
    """
    if devices is None:
        return []
    if isinstance(devices, str):
        return [devices]
    if (
        isinstance(devices, (tuple, list))
        and len(devices) == 2
        and isinstance(devices[0], str)
        and devices[0].upper() in _BACKENDS
        and not isinstance(devices[1], str)
    ):
        return [tuple(devices)]
    # The two single-spec forms were consumed above, so what is left is a
    # sequence of specs.
    return list(cast("Iterable[DeviceSpec]", devices))


# ---------------------------------------------------------------------------
# The shared buffer
# ---------------------------------------------------------------------------
class SharedBuffer:
    """One allocation addressable by several runtimes at once.

    The pages come from the *primary* device and are mapped into each
    *secondary* one; ``share_with`` adds more later and ``is_shared_with``
    reports what is already mapped. Views are cheap and non-owning: the pages
    live and die with this object, so a tensor handed out by ``torch()`` must
    not outlive it.

    Not thread-safe: concurrent dispatches sharing one buffer need external
    serialisation, which is also what the NPU's single command queue implies.
    """

    __slots__ = (
        "_attachments",
        "_primary",
        "_host_ptr",
        "_nbytes",
        "shape",
        "dtype",
        "_torch_view",
        "_numpy_view",
    )

    def __init__(
        self,
        shape: Sequence[int],
        dtype: torch.dtype,
        device: DeviceSpec,
        share: Shared = (),
    ) -> None:
        """Allocate ``shape``/``dtype`` on ``device`` and map it into ``share``.

        ``device`` is a device spec -- ``"xrt:0"``, ``("HIP", 0)``, or
        ``("XRT", handle)``; see ``_make_attachment`` for the grammar. ``share``
        is an iterable of the same, and may be empty: a buffer no other runtime
        maps is still useful to the one that owns it.

        Named to match the factories below, which is how most code allocates.

        Mapping happens here, once, because it is the expensive part and the
        whole design depends on the buffer being long-lived enough to amortise
        it.

        Raises ``SharedBufferError`` if a named device cannot be reached, for
        whatever reason that runtime reports. There is no separate "is it
        available?" predicate: asking and then doing would duplicate each
        runtime's requirements in two places and go stale, so the attempt *is*
        the check, and the exception carries the actual cause.
        """
        # Teardown state first: everything below can raise, and close() has to
        # stay callable after any of it.
        self._attachments = {}
        self._primary = None
        self._host_ptr: int | None = None
        self._torch_view: weakref.ref[Tensor] | None = None
        self._numpy_view: np.ndarray | None = None
        self.shape = tuple(shape)
        self.dtype = dtype

        _, bits, _ = _dtype_info(dtype)
        # Checked here, once, rather than left to whichever runtime allocates:
        # each reports a shape it cannot serve in its own terms and its own
        # exception type -- a raw ``mmap_range(len=0)`` from XRT, a pybind11
        # TypeError, a null pointer from HIP -- and two of those are not
        # SharedBufferError, which is what callers are told to catch.
        if any(dim < 0 for dim in self.shape):
            raise SharedBufferError(
                f"shape {tuple(self.shape)} has a negative dimension"
            )
        if math.prod(self.shape) == 0:
            raise SharedBufferError(
                f"shape {tuple(self.shape)} holds no elements; there is nothing "
                "for a device to map"
            )
        self._nbytes = math.prod(self.shape) * (bits // 8)

        # Parsed before anything is allocated: the primary allocates according
        # to who else will hold the buffer, and a device set that cannot work
        # should be refused while there is still nothing to unwind.
        secondaries = _as_device_list(share)
        primary_kind, primary_handle = _parse_spec(device)
        peers = frozenset(_parse_spec(s)[0] for s in secondaries)
        _check_device_mix({primary_kind, *peers})

        primary_att = _BACKENDS[primary_kind](primary_handle)
        self._host_ptr = primary_att.allocate(self._nbytes, peers)
        self._primary = primary_att
        self._attachments[primary_att.key] = primary_att
        try:
            for secondary in secondaries:
                self.share_with(secondary)
        except Exception:
            # A half-shared buffer is not something the caller can use or
            # reason about, so unwind rather than hand one back.
            self.close()
            raise

    # -- sharing ------------------------------------------------------------
    def share_with(self, device: DeviceSpec) -> SharedBuffer:
        """Map this buffer into another device; returns ``self`` so calls chain.

        Idempotent per device: sharing with one already attached is a no-op
        rather than an error, so a caller can ask for what it needs without
        tracking what it has already asked for.
        """
        if self._host_ptr is None:
            raise SharedBufferError("cannot share a closed buffer")
        attachment = _make_attachment(device)
        if attachment.key in self._attachments:
            return self
        _check_device_mix(
            {attachment.kind, *(a.kind for a in self._attachments.values())}
        )
        assert self._primary is not None  # a live buffer always has one
        attachment.attach(self._host_ptr, self._nbytes, self._primary)
        # Recorded before the announcements below, which can fail: an
        # attachment that has mapped the pages but is not in the dict is one
        # nothing can ever release, and a stranded mapping outlives the buffer.
        self._attachments[attachment.key] = attachment
        try:
            # Announced after the fact, so a peer that needs an address only the
            # new attachment can supply -- the HSA runtime does -- gets it once
            # that address exists.
            for existing in self._attachments.values():
                if existing is attachment:
                    continue
                existing.on_peer_attached(attachment)
                attachment.on_peer_attached(existing)
        except Exception:
            # Leave the buffer as it was found: a half-announced attachment is
            # one the runtime may not recognise at dispatch.
            del self._attachments[attachment.key]
            attachment.release()
            raise
        # Adding a HIP device changes which pointer the DLPack view should
        # carry, so a torch view minted before this one is now describing the
        # buffer as the wrong kind of memory.
        self._torch_view = None
        return self

    def is_shared_with(self, device: DeviceSpec) -> bool:
        """Whether this buffer is mapped into ``device``.

        The primary counts as shared: it is a device the buffer is reachable
        from, and callers asking "can I use it there?" mean exactly that.

        Raises for a malformed spec or an unknown kind -- those are typos, and
        answering ``False`` would let them pass as a legitimate "no".
        """
        return _make_attachment(device).key in self._attachments

    @property
    def device(self) -> DeviceSpec:
        """The device the pages were allocated on, as a spec you can pass back.

        ``devices`` is the whole set; this is the one that owns the memory.
        """
        if self._primary is None:
            raise SharedBufferError("buffer is closed")
        return self._primary.spec

    @property
    def devices(self) -> tuple[DeviceSpec, ...]:
        """Every device this buffer is mapped into, primary first."""
        return tuple(a.spec for a in self._attachments.values())

    def _one_of_kind(self, cls: type[_A]) -> _A:
        """The single attachment of one runtime.

        Raises when there is none, and when there is more than one: with two
        devices of the same runtime attached there is no defensible default,
        and picking the first would silently dispatch on the wrong one.
        """
        found = [a for a in self._attachments.values() if isinstance(a, cls)]
        if not found:
            raise SharedBufferError(
                f"buffer is not shared with any {cls.kind} device "
                f"(shared with: "
                f"{', '.join(a.kind for a in self._attachments.values()) or 'nothing'})"
            )
        if len(found) > 1:
            raise SharedBufferError(
                f"buffer is shared with {len(found)} {cls.kind} devices; "
                f"name which one"
            )
        return found[0]

    # -- per-device handles -------------------------------------------------
    @property
    def bo(self) -> pyxrt.ext.bo:
        """The ``xrt::bo`` for the one XRT device this buffer is shared with.

        Raises when there is none: a dispatch cannot name a buffer XRT does not
        know about, and returning ``None`` would defer that to a confusing
        failure inside the launch.
        """
        return self._one_of_kind(_XrtAttachment).bo

    @property
    def host_ptr(self) -> int | None:
        """The primary device's address for the pages, in this process.

        Not "the host address": each runtime holding the buffer has its own
        address for the same memory, and this is the one the pages were
        allocated at. It is what the other attachments were handed to map, and
        what the host views are built on -- which is why an XRT-owned buffer's
        BO maps at exactly this address.
        """
        return self._host_ptr

    def device_ptr(self) -> int | None:
        """The iGPU-side address of the pages."""
        return self._one_of_kind(_HipAttachment).data_ptr()

    def aie_ptr(self) -> int | None:
        """The NPU-side address of the pages under the HSA runtime.

        The address a dispatch through ROCR runs on, and the counterpart of
        ``bo`` for that runtime -- though unlike a BO it is not something a
        caller passes anywhere: the runtime resolves it from whichever address
        the caller does pass. Here to be looked at.

        Not necessarily readable from the host: when the iGPU owns the pages,
        the NPU's mapping of them is granted to the AIE agent alone.
        """
        return self._one_of_kind(_HsaAttachment).data_ptr()

    # -- views --------------------------------------------------------------
    def _dlpack_source(self) -> tuple[int, int, int]:
        """``(device_type, device_id, pointer)`` describing the DLPack view.

        A device attachment wins over the host mapping: its pointer is the
        whole reason a framework can consume this buffer without a copy. With
        no such device attached the buffer is still perfectly good host memory,
        so it is described as a CPU tensor rather than refused -- that is the
        honest answer, and it keeps ``torch.from_dlpack`` working on a buffer
        that is only shared with the NPU.
        """
        for attachment in self._attachments.values():
            device = attachment.dlpack_device()
            pointer = attachment.data_ptr()
            if device is not None and pointer is not None:
                return device[0], device[1], pointer
        if self._host_ptr is None:
            raise SharedBufferError("buffer is closed")
        return kDLCPU, 0, self._host_ptr

    def __dlpack_device__(self) -> tuple[int, int]:
        """DLPack device tuple.

        ROCm when a HIP device is attached -- the pages are host-allocated, but
        the pointer ``__dlpack__`` hands out is then the iGPU-side mapping, so
        the consumer must treat it as device memory.
        """
        device_type, device_id, _ = self._dlpack_source()
        return (device_type, device_id)

    def _as_ndarray(self) -> Any:
        """A ``nanobind.nb_ndarray`` describing this buffer.

        Built fresh per call rather than cached: it is cheap, and a cached one
        would go stale the moment ``share_with`` changed which mapping the
        buffer should describe itself by.
        """
        device_type, device_id, pointer = self._dlpack_source()
        code, bits, _ = _dtype_info(self.dtype)
        # self as owner: nanobind needs one (without it it tries to copy, which
        # it cannot do for a bare pointer), and it makes a consumer's tensor keep
        # this buffer alive. That matters -- `shared.zeros(...).torch()` drops
        # the buffer at the end of the expression, and without the back-reference
        # the tensor is left over pages __del__ has already released.
        #
        # The obvious cycle this would form, buffer -> cached view -> ndarray ->
        # buffer, is broken on the first edge: torch() holds the view weakly.
        # It has to be broken there, because nanobind's ndarray is not
        # GC-tracked, so the cycle would be invisible to the collector and
        # __del__ would never run.
        return _dlpack_ndarray()(
            pointer, list(self.shape), code, bits, device_type, device_id, self
        )

    def __dlpack__(
        self,
        *,
        stream: int | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> Any:
        """A DLPack capsule over this buffer.

        Delegated to nanobind, which implements the whole negotiation: a fresh
        capsule per call (ownership transfers to the consumer, so handing the
        same one out twice would double-free it), legacy or versioned according
        to ``max_version``, and ``BufferError`` for ``copy=True`` or a
        ``dl_device`` this buffer is not on.

        The signature is spelled out rather than ``**kwargs`` so the accepted
        arguments stay discoverable, and because ``stream`` deserves saying out
        loud: it is accepted and ignored, since the other writer of these pages
        is the NPU, which is not on a HIP stream at all. Callers must fence
        explicitly around the hand-off (``_FusedMLP.run`` does).
        """
        return self._as_ndarray().__dlpack__(
            stream=stream,
            max_version=max_version,
            dl_device=dl_device,
            copy=copy,
        )

    def torch(self) -> Tensor:
        """A torch tensor aliasing this buffer.

        An iGPU tensor when a HIP device is attached, a CPU one otherwise.
        Cached: the view is stable for as long as the attachment set is, and
        re-deriving it would build an ndarray and a capsule per call for no
        benefit.
        """
        cached = self._torch_view() if self._torch_view is not None else None
        if cached is None:
            import torch

            cached = torch.from_dlpack(self)
            self._torch_view = weakref.ref(cached)
        return cached

    def numpy(self) -> np.ndarray:
        """A host numpy view aliasing this buffer (no copy).

        Reads and writes the primary's pages directly, whatever else holds
        them. For an XRT- or HSA-owned buffer those are host pages and this is
        unremarkable. For an iGPU-owned one they are an iGPU allocation, and
        the view works because the two processors share physical memory --
        which is the premise this whole module rests on, but is worth saying
        where a raw address is handed to numpy.

        Cached like the torch view: the mapping is fixed for the buffer's
        lifetime, and callers on the dispatch path ask for it several times per
        launch.
        """
        if self._numpy_view is None:
            import numpy as np

            if self._host_ptr is None:
                raise SharedBufferError("buffer is closed")
            _, _, dt = _dtype_info(self.dtype)
            buf = (ctypes.c_char * self._nbytes).from_address(self._host_ptr)
            self._numpy_view = np.frombuffer(buf, dtype=dt).reshape(self.shape)
        return self._numpy_view

    def __getitem__(self, index: Any) -> Tensor:
        """Index the buffer as its torch view: ``buf[:4]`` is ``buf.torch()[:4]``.

        Sugar, but it removes the accessor from the common case -- slicing and
        in-place writes -- which is most of what call sites do with a buffer.
        Whole-buffer operations still go through ``torch()`` or ``numpy()``,
        which is the right seam to be explicit about.
        """
        return self.torch()[index]

    def __setitem__(self, index: Any, value: Any) -> None:
        """Write through the torch view; see ``__getitem__``."""
        self.torch()[index] = value

    def __repr__(self) -> str:
        """Shape, dtype and where it is reachable -- what a tensor repr shows.

        A buffer on the wrong device is the failure this exists to make
        visible, so the device list is the part worth printing.
        """
        if self._primary is None:
            return f"{type(self).__name__}(closed)"
        places = ", ".join(
            d if isinstance(d, str) else f"{d[0].lower()}:<handle>"
            for d in self.devices
        )
        return (
            f"{type(self).__name__}(shape={tuple(self.shape)}, "
            f"dtype={self.dtype}, devices=[{places}])"
        )

    # -- lifetime -----------------------------------------------------------
    def close(self) -> None:
        """Release every mapping and the pages. Idempotent.

        Order matters and is the reverse of construction: views first, since
        they point into pages that are about to stop being addressable; then
        the secondaries, which only borrow the primary's pages; then the
        primary, which owns them. Every secondary's undo step names an address
        that is only meaningful while the pages are still mapped, so releasing
        the primary first leaves each of them operating on freed memory.
        """
        self._torch_view = None
        self._numpy_view = None
        # The primary is inserted first and share_with() ignores a duplicate
        # key, so it is always the first entry -- reversed() releases it last
        # without needing to single it out.
        for attachment in reversed(list(self._attachments.values())):
            # Tell the others first: an attachment's address stops meaning
            # anything the moment it lets go, and whoever recorded it has to
            # forget it while it is still theirs to forget.
            for other in self._attachments.values():
                if other is not attachment:
                    other.on_peer_released(attachment)
            attachment.release()
        self._primary = None
        self._attachments.clear()
        self._host_ptr = None

    def __del__(self) -> None:
        """Best-effort release.

        Exceptions are swallowed because __del__ can run during interpreter
        teardown, where raising is reported as an unraisable and can mask the
        real cause of a shutdown failure.
        """
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Torch-shaped factories
# ---------------------------------------------------------------------------
def _normalize_size(size: tuple[int | Sequence[int], ...]) -> tuple[int, ...]:
    """``(128, 768)`` from either ``f(128, 768)`` or ``f((128, 768))``.

    torch accepts both spellings of a shape and so should this; varargs alone
    would make ``empty(t.shape)`` -- which is how you usually have a shape --
    fail with a confusing dtype error.
    """
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        return tuple(size[0])
    # Not a sequence in position 0, so by the signature's contract every
    # element is a dimension.
    return cast("tuple[int, ...]", size)


def empty(
    *size: int | Sequence[int],
    dtype: torch.dtype | None = None,
    device: DeviceSpec,
    share: Shared = (),
) -> SharedBuffer:
    """An uninitialised shared buffer, like ``torch.empty``.

    ``device`` is the runtime that allocates the pages and ``share`` the ones
    that map them; both take device strings (``"xrt:0"``) or ``(kind, handle)``
    pairs. ``share`` accepts a single device as well as a list.

    Uninitialised means uninitialised: XRT and HIP both hand back whatever was
    in those pages. Use ``zeros`` when that matters.
    """
    if dtype is None:
        import torch

        dtype = torch.get_default_dtype()
    return SharedBuffer(_normalize_size(size), dtype, device, share)


def _written(buf: SharedBuffer, fill: Callable[[Tensor], Any]) -> SharedBuffer:
    """Apply ``fill`` to the buffer's torch view and make the write visible.

    The fence is the point. On an iGPU-shared buffer the fill is a device write
    on the current stream, and the NPU is not on that stream -- so a factory
    that returned before draining it would hand back a buffer whose contents
    are not there yet, and the next dispatch would read whatever was in those
    pages. Doing it here rather than asking every caller to remember is the
    whole reason these factories exist.

    Only the current stream is drained; a buffer with no HIP device attached is
    plain host memory and its fill was synchronous.
    """
    view = buf.torch()
    fill(view)
    if view.is_cuda:
        import torch

        torch.cuda.current_stream().synchronize()
    return buf


def zeros(
    *size: int | Sequence[int],
    dtype: torch.dtype | None = None,
    device: DeviceSpec,
    share: Shared = (),
) -> SharedBuffer:
    """A zero-filled shared buffer, like ``torch.zeros``.

    Zeroed through the torch view rather than the host mapping, so on an
    iGPU-shared buffer the fill happens there. The write is fenced before
    returning; see ``_written``.
    """
    return _written(
        empty(*size, dtype=dtype, device=device, share=share),
        lambda view: view.zero_(),
    )


def ones(
    *size: int | Sequence[int],
    dtype: torch.dtype | None = None,
    device: DeviceSpec,
    share: Shared = (),
) -> SharedBuffer:
    """A one-filled shared buffer, like ``torch.ones``. See ``zeros``."""
    return _written(
        empty(*size, dtype=dtype, device=device, share=share),
        lambda view: view.fill_(1),
    )


def empty_like(
    other: SharedBuffer | Tensor,
    *,
    dtype: torch.dtype | None = None,
    device: DeviceSpec | None = None,
    share: Shared | None = None,
) -> SharedBuffer:
    """A buffer shaped like ``other``, like ``torch.empty_like``.

    ``other`` may be another ``SharedBuffer``, in which case the device set is
    inherited too and ``out = empty_like(c)`` gives a buffer that pairs with
    ``c`` on every device it is reachable from. For a plain tensor there is no
    device set to inherit, so ``device`` is required -- the same rule as
    ``empty``, for the same reason.

    Any of the three can be overridden, independently of each other.
    """
    if isinstance(other, SharedBuffer):
        if device is None:
            device = other.device
        if share is None:
            share = other.devices[1:]
    elif device is None:
        raise SharedBufferError(
            f"empty_like({type(other).__name__}) cannot infer a device; pass device="
        )
    return empty(
        tuple(other.shape),
        dtype=dtype if dtype is not None else other.dtype,
        device=device,
        share=() if share is None else share,
    )


def zeros_like(
    other: SharedBuffer | Tensor,
    *,
    dtype: torch.dtype | None = None,
    device: DeviceSpec | None = None,
    share: Shared | None = None,
) -> SharedBuffer:
    """A zero-filled buffer shaped like ``other``, like ``torch.zeros_like``.

    Inherits from ``other`` on the same terms as ``empty_like``; see ``zeros``
    for how the fill is issued.
    """
    return _written(
        empty_like(other, dtype=dtype, device=device, share=share),
        lambda view: view.zero_(),
    )


def from_tensor(
    tensor: Tensor,
    *,
    dtype: torch.dtype | None = None,
    device: DeviceSpec,
    share: Shared = (),
) -> SharedBuffer:
    """A shared buffer holding a copy of ``tensor``.

    The bridge for data that already exists somewhere else. The copy goes
    through the buffer's torch view, so torch handles the crossing -- a CPU
    tensor into an iGPU-shared buffer is a host-to-device copy, and one already
    on the iGPU never leaves it.

    This is a copy, and the only one in this module. It is what you want when
    torch allocated the tensor; ``empty`` and friends are what you want when
    the buffer can own the data from the start.

    Why copy rather than share torch's own pages
    -------------------------------------------
    An iGPU allocation *can* be handed to the NPU without copying, by exporting
    it with ``hsa_amd_portable_export_dmabuf`` and importing that fd through
    XRT's importing ``bo`` constructor; this was tried and an NPU kernel ran
    correctly on torch-allocated memory. It is not what happens here for two
    reasons. The import cost is paid per hand-off rather than once per buffer,
    which is the wrong shape for a buffer that will be dispatched on repeatedly.
    And torch's caching allocator suballocates, so a tensor usually sits at a
    non-zero offset within a larger segment and needs an XRT sub-buffer carved
    out of it -- more machinery, and a lifetime tied to torch's segment rather
    than to the tensor. Allocating the buffer here avoids both.
    """
    return _written(
        empty(
            tuple(tensor.shape),
            dtype=dtype if dtype is not None else tensor.dtype,
            device=device,
            share=share,
        ),
        lambda view: view.copy_(tensor),
    )
