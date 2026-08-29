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
    The Nth HSA agent. Not implemented yet.

A ``(kind, handle)`` pair is also accepted, and is the only way to name a
runtime object the caller already holds -- ``("XRT", pyxrt.device(0))`` when it
has to be *that* handle -- though for XRT any handle to the same device
will do, since they are interchangeable.

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

import ctypes
import functools
import glob
import math
import os
from typing import TYPE_CHECKING, Any, Callable, ClassVar, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
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
    "is_on_device",
    "ones",
    "zeros",
    "zeros_like",
]

# DLPack device types, as the spec numbers them. A buffer shared with a HIP
# device is described
# as a ROCm tensor: the pages are host-allocated, but the pointer handed out is
# the iGPU-side mapping, so the consumer must treat it as device memory. With no
# HIP device attached it is plain host memory and is described as such.
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


#: Stand-in owner for the DLPack views handed to consumers.
#:
#: nanobind needs *an* owner object or it tries to copy the contents, which it
#: cannot do for a bare pointer. Nothing Python-side actually owns these pages
#: -- the attachment does, and close() releases them explicitly -- so a
#: sentinel is the honest answer.
#:
#: Passing the SharedBuffer would read better and would keep the wrapper alive
#: behind a consumer's tensor, but it leaks: nanobind's ndarray is not
#: GC-tracked, so buffer -> cached torch view -> ndarray -> buffer is invisible
#: to the cycle collector and __del__ never runs. Measured, not assumed.
_NDARRAY_OWNER = object()


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

    def allocate(self, nbytes: int) -> int:
        """Obtain ``nbytes`` of shareable pages; returns their host address."""
        raise NotImplementedError

    def attach(self, host_ptr: int, nbytes: int) -> None:
        """Map an existing host range into this device."""
        raise NotImplementedError

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

    def allocate(self, nbytes: int) -> int:
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

    def attach(self, host_ptr: int, nbytes: int) -> None:
        import pyxrt

        self._bo = pyxrt.ext.bo(self.handle, _void_capsule(host_ptr), nbytes)

    def release(self) -> None:
        self._bo = None

    @property
    def bo(self) -> pyxrt.ext.bo | None:
        """The BO, for use as a kernel argument."""
        return self._bo


class _HipAttachment(_Attachment):
    """HIP's hold: pinned pages with an iGPU-side alias.

    As primary it calls ``hipHostMalloc``; as secondary it pins pages another
    runtime owns with ``hipHostRegister``. Either way
    ``hipHostGetDevicePointer`` yields the address the iGPU uses, which is what
    ``__dlpack__`` hands to torch.

    Why pinning and not an external-memory import
    ---------------------------------------------
    The obvious route for the secondary role -- export the BO as a dma-buf and
    import it into ROCm -- does not work. ``AMDKFD_IOC_GET_DMABUF_INFO``
    returns ``EINVAL`` for an ``amdxdna``-exported dma-buf (a ``drm``-exported
    one succeeds), because KFD can only describe buffers it can resolve back to
    an amdgpu object. That surfaces as ``hipErrorOutOfMemory`` from
    ``hipImportExternalMemory``, which is misleading. The vmem paths
    (``hipMemImportFromShareableHandle``,
    ``hsa_amd_vmem_import_shareable_handle``) fail too: they are matched-pair
    APIs that only accept handles minted by their own exporter, not arbitrary
    dma-bufs.

    ``hipHostRegister`` sidesteps all of it by never touching the fd -- it pins
    an existing host mapping and hands back a device pointer for the same pages.
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
        # Which of the two undo paths release() owes: freeing what we malloc'd,
        # or unregistering what we pinned. Never both.
        self._owned = False
        self._registered = False

    def _select(self) -> None:
        """Make this attachment's device current.

        The host-memory calls below act on whatever device HIP considers
        current, not on one passed in, so a buffer asked for HIP device 1 would
        otherwise silently land on device 0.
        """
        _hip_check(_hip().hipSetDevice(ctypes.c_int(self.index)), "hipSetDevice")

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

    def allocate(self, nbytes: int) -> int:
        self._select()
        ptr = ctypes.c_void_p()
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
        self._owned = True
        self._resolve_device_ptr()
        return self._host_ptr

    def attach(self, host_ptr: int, nbytes: int) -> None:
        self._select()
        _hip_check(
            _hip().hipHostRegister(
                ctypes.c_void_p(host_ptr),
                ctypes.c_size_t(nbytes),
                ctypes.c_uint(_HIP_HOST_REGISTER_MAPPED),
            ),
            "hipHostRegister",
        )
        self._host_ptr = host_ptr
        self._registered = True
        self._resolve_device_ptr()

    def release(self) -> None:
        # Swallowed: release() runs from close() and from __del__, and __del__
        # can run during interpreter teardown, where raising is reported as an
        # unraisable and can mask the real cause of a shutdown failure.
        try:
            if self._registered:
                _hip().hipHostUnregister(ctypes.c_void_p(self._host_ptr))
            elif self._owned:
                _hip().hipHostFree(ctypes.c_void_p(self._host_ptr))
        except Exception:
            pass
        self._registered = False
        self._owned = False
        self._device_ptr = None

    def dlpack_device(self) -> tuple[int, int]:
        return (kDLROCM, self.index)

    def data_ptr(self) -> int | None:
        return self._device_ptr


class _HsaAttachment(_Attachment):
    """Placeholder for sharing through HSA directly. Not implemented.

    Present so that "HSA" is a recognised kind everywhere a device is named,
    and asking for it fails with one clear message instead of an unknown-kind
    error that reads like a typo.

    Filling this in means ``hsa_amd_memory_lock_to_pool`` for the secondary
    role and ``hsa_amd_memory_pool_allocate`` from a fine-grained system pool
    for the primary one. Both need the ``hsa_agent_t`` rather than the plain
    index this stub accepts, so the handle convention will have to grow with
    the implementation.
    """

    kind = "HSA"

    def allocate(self, nbytes: int) -> int:
        raise SharedBufferError("HSA buffers are not implemented yet")

    def attach(self, host_ptr: int, nbytes: int) -> None:
        raise SharedBufferError("HSA sharing is not implemented yet")


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


def _make_attachment(device: DeviceSpec) -> _Attachment:
    """Build the attachment named by a device spec.

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
    try:
        cls = _BACKENDS[str(kind).upper()]
    except KeyError:
        raise SharedBufferError(
            f"unknown device kind {kind!r}; known kinds are "
            f"{', '.join(sorted(_BACKENDS))}"
        ) from None
    return cls(handle)


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
        self._torch_view: Tensor | None = None
        self._numpy_view: np.ndarray | None = None
        self.shape = tuple(shape)
        self.dtype = dtype

        _, bits, _ = _dtype_info(dtype)
        self._nbytes = math.prod(self.shape) * (bits // 8)

        primary_att = _make_attachment(device)
        self._host_ptr = primary_att.allocate(self._nbytes)
        self._primary = primary_att
        self._attachments[primary_att.key] = primary_att
        try:
            for secondary in _as_device_list(share):
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
        attachment.attach(self._host_ptr, self._nbytes)
        self._attachments[attachment.key] = attachment
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
        """Address of the pages in this process, valid for every attachment."""
        return self._host_ptr

    def device_ptr(self) -> int | None:
        """The iGPU-side address of the pages."""
        return self._one_of_kind(_HipAttachment).data_ptr()

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
        return _dlpack_ndarray()(
            pointer,
            list(self.shape),
            code,
            bits,
            device_type,
            device_id,
            _NDARRAY_OWNER,
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
        if self._torch_view is None:
            import torch

            self._torch_view = torch.from_dlpack(self)
        return self._torch_view

    def numpy(self) -> np.ndarray:
        """A host numpy view aliasing this buffer (no copy).

        Always available, whatever the buffer is shared with: every attachment
        maps the same host pages. Cached like the torch view -- the mapping is
        fixed for the buffer's lifetime, and callers on the dispatch path ask
        for it several times per launch.
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


def zeros(
    *size: int | Sequence[int],
    dtype: torch.dtype | None = None,
    device: DeviceSpec,
    share: Shared = (),
) -> SharedBuffer:
    """A zero-filled shared buffer, like ``torch.zeros``.

    Zeroed through the torch view rather than the host mapping, so on a buffer
    shared with an iGPU the fill happens there -- and, like any other device
    write, needs a ``torch.cuda.synchronize()`` before the NPU reads it.
    """
    buf = empty(*size, dtype=dtype, device=device, share=share)
    buf.torch().zero_()
    return buf


def ones(
    *size: int | Sequence[int],
    dtype: torch.dtype | None = None,
    device: DeviceSpec,
    share: Shared = (),
) -> SharedBuffer:
    """A one-filled shared buffer, like ``torch.ones``. See ``zeros``."""
    buf = empty(*size, dtype=dtype, device=device, share=share)
    buf.torch().fill_(1)
    return buf


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
    buf = empty_like(other, dtype=dtype, device=device, share=share)
    buf.torch().zero_()
    return buf


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
    buf = empty(
        tuple(tensor.shape),
        dtype=dtype if dtype is not None else tensor.dtype,
        device=device,
        share=share,
    )
    buf.torch().copy_(tensor)
    return buf
