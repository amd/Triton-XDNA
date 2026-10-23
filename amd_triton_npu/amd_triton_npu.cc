//===- amd_triton_npu.cc -------------------------------------------*- C++
//-*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifdef AMD_TRITON_NPU_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#else
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#endif

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef AMD_TRITON_NPU_PYBIND11
namespace py = pybind11;
#else
namespace py = nanobind;
#endif

//===----------------------------------------------------------------------===//
// DLPack producer for shared multi-device buffers
//
// Backs ``amd_triton_npu/backend/shared.py``: a buffer whose pages one runtime
// allocated and another mapped is described to any DLPack consumer (torch,
// CuPy, JAX) as a tensor over the mapping that consumer can reach, with no
// copy.
//
// nanobind owns the DLPack ABI. Returning an ``ndarray<array_api>`` hands
// Python a ``nanobind.nb_ndarray``, which implements the consumer-facing half
// in full: ``__dlpack_device__``, legacy vs versioned capsule negotiation off
// ``max_version``, ``BufferError`` for ``copy=True`` and for a mismatched
// ``dl_device``, and ``stream`` accepted-but-ignored. That last one is correct
// here for a reason worth stating: the other writer of these pages is the NPU,
// which is not on a HIP stream at all, so there is no producer stream to order
// against and callers must fence explicitly around the hand-off.
//
// This is why pyproject.toml floors nanobind at 3.0: 2.10.2 has the versioned-
// capsule negotiation but merely accepts and ignores ``copy`` and ``dl_device``
// instead of refusing them. shared_buffer_test.py asserts the refusals, so an
// older nanobind fails there rather than silently losing them.
//
// pybind11 has no ndarray type, so the pybind11 build below implements the
// same object by hand.
//
// This lives in the compiled plugin rather than a shim built at import time so
// that the managed tensor's deleter is an ordinary C++ function: the consumer
// holds it for the tensor's whole life, including interpreter shutdown, which
// a ctypes callback does not survive.
//
// Only the description is here. Buffer allocation, the per-runtime mapping and
// the dtype/device policy stay in Python, where they belong.
//===----------------------------------------------------------------------===//

#ifndef AMD_TRITON_NPU_PYBIND11

void init_triton_amd_triton_npu(py::module_ &m) {
  m.def(
      "dlpack_ndarray",
      [](uintptr_t data, const std::vector<size_t> &shape, uint8_t code,
         uint8_t bits, int32_t deviceType, int32_t deviceId, py::object owner) {
        // `owner` is required, not optional: without one nanobind falls back to
        // copying the contents, which it cannot do for a bare pointer, and the
        // call fails outright. The Python side passes the buffer itself, so a
        // consumer's tensor keeps the pages it describes alive -- see
        // SharedBuffer._as_ndarray in shared.py for why, and for how the cycle
        // that would otherwise form is broken.
        return py::ndarray<py::array_api>(
            reinterpret_cast<void *>(data), shape.size(), shape.data(), owner,
            /*strides=*/nullptr, py::dlpack::dtype{code, bits, 1}, deviceType,
            deviceId);
      },
      py::arg("data"), py::arg("shape"), py::arg("code"), py::arg("bits"),
      py::arg("device_type"), py::arg("device_id"), py::arg("owner"),
      "Describe a device pointer as a DLPack array. The result implements "
      "__dlpack__ and __dlpack_device__; it borrows the memory from `owner`.");
}

#else // AMD_TRITON_NPU_PYBIND11

namespace {

// Layouts from dlpack.h, ABI 1.0. The versioned envelope puts the deleter
// before the tensor; the legacy one puts it after.

struct DLDevice {
  int32_t device_type;
  int32_t device_id;
};

struct DLDataType {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
};

struct DLTensor {
  void *data;
  DLDevice device;
  int32_t ndim;
  DLDataType dtype;
  int64_t *shape;
  int64_t *strides;
  uint64_t byte_offset;
};

struct DLManagedTensor {
  DLTensor dl_tensor;
  void *manager_ctx;
  void (*deleter)(DLManagedTensor *);
};

struct DLPackVersion {
  uint32_t major;
  uint32_t minor;
};

struct DLManagedTensorVersioned {
  DLPackVersion version;
  void *manager_ctx;
  void (*deleter)(DLManagedTensorVersioned *);
  uint64_t flags;
  DLTensor dl_tensor;
};

// Owned by a managed tensor: its shape, and a reference to the owner so that
// the consumer's tensor keeps the pages alive.
struct ManagerContext {
  std::vector<int64_t> shape;
  PyObject *owner;
};

// The consumer may call a deleter from any thread, without the GIL.
template <typename Managed> void deleteManaged(Managed *self) {
  if (!self)
    return;
  auto *ctx = static_cast<ManagerContext *>(self->manager_ctx);
  if (Py_IsInitialized()) {
    PyGILState_STATE state = PyGILState_Ensure();
    Py_XDECREF(ctx->owner);
    PyGILState_Release(state);
  }
  delete ctx;
  delete self;
}

// A consumer renames the capsule once it takes ownership. One that still has
// its original name when collected was never consumed, so it is ours to free.
// IsValid rather than GetPointer, which sets an exception on a name mismatch.
template <typename Managed>
void destroyCapsule(PyObject *capsule, const char *name) {
  if (!PyCapsule_IsValid(capsule, name))
    return;
  auto *managed = static_cast<Managed *>(PyCapsule_GetPointer(capsule, name));
  managed->deleter(managed);
}

void destroyLegacyCapsule(PyObject *capsule) {
  destroyCapsule<DLManagedTensor>(capsule, "dltensor");
}

void destroyVersionedCapsule(PyObject *capsule) {
  destroyCapsule<DLManagedTensorVersioned>(capsule, "dltensor_versioned");
}

class DLPackArray {
public:
  DLPackArray(uintptr_t data, std::vector<int64_t> shape, uint8_t code,
              uint8_t bits, int32_t deviceType, int32_t deviceId,
              py::object owner)
      : data(data), shape(std::move(shape)), code(code), bits(bits),
        deviceType(deviceType), deviceId(deviceId), owner(std::move(owner)) {}

  py::tuple dlpackDevice() const {
    return py::make_tuple(deviceType, deviceId);
  }

  // `stream` is accepted and ignored, as nanobind does; see the comment at the
  // top of the file.
  py::object dlpack(const py::object &stream, const py::object &maxVersion,
                    const py::object &dlDevice, const py::object &copy) const {
    if (!copy.is_none() && copy.cast<bool>())
      throw py::buffer_error("a copy was requested, but this buffer can only "
                             "be shared in place");
    if (!dlDevice.is_none()) {
      auto requested = dlDevice.cast<std::pair<int32_t, int32_t>>();
      if (requested.first != deviceType || requested.second != deviceId)
        throw py::buffer_error("dl_device does not match the device of this "
                               "buffer");
    }
    bool versioned =
        !maxVersion.is_none() &&
        maxVersion.cast<std::pair<uint32_t, uint32_t>>().first >= 1;
    return versioned
               ? makeCapsule<DLManagedTensorVersioned>("dltensor_versioned",
                                                       destroyVersionedCapsule)
               : makeCapsule<DLManagedTensor>("dltensor", destroyLegacyCapsule);
  }

private:
  template <typename Managed>
  py::object makeCapsule(const char *name,
                         PyCapsule_Destructor destructor) const {
    auto *ctx = new ManagerContext{shape, owner.ptr()};
    Py_INCREF(ctx->owner);

    auto *managed = new Managed{};
    if constexpr (std::is_same_v<Managed, DLManagedTensorVersioned>)
      managed->version = {1, 0};
    managed->manager_ctx = ctx;
    managed->deleter = deleteManaged<Managed>;
    DLTensor &t = managed->dl_tensor;
    t.data = reinterpret_cast<void *>(data);
    t.device = {deviceType, deviceId};
    t.ndim = static_cast<int32_t>(ctx->shape.size());
    t.dtype = {code, bits, 1};
    t.shape = ctx->shape.data();
    t.strides = nullptr; // compact row-major

    PyObject *capsule = PyCapsule_New(managed, name, destructor);
    if (!capsule) {
      managed->deleter(managed);
      throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(capsule);
  }

  uintptr_t data;
  std::vector<int64_t> shape;
  uint8_t code;
  uint8_t bits;
  int32_t deviceType;
  int32_t deviceId;
  py::object owner;
};

} // namespace

void init_triton_amd_triton_npu(py::module &&m) {
  py::class_<DLPackArray>(m, "DLPackArray")
      .def("__dlpack__", &DLPackArray::dlpack, py::arg("stream") = py::none(),
           py::arg("max_version") = py::none(),
           py::arg("dl_device") = py::none(), py::arg("copy") = py::none())
      .def("__dlpack_device__", &DLPackArray::dlpackDevice);

  m.def(
      "dlpack_ndarray",
      [](uintptr_t data, std::vector<int64_t> shape, uint8_t code, uint8_t bits,
         int32_t deviceType, int32_t deviceId, py::object owner) {
        return DLPackArray(data, std::move(shape), code, bits, deviceType,
                           deviceId, std::move(owner));
      },
      py::arg("data"), py::arg("shape"), py::arg("code"), py::arg("bits"),
      py::arg("device_type"), py::arg("device_id"), py::arg("owner"),
      "Describe a device pointer as a DLPack array. The result implements "
      "__dlpack__ and __dlpack_device__; it borrows the memory from `owner`.");
}

#endif // AMD_TRITON_NPU_PYBIND11
