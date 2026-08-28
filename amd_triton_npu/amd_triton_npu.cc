//===- amd_triton_npu.cc -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace py = nanobind;

//===----------------------------------------------------------------------===//
// DLPack producer for shared multi-device buffers
//
// Backs ``amd_triton_npu/backend/shared.py``: a buffer whose pages one
// runtime allocated and another mapped is described to any DLPack consumer
// (torch, CuPy, JAX) as a tensor over the mapping that consumer can reach,
// with no copy.
//
// This lives in the compiled plugin rather than a shim built at import time so
// that the managed tensor's deleter and the capsule's destructor are ordinary
// C++ functions: both are held by the consumer for the tensor's whole life,
// including interpreter shutdown, which a ctypes callback does not survive.
//
// Only the capsule plumbing is here. Buffer allocation, the per-runtime
// mapping and the dtype/device policy stay in Python, where they belong.
//===----------------------------------------------------------------------===//

namespace {

// The six structs below are dlpack.h, ABI 1.0, duplicated rather than vendored:
// they are frozen by the standard, and a header dependency on dlpack for them
// would be heavier build coupling than the definitions are worth.

// Which device the buffer lives on. `device_type` is a kDL* constant, resolved
// by the Python side and passed in -- kDLROCM for a buffer an iGPU can reach,
// kDLCPU for one only the host and the NPU share.
struct DLDevice {
  int32_t device_type;
  int32_t device_id;
};

// Element type, as a (code, bit width) pair rather than a name -- kDLFloat/32
// is f32, kDLBfloat/16 is bf16, and so on. `lanes` is 1 for everything here.
struct DLDataType {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
};

// The tensor description proper. A NULL `strides` means compact row-major,
// which is the only layout produced here, and lets the consumer derive them.
struct DLTensor {
  void *data;
  DLDevice device;
  int32_t ndim;
  DLDataType dtype;
  int64_t *shape;
  int64_t *strides;
  uint64_t byte_offset;
};

// Legacy (pre-1.0) envelope: the tensor plus the callback that releases it.
// Carried in a capsule tagged "dltensor".
struct DLManagedTensor {
  DLTensor dl_tensor;
  void *manager_ctx;
  void (*deleter)(DLManagedTensor *);
};

// ABI version carried by the 1.0 envelope. Always {1, 0} here.
struct DLPackVersion {
  uint32_t major;
  uint32_t minor;
};

// DLPack 1.0 envelope, carried in a capsule tagged "dltensor_versioned".
// `flags` would carry e.g. the read-only bit; it is 0 here. Note the deleter
// sits *before* the tensor, unlike the legacy struct -- getting that order
// wrong silently corrupts whichever field the consumer reads.
struct DLManagedTensorVersioned {
  DLPackVersion version;
  void *manager_ctx;
  void (*deleter)(DLManagedTensorVersioned *);
  uint64_t flags;
  DLTensor dl_tensor;
};

// Both deleters free the envelope and its shape array and nothing else: the
// pages belong to whichever runtime the producer allocated them on, and that
// owner outlives every view handed out for them.

// Release a legacy managed tensor.
void deleteManagedTensor(DLManagedTensor *self) {
  if (!self)
    return;
  std::free(self->dl_tensor.shape);
  std::free(self);
}

// Release a 1.0 managed tensor. Same contract as the legacy deleter.
void deleteManagedTensorVersioned(DLManagedTensorVersioned *self) {
  if (!self)
    return;
  std::free(self->dl_tensor.shape);
  std::free(self);
}

// Capsule destructors.
//
// DLPack transfers ownership to the consumer, which renames the capsule to
// "used_dltensor" once it has taken it. A capsule still carrying its original
// name at collection time was never consumed -- a consumer that raised, or a
// caller that only inspected it -- so the producer still owns the managed
// tensor and must free it. Checking the name is what keeps this from
// double-freeing what the consumer already owns.
//
// Raw PyCapsule_New rather than nb::capsule for exactly that reason: nanobind's
// cleanup hook receives the pointer, not the capsule, so it cannot tell a
// consumed capsule from an abandoned one.

// Frees a "dltensor" capsule's managed tensor, if it still owns one.
void destroyCapsule(PyObject *capsule) {
  // IsValid rather than GetPointer + PyErr_Clear: GetPointer *sets* a
  // ValueError when the capsule has been renamed, and clearing it would also
  // discard any exception already propagating when this runs -- destructors
  // fire during GC, including inside except blocks.
  if (!PyCapsule_IsValid(capsule, "dltensor"))
    return; // renamed by the consumer: not ours to free
  deleteManagedTensor(
      static_cast<DLManagedTensor *>(PyCapsule_GetPointer(capsule, "dltensor")));
}

// Frees a "dltensor_versioned" capsule's managed tensor, if it still owns one.
void destroyCapsuleVersioned(PyObject *capsule) {
  if (!PyCapsule_IsValid(capsule, "dltensor_versioned"))
    return;
  deleteManagedTensorVersioned(static_cast<DLManagedTensorVersioned *>(
      PyCapsule_GetPointer(capsule, "dltensor_versioned")));
}

// Fill the DLTensor common to both flavours. Returns false if the shape copy
// could not be allocated.
bool fillTensor(DLTensor &t, uintptr_t data, const std::vector<int64_t> &shape,
                uint8_t code, uint8_t bits, int32_t deviceType,
                int32_t deviceId) {
  size_t ndim = shape.size();
  auto *shapeCopy =
      static_cast<int64_t *>(std::calloc(ndim ? ndim : 1, sizeof(int64_t)));
  if (!shapeCopy)
    return false;
  std::memcpy(shapeCopy, shape.data(), ndim * sizeof(int64_t));
  t.data = reinterpret_cast<void *>(data);
  t.device.device_type = deviceType;
  t.device.device_id = deviceId;
  t.ndim = static_cast<int32_t>(ndim);
  t.dtype.code = code;
  t.dtype.bits = bits;
  t.dtype.lanes = 1;
  t.shape = shapeCopy;
  // strides NULL means compact row-major; byte_offset stays 0 from the calloc.
  t.strides = nullptr;
  return true;
}

} // namespace

void init_triton_amd_triton_npu(py::module_ &m) {
  // dlpack_capsule(data, shape, code, bits, device_type, device_id, versioned)
  //
  // Allocate a managed tensor describing `data` and hand it back in the capsule
  // flavour the consumer negotiated. The Python side owns dtype and device
  // policy and passes the results in already resolved; nothing here inspects
  // the buffer.
  m.def(
      "dlpack_capsule",
      [](uintptr_t data, const std::vector<int64_t> &shape, uint8_t code,
         uint8_t bits, int32_t deviceType, int32_t deviceId,
         bool versioned) -> py::object {
        PyObject *capsule = nullptr;
        if (versioned) {
          auto *mt = static_cast<DLManagedTensorVersioned *>(
              std::calloc(1, sizeof(DLManagedTensorVersioned)));
          if (!mt)
            return py::none();
          if (!fillTensor(mt->dl_tensor, data, shape, code, bits, deviceType,
                          deviceId)) {
            std::free(mt);
            return py::none();
          }
          mt->version.major = 1;
          mt->version.minor = 0;
          mt->deleter = deleteManagedTensorVersioned;
          capsule =
              PyCapsule_New(mt, "dltensor_versioned", destroyCapsuleVersioned);
          if (!capsule)
            deleteManagedTensorVersioned(mt);
        } else {
          auto *mt = static_cast<DLManagedTensor *>(
              std::calloc(1, sizeof(DLManagedTensor)));
          if (!mt)
            return py::none();
          if (!fillTensor(mt->dl_tensor, data, shape, code, bits, deviceType,
                          deviceId)) {
            std::free(mt);
            return py::none();
          }
          mt->deleter = deleteManagedTensor;
          capsule = PyCapsule_New(mt, "dltensor", destroyCapsule);
          if (!capsule)
            deleteManagedTensor(mt);
        }
        if (!capsule) {
          // PyCapsule_New set an exception; returning a value with one live
          // would surface as SystemError instead of the real MemoryError, and
          // the Python side turns None into its own clear message.
          PyErr_Clear();
          return py::none();
        }
        return py::steal<py::object>(capsule);
      },
      py::arg("data"), py::arg("shape"), py::arg("code"), py::arg("bits"),
      py::arg("device_type"), py::arg("device_id"), py::arg("versioned"),
      "Wrap a device pointer in a DLPack capsule. Returns a 'dltensor' or "
      "'dltensor_versioned' capsule, or None if allocation failed. The "
      "consumer takes ownership; an unconsumed capsule frees itself.");
}
