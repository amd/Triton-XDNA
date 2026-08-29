//===- amd_triton_npu.cc -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace py = nanobind;

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
// This lives in the compiled plugin rather than a shim built at import time so
// that the managed tensor's deleter is an ordinary C++ function: the consumer
// holds it for the tensor's whole life, including interpreter shutdown, which
// a ctypes callback does not survive.
//
// Only the description is here. Buffer allocation, the per-runtime mapping and
// the dtype/device policy stay in Python, where they belong.
//===----------------------------------------------------------------------===//

void init_triton_amd_triton_npu(py::module_ &m) {
  m.def(
      "dlpack_ndarray",
      [](uintptr_t data, const std::vector<size_t> &shape, uint8_t code,
         uint8_t bits, int32_t deviceType, int32_t deviceId, py::object owner) {
        // `owner` is required, not optional: without one nanobind falls back to
        // copying the contents, which it cannot do for a bare pointer, and the
        // call fails outright. The Python side passes a sentinel rather than
        // the buffer -- see _NDARRAY_OWNER in shared.py for why the descriptive
        // choice leaks.
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
