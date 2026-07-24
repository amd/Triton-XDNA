// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Shared CPython glue for the generated NPU dispatch launchers (XRT xclbin/ELF
// and HSA). These helpers are signature-independent and identical across
// runtimes, so they live here instead of being emitted inline by each launcher
// generator (driver.py / hsa_driver.py).

#pragma once

#include <Python.h>
#include <stdbool.h>

// Result of resolving a kernel pointer argument: the raw device/host address and
// whether resolution succeeded (a Python error is set when `valid` is false).
struct DevicePtrInfo {
  void *dev_ptr;
  bool valid;
};

// Resolve a kernel pointer argument to a raw device address. Accepts a Python
// int (used directly), None (valid nullptr), or an object exposing data_ptr().
inline DevicePtrInfo getPointer(PyObject *obj, int idx) {
  (void)idx;
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(obj));
    return ptr_info;
  }
  if (obj == Py_None) {
    // valid nullptr
    return ptr_info;
  }
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if (ptr) {
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!ret) {
      // data_ptr() raised; the exception is already set. Do NOT touch `ret`.
      ptr_info.valid = false;
      return ptr_info;
    }
    if (!PyLong_Check(ret)) {
      Py_DECREF(ret);
      PyErr_SetString(PyExc_TypeError,
                      "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
    Py_DECREF(ret);
    return ptr_info;
  }
  PyErr_SetString(PyExc_TypeError,
                  "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}

// Total element count of a tensor-like object (product of obj.shape). Returns
// -1 and sets/print a Python error on failure.
inline long getNumElements(PyObject *obj) {
  PyObject *shape = PyObject_GetAttrString(obj, "shape");
  if (!shape) {
    PyErr_Print();
    return -1;
  }

  if (!PySequence_Check(shape)) {
    Py_DECREF(shape);
    PyErr_SetString(PyExc_TypeError, "Attribute 'shape' is not a sequence.");
    return -1;
  }

  Py_ssize_t ndim = PySequence_Size(shape);
  if (ndim < 0) {
    Py_DECREF(shape);
    PyErr_Print();
    return -1;
  }

  long num_elements = 1;
  for (Py_ssize_t i = 0; i < ndim; ++i) {
    PyObject *dim_obj = PySequence_GetItem(shape, i);
    if (!dim_obj) {
      Py_DECREF(shape);
      PyErr_Print();
      return -1;
    }

    long dim = PyLong_AsLong(dim_obj);
    Py_DECREF(dim_obj);

    if (dim == -1 && PyErr_Occurred()) {
      Py_DECREF(shape);
      PyErr_Print();
      return -1;
    }

    num_elements *= dim;
  }

  Py_DECREF(shape);
  return num_elements;
}

// Size in bytes of a single element of a tensor-like object (obj.dtype.itemsize).
// Returns -1 and sets/print a Python error on failure.
inline long getElementSizeInBytes(PyObject *obj) {
  if (!obj) return -1;

  PyObject *dtype = PyObject_GetAttrString(obj, "dtype");
  if (!dtype) {
    PyErr_Print();
    return -1;
  }

  PyObject *itemsize = PyObject_GetAttrString(dtype, "itemsize");
  Py_DECREF(dtype);
  if (!itemsize) {
    PyErr_Print();
    return -1;
  }

  long size = PyLong_AsLong(itemsize);
  Py_DECREF(itemsize);

  if (size == -1 && PyErr_Occurred()) {
    PyErr_Print();
    return -1;
  }

  return size;
}
