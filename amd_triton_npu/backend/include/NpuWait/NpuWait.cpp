// Wait for a pyxrt.run without holding the GIL.
//
// pyxrt's own run.wait / run.wait2 block in the driver with the GIL held, so
// every other Python thread stops for the whole NPU command. A GPU wait in
// torch releases it. This does the same for the NPU: it takes the xrt::run
// behind a pyxrt.run and calls wait2 with the GIL released.
//
// The xrt::run is obtained through pybind11's cpp conduit
// (`_pybind11_conduit_v1_`), which hands the pointer over only when the
// caller names the same platform ABI as the module that owns it. pybind11
// spells the ABI of a g++/libstdc++ build one way before its 3.0 and another
// way from it, so both spellings are formed for the compiler building this
// file and tried in turn. When neither matches pyxrt's, pyxrt declines and
// the caller falls back to pyxrt's own wait.

#include <Python.h>

#include <chrono>
#include <exception>
#include <string>
#include <thread>
#include <typeinfo>

#include "xrt/xrt_kernel.h"

#if !defined(__GNUC__) || defined(__clang__) || !defined(__GLIBCXX__)
#error "NpuWait is built with g++ against libstdc++"
#endif

// pybind11's spellings of this compiler's ABI, as its
// pybind11_platform_abi_id.h forms them: before 3.0, then from 3.0 on.
static const std::string abi_ids[] = {
    "_gcc_libstdcpp_cxxabi" + std::to_string(__GXX_ABI_VERSION),
    "system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_" +
        std::to_string(_GLIBCXX_USE_CXX11_ABI),
};

// The xrt::run behind `obj` if pyxrt hands it over under `id`, else nullptr
// with no Python error set.
static xrt::run *as_run(PyObject *obj, const std::string &id) {
  PyObject *ti = PyCapsule_New(const_cast<std::type_info *>(&typeid(xrt::run)),
                               typeid(std::type_info).name(), nullptr);
  if (!ti)
    return nullptr;
  PyObject *cap =
      PyObject_CallMethod(obj, "_pybind11_conduit_v1_", "y#Oy", id.data(),
                          (Py_ssize_t)id.size(), ti, "raw_pointer_ephemeral");
  Py_DECREF(ti);
  if (!cap) {
    PyErr_Clear();
    return nullptr;
  }
  void *ptr = nullptr;
  if (PyCapsule_CheckExact(cap))
    ptr = PyCapsule_GetPointer(cap, PyCapsule_GetName(cap));
  Py_DECREF(cap);
  if (!ptr)
    PyErr_Clear();
  return static_cast<xrt::run *>(ptr);
}

static xrt::run *as_run(PyObject *obj) {
  for (const std::string &id : abi_ids)
    if (xrt::run *run = as_run(obj, id))
      return run;
  return nullptr;
}

static PyObject *py_supported(PyObject *, PyObject *args) {
  PyObject *obj;
  if (!PyArg_ParseTuple(args, "O", &obj))
    return nullptr;
  return PyBool_FromLong(as_run(obj) != nullptr);
}

// Runs `fn` with the GIL released and returns what it threw, if anything.
// Every blocking call here goes through this, so other Python threads run
// while it waits.
template <typename F> static std::string without_gil(F fn) {
  std::string err;
  PyThreadState *state = PyEval_SaveThread();
  try {
    fn();
  } catch (const std::exception &e) {
    err = e.what();
  }
  PyEval_RestoreThread(state);
  return err;
}

// Same contract as pyxrt's wait2: returns once the run completed, raises
// otherwise.
static PyObject *py_wait(PyObject *, PyObject *args) {
  PyObject *obj;
  if (!PyArg_ParseTuple(args, "O", &obj))
    return nullptr;
  // `obj` is borrowed from the argument tuple, which outlives the wait, so the
  // ephemeral pointer stays valid.
  xrt::run *run = as_run(obj);
  if (!run) {
    PyErr_SetString(PyExc_TypeError,
                    "pyxrt.run does not share this module's C++ ABI");
    return nullptr;
  }
  std::string err = without_gil([run] { run->wait2(); });
  if (!err.empty()) {
    PyErr_SetString(PyExc_RuntimeError, err.c_str());
    return nullptr;
  }
  Py_RETURN_NONE;
}

// Test hook: blocks for `ms` milliseconds the way `wait` blocks, so a test can
// check that other threads run meanwhile without needing an NPU command.
static PyObject *py_block(PyObject *, PyObject *args) {
  int ms;
  if (!PyArg_ParseTuple(args, "i", &ms))
    return nullptr;
  without_gil(
      [ms] { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); });
  Py_RETURN_NONE;
}

static PyMethodDef Methods[] = {
    {"supported", py_supported, METH_VARARGS,
     "Whether this pyxrt.run can be waited on here"},
    {"wait", py_wait, METH_VARARGS, "pyxrt.run.wait2 without the GIL"},
    {"_block", py_block, METH_VARARGS, "Block like wait, for tests"},
    {nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef Module = {PyModuleDef_HEAD_INIT, "npu_wait", nullptr,
                                    -1, Methods};

PyMODINIT_FUNC PyInit_npu_wait(void) { return PyModule_Create(&Module); }
