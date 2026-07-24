# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""HSA/ROCR launcher code generation for the Triton-XDNA NPU backend.

Generates a thin C++ CPython launcher that dispatches Triton-generated NPU
kernels through ROCR. The launcher contains only the signature-specific CPython
glue (argument parsing, pointer/size resolution) and delegates all HSA work to
the shared runtime library ``libtriton_npu_hsa.so`` (see
``include/HsaRuntime/HsaRuntime.{h,cpp}``) via a small C ABI:

* ``triton_npu_hsa_prepare(pdi, insts)`` -> opaque program handle (once, from
  ``set_paths``).
* ``triton_npu_hsa_dispatch(program, n, ptrs, sizes)`` (per launch).

Because every launcher links the same shared library, the ``HsaRuntime``
singleton behind those calls is process-global: one ``hsa_init``, one queue, one
completion signal, one kernarg pool, and one vmem buffer pool are shared across
all kernel signatures. That is what allows multiple signatures in one process on
an AIE agent that permits only one queue (``QUEUES_MAX == 1``).

Used by ``NPULauncher`` in ``driver.py`` when the driver's runtime is "hsa"
(``NPUDriver("hsa")`` or ``AMD_TRITON_NPU_RUNTIME=hsa``). See
docs/hsa-zero-copy-notes.md for the deferred zero-copy data path.
"""

from .codegen import extracted_type, format_of


def _generate_hsa_launcher(constants, signature, _kernel_name) -> str:
    """Generate the thin C++ CPython launcher that dispatches via HSA/ROCR.

    The generated module exposes ``set_paths(pdi_path, insts_path)`` (which calls
    ``triton_npu_hsa_prepare`` once and stashes the handle) and ``launch(...)``
    (which marshals the tensor pointers/sizes and calls
    ``triton_npu_hsa_dispatch`` with the GIL released). All HSA state lives in
    the shared runtime library, not in this per-signature module.

    ``_kernel_name`` is accepted for signature parity with the XRT launcher
    generators but is unused: the HSA path selects work by PDI/insts address,
    not by a kernel symbol name.
    """
    args_format = "".join(format_of(extracted_type(ty)) for ty in signature.values())
    fmt = "iiiOOOO" + args_format
    args_list = (
        ", " + ", ".join(f"&_arg{i}" for i in signature) if len(signature) > 0 else ""
    )

    # Pointer (tensor) args excluding constexpr constants -- the kernel arguments.
    ptr_args = [
        (i, ty) for i, ty in signature.items() if i not in constants and ty[0] == "*"
    ]
    num_ptr_args = len(ptr_args)
    # C arrays can't be zero-length; size for at least 1 (unused when 0 tensors).
    arr_len = max(num_ptr_args, 1)

    arg_decls = " ".join(
        f"{extracted_type(ty)} _arg{i}; " for i, ty in signature.items()
    )
    # Silence unused-variable warnings for args we parse but don't dispatch
    # (scalars and constexprs); the tensor-pointer args are used via getPointer.
    used = {i for i, _ in ptr_args}
    void_casts = " ".join(f"(void)_arg{i};" for i in signature if i not in used)

    ptr_info_lines = "\n  ".join(
        f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); "
        f"if (!ptr_info{i}.valid) return NULL;"
        for i, _ in ptr_args
    )
    # Check each factor for the -1 error sentinel separately: a single
    # `product == -1` check misses e.g. nelem==-1 with itemsize>1.
    vol_lines = "\n  ".join(
        f"long nelem{i} = getNumElements(_arg{i}); "
        f"long ebytes{i} = getElementSizeInBytes(_arg{i}); "
        f"if (nelem{i} == -1 || ebytes{i} == -1) return NULL; "
        f"long tensor_volume{i} = nelem{i} * ebytes{i};"
        for i, _ in ptr_args
    )
    fill_arrays = "\n      ".join(
        f"host_ptrs[{pos}] = ptr_info{i}.dev_ptr; "
        f"sizes[{pos}] = (std::uint64_t)tensor_volume{i};"
        for pos, (i, _) in enumerate(ptr_args)
    )

    return f"""
#include <Python.h>
#include <cstdint>

#include "npu_dispatch_common.h"
#include "HsaRuntime/HsaRuntime.h"

// Number of tensor kernel arguments for this specialized launcher.
static constexpr std::uint32_t NUM_KERNARGS = {num_ptr_args};

// Handle to this module's prepared (pdi, insts) program in the shared runtime.
static triton_npu_hsa_program_t g_program = nullptr;

// Python-callable set_paths(pdi_path, insts_path): prepare (load + cache) the
// program in the shared runtime and stash its handle. Called once per module.
static PyObject* py_set_paths(PyObject* self, PyObject* args) {{
  const char* pdi;
  const char* insts;
  if (!PyArg_ParseTuple(args, "ss", &pdi, &insts)) {{
    return NULL;
  }}
  char err[512];
  g_program = triton_npu_hsa_prepare(pdi, insts, err, sizeof(err));
  if (g_program == nullptr) {{
    PyErr_SetString(PyExc_RuntimeError, err);
    return NULL;
  }}
  Py_RETURN_NONE;
}}

// Python-callable launch(gridX, gridY, gridZ, kernel_metadata, launch_metadata,
// enter_hook, exit_hook, *args): parse the arguments, run the enter/exit hooks,
// resolve each tensor's device pointer and byte size, and dispatch through the
// shared runtime (releasing the GIL for the duration of the device work).
static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject* launch_enter_hook = NULL;
  PyObject* launch_exit_hook = NULL;
  PyObject* kernel_metadata = NULL;
  PyObject* launch_metadata = NULL;
  {arg_decls}
  if (!PyArg_ParseTuple(args, \"{fmt}\", &gridX, &gridY, &gridZ,
                        &kernel_metadata, &launch_metadata,
                        &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}
  {void_casts}

  if (launch_enter_hook != Py_None) {{
    PyObject* hook_args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, hook_args);
    Py_DECREF(hook_args);
    if (!ret) return NULL;
    Py_DECREF(ret);
  }}

  {ptr_info_lines}
  {vol_lines}

  if (gridX * gridY * gridZ > 0) {{
    void* host_ptrs[{arr_len}];
    std::uint64_t sizes[{arr_len}];
    {fill_arrays}
    char err[512];
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = triton_npu_hsa_dispatch(g_program, NUM_KERNARGS, host_ptrs, sizes,
                                 err, sizeof(err));
    Py_END_ALLOW_THREADS
    if (rc != 0) {{
      PyErr_SetString(PyExc_RuntimeError, err);
      return NULL;
    }}
  }}

  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if (launch_exit_hook != Py_None) {{
    PyObject* hook_args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, hook_args);
    Py_DECREF(hook_args);
    if (!ret) return NULL;
  }}

  Py_INCREF(Py_None);
  return Py_None;
}}

// Methods exported by this per-signature dispatch extension module.
static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{"set_paths", py_set_paths, METH_VARARGS, "Set paths to aie.pdi and insts.bin"}},
  {{NULL, NULL, 0, NULL}}  // sentinel
}};

// CPython module definition for the "__npu_dispatch" extension.
static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__npu_dispatch\",
  NULL,  // documentation
  -1,    // size
  ModuleMethods
}};

// Module init entry point invoked by CPython when the extension is imported.
PyMODINIT_FUNC PyInit___npu_dispatch(void) {{
  PyObject* m = PyModule_Create(&ModuleDef);
  if (m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
