# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import hashlib
import tempfile
import sys
import sysconfig

import os, subprocess, tempfile, platform
import importlib.util
import shutil

from pathlib import Path

from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget

import aie.compiler.aiecc.main as aiecc
import air.compiler.aircc.main as aircc
from air.compiler.util import run_transform
from air.ir import *
import air.passmanager

autotune_time = False


# -------------------- Launcher ----------------------------
def _ty_to_cpp(ty):
    if ty[0] == "*":
        return "void*"
    if ty == "constexpr":
        return "PyObject*"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "bfloat16",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def _extracted_type(ty):
    if ty[0] == "*":
        return "PyObject*"
    if ty == "constexpr":
        return "PyObject*"
    return _ty_to_cpp(ty)


def _format_of(ty):
    return {
        "PyObject*": "O",
        "constexpr": "O",
        "float": "f",
        "double": "d",
        "long": "l",
        "int8_t": "b",
        "int16_t": "h",
        "int32_t": "i",
        "int64_t": "l",
        "uint8_t": "B",
        "uint16_t": "H",
        "uint32_t": "I",
        "uint64_t": "K",
    }[ty]


def _get_air_opt_path() -> str:
    """
    Get the path to air-opt binary from pip-installed mlir-air package.

    Uses the aircc module's location to find the mlir_air package root,
    then locates the air-opt binary in the bin/ directory.

    Returns:
        str: Path to air-opt binary

    Raises:
        RuntimeError: If air-opt binary not found
    """
    # aircc.__file__ gives: /path/to/mlir_air/python/air/compiler/aircc/main.py
    # We need: /path/to/mlir_air/bin/air-opt
    aircc_path = Path(aircc.__file__).resolve()
    # Navigate from .../mlir_air/python/air/compiler/aircc/main.py to .../mlir_air/
    mlir_air_root = aircc_path.parent.parent.parent.parent.parent
    air_opt_path = mlir_air_root / "bin" / "air-opt"

    if not air_opt_path.exists():
        raise RuntimeError(f"Could not find air-opt binary at {air_opt_path}")

    return str(air_opt_path)


def _get_xrt_path() -> str:
    path = os.getenv("XILINX_XRT", "")
    if path == "":
        raise Exception("XILINX_XRT is not set. Is xrt installed in system?")
    return path


def _get_aie_test_utils_path() -> str:
    path = (
        Path(aiecc.__file__).parent.parent.parent.parent.parent
        / "runtime_lib"
        / "x86_64"
        / "test_lib"
    )
    return path


def _get_air_project_path() -> Path:
    """
    Get the path for air_project directory.

    If AMD_TRITON_NPU_AIR_PROJECT_PATH is set, use that path.
    Otherwise, default to 'air_project' in the current working directory.

    Returns:
        Path: The path to the air_project directory
    """
    custom_path = os.getenv("AMD_TRITON_NPU_AIR_PROJECT_PATH")
    if custom_path:
        return Path(custom_path)
    return Path(os.getcwd()) / "air_project"


def _dump_ir_if_needed(files):
    """
    Dump intermediate IR files to the air_project directory.

    Files are always dumped to the air_project path (controlled by
    AMD_TRITON_NPU_AIR_PROJECT_PATH or defaulting to ./air_project/).
    """
    air_proj_path = _get_air_project_path()
    os.makedirs(air_proj_path, exist_ok=True)
    for f in files:
        shutil.copy(f, os.path.join(air_proj_path, os.path.basename(f)))


def get_npu_device_info():
    try:
        import re

        result = subprocess.run(
            ["xrt-smi", "examine"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        output = result.stdout

        # Match either one or two pipes with optional whitespace around them
        device_pattern = re.compile(
            r"\[(?P<bdf>[0-9a-fA-F:.]+)\]\s*\|{1,2}\s*(?P<name>.+?)\s*\|"
        )

        matches = device_pattern.findall(output)

        devices = []
        for bdf, name in matches:
            devices.append({"bdf": bdf, "name": name.strip()})

        return devices

    except subprocess.CalledProcessError as e:
        print("Failed to run xrt-smi:", e.stderr)
        return []
    except Exception as e:
        print("Unexpected error:", str(e))
        return []


def detect_npu_version():
    """Map known device names to internal NPU version strings."""
    devices = get_npu_device_info()
    for device in devices:
        name = device["name"]
        if "RyzenAI-npu1" in name:
            return "npu1"
        elif "NPU Phoenix" in name:
            return "npu1"
        elif "Strix" in name:
            return "npu2"
    raise RuntimeError("Unsupported or unrecognized NPU device found.")


def _get_transform_ir_string():
    """
    Get the transform IR string for tiling operations.

    If the environment variable AIR_TRANSFORM_TILING_SCRIPT is set,
    read the transform IR from that file. Otherwise, use the default
    hardcoded transform IR string.

    Returns:
        str: The transform IR string to use for tiling
    """
    custom_script_path = os.getenv("AIR_TRANSFORM_TILING_SCRIPT")

    if custom_script_path:
        if not os.path.isfile(custom_script_path):
            raise FileNotFoundError(
                f"AIR_TRANSFORM_TILING_SCRIPT is set to '{custom_script_path}' "
                f"but the file was not found (cwd: {os.getcwd()}). "
                f"Use an absolute path or run from the directory containing the script."
            )
        with open(custom_script_path, "r") as f:
            print(f"Using custom tiling script from: {custom_script_path}")
            return f.read()

    # Default hardcoded transform IR string
    matmul_tiling_size_l1_m = 32
    matmul_tiling_size_l1_n = 32
    matmul_tiling_size_l1_k = 32
    elemwise_tiling_size_l1_m = 32
    elemwise_tiling_size_l1_n = 32

    return f"""
        module attributes {{transform.with_named_sequence}} {{
          transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
                %mul = transform.structured.match ops{{["linalg.mul"]}} in %arg1  : (!transform.any_op) -> !transform.any_op
                %mul_1, %loop = transform.air.linalg_tile %mul [{elemwise_tiling_size_l1_m}, {elemwise_tiling_size_l1_n}]
                transform.air.linalg_promote %mul_1 {{"operands_to_promote"=[2], "memory_space"="L1"}}
                transform.air.linalg_promote %mul_1 {{"operands_to_promote"=[0,1], "memory_space"="L1"}}

                %add = transform.structured.match ops{{["linalg.add"]}} in %arg1  : (!transform.any_op) -> !transform.any_op
                %add_1, %add_loop = transform.air.linalg_tile %add [{elemwise_tiling_size_l1_m}, {elemwise_tiling_size_l1_n}]
                transform.air.linalg_promote %add_1 {{"operands_to_promote"=[2], "memory_space"="L1"}}
                transform.air.linalg_promote %add_1 {{"operands_to_promote"=[0,1], "memory_space"="L1"}}

                %matmul = transform.structured.match ops{{["linalg.matmul"]}} in %arg1  : (!transform.any_op) -> !transform.any_op
                %fill = transform.structured.match ops{{["linalg.fill"]}} in %arg1  : (!transform.any_op) -> !transform.any_op
                %matmul_1, %matmul_loop = transform.air.linalg_tile %matmul [{matmul_tiling_size_l1_m}, {matmul_tiling_size_l1_n}]
                %fill_1 = transform.air.fuse_into_containing_op %fill into %matmul_loop
                transform.air.linalg_promote %fill_1 {{"operands_to_promote"=[1], "memory_space"="L1"}}
                transform.air.linalg_promote %matmul_1 {{"operands_to_promote"=[2], "memory_space"="L1"}}
                %matmul_2, %reduction_loop = transform.air.linalg_tile %matmul_1 [0, 0, {matmul_tiling_size_l1_k}]
                transform.air.linalg_promote %matmul_2 {{"operands_to_promote"=[0,1], "memory_space"="L1"}}
            transform.yield
          }}
        }}
        """


def _ttshared_to_air(mod, gridX, gridY, gridZ):
    # Get Triton-Shared-MLIR as string
    with tempfile.TemporaryDirectory() as tmpdir:
        dst_path = os.path.join(tmpdir, "airinput.mlir")
        air_opt_path = _get_air_opt_path()
        # MLIR-AIR compilation step 1: mapping grid to air.launch
        pipeline = (
            "builtin.module("
            + ",".join(
                [
                    "air-resolve-tensor-opoperand-conflicts",
                    "air-override-memref-memory-space{scope=func memory-space=1}",
                ]
            )
            + ")"
        )
        air_context = air.ir.Context()
        air_module = Module.parse(mod, context=air_context)
        pm = air.passmanager.PassManager.parse(pipeline, context=air_context)
        pm.run(air_module.operation)
        # MLIR-AIR compilation step 2: tiling the launch body
        transform_ir_string = _get_transform_ir_string()
        transform_ir = Module.parse(transform_ir_string, context=air_context)
        run_transform(transform_ir, air_module)
        # MLIR-AIR compilation step 3: converting to AIR
        pipeline = (
            "builtin.module("
            + ",".join(
                [
                    "func.func(air-wrap-func-with-parallel{loop-bounds="
                    + str(gridX)
                    + ","
                    + str(gridY)
                    + ","
                    + str(gridZ)
                    + "})",
                    "air-par-to-launch{depth=0 has-air-segment=true}",
                    "canonicalize",
                    "cse",
                    "air-copy-to-dma",
                ]
            )
            + ")"
        )
        pm = air.passmanager.PassManager.parse(pipeline, context=air_context)
        pm.run(air_module.operation)
        with open(dst_path, "w") as f:
            f.write(str(air_module))
        _dump_ir_if_needed([dst_path])
        return air_module


def _generate_launcher(constants, signature, kernel_name):
    arg_decls = ", ".join(f"{_ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
    args_format = "".join(
        [_format_of(_extracted_type(ty)) for ty in signature.values()]
    )
    format = "iiiOOOO" + args_format
    args_list = (
        ", " + ", ".join(f"&_arg{i}" for i, ty in signature.items())
        if len(signature) > 0
        else ""
    )

    kernel_arg_decls = ", ".join(
        _ty_to_cpp(ty) if ty[0] != "*" else f"int64_t, void*"
        for i, ty in signature.items()
        if ty != "constexpr"
    )
    kernel_arg_decls += ", " if kernel_arg_decls else ""

    kernel_parameters = ", ".join(
        f"static_cast<{_ty_to_cpp(ty)}>(arg{i})" if ty[0] != "*" else f"0, &ptr_arg{i}"
        for i, ty in signature.items()
        if ty != "constexpr"
    )
    kernel_parameters += ", " if kernel_parameters else ""

    global autotune_time

    return f"""
#include <assert.h>
#include <fstream>
#include <iostream>
#include <stdbool.h>
#include <Python.h>
#include "ExecutionEngine/CRunnerUtils.h"
#include "ExecutionEngine/CRunnerUtils.cpp"

#include "test_utils.h"

#include <chrono>
#include <cstdlib>
#include <ctime>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

static char aie_path[1024] = {{0}};
static char insts_path[1024] = {{0}};

static PyObject* py_set_paths(PyObject* self, PyObject* args) {{
    const char* aie;
    const char* insts;

    if (!PyArg_ParseTuple(args, "ss", &aie, &insts)) {{
        return NULL;
    }}

    strncpy(aie_path, aie, sizeof(aie_path) - 1);
    strncpy(insts_path, insts, sizeof(insts_path) - 1);
    aie_path[sizeof(aie_path) - 1] = '\\0';
    insts_path[sizeof(insts_path) - 1] = '\\0';

    Py_RETURN_NONE;
}}

// Call to XRT goes here:
static void _launch(int gridX, int gridY, int gridZ, {', '.join(f"long size{i}" for i, ty in signature.items() if i not in constants and ty[0]=="*")}, {arg_decls}) {{
  if (gridX*gridY*gridZ > 0) {{
    std::vector<uint32_t> instr_v =
        test_utils::load_instr_binary(insts_path);

    int verbosity = 1;
    if (verbosity >= 1)
        std::cout << "Sequence instr count: " << instr_v.size() << std::endl;

    // Start the XRT test code
    // Get a device handle
    unsigned int device_index = 0;
    auto device = xrt::device(device_index);

    // Load the xclbin
    if (verbosity >= 1)
        std::cout << "Loading xclbin." << std::endl;
    auto xclbin = xrt::xclbin(std::string(aie_path));

    if (verbosity >= 1)
        std::cout << "Kernel opcode: " << "MLIR_AIE" << std::endl;
    std::string Node = "MLIR_AIE";

    // Get the kernel from the xclbin
    auto xkernels = xclbin.get_kernels();
    auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                                    [Node](xrt::xclbin::kernel &k) {{
                                    auto name = k.get_name();
                                    std::cout << "Name: " << name << std::endl;
                                    return name.rfind(Node, 0) == 0;
                                    }});
    auto kernelName = xkernel.get_name();

    if (verbosity >= 1)
        std::cout << "Registering xclbin." << std::endl;

    device.register_xclbin(xclbin);

    // get a hardware context
    if (verbosity >= 1)
        std::cout << "Getting hardware context." << std::endl;
    xrt::hw_context context(device, xclbin.get_uuid());

    // get a kernel handle
    if (verbosity >= 1)
        std::cout << "Getting handle to kernel:" << kernelName << std::endl;
    auto kernel = xrt::kernel(context, kernelName);

    // get instruction sequence
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    
    {' '.join(f'auto bo_{i} = xrt::bo(device, size{i}, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id({i+3}));' for i, ty in signature.items() if i not in constants and ty[0] == "*")}

    if (verbosity >= 1)
        std::cout << "Writing data into buffer objects." << std::endl;
    {' '.join(f'void *buf{i} = bo_{i}.map<void *>(); memcpy(buf{i}, arg{i}, size{i});' for i, ty in signature.items() if i not in constants and ty[0] == "*")}

    void *bufInstr = bo_instr.map<void *>();
    memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    {' '.join(f'bo_{i}.sync(XCL_BO_SYNC_BO_TO_DEVICE);' for i, ty in signature.items() if i not in constants and ty[0] == "*")}

    if (verbosity >= 1)
        std::cout << "Running Kernel." << std::endl;
    unsigned int opcode = 3;
    {'auto start = std::chrono::high_resolution_clock::now();' if autotune_time else ''} 
    auto run = kernel(opcode, bo_instr, instr_v.size(), {','.join(f'bo_{i}' for i, ty in signature.items() if i not in constants and ty[0] == "*")});
    run.wait();
    {'auto stop = std::chrono::high_resolution_clock::now(); float npu_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();' if autotune_time else ''}

    {'std::ofstream file("data.txt"); file << npu_time << std::endl; file.close();' if autotune_time else ''}

    if (verbosity >= 1)
        std::cout << "Copying results." << std::endl;
    // TODO: Assuming the last tensor is the only output tensor.
    bo_{next((i for i, ty in reversed(signature.items()) if i not in constants and ty[0] == "*"), None)}.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    memcpy(arg{next((i for i, ty in reversed(signature.items()) if i not in constants and ty[0] == "*"), None)}, buf{next((i for i, ty in reversed(signature.items()) if i not in constants and ty[0] == "*"), None)}, size{next((i for i, ty in reversed(signature.items()) if i not in constants and ty[0] == "*"), None)});

    if (verbosity >= 1)
        std::cout << "Launch finished." << std::endl;
  }}
}}

typedef struct _DevicePtrInfo {{
  void *dev_ptr;
  bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(obj));
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
    if(!ptr_info.dev_ptr)
      return ptr_info;
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}}

long getNumElements(PyObject *obj) {{
    PyObject *shape = PyObject_GetAttrString(obj, "shape");
    if (!shape) {{
        PyErr_Print();
        return -1;
    }}

    if (!PySequence_Check(shape)) {{
        Py_DECREF(shape);
        PyErr_SetString(PyExc_TypeError, "Attribute 'shape' is not a sequence.");
        return -1;
    }}

    Py_ssize_t ndim = PySequence_Size(shape);
    if (ndim < 0) {{
        Py_DECREF(shape);
        PyErr_Print();
        return -1;
    }}

    long num_elements = 1;
    for (Py_ssize_t i = 0; i < ndim; ++i) {{
        PyObject *dim_obj = PySequence_GetItem(shape, i);
        if (!dim_obj) {{
            Py_DECREF(shape);
            PyErr_Print();
            return -1;
        }}

        long dim = PyLong_AsLong(dim_obj);
        Py_DECREF(dim_obj);

        if (dim == -1 && PyErr_Occurred()) {{
            Py_DECREF(shape);
            PyErr_Print();
            return -1;
        }}

        num_elements *= dim;
    }}

    Py_DECREF(shape);
    return num_elements;
}}

long getElementSizeInBytes(PyObject *obj) {{
    if (!obj) return -1;

    PyObject *dtype = PyObject_GetAttrString(obj, "dtype");
    if (!dtype) {{
        PyErr_Print();
        return -1;
    }}

    PyObject *itemsize = PyObject_GetAttrString(dtype, "itemsize");
    Py_DECREF(dtype);
    if (!itemsize) {{
        PyErr_Print();
        return -1;
    }}

    long size = PyLong_AsLong(itemsize);
    Py_DECREF(itemsize);

    if (size == -1 && PyErr_Occurred()) {{
        PyErr_Print();
        return -1;
    }}

    return size;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}

  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  {"; ".join([f"long tensor_volume{i} = getNumElements(_arg{i}) * getElementSizeInBytes(_arg{i}); if (tensor_volume{i} == -1) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  _launch(gridX, gridY, gridZ, {', '.join(f"tensor_volume{i}" for i, ty in signature.items() if i not in constants and ty[0]=="*")}, {', '.join(f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items())});

  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{"set_paths", py_set_paths, METH_VARARGS, "Set paths to aie.bin and insts.bin"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__npu_dispatch\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___npu_dispatch(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""


def compile_module(launcher_src, kernel_placeholder_name):
    py_version = sys.version_info
    if platform.system() == "Windows":
        py_include_dir = os.path.join(sys.base_prefix, "include")
        py_lib_dir = os.path.join(sys.base_prefix, "libs")
        py_lib = "{name}{major}{minor}.lib".format(
            name="python", major=py_version.major, minor=py_version.minor
        )
    else:
        py_include_dir = os.path.join(
            sys.base_prefix,
            "include",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
        )
        py_lib_dir = os.path.join(sys.base_prefix, "lib")
        py_lib = "{name}{major}.{minor}".format(
            name="python", major=py_version.major, minor=py_version.minor
        )
    npu_backend_path = Path(__file__).resolve().parent
    include_dir = os.path.join(npu_backend_path, "include")
    xrt_dir = _get_xrt_path()
    aie_test_utils_dir = _get_aie_test_utils_path()

    def launch(
        gridX,
        gridY,
        gridZ,
        stream,
        cu_function,
        kernel_metadata,
        launch_metadata,
        launch_enter_hook,
        launch_exit_hook,
        *args,
    ):
        asm_src = cu_function
        kernel_name = kernel_metadata[6]  # see pack_metadata in compiler.py
        src = launcher_src.replace(kernel_placeholder_name, kernel_name)

        # Get air_project path (controlled by AMD_TRITON_NPU_AIR_PROJECT_PATH
        # or defaults to ./air_project/)
        air_proj_path = _get_air_project_path()
        os.makedirs(air_proj_path, exist_ok=True)
        Path(os.path.join(air_proj_path, "asm_src.mlir")).write_bytes(asm_src)
        air_output = _ttshared_to_air(asm_src, gridX, gridY, gridZ)
        with open(Path(os.path.join(air_proj_path, "asm_air_output.mlir")), "w") as f:
            f.write(str(air_output))

        key_data = str(air_output) + f"_timing_{autotune_time}"
        key = hashlib.md5(key_data.encode("utf-8")).hexdigest()

        cache = get_cache_manager(key)
        name = "__npu_dispatch"
        filename = f"{name}.so"
        cache_path = cache.get_file(filename)
        cache_xclbin_path = cache.get_file("aie.xclbin")
        cache_insts_path = cache.get_file("insts.bin")

        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                launcher_src_path = os.path.join(tmpdir, "main.cxx")
                so_path = os.path.join(tmpdir, "xrt_dispatch.exe")
                Path(launcher_src_path).write_text(src)
                # Compile it together.
                subprocess.check_call(
                    [
                        "g++",
                        "-std=c++23",
                        launcher_src_path,
                        f"-I{py_include_dir}",
                        f"-I{include_dir}",
                        f"-L{py_lib_dir}",
                        "-shared",
                        f"-l{py_lib}",
                        "-fPIC",
                        "-Wall",
                        f"-I{os.path.join(xrt_dir, 'include')}",
                        f"-L{os.path.join(xrt_dir, 'lib')}",
                        f"-I{os.path.join(aie_test_utils_dir, 'include')}",
                        f"-L{os.path.join(aie_test_utils_dir, 'lib')}",
                        "-luuid",
                        "-lxrt_coreutil",
                        "-lrt",
                        "-lstdc++",
                        "-lboost_program_options",
                        "-lboost_filesystem",
                        "-ltest_utils",
                        "-o",
                        so_path,
                    ]
                )

                ###### Compile to xclbin and runtime sequence
                xclbin_path = os.path.join(air_proj_path, "aie.xclbin")
                insts_path = os.path.join(air_proj_path, "insts.bin")
                aircc_options = [
                    "--device",
                    detect_npu_version(),
                    "--no-xchesscc",
                    "--no-xbridge",
                    "-i",
                    insts_path,
                    "-o",
                    xclbin_path,
                    "--peano" + " ",
                ]
                aircc.run(air_output, aircc_options)

                with open(so_path, "rb") as f:
                    cache_path = cache.put(f.read(), filename, binary=True)
                with open(xclbin_path, "rb") as f:
                    cache_xclbin_path = cache.put(f.read(), "aie.xclbin", binary=True)
                with open(insts_path, "rb") as f:
                    cache_insts_path = cache.put(f.read(), "insts.bin")

                # Check for compile-only mode
                if os.getenv("AMD_TRITON_NPU_COMPILE_ONLY", "0") == "1":
                    print(f"Compile-only mode: binaries cached at {cache_path}")
                    print(f"  xclbin: {cache_xclbin_path}")
                    print(f"  insts: {cache_insts_path}")
                    return None
        else:
            print(
                "got cache path: "
                + cache_path
                + " compilation is therefore skipped (delete cache path to force recompile)."
            )

            # Check for compile-only mode (cache hit)
            if os.getenv("AMD_TRITON_NPU_COMPILE_ONLY", "0") == "1":
                print(f"Compile-only mode (cache hit): binaries at {cache_path}")
                print(f"  xclbin: {cache_xclbin_path}")
                print(f"  insts: {cache_insts_path}")
                return None

        # Load and launch the compiled kernel.
        spec = importlib.util.spec_from_file_location(name, cache_path)
        if spec is None:
            raise RuntimeError(f"Cannot find {name} module in {cache_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.set_paths(cache_xclbin_path, cache_insts_path)
        return mod.launch(
            gridX,
            gridY,
            gridZ,
            kernel_metadata,
            launch_metadata,
            launch_enter_hook,
            launch_exit_hook,
            *args,
        )

    return launch


class NPULauncher(object):
    def __init__(self, src, metadata):
        kernel_placeholder_name = "KERNEL_NAME_PLACEHOLDER"

        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        launcher_src = _generate_launcher(constants, signature, kernel_placeholder_name)
        # Later KERNEL_NAME_PLACEHOLDER will be used to assign the kernel name
        # in the following launch function.
        self.launch = compile_module(launcher_src, kernel_placeholder_name)

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):
        self.launch(gridX, gridY, gridZ, stream, function, *args)


class NPUUtils(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(NPUUtils, cls).__new__(cls)
        return cls.instance

    # Note:
    # nvidia and amd backends have their corresponding driver.c file that exposes
    # get_device_properties and load_binary using python bindings.
    # (see third_party/nvidia/backend/driver.c)
    # These methods are then used in compiler.py to initialize handles before running
    # the triton kernels.
    # Since we recompile the kernel every time (see compile_module above),
    # and the metadata generated by these functions aren't applicable to the npu
    # backend, just define the same functions with dummy implementation.
    @staticmethod
    def get_device_properties(device):
        return {
            "max_shared_mem": 2**20,
            "multiprocessor_count": None,
            "sm_clock_rate": None,
            "mem_clock_rate": None,
            "mem_bus_width": None,
        }

    # Important note:
    # Since we cannot easy pass function pointers around, we pass along the
    # assembly source code so that compile_module above can recompile the
    # module every time.
    @staticmethod
    def load_binary(name, kernel_asm, shared, device):
        return (
            None,  # module
            kernel_asm,  # function
            None,  # n_regs
            None,  # n_spills
            sys.maxsize,  # n_max_threads
        )


class NPUDriver(DriverBase):

    def __init__(self):
        super().__init__()
        self.utils = NPUUtils()
        self.launcher_cls = NPULauncher
        self.binary_ext = "ttsharedir"

    # NPU driver won't be automatically chosen unless explicitly set through
    # triton.runtime.driver.set_active(NPUDriver())
    @staticmethod
    def is_active():
        return False

    def do_bench(
        self,
        fn,
        warmup=25,
        rep=100,
        grad_to_none=None,
        quantiles=None,
        return_mode="mean",
    ):
        assert return_mode in ["min", "max", "mean", "median", "all"]

        global autotune_time
        autotune_time = True

        fn()

        # Estimate the runtime of the function
        estimate_us = 0.0
        for _ in range(5):
            fn()
            with open("data.txt", "r") as f:
                value_str = f.read().strip()
            value = float(value_str)
            estimate_us += value

        estimate_ms = estimate_us / (5 * 1000)

        from triton import knobs

        verbose = knobs.autotuning.print
        if verbose:
            print("NPU estimate ms: ", estimate_ms)
        # compute number of warmup and repeat
        n_warmup = max(1, int(25 / estimate_ms))
        n_repeat = max(5, int(100 / estimate_ms))

        # Warm-up
        for _ in range(n_warmup):
            fn()

        times = [0.0 for i in range(n_repeat)]
        # Benchmark
        for i in range(n_repeat):
            # we don't want `fn` to accumulate gradient values
            # if it contains a backward pass. So we clear the
            # provided gradients
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            fn()
            with open("data.txt", "r") as f:
                value_str = f.read().strip()
            times[i] = float(value_str) / 1000

        if verbose:
            print("NPU KERNEL TIME (ms): ", ", ".join(str(t) for t in times))
        autotune_time = False

        from triton.testing import _summarize_statistics

        return _summarize_statistics(times, quantiles, return_mode)

    def get_benchmarker(self):
        return self.do_bench

    def get_device_capability(self):
        return ("npu", 0)

    def get_current_stream(self, device):
        return None

    def get_current_device(self):
        # NPU doesn't have a device to return. Return something.
        return "npu"

    def set_current_device(self, device):
        # NPU doesn't have a device to set
        assert device == "npu"
        return

    def get_current_target(self):
        return GPUTarget("npu", 0, 0)

    def get_active_torch_device(self):
        import torch

        return torch.device("npu")

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args

    def map_python_to_cpp_type(self, ty: str) -> str:
        return _ty_to_cpp(ty)
