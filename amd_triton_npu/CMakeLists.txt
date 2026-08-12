# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

project(amd_triton_npu)

# Build amd-triton-npu plugin with triton-shared c++ passes

option(AMD_TRITON_NPU_BUILD_NPU_BACKEND "Build AMD NPU backend" ON)

set(AMD_TRITON_NPU_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(AMD_TRITON_NPU_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/triton_shared/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files
include_directories(${Python3_INCLUDE_DIR})

if (AMD_TRITON_NPU_BUILD_NPU_BACKEND)
    add_triton_plugin(AmdTritonNpu ${CMAKE_CURRENT_SOURCE_DIR}/amd_triton_npu.cc LINK_LIBS TritonSharedAnalysis TritonTilingExtIR)
    # Triton moved its Python bindings from pybind11 to nanobind; the plugin
    # entry point takes a nanobind::module_ and needs nanobind's headers.
    target_link_libraries(AmdTritonNpu PRIVATE Python3::Module nanobind-static)
endif()
