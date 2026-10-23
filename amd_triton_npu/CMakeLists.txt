# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

project(amd_triton_npu)

# Build amd-triton-npu plugin with triton-shared c++ passes

option(AMD_TRITON_NPU_BUILD_NPU_BACKEND "Build AMD NPU backend" ON)

set(AMD_TRITON_NPU_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(AMD_TRITON_NPU_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

# triton_shared is built as a sibling plugin from wherever TRITON_PLUGIN_DIRS
# points, so find its headers through one of its targets.
get_target_property(TRITON_SHARED_ANALYSIS_DIR TritonSharedAnalysis SOURCE_DIR)
include_directories(${TRITON_SHARED_ANALYSIS_DIR}/../../include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files
include_directories(${Python3_INCLUDE_DIR})

if (AMD_TRITON_NPU_BUILD_NPU_BACKEND)
    add_triton_plugin(AmdTritonNpu ${CMAKE_CURRENT_SOURCE_DIR}/amd_triton_npu.cc LINK_LIBS TritonSharedAnalysis TritonTilingExtIR)
    # The plugin entry point takes the module type of whichever binding
    # framework Triton itself was built with. Triton builds the nanobind-static
    # target only when it uses nanobind.
    if (TARGET nanobind-static)
        target_link_libraries(AmdTritonNpu PRIVATE Python3::Module nanobind-static)
    else()
        target_compile_definitions(AmdTritonNpu PRIVATE AMD_TRITON_NPU_PYBIND11)
        target_link_libraries(AmdTritonNpu PRIVATE Python3::Module pybind11::headers)
    endif()
endif()
