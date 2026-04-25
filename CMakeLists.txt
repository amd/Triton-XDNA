# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.24)

project(triton-xdna)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(Git REQUIRED)

# =============================================================================
# Patch Management Targets
# =============================================================================

# Apply patches to third-party submodules
add_custom_target(apply-patches
	COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/apply_patches.py
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	COMMENT "Applying patches to third-party submodules..."
	USES_TERMINAL
)

# Reset submodules to clean state (removes patches)
add_custom_target(reset-submodules
	COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/apply_patches.py --reset-only
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	COMMENT "Resetting third-party submodules to clean state..."
	USES_TERMINAL
)

# Force re-apply patches (reset + apply)
add_custom_target(reapply-patches
	COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/apply_patches.py --reset
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	COMMENT "Re-applying patches to third-party submodules..."
	USES_TERMINAL
)

# =============================================================================
# Build Configuration
# =============================================================================

set(PIP_INSTALL_ARGS)  # Pip install command args
list(APPEND PIP_INSTALL_ARGS "--no-build-isolation")
list(APPEND PIP_INSTALL_ARGS "-vvv")

set(ENV TRITON_PLUGIN_DIRS="${pwd}/amd_triton_npu;${pwd}/third_party/triton_shared")

set(PIP_INSTALL_ENVS)  # Pip install environmental vars
if(DEFINED PIP_INSTALL_TARGET)
	list(APPEND PIP_INSTALL_ENVS "PIP_INSTALL_TARGET=${PIP_INSTALL_TARGET}")
endif()
if(NOT WIN32)
	list(APPEND PIP_INSTALL_ENVS "TRITON_BUILD_WITH_CLANG_LLD=true")
endif()

add_custom_target(triton-build ALL
	COMMAND ${PIP_INSTALL_ENVS}
		"TRITON_PLUGIN_DIRS=${CMAKE_CURRENT_SOURCE_DIR}/third_party/triton_shared\;${CMAKE_CURRENT_SOURCE_DIR}/amd_triton_npu"
		${Python3_EXECUTABLE} -m pip install ${PIP_INSTALL_ARGS} .
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	COMMENT "Building Triton with XDNA plugins..."
	USES_TERMINAL
)

# Ensure patches are applied before building
add_dependencies(triton-build apply-patches)
