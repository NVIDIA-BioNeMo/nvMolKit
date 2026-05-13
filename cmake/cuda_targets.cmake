# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# Options for CUDA build targets.

if(NVMOLKIT_CUDA_TARGET_MODE STREQUAL "native")
  set(CMAKE_CUDA_ARCHITECTURES native)
  message(
    STATUS
      "NVMOLKIT_CUDA_TARGET_MODE=native: Using native CUDA architecture for fast local builds"
  )
elseif(NVMOLKIT_CUDA_TARGET_MODE STREQUAL "full")
  set(_nvmolkit_cuda_arch_list
      "70-real;75-real;80-real;86-real;89-real;90-real")
  if(DEFINED CUDAToolkit_VERSION)
    string(REPLACE "." ";" _cuda_version_list "${CUDAToolkit_VERSION}")
    list(GET _cuda_version_list 0 _cuda_major)
    list(GET _cuda_version_list 1 _cuda_minor)
    math(EXPR _cuda_version_num "${_cuda_major} * 100 + ${_cuda_minor}")
    if(_cuda_major GREATER_EQUAL 13)
      list(REMOVE_ITEM _nvmolkit_cuda_arch_list "70-real")
      message(STATUS "CUDA >= 13 detected, removing Volta (70-real) arch")
    endif()
    if(_cuda_version_num GREATER_EQUAL 1208)
      list(APPEND _nvmolkit_cuda_arch_list "100-real")
      message(
        STATUS "CUDA >= 12.8 detected, enabling Blackwell (100-real) arch")
    else()
      message(
        STATUS "CUDA < 12.8 detected, Blackwell (100-real) arch not enabled")
    endif()
    if(_cuda_version_num GREATER_EQUAL 1209)
      list(APPEND _nvmolkit_cuda_arch_list "103-real")
      list(APPEND _nvmolkit_cuda_arch_list "120")
      message(
        STATUS
          "CUDA >= 12.9 detected, enabling Blackwell (103-real) and (120 + PTX) for forward compatibility"
      )
    endif()
  endif()

  set(CMAKE_CUDA_ARCHITECTURES "${_nvmolkit_cuda_arch_list}")
  message(
    STATUS
      "NVMOLKIT_CUDA_TARGET_MODE=full: Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}"
  )
else() # default
  if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(
      STATUS
        "NVMOLKIT_CUDA_TARGET_MODE=default: Using user-specified CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}"
    )
    # Check if the first architecture is below 70
    string(REPLACE ";" " " _cuda_arch_str "${CMAKE_CUDA_ARCHITECTURES}")
    string(REGEX MATCH "^[0-9]+" _first_arch "${_cuda_arch_str}")
    if(_first_arch AND _first_arch LESS 70)
      set(_original_arch "${CMAKE_CUDA_ARCHITECTURES}")
      set(CMAKE_CUDA_ARCHITECTURES
          70
          CACHE STRING "CUDA architectures" FORCE)
      message(
        WARNING
          "CMAKE_CUDA_ARCHITECTURES was set below 70, which is the minimum for nvMolKit.\n"
          "  Original value: ${_original_arch}\n"
          "  Resetting to 70.")
    endif()
  else()
    set(CMAKE_CUDA_ARCHITECTURES 80)
    message(
      STATUS
        "NVMOLKIT_CUDA_TARGET_MODE=default: No CMAKE_CUDA_ARCHITECTURES set, defaulting to 80"
    )
  endif()
endif()

# Identify CUDA compute capabilities for similarity tensor core support. The
# resulting NVMOLKIT_CUDA_CC_* defines are exposed to first-party targets via
# the nvmolkit_cuda_caps INTERFACE library so they don't leak into FetchContent
# third-party translation units.
add_library(nvmolkit_cuda_caps INTERFACE)

if(CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
  # Write a small CUDA program to detect compute capability
  file(
    WRITE "${CMAKE_BINARY_DIR}/detect_cuda_arch.cu"
    "
  #include <cuda_runtime.h>
  #include <cstdio>
  int main() {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, 0);
      printf(\"%d%d\\n\", prop.major, prop.minor);
      return 0;
  }
  ")

  # Build the program
  execute_process(
    COMMAND
      ${CMAKE_CUDA_COMPILER} --disable-warnings
      "${CMAKE_BINARY_DIR}/detect_cuda_arch.cu" -o
      "${CMAKE_BINARY_DIR}/detect_cuda_arch"
    RESULT_VARIABLE _build_result)

  if(_build_result EQUAL 0)
    execute_process(
      COMMAND "${CMAKE_BINARY_DIR}/detect_cuda_arch"
      OUTPUT_VARIABLE _native_cc
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  else()
    message(FATAL_ERROR "Failed to build detect_cuda_arch.cu")
  endif()
  foreach(
    cc IN
    ITEMS 80
          86
          89
          90
          100
          103
          120)
    if(_native_cc STREQUAL "${cc}")
      target_compile_definitions(nvmolkit_cuda_caps
                                 INTERFACE NVMOLKIT_CUDA_CC_${cc}=1)
    else()
      target_compile_definitions(nvmolkit_cuda_caps
                                 INTERFACE NVMOLKIT_CUDA_CC_${cc}=0)
    endif()
  endforeach()
else()
  foreach(
    cc IN
    ITEMS 80
          86
          89
          90
          100
          103
          120)
    string(REPLACE ";" " " _cuda_arch_str "${CMAKE_CUDA_ARCHITECTURES}")
    string(REGEX MATCH "(^| )${cc}(-real)?( |$)" _match "${_cuda_arch_str}")
    if(_match)
      target_compile_definitions(nvmolkit_cuda_caps
                                 INTERFACE NVMOLKIT_CUDA_CC_${cc}=1)
    else()
      target_compile_definitions(nvmolkit_cuda_caps
                                 INTERFACE NVMOLKIT_CUDA_CC_${cc}=0)
    endif()
  endforeach()
endif()
