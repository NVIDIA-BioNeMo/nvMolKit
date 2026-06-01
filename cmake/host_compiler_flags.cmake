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

# Per-target host compile options. The CUDA half of nvmolkit_warnings is added
# in cmake/device_compiler_flags.cmake.
add_library(nvmolkit_warnings INTERFACE)
if(NVMOLKIT_EXTRA_DEV_FLAGS)
  message(STATUS "Enabling extra development flags")
  target_compile_options(
    nvmolkit_warnings
    INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Werror;-Wall;-Wextra;-Wno-sign-compare>
  )
endif()

# -ffast-math is a project preference for optimized builds and intentionally
# stays off third-party targets compiled via FetchContent. Host-only; the CUDA
# equivalent (--use_fast_math) is added by nvmolkit_cuda_options.
add_library(nvmolkit_release_opts INTERFACE)
target_compile_options(
  nvmolkit_release_opts
  INTERFACE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-ffast-math>
)
