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

# CUDA half of nvmolkit_warnings (host target is created in
# cmake/host_compiler_flags.cmake).
if(NVMOLKIT_EXTRA_DEV_FLAGS)
  target_compile_options(
    nvmolkit_warnings
    INTERFACE
      $<$<COMPILE_LANGUAGE:CUDA>:-Werror=all-warnings;-Xcompiler=-Werror,-Wall,-Wextra>
  )
endif()

# Project-specific CUDA options. Per-config flags are layered on top of CMake's
# defaults (-O3 -DNDEBUG, -O2 -g -DNDEBUG, -g) via generator expressions, rather
# than overwriting CMAKE_CUDA_FLAGS_<CONFIG>.
add_library(nvmolkit_cuda_options INTERFACE)
target_compile_options(
  nvmolkit_cuda_options
  INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets;--default-stream=per-thread>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-g;-G>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:--use_fast_math>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:--use_fast_math;-lineinfo>
)
