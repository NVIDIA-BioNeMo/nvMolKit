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

# Sanitizers are driven by the NVMOLKIT_SANITIZER cache variable. For backwards
# compatibility, a build type of asan / tsan / ubsan auto-selects the matching
# sanitizer when NVMOLKIT_SANITIZER is unset.
set(NVMOLKIT_SANITIZER
    "none"
    CACHE STRING "Active sanitizer (none, asan, tsan, ubsan)")
set_property(CACHE NVMOLKIT_SANITIZER PROPERTY STRINGS none asan tsan ubsan)

if(NVMOLKIT_SANITIZER STREQUAL "none" AND CMAKE_BUILD_TYPE)
  string(TOLOWER "${CMAKE_BUILD_TYPE}" _nvmolkit_lc_build_type)
  if(_nvmolkit_lc_build_type MATCHES "^(asan|tsan|ubsan)$")
    set(NVMOLKIT_SANITIZER "${_nvmolkit_lc_build_type}")
    message(
      STATUS
        "NVMOLKIT_SANITIZER not set; deriving '${NVMOLKIT_SANITIZER}' from CMAKE_BUILD_TYPE"
    )
  endif()
endif()

# cmake-format: off
string(CONCAT NVMOLKIT_CTEST_ASAN_ENV_VARS
       "ASAN_OPTIONS=protect_shadow_gap=0"
       ":report_globals=1"
       ":check_initialization_order=true"
       ":detect_stack_use_after_return=true"
       ":strict_string_checks=true")
# cmake-format: on

add_library(nvmolkit_sanitizers INTERFACE)
if(NVMOLKIT_SANITIZER STREQUAL "asan")
  target_compile_options(nvmolkit_sanitizers INTERFACE -fsanitize=address -g
                                                       -O0)
  target_link_options(nvmolkit_sanitizers INTERFACE -fsanitize=address)
elseif(NVMOLKIT_SANITIZER STREQUAL "tsan")
  target_compile_options(nvmolkit_sanitizers INTERFACE -fsanitize=thread -g -O1)
  target_link_options(nvmolkit_sanitizers INTERFACE -fsanitize=thread)
elseif(NVMOLKIT_SANITIZER STREQUAL "ubsan")
  set(_nvmolkit_ubsan_checks
      "undefined,float-divide-by-zero,implicit-conversion,local-bounds,nullability"
  ) # cmake-lint: disable=C0301
  target_compile_options(nvmolkit_sanitizers
                         INTERFACE -fsanitize=${_nvmolkit_ubsan_checks} -g -O1)
  target_link_options(nvmolkit_sanitizers INTERFACE
                      -fsanitize=${_nvmolkit_ubsan_checks})
elseif(NOT NVMOLKIT_SANITIZER STREQUAL "none")
  message(FATAL_ERROR "Unknown NVMOLKIT_SANITIZER value: ${NVMOLKIT_SANITIZER}")
endif()
