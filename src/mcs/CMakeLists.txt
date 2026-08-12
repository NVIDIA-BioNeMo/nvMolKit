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

add_library(mcs_fmcs_foundations fmcs_cuda/fmcs_match_tables.cu mcs_graph.cpp)
target_include_directories(mcs_fmcs_foundations
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mcs_fmcs_foundations PUBLIC device_vector)

add_library(mcs_fmcs_core fmcs_cuda/fmcs.cu)
target_include_directories(mcs_fmcs_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                                ${CMAKE_SOURCE_DIR})
target_link_libraries(mcs_fmcs_core PUBLIC mcs_fmcs_foundations device nvtx
                                           pinned_host_allocator)
target_compile_options(
  mcs_fmcs_core
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v;--extended-lambda>)

add_library(mcs_search mcs_search.cu)
target_link_libraries(mcs_search PUBLIC mcs_fmcs_core ${RDKit_LIBS})
target_include_directories(mcs_search PUBLIC ${CMAKE_SOURCE_DIR})
