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

add_library(packed_bonds packed_bonds.cpp)

add_library(molecules molecules.cpp)
target_link_libraries(
  molecules
  PUBLIC device_vector flatBitVect OpenMP::OpenMP_CXX packed_bonds device
  PRIVATE ${RDKit_LIBS} nvtx openmp_helpers)

add_library(substruct_kernels substruct_kernels.cu)
target_link_libraries(
  substruct_kernels PUBLIC molecules device_vector flatBitVect cuda_error_check
                           device_timings)

add_library(substruct_definitions INTERFACE)

add_library(
  substructure_search
  substruct_search.cu
  gpu_executor.cu
  recursive_preprocessor.cu
  minibatch_planner.cpp
  pinned_buffer_pool.cpp
  substruct_launch_config.cpp
  substruct_search_internal.cpp)
target_link_libraries(
  substructure_search
  PUBLIC molecules
         device_vector
         flatBitVect
         cuda_error_check
         pinned_host_allocator
         device_timings
         device
         substruct_kernels
         OpenMP::OpenMP_CXX
  PRIVATE ${RDKit_LIBS})
target_compile_options(substructure_search
                       PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
