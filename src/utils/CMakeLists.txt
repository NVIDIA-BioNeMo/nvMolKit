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

add_library(exceptions exceptions.cpp)
target_link_libraries(exceptions PRIVATE CUDA::cudart)
target_compile_options(exceptions PRIVATE "-fPIC")

add_library(cuda_error_check INTERFACE)
target_link_libraries(cuda_error_check INTERFACE exceptions CUDA::cudart)

add_library(device_vector INTERFACE)
target_link_libraries(device_vector INTERFACE CUDA::cudart cuda_error_check
                                              cccl_interface)

add_library(host_vector INTERFACE)
target_link_libraries(host_vector INTERFACE CUDA::cudart cuda_error_check)

add_library(pinned_host_allocator pinned_host_allocator.cpp)
target_link_libraries(pinned_host_allocator PUBLIC host_vector)

add_library(device device.cpp)
target_link_libraries(device PRIVATE CUDA::cudart cuda_error_check)

add_library(rdkit_utils INTERFACE)

add_library(work_splitting INTERFACE)

add_library(nvtx INTERFACE)
target_link_libraries(nvtx INTERFACE CUDA::nvtx3)

add_library(openmp_helpers openmp_helpers.cpp)
target_link_libraries(openmp_helpers PRIVATE OpenMP::OpenMP_CXX)

add_library(cub_helpers INTERFACE)
target_link_libraries(cub_helpers INTERFACE CUDA::cudart)

add_library(device_timings INTERFACE)
target_link_libraries(device_timings INTERFACE cuda_error_check)

add_library(thread_safe_queue INTERFACE)

add_library(p2p p2p.cpp)
target_link_libraries(p2p PUBLIC CUDA::cudart cuda_error_check device)
