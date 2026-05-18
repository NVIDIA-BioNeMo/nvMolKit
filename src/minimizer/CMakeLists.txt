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

add_library(bfgs bfgs_minimize.cu bfgs_hessian.cu
                 bfgs_minimize_permol_kernels.cu)
target_link_libraries(
  bfgs
  PUBLIC host_vector device_vector
  PRIVATE ${RDKit_LIBS}
          batched_forcefield
          rdkit_mmff_flattened
          dist_geom
          cub_helpers
          cccl_interface
          nvmolkit_versions)
target_include_directories(bfgs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(bfgs_common bfgs_common.cpp)
target_link_libraries(
  bfgs_common
  PUBLIC host_vector device conformer_types
  PRIVATE ${RDKit_LIBS} OpenMP::OpenMP_CXX)
target_include_directories(bfgs_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(bfgs_mmff bfgs_mmff.cpp)
target_link_libraries(
  bfgs_mmff
  PUBLIC bfgs conformer_types
  PRIVATE ${RDKit_LIBS}
          bfgs_common
          ff_device_collect
          mmff_batched_forcefield
          rdkit_mmff_flattened
          forcefield_constraints
          device_vector
          OpenMP::OpenMP_CXX
          openmp_helpers
          cccl_interface)
target_include_directories(bfgs_mmff PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                            ${CMAKE_SOURCE_DIR}/src/forcefields)

add_library(bfgs_uff bfgs_uff.cpp)
target_link_libraries(
  bfgs_uff
  PUBLIC bfgs conformer_types
  PRIVATE ${RDKit_LIBS}
          bfgs_common
          ff_device_collect
          uff_batched_forcefield
          rdkit_uff_flattened
          forcefield_constraints
          device_vector
          OpenMP::OpenMP_CXX
          openmp_helpers
          cccl_interface)
target_include_directories(bfgs_uff PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                           ${CMAKE_SOURCE_DIR}/src/forcefields)
