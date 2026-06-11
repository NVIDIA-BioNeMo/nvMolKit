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

add_library(rdkit_mmff_flattened mmff_flattened_builder.cpp)
target_link_libraries(
  rdkit_mmff_flattened
  PRIVATE ${RDKit_LIBS} device_vector
  PUBLIC mmff)

add_library(rdkit_uff_flattened uff_flattened_builder.cpp)
target_link_libraries(
  rdkit_uff_flattened
  PRIVATE ${RDKit_LIBS} device_vector
  PUBLIC uff)

add_library(mmff_contribs mmff_contribs.cpp)
target_link_libraries(
  mmff_contribs
  PRIVATE ${RDKit_LIBS} CUDA::cudart device_vector ff_utils
  PUBLIC mmff)

add_library(mmff_builder mmff_builder.cpp)
target_link_libraries(
  mmff_builder
  PRIVATE ${RDKit_LIBS} CUDA::cudart device_vector ff_utils
  PUBLIC mmff)

add_library(mmff_optimize INTERFACE)
target_link_libraries(mmff_optimize INTERFACE ${RDKit_LIBS} CUDA::cudart
                                              rdkit_mmff_flattened)

add_library(rdkit_dist_geom_flattened dist_geom_flattened_builder.cpp)
target_link_libraries(
  rdkit_dist_geom_flattened
  PRIVATE ${RDKit_LIBS} device_vector
  PUBLIC dist_geom)

add_library(rdkit_bounds_matrix bounds_matrix.cpp)
target_link_libraries(rdkit_bounds_matrix PRIVATE ${RDKit_LIBS})

add_library(conformer_pruning conformer_pruning.cpp)
target_link_libraries(conformer_pruning PRIVATE ${RDKit_LIBS})
