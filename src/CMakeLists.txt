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

add_subdirectory(conformer)
add_subdirectory(data_structures)
add_subdirectory(forcefields)
add_subdirectory(minimizer)
add_subdirectory(substruct)
add_subdirectory(testutils)
add_subdirectory(utils)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/versions.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/versions.h @ONLY)

add_library(nvmolkit_versions INTERFACE)
target_include_directories(nvmolkit_versions
                           INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

add_library(similarity_kernels similarity_kernels.cu)
target_link_libraries(similarity_kernels PRIVATE cuda_error_check device
                                                 cccl_interface)

target_include_directories(similarity_kernels
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(similarity_kernels SYSTEM
                           PRIVATE ${Boost_INCLUDE_DIRS})
add_library(Similarity similarity.cpp)

target_link_libraries(
  Similarity
  PUBLIC CUDA::cudart device_vector
  PRIVATE similarity_kernels
          ${RDKit_LIBS}
          device_vector
          device
          openmp_helpers
          OpenMP::OpenMP_CXX
          cccl_interface)
target_include_directories(Similarity PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(Similarity SYSTEM PRIVATE ${Boost_INCLUDE_DIRS})
add_library(morganFingerprintKernels morgan_fingerprint_kernels.cu)
target_link_libraries(
  morganFingerprintKernels
  PRIVATE cuda_error_check flatBitVect ${RDKit_LIBS} cccl_interface
  PUBLIC device_vector flatBitVect cuda_error_check)
target_include_directories(morganFingerprintKernels
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(morganFingerprintKernels SYSTEM
                           PRIVATE ${Boost_INCLUDE_DIRS})

add_library(
  morganFingerprint morgan_fingerprint.cpp morgan_fingerprint_gpu.cpp
                    morgan_fingerprint_cpu.cpp morgan_fingerprint_common.cpp)
target_link_libraries(
  morganFingerprint
  PRIVATE ${RDKit_LIBS} OpenMP::OpenMP_CXX device openmp_helpers
  PUBLIC morganFingerprintKernels)
target_include_directories(morganFingerprint PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(symmetricEigenSolver symmetric_eigensolver.cu)
target_link_libraries(symmetricEigenSolver PRIVATE CUDA::cudart
                                                   cuda_error_check device)
target_include_directories(symmetricEigenSolver
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(triangle_smooth triangle_smooth.cu)
target_link_libraries(triangle_smooth PRIVATE ${RDKit_LIBS} device_vector)
target_include_directories(triangle_smooth PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(embedder_utils embedder_utils.cpp)
target_link_libraries(embedder_utils PRIVATE ${RDKit_LIBS} device_vector
                                             triangle_smooth nvmolkit_versions)
target_include_directories(embedder_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(etkdg_impl etkdg_impl.cpp etkdg_kernels.cu)
target_link_libraries(
  etkdg_impl
  PRIVATE embedder_utils rdkit_dist_geom_flattened OpenMP::OpenMP_CXX
  PUBLIC dist_geom ${RDKit_LIBS} device_vector)
target_include_directories(etkdg_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(
  etkdg_stages
  etkdg_stage_coordgen.cu etkdg_stage_distgeom_minimize.cu
  etkdg_stage_update_conformers.cu etkdg_stage_stereochem_checks.cu
  etkdg_stage_etk_minimization.cu)

target_link_libraries(
  etkdg_stages
  PUBLIC bfgs
  PRIVATE device
          device_vector
          ${RDKit_LIBS}
          dg_batched_forcefield
          etk_batched_forcefield
          ff_kernel_utils
          dist_geom
          rdkit_dist_geom_flattened
          etkdg)
target_include_directories(etkdg_stages PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(etkdg etkdg.cpp)
target_link_libraries(
  etkdg
  PUBLIC bfgs conformer_types
  PRIVATE ${RDKit_LIBS}
          conformer_pruning
          device
          device_coord_collector
          device_vector
          dist_geom
          embedder_utils
          etkdg_device_collect
          rdkit_dist_geom_flattened
          etkdg_impl
          etkdg_stages
          openmp_helpers
          OpenMP::OpenMP_CXX)
target_include_directories(etkdg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(butina butina.cu)
target_link_libraries(
  butina
  PUBLIC device_vector
  PRIVATE cub_helpers nvtx)
target_include_directories(butina PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(conformer_rmsd conformer_rmsd.cu)
target_link_libraries(
  conformer_rmsd
  PUBLIC device_vector
  PRIVATE cuda_error_check)
target_include_directories(conformer_rmsd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(conformer_rmsd_mol conformer_rmsd_mol.cpp)
target_link_libraries(
  conformer_rmsd_mol
  PUBLIC device_vector conformer_rmsd
  PRIVATE ${RDKit_LIBS})
target_include_directories(conformer_rmsd_mol
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(tfd)
