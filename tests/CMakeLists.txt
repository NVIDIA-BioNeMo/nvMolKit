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

add_subdirectory(integration)
add_executable(test_rdkit_comp test_rdkit_compiles.cpp)
target_link_libraries(test_rdkit_comp PRIVATE ${RDKit_LIBS})

add_executable(test_morgan_fingerprint_ref test_morgan_fingerprint_ref.cpp)
target_link_libraries(test_morgan_fingerprint_ref PRIVATE ${RDKit_LIBS})

add_executable(test_morgan_fingerprint test_morgan_fingerprint.cpp)
target_link_libraries(
  test_morgan_fingerprint PRIVATE ${RDKit_LIBS} morganFingerprint rdkit_utils
                                  molData OpenMP::OpenMP_CXX)

add_executable(test_similarity test_similarity.cpp)
target_link_libraries(test_similarity PRIVATE ${RDKit_LIBS} Similarity)

add_executable(test_cuda_error_check test_cuda_error_check.cpp)
target_link_libraries(test_cuda_error_check PRIVATE cuda_error_check)

add_executable(test_device test_device.cpp)
target_link_libraries(test_device PRIVATE device cuda_error_check CUDA::cudart)

add_executable(test_p2p test_p2p.cpp)
target_link_libraries(test_p2p PRIVATE p2p device device_vector
                                       cuda_error_check CUDA::cudart)

add_executable(test_device_timings test_device_timings.cu)
target_link_libraries(test_device_timings PRIVATE device_timings)
block()
# cmake-format and cmake-lint disagree on this wrapped property assignment.
# cmake-lint: disable=C0307
set_source_files_properties(
  test_device_timings.cu PROPERTIES COMPILE_DEFINITIONS
                                    NVMOLKIT_ENABLE_DEVICE_TIMINGS=1)
endblock()

add_executable(test_device_vector test_device_vector.cpp)
target_link_libraries(test_device_vector PRIVATE device_vector cuda_error_check
                                                 CUDA::cudart)

add_executable(test_flat_bit_vect test_flat_bit_vect.cpp)
target_link_libraries(test_flat_bit_vect PRIVATE flatBitVect)
target_include_directories(test_flat_bit_vect SYSTEM
                           PRIVATE ${Boost_INCLUDE_DIRS})

add_executable(test_flat_bit_vect_device test_flat_bit_vect_device.cu)
target_link_libraries(test_flat_bit_vect_device PRIVATE flatBitVect device
                                                        device_vector)

add_executable(test_work_splitting test_work_splitting.cpp)
target_link_libraries(test_work_splitting PRIVATE work_splitting)

add_executable(test_thread_safe_queue test_thread_safe_queue.cpp)
target_link_libraries(test_thread_safe_queue PRIVATE thread_safe_queue)

add_executable(test_openmp_helpers test_openmp_helpers.cpp)
target_link_libraries(test_openmp_helpers PRIVATE openmp_helpers
                                                  OpenMP::OpenMP_CXX)

add_library(test_utils test_utils.cpp)
target_link_libraries(
  test_utils PRIVATE ${RDKit_LIBS} GTest::gtest GTest::gtest_main GTest::gmock
                     GTest::gmock_main)

add_executable(smarts_filter smarts_filter.cpp)
target_link_libraries(smarts_filter PRIVATE molecules molData ${RDKit_LIBS})

add_executable(test_flattened_builder test_flattened_builder.cu)
target_link_libraries(
  test_flattened_builder
  PRIVATE ${RDKit_LIBS} rdkit_mmff_flattened rdkit_dist_geom_flattened
          device_vector test_utils embedder_utils)

add_executable(test_mmff test_mmff.cu)
target_link_libraries(
  test_mmff
  PRIVATE ${RDKit_LIBS}
          mmff_minimize
          rdkit_mmff_flattened
          mmff
          mmff_batched_forcefield
          mmff_builder
          mmff_contribs
          mmff_optimize
          device_vector
          ff_utils
          test_utils)

add_executable(test_uff test_uff.cu)
target_link_libraries(
  test_uff
  PRIVATE ${RDKit_LIBS}
          uff_batched_forcefield
          uff_minimize
          bfgs_common
          bfgs
          ff_device_collect
          fire_minimizer
          forcefield_constraints
          uff
          rdkit_uff_flattened
          device_vector
          ff_utils
          ff_kernel_utils
          test_utils)

add_executable(test_distgeom test_distgeom.cu)
target_link_libraries(
  test_distgeom
  PRIVATE ${RDKit_LIBS}
          rdkit_dist_geom_flattened
          batched_forcefield
          dg_batched_forcefield
          dist_geom
          device_vector
          ff_utils
          ff_kernel_utils
          test_utils
          embedder_utils)
add_executable(test_distgeom_kernels test_distgeom_kernels.cu)
target_link_libraries(
  test_distgeom_kernels
  PRIVATE ${RDKit_LIBS}
          rdkit_dist_geom_flattened
          dist_geom
          device_vector
          rdkit_bounds_matrix
          ff_utils
          ff_kernel_utils
          test_utils
          embedder_utils)

add_executable(test_rdkit_bounds_matrix test_rdkit_bounds_matrix.cpp)
target_link_libraries(
  test_rdkit_bounds_matrix PRIVATE rdkit_bounds_matrix symmetricEigenSolver
                                   ${RDKit_LIBS})

add_executable(test_triangle_smooth test_triangle_smooth.cu)
target_link_libraries(
  test_triangle_smooth PRIVATE triangle_smooth device_vector ${RDKit_LIBS}
                               test_utils)

add_executable(test_coordgen test_coordgen.cu)
target_link_libraries(
  test_coordgen
  PRIVATE rdkit_bounds_matrix
          symmetricEigenSolver
          device_vector
          ${RDKit_LIBS}
          test_utils
          dist_geom
          rdkit_dist_geom_flattened)

add_executable(test_etkdg test_etkdg.cu test_etkdg_minimize.cu)
target_link_libraries(
  test_etkdg
  PRIVATE etkdg_impl
          device_vector
          test_utils
          ${RDKit_LIBS}
          dist_geom
          etkdg_stages
          etkdg
          embedder_utils
          conformer_pruning)

add_executable(test_etkdg_device_output test_etkdg_device_output.cu)
target_link_libraries(
  test_etkdg_device_output
  PRIVATE etkdg
          device_coord_collector
          etkdg_stages
          dist_geom
          device
          device_vector
          cuda_error_check
          ${RDKit_LIBS})

add_executable(test_ff_device_output test_ff_device_output.cu)
target_link_libraries(
  test_ff_device_output
  PRIVATE mmff_minimize
          uff_minimize
          bfgs_common
          mmff
          uff
          mmff_batched_forcefield
          uff_batched_forcefield
          rdkit_mmff_flattened
          rdkit_uff_flattened
          ff_utils
          ff_kernel_utils
          forcefield_constraints
          device
          device_vector
          cuda_error_check
          ${RDKit_LIBS})

add_executable(test_etkdg_coordgen test_etkdg_coordgen.cu)
target_link_libraries(
  test_etkdg_coordgen
  PRIVATE dist_geom
          rdkit_dist_geom_flattened
          test_utils
          etkdg
          etkdg_stages
          ${RDKit_LIBS}
          device_vector)

add_executable(test_bfgs_minimizer test_bfgs_minimizer.cu)
target_link_libraries(
  test_bfgs_minimizer
  PRIVATE bfgs
          batched_forcefield
          mmff_batched_forcefield
          rdkit_mmff_flattened
          device_vector
          ${RDKit_LIBS}
          test_utils
          mmff_minimize
          bfgs_common
          ff_device_collect
          ff_utils
          forcefield_constraints
          mmff
          device)

add_executable(test_bfgs_hessian test_bfgs_hessian.cpp)
target_link_libraries(test_bfgs_hessian PRIVATE bfgs device_vector)

add_executable(test_fire_minimizer test_fire_minimizer.cu)
target_link_libraries(test_fire_minimizer PRIVATE fire_minimizer device_vector
                                                  cuda_error_check)

add_executable(test_fire_minimizer_permol test_fire_minimizer_permol.cu)
target_link_libraries(
  test_fire_minimizer_permol
  PRIVATE fire_minimizer
          batched_forcefield
          mmff_batched_forcefield
          rdkit_mmff_flattened
          device_vector
          cuda_error_check
          ${RDKit_LIBS}
          test_utils
          mmff_minimize
          mmff
          device)

add_executable(test_etkdg_etk_minimize test_etkdg_etk_minimize.cu)
target_link_libraries(
  test_etkdg_etk_minimize
  PRIVATE etkdg_stages device_vector ${RDKit_LIBS} test_utils dist_geom
          rdkit_dist_geom_flattened)

add_executable(test_etkdg_stereochem_checks test_etkdg_stereochem_checks.cu)
target_link_libraries(
  test_etkdg_stereochem_checks
  PRIVATE etkdg_stages device_vector ${RDKit_LIBS} test_utils dist_geom
          rdkit_dist_geom_flattened)

add_executable(test_conformer_checkers test_conformer_checkers.cpp)
target_link_libraries(test_conformer_checkers PRIVATE ${RDKit_LIBS}
                                                      conformer_checkers)

add_executable(test_host_vector test_host_vector.cu)
target_link_libraries(test_host_vector PRIVATE host_vector device_vector)

add_executable(test_pinned_host_allocator test_pinned_host_allocator.cpp)
target_link_libraries(test_pinned_host_allocator PRIVATE pinned_host_allocator)

add_executable(test_etkdg_result_tracker test_etkdg_result_tracker.cpp)
target_link_libraries(test_etkdg_result_tracker PRIVATE etkdg_impl)

add_executable(test_butina test_butina.cpp)
target_link_libraries(test_butina PRIVATE butina device)

add_executable(test_boolean_tree test_boolean_tree.cu)
target_link_libraries(test_boolean_tree PRIVATE substruct_definitions device)

add_executable(test_molecules test_molecules.cu)
target_link_libraries(test_molecules PRIVATE molecules cuda_error_check
                                             ${RDKit_LIBS})

add_executable(test_query test_query.cu)
target_link_libraries(test_query PRIVATE molecules device ${RDKit_LIBS})

add_executable(test_graph_labeler test_graph_labeler.cu)
target_link_libraries(
  test_graph_labeler PRIVATE molecules device substruct_validation
                             ${RDKit_LIBS})

add_executable(test_graph_labeler_recursive test_graph_labeler_recursive.cu)
target_link_libraries(
  test_graph_labeler_recursive PRIVATE substructure_search molecules device
                                       substruct_validation ${RDKit_LIBS})

add_executable(test_minibatch_planner test_minibatch_planner.cpp)
target_link_libraries(test_minibatch_planner PRIVATE substructure_search
                                                     molecules ${RDKit_LIBS})

add_executable(test_recursive_preprocessor test_recursive_preprocessor.cu)
target_link_libraries(
  test_recursive_preprocessor PRIVATE substructure_search molecules device
                                      ${RDKit_LIBS})

add_executable(test_substructure_search test_substruct_search.cu)
target_link_libraries(
  test_substructure_search PRIVATE substructure_search molecules device
                                   substruct_validation ${RDKit_LIBS})

add_executable(test_substruct_algos test_substruct_algos.cu)
target_link_libraries(
  test_substruct_algos PRIVATE substruct_definitions molecules device
                               flatBitVect device_vector device_timings)

add_executable(test_substructure_integration test_substruct_integration.cu)
target_link_libraries(
  test_substructure_integration
  PRIVATE substructure_search
          molecules
          device
          substruct_validation
          test_utils
          molData
          ${RDKit_LIBS})

add_executable(test_substruct_label_integration
               test_substruct_label_integration.cu)
target_link_libraries(
  test_substruct_label_integration PRIVATE substructure_search molecules device
                                           test_utils molData ${RDKit_LIBS})

add_executable(test_tfd_cpu test_tfd_cpu.cpp)
target_link_libraries(test_tfd_cpu PRIVATE ${RDKit_LIBS} tfd_cpu test_utils)

add_executable(test_tfd_kernels test_tfd_kernels.cu)
target_link_libraries(
  test_tfd_kernels PRIVATE ${RDKit_LIBS} tfd_cpu tfd_kernels device_vector
                           test_utils)

add_executable(test_tfd_gpu test_tfd_gpu.cpp)
target_link_libraries(test_tfd_gpu PRIVATE ${RDKit_LIBS} tfd_gpu device_vector
                                           test_utils)

add_executable(test_tfd test_tfd.cpp)
target_link_libraries(test_tfd PRIVATE ${RDKit_LIBS} tfd_gpu tfd_cpu
                                       device_vector test_utils)

set(TEST_LIST
    test_boolean_tree
    test_butina
    test_bfgs_hessian
    test_tfd_cpu
    test_tfd_kernels
    test_tfd_gpu
    test_tfd
    test_openmp_helpers
    test_conformer_checkers
    test_coordgen
    test_p2p
    test_cuda_error_check
    test_device
    test_device_timings
    test_device_vector
    test_distgeom_kernels
    test_etkdg
    test_etkdg_device_output
    test_ff_device_output
    test_etkdg_coordgen
    test_etkdg_etk_minimize
    test_etkdg_stereochem_checks
    test_flat_bit_vect
    test_flat_bit_vect_device
    test_flattened_builder
    test_graph_labeler
    test_graph_labeler_recursive
    test_minibatch_planner
    test_recursive_preprocessor
    test_host_vector
    test_bfgs_minimizer
    test_etkdg_result_tracker
    test_molecules
    test_mmff
    test_uff
    test_distgeom
    test_pinned_host_allocator
    test_triangle_smooth
    test_morgan_fingerprint
    test_morgan_fingerprint_ref
    test_query
    test_rdkit_comp
    test_rdkit_bounds_matrix
    test_substruct_algos
    test_substructure_search
    test_substructure_integration
    test_substruct_label_integration
    test_similarity
    test_thread_safe_queue
    test_work_splitting
    test_fire_minimizer
    test_fire_minimizer_permol)

# Auto populate the NVMOLKIT_TESTDATA environment variable for ctest.
foreach(arg ${TEST_LIST})
  target_link_libraries(${arg} PRIVATE GTest::gtest GTest::gtest_main
                                       GTest::gmock GTest::gmock_main)
  gtest_add_tests(TARGET ${arg})
  set(LOCAL_ENV_VARS "${NVMOLKIT_CTEST_ENV_VARS}"
                     "NVMOLKIT_TESTDATA=${CMAKE_CURRENT_SOURCE_DIR}/test_data")
  set_tests_properties(${myTests} PROPERTIES ENVIRONMENT "${LOCAL_ENV_VARS}")
endforeach()
