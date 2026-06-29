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

add_library(benchmark_utils STATIC benchmark_utils.cpp)
target_link_libraries(benchmark_utils PRIVATE ${RDKit_LIBS} test_utils
                                              OpenMP::OpenMP_CXX)

add_executable(MorganFpBench morgan_fp.cpp)
target_link_libraries(
  MorganFpBench PRIVATE nanobench ${RDKit_LIBS} OpenMP::OpenMP_CXX
                        morganFingerprint)

add_executable(BitVectorBench bit_vector.cpp)
target_link_libraries(BitVectorBench PRIVATE nanobench ${RDKit_LIBS}
                                             flatBitVect)
add_executable(EigenSolverBench eigen_solver_bench.cu)
target_link_libraries(
  EigenSolverBench PRIVATE nanobench ${RDKit_LIBS} device_vector
                           cuda_error_check symmetricEigenSolver)

add_executable(ETKDGFFBench etkdg_ff_bench.cpp)
target_link_libraries(
  ETKDGFFBench
  PRIVATE nanobench
          ${RDKit_LIBS}
          rdkit_dist_geom_flattened
          dist_geom
          device_vector
          ff_utils
          ff_kernel_utils
          test_utils
          embedder_utils
          cuda_error_check)

add_executable(mmff_multimol_bench mmff_multimol_bench.cpp)
target_link_libraries(
  mmff_multimol_bench
  PRIVATE nanobench
          ${RDKit_LIBS}
          mmff_minimize
          test_utils
          ff_utils
          cuda_error_check
          OpenMP::OpenMP_CXX
          benchmark_utils)

add_executable(ETKDGBench etkdg_embed_bench.cpp)
target_link_libraries(
  ETKDGBench
  PRIVATE nanobench
          device_vector
          dist_geom
          ${RDKit_LIBS}
          test_utils
          etkdg_impl
          etkdg_stages
          etkdg
          embedder_utils
          conformer_checkers
          benchmark_utils)

add_executable(UpdateInverseHessianBench updateInverseHessianBench.cu)
target_link_libraries(UpdateInverseHessianBench
                      PRIVATE CUDA::cudart cuda_error_check device_vector bfgs)
