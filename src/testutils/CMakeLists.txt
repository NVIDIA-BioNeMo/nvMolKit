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

if(NVMOLKIT_BUILD_TESTS OR NVMOLKIT_BUILD_BENCHMARKS)
  add_library(molData mol_data.cpp)
  target_link_libraries(molData PRIVATE ${RDKit_LIBS})

  add_library(substruct_validation substruct_validation.cu)
  target_link_libraries(
    substruct_validation
    PUBLIC ${RDKit_LIBS} substructure_search substruct_definitions
           cuda_error_check flatBitVect)
endif()

add_library(conformer_checkers conformer_checkers.cpp)
target_link_libraries(conformer_checkers PRIVATE ${RDKit_LIBS})
