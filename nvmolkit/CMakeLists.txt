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

set(CMAKE_SHARED_MODULE_PREFIX "")

# Set up the libraries and header search paths for this target
add_library(_DataStructs MODULE DataStructs.cpp)
target_link_libraries(
  _DataStructs PUBLIC ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${RDKit_LIBS}
                      Similarity device)
target_include_directories(_DataStructs PRIVATE ${Python_INCLUDE_DIRS})
installpythontarget(_DataStructs ./)

add_library(_Fingerprints MODULE fingerprints.cpp)
target_link_libraries(
  _Fingerprints
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
  ${RDKit_LIBS}
  morganFingerprint
  device
  CUDA::cudart)
target_include_directories(_Fingerprints PRIVATE ${Python_INCLUDE_DIRS})
installpythontarget(_Fingerprints ./)

add_library(_arrayHelpers MODULE array_helpers.cpp)
target_link_libraries(_arrayHelpers PUBLIC ${Boost_LIBRARIES} device_vector)
target_include_directories(
  _arrayHelpers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/utils ${Python_INCLUDE_DIRS})
target_include_directories(_arrayHelpers SYSTEM PUBLIC ${Boost_INCLUDE_DIRS})
installpythontarget(_arrayHelpers ./)

add_library(_mmffOptimization MODULE mmffOptimization.cpp)
target_link_libraries(_mmffOptimization PUBLIC ${Boost_LIBRARIES}
                                               ${PYTHON_LIBRARIES})
target_link_libraries(_mmffOptimization PRIVATE ${RDKit_LIBS} bfgs_mmff
                                                ff_utils)
target_include_directories(
  _mmffOptimization PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/utils
                           ${Python_INCLUDE_DIRS})
target_include_directories(_mmffOptimization SYSTEM
                           PUBLIC ${Boost_INCLUDE_DIRS})
installpythontarget(_mmffOptimization ./)

add_library(_uffOptimization MODULE uffOptimization.cpp)
target_link_libraries(_uffOptimization PUBLIC ${Boost_LIBRARIES}
                                              ${PYTHON_LIBRARIES})
target_link_libraries(_uffOptimization PRIVATE ${RDKit_LIBS} bfgs_uff ff_utils)
target_include_directories(
  _uffOptimization PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/utils
                          ${Python_INCLUDE_DIRS})
target_include_directories(_uffOptimization SYSTEM PUBLIC ${Boost_INCLUDE_DIRS})
installpythontarget(_uffOptimization ./)

add_library(_batchedForcefield MODULE batchedForcefield.cpp)
target_link_libraries(_batchedForcefield PUBLIC ${Boost_LIBRARIES}
                                                ${PYTHON_LIBRARIES})
target_link_libraries(
  _batchedForcefield
  PRIVATE ${RDKit_LIBS}
          bfgs_mmff
          bfgs_uff
          mmff_batched_forcefield
          uff_batched_forcefield
          batched_forcefield
          rdkit_mmff_flattened
          rdkit_uff_flattened
          device
          device_vector
          ff_utils)
target_include_directories(
  _batchedForcefield PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/utils
                            ${Python_INCLUDE_DIRS})
target_include_directories(_batchedForcefield SYSTEM
                           PUBLIC ${Boost_INCLUDE_DIRS})
installpythontarget(_batchedForcefield ./)

add_library(_embedMolecules MODULE embedMolecules.cpp)
target_link_libraries(_embedMolecules PUBLIC ${Boost_LIBRARIES}
                                             ${PYTHON_LIBRARIES} device_vector)
target_link_libraries(_embedMolecules PRIVATE ${RDKit_LIBS} dist_geom etkdg)
target_include_directories(
  _embedMolecules PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/utils
                         ${Python_INCLUDE_DIRS})
target_include_directories(_embedMolecules SYSTEM PUBLIC ${Boost_INCLUDE_DIRS})
installpythontarget(_embedMolecules ./)

add_library(_clustering MODULE clustering.cpp)
target_include_directories(
  _clustering PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/utils ${Boost_INCLUDE_DIRS}
                     ${Python_INCLUDE_DIRS})
target_link_libraries(_clustering PUBLIC ${Boost_LIBRARIES} ${PYTHON_LIBRARIES}
                                         device_vector butina device)
installpythontarget(_clustering ./)

add_library(_conformerRmsd MODULE conformerRmsd.cpp)
target_include_directories(
  _conformerRmsd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/utils ${Boost_INCLUDE_DIRS}
                        ${Python_INCLUDE_DIRS})
target_link_libraries(
  _conformerRmsd PUBLIC ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} device_vector
                        conformer_rmsd_mol device)
target_link_libraries(_conformerRmsd PRIVATE ${RDKit_LIBS})
installpythontarget(_conformerRmsd ./)

add_library(_substructure MODULE substructure.cpp)
target_include_directories(
  _substructure
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/utils ${Boost_INCLUDE_DIRS}
         ${Python_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src/substruct)
target_link_libraries(_substructure PUBLIC ${Boost_LIBRARIES}
                                           ${PYTHON_LIBRARIES})
target_link_libraries(_substructure PRIVATE ${RDKit_LIBS} substructure_search
                                            pinned_host_allocator)
installpythontarget(_substructure ./)

add_library(_TFD MODULE tfd.cpp)
target_include_directories(
  _TFD PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/utils ${Boost_INCLUDE_DIRS}
              ${Python_INCLUDE_DIRS})
target_link_libraries(_TFD PUBLIC ${Boost_LIBRARIES} ${PYTHON_LIBRARIES}
                                  device_vector)
target_link_libraries(_TFD PRIVATE ${RDKit_LIBS} tfd_gpu CUDA::cudart)
installpythontarget(_TFD ./)
