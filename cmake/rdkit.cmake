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

if(NOT NVMOLKIT_BUILD_AGAINST_PIP_RDKIT)
  find_package(RDKit REQUIRED)
  set(RDKit_LIBS
      RDKit::DataStructs
      RDKit::Depictor
      RDKit::Descriptors
      RDKit::DistGeomHelpers
      RDKit::FileParsers
      RDKit::Fingerprints
      RDKit::ForceField
      RDKit::ForceFieldHelpers
      RDKit::GraphMol
      RDKit::MolStandardize
      RDKit::MolTransforms
      RDKit::PartialCharges
      RDKit::RDGeneral
      RDKit::RDGeometryLib
      RDKit::SmilesParse
      RDKit::SubstructMatch)

  # For RDKit 2023.5 onwards, the rdkit::rdbase target improperly has hardcoded
  # interface include directories that use the python version they were built
  # against. We replace these with the right python version or remove if in a
  # C++ build.
  function(replace_or_remove_python_version list_var user_input remove)
    set(new_list "")
    foreach(item IN LISTS ${list_var})
      if(remove)
        if(item MATCHES "python3\\.[0-9]+")
          continue()
        endif()
      else()
        string(REGEX REPLACE "python3\\.[0-9]+" ${user_input} item ${item})
      endif()
      list(APPEND new_list ${item})
    endforeach()
    set(${list_var}
        ${new_list}
        PARENT_SCOPE)
  endfunction()

  # Set variable SHOULD_REMOVE = ! NVMOLKIT_BUILD_PYTHON_BINDINGS
  set(SHOULD_REMOVE NOT ${NVMOLKIT_BUILD_PYTHON_BINDINGS})
  get_property(
    rdbase_links
    TARGET RDKit::rdkit_base
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
  set(PYTHON_REPLACE_VERSION "python3.${Python_VERSION_MINOR}")
  replace_or_remove_python_version(rdbase_links ${USER_INPUT}
                                   ${PYTHON_REPLACE_VERSION} SHOULD_REMOVE)
  set_property(TARGET RDKit::rdkit_base PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                                 ${rdbase_links})

else()
  if(NOT NVMOLKIT_BUILD_AGAINST_PIP_LIBDIR)
    message(
      FATAL_ERROR
        "NVMOLKIT_BUILD_AGAINST_PIP_LIBDIR must be set for builds against pip install"
    )
  endif()
  if(NOT NVMOLKIT_BUILD_AGAINST_PIP_INCDIR)
    message(
      FATAL_ERROR
        "NVMOLKIT_BUILD_AGAINST_PIP_INCDIR must be set for builds against pip install"
    )
  endif()
  if(NOT NVMOLKIT_BUILD_AGAINST_PIP_BOOSTINCLUDEDIR)
    message(
      FATAL_ERROR
        "NVMOLKIT_BUILD_AGAINST_PIP_BOOSTINCLUDEDIR must be set for builds against pip install"
    )
  endif()

  # make a list of all files ine the libdir
  file(GLOB RDKIT_FILES ${NVMOLKIT_BUILD_AGAINST_PIP_LIBDIR}/*)
  # for each file, make an imported library with that lib as source
  message(
    STATUS "Searched for RDKit libs in: ${NVMOLKIT_BUILD_AGAINST_PIP_LIBDIR}")
  message(STATUS "Found RDKit pip libs: ${RDKIT_FILES}")
  # Populate RDKIT_LIBS with the imported libraries
  foreach(lib ${RDKIT_FILES})
    get_filename_component(libname ${lib} NAME_WE)
    add_library(${libname} SHARED IMPORTED)
    set_target_properties(${libname} PROPERTIES IMPORTED_LOCATION ${lib})
    # Set include dirs to include both NVMOLKIT_BUILD_AGAINST_PIP_INCDIR and
    # NVMOLKIT_BUILD_AGAINST_PIP_BOOSTINCLUDEDIR
    target_include_directories(
      ${libname} SYSTEM INTERFACE ${NVMOLKIT_BUILD_AGAINST_PIP_INCDIR}
                                  ${NVMOLKIT_BUILD_AGAINST_PIP_BOOSTINCLUDEDIR})
    list(APPEND RDKit_LIBS ${libname})
  endforeach()
  # cmake-lint: disable=C0103
  set(Boost_INCLUDE_DIRS ${NVMOLKIT_BUILD_AGAINST_PIP_BOOSTINCLUDEDIR})
  message(STATUS "Using boost libs from pip RDKit")
endif(NOT NVMOLKIT_BUILD_AGAINST_PIP_RDKIT)
