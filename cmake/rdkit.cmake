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

# RDKit components nvmolkit actually links against. Used by both code paths
# below: find_package on the conda/system path, and resolved against
# rdkit.libs/libRDKit<Name>-*.so.1 on the pip-build path.
set(NVMOLKIT_RDKIT_COMPONENTS
    DataStructs
    Depictor
    Descriptors
    DistGeomHelpers
    FileParsers
    Fingerprints
    ForceField
    ForceFieldHelpers
    GraphMol
    MolStandardize
    MolTransforms
    PartialCharges
    RDGeneral
    RDGeometryLib
    SmilesParse
    SubstructMatch)

if(NOT NVMOLKIT_BUILD_AGAINST_PIP_RDKIT)
  find_package(RDKit REQUIRED)
  set(RDKit_LIBS "")
  foreach(component IN LISTS NVMOLKIT_RDKIT_COMPONENTS)
    list(APPEND RDKit_LIBS RDKit::${component})
  endforeach()

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

  # Resolve each component to its hash-mangled rdkit.libs/ filename. boost.cmake
  # handles libboost_* separately; everything else in rdkit.libs/ (libcairo,
  # libfontconfig, libfreetype, libxcb*, libXau, libpixman, libpng16,
  # libquadmath, libuuid, libbz2) is an auditwheel transitive that nvmolkit does
  # not consume.
  message(
    STATUS "Searched for RDKit libs in: ${NVMOLKIT_BUILD_AGAINST_PIP_LIBDIR}")
  set(RDKit_LIBS "")
  foreach(component IN LISTS NVMOLKIT_RDKIT_COMPONENTS)
    file(GLOB MATCHES
         ${NVMOLKIT_BUILD_AGAINST_PIP_LIBDIR}/libRDKit${component}-*.so.*)
    list(LENGTH MATCHES NUM_MATCHES)
    if(NOT NUM_MATCHES EQUAL 1)
      message(
        FATAL_ERROR
          "Expected exactly one libRDKit${component}-*.so.* under "
          "${NVMOLKIT_BUILD_AGAINST_PIP_LIBDIR}, got ${NUM_MATCHES}: ${MATCHES}"
      )
    endif()
    list(GET MATCHES 0 LIB_PATH)
    get_filename_component(libname ${LIB_PATH} NAME_WE)
    add_library(${libname} SHARED IMPORTED)
    set_target_properties(${libname} PROPERTIES IMPORTED_LOCATION ${LIB_PATH})
    target_include_directories(
      ${libname} SYSTEM INTERFACE ${NVMOLKIT_BUILD_AGAINST_PIP_INCDIR}
                                  ${NVMOLKIT_BUILD_AGAINST_PIP_BOOSTINCLUDEDIR})
    list(APPEND RDKit_LIBS ${libname})
  endforeach()
  message(STATUS "Imported RDKit pip libs: ${RDKit_LIBS}")
  # cmake-format: off
  set(Boost_INCLUDE_DIRS ${NVMOLKIT_BUILD_AGAINST_PIP_BOOSTINCLUDEDIR}) # cmake-lint: disable=C0103
  # cmake-format: on
endif(NOT NVMOLKIT_BUILD_AGAINST_PIP_RDKIT)
