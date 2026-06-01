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

# cmake-lint: disable=C0103

set(BOOST_TARGET_LIBS serialization iostreams)
if(NVMOLKIT_BUILD_PYTHON_BINDINGS)
  list(APPEND BOOST_TARGET_LIBS
       "python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}")
  # Link Boost.Python.Numpy as we use boost::python::numpy in DataStructs.cpp
  list(APPEND BOOST_TARGET_LIBS
       "numpy${Python_VERSION_MAJOR}${Python_VERSION_MINOR}")
endif()

if(NVMOLKIT_BUILD_AGAINST_PIP_RDKIT)
  message(STATUS "Using boost libs from pip RDKit: ${BOOST_TARGET_LIBS}")
  # rdkit-pypi hash-mangles SONAMEs (e.g.
  # libboost_python312-ed6a74e7.so.1.85.0), so we glob per component rather than
  # calling find_package.
  set(Boost_LIBRARIES "")
  foreach(component IN LISTS BOOST_TARGET_LIBS)
    file(GLOB MATCHES
         ${NVMOLKIT_BUILD_AGAINST_PIP_LIBDIR}/libboost_${component}-*.so.*)
    list(LENGTH MATCHES NUM_MATCHES)
    if(NOT NUM_MATCHES EQUAL 1)
      message(
        FATAL_ERROR
          "Expected exactly one libboost_${component}-*.so.* under "
          "${NVMOLKIT_BUILD_AGAINST_PIP_LIBDIR}, got ${NUM_MATCHES}: ${MATCHES}"
      )
    endif()
    list(GET MATCHES 0 LIB_PATH)
    get_filename_component(libname ${LIB_PATH} NAME_WE)
    add_library(${libname} SHARED IMPORTED)
    set_target_properties(${libname} PROPERTIES IMPORTED_LOCATION ${LIB_PATH})
    target_include_directories(
      ${libname} SYSTEM INTERFACE ${NVMOLKIT_BUILD_AGAINST_PIP_BOOSTINCLUDEDIR})
    list(APPEND Boost_LIBRARIES ${libname})
  endforeach()
else()
  message(STATUS "Finding boost libs: ${BOOST_TARGET_LIBS}")
  find_package(Boost REQUIRED COMPONENTS ${BOOST_TARGET_LIBS})
endif()
