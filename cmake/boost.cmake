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

if(NVMOLKIT_BUILD_AGAINST_PIP_RDKIT)
  message(STATUS "Using boost libs from pip RDKit")
  # rdkit.cmake already enumerated every .so under rdkit.libs/ as an IMPORTED
  # target and appended each to RDKit_LIBS. Filter out the boost ones so targets
  # that link against ${Boost_LIBRARIES} (rather than ${RDKit_LIBS}) still pull
  # in libboost_python312, libboost_serialization, etc.
  set(BOOST_LIBRARIES_FROM_PIP "")
  foreach(lib IN LISTS RDKit_LIBS)
    if(lib MATCHES "^libboost_")
      list(APPEND BOOST_LIBRARIES_FROM_PIP ${lib})
    endif()
  endforeach()
  set(Boost_LIBRARIES ${BOOST_LIBRARIES_FROM_PIP})
else()
  set(BOOST_TARGET_LIBS serialization iostreams)
  if(NVMOLKIT_BUILD_PYTHON_BINDINGS)
    list(APPEND BOOST_TARGET_LIBS
         "python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}")
    # Link Boost.Python.Numpy as we use boost::python::numpy in DataStructs.cpp
    list(APPEND BOOST_TARGET_LIBS
         "numpy${Python_VERSION_MAJOR}${Python_VERSION_MINOR}")
  endif()
  message(STATUS "Finding boost libs: ${BOOST_TARGET_LIBS}")
  find_package(Boost REQUIRED COMPONENTS ${BOOST_TARGET_LIBS})
endif()
