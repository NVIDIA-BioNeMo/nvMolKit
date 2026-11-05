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

if(NVMOLKIT_BUILD_PYTHON_BINDINGS)
  # Build the internal libraries as shared libraries installed next to the
  # Python modules. Static libraries copy their objects, including each CUDA
  # fatbin, into every module that links them.
  set(BUILD_SHARED_LIBS ON)
  # Load one libcudart for the modules and the libraries.
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  set(CMAKE_INSTALL_RPATH "$ORIGIN")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

  add_custom_target(
    installPythonLibrariesTarget
    COMMAND ${CMAKE_COMMAND} --install .
    COMMENT "Installing Python libraries")

  # Add a boost python target to the
  function(installPythonTarget targetModule subpackage)
    install(TARGETS ${targetModule} DESTINATION ${subpackage})
    add_dependencies(installPythonLibrariesTarget ${targetModule})
  endfunction(
    installPythonTarget
    targetModule
    subpackage)

  # Install every shared library target defined under directory into subpackage.
  function(installPythonSharedLibraries directory subpackage)
    get_property(
      targets
      DIRECTORY ${directory}
      PROPERTY BUILDSYSTEM_TARGETS)
    foreach(target IN LISTS targets)
      get_target_property(targetType ${target} TYPE)
      if(targetType STREQUAL "SHARED_LIBRARY")
        installpythontarget(${target} ${subpackage})
      endif()
    endforeach()
    get_property(
      subdirectories
      DIRECTORY ${directory}
      PROPERTY SUBDIRECTORIES)
    foreach(subdirectory IN LISTS subdirectories)
      installpythonsharedlibraries(${subdirectory} ${subpackage})
    endforeach()
  endfunction()

else()
  # Dummy implementation to avoid branching in other cmake code
  function(installPythonTarget targetModule subpackage)

  endfunction(
    installPythonTarget
    targetModule
    subpackage)

  # Dummy implementation to avoid branching in other cmake code
  function(installPythonSharedLibraries directory subpackage)

  endfunction()

endif()
