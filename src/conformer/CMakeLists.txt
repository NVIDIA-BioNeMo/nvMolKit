# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

add_library(conformer_types INTERFACE)

add_library(device_coord_collector device_coord_collector.cpp)
target_link_libraries(
  device_coord_collector
  PUBLIC conformer_types device_vector
  PRIVATE p2p device cuda_error_check)

add_library(etkdg_device_collect etkdg_device_collect.cu)
target_link_libraries(
  etkdg_device_collect
  PUBLIC conformer_types device_coord_collector device_vector
  PRIVATE device cuda_error_check)

add_library(ff_device_collect ff_device_collect.cpp)
target_link_libraries(
  ff_device_collect
  PUBLIC conformer_types device_coord_collector device_vector
  PRIVATE p2p device cuda_error_check ${RDKit_LIBS})

add_library(conformer_coord_upload conformer_coord_upload.cpp)
target_link_libraries(
  conformer_coord_upload
  PUBLIC conformer_types device_vector
  PRIVATE cuda_error_check ${RDKit_LIBS} OpenMP::OpenMP_CXX)
