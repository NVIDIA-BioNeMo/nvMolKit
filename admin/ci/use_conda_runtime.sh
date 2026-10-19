#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Source after conda activate, before configuring, building, or running tests.
# GCC 13's private library directory can otherwise select an older libstdc++
# than prebuilt RDKit requires. LDFLAGS initializes CMake's executable, shared,
# and module linker flags, including builds driven by pip/scikit-build.
export LDFLAGS="-L${CONDA_PREFIX:?Activate the CI conda environment first}/lib -Wl,-rpath,${CONDA_PREFIX}/lib${LDFLAGS:+ ${LDFLAGS}}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
