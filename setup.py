# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

from setuptools import find_packages
from skbuild import setup

pyroot = os.getenv("CONDA_PREFIX")

cmake_extra_args = []
if pyroot:
    cmake_extra_args.append(f"-DCMAKE_PREFIX_PATH={pyroot}")

# Detect if we're doing an install against pip rdkit
nvmolkit_build_against_pip = os.getenv("NVMOLKIT_BUILD_AGAINST_PIP_RDKIT")
if nvmolkit_build_against_pip:
    NVMOLKIT_BUILD_AGAINST_PIP_LIBDIR = os.getenv("NVMOLKIT_BUILD_AGAINST_PIP_LIBDIR")
    if not NVMOLKIT_BUILD_AGAINST_PIP_LIBDIR:
        raise ValueError("NVMOLKIT_BUILD_AGAINST_PIP_LIBDIR must be set when building against pip rdkit")
    NVMOLKIT_BUILD_AGAINST_PIP_INCDIR = os.getenv("NVMOLKIT_BUILD_AGAINST_PIP_INCDIR")
    if not NVMOLKIT_BUILD_AGAINST_PIP_INCDIR:
        raise ValueError("NVMOLKIT_BUILD_AGAINST_PIP_INCDIR must be set when building against pip rdkit")
    NVMOLKIT_BUILD_AGAINST_PIP_BOOSTINCLUDEDIR = os.getenv("NVMOLKIT_BUILD_AGAINST_PIP_BOOSTINCLUDEDIR")
    if not NVMOLKIT_BUILD_AGAINST_PIP_BOOSTINCLUDEDIR:
        raise ValueError("NVMOLKIT_BUILD_AGAINST_PIP_BOOSTINCLUDEDIR must be set when building against pip rdkit")
    cmake_extra_args.extend(
        [
            "-DNVMOLKIT_BUILD_AGAINST_PIP_RDKIT=ON",
            f"-DNVMOLKIT_BUILD_AGAINST_PIP_LIBDIR={NVMOLKIT_BUILD_AGAINST_PIP_LIBDIR}",
            f"-DNVMOLKIT_BUILD_AGAINST_PIP_INCDIR={NVMOLKIT_BUILD_AGAINST_PIP_INCDIR}",
            f"-DNVMOLKIT_BUILD_AGAINST_PIP_BOOSTINCLUDEDIR={NVMOLKIT_BUILD_AGAINST_PIP_BOOSTINCLUDEDIR}",
        ]
    )

    # find_package(RDKit) is skipped on the pip path, so the version variables
    # that src/versions.h.in expects (RDKit_VERSION_{MAJOR,MINOR,PATCH}) are
    # never populated. Derive them from the RDKIT_VERSION env var the CI/dev
    # driver is required to set.
    rdkit_version_str = os.getenv("RDKIT_VERSION")
    if not rdkit_version_str:
        raise ValueError("RDKIT_VERSION must be set to a value of the form 'YYYY.M.P' when building against pip rdkit")
    rdkit_parts = rdkit_version_str.split(".")
    if len(rdkit_parts) != 3 or not all(p.isdigit() for p in rdkit_parts):
        raise ValueError(f"RDKIT_VERSION must be 'YYYY.M.P' (got: {rdkit_version_str!r})")
    cmake_extra_args.extend(
        [
            f"-DRDKit_VERSION_MAJOR={rdkit_parts[0]}",
            f"-DRDKit_VERSION_MINOR={rdkit_parts[1]}",
            f"-DRDKit_VERSION_PATCH={rdkit_parts[2]}",
        ]
    )


if __name__ == "__main__":
    setup(
        cmake_languages=("CXX", "CUDA"),
        cmake_args=[
            "-DNVMOLKIT_EXTRA_DEV_FLAGS=OFF",
            "-DNVMOLKIT_BUILD_PYTHON_BINDINGS=ON",
            "-DNVMOLKIT_BUILD_TESTS=OFF",
            "-DNVMOLKIT_BUILD_BENCHMARKS=OFF",
            f"-DNVMOLKIT_CUDA_TARGET_MODE={os.getenv('NVMOLKIT_CUDA_TARGET_MODE', 'full')}",
            f"-DCMAKE_BUILD_TYPE={os.getenv('CMAKE_BUILD_TYPE', 'Release')}",
            # "-DBoost_NO_BOOST_CMAKE=TRUE"
        ]
        + cmake_extra_args,
        packages=find_packages(include=["nvmolkit", "nvmolkit.*"], exclude=["nvmolkit.tests*"]),
        exclude_package_data={"nvmolkit": ["tests/*", "*.cpp"]},
        package_data={"nvmolkit": ["**/*.csv"]},
        cmake_install_dir="nvmolkit",
        cmake_install_target="installPythonLibrariesTarget",
    )
