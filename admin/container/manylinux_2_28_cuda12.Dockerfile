# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# manylinux_2_28 image with CUDA + rdkit native build dependencies + Python
# build dependencies pre-installed across all supported CPython interpreters.
# Used by the pip wheel CI pipeline. Hosted on the org's GHCR; the workflow
# pulls a fixed date tag rather than rebuilding. To produce a new tag (only
# needed when this Dockerfile changes), build and push manually:
#
#   tag=$(date -u +%Y.%m.%d)
#   docker build --network host \
#     -f admin/container/manylinux_2_28_cuda12.Dockerfile \
#     -t ghcr.io/nvidia-digital-bio/nvmolkit-manylinux-cuda12:${tag} .
#   docker push ghcr.io/nvidia-digital-bio/nvmolkit-manylinux-cuda12:${tag}
#
# Then bump MANYLINUX_CUDA_IMAGE in .github/workflows/pip-build.yml.
#
# Image contents:
#   - PyPA manylinux_2_28 x86_64 base, RHEL 8 with /opt/python/cp310..cp314
#   - CUDA Toolkit 12.9 (gcc-14 compatible; required because the manylinux
#     image ships gcc-toolset-14 which kuelumbus/rdkit-pypi uses)
#   - patchelf + zip/unzip (auditwheel + ad-hoc post-processing)
#   - RDKit native build deps (freetype, libpng, pixman, zlib, eigen3, cairo).
#     These mirror kuelumbus/rdkit-pypi's pyproject.toml [tool.cibuildwheel.linux]
#     before-all so the recipe build matches the upstream PyPI build env.
#   - Python build deps installed into each supported interpreter:
#     setuptools, wheel, conan>=2.0, ninja, numpy, pybind11-stubgen, Pillow.
#     These are read by the rdkit-pypi setup.py during the recipe build.
#
# CUDA major is locked to 12 for the v0.5 release line. Bumping the patch is
# safe because libcudart's SONAME is libcudart.so.12 and we declare a runtime
# dependency on nvidia-cuda-runtime-cu12 in pyproject.toml.

FROM quay.io/pypa/manylinux_2_28_x86_64

# CUDA 12.9 is the minimum patch that supports gcc 14, which is the toolchain
# the manylinux image's gcc-toolset-14 provides and which kuelumbus/rdkit-pypi
# uses to build its PyPI wheel. Earlier 12.x patches reject gcc 14 with
# "unsupported GNU version" via crt/host_config.h.
ARG CUDA_VERSION=12-9
ARG SUPPORTED_PYTHON_TAGS="cp310-cp310 cp311-cp311 cp312-cp312 cp313-cp313 cp314-cp314"

RUN dnf -y install dnf-plugins-core \
    && dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
    && dnf -y clean all

# System packages: CUDA toolkit + auditwheel/post-processing tools + RDKit
# native build deps. The rdkit deps mirror kuelumbus/rdkit-pypi's before-all;
# changes to that upstream list need to be reflected here.
RUN dnf -y install \
        cuda-toolkit-${CUDA_VERSION} \
        patchelf \
        zip unzip \
        freetype-devel \
        libpng-devel \
        pixman-devel \
        zlib-devel \
        eigen3-devel \
        cairo-devel \
    && dnf -y clean all \
    && rm -rf /var/cache/dnf

ENV PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
    CUDA_HOME=/usr/local/cuda \
    CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

# Python build dependencies, pre-installed into each supported CPython.
# Mirrors kuelumbus/rdkit-pypi's [build-system].requires so the recipe build
# inside this image does no further pip-resolution by default.
RUN for tag in ${SUPPORTED_PYTHON_TAGS}; do \
        /opt/python/${tag}/bin/python -m pip install --no-cache-dir \
            setuptools wheel \
            'conan>=2.0' ninja \
            numpy \
            pybind11-stubgen Pillow \
        || exit 1; \
    done

RUN nvcc --version
