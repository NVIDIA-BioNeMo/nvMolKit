# nvMolKit

## Documentation
Please see the official [NVIDIA nvMolKit Documentation](https://nvidia-digital-bio.github.io/nvMolKit/) for an overview of features, examples, and a detailed API reference.

## Cursor / agent skill

If you use [Cursor](https://cursor.com/) (or another agent that supports the `SKILL.md` format) to write code that calls nvMolKit, you can copy [`agent-skills/nvmolkit-usage/`](agent-skills/nvmolkit-usage/) into your project's `.cursor/skills/` (or `~/.cursor/skills/` for personal use). It gives the agent the public Python entry-point map, runtime requirements, and runnable recipes.

## Installation Guide

**IMPORTANT**: nvMolKit requires an NVIDIA GPU with compute capability 7.0 (V100) or higher to run. Check your GPU's compute capability [here](https://developer.nvidia.com/cuda-gpus).
It also requires a CUDA Driver sufficient for CUDA 12.6 or later (driver version >=560.28), though some backwards compatibility may be supported, see the [CUDA compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/index.html).

### Conda-forge Installation (Recommended)

Conda is the recommended way to install nvMolKit, matching the recommended distribution mechanism of the RDKit. First, ensure 
you have a variant of conda installed and activated, such as [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 
or [Miniforge](https://conda-forge.org/download/).

nvMolKit v0.5.0 supports RDKit 2025.03.1 through 2026.03.1. To install:

```bash
conda install -c conda-forge nvmolkit
```

**Note**: If your system's CUDA driver does not support CUDA 13, the default install may
resolve `cuda-version=13` and fall back to a CPU-only PyTorch. To ensure GPU-accelerated
PyTorch is installed:

```bash
conda install -c conda-forge nvmolkit cuda-version=12.6
```
Choose a `cuda-version` that is **≤** the CUDA version reported by `nvidia-smi` and that
has a matching PyTorch build on conda-forge. See the
[available PyTorch builds](https://anaconda.org/channels/conda-forge/packages/pytorch/files?file_q=cuda)
to find supported CUDA versions.


### Installation from Source
#### Prerequisites

##### An NVIDIA GPU
See the note above about GPU requirements.

##### System Dependencies

First, install essential build dependencies. This includes a C++ compiler and OpenMP. Eigen headers may be necesary,
sometimes RDKit includes them in some headers but the RDKit install does not always properly declare this dependency.

Example shown for installing on Ubuntu. System installs or conda installs both work.
```bash
# Update package list
sudo apt-get update

# Install build tools and development headers
sudo apt-get install build-essential libeigen3-dev
sudo apt-get install libstdc++-12-dev libomp-15-dev

# nvMolKit requires a C++ compiler. You can install it system-wide or via conda:

# Example: Install clang on Ubuntu:
sudo apt-get install clang-15 clang-format-15 clang-tidy-15

# Other options:
# - Use system GCC (already included in build-essential above)
# - Install inside a conda environment (see Python Environment Setup section below):
#   conda install -c conda-forge cxx-compiler
```

##### CUDA Installation

Install NVIDIA CUDA Toolkit (version 12.5 or later) following [NVIDIA's official installation guide](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network).

##### CMake

nvMolKit requires CMake >= 3.26. Update if needed, for example on Ubuntu:

```bash
# Remove old CMake
sudo apt remove --purge --auto-remove cmake

# Install CMake 3.30.1
wget https://github.com/Kitware/CMake/releases/download/v3.30.1/cmake-3.30.1-linux-x86_64.sh
chmod +x cmake-3.30.1-linux-x86_64.sh
sudo ./cmake-3.30.1-linux-x86_64.sh --prefix=/usr/local --skip-license

# Verify installation
cmake --version
```

A conda cmake should also work.


##### Python Environment Setup

Create a conda environment with all required dependencies (install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) or [Anaconda](https://www.anaconda.com/download) if you don't have conda):

```bash
# Create and activate environment
conda create --name nvmolkit_dev_py312 python=3.12.1
conda activate nvmolkit_dev_py312

# Install RDKit with development headers
conda install -c conda-forge rdkit=2025.03.6 rdkit-dev=2025.03.6

# Install Boost subpackages in case RDKit install did not include them transitively
conda install -c conda-forge libboost libboost-python libboost-devel libboost-headers libboost-python-devel

# Install Torch, make sure it's a GPU-enabled version. If having trouble install, check out the
# torch installation guidelines: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio
python -c "import torch; print(torch.__version__); print(f'Is a CUDA build? {torch.cuda.is_available()}')"
```

#### Installation

```bash
# Activate your environment
conda activate nvmolkit_dev_py312

# Navigate to the repo root directory
cd <path/to/nvmolkit>

# Install nvMolKit directly
#  Use all CPU cores for a faster build, or replace $(nproc) with a specific number
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) pip -v install .
```
This will build and install nvMolKit with Python bindings automatically.

To test the installation, run
```bash
pip install pytest
(cd nvmolkit/tests && pytest -v .)
```

#### Docker containers

Materials for building docker containers are in [`admin/container`](admin/container). [HPCCM](https://github.com/NVIDIA/hpc-container-maker)
builds docker files from yaml configs. See the [README](admin/container/hpccm_build.py) for config definitions.


## Developer Guide

### cppcheck Installation

For code quality analysis during development:

```bash
wget https://github.com/danmar/cppcheck/archive/2.14.2.tar.gz
tar -zxvf 2.14.2.tar.gz
cd cppcheck-2.14.2
mkdir build && cd build
cmake .. -DUSE_MATCHCOMPILER=ON -DCMAKE_BUILD_TYPE=release
make -j
sudo make install
cd ../../ && rm -rf cppcheck-2.14.2 2.14.2.tar.gz
```

### Development Build Options

#### Compilers

- **Supported Compilers**: We have tested and support clang-15 and GCC 12. Other compilers may work but are not extensively tested.

#### CMake Build Options

- **`-DCMAKE_BUILD_TYPE=<type>`**: Available options for build type include `Release`, `Debug`, `RelWithDebInfo`, `asan`, `tsan`, and `ubsan`.

- **`-DNVMOLKIT_BUILD_TESTS=ON`**: Enables building unit tests. CMake will download and build GTest automatically. Run tests with `ctest` after building.

- **`-DNVMOLKIT_BUILD_BENCHMARKS=ON`**: Enables building performance benchmarks. CMake will download and build nanobench automatically. After building, executable benchmarks will be found in `build/benchmarks`.

- **`-DNVMOLKIT_BUILD_PYTHON_BINDINGS=ON`**: Builds Python bindings using boost-python. Required for Python API access. This ensures compatibility with RDKit's Python bindings.

- **`-DNVMOLKIT_CUDA_TARGET_MODE=<mode>`**: Controls GPU target architectures. See GPU Target Architectures section below for available modes.

- **`-DNVMOLKIT_BUILD_AGAINST_PIP_RDKIT=ON`**: Build against pip-installed RDKit instead of conda. See Building Against pip-installed RDKit section below for additional required configuration. Default: `OFF`.

#### GPU Target Architectures

nvMolKit supports building for multiple GPU architectures. Build behavior is controlled by the `NVMOLKIT_CUDA_TARGET_MODE` variable:

- **`default`**: Uses `CMAKE_CUDA_ARCHITECTURES` if set, otherwise defaults to compute capability 7.0
- **`native`**: Builds only for the GPU on your current system. Fastest for local development but not portable.
- **`full`**: Builds for all architectures >= 7.0, including Blackwell (if NVCC >= 12.8). Larger binaries, longer compile time, but works on all major GPUs.

**Recommendation**: Use `native` for development, `full` for distribution.

#### Building Against pip-installed RDKit

The pip wheel pipeline reproduces the [kuelumbus/rdkit-pypi](https://github.com/kuelumbus/rdkit-pypi) build at a pinned tag to obtain bit-exact rdkit headers and matching boost headers, links nvMolKit against the pip-installed rdkit's auditwheel-rewritten libraries, and produces a `manylinux_2_28` wheel. The supported (rdkit, python) pairs are listed in [`admin/distribute/rdkit_build_matrix.yaml`](admin/distribute/rdkit_build_matrix.yaml).

The conda-based development setup above is still the recommended path for day-to-day development. The pip pipeline is driven exclusively by [cibuildwheel](https://cibuildwheel.pypa.io/) so the build environment matches kuelumbus/rdkit-pypi's CI exactly (manylinux entrypoint, gcc-toolset activation, and `before-all` system-dep install).

To build wheels locally, use the same path CI uses:

```bash
docker build -f admin/container/manylinux_2_28_cuda12.Dockerfile \
  -t nvmolkit-manylinux-cuda12:test .
pip install cibuildwheel
CIBW_MANYLINUX_X86_64_IMAGE=nvmolkit-manylinux-cuda12:test \
  RDKIT_VERSION=2026.3.1 \
  bash admin/deploy/build_pip_wheels.sh 2026.3.1 wheelhouse
```

To narrow the matrix while iterating, set `CIBW_BUILD=cp312-manylinux_x86_64` (or whichever python tag you care about) before invoking the script. Wheels land in `wheelhouse/`.

The full CI pipeline is at [`.github/workflows/pip-build.yml`](.github/workflows/pip-build.yml). It runs on demand (`workflow_dispatch` only), expands the (rdkit, python) matrix from [`admin/distribute/rdkit_build_matrix.yaml`](admin/distribute/rdkit_build_matrix.yaml), and pulls the pre-built manylinux+CUDA image from the org's GHCR (`ghcr.io/nvidia-digital-bio/nvmolkit-manylinux-cuda12`). The image is rebuilt and pushed manually when the Dockerfile changes; the build script header documents the push command.

Internally, cibuildwheel's `before-build` hook (see [`admin/distribute/cibuildwheel_before_build.sh`](admin/distribute/cibuildwheel_before_build.sh)) clones rdkit-pypi at the matching tag, runs [`admin/distribute/build_rdkit_recipe.sh`](admin/distribute/build_rdkit_recipe.sh) to reproduce its build (~30-60 min on first invocation; cached afterwards), pip-installs the matching rdkit wheel for runtime SONAME-matching libs, and stages everything at stable paths under `/tmp/nvmolkit_pip_inputs/`. setup.py picks those up via `NVMOLKIT_BUILD_AGAINST_PIP_*` env vars set in pyproject.toml's `[tool.cibuildwheel.linux].environment`.

