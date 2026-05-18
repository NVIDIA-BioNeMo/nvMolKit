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

"""nvMolKit - GPU-accelerated RDKit functionality.

nvMolKit provides GPU-accelerated implementations of common RDKit operations
to improve performance for large-scale cheminformatics workflows. APIs match
RDKit as closely as possible, but expand to support batches of molecules, which
is critical for GPU performance.

Currently supported functionality:
- Batch Morgan Fingerprint calculation
- Bulk tanimoto/cosine similarity calculations between fingerprints
- ETKDG conformer generation for multiple molecules
- MMFF optimization for multiple molecules and conformers
- UFF optimization for multiple molecules and conformers
- TFD (Torsion Fingerprint Deviation) calculation for conformer comparison
- Optional autotuning of hardware options via :mod:`nvmolkit.autotune`
  (requires the ``optuna`` package, installable via the ``autotune`` extra)
"""

VERSION = "0.5.0"
__version__ = VERSION
