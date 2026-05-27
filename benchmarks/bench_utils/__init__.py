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

"""Shared utilities for nvMolKit benchmarks.

Re-exports timing primitives, file loaders, and molecule preparation helpers
so individual bench scripts can ``from bench_utils import time_it, load_smiles``.
"""

from bench_utils.loaders import load_pickle, load_sdf, load_smarts, load_smiles
from bench_utils.molprep import clone_mols_with_conformers, embed_and_jitter, perturb_conformer, prep_mols
from bench_utils.timing import TimingResult, time_it

__all__ = [
    "TimingResult",
    "clone_mols_with_conformers",
    "embed_and_jitter",
    "load_pickle",
    "load_sdf",
    "load_smarts",
    "load_smiles",
    "perturb_conformer",
    "prep_mols",
    "time_it",
]
