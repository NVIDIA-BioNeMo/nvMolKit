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

"""Optuna-backed autotuning for nvMolKit hardware options.

This subpackage is **opt-in**: it depends on the optional ``optuna`` package.
Importing :mod:`nvmolkit.autotune` itself never fails, even when ``optuna`` is
not installed — only calling the ``tune_*`` functions raises an
:class:`ImportError` with installation instructions. Users on conda-forge can
still load tuned configurations through :func:`load` without ``optuna``.

Use :func:`is_available` to check whether autotuning is supported in the
current environment::

    from nvmolkit import autotune

    if autotune.is_available():
        result = autotune.tune_embed_molecules(mols, params)
        autotune.save(result.best_config, "etkdg_options.json")

The ``tune_*`` functions return a :class:`TuneResult` whose ``best_config`` is
a fully-populated :class:`~nvmolkit.types.HardwareOptions` suitable for passing
directly into the corresponding API on the full workload.
"""

from nvmolkit.autotune._core import (
    OPTUNA_INSTALL_HINT,
    CalibrationState,
    TuneResult,
    is_optuna_available,
)
from nvmolkit.autotune._persistence import load, save
from nvmolkit.autotune.tune_batched_forcefield import tune_batched_forcefield
from nvmolkit.autotune.tune_embed_molecules import tune_embed_molecules
from nvmolkit.autotune.tune_mmff_optimize import tune_mmff_optimize
from nvmolkit.autotune.tune_substructure import tune_substructure
from nvmolkit.autotune.tune_uff_optimize import tune_uff_optimize


def is_available() -> bool:
    """Return ``True`` if autotune dependencies are present (currently optuna)."""
    return is_optuna_available()


__all__ = [
    "CalibrationState",
    "OPTUNA_INSTALL_HINT",
    "TuneResult",
    "is_available",
    "is_optuna_available",
    "load",
    "save",
    "tune_batched_forcefield",
    "tune_embed_molecules",
    "tune_mmff_optimize",
    "tune_substructure",
    "tune_uff_optimize",
]
