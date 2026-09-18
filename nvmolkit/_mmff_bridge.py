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

"""Bridge RDKit MMFF property objects into nvMolKit's internal MMFF config.

This shim exists because RDKit ``Mol`` objects pass cleanly across the Python/C++
extension boundary as ``ROMol*``, but RDKit's Python ``MMFFMolProperties`` object
does not. It is a separate Boost.Python wrapper type owned by RDKit's forcefield
module, so nvMolKit accepts that object at the public Python API and converts it
into its own plain internal ``MMFFProperties`` transport object before calling
into the native extension.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rdkit.Chem import rdForceFieldHelpers
from rdkit.ForceField import rdForceField as _rdForceField  # noqa: F401

from nvmolkit import _batchedForcefield  # type: ignore

if TYPE_CHECKING:
    from rdkit.Chem import Mol
    from rdkit.ForceField.rdForceField import MMFFMolProperties as RDKitMMFFMolProperties


def default_rdkit_mmff_properties(mol: "Mol"):
    """Create default RDKit MMFF properties for ``mol``."""
    properties = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    if properties is None:
        raise ValueError("RDKit could not create MMFF properties for molecule")
    return properties


def make_internal_mmff_properties(
    properties: "RDKitMMFFMolProperties",
    *,
    non_bonded_threshold: float,
    ignore_interfrag_interactions: bool,
):
    """Convert an RDKit MMFF properties object into nvMolKit's internal transport.

    RDKit's Python binding only exposes setters for the scalar MMFF settings
    (variant, dielectric, per-term flags); the corresponding getters are not
    wrapped.  We read the settings through the C++ binding layer instead.
    """
    return _batchedForcefield.buildMMFFPropertiesFromRDKit(
        properties,
        float(non_bonded_threshold),
        bool(ignore_interfrag_interactions),
    )
