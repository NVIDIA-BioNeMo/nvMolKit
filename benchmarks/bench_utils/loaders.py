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

"""Shared molecule and pattern loaders for nvMolKit benchmarks.

Molecule loaders accept ``max_count`` to cap the workload, with a uniform
random sample drawn (via reservoir sampling for streaming inputs) when the
source contains more entries than requested.
"""

import pickle
import random
from functools import partial
from typing import Iterator

from rdkit import Chem, RDLogger
from tqdm.contrib.concurrent import process_map


def _mol_from_binary(binary_mol: bytes) -> Chem.Mol:
    return Chem.Mol(binary_mol)


def load_pickle(filepath: str, max_count: int = 0, seed: int | None = None) -> list[Chem.Mol]:
    """Load molecules from a pickle file containing a list of RDKit binary molecules.

    Args:
        filepath: Path to the pickle file. Must contain a list of ``bytes``
            payloads as produced by :meth:`Chem.Mol.ToBinary`.
        max_count: When positive and the source has more entries, draw a
            uniform random sample of this size before unpickling.
        seed: Optional seed for the sampling RNG.

    Returns:
        List of parsed RDKit molecules. The list is always shuffled
        (deterministic with ``seed``) so benches that consume a head slice
        get a representative cross-section rather than file-order bias.
    """
    with open(filepath, "rb") as fh:
        binary_mols = pickle.load(fh)
    rng = random.Random(seed)
    if max_count > 0 and len(binary_mols) > max_count:
        binary_mols = rng.sample(binary_mols, max_count)
    else:
        binary_mols = list(binary_mols)
        rng.shuffle(binary_mols)
    mols = process_map(
        _mol_from_binary,
        binary_mols,
        desc="Unpickling molecules",
        chunksize=1000,
    )
    print(f"  Loaded {len(mols)} molecules from {filepath}")
    return mols


def _parse_smiles(smi: str, sanitize: bool) -> Chem.Mol | None:
    return Chem.MolFromSmiles(smi, sanitize=sanitize)


def _iter_smiles_tokens(filepath: str, sanitize: bool) -> Iterator[str]:
    """Yield SMILES tokens from a file, dropping a parse-failing first line as a header."""
    with open(filepath, "r") as fh:
        first_data_seen = False
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            smi = stripped.split()[0]
            if not first_data_seen:
                first_data_seen = True
                RDLogger.DisableLog("rdApp.*")
                mol = Chem.MolFromSmiles(smi, sanitize=sanitize)
                RDLogger.EnableLog("rdApp.*")
                if mol is None:
                    continue
            yield smi


def load_smiles(
    filepath: str,
    max_count: int = 0,
    sanitize: bool = True,
    seed: int | None = None,
) -> list[Chem.Mol]:
    """Load and parse molecules from a SMILES file.

    Reservoir-samples a uniform subset when ``max_count > 0`` so the input file
    does not have to fit in memory and only the sampled SMILES are parsed. A
    10% buffer is read past ``max_count`` to absorb parse failures, after which
    the result is trimmed back to ``max_count``.

    The returned list is always shuffled (deterministic with ``seed``) so
    benches that consume a head slice get a representative cross-section
    rather than file-order bias (some upstream files are sorted by size).
    """
    read_limit = int(max_count * 1.1) if max_count > 0 else 0
    rng = random.Random(seed)

    if read_limit > 0:
        reservoir: list[str] = []
        for index, smi in enumerate(_iter_smiles_tokens(filepath, sanitize)):
            if index < read_limit:
                reservoir.append(smi)
            else:
                replace_index = rng.randint(0, index)
                if replace_index < read_limit:
                    reservoir[replace_index] = smi
        smiles_list = reservoir
    else:
        smiles_list = list(_iter_smiles_tokens(filepath, sanitize))

    mols: list[Chem.Mol] = []
    if smiles_list:
        parse_func = partial(_parse_smiles, sanitize=sanitize)
        parsed = process_map(parse_func, smiles_list, desc="Parsing molecules", chunksize=1000)
        parse_failures = 0
        for mol in parsed:
            if mol is None:
                parse_failures += 1
            else:
                mols.append(mol)
        if parse_failures > 0:
            print(f"    ({parse_failures} parse failures)")

    if max_count > 0 and len(mols) > max_count:
        mols = rng.sample(mols, max_count)
    else:
        rng.shuffle(mols)

    print(f"  Loaded {len(mols)} molecules from {filepath}")
    return mols


def load_smarts(filepath: str) -> tuple[list[Chem.Mol], list[str]]:
    """Load and parse every query pattern from a SMARTS file.

    Returns:
        ``(queries, smarts_strings)`` parallel lists.
    """
    queries: list[Chem.Mol] = []
    smarts_list: list[str] = []
    parse_failures = 0

    with open(filepath, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            smarts = stripped.split()[0]
            query = Chem.MolFromSmarts(smarts)
            if query is None:
                parse_failures += 1
                continue
            queries.append(query)
            smarts_list.append(smarts)

    print(f"  Loaded {len(queries)} SMARTS patterns from {filepath}")
    if parse_failures > 0:
        print(f"    ({parse_failures} parse failures)")
    return queries, smarts_list


def load_sdf(
    filepath: str,
    max_count: int = 0,
    seed: int | None = None,
    removeHs: bool = False,
    sanitize: bool = True,
) -> list[Chem.Mol]:
    """Load molecules from an SDF file with optional reservoir sampling.

    The returned list is always shuffled (deterministic with ``seed``) so
    benches that consume a head slice get a representative cross-section
    rather than file-order bias (some upstream files are sorted by size).
    """
    supplier = Chem.SDMolSupplier(filepath, removeHs=removeHs, sanitize=sanitize)
    read_limit = int(max_count * 1.1) if max_count > 0 else 0
    rng = random.Random(seed)

    parse_failures = 0
    if read_limit > 0:
        reservoir: list[Chem.Mol] = []
        index = 0
        for mol in supplier:
            if mol is None:
                parse_failures += 1
                continue
            if index < read_limit:
                reservoir.append(mol)
            else:
                replace_index = rng.randint(0, index)
                if replace_index < read_limit:
                    reservoir[replace_index] = mol
            index += 1
        mols = reservoir
    else:
        mols = []
        for mol in supplier:
            if mol is None:
                parse_failures += 1
                continue
            mols.append(mol)

    if max_count > 0 and len(mols) > max_count:
        mols = rng.sample(mols, max_count)
    else:
        rng.shuffle(mols)

    if parse_failures > 0:
        print(f"    ({parse_failures} parse failures)")
    print(f"  Loaded {len(mols)} molecules from {filepath}")
    return mols
