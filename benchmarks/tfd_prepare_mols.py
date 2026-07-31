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

"""Precompute molecules with conformers for TFD benchmarking and profiling.

Generates conformers once and saves the results as pickle files (one per
conformer count). This avoids the expensive ETKDG embedding during
benchmark/profile runs.

Output files: <output-dir>/prepared_mols_<N>confs.pkl

Usage:
    python benchmarks/tfd_prepare_mols.py

    # Custom settings:
    python benchmarks/tfd_prepare_mols.py \
        --smiles-file benchmarks/data/benchmark_smiles.csv \
        --output-dir benchmarks/data \
        --conformers 5 10 20 50 \
        --max-mols 500 \
        --workers 8
"""

import argparse
import multiprocessing
import os
import pickle
import sys
import time
from functools import partial

import pandas as pd
from bench_utils import available_cpu_count
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


def _generate_one(args_tuple, num_confs):
    """Generate conformers for a single molecule (picklable for multiprocessing)."""
    idx, smi = args_tuple
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumAtoms() < 4:
        return None

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42 + idx
    params.numThreads = 1
    params.useRandomCoords = True
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    if len(conf_ids) < 2:
        return None
    return Chem.RemoveHs(mol)


def main():
    default_smiles = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "benchmark_smiles.csv")
    default_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    parser = argparse.ArgumentParser(
        description="Precompute molecules with conformers for TFD benchmarking",
    )
    parser.add_argument(
        "--smiles-file",
        type=str,
        default=default_smiles,
        help="CSV file with SMILES in first column (default: benchmarks/data/benchmark_smiles.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output_dir,
        help="Output directory for pickle files (default: benchmarks/data)",
    )
    parser.add_argument(
        "--conformers",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="Conformer counts to generate (default: 5 10 20)",
    )
    parser.add_argument(
        "--max-mols",
        type=int,
        default=0,
        help="Maximum number of molecules to prepare (default: 0 = all valid SMILES)",
    )
    default_workers = max(1, available_cpu_count() // 2)
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Number of parallel workers (default: {default_workers})",
    )
    args = parser.parse_args()

    print(f"Loading SMILES from: {args.smiles_file}")
    df = pd.read_csv(args.smiles_file)
    smiles_list = df.iloc[:, 0].tolist()
    print(f"Loaded {len(smiles_list)} SMILES")

    # Build (index, smiles) pairs, filtering small molecules
    work_items = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None and mol.GetNumAtoms() >= 4:
            work_items.append((i, smi))
        if args.max_mols > 0 and len(work_items) >= args.max_mols:
            break
    print(f"Valid molecules (>=4 atoms): {len(work_items)}")
    print(f"Workers: {args.workers}")

    os.makedirs(args.output_dir, exist_ok=True)

    for num_confs in args.conformers:
        out_path = os.path.join(args.output_dir, f"prepared_mols_{num_confs}confs.pkl")

        if os.path.exists(out_path):
            print(f"\n--- {num_confs} conformers: {out_path} already exists, skipping ---")
            continue

        print(f"\n--- Generating {num_confs} conformers ({len(work_items)} molecules, {args.workers} workers) ---")
        start = time.perf_counter()

        worker_fn = partial(_generate_one, num_confs=num_confs)

        if args.workers > 1:
            with multiprocessing.Pool(args.workers) as pool:
                results = []
                chunksize = max(1, len(work_items) // (args.workers * 4))
                for i, result in enumerate(pool.imap(worker_fn, work_items, chunksize=chunksize)):
                    results.append(result)
                    if (i + 1) % 50 == 0 or (i + 1) == len(work_items):
                        print(f"  Progress: {i + 1}/{len(work_items)} molecules", flush=True)
        else:
            results = []
            for i, item in enumerate(work_items):
                results.append(worker_fn(item))
                if (i + 1) % 50 == 0 or (i + 1) == len(work_items):
                    print(f"  Progress: {i + 1}/{len(work_items)} molecules", flush=True)

        mols = [m for m in results if m is not None]
        elapsed = time.perf_counter() - start

        actual_confs = [m.GetNumConformers() for m in mols]
        total_pairs = sum(c * (c - 1) // 2 for c in actual_confs)

        with open(out_path, "wb") as f:
            pickle.dump(mols, f, protocol=pickle.HIGHEST_PROTOCOL)

        file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  {len(mols)} molecules, avg {sum(actual_confs) / len(actual_confs):.1f} conformers")
        print(f"  Total TFD pairs: {total_pairs}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Saved: {out_path} ({file_size_mb:.1f} MB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
