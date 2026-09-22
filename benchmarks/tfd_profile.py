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

"""TFD profiling script for use with NVIDIA Nsight Systems (nsys).

Designed to produce NVTX-annotated traces for analyzing GPU TFD performance.
C++ NVTX ranges (buildTFDSystem, transferToDevice, kernel launches, etc.) are
automatically captured by nsys alongside the Python-level annotations below.

Usage:
    # Profile with default sweep (multiple mol counts and conformer counts):
    nsys profile -t nvtx,osrt,cuda --gpu-metrics-devices=all --stats=true \
        -f true -o tfd_gpu_profile \
        python benchmarks/tfd_profile.py

    # Profile with specific configurations:
    python benchmarks/tfd_profile.py --num-mols 100 500 1000 --num-confs 20 50
"""

import argparse
import os
import pickle
import sys

import nvtx
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

import nvmolkit.tfd as nvmol_tfd


def generate_conformers(mol, num_confs, seed=42):
    """Generate conformers for a molecule using ETKDG."""
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1
    params.useRandomCoords = True
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    if len(conf_ids) < 2:
        return None
    return Chem.RemoveHs(mol)


def prepare_molecules(smiles_file, num_mols, num_confs):
    """Load molecules with conformers, using precomputed pickle if available."""
    data_dir = os.path.dirname(os.path.abspath(smiles_file))
    pkl_path = os.path.join(data_dir, f"prepared_mols_{num_confs}confs.pkl")

    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            all_mols = pickle.load(f)
        mols = all_mols[:num_mols]
        print(f"  Loaded {len(mols)} molecules from {pkl_path}")
        return mols

    print(f"  No precomputed file found ({pkl_path}), generating from scratch...")
    import pandas as pd

    df = pd.read_csv(smiles_file)
    smiles_list = df.iloc[:, 0].tolist()

    mols = []
    for i, smi in enumerate(smiles_list):
        if len(mols) >= num_mols:
            break
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() < 4:
            continue
        mol_with_confs = generate_conformers(mol, num_confs, seed=42 + i)
        if mol_with_confs is not None and mol_with_confs.GetNumConformers() >= 2:
            mols.append(mol_with_confs)

    return mols


def run_config(mols, num_runs, label):
    """Run profiled iterations for a single configuration."""
    actual_confs = [mol.GetNumConformers() for mol in mols]
    total_pairs = sum(c * (c - 1) // 2 for c in actual_confs)
    print(
        f"\n[{label}] {len(mols)} mols, avg {sum(actual_confs) / len(actual_confs):.1f} confs, {total_pairs} TFD pairs"
    )

    for run in range(num_runs):
        with nvtx.annotate(f"{label} GPU run {run}", color="orange"):
            nvmol_tfd.GetTFDMatrices(mols, useWeights=True, maxDev="equal", return_type="tensor")
            torch.cuda.synchronize()


def main():
    _default_smiles = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "benchmark_smiles.csv")

    parser = argparse.ArgumentParser(
        description="TFD profiling script for nsys",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Profile with default sweep:
  nsys profile -t nvtx,osrt,cuda --gpu-metrics-devices=all --stats=true \\
      -f true -o tfd_gpu python benchmarks/tfd_profile.py

  # Profile with large configurations:
  python benchmarks/tfd_profile.py --num-mols 500 1000 --num-confs 50 100
""",
    )
    parser.add_argument(
        "--num-mols",
        type=int,
        nargs="+",
        default=[50, 100, 500, 1000],
        help="Number of molecules to test (default: 50 100 500 1000)",
    )
    parser.add_argument(
        "--num-confs",
        type=int,
        nargs="+",
        default=[10, 20, 50],
        help="Number of conformers to test (default: 10 20 50)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup iterations (default: 1)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of profiled runs per configuration (default: 3)",
    )
    parser.add_argument(
        "--smiles-file",
        type=str,
        default=_default_smiles,
        help="CSV file with SMILES (default: benchmarks/data/benchmark_smiles.csv)",
    )
    args = parser.parse_args()

    print(f"Molecule counts: {args.num_mols}")
    print(f"Conformer counts: {args.num_confs}")
    print(f"Runs per config: {args.runs}")

    # === Setup: load all needed configurations ===
    configs = {}
    with nvtx.annotate("Setup", color="blue"):
        for num_confs in args.num_confs:
            max_mols = max(args.num_mols)
            print(f"\nLoading up to {max_mols} molecules with {num_confs} conformers...")
            all_mols = prepare_molecules(args.smiles_file, max_mols, num_confs)
            if not all_mols:
                print(f"  Warning: no molecules available for {num_confs} conformers, skipping")
                continue
            configs[num_confs] = all_mols

    if not configs:
        print("Error: No valid molecules prepared")
        sys.exit(1)

    # === Warmup ===
    if args.warmup > 0:
        with nvtx.annotate("Warmup", color="red"):
            first_mols = next(iter(configs.values()))
            warmup_mols = first_mols[: min(5, len(first_mols))]
            print(f"\nWarmup ({args.warmup} iteration(s)) with {len(warmup_mols)} molecules...")
            for _ in range(args.warmup):
                nvmol_tfd.GetTFDMatrices(warmup_mols, useWeights=True, maxDev="equal", return_type="tensor")
                torch.cuda.synchronize()

    # === Profiled runs ===
    with nvtx.annotate("Profiled runs", color="green"):
        for num_confs in sorted(configs.keys()):
            all_mols = configs[num_confs]
            for num_mols in args.num_mols:
                mols = all_mols[:num_mols]
                if len(mols) == 0:
                    continue
                label = f"{len(mols)}mols_{num_confs}confs"
                with nvtx.annotate(label, color="magenta"):
                    run_config(mols, args.runs, label)

    print("\nProfiling complete.")


if __name__ == "__main__":
    main()
