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

"""Benchmark for TFD (Torsion Fingerprint Deviation) calculation.

Compares:
- RDKit TorsionFingerprints.GetTFDMatrix (Python, single-threaded)
- nvMolKit GPU (CUDA) with different return types (list, numpy, tensor)

Usage:
    python tfd_bench.py [--smiles-file FILE] [--output FILE] [--no-rdkit]
    python tfd_bench.py --pkl-file data1.pkl data2.pkl [--output FILE]

Example:
    python tfd_bench.py --smiles-file data/benchmark_smiles.csv --output tfd_results.csv

    # Use precomputed ChEMBL stratified pickles directly:
    python tfd_bench.py --pkl-file ../Data/Chembl_stratified_prepared/chembl_0-20_10confs.pkl \
        --num-mols 100 500 1000 5000 --output tfd_chembl_results.csv
"""

import argparse
import os
import pickle
import sys
from typing import List

import pandas as pd
import torch
from bench_utils import (
    add_backend_selection_args,
    available_cpu_count,
    embed_and_jitter,
    load_smiles,
    print_csv_rows,
    slice_conformers,
    time_it,
    write_csv_rows,
)
from rdkit import Chem
from rdkit.Chem import TorsionFingerprints

import nvmolkit.tfd as nvmol_tfd

def generate_conformers_batch(
    mols: List[Chem.Mol],
    num_confs: int,
    seed: int = 42,
    num_workers: int = 0,
) -> List[Chem.Mol]:
    """Generate ``num_confs`` conformers per mol via embed-once-then-perturb.

    Wraps the shared :func:`bench_utils.embed_and_jitter` with TFD-specific
    constraints: requires ``num_confs >= 2`` (at least one torsion pair) and
    drops mols with fewer than 4 atoms; hydrogens are added during embedding
    and stripped from the returned mols.
    """
    if num_confs < 2:
        raise ValueError(f"num_confs must be >= 2 for TFD, got {num_confs}")
    workers = num_workers if num_workers > 0 else max(1, available_cpu_count() // 2)
    return embed_and_jitter(
        mols,
        confs_per_mol=num_confs,
        seed=seed,
        num_workers=workers,
        add_hs=True,
        min_atoms=4,
        desc=f"Embedding base conformer (1/{num_confs})",
    )


def prepare_molecules(
    input_mols: List[Chem.Mol],
    num_confs: int,
    max_mols: int = 100,
    num_workers: int = 0,
    seed: int = 42,
) -> List[Chem.Mol]:
    """Prepare molecules with conformers from the explicitly supplied inputs.

    Args:
        input_mols: Parsed RDKit molecules (used when no precomputed pickle is found).
        num_confs: Number of conformers per molecule
        max_mols: Maximum number of molecules to prepare
        num_workers: Parallel workers for ETKDG embedding (0 = auto, half of CPUs)
        seed: Seed for base embedding and conformer perturbations.

    Returns:
        List of molecules with conformers
    """
    candidates: List[Chem.Mol] = []
    for mol in input_mols:
        if mol is None or mol.GetNumAtoms() < 4:
            continue
        candidates.append(mol)
        if len(candidates) >= max_mols:
            break

    return generate_conformers_batch(candidates, num_confs, seed=seed, num_workers=num_workers)


def bench_rdkit_single(mol: Chem.Mol) -> None:
    """Benchmark RDKit TFD for a single molecule."""
    TorsionFingerprints.GetTFDMatrix(mol, useWeights=True, maxDev="equal")


def bench_rdkit_batch(mols: List[Chem.Mol]) -> None:
    """Benchmark RDKit TFD for multiple molecules (sequential)."""
    for mol in mols:
        TorsionFingerprints.GetTFDMatrix(mol, useWeights=True, maxDev="equal")


def bench_nvmol_gpu_single(mol: Chem.Mol) -> None:
    """Benchmark nvMolKit GPU TFD for a single molecule."""
    nvmol_tfd.GetTFDMatrix(mol, useWeights=True, maxDev="equal")
    torch.cuda.synchronize()


def bench_nvmol_gpu_list(mols: List[Chem.Mol]) -> None:
    """Benchmark nvMolKit GPU TFD returning Python lists."""
    nvmol_tfd.GetTFDMatrices(mols, useWeights=True, maxDev="equal", return_type="list")
    torch.cuda.synchronize()


def bench_nvmol_gpu_numpy(mols: List[Chem.Mol]) -> None:
    """Benchmark nvMolKit GPU TFD returning numpy arrays."""
    nvmol_tfd.GetTFDMatrices(mols, useWeights=True, maxDev="equal", return_type="numpy")
    torch.cuda.synchronize()


def bench_nvmol_gpu_tensor(mols: List[Chem.Mol]) -> None:
    """Benchmark nvMolKit GPU TFD returning GPU tensors (no D2H)."""
    nvmol_tfd.GetTFDMatrices(mols, useWeights=True, maxDev="equal", return_type="tensor")
    torch.cuda.synchronize()


def verify_correctness(mol: Chem.Mol, tolerance: float = 0.01) -> bool:
    """Verify nvMolKit results match RDKit (within tolerance).

    Multi-quartet torsions (rings and symmetric) are fully supported,
    so results should match RDKit closely.
    """
    rdkit_result = TorsionFingerprints.GetTFDMatrix(mol, useWeights=True, maxDev="equal")
    nvmol_result = nvmol_tfd.GetTFDMatrix(mol, useWeights=True, maxDev="equal")

    if len(rdkit_result) != len(nvmol_result):
        return False

    for rd, nv in zip(rdkit_result, nvmol_result):
        if abs(rd - nv) > tolerance:
            return False

    return True


def load_pkl_files(pkl_paths: List[str]) -> List[Chem.Mol]:
    """Load and concatenate molecules from one or more pickle files.

    Each pickle file must contain a list of RDKit Mol objects with conformers
    (as produced by prepare_chembl_stratified.py).
    """
    all_mols = []
    for path in pkl_paths:
        with open(path, "rb") as f:
            mols = pickle.load(f)
        print(f"  Loaded {len(mols)} molecules from {path}")
        all_mols.extend(mols)
    return all_mols


def run_benchmarks(
    input_mols: List[Chem.Mol] | None = None,
    skip_rdkit: bool = False,
    skip_nvmolkit: bool = False,
    output_file: str = "tfd_results.csv",
    mol_counts: List[int] | None = None,
    conformer_counts: List[int] | None = None,
    preloaded_mols: List[Chem.Mol] | None = None,
    num_workers: int = 0,
    runs: int = 3,
    warmups: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Run TFD benchmarks with various configurations.

    Args:
        input_mols: Parsed RDKit molecules without conformers (unused when preloaded_mols given).
        skip_rdkit: If True, skip RDKit benchmarks (faster for large runs)
        skip_nvmolkit: If True, skip nvMolKit GPU benchmarks (RDKit-only mode)
        output_file: Output CSV file path
        mol_counts: List of molecule counts to benchmark
        conformer_counts: List of conformer counts to benchmark
        preloaded_mols: Pre-loaded molecules with conformers (e.g. from --pkl-file).
            When provided, input_mols/smiles_file/conformer_counts are ignored and
            the actual conformer count is read from the molecules.
        num_workers: Parallel workers for ETKDG embedding (0 = auto, half of CPUs).
        runs: Number of timed repetitions per workload point.
        warmups: Number of warmup repetitions per workload point.
        seed: Sampling and conformer-preparation seed.

    Returns:
        DataFrame with benchmark results
    """
    if skip_rdkit and skip_nvmolkit:
        raise ValueError("cannot disable both RDKit and nvMolKit")

    if mol_counts is None:
        mol_counts = [1, 5, 10, 25, 50, 100]

    if preloaded_mols is not None:
        actual_confs_all = [m.GetNumConformers() for m in preloaded_mols]
        median_confs = sorted(actual_confs_all)[len(actual_confs_all) // 2]
        conformer_counts = [median_confs]
        print(f"  Using {len(preloaded_mols)} preloaded molecules (~{median_confs} conformers each)")
    elif conformer_counts is None:
        conformer_counts = [5, 10, 20]

    results: list[dict[str, object]] = []

    print("=" * 70)
    print("TFD Benchmark: RDKit vs nvMolKit (GPU)")
    print(f"Molecule counts: {mol_counts}")
    print(f"Conformer counts: {conformer_counts}")
    print("=" * 70)

    prepared_max: List[Chem.Mol] | None = None
    if preloaded_mols is None:
        max_confs = max(conformer_counts)
        print(f"\n--- Preparing molecules once with {max_confs} conformers ---")
        prepared_max = prepare_molecules(
            input_mols,
            max_confs,
            max_mols=max(mol_counts) + 20,
            num_workers=num_workers,
            seed=seed,
        )

    for num_confs in conformer_counts:
        if preloaded_mols is not None:
            all_mols = preloaded_mols[: max(mol_counts)]
        else:
            assert prepared_max is not None
            all_mols = slice_conformers(prepared_max, num_confs)

        if len(all_mols) < max(mol_counts):
            print(f"Warning: Only {len(all_mols)} molecules available")

        for num_mols in mol_counts:
            if num_mols > len(all_mols):
                print(f"Skipping {num_mols} mols (only {len(all_mols)} available)")
                continue

            mols = all_mols[:num_mols]
            actual_confs = [mol.GetNumConformers() for mol in mols]
            avg_confs = sum(actual_confs) / len(actual_confs)

            print(f"\nBenchmarking: {num_mols} molecules, ~{avg_confs:.1f} conformers each")

            # Calculate expected TFD pairs
            total_pairs = sum(c * (c - 1) // 2 for c in actual_confs)
            print(f"  Total TFD pairs: {total_pairs}")

            result = {
                "num_molecules": num_mols,
                "target_conformers": num_confs,
                "avg_conformers": avg_confs,
                "total_tfd_pairs": total_pairs,
            }

            # RDKit benchmark (single-threaded Python)
            if not skip_rdkit:
                timing = time_it(lambda: bench_rdkit_batch(mols), runs=runs, warmups=warmups)
                rdkit_time, rdkit_std = timing.mean_ms, timing.std_ms
                result["rdkit_time_ms"] = rdkit_time
                result["rdkit_std_ms"] = rdkit_std
                print(f"  RDKit (Python):     {rdkit_time:8.2f} ms (+/- {rdkit_std:.2f})")
            else:
                result["rdkit_time_ms"] = None
                result["rdkit_std_ms"] = None

            if not skip_nvmolkit:
                timing = time_it(lambda: bench_nvmol_gpu_list(mols), runs=runs, warmups=warmups)
                t, s = timing.mean_ms, timing.std_ms
                result["nvmol_gpu_list_time_ms"] = t
                result["nvmol_gpu_list_std_ms"] = s
                print(f"  nvMolKit (GPU list):  {t:8.2f} ms (+/- {s:.2f})")

                timing = time_it(lambda: bench_nvmol_gpu_numpy(mols), runs=runs, warmups=warmups)
                t, s = timing.mean_ms, timing.std_ms
                result["nvmol_gpu_numpy_time_ms"] = t
                result["nvmol_gpu_numpy_std_ms"] = s
                print(f"  nvMolKit (GPU numpy): {t:8.2f} ms (+/- {s:.2f})")

                timing = time_it(lambda: bench_nvmol_gpu_tensor(mols), runs=runs, warmups=warmups)
                t, s = timing.mean_ms, timing.std_ms
                result["nvmol_gpu_tensor_time_ms"] = t
                result["nvmol_gpu_tensor_std_ms"] = s
                print(f"  nvMolKit (GPU tensor): {t:8.2f} ms (+/- {s:.2f})")

                speedups = {}
                for key, label in [
                    ("nvmol_gpu_list_time_ms", "GPU list"),
                    ("nvmol_gpu_numpy_time_ms", "GPU numpy"),
                    ("nvmol_gpu_tensor_time_ms", "GPU tensor"),
                ]:
                    if result.get("rdkit_time_ms") and result.get(key):
                        speedups[label] = result["rdkit_time_ms"] / result[key]

                for label, val in speedups.items():
                    print(f"  Speedup {label:>10s} vs RDKit: {val:.1f}x")
            else:
                result["nvmol_gpu_list_time_ms"] = None
                result["nvmol_gpu_numpy_time_ms"] = None
                result["nvmol_gpu_tensor_time_ms"] = None

            results.append(result)

    if not results:
        raise RuntimeError("no TFD workload points could be run with the available molecules")

    # Create DataFrame and save
    df = pd.DataFrame(results)
    write_csv_rows(results, output_file)
    print("\nCSV Results:")
    print_csv_rows(results)
    print(f"\n{'=' * 70}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 70}")

    return df


def main():
    _default_smiles = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "benchmark_smiles.csv")

    parser = argparse.ArgumentParser(description="TFD Benchmark")
    parser.add_argument(
        "--smiles-file",
        type=str,
        default=_default_smiles,
        help="CSV file with SMILES (default: benchmarks/data/benchmark_smiles.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tfd_results.csv",
        help="Output CSV file for results",
    )
    parser.add_argument(
        "--num-mols",
        type=int,
        nargs="+",
        default=[1, 10, 50, 100, 500, 1000],
        help="Molecule counts to benchmark (default: 1 10 50 100 500 1000)",
    )
    parser.add_argument(
        "--num-confs",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50],
        help="Conformer counts to benchmark (default: 5 10 20 50)",
    )
    add_backend_selection_args(
        parser,
        rdkit_dest="skip_rdkit",
        nvmolkit_dest="skip_nvmolkit",
        rdkit_aliases=("--skip-rdkit",),
        nvmolkit_aliases=("--skip-nvmolkit",),
    )
    parser.add_argument(
        "--pkl-file",
        type=str,
        nargs="+",
        default=None,
        help="Precomputed pickle file(s) containing molecules with conformers. "
        "When provided, --smiles-file and --num-confs are ignored.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify correctness before benchmarking",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify correctness and exit (skip benchmarking)",
    )
    parser.add_argument(
        "--prep-workers",
        type=int,
        default=0,
        help="Parallel workers for ETKDG embedding during prep (0 = auto, half of CPUs)",
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of timed repetitions (default: 3)")
    parser.add_argument("--warmups", type=int, default=1, help="Number of warmup repetitions (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling and conformer-generation seed (default: 42)")
    args = parser.parse_args()

    if args.skip_rdkit and args.skip_nvmolkit:
        parser.error("cannot pass both --skip-rdkit and --skip-nvmolkit")

    preloaded_mols = None
    input_mols = None

    if args.pkl_file:
        print("Loading precomputed molecules from pickle file(s)...")
        preloaded_mols = load_pkl_files(args.pkl_file)
        if not preloaded_mols:
            print("Error: no molecules loaded from pickle files")
            sys.exit(1)
        print(f"Total: {len(preloaded_mols)} molecules")
    else:
        print(f"Loading SMILES from: {args.smiles_file}")
        try:
            input_mols = load_smiles(args.smiles_file, max_count=max(args.num_mols) + 100, seed=args.seed)
        except Exception as e:
            print(f"Error loading SMILES file: {e}")
            sys.exit(1)
        print(f"Loaded {len(input_mols)} molecules")

    if args.verify or args.verify_only:
        print("\nVerifying correctness...")
        if preloaded_mols is not None:
            test_mols = preloaded_mols[:50]
        else:
            test_mols = prepare_molecules(
                input_mols,
                num_confs=5,
                max_mols=50,
                num_workers=args.prep_workers,
                seed=args.seed,
            )
        if not test_mols:
            print("Error: no molecules available for TFD verification", file=sys.stderr)
            sys.exit(1)
        all_correct = True
        mismatches = 0
        for i, mol in enumerate(test_mols):
            if verify_correctness(mol):
                print(f"  Molecule {i}: OK")
            else:
                print(f"  Molecule {i}: MISMATCH")
                all_correct = False
                mismatches += 1
        if all_correct:
            print(f"All {len(test_mols)} molecules match RDKit.")
        else:
            print(
                f"Error: {mismatches}/{len(test_mols)} molecules did not match RDKit within tolerance",
                file=sys.stderr,
            )
            sys.exit(1)

        if args.verify_only:
            sys.exit(0)

    run_benchmarks(
        input_mols=input_mols,
        skip_rdkit=args.skip_rdkit,
        skip_nvmolkit=args.skip_nvmolkit,
        output_file=args.output,
        mol_counts=args.num_mols,
        conformer_counts=args.num_confs if not args.pkl_file else None,
        preloaded_mols=preloaded_mols,
        num_workers=args.prep_workers,
        runs=args.runs,
        warmups=args.warmups,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
