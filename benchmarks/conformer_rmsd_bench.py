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

"""Benchmark: GPU vs single-threaded CPU pairwise conformer RMSD.

Measures speedup of nvMolKit's GPU GetConformerRMSMatrix over RDKit's
CPU GetConformerRMSMatrix across varying conformer counts.
"""

import argparse
import csv
import multiprocessing as mp
import statistics
import time
from pathlib import Path

import torch
from bench_utils import Deadline, add_rdkit_max_seconds_arg, embed_and_jitter, load_smiles
from benchmark_timing import time_it
from rdkit import Chem
from rdkit.Chem import AllChem

from nvmolkit.conformerRmsd import GetConformerRMSMatrixBatch


def prepare_mols(
    raw_mols: list[Chem.Mol],
    confs_per_mol: int,
    seed: int,
    num_workers: int,
) -> list[Chem.Mol]:
    """Embed one base conformer per mol, then perturb to ``confs_per_mol``."""
    workers = num_workers if num_workers > 0 else max(1, mp.cpu_count() // 2)
    return embed_and_jitter(
        raw_mols,
        confs_per_mol=confs_per_mol,
        seed=seed,
        num_workers=workers,
        add_hs=True,
        min_atoms=2,
        desc=f"Embed + perturb ({confs_per_mol} confs)",
    )


def bench_rdkit_batch(payloads: list[bytes], max_seconds: float) -> tuple[float, int]:
    """One RDKit timing iteration over ``payloads``; returns ``(elapsed_s, n_done)``.

    Stops once ``max_seconds`` is exceeded (``0`` means no cap). A fresh mol is
    built per call because ``GetConformerRMSMatrix`` aligns conformers in place.
    """
    mols = [Chem.Mol(mol_bytes) for mol_bytes in payloads]
    deadline = Deadline(max_seconds)
    start = time.perf_counter()
    n_done = 0
    for mol in mols:
        AllChem.GetConformerRMSMatrix(mol, prealigned=False)
        n_done += 1
        if deadline.expired():
            break
    return time.perf_counter() - start, n_done


def bench_gpu_batch(mols: list[Chem.Mol]) -> None:
    results = GetConformerRMSMatrixBatch(mols, prealigned=False)
    torch.cuda.synchronize()


def validate(mols: list[Chem.Mol], num_check: int, tol: float) -> None:
    """Compare GPU RMSD matrices against RDKit on the first ``num_check`` mols.

    Raises ``RuntimeError`` on the first pair whose absolute diff exceeds ``tol``.
    """
    subset = mols[:num_check]
    if not subset:
        return
    print(f"\nValidation: comparing GPU vs RDKit on {len(subset)} mols (tol={tol})")
    gpu_results = GetConformerRMSMatrixBatch(subset, prealigned=False)
    torch.cuda.synchronize()
    max_abs_diff = 0.0
    for mol_idx, mol in enumerate(subset):
        rdkit_mol = Chem.Mol(mol.ToBinary())
        rdkit_rms = AllChem.GetConformerRMSMatrix(rdkit_mol, prealigned=False)
        gpu_rms = gpu_results[mol_idx].numpy().tolist()
        if len(gpu_rms) != len(rdkit_rms):
            raise RuntimeError(
                f"validation: mol {mol_idx} pair count mismatch (gpu={len(gpu_rms)}, rdkit={len(rdkit_rms)})"
            )
        for pair_idx, (gpu_val, rdkit_val) in enumerate(zip(gpu_rms, rdkit_rms)):
            diff = abs(float(gpu_val) - float(rdkit_val))
            if diff > tol:
                raise RuntimeError(
                    f"validation: mol {mol_idx} pair {pair_idx} diff {diff:.4f} > {tol} "
                    f"(gpu={gpu_val:.4f}, rdkit={rdkit_val:.4f})"
                )
            if diff > max_abs_diff:
                max_abs_diff = diff
    print(f"  OK (max abs diff {max_abs_diff:.5f})")


def _slice_to_confs(mols: list[Chem.Mol], target: int) -> list[Chem.Mol]:
    """Return copies of ``mols`` keeping only the first ``target`` conformers each."""
    out: list[Chem.Mol] = []
    for mol in mols:
        copy_mol = Chem.Mol(mol, True)  # quickCopy: keeps graph, drops conformers
        confs = list(mol.GetConformers())[:target]
        for conf in confs:
            copy_mol.AddConformer(Chem.Conformer(conf), assignId=True)
        out.append(copy_mol)
    return out


def run(
    smiles_path: str,
    num_mols: int,
    confs_per_mol_list: list[int],
    seed: int,
    prep_workers: int,
    rdkit_max_seconds: float,
    validate_count: int,
    validate_tol: float,
    no_rdkit: bool,
    no_nvmolkit: bool,
    output: str | None,
) -> None:
    if no_rdkit and no_nvmolkit:
        raise ValueError("cannot disable both RDKit and nvMolKit")
    if any(count < 2 for count in confs_per_mol_list):
        raise ValueError("every --confs_per_mol value must be >= 2")

    if not no_nvmolkit:
        print(f"GPU: {torch.cuda.get_device_name(0)}  CUDA: {torch.version.cuda}")
    print(f"Loading SMILES from {smiles_path} (target {num_mols} mols)")
    raw = load_smiles(smiles_path, max_count=num_mols, sanitize=True, seed=seed)

    max_confs = max(confs_per_mol_list)
    print(f"Preparing {len(raw)} mols x {max_confs} conformers (perturb-from-1-embed)")
    base_mols = prepare_mols(raw, confs_per_mol=max_confs, seed=seed, num_workers=prep_workers)
    if len(base_mols) > num_mols:
        base_mols = base_mols[:num_mols]
    if not base_mols:
        raise RuntimeError("no molecules survived embedding")

    avg_atoms = sum(mol.GetNumAtoms() for mol in base_mols) / len(base_mols)
    print(f"  {len(base_mols)} mols, ~{avg_atoms:.1f} heavy atoms/mol")
    if validate_count > 0 and not no_rdkit and not no_nvmolkit:
        validate(_slice_to_confs(base_mols, max_confs), validate_count, validate_tol)
    elif validate_count > 0:
        print("\nValidation skipped (requires both --rdkit and --nvmolkit enabled)")

    print(f"\nSweeping confs_per_mol: {confs_per_mol_list}")

    rows: list[dict[str, float | int | str]] = []
    for target_confs in sorted(confs_per_mol_list):
        mols = _slice_to_confs(base_mols, target_confs)
        actual_confs = [mol.GetNumConformers() for mol in mols]
        total_pairs = sum(count * (count - 1) // 2 for count in actual_confs)
        print(f"\n=== confs_per_mol={target_confs}: {len(mols)} mols, {total_pairs} RMSD pairs ===")

        row: dict[str, float | int | str] = {
            "num_mols": len(mols),
            "confs_per_mol": target_confs,
            "total_pairs": total_pairs,
            "avg_heavy_atoms": avg_atoms,
        }

        rdkit_mols_per_s: float | None = None
        rdkit_pairs_per_s: float | None = None
        if not no_rdkit:
            payloads = [mol.ToBinary() for mol in mols]
            cap_label = f"cap={rdkit_max_seconds:.0f}s" if rdkit_max_seconds > 0 else "no cap"
            print(f"  RDKit CPU (single-threaded, {cap_label}):")
            # TODO: replace this hand-rolled warmup/sample/median loop with time_it once
            # time_it can consume a Deadline and truncate a run mid-workload.
            # https://github.com/NVIDIA-BioNeMo/nvMolKit/issues/186
            bench_rdkit_batch(payloads, rdkit_max_seconds)  # warmup
            samples = [bench_rdkit_batch(payloads, rdkit_max_seconds) for _ in range(3)]
            samples.sort(key=lambda pair: pair[0] / max(pair[1], 1))
            rdkit_time_s, rdkit_done = samples[len(samples) // 2]
            per_mol_times = [elapsed / max(done, 1) for elapsed, done in samples]
            rdkit_std_s = statistics.stdev(per_mol_times) * rdkit_done if len(samples) > 1 else 0.0
            pair_count_done = sum(count * (count - 1) // 2 for count in actual_confs[:rdkit_done])
            rdkit_mols_per_s = rdkit_done / rdkit_time_s
            rdkit_pairs_per_s = pair_count_done / rdkit_time_s
            truncated = rdkit_done < len(mols)
            suffix = f" [truncated at {rdkit_done}/{len(mols)} mols]" if truncated else ""
            print(
                f"    median wall: {rdkit_time_s * 1000:.1f} +/- {rdkit_std_s * 1000:.1f} ms over {rdkit_done} mols  "
                f"({rdkit_mols_per_s:.1f} mols/s, {rdkit_pairs_per_s:.0f} pairs/s){suffix}"
            )
            row["rdkit_median_s"] = rdkit_time_s
            row["rdkit_std_s"] = rdkit_std_s
            row["rdkit_mols_processed"] = rdkit_done
            row["rdkit_truncated"] = int(truncated)
            row["rdkit_mols_per_s"] = rdkit_mols_per_s
            row["rdkit_pairs_per_s"] = rdkit_pairs_per_s

        gpu_pairs_per_s: float | None = None
        if not no_nvmolkit:
            print("  nvMolKit GPU (batched):")
            result = time_it(lambda: bench_gpu_batch(mols), runs=5, warmups=2, gpu_sync=True)
            gpu_time_s = result.median_s
            gpu_std_s = result.std_ms / 1000.0
            gpu_pairs_per_s = total_pairs / gpu_time_s
            print(
                f"    median wall: {gpu_time_s * 1000:.1f} +/- {gpu_std_s * 1000:.1f} ms  "
                f"({len(mols) / gpu_time_s:.1f} mols/s, {gpu_pairs_per_s:.0f} pairs/s)"
            )
            row["gpu_median_s"] = gpu_time_s
            row["gpu_std_s"] = gpu_std_s
            row["gpu_mols_per_s"] = len(mols) / gpu_time_s
            row["gpu_pairs_per_s"] = gpu_pairs_per_s

        if rdkit_pairs_per_s is not None and gpu_pairs_per_s is not None:
            row["speedup"] = gpu_pairs_per_s / rdkit_pairs_per_s
            print(f"  GPU speedup vs single-threaded RDKit (pairs/s): {row['speedup']:.1f}x")

        rows.append(row)

    if output and rows:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with out_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Conformer RMSD batch benchmark")
    parser.add_argument("--smiles", required=True, help="Path to smiles file")
    parser.add_argument("--num_mols", type=int, default=2000, help="Number of molecules to sample")
    parser.add_argument(
        "--confs_per_mol",
        type=int,
        nargs="+",
        default=[10, 25, 50, 100, 200],
        help="Conformers-per-molecule sweep points (each >=2)",
    )
    parser.add_argument(
        "--prep_workers", type=int, default=0, help="Workers for the embed-and-perturb prep step (0 = half of CPUs)"
    )
    add_rdkit_max_seconds_arg(
        parser,
        extra_help="The cap applies per timing iteration and truncates at a molecule boundary.",
    )
    parser.add_argument(
        "--validate_count",
        type=int,
        default=8,
        help="Number of mols to compare GPU vs RDKit before timing (0 disables; requires both backends enabled)",
    )
    parser.add_argument(
        "--validate_tol", type=float, default=0.05, help="Absolute tolerance (Angstroms) for per-pair RMSD diff"
    )
    parser.add_argument("--no_validate", action="store_true", help="Skip the GPU-vs-RDKit correctness check")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="Optional CSV output path")
    parser.add_argument("--no_rdkit", action="store_true", help="Skip RDKit CPU benchmark")
    parser.add_argument("--no_nvmolkit", action="store_true", help="Skip nvMolKit GPU benchmark")
    args = parser.parse_args()

    if args.no_rdkit and args.no_nvmolkit:
        parser.error("cannot pass both --no_rdkit and --no_nvmolkit")

    run(
        smiles_path=args.smiles,
        num_mols=args.num_mols,
        confs_per_mol_list=args.confs_per_mol,
        seed=args.seed,
        prep_workers=args.prep_workers,
        rdkit_max_seconds=args.rdkit_max_seconds,
        validate_count=0 if args.no_validate else args.validate_count,
        validate_tol=args.validate_tol,
        no_rdkit=args.no_rdkit,
        no_nvmolkit=args.no_nvmolkit,
        output=args.output,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
