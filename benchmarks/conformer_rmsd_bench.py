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

"""Benchmark: GPU vs CPU conformer RMSD matrix computation.

Measures speedup of nvMolKit's GPU GetConformerRMSMatrix over RDKit's
CPU GetConformerRMSMatrix across varying conformer counts and molecule sizes.

Usage:
    python conformer_rmsd_bench.py
    python conformer_rmsd_bench.py --num-confs 50 100 200 500
    python conformer_rmsd_bench.py --smiles "CCCCCCCCCC" --num-confs 500
"""

import argparse
import copy

import numpy as np
import torch
from bench_utils import perturb_conformer
from benchmark_timing import time_it
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

from nvmolkit.conformerRmsd import GetConformerRMSMatrix


def _numpy_kabsch_rmsd(p, q):
    """Independent Kabsch RMSD using numpy SVD."""
    p_c = p - p.mean(axis=0)
    q_c = q - q.mean(axis=0)
    H = p_c.T @ q_c
    S = np.linalg.svd(H, compute_uv=False)
    d = np.sign(np.linalg.det(H))
    S[-1] *= d if d != 0.0 else 1.0
    Sp = np.sum(p_c**2)
    Sq = np.sum(q_c**2)
    return np.sqrt(max((Sp + Sq - 2.0 * np.sum(S)) / len(p), 0.0))


def benchmark_cpu(mol, n_warmup=1, n_iter=5):
    """Benchmark RDKit CPU GetConformerRMSMatrix.

    Deep-copies the molecule before each call because GetConformerRMSMatrix
    modifies conformer coordinates in-place during alignment; reusing the
    same molecule would measure already-aligned conformers and understate
    the true CPU cost.
    """
    result = time_it(
        lambda: AllChem.GetConformerRMSMatrix(copy.deepcopy(mol), prealigned=False), runs=n_iter, warmups=n_warmup
    )
    return result.median_s


def benchmark_gpu(mol, n_warmup=2, n_iter=10):
    """Benchmark nvMolKit GPU GetConformerRMSMatrix."""
    result = time_it(
        lambda: GetConformerRMSMatrix(mol, prealigned=False), runs=n_iter, warmups=n_warmup, gpu_sync=True
    )
    return result.median_s


def run_benchmark(smiles, num_confs_list, seed=42):
    """Run CPU vs GPU benchmark for a molecule at various conformer counts."""
    mol_base = Chem.AddHs(Chem.MolFromSmiles(smiles))
    no_h_base = Chem.RemoveHs(Chem.AddHs(Chem.MolFromSmiles(smiles)))
    n_atoms = no_h_base.GetNumAtoms()

    print(f"\nMolecule: {smiles}  ({n_atoms} heavy atoms)")
    print(f"{'Confs':>8}  {'Pairs':>10}  {'CPU (ms)':>10}  {'GPU (ms)':>10}  {'Speedup':>8}  {'Match':>6}")
    print("-" * 70)

    for num_confs in num_confs_list:
        mol = Chem.RWMol(mol_base)
        mol.RemoveAllConformers()
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = seed
        params.useRandomCoords = True
        if rdDistGeom.EmbedMolecule(mol, params=params) < 0:
            print(f"{num_confs:>8}  {'skipped (embedding failed)':>50}")
            continue
        if num_confs < 2:
            print(f"{num_confs:>8}  {'skipped (need >= 2 confs for RMSD)':>50}")
            continue
        base_conf_id = mol.GetConformer().GetId()
        for conf_idx in range(1, num_confs):
            new_conf = Chem.Conformer(mol.GetConformer(base_conf_id))
            perturb_conformer(new_conf, seed=seed + conf_idx)
            mol.AddConformer(new_conf, assignId=True)
        perturb_conformer(mol.GetConformer(base_conf_id), seed=seed)
        actual_confs = mol.GetNumConformers()

        no_h = Chem.RemoveHs(mol)
        n_pairs = actual_confs * (actual_confs - 1) // 2

        # CPU benchmark
        cpu_time = benchmark_cpu(no_h)

        # GPU benchmark
        gpu_time = benchmark_gpu(no_h)

        # Correctness check against numpy Kabsch SVD (sample up to 500 pairs)
        gpu_result = GetConformerRMSMatrix(no_h, prealigned=False)
        torch.cuda.synchronize()
        gpu_rms = gpu_result.numpy().tolist()

        confs = no_h.GetConformers()
        coords = [np.array(c.GetPositions()) for c in confs]
        max_diff = 0.0
        count = 0
        max_checks = min(500, n_pairs)
        for i in range(len(confs)):
            for j in range(i):
                idx = i * (i - 1) // 2 + j
                ref = _numpy_kabsch_rmsd(coords[i], coords[j])
                max_diff = max(max_diff, abs(gpu_rms[idx] - ref))
                count += 1
                if count >= max_checks:
                    break
            if count >= max_checks:
                break
        match_ok = max_diff < 0.05

        speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")

        print(
            f"{actual_confs:>8}  {n_pairs:>10}  {cpu_time * 1000:>10.2f}  "
            f"{gpu_time * 1000:>10.2f}  {speedup:>7.1f}x  "
            f"{'OK' if match_ok else f'FAIL ({max_diff:.4f})':>6}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU vs CPU conformer RMSD matrix")
    parser.add_argument(
        "--smiles",
        nargs="+",
        default=[
            "CC(=O)Oc1ccccc1C(=O)O",  # aspirin (13 HA)
            "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",  # celecoxib (24 HA)
            "C=CC(=O)Nc1cc(OC)c(Nc2nccc(-c3cn(C)c4ccccc34)n2)cc1N(C)CCN(C)C",  # osimertinib (33 HA)
            "CC(C)CC1NC(=O)C(CC(=O)O)NC(=O)C(Cc2ccc(O)cc2)NC(=O)C(CO)NC(=O)C(Cc2c[nH]c3ccccc23)NC1=O",  # cyclic pentapeptide (~48 HA)
        ],
        help="SMILES strings to benchmark",
    )
    parser.add_argument(
        "--num-confs",
        nargs="+",
        type=int,
        default=[10, 50, 100, 200, 500],
        help="Number of conformers to generate",
    )
    args = parser.parse_args()

    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}")
    print(f"CUDA: {torch.version.cuda}")

    for smiles in args.smiles:
        run_benchmark(smiles, args.num_confs)

    print("\nDone.")


if __name__ == "__main__":
    main()
