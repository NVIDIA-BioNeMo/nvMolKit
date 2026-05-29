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

import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch
from bench_utils import load_smiles
from benchmark_timing import time_it
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.ML.Cluster.Butina import ClusterData

from nvmolkit.clustering import butina as butina_nvmol, fused_butina
from nvmolkit.fingerprints import MorganFingerprintGenerator as nvmolMorganGen
from nvmolkit.similarity import crossTanimotoSimilarity


def check_butina_correctness(hit_mat, clusts):
    hit_mat = hit_mat.clone()
    seen = set()

    for i, clust in enumerate(clusts):
        assert len(clust) > 0, "Empty cluster found"
        clust_size = len(clust)

        if clust_size == 1:
            remaining_items = []
            for remaining_clust in clusts[i:]:
                assert len(remaining_clust) == 1, "Expected all remaining clusters to be singletons"
                remaining_items.append(remaining_clust[0])

            remaining_set = set(remaining_items)
            assert len(remaining_set) == len(remaining_items), "Duplicate items in singleton clusters"
            assert remaining_set.isdisjoint(seen), "Singleton item was already seen"
            seen.update(remaining_set)
            break
        counts = hit_mat.sum(-1)
        assert clust_size == counts.max(), (
            f"Cluster size {clust_size} doesn't match max available count {counts.max()}"
        )
        for item in clust:
            assert item not in seen, f"Point {item} assigned to multiple clusters"
            seen.add(item)
            hit_mat[item, :] = False
            hit_mat[:, item] = False
    assert len(seen) == hit_mat.shape[0]


def get_fingerprints(molecules):
    nvmol_gen = nvmolMorganGen(radius=2, fpSize=1024)
    nvmol_fps = nvmol_gen.GetFingerprints(molecules, 10)
    return nvmol_fps.torch()


def resize_and_fill(distance_mat: torch.Tensor, want_size):
    current_size = distance_mat.shape[0]
    if current_size >= want_size:
        return distance_mat[:want_size, :want_size].contiguous()
    full_mat = torch.rand(want_size, want_size, dtype=distance_mat.dtype, device=distance_mat.device)
    full_mat = torch.abs(full_mat - full_mat.T).clip(0.01, 0.99)
    full_mat.fill_diagonal_(0.0)
    full_mat[:current_size, :current_size] = distance_mat
    return full_mat


def resize_and_fill_fingerprints(fps: torch.Tensor, want_size: int) -> torch.Tensor:
    current_size = fps.shape[0]
    if current_size >= want_size:
        return fps[:want_size].contiguous()
    full_fps = torch.randint(
        -(2**31),
        2**31 - 1,
        (want_size, fps.shape[1]),
        dtype=torch.int32,
        device=fps.device,
    )
    full_fps[:current_size] = fps
    return full_fps


def bench_rdkit(data, threshold, runs=3):
    result = time_it(lambda: ClusterData(data, len(data), threshold, isDistData=True, reordering=True), runs=runs)
    return result.mean_ms, result.std_ms


def bench_rdkit_with_tanimoto(rdkit_fps, threshold, runs=3):
    def _run():
        n = len(rdkit_fps)
        dist = np.empty((n, n), dtype=np.float64)
        for i in range(n):
            dist[i] = BulkTanimotoSimilarity(rdkit_fps[i], rdkit_fps)
        np.subtract(1.0, dist, out=dist)
        ClusterData(dist, n, threshold, isDistData=True, reordering=True)

    result = time_it(_run, runs=runs)
    return result.mean_ms, result.std_ms


def _tanimoto_dist(fp1, fp2):
    return 1.0 - DataStructs.TanimotoSimilarity(fp1, fp2)


def bench_rdkit_lowmem(rdkit_fps, threshold, runs=3):
    result = time_it(
        lambda: ClusterData(
            rdkit_fps,
            len(rdkit_fps),
            threshold,
            isDistData=False,
            distFunc=_tanimoto_dist,
            reordering=True,
        ),
        runs=runs,
    )
    return result.mean_ms, result.std_ms


def bench_nvmol_inner(data, threshold, neighborlist_max_size):
    butina_nvmol(data, threshold, neighborlist_max_size=neighborlist_max_size)


def bench_nvmol_with_tanimoto(fps, threshold, neighborlist_max_size):
    sim = crossTanimotoSimilarity(fps).torch()
    butina_nvmol(1.0 - sim, threshold, neighborlist_max_size=neighborlist_max_size)


VALID_BENCHMARKS = {"rdkit", "rdkit_lowmem", "fused", "nvmolkit"}
DEFAULT_SIZES = [1000, 5000, 10000, 20000, 30000, 40000]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Butina clustering")
    parser.add_argument("input_smiles_file", help="Path to input SMILES file")
    parser.add_argument("--no-rdkit", action="store_true", help="Disable RDKit benchmarks")
    parser.add_argument("--no-fused", action="store_true", help="Disable fused Butina benchmarks")
    parser.add_argument("--no-nvmolkit", action="store_true", help="Disable nvMolKit Butina benchmarks")
    parser.add_argument("--no-rdkit-lowmem", action="store_true", help="Disable RDKit low-memory benchmarks")
    parser.add_argument(
        "--include-tanimoto-matrix",
        action="store_true",
        help="Include tanimoto matrix calculation in butina timing (for fair comparison with fused_butina)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file specifying per-size benchmark selection",
    )
    parser.add_argument("--cutoff", type=float, default=None, help="Run only this cutoff value")
    parser.add_argument("--runs", type=int, default=3, help="Number of timed repetitions (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling SMILES (default: 42)")
    parser.add_argument(
        "-o", "--output", type=str, default="results.csv", help="Output CSV file path (default: results.csv)"
    )
    args = parser.parse_args()

    include_tanimoto = args.include_tanimoto_matrix
    n_runs = args.runs
    disabled = set()
    if args.no_rdkit:
        disabled.add("rdkit")
    if args.no_fused:
        disabled.add("fused")
    if args.no_nvmolkit:
        disabled.add("nvmolkit")
    if args.no_rdkit_lowmem:
        disabled.add("rdkit_lowmem")

    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        run_plan = []
        for entry in config:
            entry_runs = set(entry["run"])
            unknown = entry_runs - VALID_BENCHMARKS
            if unknown:
                print(f"Error: unknown benchmark types in config: {unknown}", file=sys.stderr)
                sys.exit(1)
            runs = entry_runs - disabled
            if runs:
                plan_entry = {"size": entry["size"], "run": runs}
                if "neighborlist_sizes" in entry:
                    plan_entry["neighborlist_sizes"] = entry["neighborlist_sizes"]
                run_plan.append(plan_entry)
    else:
        default_runs = VALID_BENCHMARKS - disabled
        run_plan = [{"size": s, "run": set(default_runs)} for s in DEFAULT_SIZES]

    run_plan = [e for e in run_plan if e["run"]]
    if not run_plan:
        print("Error: no benchmarks to run", file=sys.stderr)
        sys.exit(1)

    max_size = max(e["size"] for e in run_plan)

    mols = load_smiles(args.input_smiles_file, max_count=max_size + 100, sanitize=True, seed=args.seed)

    if include_tanimoto and len(mols) < max_size:
        print(
            f"Error: --include-tanimoto-matrix requires at least {max_size} molecules, "
            f"but only {len(mols)} were loaded from input",
            file=sys.stderr,
        )
        sys.exit(1)

    fps = get_fingerprints(mols)

    max_rdkit_fps_size = max(
        (e["size"] for e in run_plan if "rdkit_lowmem" in e["run"] or ("rdkit" in e["run"] and include_tanimoto)),
        default=0,
    )
    if max_rdkit_fps_size > 0:
        if len(mols) < max_rdkit_fps_size:
            print(
                f"Error: rdkit benchmarks with fingerprints require at least {max_rdkit_fps_size} "
                f"molecules, but only {len(mols)} were loaded from input",
                file=sys.stderr,
            )
            sys.exit(1)
        rdkit_fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
        rdkit_fps = [rdkit_fpgen.GetFingerprint(mol) for mol in mols]
    else:
        rdkit_fps = None

    output_path = args.output
    cutoffs = [args.cutoff] if args.cutoff is not None else [1e-10, 0.1, 0.2, 0.35, 1.1]
    default_nl_sizes = [8, 16, 32, 64, 128]
    results = []

    def save_results():
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        return df

    try:
        for entry in run_plan:
            size = entry["size"]
            runs = entry["run"]
            max_nl_sizes = entry.get("neighborlist_sizes", default_nl_sizes)

            need_dist = "nvmolkit" in runs or ("rdkit" in runs and not include_tanimoto)
            if include_tanimoto:
                fps_mat = fps[:size].contiguous()
                if need_dist:
                    dist_mat = 1.0 - crossTanimotoSimilarity(fps_mat).torch()
            else:
                fps_mat = resize_and_fill_fingerprints(fps, size)
                if need_dist:
                    real_size = min(size, len(mols))
                    base_dists = 1.0 - crossTanimotoSimilarity(fps[:real_size]).torch()
                    dist_mat = resize_and_fill(base_dists, size)
                    del base_dists

            for cutoff in cutoffs:
                # Don't run large sizes for edge cases.
                if cutoff in (1e-10, 1.1) and size > 20000:
                    continue

                rdkit_time, rdk_std = float("nan"), float("nan")
                if "rdkit" in runs:
                    if include_tanimoto:
                        rdkit_time, rdk_std = bench_rdkit_with_tanimoto(rdkit_fps[:size], cutoff, runs=n_runs)
                    else:
                        dist_mat_numpy = dist_mat.cpu().numpy()
                        rdkit_time, rdk_std = bench_rdkit(dist_mat_numpy, cutoff, runs=n_runs)

                rdkit_lm_time, rdkit_lm_std = float("nan"), float("nan")
                if "rdkit_lowmem" in runs:
                    print(f"Running rdkit_lowmem size {size} cutoff {cutoff}")
                    rdkit_lm_time, rdkit_lm_std = bench_rdkit_lowmem(rdkit_fps[:size], cutoff, runs=n_runs)

                fused_time, fused_std = float("nan"), float("nan")
                if "fused" in runs:
                    print(f"Running fused_butina size {size} cutoff {cutoff}")
                    fused_result = time_it(
                        lambda: fused_butina(fps_mat, cutoff=cutoff, metric="tanimoto"),
                        gpu_sync=True,
                        runs=n_runs,
                    )
                    fused_time, fused_std = fused_result.mean_ms, fused_result.std_ms

                if "nvmolkit" in runs:
                    for max_nl in max_nl_sizes:
                        print(f"Running nvmolkit size {size} cutoff {cutoff} max_nl {max_nl}")
                        if include_tanimoto:
                            nvmol_result = time_it(
                                lambda: bench_nvmol_with_tanimoto(fps_mat, cutoff, max_nl),
                                gpu_sync=True,
                                runs=n_runs,
                            )
                        else:
                            nvmol_result = time_it(
                                lambda: bench_nvmol_inner(dist_mat, cutoff, max_nl),
                                gpu_sync=True,
                                runs=n_runs,
                            )
                        nvmol_time, nvmol_std = nvmol_result.mean_ms, nvmol_result.std_ms

                        nvmol_res = butina_nvmol(dist_mat, cutoff, neighborlist_max_size=max_nl).torch()
                        torch.cuda.synchronize()
                        nvmol_clusts = [
                            tuple(torch.argwhere(nvmol_res == i).flatten().tolist())
                            for i in range(nvmol_res.max() + 1)
                        ]
                        check_butina_correctness(dist_mat <= cutoff, nvmol_clusts)

                        results.append(
                            {
                                "size": size,
                                "cutoff": cutoff,
                                "max_neighborlist_size": max_nl,
                                "rdkit_time_ms": rdkit_time,
                                "rdkit_std_ms": rdk_std,
                                "rdkit_lowmem_time_ms": rdkit_lm_time,
                                "rdkit_lowmem_std_ms": rdkit_lm_std,
                                "nvmol_time_ms": nvmol_time,
                                "nvmol_std_ms": nvmol_std,
                                "fused_butina_time_ms": fused_time,
                                "fused_butina_std_ms": fused_std,
                            }
                        )
                else:
                    results.append(
                        {
                            "size": size,
                            "cutoff": cutoff,
                            "max_neighborlist_size": float("nan"),
                            "rdkit_time_ms": rdkit_time,
                            "rdkit_std_ms": rdk_std,
                            "rdkit_lowmem_time_ms": rdkit_lm_time,
                            "rdkit_lowmem_std_ms": rdkit_lm_std,
                            "nvmol_time_ms": float("nan"),
                            "nvmol_std_ms": float("nan"),
                            "fused_butina_time_ms": fused_time,
                            "fused_butina_std_ms": fused_std,
                        }
                    )
            save_results()
    except Exception as e:
        print(f"Got exception: {e}, exiting early")
    df = save_results()
    print(df)
