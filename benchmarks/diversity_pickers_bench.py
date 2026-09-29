# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark matrix Leader selection against a one-CPU reference.

Fingerprint generation and matrix construction are outside the timed region.
"""

import argparse
import csv
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch
from bench_utils import load_smiles
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from nvmolkit.clustering import OutputMode
from nvmolkit.pickers import leader


def _distance_matrix(fps):
    matrix = np.zeros((len(fps), len(fps)), dtype=np.float32)
    for row in range(1, len(fps)):
        distances = DataStructs.BulkTanimotoSimilarity(fps[row], fps[:row], returnDistance=True)
        matrix[row, :row] = distances
        matrix[:row, row] = distances
    return matrix


def _cpu_leader(matrix, cutoff):
    available = np.ones(len(matrix), dtype=np.bool_)
    picks = []
    for candidate in range(len(matrix)):
        if available[candidate]:
            picks.append(candidate)
            available[candidate] = False
            available[candidate + 1:] &= matrix[candidate, candidate + 1:] > cutoff
    return tuple(picks)


def _time(function, runs, gpu):
    function()  # One warmup.
    if gpu:
        torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        if gpu:
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = function()
        if gpu:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return result, float(np.median(times)), float(np.std(times))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1024, 4096, 16384])
    parser.add_argument("--matrix-max-size", type=int, default=4096)
    parser.add_argument("--operations", nargs="+", choices=("leader",), default=["leader"])
    parser.add_argument("--forms", nargs="+", choices=("matrix",), default=["matrix"])
    parser.add_argument("--pick-size", type=int, default=128)
    parser.add_argument("--cutoff", type=float, default=0.5)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--prep-workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (args.runs < 1 or args.pick_size < 1 or args.prep_workers < 1 or
            args.matrix_max_size < 1 or any(size < args.pick_size for size in args.sizes)):
        parser.error("runs, pick-size, prep-workers, and matrix-max-size must be positive; sizes >= pick-size")

    sizes = sorted(set(args.sizes))
    print(f"PROGRESS stage=load size={sizes[-1]}", flush=True)
    molecules = load_smiles(args.smiles, max_count=sizes[-1] + 100, sanitize=True, seed=args.seed)
    if len(molecules) < sizes[-1]:
        parser.error(f"only {len(molecules)} molecules available")
    print(f"PROGRESS stage=fingerprints workers={args.prep_workers}", flush=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fps = generator.GetFingerprints(molecules[:sizes[-1]], numThreads=args.prep_workers)
    print(f"PROGRESS stage=prepared peak_process_rss_kib={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}",
          flush=True)
    results = []
    for size in sizes:
        print(f"PROGRESS stage=size size={size}", flush=True)
        cpu_fps = fps[:size]
        matrix = _distance_matrix(cpu_fps) if "matrix" in args.forms and size <= args.matrix_max_size else None
        gpu_matrix = torch.from_numpy(matrix).to("cuda") if matrix is not None else None
        operations = []
        if matrix is not None and "leader" in args.operations:
            operations.extend([
                ("leader", "matrix", "numpy", False, lambda: _cpu_leader(matrix, args.cutoff)),
                ("leader", "matrix", "nvmolkit", True,
                 lambda: leader(gpu_matrix, args.cutoff, output=OutputMode.RDKIT)),
            ])
        outputs = {}
        for operation, form, method, gpu, function in operations:
            picks, median_ms, std_ms = _time(function, args.runs, gpu)
            outputs[(operation, form, method)] = picks
            results.append(dict(size=size, operation=operation, form=form, method=method,
                                device=torch.cuda.get_device_name() if gpu else "1 CPU",
                                median_ms=median_ms, std_ms=std_ms, num_picks=len(picks),
                                runs=args.runs, cutoff=args.cutoff, pick_size=args.pick_size,
                                seed=args.seed, input=args.smiles))
            print(f"PROGRESS size={size} operation={operation} form={form} method={method} "
                  f"median_ms={median_ms:.3f} num_picks={len(picks)}", flush=True)
        for operation in args.operations:
            cpu_key = (operation, "matrix", "numpy")
            if cpu_key in outputs and outputs[cpu_key] != outputs[(operation, "matrix", "nvmolkit")]:
                raise AssertionError(f"picker mismatch: size={size}, {operation}, matrix")
        if results:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
    print(f"PROGRESS rows_written={len(results)} output={args.output}", flush=True)
    print(f"PROGRESS peak_process_rss_kib={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}", flush=True)


if __name__ == "__main__":
    main()
