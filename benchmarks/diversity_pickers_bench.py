# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Leader, DISE, and MaxMin selection against RDKit.

RDKit selects from Morgan bit vectors with ``LeaderPicker`` and ``MaxMinPicker``.
The DISE reference is RDKit's ``LeaderPicker`` followed by nearest-centroid
assignment with ``BulkTanimotoSimilarity``.

nvMolKit is timed in two forms, both starting from nvMolKit Morgan
fingerprints on the GPU. The fused form selects directly from fingerprints.
The matrix form builds the Tanimoto distance matrix and selects from it;
``nvmolkit_matrix_select`` times the selection alone on a prebuilt matrix.
For MaxMin, ``rdkit_matrix_select`` times ``MaxMinPicker.Pick`` on a prebuilt
condensed matrix. Fingerprint generation is outside every timed region.

nvMolKit compares distances in single precision, so a cutoff that is not a
dyadic fraction can classify a boundary distance differently than RDKit. The
``*_matches_rdkit`` columns report whether nvMolKit returned the RDKit result.

Example:
    python diversity_pickers_bench.py --smiles data/chembl_10k.smi \
        --sizes 1000 10000 --cutoffs 0.25 0.5 --pick_sizes 100
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import nvtx
import torch
from bench_utils import (
    add_backend_selection_args,
    available_cpu_count,
    load_smiles,
    print_csv_rows,
    time_it,
    write_csv_rows,
)
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters import rdSimDivPickers

from nvmolkit.clustering import OutputMode, dise, fused_dise
from nvmolkit.fingerprints import MorganFingerprintGenerator
from nvmolkit.pickers import fused_leader, fused_maxmin, leader, maxmin
from nvmolkit.similarity import crossTanimotoSimilarity

FORMS = ("fused", "matrix")


@dataclass
class Inputs:
    """Inputs for one problem size; a disabled backend's fingerprints are None."""

    num_mols: int
    rdkit_fps: list | None
    fingerprints: torch.Tensor | None
    matrix: torch.Tensor | None
    condensed: np.ndarray | None
    seed: int


def _distance_matrix(fingerprints: torch.Tensor) -> torch.Tensor:
    return 1.0 - crossTanimotoSimilarity(fingerprints).torch()


def _condensed(matrix: torch.Tensor) -> np.ndarray:
    """Lower triangle in the row order of RDKit's condensed distance matrices."""
    dense = matrix.cpu().numpy().astype(np.float64)
    return dense[np.tril_indices(len(dense), -1)]


def _clusters(centroids, assignments) -> tuple[tuple[int, ...], ...]:
    groups = [[] for _ in centroids]
    for item, cluster_id in enumerate(assignments):
        groups[int(cluster_id)].append(item)
    return tuple(tuple(sorted(group)) for group in groups)


def _canonical_clusters(clusters) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(sorted(cluster)) for cluster in clusters))


def _rdkit_leader(inputs: Inputs, cutoff: float):
    fps = inputs.rdkit_fps
    return tuple(rdSimDivPickers.LeaderPicker().LazyBitVectorPick(fps, len(fps), cutoff))


def _rdkit_dise(inputs: Inputs, cutoff: float):
    centroids = _rdkit_leader(inputs, cutoff)
    centroid_fps = [inputs.rdkit_fps[index] for index in centroids]
    assignments = [
        int(np.argmin(DataStructs.BulkTanimotoSimilarity(fp, centroid_fps, returnDistance=True)))
        for fp in inputs.rdkit_fps
    ]
    return _clusters(centroids, assignments)


def _rdkit_maxmin(inputs: Inputs, pick_size: int):
    fps = inputs.rdkit_fps
    return tuple(rdSimDivPickers.MaxMinPicker().LazyBitVectorPick(fps, len(fps), pick_size, seed=inputs.seed))


def _rdkit_maxmin_matrix(inputs: Inputs, pick_size: int):
    size = len(inputs.rdkit_fps)
    return tuple(rdSimDivPickers.MaxMinPicker().Pick(inputs.condensed, size, pick_size, seed=inputs.seed))


@dataclass(frozen=True)
class Operation:
    """RDKit and nvMolKit implementations of one selection algorithm."""

    parameter: str
    rdkit: Callable
    fused: Callable
    matrix: Callable
    rdkit_matrix: Callable | None = None
    canonical: Callable = tuple


OPERATIONS = {
    "leader": Operation(
        parameter="cutoff",
        rdkit=_rdkit_leader,
        fused=lambda fps, cutoff, seed: fused_leader(fps, cutoff, output=OutputMode.RDKIT),
        matrix=lambda matrix, cutoff, seed: leader(matrix, cutoff, output=OutputMode.RDKIT),
    ),
    "dise": Operation(
        parameter="cutoff",
        rdkit=_rdkit_dise,
        fused=lambda fps, cutoff, seed: fused_dise(fps, cutoff, output=OutputMode.RDKIT),
        matrix=lambda matrix, cutoff, seed: dise(matrix, cutoff, output=OutputMode.RDKIT),
        canonical=_canonical_clusters,
    ),
    "maxmin": Operation(
        parameter="pick_size",
        rdkit=_rdkit_maxmin,
        fused=lambda fps, pick_size, seed: fused_maxmin(fps, pick_size, seed=seed, output=OutputMode.RDKIT)[0],
        matrix=lambda matrix, pick_size, seed: maxmin(matrix, pick_size, seed=seed, output=OutputMode.RDKIT)[0],
        rdkit_matrix=_rdkit_maxmin_matrix,
    ),
}


def _time(label: str, function: Callable, runs: int, warmups: int, gpu: bool):
    """Time ``function`` and return its timing and the result of the last call."""
    last = []

    def call():
        with nvtx.annotate(label, color="green" if gpu else "blue"):
            last[:] = [function()]

    timing = time_it(call, runs=runs, warmups=warmups, gpu_sync=gpu)
    return timing, last[0]


def _timing_fields(prefix: str, timing) -> dict[str, float]:
    return {f"{prefix}_median_ms": timing.median_ms, f"{prefix}_std_ms": timing.std_ms}


def _validate_fingerprints(rdkit_fps: list, fingerprints: torch.Tensor) -> None:
    """Check that RDKit and nvMolKit select from the same fingerprints."""
    # BitVectToBinaryText uses bit 0 as the low bit of byte 0, as nvMolKit does.
    packed = np.stack([np.frombuffer(DataStructs.BitVectToBinaryText(fp), dtype=np.int32) for fp in rdkit_fps])
    if not np.array_equal(packed, fingerprints.view(torch.int32).cpu().numpy()):
        raise AssertionError("RDKit and nvMolKit Morgan fingerprints differ")


def _benchmark_point(
    name: str,
    operation: Operation,
    value: float | int,
    inputs: Inputs,
    forms: list[str],
    runs: int,
    warmups: int,
    validate: bool,
    no_rdkit: bool,
    no_nvmolkit: bool,
) -> dict[str, float | int | str | bool]:
    """Time every requested implementation at one sweep point."""
    row: dict[str, float | int | str | bool] = {
        "operation": name,
        "num_mols": inputs.num_mols,
        operation.parameter: value,
    }
    rdkit_timing = rdkit_result = None
    if not no_rdkit:
        rdkit_timing, rdkit_result = _time(
            f"rdkit_{name}", lambda: operation.rdkit(inputs, value), runs, warmups, gpu=False
        )
        row["rdkit_num_selected"] = len(rdkit_result)
        row.update(_timing_fields("rdkit", rdkit_timing))

    def compare(form: str, timing, result) -> None:
        row[f"nvmolkit_{form}_num_selected"] = len(result)
        row.update(_timing_fields(f"nvmolkit_{form}", timing))
        if rdkit_timing is not None:
            row[f"{form}_speedup"] = rdkit_timing.median_ms / timing.median_ms
            if validate:
                row[f"{form}_matches_rdkit"] = operation.canonical(result) == operation.canonical(rdkit_result)

    if not no_nvmolkit and "fused" in forms:
        compare(
            "fused",
            *_time(
                f"nvmolkit_fused_{name}",
                lambda: operation.fused(inputs.fingerprints, value, inputs.seed),
                runs,
                warmups,
                gpu=True,
            ),
        )

    if inputs.matrix is not None:
        compare(
            "matrix",
            *_time(
                f"nvmolkit_matrix_{name}",
                lambda: operation.matrix(_distance_matrix(inputs.fingerprints), value, inputs.seed),
                runs,
                warmups,
                gpu=True,
            ),
        )
        select_timing, _ = _time(
            f"nvmolkit_matrix_select_{name}",
            lambda: operation.matrix(inputs.matrix, value, inputs.seed),
            runs,
            warmups,
            gpu=True,
        )
        row.update(_timing_fields("nvmolkit_matrix_select", select_timing))
        if inputs.condensed is not None and operation.rdkit_matrix is not None:
            rdkit_select_timing, _ = _time(
                f"rdkit_matrix_select_{name}", lambda: operation.rdkit_matrix(inputs, value), runs, warmups, gpu=False
            )
            row.update(_timing_fields("rdkit_matrix_select", rdkit_select_timing))
            row["matrix_select_speedup"] = rdkit_select_timing.median_ms / select_timing.median_ms

    if row.get("fused_matches_rdkit") is False or row.get("matrix_matches_rdkit") is False:
        print(f"WARNING: nvMolKit {name} differs from RDKit", file=sys.stderr, flush=True)
    return row


def run(
    smiles_path: str,
    sizes: list[int],
    operations: list[str],
    forms: list[str],
    cutoffs: list[float],
    pick_sizes: list[int],
    matrix_max_size: int,
    radius: int,
    fp_size: int,
    runs: int,
    warmups: int,
    seed: int,
    prep_workers: int,
    validate: bool,
    no_rdkit: bool,
    no_nvmolkit: bool,
    output: str | None,
) -> list[dict[str, float | int | str | bool]]:
    """Prepare fingerprints for the largest size and benchmark each sweep point."""
    if no_rdkit and no_nvmolkit:
        raise ValueError("cannot disable both RDKit and nvMolKit")
    if any(size < 1 for size in sizes):
        raise ValueError("every --sizes value must be positive")
    if "maxmin" in operations and any(pick_size < 1 or pick_size >= min(sizes) for pick_size in pick_sizes):
        raise ValueError("every --pick_sizes value must be positive and smaller than every size")

    max_size = max(sizes)
    mols = load_smiles(smiles_path, max_count=max_size + 100, sanitize=True, seed=seed)[:max_size]
    if len(mols) < max_size:
        raise ValueError(f"requested {max_size} molecules, but only {len(mols)} valid molecules were loaded")
    workers = prep_workers if prep_workers > 0 else available_cpu_count()
    all_rdkit_fps = None
    if not no_rdkit:
        all_rdkit_fps = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size).GetFingerprints(
            mols, numThreads=workers
        )
    all_fingerprints = None
    if not no_nvmolkit:
        all_fingerprints = (
            MorganFingerprintGenerator(radius=radius, fpSize=fp_size)
            .GetFingerprints(mols, num_threads=workers)
            .torch()
        )
    if validate and all_rdkit_fps is not None and all_fingerprints is not None:
        _validate_fingerprints(all_rdkit_fps, all_fingerprints)

    rows: list[dict[str, float | int | str | bool]] = []
    try:
        for size in sorted(set(sizes)):
            inputs = None  # Release the previous size's matrix before building the next.
            use_matrix = "matrix" in forms and not no_nvmolkit and size <= matrix_max_size
            fingerprints = None if all_fingerprints is None else all_fingerprints[:size].contiguous()
            matrix = _distance_matrix(fingerprints) if use_matrix else None
            inputs = Inputs(
                num_mols=size,
                rdkit_fps=None if all_rdkit_fps is None else list(all_rdkit_fps[:size]),
                fingerprints=fingerprints,
                matrix=matrix,
                condensed=_condensed(matrix) if use_matrix and not no_rdkit and "maxmin" in operations else None,
                seed=seed,
            )
            for name in operations:
                operation = OPERATIONS[name]
                for value in cutoffs if operation.parameter == "cutoff" else pick_sizes:
                    print(f"\n=== {name}, {size} molecules, {operation.parameter}={value} ===", flush=True)
                    rows.append(
                        _benchmark_point(
                            name, operation, value, inputs, forms, runs, warmups, validate, no_rdkit, no_nvmolkit
                        )
                    )
    finally:
        if output:
            write_csv_rows(rows, Path(output))

    print("\nCSV Results:")
    print_csv_rows(rows)
    if output:
        print(f"\nWrote {output}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Leader, DISE, and MaxMin selection benchmark")
    parser.add_argument("--smiles", required=True, help="Path to a SMILES file")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000])
    parser.add_argument("--operations", choices=tuple(OPERATIONS), nargs="+", default=list(OPERATIONS))
    parser.add_argument("--forms", choices=FORMS, nargs="+", default=list(FORMS))
    parser.add_argument(
        "--cutoffs", type=float, nargs="+", default=[0.25, 0.5], help="Leader and DISE distance cutoffs"
    )
    parser.add_argument("--pick_sizes", type=int, nargs="+", default=[100], help="MaxMin selection sizes")
    parser.add_argument(
        "--matrix_max_size", type=int, default=10000, help="Largest size benchmarked in the matrix form"
    )
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--fp_size", type=int, default=1024)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prep_workers", type=int, default=0, help="Fingerprint workers (0 = all available CPUs)")
    parser.add_argument("--no_validate", action="store_true")
    parser.add_argument("--output", default=None, help="Optional CSV output path")
    add_backend_selection_args(parser)
    args = parser.parse_args()
    run(
        smiles_path=args.smiles,
        sizes=args.sizes,
        operations=args.operations,
        forms=args.forms,
        cutoffs=args.cutoffs,
        pick_sizes=args.pick_sizes,
        matrix_max_size=args.matrix_max_size,
        radius=args.radius,
        fp_size=args.fp_size,
        runs=args.runs,
        warmups=args.warmups,
        seed=args.seed,
        prep_workers=args.prep_workers,
        validate=not args.no_validate,
        no_rdkit=args.no_rdkit,
        no_nvmolkit=args.no_nvmolkit,
        output=args.output,
    )


if __name__ == "__main__":
    main()
