# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Leader selection against RDKit.

RDKit selects from Morgan bit vectors with ``LeaderPicker``.

nvMolKit starts from nvMolKit Morgan fingerprints on the GPU, builds the
Tanimoto distance matrix, and selects from it; ``nvmolkit_matrix_select``
times the selection alone on a prebuilt matrix.
Fingerprint generation is outside every timed region.

nvMolKit compares distances in single precision, so a cutoff that is not a
dyadic fraction can classify a boundary distance differently than RDKit. The
``*_matches_rdkit`` columns report whether nvMolKit returned the RDKit result.

Example:
    python diversity_pickers_bench.py --smiles data/chembl_10k.smi \
        --sizes 1000 10000 --cutoffs 0.25 0.5
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

from nvmolkit.clustering import OutputMode
from nvmolkit.fingerprints import MorganFingerprintGenerator
from nvmolkit.pickers import leader
from nvmolkit.similarity import crossTanimotoSimilarity


@dataclass
class Inputs:
    """Inputs for one problem size."""

    rdkit_fps: list
    fingerprints: torch.Tensor
    matrix: torch.Tensor | None


def _distance_matrix(fingerprints: torch.Tensor) -> torch.Tensor:
    return 1.0 - crossTanimotoSimilarity(fingerprints).torch()


def _rdkit_leader(fps: list, cutoff: float):
    return tuple(rdSimDivPickers.LeaderPicker().LazyBitVectorPick(fps, len(fps), cutoff))


@dataclass(frozen=True)
class Operation:
    """RDKit and nvMolKit implementations of one selection algorithm."""

    rdkit: Callable
    matrix: Callable


OPERATIONS = {
    "leader": Operation(
        rdkit=_rdkit_leader,
        matrix=lambda matrix, cutoff: leader(matrix, cutoff, output=OutputMode.RDKIT),
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
    cutoff: float,
    inputs: Inputs,
    runs: int,
    warmups: int,
    validate: bool,
    no_rdkit: bool,
    no_nvmolkit: bool,
) -> dict[str, float | int | str | bool]:
    """Time every requested implementation at one sweep point."""
    row: dict[str, float | int | str | bool] = {
        "operation": name,
        "num_mols": len(inputs.rdkit_fps),
        "cutoff": cutoff,
    }
    rdkit_timing = rdkit_result = None
    if not no_rdkit:
        rdkit_timing, rdkit_result = _time(
            f"rdkit_{name}", lambda: operation.rdkit(inputs.rdkit_fps, cutoff), runs, warmups, gpu=False
        )
        row["num_selected"] = len(rdkit_result)
        row.update(_timing_fields("rdkit", rdkit_timing))

    def compare(form: str, timing, result) -> None:
        row["num_selected"] = len(result)
        row.update(_timing_fields(f"nvmolkit_{form}", timing))
        if rdkit_timing is not None:
            row[f"{form}_speedup"] = rdkit_timing.median_ms / timing.median_ms
            if validate:
                row[f"{form}_matches_rdkit"] = result == rdkit_result

    if inputs.matrix is not None:
        compare(
            "matrix",
            *_time(
                f"nvmolkit_matrix_{name}",
                lambda: operation.matrix(_distance_matrix(inputs.fingerprints), cutoff),
                runs,
                warmups,
                gpu=True,
            ),
        )
        select_timing, _ = _time(
            f"nvmolkit_matrix_select_{name}",
            lambda: operation.matrix(inputs.matrix, cutoff),
            runs,
            warmups,
            gpu=True,
        )
        row.update(_timing_fields("nvmolkit_matrix_select", select_timing))

    if row.get("matrix_matches_rdkit") is False:
        print(f"WARNING: nvMolKit {name} differs from RDKit", file=sys.stderr, flush=True)
    return row


def run(
    smiles_path: str,
    sizes: list[int],
    operations: list[str],
    cutoffs: list[float],
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

    max_size = max(sizes)
    mols = load_smiles(smiles_path, max_count=max_size + 100, sanitize=True, seed=seed)[:max_size]
    if len(mols) < max_size:
        raise ValueError(f"requested {max_size} molecules, but only {len(mols)} valid molecules were loaded")
    workers = prep_workers if prep_workers > 0 else available_cpu_count()
    all_rdkit_fps = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size).GetFingerprints(
        mols, numThreads=workers
    )
    all_fingerprints = (
        MorganFingerprintGenerator(radius=radius, fpSize=fp_size).GetFingerprints(mols, num_threads=workers).torch()
    )
    if validate:
        _validate_fingerprints(all_rdkit_fps, all_fingerprints)

    rows: list[dict[str, float | int | str | bool]] = []
    try:
        for size in sorted(set(sizes)):
            inputs = None  # Release the previous size's matrix before building the next.
            use_matrix = not no_nvmolkit and size <= matrix_max_size
            fingerprints = all_fingerprints[:size].contiguous()
            matrix = _distance_matrix(fingerprints) if use_matrix else None
            inputs = Inputs(
                rdkit_fps=list(all_rdkit_fps[:size]),
                fingerprints=fingerprints,
                matrix=matrix,
            )
            for name in operations:
                for cutoff in cutoffs:
                    print(f"\n=== {name}, {size} molecules, cutoff={cutoff} ===", flush=True)
                    rows.append(
                        _benchmark_point(
                            name, OPERATIONS[name], cutoff, inputs, runs, warmups, validate, no_rdkit, no_nvmolkit
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
    parser = argparse.ArgumentParser(description="Leader selection benchmark")
    parser.add_argument("--smiles", required=True, help="Path to a SMILES file")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000])
    parser.add_argument("--operations", choices=tuple(OPERATIONS), nargs="+", default=list(OPERATIONS))
    parser.add_argument("--cutoffs", type=float, nargs="+", default=[0.25, 0.5], help="Leader distance cutoffs")
    parser.add_argument("--matrix_max_size", type=int, default=10000, help="Largest size benchmarked by nvMolKit")
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
        cutoffs=args.cutoffs,
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
