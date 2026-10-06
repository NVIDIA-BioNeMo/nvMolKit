# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark persistent nvMolKit and RDKit substructure libraries.

Each backend configuration stages and finalizes one library, timed once, then
times every requested operation against it. nvMolKit results are validated
against RDKit over the queries RDKit completed before its deadline.

Examples:
    python substruct_library_bench.py --smiles molecules.smi --smarts queries.smarts
    python substruct_library_bench.py --smiles molecules.smi --smarts queries.smarts \
        --operations has get --algorithms gsi dfs --rdkit_holders mol cached-pattern
    python substruct_library_bench.py --pickle targets.pkl --query_smiles queries.smi --num_queries 1000 \
        --query_modes serial concurrent --no-rdkit
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from bench_utils import (
    Deadline,
    TimingResult,
    add_backend_selection_args,
    add_rdkit_max_seconds_arg,
    load_pickle,
    load_smarts,
    load_smiles,
    print_csv_rows,
    throughput_per_s,
    time_it,
    write_csv_rows,
)
from rdkit import Chem
from rdkit.Chem import rdSubstructLibrary

OPERATIONS = ("has", "count", "get")


@dataclass(frozen=True)
class Measurement:
    """One-time build timings, steady-state query timing, and the last query sweep's results."""

    staging_ms: float
    finalize_ms: float
    steady_ms: float
    steady_std_ms: float
    results: list[Any]
    completed_queries: int


def run_nvmolkit_queries(
    library: Any, queries: Sequence[Any], operation: str, max_results: int, query_mode: str
) -> list[Any]:
    """Run every query, waiting on each in turn (serial) or submitting all before waiting (concurrent)."""
    if query_mode == "serial":
        run = {
            "has": library.hasMatchSync,
            "count": library.countMatchesSync,
            "get": lambda query: library.getMatchesSync(query, max_results),
        }[operation]
        return [run(query) for query in queries]
    submit = {
        "has": library.hasMatch,
        "count": library.countMatches,
        "get": lambda query: library.getMatches(query, max_results),
    }[operation]
    return [future.result() for future in [submit(query) for query in queries]]


def run_rdkit_queries(
    library: Any,
    queries: Sequence[Any],
    operation: str,
    max_results: int,
    num_threads: int,
    deadline: Deadline,
) -> list[Any]:
    """Run queries until the deadline expires; results cover a prefix of the query list."""
    options = {
        "recursionPossible": True,
        "useChirality": False,
        "useQueryQueryMatches": False,
        "numThreads": num_threads,
    }
    results = []
    for query in queries:
        if deadline.expired():
            break
        if operation == "has":
            results.append(library.HasMatch(query, **options))
        elif operation == "count":
            results.append(library.CountMatches(query, **options))
        else:
            results.append(list(library.GetMatches(query, maxResults=max_results, **options)))
    return results


def time_rdkit_operation(
    library: Any,
    queries: Sequence[Any],
    operation: str,
    *,
    max_results: int,
    num_threads: int,
    runs: int,
    max_seconds: float,
) -> tuple[TimingResult, list[Any]]:
    """Time one operation's query sweep under a shared deadline; return the timing and matching results.

    There is no warmup: time_it runs warmups without the deadline, which would defeat it. The returned results come
    from a sweep time_it keeps: a complete one, or the first sweep when even that was cut short.
    """
    latest: list[Any] = []
    kept: list[Any] | None = None

    def search(deadline: Deadline) -> None:
        nonlocal latest, kept
        latest = run_rdkit_queries(library, queries, operation, max_results, num_threads, deadline)
        if kept is None or len(latest) == len(queries):
            kept = latest

    steady = time_it(
        search,
        runs=runs,
        warmups=0,
        max_seconds=max_seconds,
        progress_getter=lambda: len(latest),
        progress_target=len(queries),
    )
    return steady, kept or []


def benchmark_rdkit(
    mols: Sequence[Any],
    queries: Sequence[Any],
    *,
    operations: Sequence[str],
    holder: str,
    num_threads: int,
    max_results: int,
    runs: int,
    max_seconds: float,
) -> dict[str, Measurement]:
    """Build one RDKit library and time every operation against it."""

    def stage() -> None:
        for mol in mols:
            library.GetMolHolder().AddMol(mol)
        if holder == "cached-pattern":
            rdSubstructLibrary.AddPatterns(library, numThreads=num_threads)

    mol_holder = rdSubstructLibrary.MolHolder() if holder == "mol" else rdSubstructLibrary.CachedMolHolder()
    library = rdSubstructLibrary.SubstructLibrary(mol_holder)
    staging = time_it(stage, runs=1, warmups=0)

    measurements = {}
    for operation in operations:
        steady, results = time_rdkit_operation(
            library,
            queries,
            operation,
            max_results=max_results,
            num_threads=num_threads,
            runs=runs,
            max_seconds=max_seconds,
        )
        measurements[operation] = Measurement(
            staging.mean_ms, 0.0, steady.mean_ms, steady.std_ms, results, steady.progress
        )
    return measurements


def benchmark_nvmolkit(
    mols: Sequence[Any],
    queries: Sequence[Any],
    *,
    operations: Sequence[str],
    query_modes: Sequence[str],
    config: Any,
    max_results: int,
    runs: int,
    warmups: int,
) -> dict[tuple[str, str], Measurement]:
    """Build one nvMolKit library and time every operation and query mode against it."""
    from nvmolkit.substruct_library import SubstructLibrary

    library = SubstructLibrary(config=config)
    staging = time_it(lambda: library.addMols(mols), runs=1, warmups=0)
    finalization = time_it(library.finalize, runs=1, warmups=0, gpu_sync=True)

    measurements = {}
    for operation in operations:
        for query_mode in query_modes:
            results: list[Any] = []

            def search(operation: str = operation, query_mode: str = query_mode) -> None:
                nonlocal results
                results = run_nvmolkit_queries(library, queries, operation, max_results, query_mode)

            steady = time_it(search, runs=runs, warmups=warmups, gpu_sync=True)
            measurements[(operation, query_mode)] = Measurement(
                staging.mean_ms, finalization.mean_ms, steady.mean_ms, steady.std_ms, results, len(queries)
            )
    return measurements


def first_mismatch(actual: Sequence[Any], expected: Sequence[Any], label: str) -> str | None:
    """Describe the first differing query over the prefix both result lists cover, or return None."""
    for query_index, (got, want) in enumerate(zip(actual, expected)):
        if got != want:
            return f"{label}: query {query_index} gave {got!r}, RDKit reference {want!r}"
    return None


def result_row(
    measurement: Measurement, *, operation: str, num_mols: int, num_queries: int, **configuration: Any
) -> dict[str, Any]:
    """One CSV row: configuration, timings, and steady and amortized throughput.

    Amortized figures charge the library's one-time staging and finalize cost to this row's operation alone.
    """
    completed = measurement.completed_queries
    pairs = num_mols * completed
    amortized_ms = measurement.staging_ms + measurement.finalize_ms + measurement.steady_ms
    return {
        "operation": operation,
        "num_mols": num_mols,
        "num_queries": num_queries,
        "completed_queries": completed,
        "positive_queries": sum(bool(result) for result in measurement.results),
        **configuration,
        "staging_ms": measurement.staging_ms,
        "finalize_ms": measurement.finalize_ms,
        "steady_ms": measurement.steady_ms,
        "steady_std_ms": measurement.steady_std_ms,
        "amortized_ms": amortized_ms,
        "steady_queries_per_s": throughput_per_s(completed, measurement.steady_ms),
        "steady_pairs_per_s": throughput_per_s(pairs, measurement.steady_ms),
        "amortized_queries_per_s": throughput_per_s(completed, amortized_ms),
    }


def load_queries(args: argparse.Namespace) -> list[Any]:
    """Load SMARTS queries, or SMILES queries with stereochemistry removed, sampling num_queries of them."""
    if args.smarts:
        queries, _ = load_smarts(args.smarts)
        if 0 < args.num_queries < len(queries):
            queries = random.Random(args.seed).sample(queries, args.num_queries)
        return queries
    queries = load_smiles(args.query_smiles, args.num_queries, args.sanitize, seed=args.seed)
    for query in queries:
        Chem.RemoveStereochemistry(query)
    return queries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent SubstructLibrary benchmark: nvMolKit vs RDKit")
    targets = parser.add_mutually_exclusive_group(required=True)
    targets.add_argument("--smiles", "-s", help="SMILES target file")
    targets.add_argument("--pickle", help="Pickled RDKit molecule binaries")
    query_inputs = parser.add_mutually_exclusive_group(required=True)
    query_inputs.add_argument("--smarts", "-q", help="SMARTS query file")
    query_inputs.add_argument("--query_smiles", help="SMILES query file; stereochemistry is ignored")
    parser.add_argument("--num_mols", "-n", type=int, default=0, help="Maximum target molecules; 0 means all")
    parser.add_argument("--num_queries", type=int, default=0, help="Sample this many queries; 0 means all")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--no_sanitize", dest="sanitize", action="store_false")
    parser.add_argument("--operations", nargs="+", choices=OPERATIONS, default=["has"])
    parser.add_argument("--max_results", type=int, default=-1, help="Result limit for get; -1 means all")
    parser.add_argument("--query_modes", nargs="+", choices=["serial", "concurrent"], default=["serial"])
    parser.add_argument("--algorithms", nargs="+", choices=["gsi", "dfs"], default=["gsi", "dfs"])
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--prep_threads", type=int, default=-1, help="Preprocessing threads (-1 = auto)")
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0], help="GPUs one library is sharded across")
    parser.add_argument("--rdkit_holders", nargs="+", choices=["mol", "cached-pattern"], default=["cached-pattern"])
    parser.add_argument("--rdkit_threads", type=int, default=-1)
    add_rdkit_max_seconds_arg(parser, extra_help="The deadline is checked between queries.")
    parser.add_argument("--runs", "-r", type=int, default=3)
    parser.add_argument(
        "--warmups", type=int, default=1, help="nvMolKit warmup sweeps; RDKit runs none so its deadline holds"
    )
    parser.add_argument("--no_validate", dest="validate", action="store_false")
    parser.add_argument("--output", "-o", help="Optional CSV output path")
    add_backend_selection_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.no_rdkit and args.no_nvmolkit:
        raise ValueError("cannot disable both backends")
    if args.max_results == 0 or args.max_results < -1:
        raise ValueError("max_results must be -1 or positive")
    if args.pickle:
        mols = load_pickle(args.pickle, args.num_mols, seed=args.seed)
    else:
        mols = load_smiles(args.smiles, args.num_mols, args.sanitize, seed=args.seed)
    queries = load_queries(args)
    if not mols or not queries:
        raise ValueError("no valid target molecules or queries loaded")

    rows = []
    references: dict[str, list[Any]] = {}
    mismatches: list[str] = []
    if not args.no_rdkit:
        for holder in args.rdkit_holders:
            measurements = benchmark_rdkit(
                mols,
                queries,
                operations=args.operations,
                holder=holder,
                num_threads=args.rdkit_threads,
                max_results=args.max_results,
                runs=args.runs,
                max_seconds=args.rdkit_max_seconds,
            )
            for operation, measurement in measurements.items():
                # RDKit holders must agree with each other before they serve as the reference.
                mismatch = first_mismatch(
                    measurement.results, references.get(operation, []), f"rdkit {holder} {operation}"
                )
                if mismatch is not None:
                    mismatches.append(mismatch)
                if len(measurement.results) > len(references.get(operation, [])):
                    references[operation] = measurement.results
                rows.append(
                    result_row(
                        measurement,
                        operation=operation,
                        num_mols=len(mols),
                        num_queries=len(queries),
                        backend="rdkit",
                        holder=holder,
                        rdkit_threads=args.rdkit_threads,
                        max_results=args.max_results,
                    )
                )

    if not args.no_nvmolkit:
        import torch

        from nvmolkit.substructure import SubstructSearchConfig

        torch.cuda.set_device(args.gpu_ids[0])
        for algorithm in args.algorithms:
            config = SubstructSearchConfig(
                batchSize=args.batch_size,
                workerThreads=args.workers,
                preprocessingThreads=args.prep_threads,
                gpuIds=args.gpu_ids,
                algorithm=algorithm,
            )
            measurements = benchmark_nvmolkit(
                mols,
                queries,
                operations=args.operations,
                query_modes=args.query_modes,
                config=config,
                max_results=args.max_results,
                runs=args.runs,
                warmups=args.warmups,
            )
            for (operation, query_mode), measurement in measurements.items():
                label = f"nvmolkit {algorithm} {query_mode} {operation}"
                if args.validate:
                    reference = references.get(operation, [])
                    print(f"VALIDATION {label}: compared {len(reference)}/{len(queries)} queries", flush=True)
                    mismatch = first_mismatch(measurement.results, reference, label)
                    if mismatch is not None:
                        mismatches.append(mismatch)
                rows.append(
                    result_row(
                        measurement,
                        operation=operation,
                        num_mols=len(mols),
                        num_queries=len(queries),
                        backend="nvmolkit",
                        algorithm=algorithm,
                        query_mode=query_mode,
                        num_gpus=len(args.gpu_ids),
                        gpu_ids=" ".join(str(gpu) for gpu in args.gpu_ids),
                        batch_size=args.batch_size,
                        workers=args.workers,
                        prep_threads=args.prep_threads,
                        max_results=args.max_results,
                    )
                )

    print_csv_rows(rows)
    write_csv_rows(rows, args.output)
    if mismatches:
        raise AssertionError("results differ from RDKit:\n" + "\n".join(mismatches))


if __name__ == "__main__":
    main()
