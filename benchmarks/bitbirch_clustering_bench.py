# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark nvMolKit BitBIRCH on real molecular fingerprints.

The optional ``bblean`` backend receives the exact same packed fingerprints as
nvMolKit. Fingerprint generation and host/device conversion are intentionally
outside the measured clustering region.
"""

import argparse
import importlib
import importlib.metadata
import math
import sys
import time
from types import ModuleType

import numpy as np
import nvtx
import torch
from bench_utils import (
    load_smiles,
    print_csv_rows,
    throughput_per_s,
    time_it,
    write_csv_rows,
)

from nvmolkit.clustering import bitbirch
from nvmolkit.fingerprints import MorganFingerprintGenerator

DEFAULT_NUM_MOLS = [1_000, 10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]


def _load_bblean(enabled: bool) -> tuple[ModuleType | None, str, str]:
    if not enabled:
        return None, "disabled", ""
    try:
        module = importlib.import_module("bblean")
    except ImportError as error:
        return None, "not_installed", str(error)
    try:
        version = importlib.metadata.version("bblean")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return module, "available", version


def _cluster_count(labels: torch.Tensor) -> int:
    return int(labels.max().item()) + 1 if labels.numel() else 0


@nvtx.annotate("bench_nvmolkit_bitbirch", color="red")
def _bench_nvmolkit(
    fingerprints: torch.Tensor,
    *,
    threshold: float,
    branching_factor: int,
    batch_size: int,
    runs: int,
    warmups: int,
) -> tuple[object, int]:
    holder: dict[str, torch.Tensor] = {}

    @nvtx.annotate("bitbirch_nvmolkit_run", color="orange")
    def run() -> None:
        holder["labels"] = bitbirch(
            fingerprints,
            threshold,
            branching_factor=branching_factor,
            batch_size=batch_size,
        ).torch()

    timing = time_it(run, runs=runs, warmups=warmups, gpu_sync=True)
    return timing, _cluster_count(holder["labels"])


@nvtx.annotate("bench_bblean_bitbirch", color="green")
def _bench_bblean(
    bblean: ModuleType,
    fingerprints: np.ndarray,
    *,
    threshold: float,
    branching_factor: int,
    runs: int,
    warmups: int,
    max_seconds: float,
) -> tuple[object, dict[str, float | int]]:
    holder: dict[str, object] = {}
    completed = [0]

    @nvtx.annotate("bitbirch_bblean_run", color="yellow")
    def run(_deadline=None) -> None:
        tree = bblean.BitBirch(
            threshold=threshold,
            branching_factor=branching_factor,
            merge_criterion="diameter",
        )
        tree.fit(fingerprints)
        holder["tree"] = tree
        completed[0] = 1

    timing = time_it(
        run,
        runs=runs,
        warmups=warmups,
        max_seconds=max_seconds,
        progress_getter=lambda: completed[0],
        progress_target=1,
    )
    tree = holder["tree"]
    return timing, _cluster_distribution(tree.get_cluster_mol_ids(), len(fingerprints))


def _cluster_distribution(clusters, num_fingerprints: int) -> dict[str, float | int]:
    sizes = np.asarray([len(cluster) for cluster in clusters], dtype=np.int64)
    if not sizes.size:
        return {
            "num_clusters": 0,
            "mean_cluster_size": math.nan,
            "cluster_size_p50": math.nan,
            "cluster_size_p90": math.nan,
            "cluster_size_p99": math.nan,
            "max_cluster_size": 0,
            "singleton_cluster_fraction": math.nan,
            "items_in_singletons_fraction": math.nan,
            "largest_cluster_fraction": math.nan,
        }
    quantiles = np.quantile(sizes, [0.5, 0.9, 0.99])
    singleton_count = int(np.count_nonzero(sizes == 1))
    return {
        "num_clusters": int(sizes.size),
        "mean_cluster_size": float(sizes.mean()),
        "cluster_size_p50": float(quantiles[0]),
        "cluster_size_p90": float(quantiles[1]),
        "cluster_size_p99": float(quantiles[2]),
        "max_cluster_size": int(sizes.max()),
        "singleton_cluster_fraction": singleton_count / int(sizes.size),
        "items_in_singletons_fraction": singleton_count / num_fingerprints,
        "largest_cluster_fraction": int(sizes.max()) / num_fingerprints,
    }


def _timing_fields(timing) -> dict[str, float | int]:
    return {
        "time_ms": timing.mean_ms,
        "std_ms": timing.std_ms,
        "runs_completed": len(timing.times_ms),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles", "-s", required=True, help="Path to a real SMILES/CXSMILES input file")
    parser.add_argument(
        "--num_mols",
        "--num-mols",
        "-n",
        type=int,
        nargs="+",
        default=DEFAULT_NUM_MOLS,
        help="Molecule counts to benchmark",
    )
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Sweep multiple thresholds; overrides --threshold",
    )
    parser.add_argument("--branching_factor", "--branching-factor", type=int, default=254)
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        type=int,
        default=1024,
        help="nvMolKit insertion batch size (default: 1024)",
    )
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--fp_size", "--fp-size", type=int, default=1024)
    parser.add_argument("--runs", "-r", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-bblean",
        "--no_bblean",
        dest="no_bblean",
        action="store_true",
        help="Skip the bblean benchmark",
    )
    parser.add_argument(
        "--no-nvmolkit",
        "--no_nvmolkit",
        dest="no_nvmolkit",
        action="store_true",
        help="Skip the nvMolKit benchmark",
    )
    parser.add_argument(
        "--bblean-max-seconds",
        "--bblean_max_seconds",
        dest="bblean_max_seconds",
        type=float,
        default=0.0,
        help=(
            "Stop starting bblean timing repetitions after this many wall-clock seconds per configuration; "
            "0 disables the limit"
        ),
    )
    parser.add_argument("--output", "-o", default=None, help="Optional path to write CSV results")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    thresholds = args.thresholds if args.thresholds is not None else [args.threshold]

    if not args.num_mols or any(num_mols <= 0 for num_mols in args.num_mols):
        parser.error("--num_mols must contain positive integers")
    if args.runs <= 0 or args.warmups < 0:
        parser.error("--runs must be positive and --warmups must be nonnegative")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.bblean_max_seconds < 0:
        parser.error("--bblean-max-seconds must be nonnegative")
    if any(not math.isfinite(threshold) or not 0 <= threshold <= 1 for threshold in thresholds):
        parser.error("thresholds must be finite and in [0, 1]")
    if args.no_nvmolkit and args.no_bblean:
        parser.error("no benchmark backend selected")

    molecule_counts = sorted(set(args.num_mols))
    max_size = molecule_counts[-1]
    bblean, bblean_status, bblean_version = _load_bblean(not args.no_bblean)
    if not args.no_bblean and bblean is None:
        print(f"bblean unavailable ({bblean_status}): {bblean_version}", file=sys.stderr)
    if args.no_nvmolkit and bblean is None:
        parser.error("bblean is unavailable and nvMolKit is disabled")

    print(
        f"Scanning {args.smiles} and sampling molecules for a maximum benchmark size of {max_size:,}",
        flush=True,
    )
    molecule_load_start = time.perf_counter()
    with nvtx.annotate("bitbirch_molecule_loading", color="cyan"):
        molecules = load_smiles(args.smiles, max_count=max_size + 100, sanitize=True, seed=args.seed)
    molecule_load_seconds = time.perf_counter() - molecule_load_start
    print(f"Molecule loading finished in {molecule_load_seconds:.3f} s", flush=True)
    if len(molecules) < max_size:
        parser.error(f"requested {max_size} molecules, but only {len(molecules)} valid molecules were loaded")

    print(f"Generating {max_size:,} Morgan fingerprints with all available CPU threads", flush=True)
    fingerprint_start = time.perf_counter()
    with nvtx.annotate("bitbirch_fingerprint_generation", color="blue"):
        fingerprints = (
            MorganFingerprintGenerator(args.radius, args.fp_size)
            .GetFingerprints(molecules[:max_size], num_threads=0)
            .torch()
        )
        torch.cuda.synchronize()
    fingerprint_seconds = time.perf_counter() - fingerprint_start
    print(f"Fingerprint generation finished in {fingerprint_seconds:.3f} s", flush=True)

    results = []
    try:
        for size in molecule_counts:
            device_fps = fingerprints[:size].contiguous()
            host_fps = device_fps.cpu().numpy().view(np.uint8).reshape(size, -1).copy() if bblean is not None else None
            for threshold in thresholds:
                print(f"\nConfiguration: {size:,} fingerprints, threshold={threshold}", flush=True)
                common_fields = {
                    "num_mols": size,
                    "threshold": threshold,
                    "branching_factor": args.branching_factor,
                    "radius": args.radius,
                    "fp_size": args.fp_size,
                    "runs": args.runs,
                    "warmups": args.warmups,
                    "seed": args.seed,
                    "fingerprint_total_seconds": fingerprint_seconds,
                    "input_file": args.smiles,
                    "input_type": "smiles",
                }
                configuration_rows = []

                if not args.no_nvmolkit:
                    print(
                        f"  Starting nvMolKit GPU: {args.warmups} warmup(s), {args.runs} timed run(s), "
                        f"batch_size={args.batch_size}",
                        flush=True,
                    )
                    torch.cuda.cudart().cudaProfilerStart()
                    try:
                        timing, num_clusters = _bench_nvmolkit(
                            device_fps,
                            threshold=threshold,
                            branching_factor=args.branching_factor,
                            batch_size=args.batch_size,
                            runs=args.runs,
                            warmups=args.warmups,
                        )
                    finally:
                        torch.cuda.cudart().cudaProfilerStop()
                    print(
                        f"  Finished nvMolKit GPU: {timing.mean_ms:.3f} +/- {timing.std_ms:.3f} ms "
                        f"({num_clusters:,} clusters)",
                        flush=True,
                    )
                    configuration_rows.append(
                        {
                            "method": "nvmolkit",
                            **common_fields,
                            "batch_size": args.batch_size,
                            "device": torch.cuda.get_device_name(device_fps.device),
                            **_timing_fields(timing),
                            "num_clusters": num_clusters,
                        }
                    )

                if bblean is not None:
                    budget = f", {args.bblean_max_seconds:g} s timing budget" if args.bblean_max_seconds > 0 else ""
                    print(
                        f"  Starting bblean CPU: {args.warmups} warmup(s), up to {args.runs} timed run(s){budget}",
                        flush=True,
                    )
                    timing, distribution = _bench_bblean(
                        bblean,
                        host_fps,
                        threshold=threshold,
                        branching_factor=args.branching_factor,
                        runs=args.runs,
                        warmups=args.warmups,
                        max_seconds=args.bblean_max_seconds,
                    )
                    print(
                        f"  Finished bblean CPU: {timing.mean_ms:.3f} +/- {timing.std_ms:.3f} ms "
                        f"over {len(timing.times_ms)} run(s) ({distribution['num_clusters']:,} clusters)",
                        flush=True,
                    )
                    configuration_rows.append(
                        {
                            "method": "bblean",
                            **common_fields,
                            "batch_size": "N/A",
                            "device": "CPU",
                            "version": bblean_version,
                            **_timing_fields(timing),
                            **distribution,
                        }
                    )

                bblean_throughput = next(
                    (
                        throughput_per_s(size, result_row["time_ms"])
                        for result_row in configuration_rows
                        if result_row["method"] == "bblean"
                    ),
                    math.nan,
                )
                for result_row in configuration_rows:
                    throughput = throughput_per_s(size, result_row["time_ms"])
                    result_row["fingerprints_per_second"] = throughput
                    result_row["vs_bblean_throughput_ratio"] = (
                        throughput / bblean_throughput
                        if result_row["method"] == "nvmolkit" and math.isfinite(bblean_throughput)
                        else "N/A"
                    )

                results.extend(configuration_rows)
                write_csv_rows(results, args.output)
    except Exception:
        write_csv_rows(results, args.output)
        raise

    print("\nCSV Results:")
    print_csv_rows(results)
    if args.output:
        print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()
