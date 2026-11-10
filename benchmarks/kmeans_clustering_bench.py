# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark nvMolKit k-means on real molecular fingerprints.

The optional ``cuml`` backend clusters the same fingerprints, unpacked to
float32 bits and L2-normalized, which is the problem nvMolKit's cosine metric
solves. Fingerprint generation, unpacking, and normalization are outside the
measured clustering region.
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

from nvmolkit.clustering import kmeans
from nvmolkit.fingerprints import MorganFingerprintGenerator, unpack_fingerprint

DEFAULT_NUM_MOLS = [1_000, 10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]
DEFAULT_N_CLUSTERS = [100, 1_000]
#: Initializations both backends implement.
CUML_INITS = ("k-means++", "random")


def _load_cuml(enabled: bool) -> tuple[ModuleType | None, str, str]:
    if not enabled:
        return None, "disabled", ""
    try:
        module = importlib.import_module("cuml")
    except ImportError as error:
        return None, "not_installed", str(error)
    try:
        version = importlib.metadata.version("cuml")
    except importlib.metadata.PackageNotFoundError:
        version = getattr(module, "__version__", "unknown")
    return module, "available", version


@nvtx.annotate("bench_nvmolkit_kmeans", color="red")
def _bench_nvmolkit(
    fingerprints: torch.Tensor,
    *,
    n_clusters: int,
    metric: str,
    init: str,
    max_iter: int,
    tol: float,
    seed: int,
    runs: int,
    warmups: int,
) -> tuple[object, int, dict[str, float | int]]:
    holder: dict[str, object] = {}

    @nvtx.annotate("kmeans_nvmolkit_run", color="orange")
    def run() -> None:
        holder["result"] = kmeans(
            fingerprints,
            n_clusters,
            metric=metric,
            init=init,
            max_iter=max_iter,
            tol=tol,
            random_state=seed,
        )

    timing = time_it(run, runs=runs, warmups=warmups, gpu_sync=True)
    result = holder["result"]
    sizes = result.cluster_sizes.torch().cpu().numpy()
    return timing, result.n_iter, _cluster_distribution(sizes, fingerprints.shape[0])


@nvtx.annotate("bench_cuml_kmeans", color="green")
def _bench_cuml(
    cuml: ModuleType,
    normalized_bits: torch.Tensor,
    *,
    n_clusters: int,
    init: str,
    max_iter: int,
    tol: float,
    seed: int,
    runs: int,
    warmups: int,
) -> tuple[object, int, dict[str, float | int]]:
    holder: dict[str, object] = {}

    @nvtx.annotate("kmeans_cuml_run", color="yellow")
    def run() -> None:
        model = cuml.KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=1,
            max_iter=max_iter,
            tol=tol,
            random_state=seed,
            output_type="numpy",
        )
        model.fit(normalized_bits)
        holder["model"] = model

    timing = time_it(run, runs=runs, warmups=warmups, gpu_sync=True)
    model = holder["model"]
    sizes = np.bincount(np.asarray(model.labels_, dtype=np.int64), minlength=n_clusters)
    return timing, int(model.n_iter_), _cluster_distribution(sizes, normalized_bits.shape[0])


def _cluster_distribution(sizes: np.ndarray, num_fingerprints: int) -> dict[str, float | int]:
    """Size statistics of the non-empty clusters."""
    sizes = sizes[sizes > 0]
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


def _normalized_bits(fingerprints: torch.Tensor) -> torch.Tensor:
    bits = unpack_fingerprint(fingerprints).to(torch.float32)
    return bits / bits.sum(dim=1, keepdim=True).sqrt().clamp_min(1.0)


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
    parser.add_argument(
        "--n_clusters",
        "--n-clusters",
        "-k",
        type=int,
        nargs="+",
        default=DEFAULT_N_CLUSTERS,
        help="Cluster counts to benchmark; counts above a molecule count are skipped for it",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["tanimoto", "cosine"],
        default=["tanimoto"],
        help="nvMolKit metrics to benchmark (cuML always solves the cosine problem)",
    )
    parser.add_argument("--init", choices=["k-means++", "random", "maxmin"], default="k-means++")
    parser.add_argument("--max_iter", "--max-iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--fp_size", "--fp-size", type=int, default=1024)
    parser.add_argument("--runs", "-r", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-cuml",
        "--no_cuml",
        dest="no_cuml",
        action="store_true",
        help="Skip the cuML benchmark",
    )
    parser.add_argument(
        "--no-nvmolkit",
        "--no_nvmolkit",
        dest="no_nvmolkit",
        action="store_true",
        help="Skip the nvMolKit benchmark",
    )
    parser.add_argument("--output", "-o", default=None, help="Optional path to write CSV results")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.num_mols or any(num_mols <= 0 for num_mols in args.num_mols):
        parser.error("--num_mols must contain positive integers")
    if any(n_clusters <= 0 for n_clusters in args.n_clusters):
        parser.error("--n_clusters must contain positive integers")
    if args.runs <= 0 or args.warmups < 0:
        parser.error("--runs must be positive and --warmups must be nonnegative")
    if args.max_iter < 0 or not math.isfinite(args.tol) or args.tol < 0:
        parser.error("--max_iter and --tol must be nonnegative")
    if args.no_nvmolkit and args.no_cuml:
        parser.error("no benchmark backend selected")
    if not args.no_cuml and args.init not in CUML_INITS:
        parser.error(f"cuML does not support init={args.init!r}; pass --no-cuml")

    molecule_counts = sorted(set(args.num_mols))
    cluster_counts = sorted(set(args.n_clusters))
    max_size = molecule_counts[-1]
    cuml, cuml_status, cuml_version = _load_cuml(not args.no_cuml)
    if not args.no_cuml and cuml is None:
        print(f"cuML unavailable ({cuml_status}): {cuml_version}", file=sys.stderr)
    if args.no_nvmolkit and cuml is None:
        parser.error("cuML is unavailable and nvMolKit is disabled")

    print(
        f"Scanning {args.smiles} and sampling molecules for a maximum benchmark size of {max_size:,}",
        flush=True,
    )
    molecule_load_start = time.perf_counter()
    with nvtx.annotate("kmeans_molecule_loading", color="cyan"):
        molecules = load_smiles(args.smiles, max_count=max_size + 100, sanitize=True, seed=args.seed)
    molecule_load_seconds = time.perf_counter() - molecule_load_start
    print(f"Molecule loading finished in {molecule_load_seconds:.3f} s", flush=True)
    if len(molecules) < max_size:
        parser.error(f"requested {max_size} molecules, but only {len(molecules)} valid molecules were loaded")

    print(f"Generating {max_size:,} Morgan fingerprints with all available CPU threads", flush=True)
    fingerprint_start = time.perf_counter()
    with nvtx.annotate("kmeans_fingerprint_generation", color="blue"):
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
            normalized_bits = _normalized_bits(device_fps) if cuml is not None else None
            for n_clusters in cluster_counts:
                if n_clusters > size:
                    print(f"\nSkipping n_clusters={n_clusters:,} for {size:,} fingerprints", flush=True)
                    continue
                print(f"\nConfiguration: {size:,} fingerprints, n_clusters={n_clusters:,}", flush=True)
                common_fields = {
                    "num_mols": size,
                    "n_clusters": n_clusters,
                    "init": args.init,
                    "max_iter": args.max_iter,
                    "tol": args.tol,
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
                    for metric in args.metrics:
                        print(
                            f"  Starting nvMolKit GPU ({metric}): {args.warmups} warmup(s), {args.runs} timed run(s)",
                            flush=True,
                        )
                        torch.cuda.cudart().cudaProfilerStart()
                        try:
                            timing, n_iter, distribution = _bench_nvmolkit(
                                device_fps,
                                n_clusters=n_clusters,
                                metric=metric,
                                init=args.init,
                                max_iter=args.max_iter,
                                tol=args.tol,
                                seed=args.seed,
                                runs=args.runs,
                                warmups=args.warmups,
                            )
                        finally:
                            torch.cuda.cudart().cudaProfilerStop()
                        print(
                            f"  Finished nvMolKit GPU ({metric}): {timing.mean_ms:.3f} +/- {timing.std_ms:.3f} ms "
                            f"({n_iter} iterations)",
                            flush=True,
                        )
                        configuration_rows.append(
                            {
                                "method": "nvmolkit",
                                "metric": metric,
                                **common_fields,
                                "device": torch.cuda.get_device_name(device_fps.device),
                                **_timing_fields(timing),
                                "n_iter": n_iter,
                                **distribution,
                            }
                        )

                if cuml is not None:
                    print(
                        f"  Starting cuML GPU (cosine): {args.warmups} warmup(s), {args.runs} timed run(s)",
                        flush=True,
                    )
                    timing, n_iter, distribution = _bench_cuml(
                        cuml,
                        normalized_bits,
                        n_clusters=n_clusters,
                        init=args.init,
                        max_iter=args.max_iter,
                        tol=args.tol,
                        seed=args.seed,
                        runs=args.runs,
                        warmups=args.warmups,
                    )
                    print(
                        f"  Finished cuML GPU (cosine): {timing.mean_ms:.3f} +/- {timing.std_ms:.3f} ms "
                        f"({n_iter} iterations)",
                        flush=True,
                    )
                    configuration_rows.append(
                        {
                            "method": "cuml",
                            "metric": "cosine",
                            **common_fields,
                            "device": torch.cuda.get_device_name(normalized_bits.device),
                            "version": cuml_version,
                            **_timing_fields(timing),
                            "n_iter": n_iter,
                            **distribution,
                        }
                    )

                cuml_throughput = next(
                    (
                        throughput_per_s(size, result_row["time_ms"])
                        for result_row in configuration_rows
                        if result_row["method"] == "cuml"
                    ),
                    math.nan,
                )
                for result_row in configuration_rows:
                    throughput = throughput_per_s(size, result_row["time_ms"])
                    result_row["fingerprints_per_second"] = throughput
                    result_row["vs_cuml_throughput_ratio"] = (
                        throughput / cuml_throughput
                        if result_row["method"] == "nvmolkit" and math.isfinite(cuml_throughput)
                        else "N/A"
                    )

                results.extend(configuration_rows)
                write_csv_rows(results, args.output)
            del normalized_bits
    except Exception:
        write_csv_rows(results, args.output)
        raise

    print("\nCSV Results:")
    print_csv_rows(results)
    if args.output:
        print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()
