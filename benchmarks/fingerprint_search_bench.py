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

"""Benchmark: GPU top-k fingerprint search vs RDKit.

Measures nvMolKit's FingerprintSearchIndex.search, with and without popcount
pruning, against RDKit Bulk*Similarity followed by a top-k selection, across
library sizes and k. Queries are molecules held out of the library.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from bench_utils import (
    Deadline,
    TimingResult,
    add_backend_selection_args,
    add_rdkit_max_seconds_arg,
    available_cpu_count,
    load_smiles,
    print_csv_rows,
    process_map_bounded,
    time_it,
    write_csv_rows,
)
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import BulkCosineSimilarity, BulkTanimotoSimilarity, ExplicitBitVect

from nvmolkit.fingerprints import MorganFingerprintGenerator
from nvmolkit.similarity import FingerprintSearchIndex

RDKIT_BULK = {"tanimoto": BulkTanimotoSimilarity, "cosine": BulkCosineSimilarity}

_worker_state: dict = {}


def rdkit_top_k(query, library_fps: list, metric: str, k: int, threshold: float) -> np.ndarray:
    """Library indices of the top k hits for one query, best first."""
    sims = np.asarray(RDKIT_BULK[metric](query, library_fps))
    hits = np.flatnonzero(sims >= threshold)
    if len(hits) > k:
        hits = hits[np.argpartition(-sims[hits], k - 1)[:k]]
    return hits[np.argsort(-sims[hits], kind="stable")]


def _rdkit_worker_init(library_binaries: list[bytes], metric: str, k: int, threshold: float) -> None:
    _worker_state["library"] = [ExplicitBitVect(binary) for binary in library_binaries]
    _worker_state["params"] = (metric, k, threshold)


def _rdkit_worker_batch(batch: list[bytes]) -> int:
    for binary in batch:
        rdkit_top_k(ExplicitBitVect(binary), _worker_state["library"], *_worker_state["params"])
    return len(batch)


def bench_rdkit(
    query_fps: list,
    library_fps: list,
    metric: str,
    k: int,
    threshold: float,
    threads: int,
    max_seconds: float,
    runs: int,
) -> tuple[TimingResult, int]:
    """Benchmark RDKit top-k search; return timing and queries completed.

    With one thread, queries run in order and the deadline is polled between queries. With more, query batches run
    in a process pool (RDKit's bulk similarity holds the GIL); each worker receives the library once at startup,
    which is included in the timing, and the deadline truncates at a completed batch.
    """
    processed_count = [0]

    if threads > 1:
        library_binaries = [fp.ToBinary() for fp in library_fps]
        query_binaries = [fp.ToBinary() for fp in query_fps]
        batch_size = max(1, len(query_binaries) // (threads * 4))
        batches = [query_binaries[start : start + batch_size] for start in range(0, len(query_binaries), batch_size)]

        def run(deadline: Deadline) -> None:
            mapped = process_map_bounded(
                _rdkit_worker_batch,
                batches,
                max_workers=threads,
                deadline=deadline,
                initializer=_rdkit_worker_init,
                initargs=(library_binaries, metric, k, threshold),
            )
            processed_count[0] = sum(mapped.results)

    else:

        def run(deadline: Deadline) -> None:
            processed_count[0] = 0
            for query in query_fps:
                rdkit_top_k(query, library_fps, metric, k, threshold)
                processed_count[0] += 1
                if deadline.expired():
                    break

    timing = time_it(
        run,
        runs=runs,
        warmups=0,
        max_seconds=max_seconds,
        progress_getter=lambda: processed_count[0],
        progress_target=len(query_fps),
    )
    if timing.progress is None:
        raise RuntimeError("bounded timing did not report progress")
    return timing, timing.progress


def run(
    smiles_path: str,
    library_sizes: list[int],
    num_queries: int,
    k_values: list[int],
    threshold: float,
    metric: str,
    radius: int,
    fp_size: int,
    seed: int,
    rdkit_threads: int,
    rdkit_max_seconds: float,
    runs: int,
    warmups: int,
    no_rdkit: bool,
    no_nvmolkit: bool,
    output: str | None,
) -> None:
    if no_rdkit and no_nvmolkit:
        raise ValueError("cannot disable both RDKit and nvMolKit")

    if not no_nvmolkit:
        print(f"GPU: {torch.cuda.get_device_name(0)}  CUDA: {torch.version.cuda}")
    max_library = max(library_sizes)
    print(f"Loading SMILES from {smiles_path} (target {max_library} library + {num_queries} query mols)")
    mols = load_smiles(smiles_path, max_count=max_library + num_queries, seed=seed)
    if len(mols) <= num_queries:
        raise RuntimeError(f"need more than {num_queries} molecules, parsed {len(mols)}")
    query_mols, library_mols = mols[:num_queries], mols[num_queries:]
    if len(library_mols) < max_library:
        print(f"  only {len(library_mols)} library mols parsed; repeating them to reach {max_library}")
    while len(library_mols) < max_library:
        library_mols += library_mols
    library_mols = library_mols[:max_library]

    rdkit_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    nvmolkit_fpgen = MorganFingerprintGenerator(radius=radius, fpSize=fp_size)
    if not no_rdkit:
        rdkit_library_all = list(rdkit_fpgen.GetFingerprints(library_mols, numThreads=available_cpu_count()))
        rdkit_queries = list(rdkit_fpgen.GetFingerprints(query_mols, numThreads=available_cpu_count()))
    if not no_nvmolkit:
        library_all = torch.as_tensor(nvmolkit_fpgen.GetFingerprints(library_mols), device="cuda")
        queries = torch.as_tensor(nvmolkit_fpgen.GetFingerprints(query_mols), device="cuda")

    rows: list[dict[str, float | int | str]] = []
    for library_size in sorted(library_sizes):
        index = None
        build_s = None
        if not no_nvmolkit:
            library = library_all[:library_size].contiguous()
            build = time_it(
                lambda library=library: FingerprintSearchIndex(library, metric=metric),
                runs=runs,
                warmups=warmups,
                gpu_sync=True,
            )
            build_s = build.median_s
            index = FingerprintSearchIndex(library, metric=metric)
            print(f"\n=== library={library_size}: index build {build_s * 1000:.1f} ms ===")
        else:
            print(f"\n=== library={library_size} ===")

        for k in sorted(k_values):
            row: dict[str, float | int | str] = {
                "library_size": library_size,
                "num_queries": len(query_mols),
                "k": k,
                "threshold": threshold,
                "metric": metric,
                "fp_size": fp_size,
            }
            print(f"  k={k}:")

            rdkit_queries_per_s: float | None = None
            if not no_rdkit:
                rdkit_timing, rdkit_done = bench_rdkit(
                    rdkit_queries,
                    rdkit_library_all[:library_size],
                    metric,
                    k,
                    threshold,
                    rdkit_threads,
                    rdkit_max_seconds,
                    runs,
                )
                rdkit_queries_per_s = rdkit_done / rdkit_timing.median_s
                truncated = rdkit_done < len(rdkit_queries)
                suffix = f" [truncated at {rdkit_done}/{len(rdkit_queries)} queries]" if truncated else ""
                print(f"    RDKit CPU ({rdkit_threads} process(es)): {rdkit_queries_per_s:.1f} queries/s{suffix}")
                row["rdkit_threads"] = rdkit_threads
                row["rdkit_median_s"] = rdkit_timing.median_s
                row["rdkit_queries_processed"] = rdkit_done
                row["rdkit_truncated"] = int(truncated)
                row["rdkit_queries_per_s"] = rdkit_queries_per_s

            if index is not None:
                row["gpu_build_s"] = build_s
                for prune, label in ((True, "gpu"), (False, "gpu_noprune")):
                    result = time_it(
                        lambda index=index, k=k, prune=prune: index.search(
                            queries, k=k, threshold=threshold, prune=prune
                        ),
                        runs=runs,
                        warmups=warmups,
                        gpu_sync=True,
                    )
                    queries_per_s = len(query_mols) / result.median_s
                    print(
                        f"    nvMolKit GPU (prune={prune}): {result.median_s * 1000:.2f} ms "
                        f"({queries_per_s:.0f} queries/s)"
                    )
                    row[f"{label}_median_s"] = result.median_s
                    row[f"{label}_queries_per_s"] = queries_per_s
                if rdkit_queries_per_s is not None:
                    row["speedup"] = row["gpu_queries_per_s"] / rdkit_queries_per_s
                    print(f"    GPU speedup vs RDKit (queries/s): {row['speedup']:.1f}x")

            rows.append(row)
        del index
        torch.cuda.empty_cache()

    if rows:
        print("\nCSV Results:")
        print_csv_rows(rows)

    if output and rows:
        out_path = Path(output)
        write_csv_rows(rows, out_path)
        print(f"\nWrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Top-k fingerprint similarity search benchmark")
    parser.add_argument(
        "--smiles",
        type=str,
        default="data/chembl_10k.smi",
        help="Path to input SMILES file. Molecules are repeated if the file is smaller than the largest library.",
    )
    parser.add_argument(
        "--library_sizes", type=int, nargs="+", default=[1000, 10000, 100000], help="Library size sweep points"
    )
    parser.add_argument("--num_queries", type=int, default=100, help="Number of held-out query molecules")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 10, 100], help="Number of neighbors sweep points")
    parser.add_argument("--threshold", type=float, default=0.0, help="Minimum similarity for a hit")
    parser.add_argument("--metric", choices=sorted(RDKIT_BULK), default="tanimoto")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--fp_size", type=int, default=2048)
    parser.add_argument(
        "--rdkit_threads", type=int, default=1, help="RDKit processes; queries are split across them (default: 1)"
    )
    add_rdkit_max_seconds_arg(
        parser,
        extra_help="The cap applies per timing iteration and truncates at a query (one process) or a query batch.",
    )
    parser.add_argument("--runs", type=int, default=3, help="Timed repetitions per measurement (default: 3)")
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="Untimed nvMolKit repetitions before timing (default: 1). RDKit is not warmed up.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="Optional CSV output path")
    add_backend_selection_args(parser)
    args = parser.parse_args()

    if args.no_rdkit and args.no_nvmolkit:
        parser.error("cannot pass both --no_rdkit and --no_nvmolkit")
    if args.runs < 1 or args.warmups < 0:
        parser.error("--runs must be positive and --warmups must be nonnegative")
    if args.rdkit_threads < 1:
        parser.error("--rdkit_threads must be at least 1")

    run(
        smiles_path=args.smiles,
        library_sizes=args.library_sizes,
        num_queries=args.num_queries,
        k_values=args.k,
        threshold=args.threshold,
        metric=args.metric,
        radius=args.radius,
        fp_size=args.fp_size,
        seed=args.seed,
        rdkit_threads=args.rdkit_threads,
        rdkit_max_seconds=args.rdkit_max_seconds,
        runs=args.runs,
        warmups=args.warmups,
        no_rdkit=args.no_rdkit,
        no_nvmolkit=args.no_nvmolkit,
        output=args.output,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
