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

"""
Substructure search benchmark comparing nvmolkit GPU substructure search against RDKit.

Compares two approaches:
  1. nvmolkit GPU-accelerated substructure search
  2. RDKit SubstructMatch API (raw or SubstructLibrary mode)

Supports three search modes:
  - hasSubstructMatch: Boolean match detection (faster)
  - countSubstructMatches: Count of matches per target/query pair
  - getSubstructMatches: Full match enumeration with optional max matches

RDKit matching modes:
  - raw: Direct mol.HasSubstructMatch/GetSubstructMatches API with multiprocessing
  - substructlib: rdSubstructLibrary.SubstructLibrary with native multithreading

Usage:
    python substruct_bench.py --smiles <smiles_file> --smarts <smarts_file>

    # Get all matches instead of just boolean:
    python substruct_bench.py --smiles <smiles_file> --smarts <smarts_file> --mode getSubstructMatches

    # Limit to first 10 matches per target/query pair:
    python substruct_bench.py --smiles <smiles_file> --smarts <smarts_file> --mode getSubstructMatches --max_matches 10

    # Skip nvmolkit (for CPU-only comparison):
    python substruct_bench.py --smiles <smiles_file> --smarts <smarts_file> --no_nvmolkit

    # Use multiprocessing for RDKit raw mode with 8 processes:
    python substruct_bench.py --smiles <smiles_file> --smarts <smarts_file> --rdkit_threads 8

    # Use SubstructLibrary with native threading:
    python substruct_bench.py --smiles <smiles_file> --smarts <smarts_file> --rdkit_match_mode substructlib --rdkit_threads 8

    # Run multiple configurations from a dataframe (smarts, batch_size, workers, prep_threads, mode, num_gpus):
    python substruct_bench.py --smiles <smiles_file> --config <config.csv>

"""

import argparse
import gc
import pickle
import random
import sys
from functools import partial
from multiprocessing import Pool
from typing import Callable, Iterator

import nvtx
import pandas as pd
from benchmark_timing import time_it as _time_it
from nvmolkit import autotune as nv_autotune
from nvmolkit.substructure import (
    SubstructSearchConfig,
    countSubstructMatches,
    getSubstructMatches,
    hasSubstructMatch,
)
from rdkit import Chem, RDLogger
from rdkit.Chem import rdSubstructLibrary
from tqdm.contrib.concurrent import process_map

OPTUNA_AVAILABLE = nv_autotune.is_available()


def time_it(func: Callable, runs: int = 1, gpu_sync: bool = False) -> tuple[float, float]:
    """Time a function and return (avg_ms, std_ms)."""
    result = _time_it(func, runs=runs, warmups=0, gpu_sync=gpu_sync)
    return result.mean_ms, result.std_ms


def load_pickle(filepath: str, max_count: int = 0, seed: int | None = None) -> list[Chem.Mol]:
    """Load molecules from a pickled file containing binary mol data.

    When ``max_count > 0``, a uniform random sample of binary mols is drawn.
    """
    with open(filepath, "rb") as f:
        binary_mols = pickle.load(f)
    if max_count > 0 and len(binary_mols) > max_count:
        binary_mols = random.Random(seed).sample(binary_mols, max_count)
    mols = process_map(
        _mol_from_binary,
        binary_mols,
        desc="Unpickling molecules",
        chunksize=1000,
    )
    print(f"  Loaded {len(mols)} molecules from {filepath}")
    return mols


def _mol_from_binary(binary_mol: bytes) -> Chem.Mol:
    """Load a molecule from RDKit binary format."""
    return Chem.Mol(binary_mol)


def _parse_smiles(smi: str, sanitize: bool) -> Chem.Mol | None:
    """Parse a single SMILES string."""
    return Chem.MolFromSmiles(smi, sanitize=sanitize)


def _iter_smiles_tokens(filepath: str, sanitize: bool) -> Iterator[str]:
    """Yield SMILES tokens from a file, skipping blanks/comments and a parse-failing first line.

    The first non-comment line is parsed quietly; if it fails to parse it is treated as a header
    and dropped, matching the original loader's behavior.
    """
    with open(filepath, "r") as f:
        first_data_seen = False
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            smi = stripped.split()[0]
            if not first_data_seen:
                first_data_seen = True
                RDLogger.DisableLog("rdApp.*")
                mol = Chem.MolFromSmiles(smi, sanitize=sanitize)
                RDLogger.EnableLog("rdApp.*")
                if mol is None:
                    continue
            yield smi


def load_smiles(filepath: str, max_count: int = 0, sanitize: bool = True, seed: int | None = None) -> list[Chem.Mol]:
    """Load and parse molecules from a SMILES file.

    When ``max_count > 0``, reservoir sampling draws a uniform random sample of lines in a single
    streaming pass (with a 10% buffer to absorb parse failures) so the file isn't fully loaded into
    memory and only the sampled SMILES are parsed.
    """
    # Use a 10% buffer to account for potential parse failures
    # "On parse failures continue down the file. Load 10% more molecules than needed"
    read_limit = int(max_count * 1.1) if max_count > 0 else 0

    if read_limit > 0:
        rng = random.Random(seed)
        reservoir: list[str] = []
        for index, smi in enumerate(_iter_smiles_tokens(filepath, sanitize)):
            if index < read_limit:
                reservoir.append(smi)
            else:
                replace_index = rng.randint(0, index)
                if replace_index < read_limit:
                    reservoir[replace_index] = smi
        smiles_list = reservoir
    else:
        smiles_list = list(_iter_smiles_tokens(filepath, sanitize))

    mols: list[Chem.Mol] = []
    if smiles_list:
        parse_func = partial(_parse_smiles, sanitize=sanitize)
        parsed = process_map(parse_func, smiles_list, desc="Parsing molecules", chunksize=1000)

        parse_failures = 0
        for mol in parsed:
            if mol is None:
                parse_failures += 1
            else:
                mols.append(mol)

        if parse_failures > 0:
            print(f"    ({parse_failures} parse failures)")

    # Trim to exactly max_count if we have more than requested
    if max_count > 0 and len(mols) > max_count:
        mols = mols[:max_count]

    print(f"  Loaded {len(mols)} molecules from {filepath}")
    return mols


def load_smarts(filepath: str, max_count: int = 0) -> tuple[list[Chem.Mol], list[str]]:
    """Load and parse query patterns from a SMARTS file."""
    queries = []
    smarts_list = []
    parse_failures = 0

    with open(filepath, "r") as f:
        for line in f:
            if max_count > 0 and len(queries) >= max_count:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            smarts = line.split()[0]
            query = Chem.MolFromSmarts(smarts)
            if query is None:
                parse_failures += 1
                continue
            queries.append(query)
            smarts_list.append(smarts)

    print(f"  Loaded {len(queries)} SMARTS patterns from {filepath}")
    if parse_failures > 0:
        print(f"    ({parse_failures} parse failures)")
    return queries, smarts_list


_worker_queries = None
_worker_params = None


def _rdkit_worker_init(query_binaries: list[bytes], max_matches: int):
    """Initialize worker process with shared query data."""
    global _worker_queries, _worker_params
    _worker_queries = [Chem.Mol(qb) for qb in query_binaries]
    _worker_params = Chem.SubstructMatchParameters()
    _worker_params.uniquify = False
    if max_matches > 0:
        _worker_params.maxMatches = max_matches


def _rdkit_worker_has(mol_binary: bytes) -> list[bool]:
    """Worker function for hasSubstructMatch multiprocessing."""
    mol = Chem.Mol(mol_binary)
    return [mol.HasSubstructMatch(q, _worker_params) for q in _worker_queries]


def _rdkit_worker_get(mol_binary: bytes) -> list[tuple]:
    """Worker function for getSubstructMatches multiprocessing."""
    mol = Chem.Mol(mol_binary)
    return [mol.GetSubstructMatches(q, _worker_params) for q in _worker_queries]


def _rdkit_worker_count(mol_binary: bytes) -> list[int]:
    """Worker function for countSubstructMatches multiprocessing."""
    mol = Chem.Mol(mol_binary)
    return [len(mol.GetSubstructMatches(q, _worker_params)) for q in _worker_queries]


@nvtx.annotate("bench_rdkit_substruct", color="green")
def bench_rdkit_substruct(
    mols: list[Chem.Mol], queries: list[Chem.Mol], runs: int, mode: str, max_matches: int, threads: int = 1
) -> tuple[float, float, list]:
    """Benchmark RDKit SubstructMatch API."""
    params = Chem.SubstructMatchParameters()
    params.uniquify = False
    if max_matches > 0:
        params.maxMatches = max_matches

    results_data = []

    if threads > 1:
        mol_binaries = [mol.ToBinary() for mol in mols]
        query_binaries = [q.ToBinary() for q in queries]
        if mode == "hasSubstructMatch":
            worker_func = _rdkit_worker_has
        elif mode == "countSubstructMatches":
            worker_func = _rdkit_worker_count
        else:
            worker_func = _rdkit_worker_get
        chunksize = max(1, len(mol_binaries) // (threads * 4))

        @nvtx.annotate("substruct_run_mp", color="yellow")
        def run():
            nonlocal results_data
            with Pool(threads, initializer=_rdkit_worker_init, initargs=(query_binaries, max_matches)) as pool:
                results_data = pool.map(worker_func, mol_binaries, chunksize=chunksize)
    else:

        @nvtx.annotate("substruct_run", color="yellow")
        def run():
            nonlocal results_data
            results_data = []
            if mode == "hasSubstructMatch":
                for mol in mols:
                    mol_results = []
                    for query in queries:
                        mol_results.append(mol.HasSubstructMatch(query, params))
                    results_data.append(mol_results)
            elif mode == "countSubstructMatches":
                for mol in mols:
                    mol_results = []
                    for query in queries:
                        mol_results.append(len(mol.GetSubstructMatches(query, params)))
                    results_data.append(mol_results)
            else:
                for mol in mols:
                    mol_results = []
                    for query in queries:
                        matches = mol.GetSubstructMatches(query, params)
                        mol_results.append(matches)
                    results_data.append(mol_results)

    avg_ms, std_ms = time_it(run, runs)
    return avg_ms, std_ms, results_data


@nvtx.annotate("bench_rdkit_substructlib", color="green")
def bench_rdkit_substructlib(
    mols: list[Chem.Mol], queries: list[Chem.Mol], runs: int, mode: str, max_matches: int, threads: int = 1
) -> tuple[float, float, list]:
    """Benchmark RDKit SubstructLibrary API with native multithreading."""
    num_mols = len(mols)
    num_queries = len(queries)

    params = Chem.SubstructMatchParameters()
    params.uniquify = False
    if max_matches > 0:
        params.maxMatches = max_matches

    results_data = [[None] * num_queries for _ in range(num_mols)]

    @nvtx.annotate("substructlib_run", color="yellow")
    def run():
        nonlocal results_data

        mol_holder = rdSubstructLibrary.CachedMolHolder()
        fp_holder = rdSubstructLibrary.PatternHolder()
        lib = rdSubstructLibrary.SubstructLibrary(mol_holder, fp_holder)
        for mol in mols:
            lib.AddMol(mol)

        results_data = [[None] * num_queries for _ in range(num_mols)]

        if mode == "hasSubstructMatch":
            for q_idx, query in enumerate(queries):
                matching_indices = lib.GetMatches(query, numThreads=threads)
                matching_set = set(matching_indices)
                for m_idx in range(num_mols):
                    results_data[m_idx][q_idx] = m_idx in matching_set
        elif mode == "countSubstructMatches":
            for q_idx, query in enumerate(queries):
                matching_indices = lib.GetMatches(query, numThreads=threads)
                matching_set = set(matching_indices)
                for m_idx in range(num_mols):
                    if m_idx in matching_set:
                        results_data[m_idx][q_idx] = len(mols[m_idx].GetSubstructMatches(query, params))
                    else:
                        results_data[m_idx][q_idx] = 0
        else:
            for q_idx, query in enumerate(queries):
                matching_indices = lib.GetMatches(query, numThreads=threads)
                matching_set = set(matching_indices)
                for m_idx in range(num_mols):
                    if m_idx in matching_set:
                        results_data[m_idx][q_idx] = mols[m_idx].GetSubstructMatches(query, params)
                    else:
                        results_data[m_idx][q_idx] = ()

    avg_ms, std_ms = time_it(run, runs)
    return avg_ms, std_ms, results_data


@nvtx.annotate("bench_nvmolkit", color="red")
def bench_nvmolkit(
    mols: list[Chem.Mol], queries: list[Chem.Mol], runs: int, mode: str, config
) -> tuple[float, float, object]:
    """Benchmark nvmolkit GPU substructure search."""
    results_data: object = None

    @nvtx.annotate("nvmolkit_run", color="orange")
    def run():
        nonlocal results_data
        if mode == "hasSubstructMatch":
            results_data = hasSubstructMatch(mols, queries, config)
        elif mode == "countSubstructMatches":
            results_data = countSubstructMatches(mols, queries, config)
        else:
            results_data = getSubstructMatches(mols, queries, config)

    avg_ms, std_ms = time_it(run, runs, gpu_sync=True)
    return avg_ms, std_ms, results_data


def _load_config_dataframe(config_path: str) -> list[dict]:
    return pd.read_csv(config_path).to_dict("records")


def main():
    parser = argparse.ArgumentParser(description="Substructure search benchmark: nvmolkit vs RDKit SubstructMatch")
    parser.add_argument("--smiles", "-s", help="Path to SMILES file with molecules to search")
    parser.add_argument("--pickle", help="Path to pickled molecules file (alternative to --smiles)")
    parser.add_argument("--smarts", "-q", help="Path to SMARTS file with query patterns")
    parser.add_argument(
        "--config",
        help=(
            "Path to config dataframe (.csv/.pkl/.pickle/.parquet) with columns: "
            "smarts, batch_size, workers, prep_threads, mode, num_gpus"
        ),
    )
    parser.add_argument("--num_mols", "-n", type=int, default=0, help="Max number of molecules (default: 0 = all)")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling SMILES when --num_mols > 0 (default: 42)",
    )
    parser.add_argument("--sanitize", action="store_true", dest="sanitize", help="Sanitize SMILES during parsing")
    parser.add_argument(
        "--no_sanitize", action="store_false", dest="sanitize", help="Skip sanitization (preprocessed SMILES)"
    )
    parser.set_defaults(sanitize=False)
    parser.add_argument("--runs", "-r", type=int, default=1, help="Number of timing runs (default: 1)")
    parser.add_argument(
        "--mode",
        "-m",
        choices=["hasSubstructMatch", "getSubstructMatches", "countSubstructMatches"],
        default="hasSubstructMatch",
        help="Search mode (default: hasSubstructMatch)",
    )
    parser.add_argument(
        "--max_matches", type=int, default=0, help="Maximum matches per target/query pair, 0 = all (default: 0)"
    )
    parser.add_argument("--no_nvmolkit", action="store_true", help="Skip nvmolkit benchmark")
    parser.add_argument("--no_rdkit", action="store_true", help="Skip RDKit benchmark")
    parser.add_argument(
        "--rdkit_match_mode",
        choices=["raw", "substructlib"],
        default="raw",
        help="RDKit matching mode: raw (direct API) or substructlib (SubstructLibrary) (default: raw)",
    )
    parser.add_argument(
        "--rdkit_threads",
        type=int,
        default=1,
        help="RDKit threads (multiprocessing for raw, native for substructlib) (default: 1)",
    )
    parser.add_argument("--batch_size", "-b", type=int, default=1024, help="nvmolkit batch size (default: 1024)")
    parser.add_argument("--workers", type=int, default=-1, help="nvmolkit GPU worker threads per GPU (-1 = auto)")
    parser.add_argument("--prep_threads", type=int, default=-1, help="nvmolkit preprocessing threads (-1 = auto)")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use (default: 1)")
    parser.add_argument("--warmup", action="store_true", dest="warmup", help="Perform warmup run (default)")
    parser.add_argument("--no_warmup", action="store_false", dest="warmup", help="Skip warmup run")
    parser.set_defaults(warmup=True)
    parser.add_argument(
        "--autotune",
        action="store_true",
        help=(
            "Tune nvmolkit SubstructSearchConfig (batchSize/workerThreads/preprocessingThreads) "
            "before timing. Requires the [autotune] extra (optuna). Single-config mode only."
        ),
    )
    parser.add_argument(
        "--autotune_save",
        type=str,
        default=None,
        help="Path to save the tuned SubstructSearchConfig as JSON (only with --autotune)",
    )
    parser.add_argument(
        "--autotune_load",
        type=str,
        default=None,
        help=(
            "Path to a previously-saved SubstructSearchConfig JSON. "
            "Overrides --batch_size/--workers/--prep_threads (and --num_gpus if gpuIds present in the file)."
        ),
    )
    parser.add_argument(
        "--autotune_trials",
        type=int,
        default=20,
        help="Number of Optuna trials when --autotune is set (default: 20)",
    )
    parser.add_argument(
        "--autotune_time_budget",
        type=float,
        default=10.0,
        help="Target wall-clock seconds per Optuna trial (default: 10.0)",
    )
    parser.add_argument(
        "--autotune_calibration_size",
        type=int,
        default=0,
        help=(
            "Number of target molecules to use per autotune trial. "
            "0 = auto-subsample (~10%% of the workload, capped at 2000). Default: 0"
        ),
    )
    parser.add_argument(
        "--autotune_seed",
        type=int,
        default=42,
        help="Seed for the Optuna sampler (default: 42)",
    )
    parser.add_argument(
        "--validate", action="store_true", dest="validate", help="Validate nvmolkit vs RDKit (default)"
    )
    parser.add_argument("--no_validate", action="store_false", dest="validate", help="Skip validation checks")
    parser.set_defaults(validate=True)

    args = parser.parse_args()

    if not args.smiles and not args.pickle:
        print("Error: Either --smiles or --pickle is required")
        sys.exit(1)

    if args.smiles and args.pickle:
        print("Error: Cannot specify both --smiles and --pickle")
        sys.exit(1)

    if args.config and args.smarts:
        print("Error: --smarts cannot be used with --config")
        sys.exit(1)

    if not args.config and not args.smarts:
        print("Error: --smarts is required unless --config is provided")
        sys.exit(1)

    input_file = args.smiles or args.pickle
    input_type = "pickle" if args.pickle else "smiles"

    sanitize_value = args.sanitize if args.smiles else "N/A"

    if args.num_gpus <= 0:
        print("Error: --num_gpus must be >= 1")
        sys.exit(1)

    if args.autotune and args.config:
        print("Error: --autotune is only supported in single-config mode (use --smarts, not --config)")
        sys.exit(1)
    if args.autotune_load and args.config:
        print(
            "Error: --autotune_load is only supported in single-config mode "
            "(it would override batch_size/workers/prep_threads/gpuIds on every row)"
        )
        sys.exit(1)
    if args.autotune and args.no_nvmolkit:
        print("Error: --autotune requires nvmolkit; remove --no_nvmolkit")
        sys.exit(1)
    if args.autotune_save and not args.autotune:
        print("Error: --autotune_save requires --autotune")
        sys.exit(1)
    if args.autotune and args.autotune_load:
        print("Error: --autotune and --autotune_load are mutually exclusive")
        sys.exit(1)

    print("\nConfiguration:")
    print(f"  Input file: {input_file} ({input_type})")
    print(f"  Sanitize: {sanitize_value}")
    print(f"  Max molecules: {args.num_mols if args.num_mols > 0 else 'all'}")
    if args.num_mols > 0:
        print(f"  Sampling seed: {args.seed}")
    print(f"  Max matches: {args.max_matches if args.max_matches > 0 else 'all'}")
    print(f"  Runs: {args.runs}")
    print(f"  Warmup: {args.warmup}")
    print(f"  Validate: {args.validate}")
    print(f"  Run nvmolkit: {not args.no_nvmolkit}")
    print(f"  Run RDKit: {not args.no_rdkit}")
    if not args.no_rdkit:
        print(f"  RDKit match mode: {args.rdkit_match_mode}")
        print(f"  RDKit threads: {args.rdkit_threads}")
    if args.config:
        print(f"  Config dataframe: {args.config}")
    else:
        print(f"  SMARTS file: {args.smarts}")
        print(f"  Mode: {args.mode}")
        if not args.no_nvmolkit:
            print(f"  nvmolkit config:")
            print(f"    batch_size: {args.batch_size}")
            print(f"    num_gpus: {args.num_gpus}")
            print(f"    workers: {args.workers if args.workers >= 0 else 'auto'}")
            print(f"    prep_threads: {args.prep_threads if args.prep_threads >= 0 else 'auto'}")

    print("\nLoading molecules...")
    if args.pickle:
        mols = load_pickle(args.pickle, args.num_mols, seed=args.seed)
    else:
        mols = load_smiles(args.smiles, args.num_mols, args.sanitize, seed=args.seed)

    if len(mols) == 0:
        print("Error: No valid molecules loaded")
        sys.exit(1)

    if args.config:
        config_rows = _load_config_dataframe(args.config)
    else:
        config_rows = [
            {
                "smarts": args.smarts,
                "batch_size": args.batch_size,
                "workers": args.workers,
                "prep_threads": args.prep_threads,
                "mode": args.mode,
                "num_gpus": args.num_gpus,
            }
        ]

    smarts_cache: dict[str, tuple[list[Chem.Mol], list[str]]] = {}
    csv_rows = []

    for config_row in config_rows:
        smarts_path = config_row["smarts"]
        mode = config_row["mode"]

        print("\nRun configuration:")
        print(f"  SMARTS file: {smarts_path}")
        print(f"  Mode: {mode}")
        if not args.no_nvmolkit:
            print(f"  nvmolkit config:")
            print(f"    batch_size: {config_row['batch_size']}")
            print(f"    num_gpus: {config_row['num_gpus']}")
            print(f"    workers: {config_row['workers'] if config_row['workers'] >= 0 else 'auto'}")
            print(f"    prep_threads: {config_row['prep_threads'] if config_row['prep_threads'] >= 0 else 'auto'}")

        if smarts_path in smarts_cache:
            queries, _ = smarts_cache[smarts_path]
        else:
            print("\nLoading SMARTS patterns...")
            queries, smarts_list = load_smarts(smarts_path)
            if len(queries) == 0:
                print("Error: No valid SMARTS patterns loaded from file")
                sys.exit(1)
            smarts_cache[smarts_path] = (queries, smarts_list)

        num_patterns = len(queries)
        print(f"\nBenchmarking substructure search ({mode}): {len(mols)} molecules × {num_patterns} patterns")
        print("=" * 70)

        results = {}
        ran_nvmolkit = False
        torch_module = None

        if not args.no_nvmolkit:
            try:
                import torch

                api_for_mode = {
                    "hasSubstructMatch": hasSubstructMatch,
                    "countSubstructMatches": countSubstructMatches,
                    "getSubstructMatches": getSubstructMatches,
                }[mode]
                gpu_ids = list(range(config_row["num_gpus"]))

                if args.autotune_load:
                    print(f"\nLoading tuned SubstructSearchConfig from {args.autotune_load}...")
                    loaded = nv_autotune.load(args.autotune_load)
                    if not isinstance(loaded, SubstructSearchConfig):
                        print(
                            f"Error: {args.autotune_load} contains {type(loaded).__name__}, "
                            "expected SubstructSearchConfig"
                        )
                        sys.exit(1)
                    config = loaded
                    if args.max_matches > 0:
                        config.maxMatches = args.max_matches
                    if not config.gpuIds:
                        config.gpuIds = gpu_ids
                    print(
                        f"  Loaded: batchSize={config.batchSize}, workerThreads={config.workerThreads}, "
                        f"preprocessingThreads={config.preprocessingThreads}, gpuIds={list(config.gpuIds)}"
                    )
                elif args.autotune:
                    if not OPTUNA_AVAILABLE:
                        print(
                            "Error: --autotune requires the optional 'optuna' dependency. "
                            "Install with `pip install nvmolkit[autotune]` or `conda install -c conda-forge optuna`."
                        )
                        sys.exit(1)
                    print(
                        f"\nAutotuning SubstructSearchConfig (mode={mode}, n_trials={args.autotune_trials}, "
                        f"per-trial target={args.autotune_time_budget:.1f}s)..."
                    )
                    explicit_calibration = None
                    if args.autotune_calibration_size > 0:
                        rng = random.Random(args.autotune_seed)
                        size = min(args.autotune_calibration_size, len(mols))
                        explicit_calibration = rng.sample(range(len(mols)), size)
                    tune_result = nv_autotune.tune_substructure(
                        mols,
                        queries,
                        api=api_for_mode,
                        maxMatches=args.max_matches,
                        gpuIds=gpu_ids,
                        n_trials=args.autotune_trials,
                        target_seconds_per_trial=args.autotune_time_budget,
                        calibration_set=explicit_calibration,
                        seed=args.autotune_seed,
                        verbose=True,
                    )
                    config = tune_result.best_config
                    print(
                        f"  Best: batchSize={config.batchSize}, workerThreads={config.workerThreads}, "
                        f"preprocessingThreads={config.preprocessingThreads} "
                        f"(throughput={tune_result.best_throughput:.2f} pairs/s, "
                        f"trials_run={tune_result.n_trials_run}, "
                        f"calibration_size={tune_result.calibration_size})"
                    )
                    if args.autotune_save:
                        nv_autotune.save(config, args.autotune_save)
                        print(f"  Saved tuned config to {args.autotune_save}")
                else:
                    config = SubstructSearchConfig()
                    config.batchSize = config_row["batch_size"]
                    config.workerThreads = config_row["workers"]
                    config.preprocessingThreads = config_row["prep_threads"]
                    config.gpuIds = gpu_ids
                    if args.max_matches > 0:
                        config.maxMatches = args.max_matches

                ran_nvmolkit = True
                torch_module = torch
                torch.cuda.cudart().cudaProfilerStart()

                if args.warmup:
                    print("\nWarming up nvmolkit...")
                    warmup_mols = mols[:10]
                    with nvtx.annotate("nvmolkit_warmup", color="purple"):
                        if mode == "hasSubstructMatch":
                            hasSubstructMatch(warmup_mols, queries, config)
                        elif mode == "countSubstructMatches":
                            countSubstructMatches(warmup_mols, queries, config)
                        else:
                            getSubstructMatches(warmup_mols, queries, config)
                        torch.cuda.synchronize()

                print("Running nvmolkit GPU benchmark...")
                nvmolkit_avg, nvmolkit_std, nvmolkit_results = bench_nvmolkit(mols, queries, args.runs, mode, config)
                print(f"  nvmolkit:        {nvmolkit_avg:10.2f} ms (± {nvmolkit_std:.2f} ms)")
                results["nvmolkit"] = (nvmolkit_avg, nvmolkit_std, nvmolkit_results)
                torch.cuda.cudart().cudaProfilerStop()

            except ImportError as e:
                print(f"  nvmolkit: SKIPPED (import error: {e})")

        if not args.no_rdkit:
            if args.rdkit_match_mode == "substructlib":
                print("\nRunning RDKit SubstructLibrary benchmark...")
                rdkit_avg, rdkit_std, rdkit_results = bench_rdkit_substructlib(
                    mols, queries, args.runs, mode, args.max_matches, args.rdkit_threads
                )
            else:
                print("\nRunning RDKit SubstructMatch benchmark...")
                rdkit_avg, rdkit_std, rdkit_results = bench_rdkit_substruct(
                    mols, queries, args.runs, mode, args.max_matches, args.rdkit_threads
                )
            print(f"  RDKit:           {rdkit_avg:10.2f} ms (± {rdkit_std:.2f} ms)")
            results["rdkit"] = (rdkit_avg, rdkit_std, rdkit_results)

        print("\n" + "=" * 70)
        print("Summary:")

        if not results:
            print("  No benchmarks were run!")
            sys.exit(1)

        baseline = None
        if "rdkit" in results:
            baseline = ("RDKit", results["rdkit"][0])

        for name, (avg_ms, std_ms, _) in results.items():
            speedup_str = ""
            if baseline and name != "rdkit":
                speedup = baseline[1] / avg_ms if avg_ms > 0 else 0
                speedup_str = f", {speedup:.1f}x vs {baseline[0]}"
            print(f"  {name:20s}: {avg_ms:10.2f} ms (± {std_ms:.2f} ms){speedup_str}")

        if args.validate and "nvmolkit" in results and "rdkit" in results:
            print("\nValidation:")
            nvmolkit_data = results["nvmolkit"][2]
            rdkit_data = results["rdkit"][2]

            if mode == "hasSubstructMatch":
                matches = 0
                total = 0
                for t in range(len(mols)):
                    for q in range(len(queries)):
                        nv_match = bool(nvmolkit_data[t][q])
                        rd_match = rdkit_data[t][q]
                        if nv_match == rd_match:
                            matches += 1
                        total += 1
                pct = 100.0 * matches / total if total > 0 else 0
                print(f"  Boolean match agreement: {matches}/{total} ({pct:.1f}%)")
            elif mode == "countSubstructMatches":
                matches = 0
                total = 0
                for t in range(len(mols)):
                    for q in range(len(queries)):
                        nv_count = int(nvmolkit_data[t][q])
                        rd_count = int(rdkit_data[t][q])
                        if nv_count == rd_count:
                            matches += 1
                        total += 1
                pct = 100.0 * matches / total if total > 0 else 0
                print(f"  Count agreement: {matches}/{total} ({pct:.1f}%)")
            else:
                matches = 0
                total = 0
                for t in range(len(mols)):
                    for q in range(len(queries)):
                        nv_matches = set(tuple(m) for m in nvmolkit_data[t][q])
                        rd_matches = set(rdkit_data[t][q])
                        if nv_matches == rd_matches:
                            matches += 1
                        total += 1
                pct = 100.0 * matches / total if total > 0 else 0
                print(f"  Full match agreement: {matches}/{total} ({pct:.1f}%)")

        if ran_nvmolkit:
            applied_batch_size = int(config.batchSize)
            applied_workers = int(config.workerThreads)
            applied_prep_threads = int(config.preprocessingThreads)
            applied_num_gpus = len(list(config.gpuIds)) if config.gpuIds else config_row["num_gpus"]
        else:
            applied_batch_size = config_row["batch_size"]
            applied_workers = config_row["workers"]
            applied_prep_threads = config_row["prep_threads"]
            applied_num_gpus = config_row["num_gpus"]

        if args.autotune:
            config_source = "autotuned"
        elif args.autotune_load:
            config_source = "loaded"
        else:
            config_source = "cli"

        for name, (avg_ms, std_ms, _) in results.items():
            batch_size = applied_batch_size if name == "nvmolkit" else "N/A"
            workers = applied_workers if name == "nvmolkit" else "N/A"
            prep_threads = applied_prep_threads if name == "nvmolkit" else "N/A"
            num_gpus = applied_num_gpus if name == "nvmolkit" else config_row["num_gpus"]
            nvmolkit_config_source = config_source if name == "nvmolkit" else "N/A"
            rdkit_threads = args.rdkit_threads if name == "rdkit" else "N/A"
            rdkit_match_mode = args.rdkit_match_mode if name == "rdkit" else "N/A"
            csv_rows.append(
                (
                    name,
                    mode,
                    smarts_path,
                    input_file,
                    input_type,
                    sanitize_value,
                    len(mols),
                    num_patterns,
                    args.max_matches,
                    batch_size,
                    num_gpus,
                    workers,
                    prep_threads,
                    nvmolkit_config_source,
                    rdkit_threads,
                    rdkit_match_mode,
                    avg_ms,
                    std_ms,
                )
            )

        if ran_nvmolkit:
            torch_module.cuda.synchronize()
            torch_module.cuda.empty_cache()
            torch_module.cuda.ipc_collect()
        gc.collect()

    print("\n\nCSV Results:")
    print(
        "method,mode,smarts,input_file,input_type,sanitize,num_mols,num_patterns,"
        "max_matches,batch_size,num_gpus,workers,prep_threads,nvmolkit_config_source,"
        "rdkit_threads,rdkit_match_mode,time_ms,std_ms"
    )
    for row in csv_rows:
        (
            name,
            mode,
            smarts_path,
            input_file,
            input_type,
            sanitize,
            num_mols,
            num_patterns,
            max_matches,
            batch_size,
            num_gpus,
            workers,
            prep_threads,
            nvmolkit_config_source,
            rdkit_threads,
            rdkit_match_mode,
            avg_ms,
            std_ms,
        ) = row
        print(
            f"{name},{mode},{smarts_path},{input_file},{input_type},{sanitize},"
            f"{num_mols},{num_patterns},{max_matches},{batch_size},{num_gpus},{workers},{prep_threads},"
            f"{nvmolkit_config_source},{rdkit_threads},{rdkit_match_mode},{avg_ms:.2f},{std_ms:.2f}"
        )


if __name__ == "__main__":
    main()
