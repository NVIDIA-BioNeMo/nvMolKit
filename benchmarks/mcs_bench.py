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

"""Maximum common substructure benchmark comparing nvMolKit against RDKit.

Usage:
    python mcs_bench.py --smiles <smiles_file>
    python mcs_bench.py --smiles <smiles_file> --num_pairs 10000 --rdkit_threads 1 8
    python mcs_bench.py --smiles <smiles_file> --config <config.csv>
"""

import argparse
import gc
import random
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Callable

import nvtx
import pandas as pd
from bench_utils import (
    Deadline,
    add_backend_selection_args,
    add_rdkit_max_seconds_arg,
    load_pickle,
    load_smiles,
    print_csv_rows,
    throughput_per_s,
    time_it_bounded,
    write_csv_rows,
)
from bench_utils import (
    time_it as _time_it,
)
from rdkit import Chem
from rdkit.Chem import rdFMCS

from nvmolkit import autotune as nv_autotune
from nvmolkit.mcs import MCSConfig, findMCS

OPTUNA_AVAILABLE = nv_autotune.is_available()

_worker_mols: list[Chem.Mol] | None = None
_worker_params: rdFMCS.MCSParameters | None = None


def time_it(func: Callable, runs: int = 1, gpu_sync: bool = False) -> tuple[float, float]:
    """Time a function and return ``(mean_ms, std_ms)``."""
    result = _time_it(func, runs=runs, warmups=0, gpu_sync=gpu_sync)
    return result.mean_ms, result.std_ms


def _load_config_dataframe(config_path: str) -> list[dict]:
    suffix = Path(config_path).suffix.lower()
    if suffix == ".csv":
        dataframe = pd.read_csv(config_path)
    elif suffix in {".pkl", ".pickle"}:
        dataframe = pd.read_pickle(config_path)
    elif suffix == ".parquet":
        dataframe = pd.read_parquet(config_path)
    else:
        raise ValueError(f"unsupported config format {suffix!r}; expected .csv, .pkl, .pickle, or .parquet")
    if dataframe.empty:
        raise ValueError("config dataframe must contain at least one row")
    return dataframe.to_dict("records")


def _row_value(row: dict, key: str, default):
    value = row.get(key, default)
    return default if pd.isna(value) else value


def _bool_value(value) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    if value in {0, 1}:
        return bool(value)
    raise ValueError(f"expected a boolean value, got {value!r}")


def _normalize_config_row(row: dict, args: argparse.Namespace) -> dict:
    config_row = {
        "num_pairs": int(_row_value(row, "num_pairs", args.num_pairs)),
        "batch_size": int(_row_value(row, "batch_size", args.batch_size)),
        "workers": int(_row_value(row, "workers", args.workers)),
        "prep_threads": int(_row_value(row, "prep_threads", args.prep_threads)),
        "executors_per_runner": int(_row_value(row, "executors_per_runner", args.executors_per_runner)),
        "num_gpus": int(_row_value(row, "num_gpus", args.num_gpus)),
        "atom_compare": str(_row_value(row, "atom_compare", args.atom_compare)).lower(),
        "bond_compare": str(_row_value(row, "bond_compare", args.bond_compare)).lower(),
        "match_valences": _bool_value(_row_value(row, "match_valences", args.match_valences)),
        "match_formal_charge": _bool_value(_row_value(row, "match_formal_charge", args.match_formal_charge)),
        "ring_matches_ring_only": _bool_value(_row_value(row, "ring_matches_ring_only", args.ring_matches_ring_only)),
        "timeout_seconds": int(_row_value(row, "timeout_seconds", args.timeout_seconds)),
    }
    if config_row["num_pairs"] <= 0:
        raise ValueError("num_pairs must be positive")
    if config_row["executors_per_runner"] != -1 and not 1 <= config_row["executors_per_runner"] <= 8:
        raise ValueError("executors_per_runner must be -1 or between 1 and 8")
    if config_row["num_gpus"] <= 0:
        raise ValueError("num_gpus must be positive")
    if config_row["atom_compare"] not in {"any", "elements", "isotopes"}:
        raise ValueError("atom_compare must be 'any', 'elements', or 'isotopes'")
    if config_row["bond_compare"] not in {"any", "order", "order_exact"}:
        raise ValueError("bond_compare must be 'any', 'order', or 'order_exact'")
    if config_row["timeout_seconds"] < 0:
        raise ValueError("timeout_seconds must be non-negative")
    return config_row


def _filter_molecules(mols: list[Chem.Mol], max_atoms: int, max_bonds: int) -> list[Chem.Mol]:
    filtered = [
        mol
        for mol in mols
        if mol is not None
        and (max_atoms <= 0 or mol.GetNumAtoms() <= max_atoms)
        and (max_bonds <= 0 or mol.GetNumBonds() <= max_bonds)
    ]
    if not filtered:
        raise ValueError("no molecules remain after max atom/bond filtering")
    return filtered


def sample_pairs(num_mols: int, num_pairs: int, seed: int) -> list[tuple[int, int]]:
    """Randomly sample unique pairs of distinct molecule indices."""
    if num_mols < 2:
        raise ValueError("at least two molecules are required for pair sampling")
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive")

    max_pairs = num_mols * (num_mols - 1) // 2
    num_pairs = min(num_pairs, max_pairs)
    if num_pairs == max_pairs:
        return [(i, j) for i in range(num_mols) for j in range(i + 1, num_mols)]

    rng = random.Random(seed)
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < num_pairs:
        idx_a, idx_b = rng.sample(range(num_mols), 2)
        pairs.add((min(idx_a, idx_b), max(idx_a, idx_b)))
    return sorted(pairs)


def _rdkit_params(config_row: dict) -> rdFMCS.MCSParameters:
    params = rdFMCS.MCSParameters()
    params.MaximizeBonds = True
    params.Timeout = config_row["timeout_seconds"]
    params.AtomCompareParameters.MatchValences = config_row["match_valences"]
    params.AtomCompareParameters.MatchFormalCharge = config_row["match_formal_charge"]
    params.AtomCompareParameters.RingMatchesRingOnly = config_row["ring_matches_ring_only"]
    params.BondCompareParameters.RingMatchesRingOnly = config_row["ring_matches_ring_only"]
    params.AtomTyper = {
        "any": rdFMCS.AtomCompare.CompareAny,
        "elements": rdFMCS.AtomCompare.CompareElements,
        "isotopes": rdFMCS.AtomCompare.CompareIsotopes,
    }[config_row["atom_compare"]]
    params.BondTyper = {
        "any": rdFMCS.BondCompare.CompareAny,
        "order": rdFMCS.BondCompare.CompareOrder,
        "order_exact": rdFMCS.BondCompare.CompareOrderExact,
    }[config_row["bond_compare"]]
    return params


def _rdkit_worker_init(mol_binaries: list[bytes], params: rdFMCS.MCSParameters) -> None:
    global _worker_mols, _worker_params
    _worker_mols = [Chem.Mol(binary) for binary in mol_binaries]
    _worker_params = params


def _rdkit_worker_pair(pair: tuple[int, int]) -> tuple[int, int]:
    if _worker_mols is None or _worker_params is None:
        raise RuntimeError("RDKit MCS worker was not initialized")
    result = rdFMCS.FindMCS([_worker_mols[pair[0]], _worker_mols[pair[1]]], _worker_params)
    return int(result.numAtoms), int(result.numBonds)


@nvtx.annotate("bench_rdkit_mcs", color="green")
def bench_rdkit_mcs(
    mols: list[Chem.Mol],
    pairs: list[tuple[int, int]],
    runs: int,
    threads: int,
    max_seconds: float,
    config_row: dict,
) -> tuple[float, float, list[tuple[int, int]], int]:
    """Benchmark RDKit FindMCS."""
    params = _rdkit_params(config_row)
    results_data: list[tuple[int, int]] = []
    pairs_done = 0

    if threads > 1:
        mol_binaries = [mol.ToBinary() for mol in mols]
        chunksize = 1 if max_seconds > 0 else max(1, len(pairs) // (threads * 8))

        @nvtx.annotate("mcs_rdkit_run_mp", color="yellow")
        def run(deadline: Deadline) -> None:
            nonlocal pairs_done, results_data
            results_data = []
            pairs_done = 0
            with Pool(threads, initializer=_rdkit_worker_init, initargs=(mol_binaries, params)) as pool:
                for result in pool.imap(_rdkit_worker_pair, pairs, chunksize=chunksize):
                    results_data.append(result)
                    pairs_done += 1
                    if deadline.expired():
                        break

    else:

        @nvtx.annotate("mcs_rdkit_run", color="yellow")
        def run(deadline: Deadline) -> None:
            nonlocal pairs_done, results_data
            results_data = []
            pairs_done = 0
            for idx_a, idx_b in pairs:
                result = rdFMCS.FindMCS([mols[idx_a], mols[idx_b]], params)
                results_data.append((int(result.numAtoms), int(result.numBonds)))
                pairs_done += 1
                if deadline.expired():
                    break

    avg_ms, std_ms, measured_pairs = time_it_bounded(
        run,
        runs,
        max_seconds,
        lambda: pairs_done,
        len(pairs),
    )
    return avg_ms, std_ms, results_data, measured_pairs


@nvtx.annotate("bench_nvmolkit_mcs", color="red")
def bench_nvmolkit_mcs(
    mols: list[Chem.Mol],
    pairs: list[tuple[int, int]],
    runs: int,
    config_row: dict,
    config: MCSConfig,
) -> tuple[float, float, object]:
    """Benchmark nvMolKit GPU MCS."""
    results_data: object = None

    @nvtx.annotate("mcs_nvmolkit_run", color="orange")
    def run() -> None:
        nonlocal results_data
        results_data = findMCS(
            mols,
            mode="pairs",
            pairs=pairs,
            atom_compare=config_row["atom_compare"],
            bond_compare=config_row["bond_compare"],
            match_valences=config_row["match_valences"],
            match_formal_charge=config_row["match_formal_charge"],
            ring_matches_ring_only=config_row["ring_matches_ring_only"],
            require_gpu=True,
            timeout_seconds=config_row["timeout_seconds"],
            config=config,
        )

    avg_ms, std_ms = time_it(run, runs, gpu_sync=True)
    return avg_ms, std_ms, results_data


def _validate_results(nvmolkit_result, rdkit_results: list[tuple[int, int]]) -> None:
    if len(rdkit_results) != len(nvmolkit_result):
        raise ValueError("validation requires a complete RDKit result set")
    for pair_idx, (rdkit_atoms, rdkit_bonds) in enumerate(rdkit_results):
        nv_atoms = int(nvmolkit_result.num_atoms[pair_idx])
        nv_bonds = int(nvmolkit_result.num_bonds[pair_idx])
        if (nv_atoms, nv_bonds) != (rdkit_atoms, rdkit_bonds):
            pair = nvmolkit_result.pairs[pair_idx]
            raise AssertionError(
                f"pair {pair_idx} {pair}: nvMolKit={nv_atoms}/{nv_bonds}, RDKit={rdkit_atoms}/{rdkit_bonds}"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MCS benchmark: nvMolKit GPU MCS vs RDKit FindMCS")
    parser.add_argument("--smiles", "-s", help="Path to SMILES file with molecules")
    parser.add_argument("--pickle", help="Path to pickled molecules file (alternative to --smiles)")
    parser.add_argument(
        "--config",
        help=(
            "Path to config dataframe (.csv/.pkl/.pickle/.parquet) with optional columns: num_pairs, "
            "batch_size, workers, prep_threads, executors_per_runner, num_gpus, atom_compare, "
            "bond_compare, match_valences, match_formal_charge, ring_matches_ring_only, timeout_seconds"
        ),
    )
    parser.add_argument("--num_mols", "-n", type=int, default=0, help="Max number of molecules (default: all)")
    parser.add_argument("--num_pairs", type=int, default=100, help="Number of random molecule pairs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for molecule and pair sampling")
    sanitize_group = parser.add_mutually_exclusive_group()
    sanitize_group.add_argument("--sanitize", action="store_true", dest="sanitize", default=True)
    sanitize_group.add_argument("--no_sanitize", action="store_false", dest="sanitize")
    parser.add_argument("--max_atoms", type=int, default=128, help="Maximum atoms per molecule; 0 disables")
    parser.add_argument("--max_bonds", type=int, default=128, help="Maximum bonds per molecule; 0 disables")
    parser.add_argument("--runs", "-r", type=int, default=1, help="Number of timing runs")
    parser.add_argument("--warmup", action="store_true", dest="warmup", help="Perform warmup run (default)")
    parser.add_argument("--no_warmup", action="store_false", dest="warmup", help="Skip warmup run")
    parser.set_defaults(warmup=True)
    add_backend_selection_args(parser)
    parser.add_argument(
        "--rdkit_threads",
        type=int,
        nargs="+",
        default=[1],
        help="RDKit process count(s) to benchmark (default: 1)",
    )
    add_rdkit_max_seconds_arg(
        parser,
        extra_help="The RDKit MCS loop stops at the next completed molecule-pair boundary.",
    )
    parser.add_argument("--batch_size", "-b", type=int, default=0, help="nvMolKit batch size (default: all pairs)")
    parser.add_argument("--workers", type=int, default=-1, help="nvMolKit GPU worker threads per GPU (-1 = auto)")
    parser.add_argument("--prep_threads", type=int, default=-1, help="nvMolKit preprocessing threads (-1 = auto)")
    parser.add_argument(
        "--executors_per_runner", type=int, default=-1, help="nvMolKit executors per runner (-1 = auto)"
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use (default: 1)")
    parser.add_argument("--atom_compare", choices=["any", "elements", "isotopes"], default="elements")
    parser.add_argument("--bond_compare", choices=["any", "order", "order_exact"], default="order")
    parser.add_argument("--match_valences", action="store_true")
    parser.add_argument("--match_formal_charge", action="store_true")
    parser.add_argument("--ring_matches_ring_only", action="store_true")
    parser.add_argument("--timeout_seconds", type=int, default=0)
    parser.add_argument(
        "--autotune",
        action="store_true",
        help=(
            "Tune nvMolKit MCSConfig before timing. Requires the [autotune] extra (optuna). Single-config mode only."
        ),
    )
    parser.add_argument(
        "--autotune_save",
        default=None,
        help="Path to save the tuned MCSConfig as JSON (only with --autotune)",
    )
    parser.add_argument(
        "--autotune_load",
        default=None,
        help=(
            "Path to a previously saved MCSConfig JSON. Overrides batch_size, workers, "
            "prep_threads, executors_per_runner, and num_gpus. Single-config mode only."
        ),
    )
    parser.add_argument("--autotune_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument(
        "--autotune_time_budget",
        type=float,
        default=10.0,
        help="Target wall-clock seconds per Optuna trial",
    )
    parser.add_argument(
        "--autotune_calibration_size",
        type=int,
        default=0,
        help="Number of pairs per autotune trial; 0 selects automatically",
    )
    parser.add_argument("--autotune_seed", type=int, default=42, help="Seed for the Optuna sampler")
    parser.add_argument("--validate", action="store_true", dest="validate", help="Validate against RDKit (default)")
    parser.add_argument("--no_validate", action="store_false", dest="validate", help="Skip validation")
    parser.set_defaults(validate=True)
    parser.add_argument("--output", "-o", help="Optional path to write CSV results")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.smiles and not args.pickle:
        print("Error: Either --smiles or --pickle is required")
        sys.exit(1)
    if args.smiles and args.pickle:
        print("Error: Cannot specify both --smiles and --pickle")
        sys.exit(1)
    if args.no_rdkit and args.no_nvmolkit:
        print("Error: cannot disable both nvMolKit and RDKit")
        sys.exit(1)
    if args.runs <= 0:
        print("Error: --runs must be positive")
        sys.exit(1)
    if any(threads <= 0 for threads in args.rdkit_threads):
        print("Error: --rdkit_threads values must be positive")
        sys.exit(1)
    if args.autotune and args.config:
        print("Error: --autotune is only supported in single-config mode")
        sys.exit(1)
    if args.autotune_load and args.config:
        print("Error: --autotune_load is only supported in single-config mode")
        sys.exit(1)
    if args.autotune and args.no_nvmolkit:
        print("Error: --autotune requires nvMolKit; remove --no_nvmolkit")
        sys.exit(1)
    if args.autotune_save and not args.autotune:
        print("Error: --autotune_save requires --autotune")
        sys.exit(1)
    if args.autotune and args.autotune_load:
        print("Error: --autotune and --autotune_load are mutually exclusive")
        sys.exit(1)

    input_file = args.smiles or args.pickle
    input_type = "pickle" if args.pickle else "smiles"
    sanitize_value = args.sanitize if args.smiles else "N/A"

    print("\nConfiguration:")
    print(f"  Input file: {input_file} ({input_type})")
    print(f"  Sanitize: {sanitize_value}")
    print(f"  Max molecules: {args.num_mols if args.num_mols > 0 else 'all'}")
    print(f"  Runs: {args.runs}")
    print(f"  Warmup: {args.warmup}")
    print(f"  Validate: {args.validate}")
    print(f"  Run nvMolKit: {not args.no_nvmolkit}")
    print(f"  Run RDKit: {not args.no_rdkit}")
    if not args.no_rdkit:
        print(f"  RDKit process counts: {args.rdkit_threads}")
    if args.config:
        print(f"  Config dataframe: {args.config}")

    print("\nLoading molecules...")
    if args.pickle:
        mols = load_pickle(args.pickle, args.num_mols, seed=args.seed)
    else:
        mols = load_smiles(args.smiles, args.num_mols, args.sanitize, seed=args.seed)
    try:
        mols = _filter_molecules(mols, args.max_atoms, args.max_bonds)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    if len(mols) < 2:
        print("Error: Need at least 2 valid molecules for pairwise MCS")
        sys.exit(1)

    try:
        raw_rows = _load_config_dataframe(args.config) if args.config else [{}]
        config_rows = [_normalize_config_row(row, args) for row in raw_rows]
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    csv_rows: list[dict[str, object]] = []
    for config_index, config_row in enumerate(config_rows):
        pairs = sample_pairs(len(mols), config_row["num_pairs"], args.seed)
        print("\nRun configuration:")
        for key, value in config_row.items():
            print(f"  {key}: {value}")
        print(f"  sampled_pairs: {len(pairs)}")
        print(f"\nBenchmarking MCS: {len(pairs)} pairs")
        print("=" * 70)

        results = {}
        ran_nvmolkit = False
        torch_module = None
        config_source = "dataframe" if args.config else "cli"
        applied_config = MCSConfig(
            batchSize=config_row["batch_size"],
            workerThreads=config_row["workers"],
            preprocessingThreads=config_row["prep_threads"],
            executorsPerRunner=config_row["executors_per_runner"],
            gpuIds=list(range(config_row["num_gpus"])),
        )
        if not args.no_nvmolkit:
            try:
                import torch

                gpu_ids = list(range(config_row["num_gpus"]))
                if args.autotune_load:
                    print(f"\nLoading tuned MCSConfig from {args.autotune_load}...")
                    loaded = nv_autotune.load(args.autotune_load)
                    if not isinstance(loaded, MCSConfig):
                        print(f"Error: {args.autotune_load} contains {type(loaded).__name__}, expected MCSConfig")
                        sys.exit(1)
                    applied_config = loaded
                    if not applied_config.gpuIds:
                        applied_config.gpuIds = gpu_ids
                    config_source = "loaded"
                    print(f"  Loaded: {applied_config.to_dict()}")
                elif args.autotune:
                    if not OPTUNA_AVAILABLE:
                        print(
                            "Error: --autotune requires the optional 'optuna' dependency. "
                            "Install with `pip install nvmolkit[autotune]` or `conda install -c conda-forge optuna`."
                        )
                        sys.exit(1)
                    explicit_calibration = None
                    if args.autotune_calibration_size > 0:
                        rng = random.Random(args.autotune_seed)
                        size = min(args.autotune_calibration_size, len(pairs))
                        explicit_calibration = rng.sample(range(len(pairs)), size)
                    print(
                        f"\nAutotuning MCSConfig (n_trials={args.autotune_trials}, "
                        f"per-trial target={args.autotune_time_budget:.1f}s)..."
                    )
                    tune_result = nv_autotune.tune_mcs(
                        mols,
                        pairs,
                        atom_compare=config_row["atom_compare"],
                        bond_compare=config_row["bond_compare"],
                        match_valences=config_row["match_valences"],
                        match_formal_charge=config_row["match_formal_charge"],
                        ring_matches_ring_only=config_row["ring_matches_ring_only"],
                        timeout_seconds=config_row["timeout_seconds"],
                        gpuIds=gpu_ids,
                        calibration_set=explicit_calibration,
                        n_trials=args.autotune_trials,
                        target_seconds_per_trial=args.autotune_time_budget,
                        seed=args.autotune_seed,
                        verbose=True,
                    )
                    applied_config = tune_result.best_config
                    config_source = "autotuned"
                    print(
                        f"  Best: {applied_config.to_dict()} "
                        f"(throughput={tune_result.best_throughput:.2f} pairs/s, "
                        f"trials_run={tune_result.n_trials_run}, calibration_size={tune_result.calibration_size})"
                    )
                    if args.autotune_save:
                        nv_autotune.save(applied_config, args.autotune_save)
                        print(f"  Saved tuned config to {args.autotune_save}")

                if args.warmup:
                    print("\nWarming up nvMolKit...")
                    warmup_pairs = pairs[: min(16, len(pairs))]
                    bench_nvmolkit_mcs(mols, warmup_pairs, 1, config_row, applied_config)

                print("\nRunning nvMolKit GPU MCS benchmark...")
                torch.cuda.cudart().cudaProfilerStart()
                nv_avg, nv_std, nv_results = bench_nvmolkit_mcs(
                    mols,
                    pairs,
                    args.runs,
                    config_row,
                    applied_config,
                )
                torch.cuda.cudart().cudaProfilerStop()
                print(f"  nvMolKit: {nv_avg:10.2f} ms (+/- {nv_std:.2f} ms)")
                results["nvmolkit"] = (nv_avg, nv_std, nv_results, len(pairs))
                ran_nvmolkit = True
                torch_module = torch
            except ImportError as exc:
                print(f"  nvMolKit: SKIPPED (import error: {exc})")

        rdkit_variants: list[tuple[str, int]] = []
        if not args.no_rdkit:
            for rdkit_threads in args.rdkit_threads:
                name = f"rdkit_t{rdkit_threads}"
                print(f"\nRunning RDKit FindMCS benchmark (processes={rdkit_threads})...")
                rd_avg, rd_std, rd_results, rd_pairs = bench_rdkit_mcs(
                    mols,
                    pairs,
                    args.runs,
                    rdkit_threads,
                    args.rdkit_max_seconds,
                    config_row,
                )
                print(f"  {name}: {rd_avg:10.2f} ms (+/- {rd_std:.2f} ms), {rd_pairs}/{len(pairs)} pairs")
                results[name] = (rd_avg, rd_std, rd_results, rd_pairs)
                rdkit_variants.append((name, rdkit_threads))

        if not results:
            print("  No benchmarks were run!")
            sys.exit(1)

        baseline_name = None
        baseline_throughput = 0.0
        for name, _threads in rdkit_variants:
            avg_ms, _std_ms, _data, pairs_done = results[name]
            throughput = throughput_per_s(pairs_done, avg_ms)
            if throughput > baseline_throughput:
                baseline_name = name
                baseline_throughput = throughput

        print("\n" + "=" * 70)
        print("Summary:")
        for name, (avg_ms, std_ms, _data, pairs_done) in results.items():
            throughput = throughput_per_s(pairs_done, avg_ms)
            speedup = ""
            if name == "nvmolkit" and baseline_throughput > 0:
                speedup = f", {throughput / baseline_throughput:.1f}x vs {baseline_name}"
            print(f"  {name:20s}: {avg_ms:10.2f} ms (+/- {std_ms:.2f} ms), {throughput:,.0f} pairs/s{speedup}")

        validation_name = next(
            (name for name, _threads in rdkit_variants if results[name][3] == len(pairs)),
            None,
        )
        if args.validate and "nvmolkit" in results and rdkit_variants:
            print("\nValidation:")
            if validation_name is None:
                print("  Skipped: every RDKit run hit the max-seconds budget")
            else:
                _validate_results(results["nvmolkit"][2], results[validation_name][2])
                print(f"  Atom/bond counts agree for all {len(pairs)} pairs against {validation_name}")

        rdkit_thread_map = dict(rdkit_variants)
        for name, (avg_ms, std_ms, _data, pairs_done) in results.items():
            is_nvmolkit = name == "nvmolkit"
            is_rdkit = name in rdkit_thread_map
            throughput = throughput_per_s(pairs_done, avg_ms)
            csv_rows.append(
                {
                    "method": name,
                    "config_index": config_index,
                    "input_file": input_file,
                    "input_type": input_type,
                    "sanitize": sanitize_value,
                    "num_mols": len(mols),
                    "num_pairs": len(pairs),
                    "atom_compare": config_row["atom_compare"],
                    "bond_compare": config_row["bond_compare"],
                    "match_valences": config_row["match_valences"],
                    "match_formal_charge": config_row["match_formal_charge"],
                    "ring_matches_ring_only": config_row["ring_matches_ring_only"],
                    "timeout_seconds": config_row["timeout_seconds"],
                    "batch_size": applied_config.batchSize if is_nvmolkit else "N/A",
                    "workers": applied_config.workerThreads if is_nvmolkit else "N/A",
                    "prep_threads": applied_config.preprocessingThreads if is_nvmolkit else "N/A",
                    "executors_per_runner": applied_config.executorsPerRunner if is_nvmolkit else "N/A",
                    "num_gpus": len(applied_config.gpuIds) if is_nvmolkit else "N/A",
                    "nvmolkit_config_source": config_source if is_nvmolkit else "N/A",
                    "rdkit_threads": rdkit_thread_map[name] if is_rdkit else "N/A",
                    "rdkit_max_seconds": args.rdkit_max_seconds if is_rdkit else "N/A",
                    "time_ms": round(avg_ms, 2),
                    "std_ms": round(std_ms, 2),
                    "pairs_processed": pairs_done,
                    "pairs_per_second": round(throughput, 2),
                    "vs_rdkit_throughput_ratio": (
                        throughput / baseline_throughput if is_nvmolkit and baseline_throughput > 0 else "N/A"
                    ),
                }
            )

        if ran_nvmolkit:
            torch_module.cuda.synchronize()
            torch_module.cuda.empty_cache()
            torch_module.cuda.ipc_collect()
        gc.collect()

    print("\n\nCSV Results:")
    print_csv_rows(csv_rows)
    if args.output:
        write_csv_rows(csv_rows, args.output)
        print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()
