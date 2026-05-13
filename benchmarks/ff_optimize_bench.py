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

"""Force-field optimization benchmark comparing nvmolkit GPU MMFF/UFF against RDKit.

Pre-embeds molecules with RDKit ETKDG, then times either
:func:`nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs` or
:func:`nvmolkit.uffOptimization.UFFOptimizeMoleculesConfs` versus their RDKit
``AllChem.MMFFOptimizeMoleculeConfs`` / ``AllChem.UFFOptimizeMoleculeConfs``
counterparts. Reports per-method wall-clock timings and (when validation is
enabled) absolute energy deltas between the two implementations.

Usage:
    python ff_optimize_bench.py --smiles data/chembl_10k.smi --ff mmff --num_mols 200 --confs_per_mol 5
    python ff_optimize_bench.py --sdf data/MPCONF196.sdf --ff uff --confs_per_mol 1 --no_rdkit
    python ff_optimize_bench.py --pickle prepared.pkl --ff mmff --batch_size 512 --batches_per_gpu 4
"""

import argparse
import gc
import math
import random
import statistics
import sys

import nvtx
import torch
from bench_utils import (
    clone_mols_with_conformers,
    load_pickle,
    load_sdf,
    load_smiles,
    prep_mols,
    time_it,
)
from nvmolkit import autotune as nv_autotune
from nvmolkit.types import HardwareOptions
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

OPTUNA_AVAILABLE = nv_autotune.is_available()


def _embed_conformers(mols: list[Chem.Mol], confs_per_mol: int, seed: int) -> list[Chem.Mol]:
    """Generate ``confs_per_mol`` conformers per molecule using RDKit ETKDGv3.

    Molecules where embedding fails to produce at least one conformer are
    dropped; a count is printed.
    """
    params = rdDistGeom.ETKDGv3()
    params.useRandomCoords = True
    params.randomSeed = seed

    embedded: list[Chem.Mol] = []
    drop_count = 0
    for mol in mols:
        try:
            conf_ids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=confs_per_mol, params=params)
            if not conf_ids:
                drop_count += 1
                continue
            embedded.append(mol)
        except Exception:
            drop_count += 1
    if drop_count > 0:
        print(f"  Dropped {drop_count} molecules during embedding (no conformer generated)")
    return embedded


def _flatten_energies(per_mol: list[list[float]]) -> list[float]:
    """Flatten ``[[e0, e1, ...], [e0, e1, ...], ...]`` returned by nvmolkit."""
    flat: list[float] = []
    for energies in per_mol:
        flat.extend(float(e) for e in energies)
    return flat


def _flatten_rdkit_energies(per_mol: list[list[tuple[int, float]]]) -> tuple[list[float], int]:
    """Flatten ``[[(status, energy), ...], ...]`` returned by RDKit FF APIs.

    Returns ``(energies, not_converged_count)`` where ``not_converged_count`` is
    the number of conformers RDKit reported with a non-zero status flag.
    """
    flat: list[float] = []
    not_converged = 0
    for entries in per_mol:
        for status, energy in entries:
            if status != 0:
                not_converged += 1
            flat.append(float(energy))
    return flat, not_converged


def _energy_diff_summary(
    rdkit_energies: list[float],
    nvmolkit_energies: list[float],
) -> tuple[float, float, int]:
    """Mean / max absolute energy difference and the number of paired conformers.

    Conformers where either side reports ``nan``/``inf`` are skipped.
    """
    paired = min(len(rdkit_energies), len(nvmolkit_energies))
    deltas: list[float] = []
    for i in range(paired):
        rd_e = rdkit_energies[i]
        nv_e = nvmolkit_energies[i]
        if not math.isfinite(rd_e) or not math.isfinite(nv_e):
            continue
        deltas.append(abs(rd_e - nv_e))
    if not deltas:
        return float("nan"), float("nan"), 0
    return statistics.mean(deltas), max(deltas), len(deltas)


@nvtx.annotate("bench_nvmolkit_ff", color="red")
def bench_nvmolkit(
    mols: list[Chem.Mol],
    ff: str,
    max_iters: int,
    hardware_options,
    runs: int,
    warmup: bool,
) -> tuple[float, float, list[float]]:
    """Benchmark nvmolkit MMFF/UFF optimization; return ``(mean_ms, std_ms, energies)``."""
    if ff == "mmff":
        from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs as _OptimizeConfs
    elif ff == "uff":
        from nvmolkit.uffOptimization import UFFOptimizeMoleculesConfs as _OptimizeConfs
    else:
        raise ValueError(f"Unknown ff: {ff!r}")

    last_energies: list[list[float]] = [[]]

    @nvtx.annotate("ff_nvmolkit_run", color="orange")
    def run() -> None:
        cloned = clone_mols_with_conformers(mols)
        last_energies[0] = _OptimizeConfs(cloned, maxIters=max_iters, hardwareOptions=hardware_options)

    if warmup:
        warmup_mols = clone_mols_with_conformers(mols[: min(4, len(mols))])
        with nvtx.annotate("ff_nvmolkit_warmup", color="purple"):
            _OptimizeConfs(warmup_mols, maxIters=max_iters, hardwareOptions=hardware_options)
        torch.cuda.synchronize()

    result = time_it(run, runs=runs, warmups=0, gpu_sync=True)
    return result.mean_ms, result.std_ms, _flatten_energies(last_energies[0])


@nvtx.annotate("bench_rdkit_ff", color="green")
def bench_rdkit(
    mols: list[Chem.Mol],
    ff: str,
    max_iters: int,
    runs: int,
    warmup: bool,
    num_threads: int,
) -> tuple[float, float, list[float]]:
    """Benchmark RDKit MMFF/UFF optimization; return ``(mean_ms, std_ms, energies)``."""
    if ff == "mmff":
        rdkit_optimize = lambda mol: AllChem.MMFFOptimizeMoleculeConfs(  # noqa: E731
            mol, numThreads=num_threads, maxIters=max_iters
        )
    elif ff == "uff":
        rdkit_optimize = lambda mol: AllChem.UFFOptimizeMoleculeConfs(  # noqa: E731
            mol, numThreads=num_threads, maxIters=max_iters
        )
    else:
        raise ValueError(f"Unknown ff: {ff!r}")

    last_results: list[list[list[tuple[int, float]]]] = [[]]

    @nvtx.annotate("ff_rdkit_run", color="yellow")
    def run() -> None:
        cloned = clone_mols_with_conformers(mols)
        last_results[0] = [rdkit_optimize(mol) for mol in cloned]

    if warmup:
        warmup_mols = clone_mols_with_conformers(mols[: min(4, len(mols))])
        for warmup_mol in warmup_mols:
            rdkit_optimize(warmup_mol)

    result = time_it(run, runs=runs, warmups=0, gpu_sync=False)
    energies, not_converged = _flatten_rdkit_energies(last_results[0])
    if not_converged > 0:
        print(f"  RDKit: {not_converged} conformer(s) reported non-zero status (not converged)")
    return result.mean_ms, result.std_ms, energies


def _build_hardware_options(
    batch_size: int,
    batches_per_gpu: int,
    prep_threads: int,
    num_gpus: int,
) -> HardwareOptions:
    return HardwareOptions(
        preprocessingThreads=prep_threads,
        batchSize=batch_size,
        batchesPerGpu=batches_per_gpu,
        gpuIds=list(range(num_gpus)) if num_gpus > 0 else None,
    )


CSV_HEADER = (
    "method,ff,input_file,input_type,num_mols,confs_per_mol,max_iters,"
    "batch_size,batches_per_gpu,prep_threads,num_gpus,nvmolkit_config_source,"
    "rdkit_threads,time_ms,std_ms,energies_compared,mean_abs_energy_diff,max_abs_energy_diff"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Force-field optimization benchmark: nvmolkit vs RDKit (MMFF or UFF)",
    )
    parser.add_argument("--smiles", "-s", help="Path to SMILES file with molecules")
    parser.add_argument("--sdf", help="Path to SDF file (alternative to --smiles)")
    parser.add_argument("--pickle", help="Path to pickled RDKit binary molecules (alternative to --smiles)")
    parser.add_argument("--num_mols", "-n", type=int, default=0, help="Max number of molecules (default: 0 = all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and ETKDG (default: 42)")
    parser.add_argument(
        "--sanitize", action="store_true", dest="sanitize", help="Sanitize molecules during parsing (default)"
    )
    parser.add_argument("--no_sanitize", action="store_false", dest="sanitize", help="Skip sanitization at parse time")
    parser.set_defaults(sanitize=True)

    parser.add_argument("--ff", choices=["mmff", "uff"], required=True, help="Force field to optimize: mmff or uff")
    parser.add_argument(
        "--confs_per_mol",
        "-c",
        type=int,
        default=10,
        help="Conformers per molecule to embed before timing (default: 10)",
    )
    parser.add_argument("--max_iters", type=int, default=200, help="Maximum FF optimization iterations (default: 200)")

    parser.add_argument("--runs", "-r", type=int, default=1, help="Number of timing runs (default: 1)")
    parser.add_argument("--warmup", action="store_true", dest="warmup", help="Perform a warmup run (default)")
    parser.add_argument("--no_warmup", action="store_false", dest="warmup", help="Skip warmup")
    parser.set_defaults(warmup=True)

    parser.add_argument("--no_nvmolkit", action="store_true", help="Skip nvmolkit benchmark")
    parser.add_argument("--no_rdkit", action="store_true", help="Skip RDKit benchmark")
    parser.add_argument(
        "--rdkit_threads",
        type=int,
        default=1,
        help="Threads passed to RDKit FF optimizer via numThreads (default: 1)",
    )

    parser.add_argument("--batch_size", "-b", type=int, default=1024, help="nvmolkit batch size (default: 1024)")
    parser.add_argument(
        "--batches_per_gpu", type=int, default=-1, help="nvmolkit concurrent batches per GPU (-1 = library default)"
    )
    parser.add_argument(
        "--prep_threads", type=int, default=-1, help="nvmolkit preprocessing threads (-1 = library default)"
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use (default: 1)")

    parser.add_argument(
        "--autotune",
        action="store_true",
        help=(
            "Tune nvmolkit HardwareOptions (batchSize/batchesPerGpu) before timing using "
            "tune_mmff_optimize or tune_uff_optimize (depending on --ff). Requires the "
            "[autotune] extra (optuna)."
        ),
    )
    parser.add_argument(
        "--autotune_save",
        type=str,
        default=None,
        help="Path to save the tuned HardwareOptions as JSON (only with --autotune)",
    )
    parser.add_argument(
        "--autotune_load",
        type=str,
        default=None,
        help=(
            "Path to a previously-saved HardwareOptions JSON. "
            "Overrides --batch_size/--batches_per_gpu/--prep_threads (and --num_gpus if gpuIds present in the file)."
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
            "Number of molecules to use per autotune trial. "
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
        "--validate", action="store_true", dest="validate", help="Compute absolute energy diffs vs RDKit (default)"
    )
    parser.add_argument("--no_validate", action="store_false", dest="validate", help="Skip energy validation")
    parser.set_defaults(validate=True)

    parser.add_argument("--output", "-o", default=None, help="Optional path to write the CSV results")

    args = parser.parse_args()

    input_paths = [p for p in (args.smiles, args.sdf, args.pickle) if p]
    if not input_paths:
        print("Error: One of --smiles, --sdf, or --pickle is required")
        sys.exit(1)
    if len(input_paths) > 1:
        print("Error: --smiles, --sdf, and --pickle are mutually exclusive")
        sys.exit(1)
    if args.num_gpus <= 0:
        print("Error: --num_gpus must be >= 1")
        sys.exit(1)
    if args.no_nvmolkit and args.no_rdkit:
        print("Error: cannot disable both nvmolkit and RDKit")
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
    input_file = input_paths[0]
    if args.smiles:
        input_type = "smiles"
    elif args.sdf:
        input_type = "sdf"
    else:
        input_type = "pickle"

    print("\nConfiguration:")
    print(f"  Input: {input_file} ({input_type})")
    print(f"  Force field: {args.ff.upper()}")
    print(f"  Max molecules: {args.num_mols if args.num_mols > 0 else 'all'}")
    print(f"  Conformers per mol: {args.confs_per_mol}")
    print(f"  Max FF iterations: {args.max_iters}")
    print(f"  Runs: {args.runs}")
    print(f"  Warmup: {args.warmup}")
    print(f"  Validate (energy diffs): {args.validate}")
    print(f"  Run nvmolkit: {not args.no_nvmolkit}")
    print(f"  Run RDKit: {not args.no_rdkit}")
    if not args.no_rdkit:
        print(f"  RDKit threads: {args.rdkit_threads}")
    if not args.no_nvmolkit:
        print("  nvmolkit hardware:")
        print(f"    batch_size: {args.batch_size}")
        print(f"    batches_per_gpu: {args.batches_per_gpu if args.batches_per_gpu > 0 else 'auto'}")
        print(f"    prep_threads: {args.prep_threads if args.prep_threads > 0 else 'auto'}")
        print(f"    num_gpus: {args.num_gpus}")

    print("\nLoading molecules...")
    if args.smiles:
        raw_mols = load_smiles(args.smiles, args.num_mols, args.sanitize, seed=args.seed)
    elif args.sdf:
        raw_mols = load_sdf(args.sdf, args.num_mols, seed=args.seed, sanitize=args.sanitize)
    else:
        raw_mols = load_pickle(args.pickle, args.num_mols, seed=args.seed)
    if not raw_mols:
        print("Error: No valid molecules loaded")
        sys.exit(1)

    print("\nPreparing molecules (AddHs / sanitize / clear conformers)...")
    mols = prep_mols(raw_mols)
    if not mols:
        print("Error: No molecules survived preparation")
        sys.exit(1)
    print(f"  {len(mols)} molecules ready")

    print(f"\nEmbedding {args.confs_per_mol} conformer(s) per molecule with RDKit ETKDGv3...")
    mols = _embed_conformers(mols, args.confs_per_mol, args.seed)
    if not mols:
        print("Error: No molecules retained after embedding")
        sys.exit(1)
    total_confs = sum(m.GetNumConformers() for m in mols)
    print(f"  {len(mols)} molecules with {total_confs} conformers ready")

    results: dict[str, tuple[float, float, list[float]]] = {}

    hardware_options = None
    config_source = "cli"
    if not args.no_nvmolkit:
        gpu_ids = list(range(args.num_gpus))
        if args.autotune_load:
            print(f"\nLoading tuned HardwareOptions from {args.autotune_load}...")
            loaded = nv_autotune.load(args.autotune_load)
            if not isinstance(loaded, HardwareOptions):
                print(f"Error: {args.autotune_load} contains {type(loaded).__name__}, expected HardwareOptions")
                sys.exit(1)
            hardware_options = loaded
            if not hardware_options.gpuIds:
                hardware_options.gpuIds = gpu_ids
            config_source = "loaded"
            print(
                f"  Loaded: batchSize={hardware_options.batchSize}, "
                f"batchesPerGpu={hardware_options.batchesPerGpu}, "
                f"preprocessingThreads={hardware_options.preprocessingThreads}, "
                f"gpuIds={list(hardware_options.gpuIds) if hardware_options.gpuIds else []}"
            )
        elif args.autotune:
            if not OPTUNA_AVAILABLE:
                print(
                    "Error: --autotune requires the optional 'optuna' dependency. "
                    "Install with `pip install nvmolkit[autotune]` or `conda install -c conda-forge optuna`."
                )
                sys.exit(1)
            print(
                f"\nAutotuning HardwareOptions for {args.ff.upper()} "
                f"(n_trials={args.autotune_trials}, per-trial target={args.autotune_time_budget:.1f}s)..."
            )
            explicit_calibration = None
            if args.autotune_calibration_size > 0:
                rng = random.Random(args.autotune_seed)
                size = min(args.autotune_calibration_size, len(mols))
                explicit_calibration = rng.sample(range(len(mols)), size)
            tune_kwargs = dict(
                maxIters=args.max_iters,
                gpuIds=gpu_ids,
                n_trials=args.autotune_trials,
                target_seconds_per_trial=args.autotune_time_budget,
                calibration_set=explicit_calibration,
                seed=args.autotune_seed,
                verbose=True,
            )
            if args.ff == "mmff":
                tune_result = nv_autotune.tune_mmff_optimize(mols, **tune_kwargs)
            else:
                tune_result = nv_autotune.tune_uff_optimize(mols, **tune_kwargs)
            hardware_options = tune_result.best_config
            config_source = "autotuned"
            print(
                f"  Best: batchSize={hardware_options.batchSize}, "
                f"batchesPerGpu={hardware_options.batchesPerGpu} "
                f"(throughput={tune_result.best_throughput:.2f} confs/s, "
                f"trials_run={tune_result.n_trials_run}, "
                f"calibration_size={tune_result.calibration_size})"
            )
            if args.autotune_save:
                nv_autotune.save(hardware_options, args.autotune_save)
                print(f"  Saved tuned config to {args.autotune_save}")
        else:
            hardware_options = _build_hardware_options(
                args.batch_size, args.batches_per_gpu, args.prep_threads, args.num_gpus
            )

        torch.cuda.cudart().cudaProfilerStart()
        print(f"\nRunning nvmolkit {args.ff.upper()} optimize benchmark...")
        nv_avg, nv_std, nv_energies = bench_nvmolkit(
            mols, args.ff, args.max_iters, hardware_options, args.runs, args.warmup
        )
        print(f"  nvmolkit:        {nv_avg:10.2f} ms (+/- {nv_std:.2f} ms)")
        results["nvmolkit"] = (nv_avg, nv_std, nv_energies)
        torch.cuda.cudart().cudaProfilerStop()

    if not args.no_rdkit:
        print(f"\nRunning RDKit {args.ff.upper()} optimize benchmark...")
        rd_avg, rd_std, rd_energies = bench_rdkit(
            mols, args.ff, args.max_iters, args.runs, args.warmup, args.rdkit_threads
        )
        print(f"  RDKit:           {rd_avg:10.2f} ms (+/- {rd_std:.2f} ms)")
        results["rdkit"] = (rd_avg, rd_std, rd_energies)

    if not results:
        print("Error: No benchmarks were run")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Summary:")
    baseline_ms = results.get("rdkit", (None, None, None))[0]
    for name, (avg_ms, std_ms, _) in results.items():
        speedup = ""
        if baseline_ms is not None and name != "rdkit" and avg_ms > 0:
            speedup = f", {baseline_ms / avg_ms:.1f}x vs RDKit"
        print(f"  {name:20s}: {avg_ms:10.2f} ms (+/- {std_ms:.2f} ms){speedup}")

    energy_mean = float("nan")
    energy_max = float("nan")
    energy_pairs = 0
    if args.validate and "nvmolkit" in results and "rdkit" in results:
        print(f"\nValidation ({args.ff.upper()} energies)...")
        energy_mean, energy_max, energy_pairs = _energy_diff_summary(results["rdkit"][2], results["nvmolkit"][2])
        if energy_pairs > 0:
            print(
                f"  |RDKit - nvmolkit|: mean={energy_mean:.4f}, max={energy_max:.4f} "
                f"kcal/mol over {energy_pairs} paired conformers"
            )
        else:
            print("  No paired conformers with finite energies on both sides")

    if "nvmolkit" in results and hardware_options is not None:
        applied_batch_size = int(hardware_options.batchSize)
        applied_batches_per_gpu = int(hardware_options.batchesPerGpu)
        applied_prep_threads = int(hardware_options.preprocessingThreads)
        applied_num_gpus = len(list(hardware_options.gpuIds)) if hardware_options.gpuIds else args.num_gpus
    else:
        applied_batch_size = args.batch_size
        applied_batches_per_gpu = args.batches_per_gpu
        applied_prep_threads = args.prep_threads
        applied_num_gpus = args.num_gpus

    csv_rows: list[str] = []
    for name, (avg_ms, std_ms, energies) in results.items():
        is_nv = name == "nvmolkit"
        batch_size = applied_batch_size if is_nv else "N/A"
        batches_per_gpu = applied_batches_per_gpu if is_nv else "N/A"
        prep_threads = applied_prep_threads if is_nv else "N/A"
        num_gpus = applied_num_gpus if is_nv else "N/A"
        nvmolkit_config_source = config_source if is_nv else "N/A"
        rdkit_threads = args.rdkit_threads if name == "rdkit" else "N/A"
        mean_diff = energy_mean if (args.validate and is_nv) else "N/A"
        max_diff = energy_max if (args.validate and is_nv) else "N/A"
        pairs = energy_pairs if (args.validate and is_nv) else "N/A"
        csv_rows.append(
            f"{name},{args.ff},{input_file},{input_type},{len(mols)},{args.confs_per_mol},"
            f"{args.max_iters},{batch_size},{batches_per_gpu},{prep_threads},{num_gpus},"
            f"{nvmolkit_config_source},{rdkit_threads},{avg_ms:.2f},{std_ms:.2f},"
            f"{pairs},{mean_diff},{max_diff}"
        )

    print("\n\nCSV Results:")
    print(CSV_HEADER)
    for row in csv_rows:
        print(row)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(CSV_HEADER + "\n")
            for row in csv_rows:
                fh.write(row + "\n")
        print(f"\nWrote results to {args.output}")

    if not args.no_nvmolkit:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


if __name__ == "__main__":
    main()
