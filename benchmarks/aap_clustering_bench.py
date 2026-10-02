# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Atom-Atom Path similarity and directed sphere exclusion clustering."""

import argparse
import importlib
import math
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from bench_utils import (
    add_backend_selection_args,
    load_csv,
    load_sdf,
    load_smiles,
    print_csv_rows,
    throughput_per_s,
    time_it,
    write_csv_rows,
)
from rdkit import Chem
from rdkit.SimDivFilters import rdSimDivPickers

from nvmolkit.clustering import fused_dise
from nvmolkit.similarity import AAPMetric, aap_similarity

SUPPORTED_BOND_TYPES = {
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
}
DEFAULT_INPUT = Path(__file__).resolve().parent / "data" / "chembl_10k.smi"


def _load_rdkit_reference():
    try:
        return importlib.import_module("rdkit.Contrib.AtomAtomSimilarity.AtomAtomPathSimilarity")
    except ImportError:
        return None


def _load_ligand_clustering_cpu_reference():
    for module_name in ("ligand_clustering_cpu", "ligand_clustering.ligand_clustering_cpu"):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            pass
    return None


def _filter_supported_molecules(molecules, max_atoms):
    return [
        molecule
        for molecule in molecules
        if 0 < molecule.GetNumAtoms() <= max_atoms
        and all(bond.GetBondType() in SUPPORTED_BOND_TYPES for bond in molecule.GetBonds())
    ]


def _sample_pairs(num_molecules, num_pairs, seed):
    rng = random.Random(seed)
    pairs = []
    for _ in range(num_pairs):
        left = rng.randrange(num_molecules)
        right = rng.randrange(num_molecules - 1)
        if right >= left:
            right += 1
        pairs.append((left, right))
    return pairs


def _rdkit_pair_scores(molecules, pairs, max_path_length, reference):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return [
            reference.AtomAtomPathSimilarity(
                molecules[left],
                molecules[right],
                reference.getpathintegers(molecules[left], max_path_length),
                reference.getpathintegers(molecules[right], max_path_length),
            )
            for left, right in pairs
        ]


def _ligand_clustering_cpu_pair_scores(molecules, pairs, max_path_length, reference):
    return [
        reference.atom_atom_path_similarity(
            molecules[left],
            molecules[right],
            reference.get_path_integers(molecules[left], max_path_length),
            reference.get_path_integers(molecules[right], max_path_length),
        )
        for left, right in pairs
    ]


def _remap_clusters_by_size(labels, num_clusters):
    sizes = [0] * num_clusters
    for label in labels:
        sizes[label] += 1
    order = sorted(range(num_clusters), key=lambda cluster: -sizes[cluster])
    remap = {old: new + 1 for new, old in enumerate(order)}
    return [remap[label] for label in labels]


def _ligand_clustering_cpu_cluster(molecules, threshold, max_path_length, reference):
    """Run exact CPU AAP primitives with directed sphere exclusion.

    The timed region starts with pre-parsed RDKit molecules and includes the
    descriptor and exact Hungarian AAP computations.
    """
    descriptors = reference.precompute_path_integers_batch(molecules, max_path_length)
    labels = [-1] * len(molecules)
    cluster_id = 0
    for centroid in range(len(molecules)):
        if labels[centroid] >= 0:
            continue
        labels[centroid] = cluster_id
        for candidate in range(len(molecules)):
            if labels[candidate] >= 0:
                continue
            similarity = reference.atom_atom_path_similarity(
                molecules[centroid],
                molecules[candidate],
                descriptors[centroid],
                descriptors[candidate],
            )
            if similarity >= threshold:
                labels[candidate] = cluster_id
        cluster_id += 1
    return _remap_clusters_by_size(labels, cluster_id)


def _parse_priority_value(raw_value):
    cleaned = raw_value.strip().lstrip("<>").strip()
    value = float(cleaned)
    if not math.isfinite(value):
        raise ValueError(f"non-finite priority value: {raw_value!r}")
    return value


def _filter_sortable_molecules(molecules, sort_tag):
    sortable = []
    for molecule in molecules:
        if not molecule.HasProp(sort_tag):
            continue
        try:
            _parse_priority_value(molecule.GetProp(sort_tag))
        except (TypeError, ValueError):
            continue
        sortable.append(molecule)
    return sortable


def _priority_order(molecules, sort_tag, descending=False):
    indexed = list(enumerate(molecules))
    if sort_tag:

        def priority(item):
            value = _parse_priority_value(item[1].GetProp(sort_tag))
            return -value if descending else value

        indexed.sort(key=priority)
    return indexed


def _rdkit_aap_dise_cluster(
    molecules, threshold, max_path_length, sort_tag, sort_descending, reference, phase_timings=None
):
    """Run the complete RDKit AAP and directed sphere exclusion workflow.

    The timed region includes priority cleanup, sorting, centroid selection
    with ``LeaderPicker``, and nearest-centroid assignment.
    """
    start = time.perf_counter()
    indexed = _priority_order(molecules, sort_tag, sort_descending)
    ordering_done = time.perf_counter()
    ordered_molecules = [molecule for _, molecule in indexed]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        descriptors = [reference.getpathintegers(molecule, max_path_length) for molecule in ordered_molecules]
    descriptors_done = time.perf_counter()
    similarity_cache = {}

    def similarity(left, right):
        if left == right:
            return 1.0
        key = (left, right) if left < right else (right, left)
        if key not in similarity_cache:
            similarity_cache[key] = reference.AtomAtomPathSimilarity(
                ordered_molecules[key[0]],
                ordered_molecules[key[1]],
                descriptors[key[0]],
                descriptors[key[1]],
            )
        return similarity_cache[key]

    picker = rdSimDivPickers.LeaderPicker()
    centroids = list(
        picker.LazyPick(
            lambda left, right: 1.0 - similarity(left, right),
            len(ordered_molecules),
            1.0 - threshold,
        )
    )
    selection_done = time.perf_counter()
    centroid_set = set(centroids)
    ordered_labels = [-1] * len(ordered_molecules)
    for cluster_id, centroid in enumerate(centroids):
        ordered_labels[centroid] = cluster_id
    for molecule_index in range(len(ordered_molecules)):
        if molecule_index in centroid_set:
            continue
        ordered_labels[molecule_index] = max(
            range(len(centroids)),
            key=lambda cluster_id: similarity(centroids[cluster_id], molecule_index),
        )

    labels = [-1] * len(molecules)
    for (input_index, _), label in zip(indexed, ordered_labels, strict=True):
        labels[input_index] = label
    assignment_done = time.perf_counter()
    if phase_timings is not None:
        phase_timings.append(
            {
                "ordering_time_ms": (ordering_done - start) * 1000.0,
                "descriptor_time_ms": (descriptors_done - ordering_done) * 1000.0,
                "centroid_selection_time_ms": (selection_done - descriptors_done) * 1000.0,
                "nearest_assignment_time_ms": (assignment_done - selection_done) * 1000.0,
                "similarity_evaluations": len(similarity_cache),
            }
        )
    return labels


def _cluster_agreement(left, right):
    agreements = 0
    comparisons = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            agreements += (left[first] == left[second]) == (right[first] == right[second])
            comparisons += 1
    return agreements / comparisons if comparisons else 1.0


def _add_cluster_stats(row, labels):
    _, sizes = np.unique(labels, return_counts=True)
    row["num_clusters"] = len(sizes)
    row["largest_cluster_size"] = int(np.max(sizes))
    row["median_cluster_size"] = float(np.median(sizes))
    row["singleton_cluster_fraction"] = float(np.mean(sizes == 1))


def _add_priority_stats(row, molecules, sort_tag, descending):
    values = [_parse_priority_value(molecule.GetProp(sort_tag)) for molecule in molecules]
    row["priority_sort"] = sort_tag
    row["priority_direction"] = "descending" if descending else "ascending"
    row["priority_distinct_values"] = len(set(values))
    row["priority_tie_fraction"] = 1.0 - len(set(values)) / len(values)


def _time_callable(function, runs, warmup, gpu_sync=False):
    result = None

    def invoke():
        nonlocal result
        result = function()

    timing = time_it(invoke, runs=runs, warmups=1 if warmup else 0, gpu_sync=gpu_sync)
    return timing, result


def _aap_metric(args):
    return AAPMetric(
        max_path_length=args.max_path_length,
        histogram_bins=args.histogram_bins,
        sinkhorn_iterations=args.sinkhorn_iterations,
        sinkhorn_temperature=args.sinkhorn_temperature,
    )


def _base_row(args, operation, method, count, timing, molecules, status="ok"):
    atom_counts = [molecule.GetNumAtoms() for molecule in molecules]
    is_pair = operation == "similarity"
    is_nvmolkit = method == "nvmolkit_gpu"
    return {
        "method": method,
        "operation": operation,
        "status": status,
        "timing_scope": "from_preparsed_rdkit_molecules",
        "input_file": str(args.sdf or args.csv or args.smiles),
        "num_mols": len(molecules),
        "num_pairs": count if is_pair else "N/A",
        "mols_processed": count if not is_pair else "N/A",
        "time_ms": round(timing.mean_ms, 4) if timing is not None else "",
        "std_ms": round(timing.std_ms, 4) if timing is not None else "",
        "pairs_per_second": (
            round(throughput_per_s(count, timing.mean_ms), 2) if is_pair and timing is not None else ""
        ),
        "vs_ligand_clustering_cpu_pair_throughput_ratio": "",
        "threshold": args.threshold,
        "max_path_length": args.max_path_length,
        "histogram_bins": args.histogram_bins if is_nvmolkit else "",
        "sinkhorn_iterations": args.sinkhorn_iterations if is_nvmolkit else "",
        "sinkhorn_temperature": args.sinkhorn_temperature if is_nvmolkit else "",
        "mean_atoms": round(float(np.mean(atom_counts)), 2),
        "min_atoms": min(atom_counts),
        "max_atoms": max(atom_counts),
    }


def _set_pair_comparison(rows, outputs, reference_method, field_prefix):
    if reference_method not in outputs:
        return
    expected = np.asarray(outputs[reference_method])
    for row in rows:
        method = row["method"]
        if method not in outputs:
            continue
        actual = np.asarray(outputs[method])
        correlation = float(np.corrcoef(expected, actual)[0, 1]) if np.std(expected) and np.std(actual) else math.nan
        row[f"{field_prefix}_correlation"] = correlation
        row[f"{field_prefix}_mae"] = float(np.mean(np.abs(expected - actual)))


def _benchmark_pairs(args, molecules, rdkit_reference, ligand_clustering_cpu_reference):
    pairs = _sample_pairs(len(molecules), args.num_pairs, args.seed)
    metric = _aap_metric(args)
    rows = []
    outputs = {}

    if not args.no_nvmolkit:
        timing, scores = _time_callable(
            lambda: [aap_similarity(molecules[left], molecules[right], metric=metric) for left, right in pairs],
            args.runs,
            args.warmup,
            gpu_sync=True,
        )
        outputs["nvmolkit_gpu"] = scores
        rows.append(_base_row(args, "similarity", "nvmolkit_gpu", len(pairs), timing, molecules))

    if not args.no_rdkit:
        if rdkit_reference is None:
            rows.append(_base_row(args, "similarity", "rdkit_contrib", len(pairs), None, molecules, "unavailable"))
        else:
            timing, scores = _time_callable(
                lambda: _rdkit_pair_scores(molecules, pairs, args.max_path_length, rdkit_reference),
                args.runs,
                args.warmup,
            )
            outputs["rdkit_contrib"] = scores
            rows.append(_base_row(args, "similarity", "rdkit_contrib", len(pairs), timing, molecules))

    if not args.no_ligand_clustering_cpu:
        if ligand_clustering_cpu_reference is None:
            rows.append(
                _base_row(
                    args,
                    "similarity",
                    "ligand_clustering_cpu",
                    len(pairs),
                    None,
                    molecules,
                    "unavailable",
                )
            )
        else:
            timing, scores = _time_callable(
                lambda: _ligand_clustering_cpu_pair_scores(
                    molecules, pairs, args.max_path_length, ligand_clustering_cpu_reference
                ),
                args.runs,
                args.warmup,
            )
            outputs["ligand_clustering_cpu"] = scores
            rows.append(_base_row(args, "similarity", "ligand_clustering_cpu", len(pairs), timing, molecules))

    _set_pair_comparison(rows, outputs, "rdkit_contrib", "vs_rdkit_contrib")
    _set_pair_comparison(rows, outputs, "ligand_clustering_cpu", "vs_ligand_clustering_cpu")
    by_method = {row["method"]: row for row in rows}
    if "nvmolkit_gpu" in outputs and "ligand_clustering_cpu" in outputs:
        by_method["nvmolkit_gpu"]["vs_ligand_clustering_cpu_pair_throughput_ratio"] = round(
            by_method["nvmolkit_gpu"]["pairs_per_second"] / by_method["ligand_clustering_cpu"]["pairs_per_second"],
            4,
        )
    return rows


def _benchmark_clustering(args, molecules, ligand_clustering_cpu_reference):
    metric = _aap_metric(args)
    rows = []
    outputs = {}

    if not args.no_nvmolkit:
        timing, result = _time_callable(
            lambda: fused_dise(
                molecules,
                cutoff=1.0 - args.threshold,
                metric=metric,
                assignment="first",
            ),
            args.runs,
            args.warmup,
            gpu_sync=True,
        )
        labels = result.cluster_ids.numpy().tolist()
        outputs["nvmolkit_gpu"] = labels
        row = _base_row(args, "clustering", "nvmolkit_gpu", len(molecules), timing, molecules)
        _add_cluster_stats(row, labels)
        rows.append(row)

    if not args.no_rdkit:
        rows.append(
            _base_row(
                args,
                "clustering",
                "rdkit_contrib",
                len(molecules),
                None,
                molecules,
                "unsupported",
            )
        )

    if not args.no_ligand_clustering_cpu:
        if ligand_clustering_cpu_reference is None:
            rows.append(
                _base_row(
                    args,
                    "clustering",
                    "ligand_clustering_cpu",
                    len(molecules),
                    None,
                    molecules,
                    "unavailable",
                )
            )
        elif len(molecules) > args.cpu_max_size:
            rows.append(
                _base_row(
                    args,
                    "clustering",
                    "ligand_clustering_cpu",
                    len(molecules),
                    None,
                    molecules,
                    "size_limit",
                )
            )
        else:
            timing, labels = _time_callable(
                lambda: _ligand_clustering_cpu_cluster(
                    molecules,
                    args.threshold,
                    args.max_path_length,
                    ligand_clustering_cpu_reference,
                ),
                args.runs,
                args.warmup,
            )
            outputs["ligand_clustering_cpu"] = labels
            row = _base_row(args, "clustering", "ligand_clustering_cpu", len(molecules), timing, molecules)
            _add_cluster_stats(row, labels)
            rows.append(row)

    by_method = {row["method"]: row for row in rows}
    if "nvmolkit_gpu" in outputs and "ligand_clustering_cpu" in outputs:
        agreement = _cluster_agreement(outputs["nvmolkit_gpu"], outputs["ligand_clustering_cpu"])
        for method in ("nvmolkit_gpu", "ligand_clustering_cpu"):
            by_method[method]["vs_ligand_clustering_cpu_cluster_agreement"] = agreement
        by_method["nvmolkit_gpu"]["vs_ligand_clustering_cpu_time_speedup"] = round(
            by_method["ligand_clustering_cpu"]["time_ms"] / by_method["nvmolkit_gpu"]["time_ms"],
            4,
        )
    return rows


def _benchmark_dise(args, molecules, rdkit_reference):
    rows = []
    outputs = {}
    ordering_scope = "activity_order" if args.sort_tag else "input_order"
    timing_scope = f"{ordering_scope}+aap_descriptors+centroid_selection+nearest_assignment"
    if not args.no_nvmolkit:
        gpu_labels = None

        def run_gpu_workflow():
            nonlocal gpu_labels
            indexed = _priority_order(molecules, args.sort_tag, args.sort_descending)
            result = fused_dise(
                [molecule for _, molecule in indexed],
                cutoff=1.0 - args.threshold,
                metric=_aap_metric(args),
                assignment="nearest",
            )
            ordered_labels = result.cluster_ids.numpy().tolist()
            gpu_labels = [-1] * len(molecules)
            for (input_index, _), label in zip(indexed, ordered_labels, strict=True):
                gpu_labels[input_index] = label

        timing, _ = _time_callable(
            run_gpu_workflow,
            args.runs,
            args.warmup,
            gpu_sync=True,
        )
        row = _base_row(args, "dise", "nvmolkit_gpu", len(molecules), timing, molecules)
        row["timing_scope"] = timing_scope
        _add_cluster_stats(row, gpu_labels)
        _add_priority_stats(row, molecules, args.sort_tag, args.sort_descending)
        rows.append(row)
        outputs["nvmolkit_gpu"] = gpu_labels
    if args.no_rdkit:
        return rows
    if rdkit_reference is None:
        rows.append(
            _base_row(
                args,
                "dise",
                "rdkit_contrib_aap_leader",
                len(molecules),
                None,
                molecules,
                "unavailable",
            )
        )
    elif len(molecules) > args.rdkit_workflow_max_size:
        rows.append(
            _base_row(
                args,
                "dise",
                "rdkit_contrib_aap_leader",
                len(molecules),
                None,
                molecules,
                "size_limit",
            )
        )
    else:
        phase_timings = []
        timing, labels = _time_callable(
            lambda: _rdkit_aap_dise_cluster(
                molecules,
                args.threshold,
                args.max_path_length,
                args.sort_tag,
                args.sort_descending,
                rdkit_reference,
                phase_timings,
            ),
            args.runs,
            args.warmup,
        )
        row = _base_row(
            args,
            "dise",
            "rdkit_contrib_aap_leader",
            len(molecules),
            timing,
            molecules,
        )
        row["timing_scope"] = timing_scope
        _add_cluster_stats(row, labels)
        _add_priority_stats(row, molecules, args.sort_tag, args.sort_descending)
        measured_phases = phase_timings[-args.runs :]
        for field in measured_phases[0]:
            row[field] = round(float(np.mean([sample[field] for sample in measured_phases])), 4)
        rows.append(row)
        outputs["rdkit_contrib_aap_leader"] = labels

    by_method = {row["method"]: row for row in rows}
    if "nvmolkit_gpu" in outputs and "rdkit_contrib_aap_leader" in outputs:
        agreement = _cluster_agreement(outputs["nvmolkit_gpu"], outputs["rdkit_contrib_aap_leader"])
        for method in ("nvmolkit_gpu", "rdkit_contrib_aap_leader"):
            by_method[method]["vs_rdkit_cluster_agreement"] = agreement
        by_method["nvmolkit_gpu"]["vs_rdkit_time_speedup"] = round(
            by_method["rdkit_contrib_aap_leader"]["time_ms"] / by_method["nvmolkit_gpu"]["time_ms"],
            4,
        )
    return rows


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Atom-Atom Path (AAP) similarity and directed sphere exclusion (DISE) clustering benchmark"
    )
    parser.add_argument("--smiles", "-s", default=str(DEFAULT_INPUT), help="Input SMILES file")
    parser.add_argument("--sdf", help="Input SDF file; overrides --csv and --smiles")
    parser.add_argument("--csv", help="Input scored CSV file; overrides --smiles")
    parser.add_argument("--smiles-column", default="smiles", help="SMILES column used with --csv")
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 128, 512], help="Clustering sizes")
    parser.add_argument(
        "--sample-pool-size",
        type=int,
        help="Deterministic sampled input pool shared across separate benchmark invocations",
    )
    parser.add_argument("--num_pairs", type=int, default=128, help="Random molecule pairs for similarity timing")
    parser.add_argument("--runs", "-r", type=int, default=3, help="Number of timing runs")
    parser.add_argument("--seed", type=int, default=42, help="Molecule and pair sampling seed")
    parser.add_argument("--threshold", type=float, default=0.217, help="Inclusive clustering similarity threshold")
    parser.add_argument(
        "--sort-tag",
        help="Numeric priority property cleaned and sorted ascending inside timed DISE runs",
    )
    parser.add_argument(
        "--sort-descending",
        action="store_true",
        help="Sort larger priority values first inside timed DISE runs",
    )
    parser.add_argument("--max_atoms", type=int, default=64, help="Maximum atoms retained, at most 64")
    parser.add_argument("--max_path_length", type=int, default=7)
    parser.add_argument("--histogram_bins", type=int, default=2048)
    parser.add_argument("--sinkhorn_iterations", type=int, default=8)
    parser.add_argument("--sinkhorn_temperature", type=float, default=0.104)
    parser.add_argument(
        "--cpu-max-size",
        "--cpu_max_size",
        type=int,
        default=128,
        help="Largest clustering size run with ligand_clustering CPU",
    )
    parser.add_argument(
        "--rdkit-workflow-max-size",
        type=int,
        default=512,
        help="Largest full RDKit AAP+DISE workflow size (default: 512)",
    )
    parser.add_argument(
        "--operation",
        choices=["similarity", "clustering", "dise", "both", "all"],
        default="both",
    )
    parser.add_argument("--warmup", action="store_true", dest="warmup", help="Perform one warmup run (default)")
    parser.add_argument("--no_warmup", action="store_false", dest="warmup", help="Skip warmup")
    parser.set_defaults(warmup=True)
    add_backend_selection_args(parser)
    parser.add_argument(
        "--no-ligand-clustering-cpu",
        "--no_ligand_clustering_cpu",
        action="store_true",
        help="Skip the ligand_clustering CPU reference",
    )
    parser.add_argument("--output", "-o", help="Optional path to write CSV results")
    return parser


def main():
    args = _build_parser().parse_args()
    wants_pairs = args.operation in ("similarity", "both", "all")
    wants_clusters = args.operation in ("clustering", "dise", "both", "all")
    if args.runs <= 0:
        print("Error: --runs must be positive", file=sys.stderr)
        sys.exit(1)
    if wants_pairs and args.num_pairs <= 0:
        print("Error: --num_pairs must be positive for similarity benchmarks", file=sys.stderr)
        sys.exit(1)
    if wants_clusters and (not args.sizes or any(size <= 0 for size in args.sizes)):
        print("Error: --sizes must contain positive values", file=sys.stderr)
        sys.exit(1)
    largest_size = max(args.sizes) if wants_clusters else 0
    required_count = max(largest_size, 2 if wants_pairs else 0)
    if args.sample_pool_size is not None and args.sample_pool_size < required_count:
        print(f"Error: --sample-pool-size must be at least {required_count}", file=sys.stderr)
        sys.exit(1)
    if not 1 <= args.max_atoms <= 64:
        print("Error: --max_atoms must be between 1 and 64", file=sys.stderr)
        sys.exit(1)
    if not 0.0 <= args.threshold <= 1.0:
        print("Error: --threshold must be between 0 and 1", file=sys.stderr)
        sys.exit(1)
    if args.cpu_max_size < 0:
        print("Error: --cpu-max-size must be non-negative", file=sys.stderr)
        sys.exit(1)
    if args.rdkit_workflow_max_size < 0:
        print("Error: --rdkit-workflow-max-size must be non-negative", file=sys.stderr)
        sys.exit(1)
    if args.no_nvmolkit and args.no_rdkit and args.no_ligand_clustering_cpu:
        print("Error: cannot disable every benchmark backend", file=sys.stderr)
        sys.exit(1)
    if args.operation in ("dise", "all") and not args.sort_tag:
        print(
            "Error: the complete clustering workflow requires --sort-tag; unordered DISE benchmarks are invalid",
            file=sys.stderr,
        )
        sys.exit(1)

    rdkit_reference = None if args.no_rdkit else _load_rdkit_reference()
    ligand_clustering_cpu_reference = (
        None if args.no_ligand_clustering_cpu else _load_ligand_clustering_cpu_reference()
    )
    if rdkit_reference is None and not args.no_rdkit:
        print("RDKit Contrib AAP unavailable; its timing fields will be empty")
    if ligand_clustering_cpu_reference is None and not args.no_ligand_clustering_cpu:
        print("ligand_clustering CPU unavailable; its timing fields will be empty")

    clustering_pool_size = largest_size * 3 if wants_clusters else 0
    pair_pool_size = max(2, args.num_pairs * 2) if wants_pairs else 0
    requested = args.sample_pool_size or max(clustering_pool_size, pair_pool_size)
    try:
        if args.sdf:
            loaded = load_sdf(args.sdf, max_count=requested, sanitize=True, seed=args.seed)
        elif args.csv:
            loaded = load_csv(
                args.csv,
                smiles_column=args.smiles_column,
                property_columns=[args.sort_tag] if args.sort_tag else [],
                max_count=requested,
                sanitize=True,
                seed=args.seed,
            )
        else:
            loaded = load_smiles(args.smiles, max_count=requested, sanitize=True, seed=args.seed)
    except (OSError, ValueError) as error:
        print(f"Error loading input: {error}", file=sys.stderr)
        sys.exit(1)
    molecules = _filter_supported_molecules(loaded, args.max_atoms)
    if args.operation in ("dise", "all"):
        supported_count = len(molecules)
        molecules = _filter_sortable_molecules(molecules, args.sort_tag)
        dropped_count = supported_count - len(molecules)
        if dropped_count:
            print(f"  Dropped {dropped_count} molecules without a valid {args.sort_tag!r} priority value")
    if len(molecules) < required_count:
        print(
            f"Error: need {required_count} supported molecules, retained {len(molecules)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\nConfiguration:")
    print(f"  Input: {args.sdf or args.csv or args.smiles}")
    print(f"  Retained molecules: {len(molecules)}")
    print(f"  Clustering sizes: {args.sizes if wants_clusters else 'N/A'}")
    print(f"  Pair count: {args.num_pairs}")
    print(f"  Runs: {args.runs}")
    print(f"  Threshold: {args.threshold}")
    print(f"  DISE priority sort: {args.sort_tag or 'none'}")
    print(f"  DISE priority direction: {'descending' if args.sort_descending else 'ascending'}")
    print(f"  Run nvMolKit GPU: {not args.no_nvmolkit}")
    print(f"  Run RDKit Contrib similarity: {rdkit_reference is not None and not args.no_rdkit}")
    print(
        "  Run ligand_clustering CPU: "
        f"{ligand_clustering_cpu_reference is not None and not args.no_ligand_clustering_cpu}"
    )

    rows = []
    if args.operation in ("similarity", "both", "all"):
        print(f"\nBenchmarking {args.num_pairs} pair similarities...")
        rows.extend(
            _benchmark_pairs(
                args,
                molecules,
                rdkit_reference,
                ligand_clustering_cpu_reference,
            )
        )
    if args.operation in ("clustering", "both", "all"):
        for size in args.sizes:
            print(f"\nBenchmarking clustering for {size} molecules...")
            rows.extend(_benchmark_clustering(args, molecules[:size], ligand_clustering_cpu_reference))
            write_csv_rows(rows, args.output)
    if args.operation in ("dise", "all"):
        for size in args.sizes:
            print(f"\nBenchmarking complete RDKit AAP+DISE workflow for {size} molecules...")
            rows.extend(_benchmark_dise(args, molecules[:size], rdkit_reference))
            write_csv_rows(rows, args.output)

    if not rows:
        print("Error: no benchmark rows were produced", file=sys.stderr)
        sys.exit(1)

    print("\nResults:")
    print_csv_rows(rows)
    write_csv_rows(rows, args.output)
    if args.output:
        print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()
