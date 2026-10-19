# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark batched Morgan fingerprint generation with nvMolKit and RDKit."""

import argparse
from collections.abc import Sequence

import nvtx
import torch
from bench_utils import (
    Deadline,
    TimingResult,
    add_backend_selection_args,
    add_rdkit_max_seconds_arg,
    load_pickle,
    load_sdf,
    load_smiles,
    print_csv_rows,
    throughput_per_s,
    time_it,
    write_csv_rows,
)
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from nvmolkit.fingerprints import MorganFingerprintGenerator, unpack_fingerprint

SUPPORTED_FP_SIZES = (128, 256, 512, 1024, 2048)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Morgan fingerprint benchmark: nvMolKit vs RDKit")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--smiles", "-s", help="Path to a SMILES file")
    inputs.add_argument("--sdf", help="Path to an SDF file")
    inputs.add_argument("--pickle", help="Path to pickled RDKit binary molecules")
    parser.add_argument("--num_mols", "-n", type=int, default=0, help="Max molecules (default: 0 = all)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling and shuffle seed (default: 42)")
    parser.add_argument(
        "--sanitize", action="store_true", dest="sanitize", help="Sanitize molecules during parsing (default)"
    )
    parser.add_argument("--no_sanitize", action="store_false", dest="sanitize", help="Skip parse-time sanitization")
    parser.set_defaults(sanitize=True)

    parser.add_argument("--radius", type=int, default=2, help="Morgan radius (default: 2)")
    parser.add_argument(
        "--fp_size",
        type=int,
        choices=SUPPORTED_FP_SIZES,
        default=2048,
        help="Fingerprint bit count (default: 2048)",
    )
    parser.add_argument("--rdkit_threads", type=int, default=1, help="RDKit fingerprint threads (default: 1)")
    add_rdkit_max_seconds_arg(
        parser,
        extra_help=f"With a cap, RDKit fingerprints {RDKIT_CHUNK} molecules per call and stops at a chunk boundary.",
    )
    parser.add_argument(
        "--prep_threads",
        type=int,
        nargs="+",
        default=[0],
        metavar="N",
        help="one or more nvMolKit CPU preprocessing thread counts (default: 0 = all available)",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="CUDA device for nvMolKit (default: 0)")

    parser.add_argument("--runs", "-r", type=int, default=3, help="Timed runs (default: 3)")
    parser.add_argument("--warmups", type=int, default=1, help="Warmup runs (default: 1)")
    add_backend_selection_args(parser)
    parser.add_argument(
        "--validate", action="store_true", dest="validate", help="Validate matching fingerprints (default)"
    )
    parser.add_argument("--no_validate", action="store_false", dest="validate", help="Skip fingerprint validation")
    parser.set_defaults(validate=True)
    parser.add_argument(
        "--validation_mols",
        type=int,
        default=100,
        help="Maximum fingerprints to compare during validation (default: 100; 0 = all)",
    )
    parser.add_argument("--output", "-o", default=None, help="Optional CSV output path")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_mols < 0:
        raise ValueError("--num_mols must be non-negative")
    if args.radius < 0:
        raise ValueError("--radius must be non-negative")
    if args.rdkit_threads < 0:
        raise ValueError("--rdkit_threads must be non-negative")
    if any(prep_threads < 0 for prep_threads in args.prep_threads):
        raise ValueError("--prep_threads must be non-negative")
    if args.gpu_id < 0:
        raise ValueError("--gpu_id must be non-negative")
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")
    if args.validation_mols < 0:
        raise ValueError("--validation_mols must be non-negative")
    if args.no_nvmolkit and args.no_rdkit:
        raise ValueError("cannot disable both nvMolKit and RDKit")


def _load_molecules(args: argparse.Namespace) -> tuple[list[Chem.Mol], str, str]:
    if args.smiles:
        return (
            load_smiles(args.smiles, args.num_mols, sanitize=args.sanitize, seed=args.seed),
            args.smiles,
            "smiles",
        )
    if args.sdf:
        return (
            load_sdf(args.sdf, args.num_mols, sanitize=args.sanitize, seed=args.seed),
            args.sdf,
            "sdf",
        )
    return load_pickle(args.pickle, args.num_mols, seed=args.seed), args.pickle, "pickle"


RDKIT_CHUNK = 50000


@nvtx.annotate("bench_rdkit_morgan", color="green")
def _bench_rdkit(
    generator,
    mols: Sequence[Chem.Mol],
    num_threads: int,
    runs: int,
    warmups: int,
    max_seconds: float = 0.0,
) -> tuple[TimingResult, Sequence, int]:
    """Time RDKit fingerprinting; return ``(timing, fingerprints, molecules_processed)``.

    Without a cap the whole input is one ``GetFingerprints`` call. With ``max_seconds > 0`` one deadline
    is shared across the timed runs and molecules are fingerprinted in chunks, stopping at a chunk
    boundary; timing and the processed count come from the same retained sample set.
    """
    fingerprints: list = []
    processed = [0]
    complete_fingerprints = None

    def compute(deadline: Deadline | None = None) -> None:
        nonlocal fingerprints, complete_fingerprints
        if deadline is None or not deadline.active:
            fingerprints = list(generator.GetFingerprints(mols, numThreads=num_threads))
            processed[0] = len(mols)
            complete_fingerprints = fingerprints
            return
        fingerprints = []
        for start in range(0, len(mols), RDKIT_CHUNK):
            fingerprints.extend(generator.GetFingerprints(mols[start : start + RDKIT_CHUNK], numThreads=num_threads))
            if deadline.expired():
                break
        processed[0] = len(fingerprints)
        if processed[0] == len(mols):
            complete_fingerprints = fingerprints

    def production_run(deadline: Deadline | None = None) -> None:
        with nvtx.annotate("morgan_rdkit_production_run", color="yellow"):
            compute(deadline)

    if max_seconds > 0:
        timing = time_it(
            production_run,
            runs=runs,
            warmups=warmups,
            max_seconds=max_seconds,
            progress_getter=lambda: processed[0],
            progress_target=len(mols),
        )
        measured = timing.progress if timing.progress is not None else processed[0]
    else:
        timing = time_it(production_run, runs=runs, warmups=warmups)
        measured = len(mols)
    if measured == len(mols) and complete_fingerprints is not None:
        fingerprints = complete_fingerprints
    return timing, fingerprints, measured


def _bench_nvmolkit(
    generator: MorganFingerprintGenerator,
    mols: list[Chem.Mol],
    prep_threads: int,
    gpu_id: int,
    runs: int,
    warmups: int,
) -> tuple[TimingResult, torch.Tensor]:
    fingerprints: torch.Tensor | None = None
    with torch.cuda.device(gpu_id):
        stream = torch.cuda.Stream(device=gpu_id)

        def compute() -> None:
            nonlocal fingerprints
            fingerprints = generator.GetFingerprints(mols, num_threads=prep_threads, stream=stream).torch()

        def production_run() -> None:
            with nvtx.annotate("morgan_nvmolkit_production_run", color="orange"):
                compute()

        torch.cuda.cudart().cudaProfilerStart()
        with nvtx.annotate("bench_nvmolkit_morgan", color="red"):
            for _ in range(warmups):
                with nvtx.annotate("morgan_nvmolkit_warmup", color="purple"):
                    compute()
                    torch.cuda.synchronize()
            timing = time_it(production_run, runs=runs, warmups=0, gpu_sync=True)
        torch.cuda.cudart().cudaProfilerStop()

    assert fingerprints is not None
    return timing, fingerprints


def _validate_fingerprints(rdkit_fps: Sequence, nvmolkit_fps: torch.Tensor, max_count: int) -> int:
    count = min(len(rdkit_fps), len(nvmolkit_fps))
    if max_count > 0:
        count = min(count, max_count)
    if len(rdkit_fps) != len(nvmolkit_fps):
        raise AssertionError(f"fingerprint count mismatch: RDKit={len(rdkit_fps)}, nvMolKit={len(nvmolkit_fps)}")

    actual = unpack_fingerprint(nvmolkit_fps[:count]).cpu()
    for index in range(count):
        expected = torch.tensor(rdkit_fps[index].ToList(), dtype=torch.bool)
        mismatch_count = int(torch.count_nonzero(actual[index] != expected).item())
        if mismatch_count:
            raise AssertionError(f"fingerprint {index} differs in {mismatch_count} bits")
    return count


def _result_row(
    method: str,
    timing: TimingResult,
    *,
    input_file: str,
    input_type: str,
    num_mols: int,
    args: argparse.Namespace,
    prep_threads: int | None = None,
    rdkit_mols_per_second: float | None = None,
) -> dict[str, object]:
    is_rdkit = method == "rdkit"
    is_nvmolkit = method == "nvmolkit"
    mols_per_second = throughput_per_s(num_mols, timing.median_ms)
    if is_nvmolkit and rdkit_mols_per_second is not None:
        vs_rdkit_throughput_ratio: float | str = round(mols_per_second / rdkit_mols_per_second, 4)
    else:
        vs_rdkit_throughput_ratio = "N/A"
    return {
        "method": method,
        "input_file": input_file,
        "input_type": input_type,
        "num_mols": num_mols,
        "radius": args.radius,
        "fp_size": args.fp_size,
        "rdkit_threads": args.rdkit_threads if is_rdkit else "N/A",
        "prep_threads": prep_threads if is_nvmolkit else "N/A",
        "gpu_id": args.gpu_id if is_nvmolkit else "N/A",
        "runs": args.runs,
        "warmups": args.warmups,
        "median_ms": round(timing.median_ms, 3),
        "mean_ms": round(timing.mean_ms, 3),
        "std_ms": round(timing.std_ms, 3),
        "mols_per_second": round(mols_per_second, 2),
        "vs_rdkit_throughput_ratio": vs_rdkit_throughput_ratio,
    }


def main() -> None:
    """Run the command-line benchmark."""
    parser = _build_parser()
    args = parser.parse_args()
    try:
        _validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    print("\nConfiguration:")
    input_file = args.smiles or args.sdf or args.pickle
    print(f"  Input: {input_file}")
    print(f"  Max molecules: {args.num_mols if args.num_mols else 'all'}")
    print(f"  Sampling seed: {args.seed}")
    print(f"  Radius: {args.radius}")
    print(f"  Fingerprint size: {args.fp_size}")
    print(f"  Runs / warmups: {args.runs} / {args.warmups}")
    if not args.no_rdkit:
        print(f"  RDKit threads: {args.rdkit_threads if args.rdkit_threads else 'all available'}")
    if not args.no_nvmolkit:
        prep_threads = [str(value) if value else "all available" for value in args.prep_threads]
        print(f"  nvMolKit preprocessing threads: {', '.join(prep_threads)}")
        print(f"  nvMolKit GPU ID: {args.gpu_id}")

    print("\nLoading molecules...")
    mols, input_file, input_type = _load_molecules(args)
    if not mols:
        parser.error("no valid molecules loaded")

    rdkit_result: tuple[TimingResult, Sequence] | None = None
    if not args.no_rdkit:
        print("\nRunning RDKit Morgan fingerprint benchmark...")
        rdkit_generator = rdFingerprintGenerator.GetMorganGenerator(radius=args.radius, fpSize=args.fp_size)
        timing, fingerprints, rdkit_processed = _bench_rdkit(
            rdkit_generator, mols, args.rdkit_threads, args.runs, args.warmups, args.rdkit_max_seconds
        )
        rdkit_result = (timing, fingerprints)
        print(
            f"  median {timing.median_ms:.3f} ms (+/- {timing.std_ms:.3f} ms), "
            f"{throughput_per_s(rdkit_processed, timing.median_ms):.2f} mols/s"
            f"  [processed {rdkit_processed}/{len(mols)} mols]"
        )

    nvmolkit_results: list[tuple[int, TimingResult]] = []
    validated = False
    if not args.no_nvmolkit:
        nvmolkit_generator = MorganFingerprintGenerator(radius=args.radius, fpSize=args.fp_size)
        for prep_threads in args.prep_threads:
            thread_label = prep_threads if prep_threads else "all available"
            print(f"\nRunning nvMolKit Morgan fingerprint benchmark ({thread_label} preprocessing threads)...")
            timing, fingerprints = _bench_nvmolkit(
                nvmolkit_generator,
                mols,
                prep_threads,
                args.gpu_id,
                args.runs,
                args.warmups,
            )
            nvmolkit_results.append((prep_threads, timing))
            print(
                f"  median {timing.median_ms:.3f} ms (+/- {timing.std_ms:.3f} ms), "
                f"{throughput_per_s(len(mols), timing.median_ms):.2f} mols/s"
            )
            if args.validate and rdkit_result is not None:
                checked = _validate_fingerprints(
                    rdkit_result[1], fingerprints[: len(rdkit_result[1])], args.validation_mols
                )
                print(f"\nValidation passed for {checked} fingerprints (prep_threads={prep_threads})")
                validated = True

    if args.validate and not validated:
        print("\nValidation skipped (both backends are required)")

    rdkit_mols_per_second = None
    rows = []
    if rdkit_result is not None:
        rdkit_mols_per_second = throughput_per_s(rdkit_processed, rdkit_result[0].median_ms)
        rows.append(
            _result_row(
                "rdkit",
                rdkit_result[0],
                input_file=input_file,
                input_type=input_type,
                num_mols=rdkit_processed,
                args=args,
            )
        )
    for prep_threads, timing in nvmolkit_results:
        if rdkit_mols_per_second is not None:
            nvmolkit_mols_per_second = throughput_per_s(len(mols), timing.median_ms)
            print(
                f"\nnvMolKit throughput relative to RDKit (prep_threads={prep_threads}): "
                f"{nvmolkit_mols_per_second / rdkit_mols_per_second:.2f}x"
            )
        rows.append(
            _result_row(
                "nvmolkit",
                timing,
                prep_threads=prep_threads,
                input_file=input_file,
                input_type=input_type,
                num_mols=len(mols),
                args=args,
                rdkit_mols_per_second=rdkit_mols_per_second,
            )
        )
    print("\nCSV Results:")
    print_csv_rows(rows)
    if args.output:
        write_csv_rows(rows, args.output)
        print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()
