# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark selected 3D properties against RDKit.

The primary nvMolKit timing starts from RDKit molecules and includes
coordinate and atom-weight extraction, transfer, and calculation. ``--device_input`` additionally times the chained-workflow
path, which reads coordinates from a ``Device3DResult`` prepared outside the
timed region.

Example:
    python descriptors3d_bench.py --smiles data/chembl_10k.smi \
        --num_mols 100 1000 --confs_per_mol 1 10
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np
import rdkit
import torch
from bench_utils import (
    Deadline,
    add_backend_selection_args,
    add_rdkit_max_seconds_arg,
    available_cpu_count,
    embed_and_jitter,
    load_smiles,
    print_csv_rows,
    process_map_bounded,
    process_pool,
    slice_conformers,
    throughput_per_s,
    time_it,
    write_csv_rows,
)
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from nvmolkit.descriptors3d import PER_ATOM_PROPERTIES, Calc3DProperties, Property3D
from nvmolkit.types import AsyncGpuResult, Device3DResult, HardwareOptions, PrecisionMode

PRECISIONS = {"single": PrecisionMode.SINGLE, "full": PrecisionMode.FULL}
# Default (relative, absolute) tolerances against RDKit at each precision's rounding level.
VALIDATION_TOLERANCES = {PrecisionMode.SINGLE: (2e-6, 1e-3), PrecisionMode.FULL: (2e-10, 2e-8)}
# Absolute tolerances for USR skews (whole molecule, USRCAT atom classes): the cube root of a near-zero third
# moment turns rounding noise into visible values, in RDKit as well (a two-atom class has an exact skew of 0).
USR_SKEW_TOLERANCES = {PrecisionMode.SINGLE: (5e-3, 0.1), PrecisionMode.FULL: (1e-4, 1e-3)}
# Relative tolerances for the DCLV properties, which sum float32 dot contributions over every atom.
DCLV_RELATIVE_TOLERANCES = {PrecisionMode.SINGLE: 1e-5, PrecisionMode.FULL: 1e-9}
# Absolute tolerances for EEM charges.
EEM_TOLERANCES = {PrecisionMode.SINGLE: 2e-6, PrecisionMode.FULL: 1e-9}

PROPERTY_SETS = {
    "single": (Property3D.RADIUS_OF_GYRATION,),
    "all": tuple(Property3D),
}

RDKIT_CALCULATORS = {
    Property3D.PMI1: rdMolDescriptors.CalcPMI1,
    Property3D.PMI2: rdMolDescriptors.CalcPMI2,
    Property3D.PMI3: rdMolDescriptors.CalcPMI3,
    Property3D.RADIUS_OF_GYRATION: rdMolDescriptors.CalcRadiusOfGyration,
    Property3D.NPR1: rdMolDescriptors.CalcNPR1,
    Property3D.NPR2: rdMolDescriptors.CalcNPR2,
    Property3D.INERTIAL_SHAPE_FACTOR: rdMolDescriptors.CalcInertialShapeFactor,
    Property3D.ECCENTRICITY: rdMolDescriptors.CalcEccentricity,
    Property3D.ASPHERICITY: rdMolDescriptors.CalcAsphericity,
}

DCLV_GETTERS = {
    Property3D.DCLV_SURFACE_AREA: "GetSurfaceArea",
    Property3D.DCLV_POLAR_SURFACE_AREA: "GetPolarSurfaceArea",
    Property3D.DCLV_VOLUME: "GetVolume",
    Property3D.DCLV_VDW_VOLUME: "GetVDWVolume",
    Property3D.DCLV_POLAR_VOLUME: "GetPolarVolume",
    Property3D.DCLV_COMPACTNESS: "GetCompactness",
    Property3D.DCLV_PACKING_DENSITY: "GetPackingDensity",
}


def _calc_rdkit_eem(mol: Chem.Mol, conf_id: int) -> list[float]:
    """Reference charges, with NaN for unsupported types or invalid coordinates."""
    missing = [math.nan] * mol.GetNumAtoms()
    molecule = Chem.Mol(mol)
    try:
        Chem.Kekulize(molecule, clearAromaticFlags=True)
    except Chem.MolSanitizeException:
        return missing
    supported_elements = {1: {1, 6, 7, 8, 9, 16, 17, 35}, 2: {6, 7, 8, 15, 16}, 3: {6, 7}}
    for atom in molecule.GetAtoms():
        bond_order = max((int(bond.GetBondTypeAsDouble()) for bond in atom.GetBonds()), default=1)
        if atom.GetAtomicNum() not in supported_elements.get(bond_order, set()):
            return missing
    positions = molecule.GetConformer(conf_id).GetPositions()
    if not np.isfinite(positions).all() or len(np.unique(positions, axis=0)) != len(positions):
        return missing
    return rdMolDescriptors.CalcEEMcharges(molecule, confId=conf_id)


def _calc_rdkit_property(mol: Chem.Mol, conf_id: int, prop: Property3D) -> float | list[float]:
    """Calculate one property using the options mirrored by the benchmark."""
    if prop == Property3D.SPHEROCITY_INDEX:
        return rdMolDescriptors.CalcSpherocityIndex(mol, confId=conf_id)
    if prop == Property3D.PBF:
        reference_mol = Chem.Mol(mol)
        reference_mol.ClearComputedProps()
        return rdMolDescriptors.CalcPBF(reference_mol, confId=conf_id)
    if prop == Property3D.WHIM:
        return rdMolDescriptors.CalcWHIM(mol, confId=conf_id)
    if prop == Property3D.RDF:
        return rdMolDescriptors.CalcRDF(mol, confId=conf_id)
    if prop == Property3D.MORSE:
        return rdMolDescriptors.CalcMORSE(mol, confId=conf_id)
    if prop == Property3D.AUTOCORR3D:
        return rdMolDescriptors.CalcAUTOCORR3D(mol, confId=conf_id)
    if prop in (Property3D.USR, Property3D.USRCAT) and mol.GetNumAtoms() < 3:
        # RDKit raises for fewer than three atoms; nvMolKit reports a NaN row.
        return [math.nan] * (12 if prop == Property3D.USR else 60)
    if prop == Property3D.USR:
        return rdMolDescriptors.GetUSR(mol, confId=conf_id)
    if prop == Property3D.USRCAT:
        return rdMolDescriptors.GetUSRCAT(mol, confId=conf_id)
    if prop == Property3D.EEM_CHARGES:
        return _calc_rdkit_eem(mol, conf_id)
    if prop == Property3D.GETAWAY:
        if len(Chem.GetMolFrags(mol)) > 1:
            # RDKit's GETAWAY does not finish in reasonable time on multi-fragment molecules.
            return [math.nan] * 273
        return rdMolDescriptors.CalcGETAWAY(mol, confId=conf_id)
    return RDKIT_CALCULATORS[prop](mol, confId=conf_id, useAtomicMasses=True)


def _pack_device_coordinates(mols: list[Chem.Mol]) -> Device3DResult:
    """Copy an RDKit conformer batch once into the public device representation."""
    coordinate_arrays: list[np.ndarray] = []
    atom_starts = [0]
    mol_indices: list[int] = []
    conf_indices: list[int] = []
    for mol_idx, mol in enumerate(mols):
        for conf in mol.GetConformers():
            positions = np.asarray(conf.GetPositions(), dtype=np.float64)
            coordinate_arrays.append(positions)
            atom_starts.append(atom_starts[-1] + len(positions))
            mol_indices.append(mol_idx)
            conf_indices.append(conf.GetId())

    if not coordinate_arrays:
        raise ValueError("benchmark batch must contain at least one conformer")
    values = torch.as_tensor(np.concatenate(coordinate_arrays), dtype=torch.float64, device="cuda")
    starts = torch.tensor(atom_starts, dtype=torch.int32, device="cuda")
    molecule_ids = torch.tensor(mol_indices, dtype=torch.int32, device="cuda")
    conformer_ids = torch.tensor(conf_indices, dtype=torch.int32, device="cuda")
    gpu_id = values.device.index
    return Device3DResult(
        values=AsyncGpuResult(values, gpu_id),
        atom_starts=AsyncGpuResult(starts, gpu_id),
        mol_indices=AsyncGpuResult(molecule_ids, gpu_id),
        conf_indices=AsyncGpuResult(conformer_ids, gpu_id),
        gpu_id=gpu_id,
        n_mols=len(mols),
    )


def _dclv_radii(mol: Chem.Mol) -> list[float]:
    """Van der Waals radii per atom; RDKit 2025.09.1 has no default-radii DCLV constructor."""
    table = Chem.GetPeriodicTable()
    return [table.GetRvdw(atom.GetAtomicNum()) for atom in mol.GetAtoms()]


def _calc_rdkit_conformer(
    mol: Chem.Mol, conf_id: int, properties: tuple[Property3D, ...]
) -> dict[Property3D, float | list[float]]:
    dclv = (
        rdMolDescriptors.DoubleCubicLatticeVolume(mol, _dclv_radii(mol), confId=conf_id)
        if any(prop in DCLV_GETTERS for prop in properties)
        else None
    )
    return {
        prop: getattr(dclv, DCLV_GETTERS[prop])() if prop in DCLV_GETTERS else _calc_rdkit_property(mol, conf_id, prop)
        for prop in properties
    }


def _calc_rdkit(mols: list[Chem.Mol], properties: tuple[Property3D, ...]) -> dict[Property3D, np.ndarray]:
    """Calculate reference arrays in molecule/conformer order."""
    values = {prop: [] for prop in properties}
    for mol in mols:
        for conf in mol.GetConformers():
            for prop, value in _calc_rdkit_conformer(mol, conf.GetId(), properties).items():
                values[prop].append(value)
    return {
        prop: np.concatenate([np.asarray(row, dtype=np.float64) for row in rows])
        if prop in PER_ATOM_PROPERTIES
        else np.asarray(rows, dtype=np.float64)
        for prop, rows in values.items()
    }


# RDKit before 2026.9.2 returns wrong EEM charges for charged molecules (rdkit/rdkit#9688); nvMolKit's are correct,
# so only neutral molecules are compared against older RDKit.
RDKIT_EEM_CHARGES_CORRECT = tuple(int(part) for part in re.findall(r"\d+", rdkit.__version__)[:3]) >= (2026, 9, 2)


def _eem_comparable_rows(mols: list[Chem.Mol]) -> np.ndarray:
    """Mask of the per-atom EEM rows, in molecule/conformer order, where the installed RDKit is a valid reference."""
    return np.concatenate(
        [
            np.full(mol.GetNumAtoms(), RDKIT_EEM_CHARGES_CORRECT or Chem.GetFormalCharge(mol) == 0)
            for mol in mols
            for _ in mol.GetConformers()
        ]
        or [np.zeros(0, dtype=bool)]
    )


def _validate(
    mols: list[Chem.Mol],
    properties: tuple[Property3D, ...],
    count: int,
    tolerance: float | None,
    device_input: bool,
    precision: PrecisionMode,
) -> None:
    """Check the nvMolKit input paths being timed against RDKit."""
    check_mols = mols[:count]
    if not check_mols:
        return
    expected = _calc_rdkit(check_mols, properties)
    results = [Calc3DProperties(check_mols, properties, precision=precision)]
    if device_input:
        coordinates = _pack_device_coordinates(check_mols)
        results.append(Calc3DProperties(check_mols, properties, coordinates=coordinates, precision=precision))
    rtol, default_atol = VALIDATION_TOLERANCES[precision]
    atol = default_atol if tolerance is None else tolerance
    for result in results:
        for prop in properties:
            actual = result[prop.value].numpy()
            if prop == Property3D.GETAWAY:
                # Rows RDKit could not compute (multi-fragment molecules) are NaN and not compared.
                computed = ~np.isnan(expected[prop]).all(axis=1)
                np.testing.assert_allclose(actual[computed], expected[prop][computed], rtol=0, atol=max(atol, 1.1e-3))
                continue
            if prop in DCLV_GETTERS:
                np.testing.assert_allclose(actual, expected[prop], rtol=DCLV_RELATIVE_TOLERANCES[precision], atol=atol)
                continue
            if prop == Property3D.EEM_CHARGES:
                eem_atol = EEM_TOLERANCES[precision] if tolerance is None else tolerance
                comparable = _eem_comparable_rows(check_mols)
                np.testing.assert_allclose(
                    actual[comparable], expected[prop][comparable], rtol=0, atol=eem_atol, equal_nan=True
                )
                continue
            if prop not in (Property3D.USR, Property3D.USRCAT):
                np.testing.assert_allclose(actual, expected[prop], rtol=rtol, atol=atol)
                continue
            # Rows of 12-value blocks: 4 reference points x (mean, standard deviation, skew).
            actual = np.asarray(actual, dtype=np.float64).reshape(len(actual), -1, 4, 3)
            reference = np.asarray(expected[prop]).reshape(actual.shape)
            np.testing.assert_allclose(actual[..., :2], reference[..., :2], rtol=rtol, atol=atol)
            whole_molecule_skew, class_skew = (
                USR_SKEW_TOLERANCES[precision] if tolerance is None else (tolerance, tolerance)
            )
            np.testing.assert_allclose(actual[:, :1, :, 2], reference[:, :1, :, 2], rtol=0, atol=whole_molecule_skew)
            np.testing.assert_allclose(actual[:, 1:, :, 2], reference[:, 1:, :, 2], rtol=0, atol=class_skew)


_WORKER_MOLS: list[Chem.Mol] = []
_WORKER_PROPERTIES: tuple[Property3D, ...] = ()


def _rdkit_worker_init(mol_binaries: list[bytes], property_values: list[str]) -> None:
    """Load the batch into a worker once, outside the timed region."""
    global _WORKER_MOLS, _WORKER_PROPERTIES  # noqa: PLW0603 - process-local worker cache
    _WORKER_MOLS = [Chem.Mol(binary) for binary in mol_binaries]
    _WORKER_PROPERTIES = tuple(Property3D(value) for value in property_values)


def _rdkit_worker_range(bounds: tuple[int, int]) -> int:
    """Calculate every property for the worker's molecules in ``[start, stop)``; return conformers done."""
    start, stop = bounds
    done = 0
    for mol in _WORKER_MOLS[start:stop]:
        for conf in mol.GetConformers():
            _calc_rdkit_conformer(mol, conf.GetId(), _WORKER_PROPERTIES)
            done += 1
    return done


def _time_rdkit(
    mols: list[Chem.Mol],
    properties: tuple[Property3D, ...],
    threads: int,
    runs: int,
    warmups: int,
    max_seconds: float,
):
    """Time RDKit across ``threads`` processes; return ``(timing, conformers_measured)``.

    Molecules are sent to the workers once when the pool starts, so the timed region only dispatches
    index ranges. With ``max_seconds > 0`` one deadline is shared across the timed runs and a run stops
    at the next chunk boundary; throughput is then reported on the conformers actually completed.
    """
    chunk = max(1, len(mols) // (threads * 8))
    ranges = [(start, min(start + chunk, len(mols))) for start in range(0, len(mols), chunk)]
    total = sum(mol.GetNumConformers() for mol in mols)
    done = [0]
    initargs = ([mol.ToBinary() for mol in mols], [prop.value for prop in properties])
    with process_pool(threads, _rdkit_worker_init, initargs) as pool:

        def run(deadline: Deadline | None = None) -> None:
            mapped = process_map_bounded(
                _rdkit_worker_range,
                ranges,
                max_workers=threads,
                deadline=deadline if deadline is not None else Deadline(0),
                pool=pool,
            )
            done[0] = sum(mapped.results)

        if max_seconds > 0:
            timing = time_it(
                run,
                runs=runs,
                warmups=0,
                max_seconds=max_seconds,
                progress_getter=lambda: done[0],
                progress_target=total,
            )
            measured = timing.progress if timing.progress is not None else done[0]
        else:
            timing = time_it(run, runs=runs, warmups=warmups)
            measured = total
    return timing, measured


def _timing_fields(prefix: str, result, num_conformers: int) -> dict[str, float]:
    return {
        f"{prefix}_median_ms": result.median_ms,
        f"{prefix}_std_ms": result.std_ms,
        f"{prefix}_conformers_per_s": throughput_per_s(num_conformers, result.median_ms),
    }


def run(
    smiles_path: str,
    num_mols_list: list[int],
    conformers_per_mol_list: list[int],
    property_set_names: list[str],
    runs: int,
    warmups: int,
    seed: int,
    prep_workers: int,
    prep_threads: int,
    rdkit_threads: int,
    rdkit_max_seconds: float,
    validate_count: int,
    validate_tolerance: float | None,
    device_input: bool,
    precision: PrecisionMode,
    no_rdkit: bool,
    no_nvmolkit: bool,
    output: str | None,
    exclude: tuple[str, ...] = (),
) -> list[dict[str, float | int | str]]:
    """Prepare one maximum-size batch and benchmark requested sweep points."""
    if no_rdkit and no_nvmolkit:
        raise ValueError("cannot disable both RDKit and nvMolKit")
    if any(count < 1 for count in num_mols_list):
        raise ValueError("every --num_mols value must be positive")
    if any(count < 1 for count in conformers_per_mol_list):
        raise ValueError("every --confs_per_mol value must be positive")

    unknown = sorted(set(exclude) - {prop.value for prop in Property3D})
    if unknown:
        raise ValueError(f"--exclude names unknown properties: {', '.join(unknown)}")
    selections = {
        name: tuple(prop for prop in PROPERTY_SETS[name] if prop.value not in exclude) for name in property_set_names
    }
    emptied = [name for name, properties in selections.items() if not properties]
    if emptied:
        raise ValueError(f"--exclude removes every property from: {', '.join(emptied)}")

    max_mols = max(num_mols_list)
    hardware_options = HardwareOptions(preprocessingThreads=prep_threads)
    max_conformers = max(conformers_per_mol_list)
    raw_mols = load_smiles(smiles_path, max_count=max_mols, sanitize=True, seed=seed)
    workers = prep_workers if prep_workers > 0 else available_cpu_count()
    prepared = embed_and_jitter(
        raw_mols,
        confs_per_mol=max_conformers,
        seed=seed,
        num_workers=workers,
        add_hs=True,
        min_atoms=1,
        desc=f"Embed + perturb ({max_conformers} confs)",
    )
    if not prepared:
        raise RuntimeError("no molecules survived conformer preparation")

    rows: list[dict[str, float | int | str]] = []
    for requested_mols in sorted(set(num_mols_list)):
        for requested_conformers in sorted(set(conformers_per_mol_list)):
            mols = slice_conformers(prepared[:requested_mols], requested_conformers)
            num_conformers = sum(mol.GetNumConformers() for mol in mols)
            avg_atoms = sum(mol.GetNumAtoms() for mol in mols) / len(mols)
            coordinates = _pack_device_coordinates(mols) if device_input and not no_nvmolkit else None

            for property_set_name in property_set_names:
                properties = selections[property_set_name]
                excluded = [prop.value for prop in PROPERTY_SETS[property_set_name] if prop.value in exclude]
                print(
                    f"\n=== {len(mols)} mols, {num_conformers} conformers, "
                    f"{avg_atoms:.1f} atoms/mol, properties={property_set_name} ==="
                )
                if validate_count > 0 and not no_rdkit and not no_nvmolkit:
                    _validate(
                        mols, properties, min(validate_count, len(mols)), validate_tolerance, device_input, precision
                    )

                row: dict[str, float | int | str] = {
                    "num_mols": len(mols),
                    "conformers_per_mol": requested_conformers,
                    "num_conformers": num_conformers,
                    "avg_atoms": avg_atoms,
                    "property_set": property_set_name + "".join(f"-{name}" for name in excluded),
                    "precision": str(precision),
                    "prep_threads": prep_threads,
                    "num_properties": len(properties),
                }

                rdkit_timing = None
                if not no_rdkit and Property3D.GETAWAY in properties:
                    skipped = sum(len(Chem.GetMolFrags(mol)) > 1 for mol in mols)
                    if skipped:
                        print(f"RDKit GETAWAY skips {skipped} multi-fragment molecules (it would not finish on them)")
                    row["rdkit_getaway_skipped_mols"] = skipped
                if not no_rdkit:
                    rdkit_timing, rdkit_conformers = _time_rdkit(
                        mols, properties, rdkit_threads, runs, warmups, rdkit_max_seconds
                    )
                    row.update(_timing_fields("rdkit", rdkit_timing, rdkit_conformers))
                    row["rdkit_threads"] = rdkit_threads
                    row["rdkit_conformers_processed"] = rdkit_conformers

                if not no_nvmolkit:
                    nvmolkit_timing = time_it(
                        lambda mols=mols, properties=properties: Calc3DProperties(
                            mols, properties, precision=precision, hardwareOptions=hardware_options
                        ),
                        runs=runs,
                        warmups=warmups,
                        gpu_sync=True,
                    )
                    row.update(_timing_fields("nvmolkit", nvmolkit_timing, num_conformers))
                    if rdkit_timing is not None and row["rdkit_conformers_per_s"] > 0:
                        row["speedup"] = row["nvmolkit_conformers_per_s"] / row["rdkit_conformers_per_s"]
                    if coordinates is not None:
                        device_timing = time_it(
                            lambda mols=mols, properties=properties, coordinates=coordinates: Calc3DProperties(
                                mols,
                                properties,
                                coordinates=coordinates,
                                precision=precision,
                                hardwareOptions=hardware_options,
                            ),
                            runs=runs,
                            warmups=warmups,
                            gpu_sync=True,
                        )
                        row.update(_timing_fields("nvmolkit_device_input", device_timing, num_conformers))
                        if rdkit_timing is not None and row["rdkit_conformers_per_s"] > 0:
                            row["device_input_speedup"] = (
                                row["nvmolkit_device_input_conformers_per_s"] / row["rdkit_conformers_per_s"]
                            )

                rows.append(row)

    print("\nCSV Results:")
    print_csv_rows(rows)
    if output:
        write_csv_rows(rows, Path(output))
        print(f"\nWrote {output}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="3D property benchmark")
    parser.add_argument("--smiles", required=True, help="Path to a SMILES file")
    parser.add_argument("--num_mols", type=int, nargs="+", default=[100, 1000])
    parser.add_argument("--confs_per_mol", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--property_sets", choices=tuple(PROPERTY_SETS), nargs="+", default=list(PROPERTY_SETS))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prep_workers",
        type=int,
        default=0,
        help="Workers for the untimed embed-and-perturb prep (0 = all available CPUs)",
    )
    parser.add_argument(
        "--prep_threads",
        type=int,
        default=-1,
        help="nvMolKit CPU preprocessing threads inside the timed Calc3DProperties call (-1 = library default)",
    )
    parser.add_argument("--rdkit_threads", type=int, default=1, help="Processes for the RDKit comparison (default: 1)")
    add_rdkit_max_seconds_arg(
        parser, extra_help="One deadline is shared across timed runs; a run stops at the next chunk boundary."
    )
    parser.add_argument("--validate_count", type=int, default=8)
    parser.add_argument(
        "--validate_tolerance",
        type=float,
        default=None,
        help="Absolute RDKit validation tolerance (default depends on --precision)",
    )
    parser.add_argument("--precision", choices=tuple(PRECISIONS), default="single")
    parser.add_argument("--no_validate", action="store_true")
    parser.add_argument(
        "--device_input",
        action="store_true",
        help="Also time calculation from pre-staged device coordinates (chained-workflow path)",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        choices=[prop.value for prop in Property3D],
        help="Property names to drop from every property set, e.g. to time 'all' without a new feature",
    )
    parser.add_argument("--output", default=None, help="Optional CSV output path")
    add_backend_selection_args(parser)
    args = parser.parse_args()
    run(
        smiles_path=args.smiles,
        num_mols_list=args.num_mols,
        conformers_per_mol_list=args.confs_per_mol,
        property_set_names=args.property_sets,
        runs=args.runs,
        warmups=args.warmups,
        seed=args.seed,
        prep_workers=args.prep_workers,
        prep_threads=args.prep_threads,
        rdkit_threads=args.rdkit_threads,
        rdkit_max_seconds=args.rdkit_max_seconds,
        validate_count=0 if args.no_validate else args.validate_count,
        validate_tolerance=args.validate_tolerance,
        precision=PRECISIONS[args.precision],
        device_input=args.device_input,
        no_rdkit=args.no_rdkit,
        no_nvmolkit=args.no_nvmolkit,
        output=args.output,
        exclude=tuple(args.exclude),
    )


if __name__ == "__main__":
    main()
