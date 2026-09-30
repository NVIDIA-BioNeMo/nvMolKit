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
from pathlib import Path

import numpy as np
import torch
from bench_utils import (
    add_backend_selection_args,
    available_cpu_count,
    embed_and_jitter,
    load_smiles,
    print_csv_rows,
    slice_conformers,
    throughput_per_s,
    time_it,
    write_csv_rows,
)
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from nvmolkit.descriptors3d import Calc3DProperties, Property3D
from nvmolkit.types import AsyncGpuResult, Device3DResult, PrecisionMode

PRECISIONS = {"single": PrecisionMode.SINGLE, "full": PrecisionMode.FULL}
# Default (relative, absolute) tolerances against RDKit at each precision's rounding level.
VALIDATION_TOLERANCES = {PrecisionMode.SINGLE: (2e-6, 1e-3), PrecisionMode.FULL: (2e-10, 2e-8)}

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


def _calc_rdkit(mols: list[Chem.Mol], properties: tuple[Property3D, ...]) -> dict[Property3D, np.ndarray]:
    """Calculate reference arrays in molecule/conformer order."""
    return {
        prop: np.asarray(
            [_calc_rdkit_property(mol, conf.GetId(), prop) for mol in mols for conf in mol.GetConformers()],
            dtype=np.float64,
        )
        for prop in properties
    }


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
            np.testing.assert_allclose(result[prop.value].numpy(), expected[prop], rtol=rtol, atol=atol)


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
    validate_count: int,
    validate_tolerance: float | None,
    device_input: bool,
    precision: PrecisionMode,
    no_rdkit: bool,
    no_nvmolkit: bool,
    output: str | None,
) -> list[dict[str, float | int | str]]:
    """Prepare one maximum-size batch and benchmark requested sweep points."""
    if no_rdkit and no_nvmolkit:
        raise ValueError("cannot disable both RDKit and nvMolKit")
    if any(count < 1 for count in num_mols_list):
        raise ValueError("every --num_mols value must be positive")
    if any(count < 1 for count in conformers_per_mol_list):
        raise ValueError("every --confs_per_mol value must be positive")

    max_mols = max(num_mols_list)
    max_conformers = max(conformers_per_mol_list)
    raw_mols = load_smiles(smiles_path, max_count=max_mols, sanitize=True, seed=seed)
    workers = prep_workers if prep_workers > 0 else max(1, available_cpu_count() // 2)
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
                properties = PROPERTY_SETS[property_set_name]
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
                    "property_set": property_set_name,
                    "precision": str(precision),
                    "num_properties": len(properties),
                }

                rdkit_timing = None
                if not no_rdkit:
                    rdkit_timing = time_it(
                        lambda mols=mols, properties=properties: _calc_rdkit(mols, properties),
                        runs=runs,
                        warmups=warmups,
                    )
                    row.update(_timing_fields("rdkit", rdkit_timing, num_conformers))

                if not no_nvmolkit:
                    nvmolkit_timing = time_it(
                        lambda mols=mols, properties=properties: Calc3DProperties(
                            mols, properties, precision=precision
                        ),
                        runs=runs,
                        warmups=warmups,
                        gpu_sync=True,
                    )
                    row.update(_timing_fields("nvmolkit", nvmolkit_timing, num_conformers))
                    if rdkit_timing is not None:
                        row["speedup"] = rdkit_timing.median_ms / nvmolkit_timing.median_ms
                    if coordinates is not None:
                        device_timing = time_it(
                            lambda mols=mols, properties=properties, coordinates=coordinates: Calc3DProperties(
                                mols, properties, coordinates=coordinates, precision=precision
                            ),
                            runs=runs,
                            warmups=warmups,
                            gpu_sync=True,
                        )
                        row.update(_timing_fields("nvmolkit_device_input", device_timing, num_conformers))
                        if rdkit_timing is not None:
                            row["device_input_speedup"] = rdkit_timing.median_ms / device_timing.median_ms

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
    parser.add_argument("--prep_workers", type=int, default=0, help="Preparation workers (0 = half of available CPUs)")
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
        validate_count=0 if args.no_validate else args.validate_count,
        validate_tolerance=args.validate_tolerance,
        precision=PRECISIONS[args.precision],
        device_input=args.device_input,
        no_rdkit=args.no_rdkit,
        no_nvmolkit=args.no_nvmolkit,
        output=args.output,
    )


if __name__ == "__main__":
    main()
