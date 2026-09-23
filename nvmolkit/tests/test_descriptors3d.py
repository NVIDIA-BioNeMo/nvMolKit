# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc

import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdMolDescriptors
from rdkit.Chem.rdDistGeom import EmbedParameters
from rdkit.Geometry import Point3D

from nvmolkit.descriptors3d import Calc3DProperties, Device3DPropertyResult, Property3D
from nvmolkit.embedMolecules import EmbedMolecules
from nvmolkit.types import AsyncGpuResult, CoordinateOutput, Device3DResult, PrecisionMode

PRECISIONS = [PrecisionMode.SINGLE, PrecisionMode.FULL]


def _assert_matches_rdkit(actual, expected, precision=PrecisionMode.SINGLE):
    """Compare against RDKit at the rounding level of the precision mode."""
    actual = np.asarray(actual)
    if precision == PrecisionMode.FULL:
        assert actual.dtype == np.float64
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-8)
    else:
        assert actual.dtype == np.float32
        # Near-zero moments (linear molecules) are resolved relative to the batch's moment scale.
        scale = max(float(np.abs(expected).max(initial=0.0)), 1.0)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6 * scale)


def _embed(smiles, num_confs, seed):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = seed
    rdDistGeom.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    return mol


def _reference_rows(mols, properties, use_atomic_masses):
    calculators = {
        Property3D.PMI1: rdMolDescriptors.CalcPMI1,
        Property3D.PMI2: rdMolDescriptors.CalcPMI2,
        Property3D.PMI3: rdMolDescriptors.CalcPMI3,
        Property3D.RADIUS_OF_GYRATION: rdMolDescriptors.CalcRadiusOfGyration,
    }
    rows = [
        [calculators[prop](mol, confId=conf.GetId(), useAtomicMasses=use_atomic_masses) for prop in properties]
        for mol in mols
        for conf in mol.GetConformers()
    ]
    return np.asarray(rows, dtype=np.float64)


def _device_result_from_molecules(mols):
    coordinate_arrays = []
    atom_starts = [0]
    mol_indices = []
    conf_indices = []
    for mol_idx, mol in enumerate(mols):
        for conf in mol.GetConformers():
            positions = np.asarray(conf.GetPositions(), dtype=np.float64)
            coordinate_arrays.append(positions)
            atom_starts.append(atom_starts[-1] + len(positions))
            mol_indices.append(mol_idx)
            conf_indices.append(conf.GetId())

    values = torch.as_tensor(np.concatenate(coordinate_arrays), dtype=torch.float64, device="cuda")
    starts = torch.tensor(atom_starts, dtype=torch.int32, device="cuda")
    mol_idx = torch.tensor(mol_indices, dtype=torch.int32, device="cuda")
    conf_idx = torch.tensor(conf_indices, dtype=torch.int32, device="cuda")
    gpu_id = values.device.index
    return Device3DResult(
        values=AsyncGpuResult(values, gpu_id),
        atom_starts=AsyncGpuResult(starts, gpu_id),
        mol_indices=AsyncGpuResult(mol_idx, gpu_id),
        conf_indices=AsyncGpuResult(conf_idx, gpu_id),
        gpu_id=gpu_id,
        n_mols=len(mols),
    )


def _mol_with_conformers(smiles, coordinate_sets):
    mol = Chem.MolFromSmiles(smiles)
    for coordinates in coordinate_sets:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for atom_idx, (x, y, z) in enumerate(coordinates):
            conf.SetAtomPosition(atom_idx, Point3D(float(x), float(y), float(z)))
        mol.AddConformer(conf, assignId=True)
    return mol


def _reference_device_rows(mols, coordinates, properties, use_atomic_masses=True):
    calculators = {
        Property3D.PMI1: rdMolDescriptors.CalcPMI1,
        Property3D.PMI2: rdMolDescriptors.CalcPMI2,
        Property3D.PMI3: rdMolDescriptors.CalcPMI3,
        Property3D.RADIUS_OF_GYRATION: rdMolDescriptors.CalcRadiusOfGyration,
    }
    values = coordinates.values.numpy()
    atom_starts = coordinates.atom_starts.torch().tolist()
    mol_indices = coordinates.mol_indices.torch().tolist()
    rows = []
    for row_idx, mol_idx in enumerate(mol_indices):
        mol = Chem.Mol(mols[mol_idx])
        mol.RemoveAllConformers()
        conf = Chem.Conformer(mol.GetNumAtoms())
        positions = values[atom_starts[row_idx] : atom_starts[row_idx + 1]]
        for atom_idx, (x, y, z) in enumerate(positions):
            conf.SetAtomPosition(atom_idx, Point3D(float(x), float(y), float(z)))
        conf_id = mol.AddConformer(conf, assignId=True)
        rows.append([calculators[prop](mol, confId=conf_id, useAtomicMasses=use_atomic_masses) for prop in properties])
    return np.asarray(rows, dtype=np.float64)


@pytest.mark.parametrize("use_atomic_masses", [True, False])
@pytest.mark.parametrize("precision", PRECISIONS)
def test_shape_properties_match_rdkit_for_mixed_batch_and_selection(use_atomic_masses, precision):
    # Hexadecane (50 atoms with Hs) takes several strides through the per-conformer atom loop.
    mols = [_embed("CCO", 3, 7), _embed("c1ccccc1", 2, 11), Chem.MolFromSmiles("CC"), _embed("C" * 16, 2, 13)]
    properties = (
        Property3D.RADIUS_OF_GYRATION,
        Property3D.PMI3,
        Property3D.PMI1,
        Property3D.PMI2,
    )

    result = Calc3DProperties(mols, properties, useAtomicMasses=use_atomic_masses, precision=precision)
    expected = _reference_rows(mols, properties, use_atomic_masses)

    assert isinstance(result, Device3DPropertyResult)
    assert tuple(result) == tuple(prop.value for prop in properties)
    for column, prop in enumerate(properties):
        assert isinstance(result[prop.value], AsyncGpuResult)
        assert result[prop.value].torch().shape == (expected.shape[0],)
        _assert_matches_rdkit(result[prop.value].numpy(), expected[:, column], precision)


def test_rows_are_labeled_by_molecule_and_conformer_position():
    mols = [_embed("CCO", 3, 7), Chem.MolFromSmiles("CC"), _embed("c1ccccc1", 2, 11)]
    result = Calc3DProperties(mols, (Property3D.PMI1, Property3D.RADIUS_OF_GYRATION))
    expected = _reference_rows(mols, (Property3D.PMI1, Property3D.RADIUS_OF_GYRATION), True)

    assert result.n_mols == 3
    assert result.n_conformers == 5
    assert result.gpu_id == result["PMI1"].device.index
    assert result.mol_indices.torch().tolist() == [0, 0, 0, 2, 2]
    assert result.conf_indices.torch().tolist() == [0, 1, 2, 0, 1]
    assert result[Property3D.RADIUS_OF_GYRATION] is result["RadiusOfGyration"]
    assert Property3D.PMI1 in result and "PMI2" not in result and "NotAProperty" not in result

    dense = result.dense()
    assert dense.conf_mask.tolist() == [[True, True, True], [False, False, False], [True, True, False]]
    assert tuple(dense.values) == ("PMI1", "RadiusOfGyration")
    for column, name in enumerate(dense.values):
        values = dense.values[name]
        assert values.shape == (3, 3)
        assert torch.isnan(values[~dense.conf_mask]).all()
        _assert_matches_rdkit(values[dense.conf_mask].cpu().numpy(), expected[:, column])

    empty = Calc3DProperties([Chem.MolFromSmiles("CC")], "PMI1").dense(pad_value=0.0)
    assert empty.values["PMI1"].shape == (1, 0) and empty.conf_mask.shape == (1, 0)


def test_device_coordinates_are_reused_and_keep_requested_schema():
    mols = [_embed("CCCC", 4, 19), _embed("CC(=O)O", 2, 23)]
    coordinates = _device_result_from_molecules(mols)
    coordinate_pointer = coordinates.values.torch().data_ptr()
    properties = ("PMI2", "RadiusOfGyration")

    result = Calc3DProperties(mols, properties, coordinates=coordinates)
    expected_properties = (Property3D.PMI2, Property3D.RADIUS_OF_GYRATION)
    expected = _reference_rows(mols, expected_properties, True)

    assert coordinates.values.torch().data_ptr() == coordinate_pointer
    assert result.mol_indices is coordinates.mol_indices
    assert result.conf_indices is coordinates.conf_indices
    assert tuple(result) == properties
    for column, property_name in enumerate(properties):
        _assert_matches_rdkit(result[property_name].numpy(), expected[:, column])


def test_explicit_stream_and_result_accessors():
    mol = _embed("CCN", 3, 31)
    properties = (Property3D.PMI1, Property3D.RADIUS_OF_GYRATION)
    stream = torch.cuda.Stream()

    result = Calc3DProperties(mol, properties, stream=stream)
    stream.synchronize()
    expected = _reference_rows([mol], properties, True)

    assert result["PMI1"].device == stream.device
    assert set(result) == {"PMI1", "RadiusOfGyration"}
    converted = {name: value.torch() for name, value in result.items()}
    assert converted["PMI1"].shape == (mol.GetNumConformers(),)
    for column, property_name in enumerate(("PMI1", "RadiusOfGyration")):
        _assert_matches_rdkit(result[property_name].numpy(), expected[:, column])

    single_property = Calc3DProperties(mol, "PMI2")
    assert tuple(single_property) == ("PMI2",)
    assert single_property["PMI2"].torch().shape == (mol.GetNumConformers(),)


@pytest.mark.parametrize("use_atomic_masses", [True, False])
@pytest.mark.parametrize("precision", PRECISIONS)
def test_degenerate_geometries_and_empty_inputs_match_rdkit(use_atomic_masses, precision):
    mols = [
        _mol_with_conformers("[He]", [[(4.0, -3.0, 2.0)]]),
        _mol_with_conformers("CCC", [[(-2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (3.0, 0.0, 0.0)]]),
        _mol_with_conformers("CCO", [[(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.5, 1.5, 0.0)]]),
        _mol_with_conformers("CC", [[(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)]]),
        Chem.MolFromSmiles("c1ccccc1"),
    ]
    properties = tuple(Property3D)

    result = Calc3DProperties(mols, properties, useAtomicMasses=use_atomic_masses, precision=precision)
    expected = _reference_rows(mols, properties, use_atomic_masses)

    assert isinstance(result, Device3DPropertyResult)
    for column, prop in enumerate(properties):
        _assert_matches_rdkit(result[prop.value].numpy(), expected[:, column], precision)

    empty_batch = Calc3DProperties([], properties)
    no_conformers = Calc3DProperties([Chem.MolFromSmiles("CC")], "PMI1")
    assert all(value.torch().shape == (0,) for value in empty_batch.values())
    assert no_conformers["PMI1"].torch().shape == (0,)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_embed_device_output_chains_directly_into_3d_properties(precision):
    mols = [Chem.AddHs(Chem.MolFromSmiles("CCO")), Chem.AddHs(Chem.MolFromSmiles("CCCO"))]
    params = EmbedParameters()
    params.useRandomCoords = True
    params.randomSeed = 0xBEEF
    coordinates = EmbedMolecules(
        mols,
        params,
        confsPerMolecule=3,
        output=CoordinateOutput.DEVICE,
    )
    properties = tuple(Property3D)

    result = Calc3DProperties(mols, properties, coordinates=coordinates, precision=precision)
    expected = _reference_device_rows(mols, coordinates, properties)

    assert all(mol.GetNumConformers() == 0 for mol in mols)
    assert isinstance(result, Device3DPropertyResult)
    assert tuple(result) == tuple(prop.value for prop in properties)
    for column, prop in enumerate(properties):
        _assert_matches_rdkit(result[prop.value].numpy(), expected[:, column], precision)


@pytest.mark.parametrize(("atom_starts", "bad_row"), [([-4, 0, 4], 0), ([4, 8, 12], 1)])
def test_device_atom_starts_outside_values_produce_nan(atom_starts, bad_row):
    square = [(1.0, 1.0, 0.0), (-1.0, 1.0, 0.0), (-1.0, -1.0, 0.0), (1.0, -1.0, 0.0)]
    mols = [_mol_with_conformers("C1CCC1", [square, square])]
    valid = _device_result_from_molecules(mols)
    gpu_id = valid.gpu_id
    # Every row keeps the molecule's 4-atom count, so only the offset bounds reject the bad row.
    coordinates = Device3DResult(
        values=valid.values,
        atom_starts=AsyncGpuResult(torch.tensor(atom_starts, dtype=torch.int32, device="cuda"), gpu_id),
        mol_indices=valid.mol_indices,
        conf_indices=valid.conf_indices,
        gpu_id=gpu_id,
        n_mols=1,
    )

    result = Calc3DProperties(
        mols, "RadiusOfGyration", coordinates=coordinates, useAtomicMasses=False, precision=PrecisionMode.FULL
    )
    values = result["RadiusOfGyration"].numpy()
    assert np.isnan(values[bad_row])
    np.testing.assert_allclose(values[1 - bad_row], np.sqrt(2.0), rtol=1e-12)


def test_extracted_async_result_keeps_device_inputs_alive_on_explicit_stream():
    mols = [_embed("CCN", 4, 41), _embed("CCCO", 2, 43)]
    expected = _reference_rows(mols, (Property3D.PMI3,), True)[:, 0]
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        coordinates = _device_result_from_molecules(mols)
        result = Calc3DProperties(mols, ("PMI3",), coordinates=coordinates, stream=stream)
        pmi3 = result["PMI3"]

    del result
    del coordinates
    gc.collect()
    _assert_matches_rdkit(pmi3.numpy(), expected)


def test_property_and_coordinate_contract_errors_are_clear():
    mol = _embed("CCO", 1, 37)
    with pytest.raises(ValueError, match="at least one"):
        Calc3DProperties(mol, [])
    with pytest.raises(ValueError, match="duplicates"):
        Calc3DProperties(mol, ["PMI1", "PMI1"])
    with pytest.raises(ValueError, match="Unknown 3D property"):
        Calc3DProperties(mol, ["NotAProperty"])

    with pytest.raises(TypeError, match="Device3DResult"):
        Calc3DProperties(mol, ["PMI1"], coordinates=object())

    coordinates = _device_result_from_molecules([mol])
    with pytest.raises(ValueError, match=r"coordinates\.n_mols"):
        Calc3DProperties([mol, mol], ["PMI1"], coordinates=coordinates)
