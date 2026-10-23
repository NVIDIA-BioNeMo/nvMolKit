# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import re

import numpy as np
import pytest
import rdkit
import torch
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdMolDescriptors
from rdkit.Chem.rdDistGeom import EmbedParameters
from rdkit.Geometry import Point3D

import nvmolkit.descriptors3d as descriptors3d_module
from nvmolkit.descriptors3d import (
    PER_ATOM_PROPERTIES,
    Calc3DProperties,
    DclvOptions,
    Device3DPropertyResult,
    GetawayOptions,
    MomentOptions,
    Property3D,
    Property3DOptions,
    WhimOptions,
)
from nvmolkit.embedMolecules import EmbedMolecules
from nvmolkit.types import AsyncGpuResult, CoordinateOutput, Device3DResult, HardwareOptions, PrecisionMode

PRECISIONS = [PrecisionMode.SINGLE, PrecisionMode.FULL]
VECTOR_PROPERTIES = (
    Property3D.WHIM,
    Property3D.RDF,
    Property3D.MORSE,
    Property3D.AUTOCORR3D,
    Property3D.USR,
    Property3D.USRCAT,
    Property3D.GETAWAY,
)
DCLV_PROPERTIES = (
    Property3D.DCLV_SURFACE_AREA,
    Property3D.DCLV_POLAR_SURFACE_AREA,
    Property3D.DCLV_VOLUME,
    Property3D.DCLV_VDW_VOLUME,
    Property3D.DCLV_POLAR_VOLUME,
    Property3D.DCLV_COMPACTNESS,
    Property3D.DCLV_PACKING_DENSITY,
)
# Shape and moment scalars, compared against RDKit by _rdkit_property; the DCLV scalars have their own tests.
SCALAR_PROPERTIES = tuple(
    prop for prop in Property3D if prop not in VECTOR_PROPERTIES + DCLV_PROPERTIES and prop not in PER_ATOM_PROPERTIES
)
PAIRWISE_PROPERTIES = (Property3D.RDF, Property3D.MORSE, Property3D.AUTOCORR3D)
RDKIT_PAIRWISE = {
    Property3D.RDF: rdMolDescriptors.CalcRDF,
    Property3D.MORSE: rdMolDescriptors.CalcMORSE,
    Property3D.AUTOCORR3D: rdMolDescriptors.CalcAUTOCORR3D,
}
USR_PROPERTIES = (Property3D.USR, Property3D.USRCAT)
RDKIT_USR = {Property3D.USR: rdMolDescriptors.GetUSR, Property3D.USRCAT: rdMolDescriptors.GetUSRCAT}


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


def _assert_rounded_matches_rdkit(actual, expected, precision):
    """Compare WHIM, RDF or MORSE rows, which RDKit rounds to thousandths, against RDKit.

    One rounding unit (0.001) covers a value whose last digit rounds differently. SINGLE additionally
    allows the float32 conversion of the rounded values.
    """
    rtol = 0 if precision == PrecisionMode.FULL else 1e-6
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=0.0011, equal_nan=True)


def _moment_options(use_atomic_masses):
    return Property3DOptions(moments=MomentOptions(useAtomicMasses=use_atomic_masses))


def _embed(smiles, num_confs, seed):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = seed
    rdDistGeom.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    return mol


def _rdkit_property(mol, conf_id, prop, use_atomic_masses):
    calculators = {
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
    if prop == Property3D.SPHEROCITY_INDEX:
        return rdMolDescriptors.CalcSpherocityIndex(mol, confId=conf_id)
    if prop == Property3D.PBF:
        reference_mol = Chem.Mol(mol)
        reference_mol.ClearComputedProps()
        return rdMolDescriptors.CalcPBF(reference_mol, confId=conf_id)
    if prop == Property3D.WHIM:
        return rdMolDescriptors.CalcWHIM(mol, confId=conf_id)
    return calculators[prop](mol, confId=conf_id, useAtomicMasses=use_atomic_masses)


def _reference_rows(mols, properties, use_atomic_masses):
    rows = [
        [_rdkit_property(mol, conf.GetId(), prop, use_atomic_masses) for prop in properties]
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
        conf.Set3D(True)
        for atom_idx, (x, y, z) in enumerate(coordinates):
            conf.SetAtomPosition(atom_idx, Point3D(float(x), float(y), float(z)))
        mol.AddConformer(conf, assignId=True)
    return mol


def _reference_device_rows(mols, coordinates, properties, use_atomic_masses=True):
    values = coordinates.values.numpy()
    atom_starts = coordinates.atom_starts.torch().tolist()
    mol_indices = coordinates.mol_indices.torch().tolist()
    rows = []
    for row_idx, mol_idx in enumerate(mol_indices):
        mol = Chem.Mol(mols[mol_idx])
        mol.RemoveAllConformers()
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.Set3D(True)
        positions = values[atom_starts[row_idx] : atom_starts[row_idx + 1]]
        for atom_idx, (x, y, z) in enumerate(positions):
            conf.SetAtomPosition(atom_idx, Point3D(float(x), float(y), float(z)))
        conf_id = mol.AddConformer(conf, assignId=True)
        rows.append([_rdkit_property(mol, conf_id, prop, use_atomic_masses) for prop in properties])
    return np.asarray(rows, dtype=np.float64)


@pytest.mark.parametrize("use_atomic_masses", [True, False])
@pytest.mark.parametrize("precision", PRECISIONS)
def test_shape_properties_match_rdkit_for_mixed_batch_and_selection(use_atomic_masses, precision):
    # Hexadecane (50 atoms with Hs) takes several strides through the per-conformer atom loop.
    mols = [_embed("CCO", 3, 7), _embed("c1ccccc1", 2, 11), Chem.MolFromSmiles("CC"), _embed("C" * 16, 2, 13)]
    properties = SCALAR_PROPERTIES

    result = Calc3DProperties(mols, properties, options=_moment_options(use_atomic_masses), precision=precision)
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
    properties = SCALAR_PROPERTIES

    result = Calc3DProperties(mols, properties, options=_moment_options(use_atomic_masses), precision=precision)
    expected = _reference_rows(mols, properties, use_atomic_masses)

    assert isinstance(result, Device3DPropertyResult)
    for column, prop in enumerate(properties):
        _assert_matches_rdkit(result[prop.value].numpy(), expected[:, column], precision)

    empty_batch = Calc3DProperties([], properties)
    no_conformers = Calc3DProperties([Chem.MolFromSmiles("CC")], "PMI1")
    assert all(value.torch().shape == (0,) for value in empty_batch.values())
    assert no_conformers["PMI1"].torch().shape == (0,)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_spherocity_ignores_atomic_mass_option(precision):
    mol = _mol_with_conformers(
        "COPF",
        [[(0.0, 0.0, 0.0), (1.4, 0.1, 0.2), (-0.3, 1.7, -0.1), (0.2, -0.4, 2.1)]],
    )
    properties = (Property3D.SPHEROCITY_INDEX,)

    mass_weighted = Calc3DProperties(mol, properties, options=_moment_options(True), precision=precision)
    unit_weighted = Calc3DProperties(mol, properties, options=_moment_options(False), precision=precision)
    expected = _reference_rows([mol], properties, use_atomic_masses=False)

    for column, prop in enumerate(properties):
        _assert_matches_rdkit(mass_weighted[prop].numpy(), expected[:, column], precision)
        np.testing.assert_array_equal(mass_weighted[prop].numpy(), unit_weighted[prop].numpy())


@pytest.mark.parametrize("precision", PRECISIONS)
def test_scalar_properties_are_translation_invariant_far_from_origin(precision):
    mols = [_embed("CC(=O)Nc1ccc(O)cc1", 2, 59), _embed("c1ccc2ccccc2c1", 1, 61)]
    translated = []
    for mol in mols:
        moved = Chem.Mol(mol)
        for conf in moved.GetConformers():
            for atom_idx in range(moved.GetNumAtoms()):
                position = conf.GetAtomPosition(atom_idx)
                conf.SetAtomPosition(atom_idx, Point3D(position.x + 1e4, position.y - 4e3, position.z + 6e3))
        translated.append(moved)

    near = Calc3DProperties(mols, SCALAR_PROPERTIES, precision=precision)
    far = Calc3DProperties(translated, SCALAR_PROPERTIES, precision=precision)

    for prop in SCALAR_PROPERTIES:
        expected = near[prop].numpy()
        scale = max(float(np.abs(expected).max(initial=0.0)), 1.0)
        np.testing.assert_allclose(far[prop].numpy(), expected, rtol=2e-6, atol=2e-6 * scale, err_msg=prop.value)

    near_pairwise = Calc3DProperties(mols, PAIRWISE_PROPERTIES, precision=precision)
    far_pairwise = Calc3DProperties(translated, PAIRWISE_PROPERTIES, precision=precision)
    for prop in PAIRWISE_PROPERTIES:
        _assert_rounded_matches_rdkit(far_pairwise[prop].numpy(), near_pairwise[prop].numpy(), precision)
    near_usr = Calc3DProperties(mols, USR_PROPERTIES, precision=precision)
    far_usr = Calc3DProperties(translated, USR_PROPERTIES, precision=precision)
    for prop in USR_PROPERTIES:
        _assert_usr_matches(
            far_usr[prop].numpy(),
            near_usr[prop].numpy(),
            precision,
            class_blocks=prop == Property3D.USRCAT,
        )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_projection_family_matches_rdkit_and_preserves_vector_shape(precision):
    mols = [_embed("CCCO", 3, 47), _embed("c1ccncc1", 2, 53)]
    threshold = 0.01
    result = Calc3DProperties(
        mols,
        (Property3D.PBF, Property3D.WHIM),
        options=Property3DOptions(whim=WhimOptions(threshold=threshold)),
        precision=precision,
    )
    expected_pbf = np.asarray(
        [_rdkit_property(mol, conf.GetId(), Property3D.PBF, True) for mol in mols for conf in mol.GetConformers()]
    )
    expected_whim = np.asarray(
        [
            rdMolDescriptors.CalcWHIM(mol, confId=conf.GetId(), thresh=threshold)
            for mol in mols
            for conf in mol.GetConformers()
        ]
    )

    assert result[Property3D.PBF].torch().shape == (5,)
    assert result[Property3D.WHIM].torch().shape == (5, 114)
    _assert_matches_rdkit(result[Property3D.PBF].numpy(), expected_pbf, precision)
    _assert_rounded_matches_rdkit(result[Property3D.WHIM].numpy(), expected_whim, precision)

    dense = result.dense()
    assert dense.values["PBF"].shape == (2, 3)
    assert dense.values["WHIM"].shape == (2, 3, 114)
    assert torch.isnan(dense.values["WHIM"][1, 2]).all()


@pytest.mark.parametrize("precision", PRECISIONS)
def test_whim_matches_rdkit_for_large_molecules(precision):
    # A 130-atom chain and a small molecule share one launch, so per-conformer symmetry scratch slots of
    # very different sizes coexist.
    rng = np.random.default_rng(3)
    steps = rng.normal(size=(130, 3))
    chain = np.cumsum(1.5 * steps / np.linalg.norm(steps, axis=1, keepdims=True), axis=0)
    mols = [_mol_with_conformers("C" * 130, [chain]), _embed("CCCO", 1, 47)]
    result = Calc3DProperties(mols, (Property3D.PBF, Property3D.WHIM), precision=precision)

    expected_whim = np.asarray(
        [rdMolDescriptors.CalcWHIM(mol, confId=conf.GetId()) for mol in mols for conf in mol.GetConformers()]
    )
    expected_pbf = np.asarray(
        [_rdkit_property(mol, conf.GetId(), Property3D.PBF, True) for mol in mols for conf in mol.GetConformers()]
    )
    _assert_rounded_matches_rdkit(result[Property3D.WHIM].numpy(), expected_whim, precision)
    _assert_matches_rdkit(result[Property3D.PBF].numpy(), expected_pbf, precision)


def test_projection_family_reuses_device_coordinates_and_ignores_mass_option():
    mols = [_embed("CCCO", 2, 59)]
    coordinates = _device_result_from_molecules(mols)
    properties = (Property3D.PBF, Property3D.WHIM)

    mass_weighted = Calc3DProperties(
        mols, properties, coordinates=coordinates, options=_moment_options(True), precision=PrecisionMode.FULL
    )
    unit_weighted = Calc3DProperties(
        mols, properties, coordinates=coordinates, options=_moment_options(False), precision=PrecisionMode.FULL
    )
    for prop in properties:
        np.testing.assert_array_equal(mass_weighted[prop].numpy(), unit_weighted[prop].numpy())


@pytest.mark.parametrize("precision", PRECISIONS)
def test_whim_degenerate_geometries_match_rdkit(precision):
    mols = [
        _mol_with_conformers("[He]", [[(4.0, -3.0, 2.0)]]),
        _mol_with_conformers("CCC", [[(-2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (3.0, 0.0, 0.0)]]),
        _mol_with_conformers("CCO", [[(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.5, 1.5, 0.0)]]),
        _mol_with_conformers("CC", [[(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)]]),
    ]
    result = Calc3DProperties(mols, Property3D.WHIM, precision=precision)
    expected = np.asarray([rdMolDescriptors.CalcWHIM(mol) for mol in mols])

    _assert_rounded_matches_rdkit(result[Property3D.WHIM].numpy(), expected, precision)


def _rdkit_pairwise_rows(mols, prop):
    return np.asarray([RDKIT_PAIRWISE[prop](mol, confId=conf.GetId()) for mol in mols for conf in mol.GetConformers()])


@pytest.mark.parametrize("precision", PRECISIONS)
def test_pairwise_family_matches_rdkit_and_preserves_vector_shape(precision):
    rng = np.random.default_rng(5)
    steps = rng.normal(size=(130, 3))
    chain = np.cumsum(1.5 * steps / np.linalg.norm(steps, axis=1, keepdims=True), axis=0)
    mols = [
        _embed("CC(=O)Nc1ccc(O)cc1", 3, 67),
        _embed("c1ccncc1", 2, 71),
        _mol_with_conformers("C" * 130, [chain]),
        # Bromine gives RDKit NaN weights (MORSE NaN, AUTOCORR3D 0); the salt adds disconnected pairs.
        _embed("CC(=O)Nc1ccc(Br)cc1.Cl", 2, 73),
    ]
    result = Calc3DProperties(mols, PAIRWISE_PROPERTIES, precision=precision)

    assert result[Property3D.RDF].torch().shape == (8, 210)
    assert result[Property3D.MORSE].torch().shape == (8, 224)
    assert result[Property3D.AUTOCORR3D].torch().shape == (8, 80)
    for prop in PAIRWISE_PROPERTIES:
        _assert_rounded_matches_rdkit(result[prop].numpy(), _rdkit_pairwise_rows(mols, prop), precision)
        # One pass over atom pairs serves every pairwise property; requesting one alone must not change it.
        alone = Calc3DProperties(mols, prop, precision=precision)
        np.testing.assert_array_equal(alone[prop].numpy(), result[prop].numpy())

    dense = result.dense()
    assert dense.values["RDF"].shape == (4, 3, 210)
    assert dense.values["AUTOCORR3D"].shape == (4, 3, 80)
    assert torch.isnan(dense.values["MORSE"][1, 2]).all()


@pytest.mark.parametrize("precision", PRECISIONS)
def test_pairwise_degenerate_geometries_match_rdkit(precision):
    mols = [
        _mol_with_conformers("[He]", [[(4.0, -3.0, 2.0)]]),
        _mol_with_conformers("CO", [[(0.0, 0.0, 0.0), (1.4, 0.0, 0.0)]]),
        _mol_with_conformers("CC", [[(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)]]),
    ]
    result = Calc3DProperties(mols, PAIRWISE_PROPERTIES, precision=precision)
    for prop in PAIRWISE_PROPERTIES:
        _assert_rounded_matches_rdkit(result[prop].numpy(), _rdkit_pairwise_rows(mols, prop), precision)
    np.testing.assert_array_equal(result[Property3D.RDF].numpy()[0], 0)


def _assert_usr_matches(actual, expected, precision, *, class_blocks):
    """Compare USR-layout rows (blocks of 4 reference points x mean, standard deviation, skew).

    The skew is the cube root of the standardized third moment, so near-symmetric distance
    distributions turn rounding noise into visible values: a two-atom USRCAT class has an exact skew
    of 0, yet RDKit reports float64 noise and SINGLE float32 noise (measured up to 0.08). Means and
    standard deviations are compared tightly; the skew tolerance covers that noise.
    """
    actual = np.asarray(actual, dtype=np.float64).reshape(len(expected), -1, 4, 3)
    expected = np.asarray(expected).reshape(actual.shape)
    single = precision == PrecisionMode.SINGLE
    np.testing.assert_allclose(
        actual[..., :2], expected[..., :2], rtol=1e-5 if single else 1e-9, atol=1e-5 if single else 1e-9
    )
    whole_molecule_skew = 5e-3 if single else 1e-4
    np.testing.assert_allclose(actual[:, 0, :, 2], expected[:, 0, :, 2], rtol=0, atol=whole_molecule_skew)
    if class_blocks:
        np.testing.assert_allclose(actual[:, 1:, :, 2], expected[:, 1:, :, 2], rtol=0, atol=0.1 if single else 1e-3)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_usr_family_matches_rdkit(precision):
    mols = [
        _embed("CC(=O)Nc1ccc(O)cc1", 3, 79),
        _embed("c1ccncc1", 2, 83),
        # Atom 1 is closer to the centroid than atom 0 by 5e-9 A, which float32 distances tie (picking atom
        # 0); the closest reference atom must come from float64 comparisons, as in RDKit.
        _mol_with_conformers(
            "C.C.C.C", [[(0.6656854354786497, 0.0, 0.0), (-0.5, 0.0, 0.1), (0.4, 3.0, 0.0), (-0.2, -3.2, 0.5)]]
        ),
    ]
    result = Calc3DProperties(mols, USR_PROPERTIES, precision=precision)
    assert result[Property3D.USR].torch().shape == (6, 12)
    assert result[Property3D.USRCAT].torch().shape == (6, 60)

    for prop in USR_PROPERTIES:
        expected = np.asarray(
            [RDKIT_USR[prop](mol, confId=conf.GetId()) for mol in mols for conf in mol.GetConformers()]
        )
        _assert_usr_matches(result[prop].numpy(), expected, precision, class_blocks=prop == Property3D.USRCAT)
        alone = Calc3DProperties(mols, prop, precision=precision)
        np.testing.assert_array_equal(alone[prop].numpy(), result[prop].numpy())
    # USRCAT starts with USR.
    np.testing.assert_array_equal(result[Property3D.USRCAT].numpy()[:, :12], result[Property3D.USR].numpy())


@pytest.mark.parametrize("precision", PRECISIONS)
def test_usr_needs_three_atoms(precision):
    mols = [
        _mol_with_conformers("CO", [[(0.0, 0.0, 0.0), (1.4, 0.0, 0.0)]]),
        _mol_with_conformers("CCO", [[(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (2.0, 1.4, 0.0)]]),
    ]
    result = Calc3DProperties(mols, USR_PROPERTIES, precision=precision)
    # RDKit raises for fewer than three atoms; nvMolKit reports NaN for that row.
    with pytest.raises(ValueError):
        rdMolDescriptors.GetUSR(mols[0])
    for prop in USR_PROPERTIES:
        values = result[prop].numpy()
        assert np.isnan(values[0]).all()
        _assert_usr_matches(
            values[1:],
            np.asarray([RDKIT_USR[prop](mols[1])]),
            precision,
            class_blocks=prop == Property3D.USRCAT,
        )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_usr_nan_coordinates_give_nan(precision):
    # A NaN coordinate makes the centroid, and so every centered coordinate, NaN: every value of a non-empty
    # atom subset is NaN, which covers every value RDKit reports as NaN (RDKit measures from raw coordinates,
    # so some of its subsets stay finite). Empty USRCAT classes stay 0 in both.
    mol = _mol_with_conformers("CCO", [[(0.0, 0.0, 0.0), (1.5, float("nan"), 0.0), (2.0, 1.4, 0.0)]])
    result = Calc3DProperties([mol], USR_PROPERTIES, precision=precision)
    assert np.isnan(result[Property3D.USR].numpy()).all()
    usrcat = result[Property3D.USRCAT].numpy()[0]
    rdkit_usrcat = np.asarray(RDKIT_USR[Property3D.USRCAT](mol))
    assert np.isnan(usrcat[np.isnan(rdkit_usrcat)]).all()
    np.testing.assert_array_equal(usrcat[~np.isnan(usrcat)], 0)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_autocorr3d_non_finite_coordinates_match_rdkit(precision):
    mols = [
        _mol_with_conformers("CCCO", [[(0.0, 0.0, 0.0), (1.5, float("nan"), 0.0), (2.0, 1.4, 0.0), (3.4, 1.5, 0.2)]]),
        _mol_with_conformers("CCCO", [[(0.0, 0.0, 0.0), (1.5, 0.2, 0.0), (2.0, 1.4, float("inf")), (3.4, 1.5, 0.2)]]),
        _embed("CCCO", 1, 47),
    ]
    result = Calc3DProperties(mols, Property3D.AUTOCORR3D, precision=precision)
    _assert_rounded_matches_rdkit(
        result[Property3D.AUTOCORR3D].numpy(), _rdkit_pairwise_rows(mols, Property3D.AUTOCORR3D), precision
    )


def _rdkit_getaway_rows(mols, precision_digits=2):
    return np.asarray(
        [
            rdMolDescriptors.CalcGETAWAY(mol, confId=conf.GetId(), precision=precision_digits)
            for mol in mols
            for conf in mol.GetConformers()
        ]
    )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_getaway_matches_rdkit(precision):
    rng = np.random.default_rng(11)
    steps = rng.normal(size=(60, 3))
    chain = np.cumsum(1.5 * steps / np.linalg.norm(steps, axis=1, keepdims=True), axis=0)
    mols = [
        _embed("CC(=O)Nc1ccc(O)cc1", 3, 97),
        # Planar: X^T X has a zero singular value, dropped from the pseudo-inverse, and HIC uses D = 2.
        _embed("c1ccncc1", 2, 101),
        _mol_with_conformers("C" * 60, [chain]),
    ]
    result = Calc3DProperties(mols, Property3D.GETAWAY, precision=precision)
    assert result[Property3D.GETAWAY].torch().shape == (6, 273)
    _assert_rounded_matches_rdkit(result[Property3D.GETAWAY].numpy(), _rdkit_getaway_rows(mols), precision)


@pytest.fixture(scope="module")
def getaway_salt():
    # RDKit's GETAWAY takes seconds on this 4-atom salt, so its reference is computed once.
    mol = _mol_with_conformers("CCO.N", [[(0.0, 0.0, 0.0), (1.5, 0.1, 0.0), (2.1, 1.4, 0.2), (5.0, 1.0, -1.0)]])
    return mol, _rdkit_getaway_rows([mol])


@pytest.mark.parametrize("precision", PRECISIONS)
def test_getaway_counts_disconnected_pairs_in_totals_like_rdkit(getaway_salt, precision):
    mol, expected = getaway_salt
    result = Calc3DProperties([mol], Property3D.GETAWAY, precision=precision)
    _assert_rounded_matches_rdkit(result[Property3D.GETAWAY].numpy(), expected, precision)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_getaway_hic_uses_default_conformer_dimension(precision):
    # RDKit's HIC takes its 2D/3D choice from PBF(mol), i.e. the default (first) conformer: planar here.
    planar = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (2.1, 1.4, 0.0), (3.6, 1.5, 0.0)]
    bent = [(0.0, 0.0, 0.0), (1.5, 0.1, 0.3), (2.1, 1.4, -0.4), (3.6, 1.5, 0.6)]
    mol = _mol_with_conformers("CCCO", [planar, bent])
    result = Calc3DProperties([mol], Property3D.GETAWAY, precision=precision)
    _assert_rounded_matches_rdkit(result[Property3D.GETAWAY].numpy(), _rdkit_getaway_rows([mol]), precision)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_getaway_tiny_leverage_matches_rdkit(precision):
    # The central carbon sits 0.003 A from the centroid: its leverage (~2e-6) prints in scientific notation
    # in RDKit's ITH/ISH digit-string clustering.
    corner = 0.89
    coordinates = [
        (0.003, 0.0, 0.0),
        (corner, corner, corner),
        (corner, -corner, -corner),
        (-corner, corner, -corner),
        (-corner, -corner, corner),
    ]
    mol = _mol_with_conformers("C(C)(C)(C)C", [coordinates])
    result = Calc3DProperties([mol], Property3D.GETAWAY, precision=precision)
    _assert_rounded_matches_rdkit(result[Property3D.GETAWAY].numpy(), _rdkit_getaway_rows([mol]), precision)


@pytest.mark.parametrize("precision_digits", [1, 3])
def test_getaway_precision_option_matches_rdkit(precision_digits):
    mols = [_embed("CC(=O)Nc1ccc(O)cc1", 2, 103)]
    result = Calc3DProperties(
        mols,
        Property3D.GETAWAY,
        options=Property3DOptions(getaway=GetawayOptions(precision=precision_digits)),
        precision=PrecisionMode.FULL,
    )
    _assert_rounded_matches_rdkit(
        result[Property3D.GETAWAY].numpy(), _rdkit_getaway_rows(mols, precision_digits), PrecisionMode.FULL
    )


@pytest.mark.parametrize("precision_digits", [-1, 0, 7])
def test_getaway_rejects_out_of_range_precision(precision_digits):
    options = Property3DOptions(getaway=GetawayOptions(precision=precision_digits))
    with pytest.raises(ValueError, match="GETAWAY precision"):
        Calc3DProperties([_embed("CCO", 1, 5)], Property3D.GETAWAY, options=options)


def test_getaway_rejects_non_integer_precision():
    options = Property3DOptions(getaway=GetawayOptions(precision=2.5))
    with pytest.raises(TypeError, match="GETAWAY precision"):
        Calc3DProperties([_embed("CCO", 1, 5)], Property3D.GETAWAY, options=options)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_getaway_non_finite_coordinates_give_nan_information_indices(precision):
    mols = [
        _mol_with_conformers("CCCO", [[(0.0, 0.0, 0.0), (1.5, float("nan"), 0.0), (2.0, 1.4, 0.0), (3.4, 1.5, 0.2)]]),
        _mol_with_conformers("CCCO", [[(0.0, 0.0, 0.0), (1.5, 0.2, 0.0), (2.0, 1.4, float("inf")), (3.4, 1.5, 0.2)]]),
        _embed("CCCO", 1, 47),
    ]
    values = Calc3DProperties(mols, Property3D.GETAWAY, precision=precision)[Property3D.GETAWAY].numpy()
    assert np.isnan(values[:2, :2]).all()  # ITH and ISH
    _assert_rounded_matches_rdkit(values[2:], _rdkit_getaway_rows(mols[2:]), precision)


@pytest.mark.parametrize("precision_digits", [0, 2.5])
def test_getaway_precision_ignored_when_not_requested(precision_digits):
    options = Property3DOptions(getaway=GetawayOptions(precision=precision_digits))
    result = Calc3DProperties([_embed("CCO", 1, 5)], Property3D.PMI1, options=options)
    assert result[Property3D.PMI1].torch().shape == (1,)


def _rdkit_dclv_rows(mols, options=None):
    """RDKit's DoubleCubicLatticeVolume getters, one column per DCLV_PROPERTIES entry."""
    options = DclvOptions() if options is None else options
    table = Chem.GetPeriodicTable()
    rows = []
    for mol in mols:
        # RDKit 2025.09.1 has no default-radii constructor, so pass the van der Waals radii it would use.
        radii = [table.GetRvdw(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
        for conf in mol.GetConformers():
            dclv = rdMolDescriptors.DoubleCubicLatticeVolume(
                mol, radii, probeRadius=options.probeRadius, confId=conf.GetId()
            )
            polar = {"includeSandP": options.includeSandP, "includeHs": options.includeHs}
            rows.append(
                [
                    dclv.GetSurfaceArea(),
                    dclv.GetPolarSurfaceArea(**polar),
                    dclv.GetVolume(),
                    dclv.GetVDWVolume(),
                    dclv.GetPolarVolume(**polar),
                    dclv.GetCompactness(),
                    dclv.GetPackingDensity(),
                ]
            )
    return np.asarray(rows)


def _dclv_columns(mols, precision=PrecisionMode.FULL, **kwargs):
    """Calculate every DCLV property and stack them in DCLV_PROPERTIES order."""
    result = Calc3DProperties(mols, DCLV_PROPERTIES, precision=precision, **kwargs)
    return np.column_stack([result[prop].numpy() for prop in DCLV_PROPERTIES])


def _assert_dclv_matches(actual, expected, precision):
    """Compare DCLV rows.

    The values count exposed surface dots, so a dot within rounding of a neighboring sphere could flip and move a
    value by one dot's share (~1e-3 relative); none does for these geometries, and float32 stays within 1e-5.
    """
    rtol = 1e-9 if precision == PrecisionMode.FULL else 1e-5
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=1e-9, equal_nan=True)


def _cubic_cluster(points_per_side, spacing):
    grid = np.arange(points_per_side) * spacing
    return np.asarray([(x, y, z) for x in grid for y in grid for z in grid])


@pytest.mark.parametrize("precision", PRECISIONS)
def test_dclv_matches_rdkit(precision):
    mols = [
        _embed("CC(=O)Nc1ccc(O)cc1", 3, 107),
        _embed("CS(=O)(=O)Nc1ccc(P(=O)(O)O)cc1", 2, 109),
        _embed("CC(=O)Nc1ccc(Br)cc1.Cl", 1, 113),
        # 216 atoms 1.1 A apart: each atom has more neighbors than the per-warp neighbor cache holds.
        _mol_with_conformers(".".join(["C"] * 216), [_cubic_cluster(6, 1.1)]),
        _mol_with_conformers("[He]", [[(4.0, -3.0, 2.0)]]),
        _mol_with_conformers("CC", [[(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)]]),
    ]
    result = Calc3DProperties(mols, DCLV_PROPERTIES, precision=precision)
    assert tuple(result) == tuple(prop.value for prop in DCLV_PROPERTIES)
    assert all(result[prop].torch().shape == (9,) for prop in DCLV_PROPERTIES)
    actual = np.column_stack([result[prop].numpy() for prop in DCLV_PROPERTIES])
    _assert_dclv_matches(actual, _rdkit_dclv_rows(mols), precision)
    # The probe-expanded and bare-sphere dot passes run only for the properties that need them; a property's value
    # must not depend on what else is requested.
    for prop in DCLV_PROPERTIES:
        alone = Calc3DProperties(mols, prop, precision=precision)
        np.testing.assert_array_equal(alone[prop].numpy(), result[prop].numpy(), err_msg=prop.value)


@pytest.mark.parametrize(
    "options",
    [
        DclvOptions(probeRadius=0.0),
        DclvOptions(probeRadius=2.0, includeSandP=True),
        DclvOptions(includeHs=True),
        DclvOptions(includeSandP=True, includeHs=True),
    ],
)
def test_dclv_options_match_rdkit(options):
    mols = [_embed("CS(=O)(=O)Nc1ccc(P(=O)(O)O)cc1", 2, 127)]
    actual = _dclv_columns(mols, options=Property3DOptions(dclv=options))
    _assert_dclv_matches(actual, _rdkit_dclv_rows(mols, options), PrecisionMode.FULL)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_dclv_zero_radius_atoms_give_nan(precision):
    # RDKit crashes on dummy atoms (radius 0), so such conformers have no reference value.
    coordinates = [(0.0, 0.0, 0.0), (1.5, 0.1, 0.0), (2.1, 1.4, 0.2)]
    mols = [
        _mol_with_conformers("CCO.*", [[*coordinates, (0.8, 0.9, 0.1)]]),
        _mol_with_conformers("CCO", [coordinates]),
    ]
    values = _dclv_columns(mols, precision)
    assert np.isnan(values[0]).all()
    _assert_dclv_matches(values[1:], _rdkit_dclv_rows(mols[1:]), precision)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_dclv_non_finite_coordinates_give_nan(precision):
    mols = [
        _mol_with_conformers("CCO", [[(0.0, 0.0, 0.0), (1.5, float("nan"), 0.0), (2.1, 1.4, 0.2)]]),
        _mol_with_conformers("CCO", [[(0.0, 0.0, 0.0), (1.5, 0.1, 0.0), (2.1, 1.4, float("inf"))]]),
        _embed("CCCO", 1, 47),
    ]
    values = _dclv_columns(mols, precision)
    assert np.isnan(values[:2]).all()
    _assert_dclv_matches(values[2:], _rdkit_dclv_rows(mols[2:]), precision)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_dclv_device_coordinates_far_from_origin(precision):
    mols = [_embed("CC(=O)Nc1ccc(O)cc1", 2, 131)]
    coordinates = _device_result_from_molecules(mols)
    near = _dclv_columns(mols, precision, coordinates=coordinates)
    _assert_dclv_matches(near, _rdkit_dclv_rows(mols), precision)
    coordinates.values.torch().add_(torch.tensor([1e4, -4e3, 6e3], dtype=torch.float64, device="cuda"))
    far = _dclv_columns(mols, precision, coordinates=coordinates)
    _assert_dclv_matches(far, near, precision)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_dclv_stream_ordering_empty_inputs_and_mixed_selection(precision):
    mols = [_embed("CCO", 3, 137), Chem.MolFromSmiles("CC"), _embed("CCN", 2, 139)]
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        coordinates = _device_result_from_molecules(mols)
        result = Calc3DProperties(
            mols, (*DCLV_PROPERTIES, Property3D.PMI1), coordinates=coordinates, precision=precision, stream=stream
        )
        values = [result[prop] for prop in DCLV_PROPERTIES]
        moments = result[Property3D.PMI1]
        labels = (result.mol_indices, result.conf_indices)
    del result, coordinates
    gc.collect()
    _assert_dclv_matches(np.column_stack([value.numpy() for value in values]), _rdkit_dclv_rows(mols), precision)
    _assert_matches_rdkit(moments.numpy(), _reference_rows(mols, (Property3D.PMI1,), True)[:, 0], precision)
    np.testing.assert_array_equal(labels[0].numpy(), [0, 0, 0, 2, 2])
    np.testing.assert_array_equal(labels[1].numpy(), [0, 1, 2, 0, 1])
    for empty_mols in ([], [Chem.MolFromSmiles("CC")], [Chem.MolFromSmiles("")]):
        empty = Calc3DProperties(empty_mols, DCLV_PROPERTIES, precision=precision, stream=stream)
        assert all(empty[prop].torch().shape == (0,) for prop in DCLV_PROPERTIES)


def test_dclv_rejects_invalid_probe_radius_only_when_requested():
    mol = _embed("CCO", 1, 5)
    for probe_radius in (-0.1, float("nan"), float("inf")):
        options = Property3DOptions(dclv=DclvOptions(probeRadius=probe_radius))
        with pytest.raises(ValueError, match="probe radius"):
            Calc3DProperties(mol, Property3D.DCLV_VDW_VOLUME, options=options)
        assert Calc3DProperties(mol, "PMI1", options=options)["PMI1"].torch().shape == (1,)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_autocorr3d_accepts_batches_without_bonds(precision):
    mols = [
        _mol_with_conformers("[He]", [[(4.0, -3.0, 2.0)]]),
        _mol_with_conformers("[Na+].[Cl-]", [[(0.0, 0.0, 0.0), (2.8, 0.0, 0.0)]]),
    ]
    result = Calc3DProperties(mols, Property3D.AUTOCORR3D, precision=precision)
    _assert_rounded_matches_rdkit(
        result[Property3D.AUTOCORR3D].numpy(), _rdkit_pairwise_rows(mols, Property3D.AUTOCORR3D), precision
    )


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
    properties = SCALAR_PROPERTIES

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

    properties = SCALAR_PROPERTIES
    result = Calc3DProperties(
        mols, properties, coordinates=coordinates, options=_moment_options(False), precision=PrecisionMode.FULL
    )
    expected = _reference_rows(mols, properties, use_atomic_masses=False)
    for column, prop in enumerate(properties):
        values = result[prop].numpy()
        assert np.isnan(values[bad_row])
        _assert_matches_rdkit(
            values[1 - bad_row : 2 - bad_row],
            expected[1 - bad_row : 2 - bad_row, column],
            PrecisionMode.FULL,
        )

    projection = Calc3DProperties(
        mols,
        (Property3D.PBF, Property3D.WHIM),
        coordinates=coordinates,
        precision=PrecisionMode.FULL,
    )
    assert np.isnan(projection[Property3D.PBF].numpy()[bad_row])
    assert np.isnan(projection[Property3D.WHIM].numpy()[bad_row]).all()
    valid_row = 1 - bad_row
    expected_pbf = _rdkit_property(mols[0], valid_row, Property3D.PBF, True)
    expected_whim = rdMolDescriptors.CalcWHIM(mols[0], confId=valid_row)
    np.testing.assert_allclose(projection[Property3D.PBF].numpy()[valid_row], expected_pbf, rtol=2e-10, atol=2e-8)
    np.testing.assert_allclose(
        projection[Property3D.WHIM].numpy()[valid_row], expected_whim, rtol=0, atol=0.0011, equal_nan=True
    )


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
    single_thread = Calc3DProperties(mol, ["PMI1", "WHIM"], hardwareOptions=HardwareOptions(preprocessingThreads=1))
    all_threads = Calc3DProperties(mol, ["PMI1", "WHIM"])
    for name in ("PMI1", "WHIM"):
        np.testing.assert_array_equal(single_thread[name].numpy(), all_threads[name].numpy())
    with pytest.raises(ValueError, match="Thread count"):
        Calc3DProperties(mol, ["PMI1"], hardwareOptions=HardwareOptions(preprocessingThreads=0))
    with pytest.raises(TypeError, match="HardwareOptions"):
        Calc3DProperties(mol, ["PMI1"], hardwareOptions=WhimOptions())
    # Options for families that are not requested are ignored, even when invalid.
    invalid_whim = Property3DOptions(whim=WhimOptions(threshold=float("nan")))
    assert Calc3DProperties(mol, ["PMI1"], options=invalid_whim)["PMI1"].torch().shape == (1,)
    with pytest.raises(TypeError, match="Property3DOptions"):
        Calc3DProperties(mol, ["PMI1"], options=WhimOptions())
    with pytest.raises(ValueError, match="WHIM threshold"):
        Calc3DProperties(mol, ["WHIM"], options=Property3DOptions(whim=WhimOptions(threshold=-0.1)))
    with pytest.raises(ValueError, match="WHIM threshold"):
        Calc3DProperties(mol, ["WHIM"], options=Property3DOptions(whim=WhimOptions(threshold=float("nan"))))

    with pytest.raises(TypeError, match="Device3DResult"):
        Calc3DProperties(mol, ["PMI1"], coordinates=object())

    coordinates = _device_result_from_molecules([mol])
    with pytest.raises(ValueError, match=r"coordinates\.n_mols"):
        Calc3DProperties([mol, mol], ["PMI1"], coordinates=coordinates)


# float32 output rounding of charges up to ~20; the float32 solve is refined against a float64 residual.
EEM_SINGLE_ATOL = 2e-6
EEM_SMILES = ("CC(=O)[O-]", "C[NH3+]", "c1ccncc1O", "O=c1cc[nH]cc1", "CS(=O)(=O)NCl", "FC(F)(F)c1ccc(Br)cc1")


# RDKit before 2026.9.2 solved the transposed EEM system (rdkit/rdkit#9688), so its charges for charged molecules
# sum to the negated formal charge. nvMolKit gives the correct values; only neutral molecules are compared against
# older RDKit, and the golden tests below cover the charged ones.
RDKIT_EEM_CHARGES_CORRECT = tuple(int(part) for part in re.findall(r"\d+", rdkit.__version__)[:3]) >= (2026, 9, 2)
requires_rdkit_eem_fix = pytest.mark.skipif(
    not RDKIT_EEM_CHARGES_CORRECT, reason="RDKit before 2026.9.2 returns wrong EEM charges for charged molecules"
)


def _rdkit_eem_reference_is_correct(mol):
    return RDKIT_EEM_CHARGES_CORRECT or Chem.GetFormalCharge(mol) == 0


def _rdkit_eem_rows(mols):
    """RDKit EEM charges of every conformer, concatenated in molecule, then conformer order.

    Returns the charges and a mask of the rows where RDKit's value is correct for the installed version.
    """
    rows, comparable = [], []
    for mol in mols:
        for conf in mol.GetConformers():
            charges = np.asarray(rdMolDescriptors.CalcEEMcharges(mol, confId=conf.GetId()), dtype=np.float64)
            rows.append(charges)
            comparable.append(np.full(len(charges), _rdkit_eem_reference_is_correct(mol)))
    return np.concatenate(rows), np.concatenate(comparable)


def _assert_eem_matches_rdkit(actual, reference, precision):
    expected, comparable = reference
    actual = actual[comparable]
    expected = expected[comparable]
    if precision == PrecisionMode.FULL:
        assert actual.dtype == np.float64
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10, equal_nan=True)
    else:
        assert actual.dtype == np.float32
        np.testing.assert_allclose(actual, expected, rtol=0, atol=EEM_SINGLE_ATOL, equal_nan=True)


def _atom_sums(result, n_mols):
    starts = result.atom_starts.torch().tolist()
    charges = result[Property3D.EEM_CHARGES].numpy()
    return [charges[starts[row] : starts[row + 1]].sum() for row in range(n_mols)]


@pytest.mark.parametrize("precision", PRECISIONS)
def test_eem_charges_are_per_atom_and_match_rdkit(precision):
    mols = [_embed(smiles, 2, 7 + idx) for idx, smiles in enumerate(EEM_SMILES)]
    result = Calc3DProperties(mols, [Property3D.NPR1, Property3D.EEM_CHARGES], precision=precision)

    total_atoms = sum(mol.GetNumAtoms() * mol.GetNumConformers() for mol in mols)
    expected_starts = np.cumsum([0] + [mol.GetNumAtoms() for mol in mols for _ in mol.GetConformers()])
    assert result[Property3D.NPR1].torch().shape == (result.n_conformers,)
    assert result[Property3D.EEM_CHARGES].torch().shape == (total_atoms,)
    np.testing.assert_array_equal(result.atom_starts.numpy(), expected_starts)
    _assert_eem_matches_rdkit(result[Property3D.EEM_CHARGES].numpy(), _rdkit_eem_rows(mols), precision)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_eem_charges_sum_to_formal_charge(precision):
    smiles = ("CC(=O)[O-]", "C[NH3+]", "CCO", "[NH4+]", "[O-]C(=O)CC[NH3+]")
    mols = [_embed(smi, 1, 3) for smi in smiles]
    result = Calc3DProperties(mols, Property3D.EEM_CHARGES, precision=precision)
    expected = [Chem.GetFormalCharge(mol) for mol in mols]
    assert expected == [-1, 1, 0, 1, 0]
    np.testing.assert_allclose(
        _atom_sums(result, len(mols)), expected, atol=1e-5 if precision == PrecisionMode.SINGLE else 1e-9
    )


def _eem_golden_molecule(smiles, coordinates):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    conf = Chem.Conformer(mol.GetNumAtoms())
    for idx, xyz in enumerate(coordinates):
        conf.SetAtomPosition(idx, Point3D(*xyz))
    mol.AddConformer(conf)
    return mol


_TETRAHEDRAL_NH = 1.01 / 3**0.5
# Charges from solving the published EEM equations (Mortier 1986) with RDKit's parameters, as rdkit/rdkit#9688 does:
# the geometries and values of its reproduction. Independent of the installed RDKit, so these hold for any version.
EEM_GOLDEN = (
    ("[OH-]", [(0, 0, 0), (0.97, 0, 0)], [-1.0632, 0.0632]),
    (
        "[NH4+]",
        [
            (0, 0, 0),
            (_TETRAHEDRAL_NH, _TETRAHEDRAL_NH, _TETRAHEDRAL_NH),
            (_TETRAHEDRAL_NH, -_TETRAHEDRAL_NH, -_TETRAHEDRAL_NH),
            (-_TETRAHEDRAL_NH, _TETRAHEDRAL_NH, -_TETRAHEDRAL_NH),
            (-_TETRAHEDRAL_NH, -_TETRAHEDRAL_NH, _TETRAHEDRAL_NH),
        ],
        [-0.9845, 0.4961, 0.4961, 0.4961, 0.4961],
    ),
)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_eem_charged_molecules_match_golden_values(precision):
    mols = [_eem_golden_molecule(smiles, coordinates) for smiles, coordinates, _ in EEM_GOLDEN]
    result = Calc3DProperties(mols, Property3D.EEM_CHARGES, precision=precision)
    expected = np.concatenate([charges for _, _, charges in EEM_GOLDEN])
    # The reference values are rounded to four decimals.
    np.testing.assert_allclose(result[Property3D.EEM_CHARGES].numpy(), expected, rtol=0, atol=1e-4)


@requires_rdkit_eem_fix
@pytest.mark.parametrize("precision", PRECISIONS)
def test_eem_golden_molecules_match_fixed_rdkit(precision):
    mols = [_eem_golden_molecule(smiles, coordinates) for smiles, coordinates, _ in EEM_GOLDEN]
    result = Calc3DProperties(mols, Property3D.EEM_CHARGES, precision=precision)
    _assert_eem_matches_rdkit(result[Property3D.EEM_CHARGES].numpy(), _rdkit_eem_rows(mols), precision)


def test_eem_dense_pads_conformers_and_atoms():
    mols = [_embed("CCO", 3, 1), Chem.AddHs(Chem.MolFromSmiles("CC")), _embed("c1ccncc1O", 1, 2)]
    result = Calc3DProperties(mols, [Property3D.PBF, Property3D.EEM_CHARGES], precision=PrecisionMode.FULL)
    dense = result.dense()

    max_atoms = max(mol.GetNumAtoms() for mol in mols)
    charges = dense.values[Property3D.EEM_CHARGES.value]
    assert charges.shape == (3, 3, max_atoms)
    assert dense.values[Property3D.PBF.value].shape == (3, 3)
    assert dense.atom_mask.shape == (3, 3, max_atoms)
    assert int(dense.atom_mask.sum()) == result[Property3D.EEM_CHARGES].torch().numel()
    torch.testing.assert_close(charges[dense.atom_mask], result[Property3D.EEM_CHARGES].torch())
    assert torch.isnan(charges[~dense.atom_mask]).all()
    assert dense.conf_mask.tolist() == [[True, True, True], [False, False, False], [True, False, False]]


@pytest.mark.parametrize("precision", PRECISIONS)
def test_eem_device_coordinates_share_atom_starts(precision):
    mols = [Chem.AddHs(Chem.MolFromSmiles(smiles)) for smiles in ("CC(=O)[O-]", "c1ccncc1O")]
    params = EmbedParameters()
    params.randomSeed = 0xC0FFEE
    coordinates = EmbedMolecules(mols, params, confsPerMolecule=2, output=CoordinateOutput.DEVICE)

    result = Calc3DProperties(mols, Property3D.EEM_CHARGES, coordinates=coordinates, precision=precision)

    assert result.atom_starts is coordinates.atom_starts
    values = coordinates.values.numpy()
    atom_starts = coordinates.atom_starts.torch().tolist()
    expected = np.full(len(values), np.nan)
    comparable = np.zeros(len(values), dtype=bool)
    for row, mol_idx in enumerate(coordinates.mol_indices.torch().tolist()):
        mol = Chem.Mol(mols[mol_idx])
        conf = Chem.Conformer(mol.GetNumAtoms())
        for atom_idx, (x, y, z) in enumerate(values[atom_starts[row] : atom_starts[row + 1]]):
            conf.SetAtomPosition(atom_idx, Point3D(float(x), float(y), float(z)))
        conf_id = mol.AddConformer(conf, assignId=True)
        expected[atom_starts[row] : atom_starts[row + 1]] = rdMolDescriptors.CalcEEMcharges(mol, confId=conf_id)
        comparable[atom_starts[row] : atom_starts[row + 1]] = _rdkit_eem_reference_is_correct(mol)
    _assert_eem_matches_rdkit(result[Property3D.EEM_CHARGES].numpy(), (expected, comparable), precision)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_eem_atoms_without_parameters_give_nan_for_their_conformers(precision):
    # RDKit has no EEM parameters past bromine and reads beyond its tables for iodine.
    iodide = _embed("CCI", 2, 4)
    ethanol = _embed("CCO", 1, 4)
    result = Calc3DProperties([iodide, ethanol], Property3D.EEM_CHARGES, precision=precision)
    charges = result[Property3D.EEM_CHARGES].numpy()
    iodide_rows = 2 * iodide.GetNumAtoms()
    assert np.isnan(charges[:iodide_rows]).all()
    _assert_eem_matches_rdkit(charges[iodide_rows:], _rdkit_eem_rows([ethanol]), precision)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_eem_atoms_with_zero_parameters_give_nan_unlike_rdkit(precision):
    # Deliberately differs from RDKit: its parameter tables hold zeros for elements and bond orders they do not
    # cover (here Si and single-bonded P), which RDKit silently solves with, returning charges of -10 or beyond.
    # nvMolKit reports NaN for those conformers instead.
    uncovered = [_embed(smiles, 1, 5) for smiles in ("C[Si](C)(C)C", "CP(C)C")]
    ethanol = _embed("CCO", 1, 5)
    result = Calc3DProperties([*uncovered, ethanol], Property3D.EEM_CHARGES, precision=precision)
    charges = result[Property3D.EEM_CHARGES].numpy()
    uncovered_rows = sum(mol.GetNumAtoms() for mol in uncovered)
    assert np.isnan(charges[:uncovered_rows]).all()
    assert np.abs(_rdkit_eem_rows(uncovered)[0]).max() > 5  # RDKit's zero-parameter charges
    _assert_eem_matches_rdkit(charges[uncovered_rows:], _rdkit_eem_rows([ethanol]), precision)


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("smiles", ["[Br-]", "CCO"])
def test_eem_nonfinite_coordinates_invalidate_only_their_conformer(precision, invalid, smiles):
    molecule = Chem.MolFromSmiles(smiles)
    for _ in range(2):
        conformer = Chem.Conformer(molecule.GetNumAtoms())
        for atom_idx in range(molecule.GetNumAtoms()):
            conformer.SetAtomPosition(atom_idx, Point3D(1.5 * atom_idx, 0, 0))
        molecule.AddConformer(conformer, assignId=True)
    molecule.GetConformer(0).SetAtomPosition(0, Point3D(invalid, 0, 0))
    result = Calc3DProperties([molecule], Property3D.EEM_CHARGES, precision=precision)
    charges = result[Property3D.EEM_CHARGES].numpy()
    assert np.isnan(charges[: molecule.GetNumAtoms()]).all()
    valid = charges[molecule.GetNumAtoms() :]
    np.testing.assert_allclose(valid.sum(), Chem.GetFormalCharge(molecule), atol=1e-5)
    expected = np.asarray(rdMolDescriptors.CalcEEMcharges(molecule, confId=1))
    comparable = np.full(len(expected), _rdkit_eem_reference_is_correct(molecule))
    _assert_eem_matches_rdkit(valid, (expected, comparable), precision)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_eem_global_workspace_and_custom_stream(precision):
    molecule = Chem.MolFromSmiles("O" + "C" * 119)
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for atom_idx in range(molecule.GetNumAtoms()):
        angle = 1.9 * atom_idx
        conformer.SetAtomPosition(atom_idx, Point3D(1.4 * np.cos(angle), 1.4 * np.sin(angle), 0.55 * atom_idx))
    molecule.AddConformer(conformer, assignId=True)
    stream = torch.cuda.Stream()
    result = Calc3DProperties([molecule], Property3D.EEM_CHARGES, precision=precision, stream=stream)
    _assert_eem_matches_rdkit(result[Property3D.EEM_CHARGES].numpy(), _rdkit_eem_rows([molecule]), precision)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_eem_budget_limited_grid_preserves_mixed_conformer_order(precision):
    # 256 atoms need ~262/521 KiB per block in single/full precision. The 64 MiB budget caps
    # the grid at 250/125 blocks on GPUs with enough SMs; 515 conformers require scratch reuse.
    large = Chem.MolFromSmiles("O" + "C" * 255)
    conformer = Chem.Conformer(large.GetNumAtoms())
    for atom_idx in range(large.GetNumAtoms()):
        angle = 1.9 * atom_idx
        conformer.SetAtomPosition(atom_idx, Point3D(1.4 * np.cos(angle), 1.4 * np.sin(angle), 0.55 * atom_idx))
    large.AddConformer(conformer, assignId=True)
    small = _embed("CCO", 2, 42)
    molecules = [large, *([small] * 128), large, *([small] * 128), large]
    expected = np.concatenate(
        [
            rdMolDescriptors.CalcEEMcharges(mol, confId=conf.GetId())
            for mol in molecules
            for conf in mol.GetConformers()
        ]
    )
    stream = torch.cuda.Stream()
    for _ in range(2):
        result = Calc3DProperties(molecules, Property3D.EEM_CHARGES, precision=precision, stream=stream)
        _assert_eem_matches_rdkit(
            result[Property3D.EEM_CHARGES].numpy(), (expected, np.ones(len(expected), dtype=bool)), precision
        )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_eem_empty_batches_and_molecules_without_conformers(precision):
    for molecules in ([], [Chem.MolFromSmiles("CCO")]):
        result = Calc3DProperties(molecules, Property3D.EEM_CHARGES, precision=precision)
        assert result.n_conformers == 0
        assert result[Property3D.EEM_CHARGES].torch().numel() == 0
        dense = result.dense()
        assert dense.values[Property3D.EEM_CHARGES.value].shape == (len(molecules), 0, 0)


def _with_atom_starts(coordinates, atom_starts):
    starts = torch.tensor(atom_starts, dtype=torch.int32, device="cuda")
    return Device3DResult(
        values=coordinates.values,
        atom_starts=AsyncGpuResult(starts, coordinates.gpu_id),
        mol_indices=coordinates.mol_indices,
        conf_indices=coordinates.conf_indices,
        gpu_id=coordinates.gpu_id,
        n_mols=coordinates.n_mols,
    )


@pytest.mark.parametrize(
    ("atom_starts", "valid_conformer", "first_row"),
    [([4, 8, 12], 0, 4), ([-4, 0, 4], 1, 0)],
    ids=["past_the_rows", "negative_start"],
)
def test_dense_charges_leave_invalid_atom_ranges_nan(atom_starts, valid_conformer, first_row):
    square = [(0, 0, 0), (1.5, 0, 0), (1.5, 1.5, 0), (0, 1.5, 1.0)]
    shifted = [(x + 0.3 * idx, y, z) for idx, (x, y, z) in enumerate(square)]
    mol = _mol_with_conformers("CCCC", [square, shifted])
    coordinates = _with_atom_starts(_device_result_from_molecules([mol]), atom_starts)

    result = Calc3DProperties([mol], Property3D.EEM_CHARGES, coordinates=coordinates, precision=PrecisionMode.FULL)
    dense = result.dense()

    charges = result[Property3D.EEM_CHARGES].numpy()
    invalid_conformer = 1 - valid_conformer
    assert dense.atom_mask.tolist() == [[[valid_conformer == 0] * 4, [valid_conformer == 1] * 4]]
    np.testing.assert_array_equal(
        dense.values[Property3D.EEM_CHARGES.value][0, valid_conformer].cpu().numpy(),
        charges[first_row : first_row + 4],
    )
    assert torch.isnan(dense.values[Property3D.EEM_CHARGES.value][0, invalid_conformer]).all()
    assert not np.isnan(charges[first_row : first_row + 4]).any()


def test_dense_scalar_properties_do_not_build_atom_gather_indexes(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("per-atom gather indexes built for scalar properties")

    monkeypatch.setattr(descriptors3d_module, "_atom_row_gather", fail)
    mols = [_embed("CCO", 2, 1), _embed("CCCO", 1, 2)]
    result = Calc3DProperties(mols, [Property3D.PMI1, Property3D.PBF], precision=PrecisionMode.FULL)
    dense = result.dense()
    assert dense.values[Property3D.PMI1.value].shape == (2, 2)
    assert dense.conf_mask.tolist() == [[True, True], [True, False]]
    assert dense.atom_mask.shape == (2, 2, max(mol.GetNumAtoms() for mol in mols))
    assert int(dense.atom_mask.sum()) == sum(mol.GetNumAtoms() * mol.GetNumConformers() for mol in mols)
