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

from nvmolkit.descriptors3d import (
    Calc3DProperties,
    Device3DPropertyResult,
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
)
SCALAR_PROPERTIES = tuple(prop for prop in Property3D if prop not in VECTOR_PROPERTIES)
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
