# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import descriptors3d_bench as bench
import numpy as np
import pytest
from rdkit import Chem


@pytest.fixture
def molecules():
    result = []
    for smiles, conformer_ids in [("CCO", [7, 11]), ("CC", []), ("CO", [23])]:
        molecule = Chem.MolFromSmiles(smiles)
        for conformer_id in conformer_ids:
            conformer = Chem.Conformer(molecule.GetNumAtoms())
            conformer.SetId(conformer_id)
            for atom in range(molecule.GetNumAtoms()):
                conformer.SetAtomPosition(atom, (1.2 * atom, 0.2 * atom * atom, 0.01 * conformer_id * atom))
            molecule.AddConformer(conformer, assignId=False)
        result.append(molecule)
    return result


@pytest.mark.parametrize(
    "properties",
    [
        (bench.Property3D.DCLV_VOLUME,),
        tuple(bench.DCLV_GETTERS),
        tuple(bench.Property3D),
        (bench.Property3D.DCLV_VOLUME, bench.Property3D.PMI1, bench.Property3D.DCLV_VOLUME),
        (
            bench.Property3D.DCLV_VOLUME,
            bench.Property3D.PMI1,
            bench.Property3D.DCLV_SURFACE_AREA,
            bench.Property3D.USR,
        ),
    ],
)
def test_dclv_reference_reuses_one_construction_per_conformer_and_timed_call(monkeypatch, molecules, properties):
    constructor = bench.rdMolDescriptors.DoubleCubicLatticeVolume
    expected = {prop: [] for prop in properties}
    for molecule in molecules:
        for conformer in molecule.GetConformers():
            for prop in expected:
                value = (
                    getattr(
                        constructor(molecule, bench._dclv_radii(molecule), confId=conformer.GetId()),
                        bench.DCLV_GETTERS[prop],
                    )()
                    if prop in bench.DCLV_GETTERS
                    else bench._calc_rdkit_property(molecule, conformer.GetId(), prop)
                )
                expected[prop].append(value)
    constructions = []

    def counted_constructor(molecule, radii, **kwargs):
        assert radii == bench._dclv_radii(molecule)
        constructions.append((molecule, kwargs["confId"]))
        return constructor(molecule, radii, **kwargs)

    monkeypatch.setattr(bench.rdMolDescriptors, "DoubleCubicLatticeVolume", counted_constructor)
    expected_constructions = [
        (molecule, conformer.GetId()) for molecule in molecules for conformer in molecule.GetConformers()
    ]
    for repeat in range(2):
        actual = bench._calc_rdkit(molecules, properties)
        assert constructions == expected_constructions * (repeat + 1)
        assert tuple(actual) == tuple(expected)
        for prop in expected:
            assert actual[prop].dtype == np.float64
            np.testing.assert_allclose(actual[prop], expected[prop], rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("properties", [(), (bench.Property3D.PMI1,), (bench.Property3D.USR,)])
def test_reference_without_dclv_does_not_construct_it(monkeypatch, molecules, properties):
    def unexpected_constructor(*args, **kwargs):
        pytest.fail("DCLV constructed when no DCLV property was requested")

    monkeypatch.setattr(bench.rdMolDescriptors, "DoubleCubicLatticeVolume", unexpected_constructor)
    actual = bench._calc_rdkit(molecules, properties)
    assert tuple(actual) == properties
    for prop in properties:
        expected = [
            bench._calc_rdkit_property(molecule, conformer.GetId(), prop)
            for molecule in molecules
            for conformer in molecule.GetConformers()
        ]
        np.testing.assert_allclose(actual[prop], expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("empty_molecules", [[], [Chem.MolFromSmiles("CCO")]])
def test_empty_reference_has_empty_float64_columns(monkeypatch, empty_molecules):
    def unexpected_constructor(*args, **kwargs):
        pytest.fail("DCLV constructed without a conformer")

    monkeypatch.setattr(bench.rdMolDescriptors, "DoubleCubicLatticeVolume", unexpected_constructor)
    properties = tuple(bench.DCLV_GETTERS) + (bench.Property3D.PMI1,)
    actual = bench._calc_rdkit(empty_molecules, properties)
    assert tuple(actual) == properties
    assert all(values.shape == (0,) and values.dtype == np.float64 for values in actual.values())


def test_dclv_reference_propagates_constructor_errors(monkeypatch, molecules):
    def failed_constructor(*args, **kwargs):
        raise ValueError("invalid reference geometry")

    monkeypatch.setattr(bench.rdMolDescriptors, "DoubleCubicLatticeVolume", failed_constructor)
    with pytest.raises(ValueError, match="invalid reference geometry"):
        bench._calc_rdkit(molecules, (bench.Property3D.DCLV_VOLUME,))


@pytest.mark.parametrize("bounds,expected_ids", [((0, 3), [7, 11, 23]), ((1, 3), [23]), ((0, 0), [])])
@pytest.mark.parametrize(
    "properties",
    [
        tuple(bench.DCLV_GETTERS),
        (bench.Property3D.PMI1,),
        (bench.Property3D.DCLV_VOLUME, bench.Property3D.PMI1, bench.Property3D.USR),
    ],
)
def test_timed_worker_constructs_dclv_once_and_reads_each_requested_getter(
    monkeypatch, molecules, bounds, expected_ids, properties
):
    constructor = bench.rdMolDescriptors.DoubleCubicLatticeVolume
    calculator = bench._calc_rdkit_property
    constructions = []
    getters = []
    other_properties = []

    def counted_property(molecule, conf_id, prop):
        other_properties.append((conf_id, prop))
        return calculator(molecule, conf_id, prop)

    class CountedDclv:
        def __init__(self, molecule, radii, *, confId):
            constructions.append(confId)
            self.conf_id = confId
            self.reference = constructor(molecule, radii, confId=confId)

        def __getattr__(self, name):
            def read_property():
                getters.append((self.conf_id, name))
                return getattr(self.reference, name)()

            return read_property

    monkeypatch.setattr(bench.rdMolDescriptors, "DoubleCubicLatticeVolume", CountedDclv)
    monkeypatch.setattr(bench, "_calc_rdkit_property", counted_property)
    monkeypatch.setattr(bench, "_WORKER_MOLS", molecules)
    monkeypatch.setattr(bench, "_WORKER_PROPERTIES", properties)
    dclv_properties = [prop for prop in properties if prop in bench.DCLV_GETTERS]
    for repeat in range(2):
        assert bench._rdkit_worker_range(bounds) == len(expected_ids)
        assert constructions == (expected_ids * (repeat + 1) if dclv_properties else [])
        assert getters == [
            (conf_id, bench.DCLV_GETTERS[prop]) for conf_id in expected_ids for prop in dclv_properties
        ] * (repeat + 1)
        assert other_properties == [
            (conf_id, prop) for conf_id in expected_ids for prop in properties if prop not in bench.DCLV_GETTERS
        ] * (repeat + 1)
