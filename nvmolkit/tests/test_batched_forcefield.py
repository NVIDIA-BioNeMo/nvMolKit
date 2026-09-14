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

import os

import pytest
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
from rdkit.ForceField import rdForceField as _rdForceField  # noqa: F401
from rdkit.Geometry import Point3D

from nvmolkit.batchedForcefield import MMFFBatchedForcefield, UFFBatchedForcefield
from nvmolkit.types import HardwareOptions, PrecisionMode


@pytest.fixture(params=[PrecisionMode.FULL, PrecisionMode.SINGLE], ids=["full", "single"])
def precision(request):
    return request.param


def load_reference_mol():
    # File has a .mol2 extension for historical reasons but is actually MOL V2000 format.
    # The header claims 2D but coords are 3D, which RDKit warns about; silence that warning.
    mol_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "tests",
        "test_data",
        "rdkit_smallmol_1.mol2",
    )
    if not os.path.exists(mol_path):
        pytest.skip(f"Test data file not found: {mol_path}")
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.ERROR)
    try:
        mol = Chem.MolFromMolFile(mol_path, sanitize=False, removeHs=False)
    finally:
        logger.setLevel(RDLogger.WARNING)
    if mol is None:
        pytest.skip("Failed to load rdkit_smallmol_1.mol2")
    Chem.SanitizeMol(mol)
    return mol


def make_embedded_mol(smiles: str, num_confs: int = 1, seed: int = 0xC0FFEE):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    rdDistGeom.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    return mol


def make_fragmented_mol():
    mol = make_embedded_mol("CC.CC")
    conf = mol.GetConformer()
    fragments = Chem.GetMolFrags(mol)
    if len(fragments) != 2:
        raise AssertionError("Expected two fragments for interfragment interaction test")
    anchor = conf.GetAtomPosition(fragments[0][0])
    moved = conf.GetAtomPosition(fragments[1][0])
    shift = Point3D(anchor.x - moved.x + 2.0, anchor.y - moved.y, anchor.z - moved.z)
    for atom_idx in fragments[1]:
        pos = conf.GetAtomPosition(atom_idx)
        conf.SetAtomPosition(atom_idx, Point3D(pos.x + shift.x, pos.y + shift.y, pos.z + shift.z))
    return mol


def clone_mols(molecules):
    return [Chem.Mol(mol) for mol in molecules]


def perturb_conformers(mol, amplitude: float = 0.3, seed: int = 0xBEEF):
    """Displace every atom of every conformer by a small random vector in-place.

    Used to force starting geometries away from the forcefield minimum so that
    relative-offset constraints (which anchor to the starting geometry) produce
    observably different results vs. an unconstrained minimize.
    """
    import random

    rng = random.Random(seed)
    for conf in mol.GetConformers():
        for atom_idx in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(atom_idx)
            conf.SetAtomPosition(
                atom_idx,
                Point3D(
                    pos.x + rng.uniform(-amplitude, amplitude),
                    pos.y + rng.uniform(-amplitude, amplitude),
                    pos.z + rng.uniform(-amplitude, amplitude),
                ),
            )
    return mol


def make_rdkit_mmff_properties(mol, settings: dict | None = None):
    settings = {} if settings is None else settings
    variant = settings.get("variant", "MMFF94")
    mmff_props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant=variant)
    if mmff_props is None:
        raise ValueError("RDKit could not create MMFF properties for molecule")
    mmff_props.SetMMFFVariant(variant)
    mmff_props.SetMMFFDielectricConstant(settings.get("dielectric_constant", 1.0))
    mmff_props.SetMMFFDielectricModel(settings.get("dielectric_model", 1))
    mmff_props.SetMMFFBondTerm(settings.get("bond_term", True))
    mmff_props.SetMMFFAngleTerm(settings.get("angle_term", True))
    mmff_props.SetMMFFStretchBendTerm(settings.get("stretch_bend_term", True))
    mmff_props.SetMMFFOopTerm(settings.get("oop_term", True))
    mmff_props.SetMMFFTorsionTerm(settings.get("torsion_term", True))
    mmff_props.SetMMFFVdWTerm(settings.get("vdw_term", True))
    mmff_props.SetMMFFEleTerm(settings.get("ele_term", True))
    return mmff_props


def make_rdkit_mmff_forcefield(
    mol,
    properties=None,
    conf_id: int = -1,
    nonBondedThreshold: float = 100.0,
    ignoreInterfragInteractions: bool = True,
):
    mmff_props = make_rdkit_mmff_properties(mol) if properties is None else properties
    return rdForceFieldHelpers.MMFFGetMoleculeForceField(
        mol,
        mmff_props,
        nonBondedThresh=nonBondedThreshold,
        confId=conf_id,
        ignoreInterfragInteractions=ignoreInterfragInteractions,
    )


def get_mmff_reference_energy_and_grad(
    mol,
    properties=None,
    conf_id: int = -1,
    nonBondedThreshold: float = 100.0,
    ignoreInterfragInteractions: bool = True,
    configure_forcefield=None,
):
    ff = make_rdkit_mmff_forcefield(
        mol,
        properties=properties,
        conf_id=conf_id,
        nonBondedThreshold=nonBondedThreshold,
        ignoreInterfragInteractions=ignoreInterfragInteractions,
    )
    if configure_forcefield is not None:
        configure_forcefield(ff)
    return ff.CalcEnergy(), list(ff.CalcGrad())


def assert_energy_and_gradient_close(got_energy, want_energy, got_grad, want_grad, precision):
    energy_tol = 1e-5
    gradient_tol = 5e-3 if precision == PrecisionMode.SINGLE else 1e-4
    assert got_energy == pytest.approx(want_energy, rel=energy_tol, abs=energy_tol)
    assert got_grad == pytest.approx(want_grad, rel=gradient_tol, abs=gradient_tol)


def _assert_batched_compute_matches_rdkit_mmff(mol_specs, precision):
    """Verify batched MMFF energies and gradients against RDKit.

    Build a single MMFFBatchedForcefield from ``mol_specs`` and compare each
    molecule using its configured properties and optional constraint.

    When any mol has a constraint configured, additionally verify that the
    constraint has an observable effect on that mol's energy AND gradient — as
    long as RDKit's own constrained-vs-unconstrained delta is above a small
    threshold (which guards against pathological zero-contribution geometries,
    e.g. an undisplaced position constraint).

    Each spec is a dict with keys:
        factory: callable returning an RDKit mol
        property_settings (optional): dict for ``make_rdkit_mmff_properties`` or ``None``
        non_bonded_threshold (optional): defaults to 100.0
        ignore_interfrag_interactions (optional): defaults to True
        configure_batch (optional): ``lambda element: ...`` adding a constraint
        configure_rdkit (optional): ``lambda ff: ...`` adding the matching RDKit constraint
    """
    mols = [spec["factory"]() for spec in mol_specs]
    properties = [
        make_rdkit_mmff_properties(mol, spec["property_settings"]) if spec.get("property_settings") else None
        for mol, spec in zip(mols, mol_specs)
    ]
    thresholds = [spec.get("non_bonded_threshold", 100.0) for spec in mol_specs]
    interfrags = [spec.get("ignore_interfrag_interactions", True) for spec in mol_specs]

    forcefield = MMFFBatchedForcefield(
        clone_mols(mols),
        properties=properties,
        nonBondedThreshold=thresholds,
        ignoreInterfragInteractions=interfrags,
        precision=precision,
    )
    for mol_idx, spec in enumerate(mol_specs):
        if spec.get("configure_batch") is not None:
            spec["configure_batch"](forcefield[mol_idx])
    got_energies = forcefield.compute_energy()
    got_grads = forcefield.compute_gradients()

    want_energies = []
    want_grads = []
    for mol, spec, prop, threshold, interfrag in zip(mols, mol_specs, properties, thresholds, interfrags):
        want_energy, want_grad = get_mmff_reference_energy_and_grad(
            Chem.Mol(mol),
            properties=prop,
            nonBondedThreshold=threshold,
            ignoreInterfragInteractions=interfrag,
            configure_forcefield=spec.get("configure_rdkit"),
        )
        want_energies.append(want_energy)
        want_grads.append(want_grad)

    for mol_idx in range(len(mol_specs)):
        assert_energy_and_gradient_close(
            got_energies[mol_idx][0],
            want_energies[mol_idx],
            got_grads[mol_idx][0],
            want_grads[mol_idx],
            precision,
        )

    if not any(spec.get("configure_batch") is not None for spec in mol_specs):
        return

    unconstrained_forcefield = MMFFBatchedForcefield(
        clone_mols(mols),
        properties=properties,
        nonBondedThreshold=thresholds,
        ignoreInterfragInteractions=interfrags,
        precision=precision,
    )
    unconstrained_energies = unconstrained_forcefield.compute_energy()
    unconstrained_grads = unconstrained_forcefield.compute_gradients()

    for mol_idx, (mol, spec, prop, threshold, interfrag) in enumerate(
        zip(mols, mol_specs, properties, thresholds, interfrags)
    ):
        if spec.get("configure_batch") is None:
            continue
        want_unconstrained_energy, want_unconstrained_grad = get_mmff_reference_energy_and_grad(
            Chem.Mol(mol),
            properties=prop,
            nonBondedThreshold=threshold,
            ignoreInterfragInteractions=interfrag,
        )
        rdkit_energy_delta = abs(want_energies[mol_idx] - want_unconstrained_energy)
        rdkit_grad_delta = max(
            abs(got_component - want_component)
            for got_component, want_component in zip(want_grads[mol_idx], want_unconstrained_grad)
        )
        if rdkit_energy_delta <= 1e-6 and rdkit_grad_delta <= 1e-6:
            continue
        assert abs(got_energies[mol_idx][0] - unconstrained_energies[mol_idx][0]) > 1e-6, (
            f"mol {mol_idx}: constraint had no observable effect on energy"
        )
        assert (
            max(
                abs(got_component - unconstrained_component)
                for got_component, unconstrained_component in zip(
                    got_grads[mol_idx][0], unconstrained_grads[mol_idx][0]
                )
            )
            > 1e-6
        ), f"mol {mol_idx}: constraint had no observable effect on gradient"


def test_mmff_batched_forcefield_properties_match_rdkit(precision):
    """Test varied per-molecule property configurations.

    Cover defaults, MMFF variants, dielectric models, term toggles, and
    fragmented molecules with interfragment interactions.
    """
    _assert_batched_compute_matches_rdkit_mmff(
        [
            {"factory": load_reference_mol},
            {"factory": lambda: make_embedded_mol("CC(=O)NC"), "property_settings": {"variant": "MMFF94s"}},
            {
                "factory": load_reference_mol,
                "property_settings": {"dielectric_constant": 2.5, "dielectric_model": 2},
            },
            {
                "factory": load_reference_mol,
                "property_settings": {
                    "bond_term": False,
                    "angle_term": False,
                    "stretch_bend_term": False,
                    "oop_term": False,
                    "torsion_term": False,
                },
            },
            {
                "factory": make_fragmented_mol,
                "non_bonded_threshold": 25.0,
                "ignore_interfrag_interactions": False,
            },
        ],
        precision,
    )


def test_mmff_batched_forcefield_reads_externally_configured_properties(precision):
    """Test externally configured RDKit MMFF properties.

    Configure properties via raw ``rdForceFieldHelpers.MMFFGetMoleculeProperties``
    plus direct ``SetMMFF*Term``/``SetMMFFDielectricConstant`` calls, with no
    nvmolkit helpers in the path, then pass the object to ``MMFFBatchedForcefield``.

    Needed because of our workaround for RDKit bug https://github.com/rdkit/rdkit/issues/9253
    """
    mol = make_embedded_mol("CCO")

    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    assert props is not None
    props.SetMMFFBondTerm(False)
    props.SetMMFFTorsionTerm(False)
    props.SetMMFFDielectricConstant(2.5)

    forcefield = MMFFBatchedForcefield(clone_mols([mol]), properties=[props], precision=precision)
    got_energy = forcefield.compute_energy()[0][0]
    got_grad = forcefield.compute_gradients()[0][0]

    rd_ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, props)
    rd_energy = rd_ff.CalcEnergy()
    rd_grad = list(rd_ff.CalcGrad())
    assert_energy_and_gradient_close(got_energy, rd_energy, got_grad, rd_grad, precision)

    default_forcefield = MMFFBatchedForcefield(clone_mols([mol]), precision=precision)
    default_energy = default_forcefield.compute_energy()[0][0]
    assert abs(got_energy - default_energy) > 1e-6, (
        "term toggles on externally-configured MMFFMolProperties had no observable effect on the batched energy"
    )


def test_mmff_batched_forcefield_constraints_match_rdkit(precision):
    """Test all five MMFF constraint types against RDKit.

    Apply one constraint type per molecule, with some molecules also carrying
    non-default property settings to exercise the properties-plus-constraints path.
    """
    _assert_batched_compute_matches_rdkit_mmff(
        [
            {
                "factory": lambda: make_embedded_mol("CCO"),
                "configure_batch": lambda element: element.add_distance_constraint(0, 2, False, 0.0, 1.5, 25.0),
                "configure_rdkit": lambda ff: ff.MMFFAddDistanceConstraint(0, 2, False, 0.0, 1.5, 25.0),
            },
            {
                "factory": lambda: make_embedded_mol("CCO"),
                "property_settings": {"dielectric_constant": 2.0, "dielectric_model": 2},
                "configure_batch": lambda element: element.add_distance_constraint(0, 2, True, 0.3, 0.6, 15.0),
                "configure_rdkit": lambda ff: ff.MMFFAddDistanceConstraint(0, 2, True, 0.3, 0.6, 15.0),
            },
            {
                "factory": lambda: make_embedded_mol("CCO"),
                "configure_batch": lambda element: element.add_position_constraint(0, 0.1, 50.0),
                "configure_rdkit": lambda ff: ff.MMFFAddPositionConstraint(0, 0.1, 50.0),
            },
            {
                "factory": lambda: make_embedded_mol("CCC"),
                "configure_batch": lambda element: element.add_angle_constraint(0, 1, 2, True, 5.0, 10.0, 20.0),
                "configure_rdkit": lambda ff: ff.MMFFAddAngleConstraint(0, 1, 2, True, 5.0, 10.0, 20.0),
            },
            {
                "factory": lambda: make_embedded_mol("CCCC"),
                "property_settings": {"variant": "MMFF94s"},
                "configure_batch": lambda element: element.add_torsion_constraint(0, 1, 2, 3, True, 15.0, 30.0, 12.0),
                "configure_rdkit": lambda ff: ff.MMFFAddTorsionConstraint(0, 1, 2, 3, True, 15.0, 30.0, 12.0),
            },
        ],
        precision,
    )


def test_mmff_batched_forcefield_multi_conformer_matches_rdkit(precision):
    mols = [
        make_embedded_mol("CCO", num_confs=2),
        make_embedded_mol("CCCO", num_confs=5),
        make_embedded_mol("c1ccccc1CO", num_confs=3),
    ]
    conf_counts = [mol.GetNumConformers() for mol in mols]
    ff_mols = clone_mols(mols)
    ff = MMFFBatchedForcefield(ff_mols, precision=precision)

    opt_energies, converged = ff.minimize()

    assert len(opt_energies) == len(mols)
    assert len(converged) == len(mols)

    for mol_idx, mol in enumerate(mols):
        assert len(opt_energies[mol_idx]) == conf_counts[mol_idx]
        assert len(converged[mol_idx]) == conf_counts[mol_idx]
        assert all(converged[mol_idx])

        for conf_idx, conf in enumerate(mol.GetConformers()):
            ref_mol = Chem.Mol(mol)
            ref_ff = make_rdkit_mmff_forcefield(ref_mol, conf_id=conf.GetId())
            ref_ff.Minimize()
            want_energy = ref_ff.CalcEnergy()
            tolerance = 1e-4 if precision == PrecisionMode.SINGLE else 1e-5
            assert opt_energies[mol_idx][conf_idx] == pytest.approx(want_energy, rel=tolerance, abs=tolerance)

        for conf_idx, conf in enumerate(ff_mols[mol_idx].GetConformers()):
            ref_conf = mols[mol_idx].GetConformer(conf.GetId())
            for atom_idx in range(ff_mols[mol_idx].GetNumAtoms()):
                got = conf.GetAtomPosition(atom_idx)
                orig = ref_conf.GetAtomPosition(atom_idx)
                assert abs(got.x - orig.x) > 1e-10 or abs(got.y - orig.y) > 1e-10 or abs(got.z - orig.z) > 1e-10, (
                    f"Mol {mol_idx} conformer {conf_idx} positions were not written back"
                )


@pytest.mark.parametrize(
    "ff_factory",
    [
        pytest.param(MMFFBatchedForcefield, id="mmff"),
        pytest.param(UFFBatchedForcefield, id="uff"),
    ],
)
def test_batched_forcefield_metadata_and_element_view(ff_factory):
    mols = [make_embedded_mol("CCO"), make_embedded_mol("CCCC")]
    forcefield = ff_factory(clone_mols(mols))

    assert len(forcefield) == len(mols)
    assert forcefield.num_molecules == len(mols)
    assert forcefield.data_dim == 3
    for mol_idx, mol in enumerate(mols):
        assert forcefield[mol_idx].num_atoms == mol.GetNumAtoms()


@pytest.mark.parametrize(
    "ff_factory",
    [
        pytest.param(lambda mols, precision: MMFFBatchedForcefield(mols, precision=precision), id="mmff"),
        pytest.param(lambda mols, precision: UFFBatchedForcefield(mols, precision=precision), id="uff"),
    ],
)
def test_batched_forcefield_lazy_build_and_rebuild(ff_factory, precision):
    mol = make_embedded_mol("CCO")
    forcefield = ff_factory([Chem.Mol(mol)], precision)

    assert forcefield._native_ff is None
    assert forcefield._dirty is True

    first_energy = forcefield.compute_energy()[0][0]
    first_native = forcefield._native_ff

    assert first_native is not None
    assert forcefield._dirty is False
    assert forcefield.compute_energy()[0][0] == pytest.approx(first_energy, rel=1e-5, abs=1e-5)
    assert forcefield._native_ff is first_native

    forcefield[0].add_distance_constraint(0, 2, True, 0.2, 0.4, 25.0)
    assert forcefield._dirty is True

    forcefield.rebuild()
    assert forcefield._dirty is False
    assert forcefield._native_ff is not first_native


@pytest.mark.parametrize(
    "ff_factory",
    [
        pytest.param(MMFFBatchedForcefield, id="mmff"),
        pytest.param(UFFBatchedForcefield, id="uff"),
    ],
)
@pytest.mark.parametrize(
    "apply_bad_constraint",
    [
        pytest.param(lambda element: element.add_distance_constraint(0, 99, False, 0.0, 1.0, 10.0), id="distance"),
        pytest.param(lambda element: element.add_position_constraint(99, 0.1, 10.0), id="position"),
        pytest.param(lambda element: element.add_angle_constraint(0, 1, 99, True, 0.0, 10.0, 10.0), id="angle"),
        pytest.param(lambda element: element.add_torsion_constraint(0, 1, 2, 99, True, 0.0, 10.0, 10.0), id="torsion"),
    ],
)
def test_batched_forcefield_invalid_indices(ff_factory, apply_bad_constraint):
    forcefield = ff_factory([make_embedded_mol("CCCC")])

    with pytest.raises(IndexError, match="Batch element index"):
        forcefield[1]

    with pytest.raises(IndexError, match="Atom index"):
        apply_bad_constraint(forcefield[0])


# Per-mol constraint specs used across batched MMFF tests. Each entry drives both the
# nvMolKit constraint application and the matching RDKit reference. Different atom indices,
# constraint kinds, and parameter magnitudes verify the batched path routes each mol's
# constraints correctly.
_MMFF_BATCH_CONSTRAINT_SPECS = [
    {
        "smiles": "CCCCO",
        "num_confs": 3,
        "apply": lambda element: element.add_distance_constraint(0, 2, True, -0.05, 0.05, 100.0),
        "apply_rdkit": lambda ff: ff.MMFFAddDistanceConstraint(0, 2, True, -0.05, 0.05, 100.0),
    },
    {
        "smiles": "CCCCCCO",
        "num_confs": 2,
        "apply": lambda element: element.add_angle_constraint(0, 1, 2, True, -5.0, 5.0, 30.0),
        "apply_rdkit": lambda ff: ff.MMFFAddAngleConstraint(0, 1, 2, True, -5.0, 5.0, 30.0),
    },
    {
        "smiles": "c1ccccc1CCO",
        "num_confs": 4,
        "apply": lambda element: element.add_position_constraint(0, 0.1, 50.0),
        "apply_rdkit": lambda ff: ff.MMFFAddPositionConstraint(0, 0.1, 50.0),
    },
    {
        "smiles": "CCCC",
        "num_confs": 2,
        "apply": lambda element: element.add_torsion_constraint(0, 1, 2, 3, True, -10.0, 10.0, 15.0),
        "apply_rdkit": lambda ff: ff.MMFFAddTorsionConstraint(0, 1, 2, 3, True, -10.0, 10.0, 15.0),
    },
]


def _build_constrained_mmff_batch(
    specs=_MMFF_BATCH_CONSTRAINT_SPECS, hardwareOptions=None, precision=PrecisionMode.FULL
):
    mols = [perturb_conformers(make_embedded_mol(s["smiles"], num_confs=s["num_confs"])) for s in specs]
    ff_mols = clone_mols(mols)
    ff = MMFFBatchedForcefield(ff_mols, hardwareOptions=hardwareOptions, precision=precision)
    for idx, spec in enumerate(specs):
        spec["apply"](ff[idx])
    return mols, ff_mols, ff


def _build_stable_hardware_options_batch(forcefield_type, hardware_options, precision):
    source_mols = [
        make_embedded_mol("CCO", num_confs=2),
        make_embedded_mol("CCCO", num_confs=2),
        make_embedded_mol("c1ccccc1", num_confs=2),
    ]
    return forcefield_type(clone_mols(source_mols), hardwareOptions=hardware_options, precision=precision)


def _assert_batched_minimize_matches_rdkit(specs, mols, opt_energies, converged, make_ref_ff, precision):
    """Compare nvMolKit and RDKit minimization results.

    Compare each molecule and conformer while each molecule carries a different
    constraint from ``specs``.
    """
    for mol_idx, (mol, spec) in enumerate(zip(mols, specs)):
        assert len(opt_energies[mol_idx]) == mol.GetNumConformers()
        assert all(converged[mol_idx]), f"Mol {mol_idx} failed to converge"

        for conf_idx, conf in enumerate(mol.GetConformers()):
            ref_mol = Chem.Mol(mol)
            ref_ff = make_ref_ff(ref_mol, conf.GetId())
            spec["apply_rdkit"](ref_ff)
            ref_ff.Minimize(maxIts=500)
            want_energy = ref_ff.CalcEnergy()
            rel_tol, abs_tol = (2e-2, 1e-1) if precision == PrecisionMode.SINGLE else (1e-3, 1e-3)
            assert opt_energies[mol_idx][conf_idx] == pytest.approx(want_energy, rel=rel_tol, abs=abs_tol)


def test_mmff_batched_minimize_with_constraints_batch_matches_rdkit(precision):
    """Test constrained batch minimization against RDKit.

    Use different constraint types, molecule sizes, and conformer counts, then
    compare each molecule and conformer energy to RDKit with the same constraint.
    """
    mols, _, ff = _build_constrained_mmff_batch(precision=precision)
    opt_energies, converged = ff.minimize(maxIters=500)

    def make_ref(mol, conf_id):
        return make_rdkit_mmff_forcefield(mol, conf_id=conf_id)

    _assert_batched_minimize_matches_rdkit(
        _MMFF_BATCH_CONSTRAINT_SPECS, mols, opt_energies, converged, make_ref, precision
    )


def test_mmff_batched_minimize_respects_maxiters_and_forcetol(precision):
    """Test that maxIters and forceTol are passed through.

    A single-iteration minimization should not converge and should leave energies
    closer to the starting point than a generous-iteration minimization.
    """
    perturbed_mols = [
        perturb_conformers(make_embedded_mol("CCCO", num_confs=2)),
        perturb_conformers(make_embedded_mol("c1ccccc1CCO", num_confs=2)),
    ]
    starting_energies = MMFFBatchedForcefield(clone_mols(perturbed_mols), precision=precision).compute_energy()

    tight_ff = MMFFBatchedForcefield(clone_mols(perturbed_mols), precision=precision)
    tight_energies, tight_converged = tight_ff.minimize(maxIters=1, forceTol=1e-12)
    for mol_converged in tight_converged:
        assert not any(mol_converged), "1 iteration at 1e-12 forceTol should not converge"

    loose_ff = MMFFBatchedForcefield(clone_mols(perturbed_mols), precision=precision)
    loose_energies, loose_converged = loose_ff.minimize(maxIters=500, forceTol=1e-4)
    for mol_converged in loose_converged:
        assert all(mol_converged), "500 iterations at 1e-4 forceTol should converge"

    for mol_idx in range(len(perturbed_mols)):
        for conf_idx in range(len(starting_energies[mol_idx])):
            assert loose_energies[mol_idx][conf_idx] < tight_energies[mol_idx][conf_idx], (
                "loose minimize should reach lower energy than 1-iter minimize"
            )
            assert tight_energies[mol_idx][conf_idx] <= starting_energies[mol_idx][conf_idx] + 1e-6, (
                "1-iter minimize should not increase energy above starting point"
            )


@pytest.mark.parametrize("batch_size", [0, 2])
@pytest.mark.parametrize("batches_per_gpu", [1, 3])
def test_mmff_batched_minimize_single_gpu_hardware_options_matches_default(batch_size, batches_per_gpu, precision):
    """HardwareOptions must produce same results as default on a varied constrained batch.

    The constrained batch's uneven conformer counts make ``batchSize=2`` split a molecule's
    conformers across worker batches. Single precision can settle those perturbed, constrained
    conformers in different basins when the batch layout changes, so it uses an unconstrained
    batch that isolates scheduling from basin selection.
    """
    hw_opts = HardwareOptions(gpuIds=[0], batchSize=batch_size, batchesPerGpu=batches_per_gpu)
    if precision == PrecisionMode.FULL:
        _, _, default_ff = _build_constrained_mmff_batch()
        _, _, tuned_ff = _build_constrained_mmff_batch(hardwareOptions=hw_opts)
    else:
        default_ff = _build_stable_hardware_options_batch(MMFFBatchedForcefield, None, precision)
        tuned_ff = _build_stable_hardware_options_batch(MMFFBatchedForcefield, hw_opts, precision)
    default_energies, default_converged = default_ff.minimize(maxIters=500)
    tuned_energies, tuned_converged = tuned_ff.minimize(maxIters=500)

    assert tuned_converged == default_converged
    for mol_idx in range(len(tuned_energies)):
        for conf_idx in range(len(default_energies[mol_idx])):
            assert tuned_energies[mol_idx][conf_idx] == pytest.approx(
                default_energies[mol_idx][conf_idx], rel=1e-4, abs=1e-4
            )


def test_mmff_batched_minimize_multi_gpu_matches_single_gpu(precision):
    if torch.cuda.device_count() < 2:
        pytest.skip("Test requires at least 2 GPUs")

    _, _, single_ff = _build_constrained_mmff_batch(hardwareOptions=HardwareOptions(gpuIds=[0]), precision=precision)
    single_energies, single_converged = single_ff.minimize(maxIters=500)

    _, _, multi_ff = _build_constrained_mmff_batch(hardwareOptions=HardwareOptions(gpuIds=[0, 1]), precision=precision)
    multi_energies, multi_converged = multi_ff.minimize(maxIters=500)

    assert multi_converged == single_converged
    for mol_idx in range(len(single_energies)):
        for conf_idx in range(len(single_energies[mol_idx])):
            assert multi_energies[mol_idx][conf_idx] == pytest.approx(
                single_energies[mol_idx][conf_idx], rel=1e-4, abs=1e-4
            )


def make_rdkit_uff_forcefield(
    mol,
    conf_id: int = -1,
    vdwThreshold: float = 10.0,
    ignoreInterfragInteractions: bool = True,
):
    return rdForceFieldHelpers.UFFGetMoleculeForceField(
        mol,
        vdwThresh=vdwThreshold,
        confId=conf_id,
        ignoreInterfragInteractions=ignoreInterfragInteractions,
    )


def get_uff_reference_energy_and_grad(
    mol,
    conf_id: int = -1,
    vdwThreshold: float = 10.0,
    ignoreInterfragInteractions: bool = True,
    configure_forcefield=None,
):
    ff = make_rdkit_uff_forcefield(
        mol,
        conf_id=conf_id,
        vdwThreshold=vdwThreshold,
        ignoreInterfragInteractions=ignoreInterfragInteractions,
    )
    ff.Initialize()
    if configure_forcefield is not None:
        configure_forcefield(ff)
    return ff.CalcEnergy(), list(ff.CalcGrad())


def _assert_batched_compute_matches_rdkit_uff(mol_specs, precision):
    """UFF analog of _assert_batched_compute_matches_rdkit_mmff.

    Each spec is a dict with keys:
        factory: callable returning an RDKit mol
        vdw_threshold (optional): defaults to 10.0
        ignore_interfrag_interactions (optional): defaults to True
        configure_batch (optional): ``lambda element: ...`` adding a constraint
        configure_rdkit (optional): ``lambda ff: ...`` adding the matching RDKit constraint
    """
    mols = [spec["factory"]() for spec in mol_specs]
    vdw_thresholds = [spec.get("vdw_threshold", 10.0) for spec in mol_specs]
    interfrags = [spec.get("ignore_interfrag_interactions", True) for spec in mol_specs]

    forcefield = UFFBatchedForcefield(
        clone_mols(mols),
        vdwThreshold=vdw_thresholds,
        ignoreInterfragInteractions=interfrags,
        precision=precision,
    )
    for mol_idx, spec in enumerate(mol_specs):
        if spec.get("configure_batch") is not None:
            spec["configure_batch"](forcefield[mol_idx])
    got_energies = forcefield.compute_energy()
    got_grads = forcefield.compute_gradients()

    want_energies = []
    want_grads = []
    for mol, spec, vdw_threshold, interfrag in zip(mols, mol_specs, vdw_thresholds, interfrags):
        want_energy, want_grad = get_uff_reference_energy_and_grad(
            Chem.Mol(mol),
            vdwThreshold=vdw_threshold,
            ignoreInterfragInteractions=interfrag,
            configure_forcefield=spec.get("configure_rdkit"),
        )
        want_energies.append(want_energy)
        want_grads.append(want_grad)

    for mol_idx in range(len(mol_specs)):
        assert_energy_and_gradient_close(
            got_energies[mol_idx][0],
            want_energies[mol_idx],
            got_grads[mol_idx][0],
            want_grads[mol_idx],
            precision,
        )

    if not any(spec.get("configure_batch") is not None for spec in mol_specs):
        return

    unconstrained_forcefield = UFFBatchedForcefield(
        clone_mols(mols),
        vdwThreshold=vdw_thresholds,
        ignoreInterfragInteractions=interfrags,
        precision=precision,
    )
    unconstrained_energies = unconstrained_forcefield.compute_energy()
    unconstrained_grads = unconstrained_forcefield.compute_gradients()

    for mol_idx, (mol, spec, vdw_threshold, interfrag) in enumerate(zip(mols, mol_specs, vdw_thresholds, interfrags)):
        if spec.get("configure_batch") is None:
            continue
        want_unconstrained_energy, want_unconstrained_grad = get_uff_reference_energy_and_grad(
            Chem.Mol(mol), vdwThreshold=vdw_threshold, ignoreInterfragInteractions=interfrag
        )
        rdkit_energy_delta = abs(want_energies[mol_idx] - want_unconstrained_energy)
        rdkit_grad_delta = max(
            abs(got_component - want_component)
            for got_component, want_component in zip(want_grads[mol_idx], want_unconstrained_grad)
        )
        if rdkit_energy_delta <= 1e-6 and rdkit_grad_delta <= 1e-6:
            continue
        assert abs(got_energies[mol_idx][0] - unconstrained_energies[mol_idx][0]) > 1e-6, (
            f"mol {mol_idx}: constraint had no observable effect on energy"
        )
        assert (
            max(
                abs(got_component - unconstrained_component)
                for got_component, unconstrained_component in zip(
                    got_grads[mol_idx][0], unconstrained_grads[mol_idx][0]
                )
            )
            > 1e-6
        ), f"mol {mol_idx}: constraint had no observable effect on gradient"


def test_uff_batched_forcefield_settings_match_rdkit(precision):
    """Batch of mols with varied vdw thresholds and interfragment settings."""
    _assert_batched_compute_matches_rdkit_uff(
        [
            {"factory": lambda: make_embedded_mol("CCO")},
            {"factory": lambda: make_embedded_mol("CC(=O)NC")},
            {
                "factory": make_fragmented_mol,
                "vdw_threshold": 5.0,
                "ignore_interfrag_interactions": False,
            },
        ],
        precision,
    )


def test_uff_batched_forcefield_constraints_match_rdkit(precision):
    """Batch of mols with all 5 UFF constraint types applied (one per mol)."""
    _assert_batched_compute_matches_rdkit_uff(
        [
            {
                "factory": lambda: make_embedded_mol("CCO"),
                "configure_batch": lambda element: element.add_distance_constraint(0, 2, False, 0.0, 1.5, 25.0),
                "configure_rdkit": lambda ff: ff.UFFAddDistanceConstraint(0, 2, False, 0.0, 1.5, 25.0),
            },
            {
                "factory": lambda: make_embedded_mol("CCO"),
                "configure_batch": lambda element: element.add_distance_constraint(0, 2, True, 0.3, 0.6, 15.0),
                "configure_rdkit": lambda ff: ff.UFFAddDistanceConstraint(0, 2, True, 0.3, 0.6, 15.0),
            },
            {
                "factory": lambda: make_embedded_mol("CCO"),
                "configure_batch": lambda element: element.add_position_constraint(0, 0.1, 50.0),
                "configure_rdkit": lambda ff: ff.UFFAddPositionConstraint(0, 0.1, 50.0),
            },
            {
                "factory": lambda: make_embedded_mol("CCC"),
                "configure_batch": lambda element: element.add_angle_constraint(0, 1, 2, True, 5.0, 10.0, 20.0),
                "configure_rdkit": lambda ff: ff.UFFAddAngleConstraint(0, 1, 2, True, 5.0, 10.0, 20.0),
            },
            {
                "factory": lambda: make_embedded_mol("CCCC"),
                "configure_batch": lambda element: element.add_torsion_constraint(0, 1, 2, 3, True, 15.0, 30.0, 12.0),
                "configure_rdkit": lambda ff: ff.UFFAddTorsionConstraint(0, 1, 2, 3, True, 15.0, 30.0, 12.0),
            },
        ],
        precision,
    )


def test_uff_batched_forcefield_multi_conformer_matches_rdkit(precision):
    mols = [
        make_embedded_mol("CCO", num_confs=2),
        make_embedded_mol("CCCO", num_confs=5),
        make_embedded_mol("c1ccccc1CO", num_confs=3),
    ]
    conf_counts = [mol.GetNumConformers() for mol in mols]
    ff_mols = clone_mols(mols)
    ff = UFFBatchedForcefield(ff_mols, precision=precision)

    opt_energies, converged = ff.minimize()

    assert len(opt_energies) == len(mols)
    assert len(converged) == len(mols)

    for mol_idx, mol in enumerate(mols):
        assert len(opt_energies[mol_idx]) == conf_counts[mol_idx]
        assert len(converged[mol_idx]) == conf_counts[mol_idx]
        assert all(converged[mol_idx])

        for conf_idx, conf in enumerate(mol.GetConformers()):
            ref_mol = Chem.Mol(mol)
            ref_ff = make_rdkit_uff_forcefield(ref_mol, conf_id=conf.GetId())
            ref_ff.Initialize()
            ref_ff.Minimize(maxIts=1000)
            want_energy = ref_ff.CalcEnergy()
            tolerance = 1e-4 if precision == PrecisionMode.SINGLE else 1e-5
            assert opt_energies[mol_idx][conf_idx] == pytest.approx(want_energy, rel=tolerance, abs=tolerance)

        for conf_idx, conf in enumerate(ff_mols[mol_idx].GetConformers()):
            ref_conf = mols[mol_idx].GetConformer(conf.GetId())
            for atom_idx in range(ff_mols[mol_idx].GetNumAtoms()):
                got = conf.GetAtomPosition(atom_idx)
                orig = ref_conf.GetAtomPosition(atom_idx)
                assert abs(got.x - orig.x) > 1e-10 or abs(got.y - orig.y) > 1e-10 or abs(got.z - orig.z) > 1e-10, (
                    f"Mol {mol_idx} conformer {conf_idx} positions were not written back"
                )


_UFF_BATCH_CONSTRAINT_SPECS = [
    {
        "smiles": "CCCCO",
        "num_confs": 3,
        "apply": lambda element: element.add_distance_constraint(0, 2, True, -0.05, 0.05, 100.0),
        "apply_rdkit": lambda ff: ff.UFFAddDistanceConstraint(0, 2, True, -0.05, 0.05, 100.0),
    },
    {
        "smiles": "CCCCCCO",
        "num_confs": 1,
        "apply": lambda element: element.add_angle_constraint(0, 1, 2, True, -5.0, 5.0, 30.0),
        "apply_rdkit": lambda ff: ff.UFFAddAngleConstraint(0, 1, 2, True, -5.0, 5.0, 30.0),
    },
    {
        "smiles": "c1ccccc1CCO",
        "num_confs": 4,
        "apply": lambda element: element.add_position_constraint(0, 0.1, 50.0),
        "apply_rdkit": lambda ff: ff.UFFAddPositionConstraint(0, 0.1, 50.0),
    },
    {
        "smiles": "CCCC",
        "num_confs": 1,
        "apply": lambda element: element.add_torsion_constraint(0, 1, 2, 3, True, -10.0, 10.0, 15.0),
        "apply_rdkit": lambda ff: ff.UFFAddTorsionConstraint(0, 1, 2, 3, True, -10.0, 10.0, 15.0),
    },
]


def _build_constrained_uff_batch(
    specs=_UFF_BATCH_CONSTRAINT_SPECS, hardwareOptions=None, precision=PrecisionMode.FULL
):
    mols = [perturb_conformers(make_embedded_mol(s["smiles"], num_confs=s["num_confs"])) for s in specs]
    ff_mols = clone_mols(mols)
    ff = UFFBatchedForcefield(ff_mols, hardwareOptions=hardwareOptions, precision=precision)
    for idx, spec in enumerate(specs):
        spec["apply"](ff[idx])
    return mols, ff_mols, ff


def test_uff_batched_minimize_with_constraints_batch_matches_rdkit(precision):
    mols, _, ff = _build_constrained_uff_batch(precision=precision)
    opt_energies, converged = ff.minimize(maxIters=500)

    def make_ref(mol, conf_id):
        ref_ff = make_rdkit_uff_forcefield(mol, conf_id=conf_id)
        ref_ff.Initialize()
        return ref_ff

    _assert_batched_minimize_matches_rdkit(
        _UFF_BATCH_CONSTRAINT_SPECS, mols, opt_energies, converged, make_ref, precision
    )


@pytest.mark.parametrize("batch_size", [0, 2])
@pytest.mark.parametrize("batches_per_gpu", [1, 3])
def test_uff_batched_minimize_single_gpu_hardware_options_matches_default(batch_size, batches_per_gpu, precision):
    # Isolate scheduling from optimizer basin selection.
    default_ff = _build_stable_hardware_options_batch(UFFBatchedForcefield, None, precision)
    default_energies, default_converged = default_ff.minimize(maxIters=500)

    hw_opts = HardwareOptions(gpuIds=[0], batchSize=batch_size, batchesPerGpu=batches_per_gpu)
    tuned_ff = _build_stable_hardware_options_batch(UFFBatchedForcefield, hw_opts, precision)
    tuned_energies, tuned_converged = tuned_ff.minimize(maxIters=500)

    assert tuned_converged == default_converged
    for mol_idx in range(len(tuned_energies)):
        for conf_idx in range(len(default_energies[mol_idx])):
            assert tuned_energies[mol_idx][conf_idx] == pytest.approx(
                default_energies[mol_idx][conf_idx], rel=1e-4, abs=1e-4
            )


def test_uff_batched_minimize_multi_gpu_matches_single_gpu(precision):
    if torch.cuda.device_count() < 2:
        pytest.skip("Test requires at least 2 GPUs")

    _, _, single_ff = _build_constrained_uff_batch(hardwareOptions=HardwareOptions(gpuIds=[0]), precision=precision)
    single_energies, single_converged = single_ff.minimize(maxIters=500)

    _, _, multi_ff = _build_constrained_uff_batch(hardwareOptions=HardwareOptions(gpuIds=[0, 1]), precision=precision)
    multi_energies, multi_converged = multi_ff.minimize(maxIters=500)

    assert multi_converged == single_converged
    for mol_idx in range(len(single_energies)):
        for conf_idx in range(len(single_energies[mol_idx])):
            assert multi_energies[mol_idx][conf_idx] == pytest.approx(
                single_energies[mol_idx][conf_idx], rel=1e-4, abs=1e-4
            )
