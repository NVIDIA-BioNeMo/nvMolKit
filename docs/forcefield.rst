.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Force field optimization on GPU
===============================

nvMolKit provides GPU-accelerated MMFF94 and UFF force fields for batches of
molecules with multiple conformers each. All conformers of each molecule are
evaluated together, and results are nested as ``list[list[...]]`` — outer
per-molecule, inner per-conformer.

Two layers of API are available:

- :func:`nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs` and
  :func:`nvmolkit.uffOptimization.UFFOptimizeMoleculesConfs` are drop-in
  batched replacements for RDKit's ``AllChem.MMFFOptimizeMoleculeConfs`` and
  ``AllChem.UFFOptimizeMoleculeConfs``.
- :class:`nvmolkit.batchedForcefield.MMFFBatchedForcefield` and
  :class:`nvmolkit.batchedForcefield.UFFBatchedForcefield` expose the same
  minimizer with per-element knobs such as custom properties, non-bonded
  thresholds, and distance / position / angle / torsion constraints.

Drop-in replacement for RDKit's MMFF/UFF optimize-confs
-------------------------------------------------------

The top-level functions mirror RDKit's ``MMFFOptimizeMoleculeConfs`` and
``UFFOptimizeMoleculeConfs`` but accept a list of molecules and optimize all
conformers of all molecules as one GPU batch. Optimized coordinates are
written back into the RDKit conformers in-place.

The example below uses three heterogeneous molecules with different numbers
of conformers each:

.. code-block:: python

    from rdkit import Chem
    from rdkit.Chem.rdDistGeom import ETKDGv3
    from nvmolkit.embedMolecules import EmbedMolecules
    from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs

    smiles = ["CCO", "c1ccccc1O", "CC(=O)NC1=CC=C(C=C1)O"]
    mols = [Chem.AddHs(Chem.MolFromSmiles(smi)) for smi in smiles]

    params = ETKDGv3()

    # Heterogeneous conformer counts require one EmbedMolecules call per
    # count; if every molecule needed the same number of conformers, a
    # single batched call would be more efficient.
    EmbedMolecules([mols[0]], params, confsPerMolecule=3)
    EmbedMolecules([mols[1]], params, confsPerMolecule=5)
    EmbedMolecules([mols[2]], params, confsPerMolecule=10)

    energies = MMFFOptimizeMoleculesConfs(mols, maxIters=200)
    # energies[0] -> 3 floats, energies[1] -> 5 floats, energies[2] -> 10 floats

The UFF variant has the same shape:

.. code-block:: python

    from nvmolkit.uffOptimization import UFFOptimizeMoleculesConfs

    energies = UFFOptimizeMoleculesConfs(mols, maxIters=1000)

Both functions accept ``hardwareOptions`` (see :ref:`async-results` and the
Hardware targeting section of the overview) and a small set of physics knobs:

- ``nonBondedThreshold`` (MMFF) / ``vdwThreshold`` (UFF): non-bonded cutoff
  distance. Scalar values are broadcast to every molecule; a per-molecule
  sequence may also be provided.
- ``ignoreInterfragInteractions``: whether to omit non-bonded terms between
  fragments. Also scalar-or-per-molecule.
- ``properties`` (MMFF only): RDKit ``MMFFMolProperties`` object, a
  per-molecule sequence of such objects, or ``None`` for default MMFF94.

Batched forcefield for fine-grained control
-------------------------------------------

:class:`~nvmolkit.batchedForcefield.MMFFBatchedForcefield` and
:class:`~nvmolkit.batchedForcefield.UFFBatchedForcefield` wrap the same
minimizer but let you configure per-batch-element settings and add
constraints. They are the right choice when:

- Different molecules in the batch need different ``MMFFMolProperties``,
  non-bonded cutoffs, or interfragment-interaction behavior.
- You need distance, position, angle, or torsion constraints on specific
  atoms of specific molecules.

Construction mirrors the drop-in API, but every scalar argument also accepts
a per-molecule sequence:

.. code-block:: python

    from rdkit.Chem import AllChem
    from nvmolkit.batchedForcefield import MMFFBatchedForcefield

    # Molecule 0: None for default MMFF94 settings
    # Molecule 1: unmodified from defaults
    props_b = AllChem.MMFFGetMoleculeProperties(mols[1])

    # Molecule 2: MMFF94s variant with a non-default dielectric.
    props_c = AllChem.MMFFGetMoleculeProperties(mols[2])
    props_c.SetMMFFVariant("MMFF94s")
    props_c.SetMMFFDielectricConstant(4.0)
    props_c.SetMMFFDielectricModel(2)  # 1 = constant, 2 = distance-dependent

    ff = MMFFBatchedForcefield(
        mols,
        properties=[None, props_b, props_c],
        nonBondedThreshold=[100.0, 25.0, 25.0],
        ignoreInterfragInteractions=[True, False, True],
    )

Every per-term toggle that RDKit's ``MMFFMolProperties`` exposes is honored:
``SetMMFFBondTerm``, ``SetMMFFAngleTerm``, ``SetMMFFStretchBendTerm``,
``SetMMFFOopTerm``, ``SetMMFFTorsionTerm``, ``SetMMFFVdWTerm``, and
``SetMMFFEleTerm``. Configure the RDKit object before passing it in; the
batched forcefield reads the settings at build time.

Per-element constraints are added through the ``ff[i]`` view. Constraints
attached to molecule ``i`` apply to every conformer of that molecule. The
``relative`` flag on distance, angle, and torsion constraints switches the
bounds between absolute values and offsets from the current geometry.

Distance constraint — hold atoms 0 and 2 of the first molecule between 0.2 Å
below and 0.5 Å above their current separation:

.. code-block:: python

    ff[0].add_distance_constraint(
        idx1=0, idx2=2,
        relative=True,
        min_len=-0.2, max_len=0.5,
        force_constant=20.0,
    )

Position constraint — pin atom 0 of the first molecule within 0.1 Å of its
starting coordinates:

.. code-block:: python

    ff[0].add_position_constraint(idx=0, max_displ=0.1, force_constant=50.0)

Angle constraint — restrain an absolute angle on the second molecule:

.. code-block:: python

    ff[1].add_angle_constraint(
        idx1=0, idx2=1, idx3=2,
        relative=False,
        min_angle_deg=100.0, max_angle_deg=120.0,
        force_constant=25.0,
    )

Torsion constraint — restrain a dihedral on the third molecule:

.. code-block:: python

    ff[2].add_torsion_constraint(
        idx1=0, idx2=1, idx3=2, idx4=3,
        relative=False,
        min_dihedral_deg=-10.0, max_dihedral_deg=10.0,
        force_constant=8.0,
    )

Once the batch is configured, ``minimize`` runs BFGS on every conformer of
every molecule and writes optimized coordinates back into the RDKit
conformers:

.. code-block:: python

    energies, converged = ff.minimize(maxIters=200, forceTol=1e-4)
    # energies[i][j] and converged[i][j] — molecule i, conformer j.

Constraints and properties are resolved lazily before the first evaluation;
if you add or remove constraints after a previous call, the next call
rebuilds the native forcefield automatically. Call
:meth:`~nvmolkit.batchedForcefield.MMFFBatchedForcefield.rebuild` explicitly
if you need to force a rebuild.

The UFF variant is analogous and takes ``vdwThreshold`` instead of
``nonBondedThreshold``; it has no ``properties`` argument:

.. code-block:: python

    from nvmolkit.batchedForcefield import UFFBatchedForcefield

    ff = UFFBatchedForcefield(mols, vdwThreshold=[10.0, 8.0, 8.0])
    ff[0].add_position_constraint(0, 0.1, 50.0)
    energies, converged = ff.minimize()

Energy and gradient evaluation
------------------------------

Both batched forcefields also expose ``compute_energy`` and
``compute_gradients`` for inspecting the current state without running the
minimizer:

.. code-block:: python

    ff = MMFFBatchedForcefield(mols)

    energies = ff.compute_energy()
    # energies[i][j] — one float per conformer.

    grads = ff.compute_gradients()
    # grads[i][j] — flattened [x0, y0, z0, x1, y1, z1, ...] per conformer.

.. note::
    ``compute_energy`` and ``compute_gradients`` are intended for correctness
    checks and Python workflows that need per-call energies or gradients.
    Each call incurs Python-side overhead that can dominate the actual GPU
    work, so they are not the performant path. For high-throughput
    optimization, use :meth:`~nvmolkit.batchedForcefield.MMFFBatchedForcefield.minimize`
    or the top-level
    :func:`~nvmolkit.mmffOptimization.MMFFOptimizeMoleculesConfs` /
    :func:`~nvmolkit.uffOptimization.UFFOptimizeMoleculesConfs` functions.
