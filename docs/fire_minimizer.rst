.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

FIRE minimizer
==============

nvMolKit provides the Fast Inertial Relaxation Engine (FIRE) as an
alternative to BFGS for MMFF94 and UFF geometry optimization. FIRE is a
local optimizer: it relaxes each input geometry toward a nearby minimum of
the selected force field, but does not search globally for the
lowest-energy conformer.

The implementation follows the FIRE 2.0 update scheme. See the original
`FIRE paper <https://doi.org/10.1103/PhysRevLett.97.170201>`_, the
`FIRE 2.0 paper <https://doi.org/10.1016/j.commatsci.2020.109584>`_, and
the `ASE FIRE2 reference implementation
<https://docs.ase-lib.org/_modules/ase/optimize/fire2.html>`_.

Using FIRE
----------

Select FIRE with ``minimizerKind="FIRE"``. BFGS remains the default.
The return value and in-place coordinate updates are the same for both
minimizers.

.. code-block:: python

    from rdkit import Chem
    from rdkit.Chem.rdDistGeom import ETKDGv3

    from nvmolkit.embedMolecules import EmbedMolecules
    from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs
    from nvmolkit.types import FireOptions

    smiles = ["CCO", "c1ccccc1O", "CC(=O)NC1=CC=C(C=C1)O"]
    mols = [Chem.AddHs(Chem.MolFromSmiles(smiles)) for smiles in smiles]

    params = ETKDGv3()
    params.useRandomCoords = True
    EmbedMolecules(mols, params, confsPerMolecule=10)

    options = FireOptions()
    options.gradTol = 1e-4

    energies = MMFFOptimizeMoleculesConfs(
        mols,
        maxIters=200,
        minimizerKind="FIRE",
        fireOptions=options,
    )
    # energies[i][j] is the final energy of conformer j of molecule i.

UFF uses the same FIRE options:

.. code-block:: python

    from nvmolkit.uffOptimization import UFFOptimizeMoleculesConfs

    energies = UFFOptimizeMoleculesConfs(
        mols,
        maxIters=400,
        minimizerKind="FIRE",
        fireOptions=options,
    )

The batched-forcefield interfaces also accept ``minimizerKind="FIRE"`` and
``fireOptions=options`` in their ``minimize`` methods.

How the algorithm works
-----------------------

FIRE combines a molecular-dynamics-like integrator with adaptive damping.
For coordinates :math:`\mathbf{x}`, velocity :math:`\mathbf{v}`, and force
:math:`\mathbf{F}=-\nabla E`, each system maintains a time step
:math:`\Delta t`, a mixing coefficient :math:`\alpha`, and a count of
consecutive downhill steps. The power

.. math::

   P = \mathbf{v} \cdot \mathbf{F}

indicates whether the velocity is aligned with the force.

Each FIRE 2.0 iteration performs the following operations independently for
each conformer:

#. If :math:`P > 0`, increment the downhill-step count. After
   ``nMinForIncrease`` consecutive positive-power steps, increase
   :math:`\Delta t` by ``timeStepIncrement`` up to its configured maximum,
   and decrease :math:`\alpha` by ``alphaDecrement``.
#. If :math:`P \leq 0`, reduce :math:`\Delta t` by
   ``timeStepDecrement``, reset :math:`\alpha` and the downhill-step count,
   take the FIRE 2.0 half-step back, and zero the velocity.
#. Apply a semi-implicit Euler force kick. With ``useMass=False`` this is
   :math:`\mathbf{v}\leftarrow\mathbf{v}+\Delta t\,\mathbf{F}`. With
   ``useMass=True``, each atom's force is divided by its mass.
#. Mix the velocity toward the force direction while preserving the velocity
   norm:

   .. math::

      \mathbf{v}\leftarrow
      (1-\alpha)\mathbf{v}
      +\alpha\,|\mathbf{v}|\,\frac{\mathbf{F}}{|\mathbf{F}|}.

#. Limit the displacement norm to ``dMax`` and update
   :math:`\mathbf{x}\leftarrow\mathbf{x}+\Delta t\,\mathbf{v}`.

A conformer is converged when the 2-norm of its full gradient is no greater
than ``gradTol``. The implementation batches independent conformers on the
GPU; the adaptive state and convergence decision remain per conformer.

Compared with the original FIRE algorithm, FIRE 2.0 uses the semi-implicit
Euler integration order and takes a half-step backward when the power becomes
non-positive. These changes improve stability and are enabled by default
through ``takeHalfStepBack=True``.

Parameters and tuning
---------------------

The default values in :class:`nvmolkit.types.FireOptions` are not the
literal ASE FIRE2 defaults. They were selected with an
`Optuna <https://optuna.org/>`_ study at a fixed budget of 200 iterations.
The training data were sampled from the Enamine REAL 10M collection:
1,000 molecules were embedded with ETKDG, ten MMFF94-minimized conformers
were generated per molecule, and Gaussian Cartesian noise with
:math:`\sigma=0.1` Å was added to produce 10,000 non-minimum starting
geometries. Every Optuna trial was re-scored with RDKit MMFF94, and the final
study minimized the mean residual gradient norm per atom.

The study tuned the initial and limiting time steps, displacement cap,
positive-power waiting period, velocity-mixing schedule, and time-step
growth/decay factors. Mass weighting, ABC correction, and plateau-based
stopping were disabled. The selected defaults are:

.. list-table:: Tuned ``FireOptions`` defaults
   :header-rows: 1
   :widths: 42 24 34

   * - Option
     - Default
     - Meaning
   * - ``dtInit``
     - 0.0035256955
     - Initial time step in ps
   * - ``dtMinFactor``
     - 0.0001457029
     - Minimum time step relative to ``dtInit``
   * - ``dtMaxFactor``
     - 5.3536467
     - Maximum time step relative to ``dtInit``
   * - ``dMax``
     - 0.6925294
     - Maximum displacement norm in Å
   * - ``nMinForIncrease``
     - 3
     - Positive-power steps before increasing the time step
   * - ``timeStepIncrement``
     - 1.2751646
     - Time-step multiplier while power stays positive
   * - ``timeStepDecrement``
     - 0.6158984
     - Time-step multiplier after non-positive power
   * - ``alphaInit``
     - 0.2890058
     - Initial velocity-mixing coefficient
   * - ``alphaDecrement``
     - 0.9574426
     - Mixing-coefficient multiplier while power stays positive
   * - ``gradTol``
     - 1e-4
     - Full-system gradient 2-norm convergence threshold

Although tuned on MMFF94, these are algorithmic integration parameters rather
than MMFF term parameters. The same schedule transferred to other force-field
surfaces and is shared by the MMFF94 and UFF interfaces. nvMolKit's UFF
validation independently checks that FIRE lowers the starting energy and
approaches the RDKit UFF minimum. As with any local minimizer, unusual
potentials, constraints, or convergence budgets may benefit from retuning
``dtInit``, ``dMax``, and ``gradTol``.

.. _fire-validation:

Validation
----------

Algorithmic parity
^^^^^^^^^^^^^^^^^^

The implementation was compared step-by-step with ASE FIRE2 using identical
starting coordinates, RDKit MMFF94 energies and gradients, and FIRE
parameters. The validation includes a focused Python parity test at 1, 10,
50, and 200 steps, a 1,000-molecule dataset harness, and native unit tests
that compare batched GPU trajectories with a single-system reference
integrator. Position drift after ten steps in the dataset harness is on the
order of :math:`10^{-7}` Å.

MMFF minimization study
^^^^^^^^^^^^^^^^^^^^^^^

The minimization study used the 10,000 perturbed Enamine REAL 10M conformers
described above. RDKit MMFF94, nvMolKit BFGS at 200 steps, nvMolKit FIRE at
200 steps, and nvMolKit FIRE at 400 steps all started from the same
coordinates. Final geometries from every arm were re-scored with RDKit MMFF94
so energy and residual-gradient comparisons use one implementation.

.. list-table:: Final RDKit-MMFF94 energy over 10,000 conformers
   :header-rows: 1
   :widths: 36 20 20 24

   * - Arm
     - Mean (kcal/mol)
     - Median (kcal/mol)
     - Correlation with RDKit
   * - RDKit MMFF94, 200 steps
     - 30.0764
     - 32.4748
     - 1.000000
   * - nvMolKit BFGS, 200 steps
     - 30.2991
     - 32.6047
     - 0.996979
   * - nvMolKit FIRE, 200 steps
     - 29.9992
     - 32.4451
     - 0.999752
   * - nvMolKit FIRE, 400 steps
     - 29.9979
     - 32.4442
     - 0.999752

At 200 steps, the median absolute FIRE-to-RDKit energy difference was already
near parity; 90% of conformers were within
:math:`6.41\times10^{-5}` kcal/mol/atom. At 400 steps the central
distribution tightened further: the median signed difference was
:math:`2.10\times10^{-6}` kcal/mol/atom and 90% were within
:math:`1.25\times10^{-5}` kcal/mol/atom. The remaining tail is dominated by
arms reaching different nearby minima rather than a systematic energy offset.

.. figure:: _static/fire_validation_energy_distributions.png
   :alt: Histograms of final energy differences for BFGS and FIRE at 200 and 400 steps.
   :width: 100%

   Final-energy difference distributions. The right panel focuses on the
   central 90% of the FIRE distributions so the improvement from 200 to 400
   FIRE steps is visible.

.. figure:: _static/fire_validation_bfgs_fire_scatter.png
   :alt: Pairwise scatter plots comparing final BFGS and FIRE MMFF energies.
   :width: 100%

   Pairwise BFGS/FIRE final energies from identical starting conformers.
   Most points lie on the parity line; points away from the diagonal identify
   conformers for which the optimizers reached different local minima within
   the iteration budget.

Residual gradients show that the energy agreement is not caused by accepting
high-force structures:

.. list-table:: Residual RDKit-MMFF94 gradient norm over 10,000 conformers
   :header-rows: 1
   :widths: 36 16 16 16 16

   * - Arm
     - Median
     - p90
     - p99
     - Maximum
   * - RDKit MMFF94, 200 steps
     - 7.72e-5
     - 1.59e-3
     - 0.605
     - 44.97
   * - nvMolKit BFGS, 200 steps
     - 2.16e-3
     - 2.28e-2
     - 0.961
     - 51.28
   * - nvMolKit FIRE, 200 steps
     - 1.02e-3
     - 6.55e-3
     - 0.0427
     - 0.250
   * - nvMolKit FIRE, 400 steps
     - 2.09e-4
     - 1.42e-3
     - 0.0106
     - 0.0742

Performance
-----------

.. note::

   Performance results will be added here.

Choosing FIRE or BFGS
---------------------

BFGS remains the default and is the conservative choice when reproducing an
existing nvMolKit or RDKit workflow. FIRE is useful when a robust,
low-state optimizer is desirable or when a fixed iteration budget leaves a
heavy tail of BFGS residual gradients. Both algorithms are local optimizers,
and neither guarantees that two starting geometries—or two algorithms—will
reach the same basin.
