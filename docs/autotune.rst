.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Hardware autotuning
===================

The :class:`~nvmolkit.types.HardwareOptions` and
:class:`~nvmolkit.substructure.SubstructSearchConfig` knobs that control
batching, threading, and GPU dispatch can have a substantial effect on
throughput, and the best values depend on the GPU model, the size of the
input, and the molecules' chemistry. The :mod:`nvmolkit.autotune` subpackage
runs a short Optuna study on a representative slice of the workload and
returns a config object you can persist and reuse on the full data set.

Installing optuna
-----------------

The autotuner depends on the optional ``optuna`` package. ``optuna`` is **not**
required to use the rest of nvMolKit, and it is **not** installed by the
conda-forge nvMolKit package by default.

Install optuna explicitly:

- Pip users:

  .. code-block:: bash

      pip install nvMolKit[autotune]
      # or
      pip install optuna

- Conda-forge users:

  .. code-block:: bash

      conda install -c conda-forge optuna

Use :func:`nvmolkit.autotune.is_available` to detect whether optuna is
importable in the current environment without triggering the import:

.. code-block:: python

    from nvmolkit import autotune

    if not autotune.is_available():
        raise SystemExit("optuna is not installed; see the autotune docs")

Calling any ``tune_*`` function without ``optuna`` raises an
:class:`ImportError` whose message contains the same install hints as above.

Quick start: tune ETKDG embedding
---------------------------------

Each ``tune_*`` function returns a :class:`~nvmolkit.autotune.TuneResult` whose
``best_config`` field is the tuned options object — drop it directly into the
matching nvMolKit API:

.. code-block:: python

    from rdkit import Chem
    from rdkit.Chem.AllChem import ETKDGv3

    from nvmolkit import autotune
    from nvmolkit.embedMolecules import EmbedMolecules

    mols = [Chem.AddHs(Chem.MolFromSmiles(s)) for s in many_smiles]
    params = ETKDGv3()
    params.useRandomCoords = True

    result = autotune.tune_embed_molecules(
        mols,
        params,
        confsPerMolecule=10,
        n_trials=30,
        target_seconds_per_trial=10.0,
    )

    print(f"Tuned options: {result.best_config.to_dict()}")
    print(f"Throughput: {result.best_throughput:.1f} conformers/s "
          f"on {result.calibration_size} molecules over {result.n_trials_run} trials")

    # Apply the tuned options to the full workload
    EmbedMolecules(mols, params, confsPerMolecule=10, hardwareOptions=result.best_config)

Saving and reloading a tuned config
-----------------------------------

Tuning is expensive, so the result is worth caching. The persistence helpers
work even without ``optuna`` installed, so a config tuned on one machine can
be loaded on a conda-forge install with no autotune extra:

.. code-block:: python

    autotune.save(result.best_config, "etkdg_options.json")

    # Later, possibly on a machine without optuna:
    options = autotune.load("etkdg_options.json")
    EmbedMolecules(mols, params, confsPerMolecule=10, hardwareOptions=options)

The same ``save``/``load`` pair handles
:class:`~nvmolkit.substructure.SubstructSearchConfig` instances; the type is
encoded in the JSON payload and dispatched on load.

Calibration set sizing and the time budget
------------------------------------------

By default the tuner runs each trial on a 10% subsample of the workload,
capped at 2000 molecules. Before the Optuna study starts, a single warm-up
trial runs the *default* configuration. If that warm-up exceeds twice
``target_seconds_per_trial``, the calibration slice is shrunk by half and the
warm-up retries (up to three shrinks). This keeps tuning cheap on huge inputs.

Override these knobs explicitly when needed:

.. code-block:: python

    result = autotune.tune_embed_molecules(
        mols,
        params,
        calibration_set=[i for i in range(500)],   # explicit indices
        target_seconds_per_trial=5.0,              # tighter budget per trial
        n_trials=20,
        verbose=True,
    )

Tuning per GPU configuration
----------------------------

The tuner does **not** search over GPU subsets. Instead, fix ``gpuIds`` to the
hardware configuration you want to evaluate and call ``tune_*`` for each:

.. code-block:: python

    single_gpu = autotune.tune_embed_molecules(mols, params, gpuIds=[0])
    multi_gpu  = autotune.tune_embed_molecules(mols, params, gpuIds=[0, 1])

Compare ``best_throughput`` across runs to pick the deployment configuration.

Other supported APIs
--------------------

The following wrappers are available; each returns the appropriate config
type:

* :func:`nvmolkit.autotune.tune_embed_molecules`
* :func:`nvmolkit.autotune.tune_mmff_optimize`
* :func:`nvmolkit.autotune.tune_uff_optimize`
* :func:`nvmolkit.autotune.tune_batched_forcefield`
* :func:`nvmolkit.autotune.tune_substructure`

Tuning a batched forcefield
---------------------------

Because :class:`~nvmolkit.batchedForcefield.MMFFBatchedForcefield` and
:class:`~nvmolkit.batchedForcefield.UFFBatchedForcefield` accept per-element
constraints and properties, the wrapper takes a *factory* callable that
rebuilds a fresh forcefield with the trial-specific
:class:`~nvmolkit.types.HardwareOptions`:

.. code-block:: python

    from nvmolkit.batchedForcefield import MMFFBatchedForcefield

    def factory(mols, hw_options):
        ff = MMFFBatchedForcefield(mols, hardwareOptions=hw_options)
        for i in range(len(ff)):
            ff[i].add_position_constraint(0, 0.1, 50.0)
        return ff

    result = autotune.tune_batched_forcefield(
        mols,
        factory,
        maxIters=100,
        n_trials=20,
    )
    options = result.best_config

Reference
---------

* :class:`nvmolkit.autotune.TuneResult`
* :func:`nvmolkit.autotune.is_available`
* :func:`nvmolkit.autotune.save`
* :func:`nvmolkit.autotune.load`
