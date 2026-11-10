.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Clustering and diversity selection
==================================

:mod:`nvmolkit.clustering` provides Butina and directed sphere exclusion (DISE) clustering, and
:mod:`nvmolkit.pickers` provides Leader and MaxMin selection.

Matrix and fused forms
----------------------

Each algorithm has two forms:

.. list-table::
   :header-rows: 1

   * - Algorithm
     - Matrix form
     - Fused form
   * - Butina
     - :func:`~nvmolkit.clustering.butina`
     - :func:`~nvmolkit.clustering.fused_butina`
   * - Leader
     - :func:`~nvmolkit.pickers.leader`
     - :func:`~nvmolkit.pickers.fused_leader`
   * - MaxMin
     - :func:`~nvmolkit.pickers.maxmin`
     - :func:`~nvmolkit.pickers.fused_maxmin`
   * - DISE
     - :func:`~nvmolkit.clustering.dise`
     - :func:`~nvmolkit.clustering.fused_dise`

The matrix form takes a precomputed distance matrix, from any source. Its memory
grows with the square of the number of items.

The fused form takes fingerprints or molecules and a similarity metric, and
computes distances as it needs them, so its memory grows linearly. Use it when
the distance matrix would not fit in GPU memory.

Both forms return the same result for the same distances. Cutoffs and
thresholds are distances, ``1 - similarity``: a similarity threshold of 0.7 is a
cutoff of 0.3.

Metrics
-------

Fused forms select a similarity metric with ``metric=``, given as a name or as a
metric object:

.. list-table::
   :header-rows: 1

   * - Metric
     - Name
     - Input
   * - :class:`~nvmolkit.similarity.TanimotoMetric`
     - ``"tanimoto"``
     - Packed ``int32`` or ``uint32`` fingerprints, shape ``(N, num_words)``
   * - :class:`~nvmolkit.similarity.CosineMetric`
     - ``"cosine"``
     - Packed ``int32`` or ``uint32`` fingerprints, shape ``(N, num_words)``
   * - :class:`~nvmolkit.similarity.AAPMetric`
     - ``"aap"``
     - Sequence of RDKit molecules

Every fused form accepts every metric, except that
:func:`~nvmolkit.clustering.fused_butina` does not yet support
:class:`~nvmolkit.similarity.AAPMetric`. Metric objects carry parameters; for
example, ``AAPMetric(max_path_length=8)``.

.. code-block:: python

    from rdkit import Chem

    from nvmolkit.clustering import OutputMode
    from nvmolkit.fingerprints import MorganFingerprintGenerator
    from nvmolkit.pickers import fused_leader, fused_maxmin
    from nvmolkit.similarity import AAPMetric

    smiles = ["CCO", "CCN", "CCCC", "c1ccccc1", "c1ccncc1", "CC(=O)O"]
    molecules = [Chem.MolFromSmiles(value) for value in smiles]
    fingerprints = MorganFingerprintGenerator(radius=2, fpSize=2048).GetFingerprints(molecules)

    leaders = fused_leader(fingerprints, 0.6, metric="tanimoto", output=OutputMode.RDKIT)
    picks, last_distance = fused_maxmin(fingerprints, 3, metric="tanimoto", seed=23, output=OutputMode.RDKIT)
    aap_leaders = fused_leader(molecules, 0.8, metric=AAPMetric(max_path_length=8), output=OutputMode.RDKIT)

Output modes
------------

``output=OutputMode.DEVICE`` (the default) returns GPU buffers: clustering
functions return a :class:`~nvmolkit.clustering.ClusterDeviceResult`, and
pickers return an :class:`~nvmolkit.types.AsyncGpuResult` of selected indices.
``output=OutputMode.RDKIT`` returns host tuples of input indices in RDKit's
format.

Differences from RDKit
----------------------

Leader and MaxMin follow RDKit's ``LeaderPicker`` and ``MaxMinPicker``, and a
given MaxMin ``seed`` selects the same first pick as RDKit. DISE uses the
leaders ``LeaderPicker`` selects as centroids and adds cluster assignment.

Leader, MaxMin, and DISE compare distances in ``float32``, while RDKit uses double
precision, so a distance within ``float32`` rounding of a cutoff or threshold can be
classified differently. For Tanimoto distances this happens only at values that
``float32`` cannot represent exactly, such as 0.3; values such as 0.25 and 0.5
give RDKit's results. A distance matrix built from nvMolKit similarities can
differ from fused distances in the last ``float32`` bit.
