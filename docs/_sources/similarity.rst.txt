.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Molecular similarity on GPU
===========================

nvMolKit provides GPU-accelerated Tanimoto and cosine similarities. Each call returns an ``n × m`` matrix between two batches (or all-to-all if the second batch is omitted).

Inputs and format
-----------------

- Accepts either an ``AsyncGpuResult`` produced by nvMolKit, or a CUDA ``torch.Tensor`` with dtype ``int32``/``uint32`` and shape ``(num_fingerprints, fp_size/32)``.
- Fingerprints must be packed into 32-bit integers (rows = fingerprints, columns = 32-bit blocks). Both inputs (if two are provided) must have the same ``fp_size``.
- How to preprocess fingerprints for nvMolKit:
    - Fingerprints from nvMolKit: already packed in the correct format; pass directly.
    - Fingerprints from a database: store or convert to ``numpy`` ``uint32`` arrays with shape ``(n, fp_size/32)``, then use ``torch.as_tensor(..., device='cuda')`` to move to GPU.
    - Fingerprints from RDKit or other sources: generate a per-molecule ``numpy`` bit array (e.g., ``GetFingerprintAsNumPy`` for RDKit) and pack once using ``nvmolkit.fingerprints.pack_fingerprint``. Avoid per-query packing for large datasets—preprocess and store packed arrays for efficiency.

Examples
--------

1) From nvMolKit Morgan fingerprints (zero-copy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rdkit import Chem
    from nvmolkit.fingerprints import MorganFingerprintGenerator
    from nvmolkit.similarity import crossTanimotoSimilarity, crossCosineSimilarity
    import torch

    mols = [Chem.MolFromSmiles(smi) for smi in ["c1ccccc1", "CCO", "CCN"]]

    fpgen = MorganFingerprintGenerator(radius=2, fpSize=1024)
    fps = fpgen.GetFingerprints(mols, num_threads=0)

    # All-to-all Tanimoto similarity
    sims = crossTanimotoSimilarity(fps).torch()  # [n, n]

2) From a database (numpy -> torch on GPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assume your database stores fingerprints as packed 32-bit integers with dtype ``uint32`` and shape ``(n, fp_size/32)``.
If instead you have a bool dtype, use nvMolKit's ``pack_fingerprint`` helper on torch tensors.

.. code-block:: python

    import numpy as np
    import torch
    from nvmolkit.similarity import crossTanimotoSimilarity

    # Load packed fingerprints: shape [n, fp_size/32], dtype uint32
    packed_np_a = np.load("fingerprints_a_uint32.npy")  # e.g., (N, 1024//32)
    packed_np_b = np.load("fingerprints_b_uint32.npy")  # e.g., (M, 1024//32)

    # Move to GPU as torch tensors (zero-copy from numpy memory semantics, then CUDA copy)
    fps_a = torch.as_tensor(packed_np_a, device='cuda')
    fps_b = torch.as_tensor(packed_np_b, device='cuda')

    # Compute cross-similarity
    sims = crossTanimotoSimilarity(fps_a, fps_b).torch()  # [N_a, N_b]

3) From RDKit fingerprints 
~~~~~~~~~~~~~~~~~~~~~~~~~~

Generating packed fingerprints from RDKit has extra CPU overhead; for large collections, precompute once and store the packed array. Below uses RDKit's per-molecule numpy bit vector and nvMolKit's pack helper.

.. code-block:: python

    import numpy as np
    import torch
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from nvmolkit.fingerprints import pack_fingerprint
    from nvmolkit.similarity import crossTanimotoSimilarity

    mols = [Chem.MolFromSmiles(smi) for smi in ["c1ccccc1", "CCO", "CCN"]]
    fp_size = 2048
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)

    bitrows = []
    for mol in mols:
        arr = gen.GetFingerprintAsNumPy(mol)  # shape [fp_size,]
        bitrows.append(arr.astype(np.bool_))
    bool_np = np.stack(bitrows, axis=0)  # shape [n, fp_size], bool

    bool_torch = torch.as_tensor(bool_np, device='cuda') 
    packed = pack_fingerprint(bool_torch)  # shape [n, fp_size/32]

    sims = crossTanimotoSimilarity(packed).torch()  # [n, n]

Atom-Atom Path similarity
-------------------------

:func:`nvmolkit.similarity.aap_similarity` compares two RDKit molecules with
approximate Atom-Atom Path (AAP) similarity. Parameters are set with
:class:`nvmolkit.similarity.AAPMetric`, which is also the ``metric`` accepted by
the fused clustering functions (see :doc:`clustering`).

.. code-block:: python

    from rdkit import Chem

    from nvmolkit.similarity import AAPMetric, aap_similarity

    score = aap_similarity(
        Chem.MolFromSmiles("CCCO"),
        Chem.MolFromSmiles("CCCN"),
        metric=AAPMetric(max_path_length=7),
    )
