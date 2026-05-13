.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NVIDIA nvMolKit Documentation
===================================

nvMolKit Introduction
---------------------

nvMolKit is a CUDA-backed python library for accelerating common RDKit molecular operations. nvMolKit links to RDKit and nvMolKit operations work on RDKit RDMol objects.

nvMolKit mimics RDKit's API where possible, but provides batch-oriented versions of these operations to enable efficient parallel processing of multiple molecules on the GPU.

For operations that don't modify RDKit structures, nvMolKit returns asynchronous GPU results, which can be converted to torch Tensors or numpy arrays. See the :ref:`async-results` section for more details.

An example using nvMolKit to compute Morgan fingerprints in parallel on the GPU is shown below:

.. code-block:: python

    import torch
    # RDKit API as common base
    from rdkit import Chem
    mols = [Chem.MolFromSmiles(smi) for smi in ['C1CCCCC1', 'C1CCCCC2CCCCC12', "COO"]]

    # Fingerprints via RDKit
    from rdkit.Chem import rdFingerprintGenerator
    rdkit_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    rdkit_fps = [rdkit_fpgen.GetFingerprint(mol) for mol in mols]  # Sequential processing, list of RDKit fingerprints

    # Fingerprints via nvMolKit
    from nvmolkit.fingerprints import MorganFingerprintGenerator
    nvmolkit_fpgen = MorganFingerprintGenerator(radius=2, fpSize=1024)
    nvmolkit_fps = nvmolkit_fpgen.GetFingerprints(mols)  # Parallel GPU processing, matrix with each row being a fingerprint
    torch.cuda.synchronize()
    print(nvmolkit_fps.torch())

For APIs that modify RDKit structures, nvMolKit applies changes in-place as is done with RDKit. An example of conformer generation:

.. code-block:: python

    from rdkit.Chem.rdDistGeom import ETKDGv3
    from rdkit.Chem import MolFromSmiles, AddHs
    from rdkit.Chem.AllChem import EmbedMultipleConfs
    from nvmolkit.embedMolecules import EmbedMolecules as nvMolKitEmbed

    # ETKDG conformer generation via nvMolKit
    mols = [AddHs(MolFromSmiles(smi)) for smi in ['C1CCCCC1', 'C1CCCCC2CCCCC12', "COO"]]
    params = ETKDGv3()
    params.useRandomCoords = True  # Required for nvMolKit
    nvMolKitEmbed(
        molecules=mols,
        params=params,
        confsPerMolecule=5,
        maxIterations=-1,  # Automatic iteration calculation
    )
    # RDKit version would be
    # for mol in mols:
    #     EmbedMultipleConfs(mol, numConfs=5, params=params)
    for mol in mols:
        print(mol.GetNumConformers())


For more fully-fledged examples, check out the Jupyter notebooks in the `examples` folder of the repository.

Source Code
-----------
nvMolKit is open source under the Apache License, and is available on `GitHub <https://github.com/NVIDIA-Digital-Bio/nvMolKit>`_.

Installation
------------

.. important::

   nvMolKit requires an NVIDIA GPU with compute capability 7.0 (V100) or higher. You can check your GPU's compute capability at the `NVIDIA CUDA GPUs page <https://developer.nvidia.com/cuda-gpus>`_.
   A CUDA Driver compatible with CUDA 12.6 or later is also required (driver version >=560.28). Some degree of backward compatibility may be available; for details, see the `CUDA compatibility guide <https://docs.nvidia.com/deploy/cuda-compatibility/index.html>`_.

Conda Forge
^^^^^^^^^^^

Conda is the recommended way to install nvMolKit, in line with RDKit's recommended installation practice. First, ensure
you have a conda-based environment manager installed and activated, such as `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or `Miniforge <https://conda-forge.org/download/>`_.

nvMolKit v0.4.0 supports RDKit 2024.09.6 through 2025.09.5.

To install with conda, run::

    conda install -c conda-forge nvmolkit



From Source
^^^^^^^^^^^

nvMolKit can be installed from source using a C++ and CUDA compiler. See installation instructions in the `GitHub README <https://github.com/NVIDIA-Digital-Bio/nvMolKit>`_.


Features
--------

nvMolKit currently supports the following features:

* **Morgan Fingerprints**: Generate Morgan fingerprints for batches of molecules in parallel on GPU
    * Supports fingerprint sizes 128, 256, 512, 1024, and 2048 bits
    * Does not yet support countSimulation and other non-default options

* **Molecular Similarity**: Fast GPU-accelerated similarity calculations (see :doc:`similarity`)
    * Tanimoto and cosine Similarity
    * Supports all-to-all comparisons between fingerprints in a batch or between two batches of fingerprints
    * Supports compute in chunks to limit GPU memory usage

* **ETKDG Conformer Generation**: GPU-accelerated 3D conformer generation using Experimental-Torsion Knowledge-based Distance Geometry
    * Batch processing of multiple molecules with multiple conformers per molecule
    * Supports multiple GPUs
    * Does not support all RDKit `EmbedParameters` options. Defaults in ETKDGv3() are supported with a few exceptions (see API documentation)

* **Geometry Relaxation**: GPU-accelerated force field optimization of conformers
    * MMFF94 and UFF force fields
    * Batch optimization of multiple molecules and conformers
    * Supports multiple GPUs

* **Butina clustering**: GPU-accelerated clustering from a distance matrix via the Taylor-Butina method

* **Substructure Search**: GPU-accelerated substructure matching against batches of molecules
    * Supports SMILES and recursive SMARTS-based query molecules via RDKit
    * Does not yet support chirality-aware matching, enhanced stereochemistry, or other advanced RDKit ``SubstructMatchParameters`` options

* **Conformer RMSD**: GPU-accelerated pairwise RMSD matrix computation for conformer ensembles

* **Torsion Fingerprint Deviation (TFD)**: GPU-accelerated TFD computation for comparing conformer geometry
    * Batch processing of multiple molecules with all-pairs conformer comparison

.. _async-results:

Asynchronous GPU Results
------------------------

nvMolKit operations that return GPU-resident data (such as fingerprinting or similarity) return an ``AsyncGpuResult`` object.
This object wraps a GPU computation and allows you to access the results in different formats.

.. code-block:: python

    # Example using fingerprints
    from nvmolkit.fingerprints import MorganFingerprintGenerator
    fpgen = MorganFingerprintGenerator(radius=2, fpSize=1024)
    
    # Get fingerprints - returns AsyncGpuResult. This can be passed to other functions that accept AsyncGpuResult such as similarity.
    # It can be passed to other nvMolKit functions without synchronization, so that multiple operations on the same GPU can be queued
    # before the first one finishes
    result = fpgen.GetFingerprints(mols)

    # To access a result, first synchronize then convert to desired format
    torch.cuda.synchronize()
    
    # Convert to torch tensor (stays on GPU, zero copy)
    fps_torch = result.torch()
    
    # Convert to numpy array (moves to CPU)
    fps_numpy = result.numpy()

The "asynchronous" nature of nvMolKit operations allows you to queue multiple GPU operations without waiting for each to complete. 
You can then choose when to synchronize with the GPU and retrieve results or launch additional operations. Numpy conversions involve
synchronizing with the GPU before copy to the GPU. For torch operations, synchronization can be achieved at any time via `torch.cuda.synchronize()`.

Most nvMolKit operations accept an optional ``stream`` parameter (a ``torch.cuda.Stream``) to control which CUDA stream
the operation runs on. If not specified, the current torch stream is used.


Hardware targeting
------------------

Some operations (currently conformer generation and energy relaxation) support multiple GPUs,and have options for
controlling tunable performance parameters. The ``HardwareOptions`` class can be used to specify these options.

An example:

.. code-block:: python

    from nvmolkit.types import HardwareOptions
    from nvmolkit.embedMolecules import EmbedMolecules

    options = HardwareOptions()

    # Target GPUs 0, 1 and 2. Defaults to using all GPUs detected
    options.gpuIds = [0, 1, 2]

    # Use 12 threads for parallel preprocessing. Defaults to using all CPUs detected
    options.preprocessingThreads = 12

    # Divide up the work into batches of 500 conformers at a time. nvMolKit will pick a reasonable default but
    # optimal values may depend on the GPU.
    options.batchSize = 500

    # Process 4 batches per GPU in parallel
    options.batchesPerGpu = 4
    EmbedMolecules(mols, ..., hardwareOptions=options)

 
 
.. toctree::
   :maxdepth: 1
   :hidden:

   Overview <self>


Guides
------

.. toctree::
   :maxdepth: 1

   similarity
   forcefield
   autotune


API Reference
-------------

.. toctree::
   :maxdepth: 1

   api/nvmolkit

What's New
----------

.. toctree::
   :maxdepth: 1

   changelog
