.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

.. module:: nvmolkit
.. currentmodule:: nvmolkit

nvMolKit APIs
=============


Fingerprint Generation
----------------------

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   fingerprints.MorganFingerprintGenerator
   fingerprints.pack_fingerprint
   fingerprints.unpack_fingerprint

Similarity Calculations
-----------------------

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   similarity.crossTanimotoSimilarity
   similarity.crossTanimotoSimilarityMemoryConstrained
   similarity.crossCosineSimilarity
   similarity.crossCosineSimilarityMemoryConstrained


ETKDG Conformer Generation
--------------------------

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   embedMolecules.EmbedMolecules

MMFF Optimization
-----------------

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   mmffOptimization.MMFFOptimizeMoleculesConfs

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   batchedForcefield.MMFFBatchedForcefield
   batchedForcefield.MMFFBatchElement

UFF Optimization
----------------

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   uffOptimization.UFFOptimizeMoleculesConfs

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   batchedForcefield.UFFBatchedForcefield
   batchedForcefield.UFFBatchElement

Butina Clustering
-----------------

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   clustering.butina

Substructure Search
-------------------

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   substructure.hasSubstructMatch
   substructure.countSubstructMatches
   substructure.getSubstructMatches

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   substructure.SubstructSearchConfig
   substructure.SubstructMatchResults

Conformer RMSD
--------------

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   conformerRmsd.GetConformerRMSMatrix
   conformerRmsd.GetConformerRMSMatrixBatch

Torsion Fingerprint Deviation (TFD)
-----------------------------------

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   tfd.GetTFDMatrix
   tfd.GetTFDMatrices

Types
-----

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   types.AsyncGpuResult
   types.HardwareOptions

Hardware Autotuning (optional ``optuna`` extra)
-----------------------------------------------

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   autotune.is_available
   autotune.tune_embed_molecules
   autotune.tune_mmff_optimize
   autotune.tune_uff_optimize
   autotune.tune_batched_forcefield
   autotune.tune_substructure
   autotune.save
   autotune.load

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   autotune.TuneResult
