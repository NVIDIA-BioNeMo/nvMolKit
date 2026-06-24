# Changelog

## 0.5.1 - 2026-06-24

### Summary

nvMolKit 0.5.1 is a bug fix and quality-of-life release.

### Contributors
- Kevin Boyd (@scal444)
- Eva Xue (@evasnow1992)
- Clay Moore (@mooreneural)

### Bug Fixes
- Fix pip-built wheels failing to load in minimal manylinux containers ([8ced6c8](https://github.com/NVIDIA-BioNeMo/nvMolKit/commit/8ced6c80e82b3ef486330cafe46692d208df7bef)).
- Fix TFD lower-triangle pair decoding for large conformer counts and empty-ring torsion handling ([d265234](https://github.com/NVIDIA-BioNeMo/nvMolKit/commit/d265234ca47423a0635c3fe4949086738c10fb08)).
- Match RDKit's fixed BFGS gradient-convergence denominator behavior with RDKit 2026.03+ ([ef5ad6b](https://github.com/NVIDIA-BioNeMo/nvMolKit/commit/ef5ad6b8bceadc94cbbf8536944ef9840e82833e)).
- Fix Morgan fingerprints for batches containing only molecules larger than the GPU buckets ([#195](https://github.com/NVIDIA-BioNeMo/nvMolKit/issues/195)).
- Fix Morgan fingerprint scratch reuse across GPU dispatch rounds ([#197](https://github.com/NVIDIA-BioNeMo/nvMolKit/issues/197)).
- Fix fused Butina neighbor-count launch overflow for very large clustering inputs ([529dd26](https://github.com/NVIDIA-BioNeMo/nvMolKit/commit/529dd260ee507315780b5d3f8b6ea494471d48f7)).
- Fix ETKDG chirality and bounds behavior under the RDKit structure-flag API ([#202](https://github.com/NVIDIA-BioNeMo/nvMolKit/issues/202)).
- Fix disconnected SMARTS queries terminating during multithreaded substructure preprocessing ([#203](https://github.com/NVIDIA-BioNeMo/nvMolKit/issues/203)).
- Fix recursive SMARTS extraction for pip RDKit builds ([eb4452c](https://github.com/NVIDIA-BioNeMo/nvMolKit/commit/eb4452c96c7a585e5f2c421a240b3f405f0e69a4)).
- Fix standalone import of `_mmffOptimization` and `_uffOptimization` before `nvmolkit.types` ([f1f491d](https://github.com/NVIDIA-BioNeMo/nvMolKit/commit/f1f491dcdbdb2978cbaa0701a3764e72bc7a50a2)).

### Quality of Life
- Accept `AsyncGpuResult`, CUDA tensors, CPU tensors, and NumPy arrays consistently in clustering and similarity wrappers ([ebc8fff](https://github.com/NVIDIA-BioNeMo/nvMolKit/commit/ebc8fff478bca6551b2f80b2deac59ff26f78b7a)).
- Improve autotune search defaults with stepped batch-size ranges and physical CPU core counts ([3786668](https://github.com/NVIDIA-BioNeMo/nvMolKit/commit/3786668d211297cdb8aca8af5875934a108a8073)).

## 0.5.0 - 2026-05-13

### Summary

nvMolKit 0.5.0 adds three new GPU-accelerated APIs: Torsion Fingerprint Deviation (TFD), pairwise conformer RMSD, and UFF force field optimization. It also introduces a `BatchedForcefield` Python API for MMFF and UFF with constraints, custom options, and multi-conformer minimization; a low-memory fused Butina clustering path that avoids the O(N²) distance matrix; a Python autotuning framework for the main APIs; and optional device-side output for ETKDG and forcefield optimization. Blackwell / L-class GPUs (including sm_103/B300) are now supported, the supported RDKit range is now 2025.03.1 through 2026.03.1, and nvMolKit is available via `pip install nvmolkit`.

### Contributors
- Kevin Boyd (@scal444)
- Eva Xue (@evasnow1992)
- Alireza Moradzadeh (@moradza)
- Andrei Volgin (@volgin)

### Features
- GPU-accelerated Torsion Fingerprint Deviation (TFD) for batch all-pairs conformer comparison ([#71](https://github.com/NVIDIA-Digital-Bio/nvMolKit/issues/71))
- GPU-accelerated pairwise conformer RMSD matrix computation by @volgin
- GPU-accelerated UFF force field, supporting all options that the new `BatchedForcefield` Python API provides for MMFF ([#114](https://github.com/NVIDIA-Digital-Bio/nvMolKit/issues/114))
- New `BatchedForcefield` Python API exposing per-molecule control over forcefield minimization (MMFF or UFF), and through it custom MMFF optimization options (max iterations, energy/gradient tolerances, non-bonded cutoff) ([#70](https://github.com/NVIDIA-Digital-Bio/nvMolKit/issues/70))
- Distance and position constraints on forcefield optimization (MMFF and UFF) ([#26](https://github.com/NVIDIA-Digital-Bio/nvMolKit/issues/26))
- Multi-conformer minimization in the `BatchedForcefield` API
- `HardwareOptions` support for MMFF minimization, matching the ETKDG hardware-targeting API
- Device-side output for ETKDG and forcefield optimization, allowing GPU tensors to flow between nvMolKit calls without round-tripping through host memory ([#140](https://github.com/NVIDIA-Digital-Bio/nvMolKit/issues/140))
- Python autotuning library for the main APIs (`nvmolkit.autotune`), including ETKDG, forcefield optimization, and substructure search, with configuration serialization ([#141](https://github.com/NVIDIA-Digital-Bio/nvMolKit/issues/141))
- Low-memory fused Butina clustering that computes Tanimoto similarities on the fly with Triton-backed kernels, avoiding the O(N²) distance matrix and enabling clustering of larger fingerprint datasets on a single GPU ([#110](https://github.com/NVIDIA-Digital-Bio/nvMolKit/issues/110))
- Support for Blackwell and L-class GPUs, including sm_103 SASS for B300

### Bug Fixes
- Fix latent stream-ordering bug in the MMFF/BFGS minimizer that could race with subsequent operations ([#172](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/172))
- Fix `int32` overflow in substructure pair indexing for batches where `numTargets * numQueries` exceeds `INT32_MAX`, which previously caused out-of-bounds writes in `hasSubstructMatch` and `countSubstructMatches` ([#169](https://github.com/NVIDIA-Digital-Bio/nvMolKit/issues/169))
- Fix shared-memory overflow in the substructure recursive preprocessor caused by an incorrect config setting ([#98](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/98))
- Fix empty result handling in substructure search with `uniquify` when all matches were already unique ([#112](https://github.com/NVIDIA-Digital-Bio/nvMolKit/issues/112))

### Miscellaneous
- pip wheel distribution pipeline (`pip install nvmolkit`) with manylinux_2_28 wheels for CPython 3.11-3.14 ([#15](https://github.com/NVIDIA-Digital-Bio/nvMolKit/issues/15))
- RDKit support range is now 2025.03.1 through 2026.03.1
- Validate `batchesPerGpu` in `HardwareOptions` so every consumer gets a clean `ValueError` instead of a cryptic C++ error from the MMFF / ETKDG layer ([#103](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/103))
- Validate `neighborlist_max_size` in `butina()` before reaching the GPU ([#104](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/104))
- Validate MMFF atom types up front and report every failing molecule instead of hitting a `PRECONDITION` assertion mid-batch ([#106](https://github.com/NVIDIA-Digital-Bio/nvMolKit/issues/106))


## 0.4.0 - 2026-02-23

### Summary

nvMolKit 0.4.0 adds GPU-accelerated substructure searching, optional stream control across Python APIs, and enhancements to Butina clustering.

### Contributors
- Kevin Boyd (@scal444)
- Eva Xue (@evasnow1992)

### Features
- GPU-accelerated substructure search with `hasSubstructMatch`, `countSubstructMatches`, and `getSubstructMatches`. Supports batch queries against batch targets with SMARTS-based query molecules.
- Optional `stream` parameter added to fingerprint generation, similarity, and Butina clustering APIs, enabling explicit CUDA stream control
- Butina clustering now supports optional centroid reporting via the `return_centroids` parameter ([#82](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/82))
- Butina clustering performance improved by replacing CPU loops with CUDA Graph conditional nodes ([#72](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/72))

### Bug Fixes
- Fix data races when torch operations immediately followed nvMolKit calls on the default stream (Issue [#84](https://github.com/NVIDIA-Digital-Bio/nvMolKit/issues/84)). Operations now correctly use the current stream or an explicit `stream` parameter ([#36](https://github.com/NVIDIA-Digital-Bio/nvMolKit/issues/36)).
- Fix `setup.py` compatibility on some Python versions and rework CUDA target detection ([#68](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/68))


## 0.3.0 - 2025-12-12

### Summary

nvMolKit 0.3.0 adds Butina clustering support, improved performance to MMFF relaxation and conformer generation, and increased compatibility with libraries and compilers.

### Contributors
- Kevin Boyd (@scal444)
- Eva Xue (@evasnow1992)
- Xuangui Huang (@stslxg-nv)

### Features
- Butina clustering API enabled, using distance matrix input. On an H200 GPU, speedups of 400-1000x can be achieved on datasets up to 60k molecules
- Improvements to BFGS minimizer. Up to 5x speedup compared to nvMolKit v0.2 on batches of small molecules (<20 atoms), with ~10-20% speedup in the general case. Applies to both MMFF relaxation and conformer generation.
- Conda-forge releases now support RDKit versions 2024.9.6 to 2025.9.3

### Bug Fixes
- Fixed a bug where synchronizations on the wrong stream could lead to data races in tests (Issue #28)
- Fixed several areas where a memcpy could go out of scope before completing (Issue #28, Issue #29)
- Fixed a bug where ETKDG would exit early with small CPU counts due to an incorrect identification of resource mis-configuration (Issue #31)

### Miscellaneous
- (C++) Added support for CUB/CCCL > v2.8
- (C++) Added support for externally specified CCCL
- (C++) Added support for CUDA 13.0


## 0.2.0 - 2025-10-24

### Summary

nvMolKit 0.2.0 comes with significant usability and feature-completeness improvements to existing functionality. It is also
the first release to have a [conda-forge release](https://anaconda.org/conda-forge/nvmolkit).

### Contributors
- Kevin Boyd (@scal444)
- Eva Xue (@evasnow1992)
- Ignacio Pickering (@ignaciojpickering)

### Features
- Add memory-segmented cross-similarity code, enabling larger datasets on systems with limited GPU memory ([#13](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/13))
- Support conformer deduplication in ETKDG conformer generation ([#14](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/14))
- Allow molecules > 256 atoms in conformer generation and MMFF optimization ([#16](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/16))
- Enable all combinations of (ET)(K)(DG) in conformer generator ([#17](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/17))

### Bug Fixes
- Fix compilation error on C++ build with target=native on Hopper architecture GPUs. ([#6](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/6))
- Fix lack of device-set cleanup in multi-GPU code ([#8](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/8))
- Fix bug in fingerprint bool->bitfield packing/unpacking code ([#11](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/11))
- Fix integer overflow leading to incorrect allocations in similarity calculation code. ([#20](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/20))
- Fix crash in most multithreaded APIs whenever exceptions are thrown inside of OpenMP loop. Exceptions now properly propagated to python ([#18](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/18))

### Miscellaneous
- Removed unsupported Bulk Similarity APIs ([#12](https://github.com/NVIDIA-Digital-Bio/nvMolKit/pull/12))
