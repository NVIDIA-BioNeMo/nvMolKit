---
name: nvmolkit-usage
description: Write code that calls the installed nvMolKit Python API for GPU-accelerated, batched RDKit-style operations - Morgan fingerprints, Tanimoto/cosine similarity, ETKDG conformer embedding, MMFF/UFF optimization, TFD, conformer RMSD, Butina clustering, and substructure search. Use when the user is importing `nvmolkit.*`, debugging an `nvmolkit` call, choosing between nvMolKit and RDKit for a batched cheminformatics workflow, or wiring nvMolKit results into a torch/numpy pipeline. Out of scope: building nvMolKit from source.
---

# nvMolKit usage

## What nvMolKit is

GPU-accelerated, batched implementations of common RDKit operations. APIs mirror RDKit where possible but are batch-oriented: they take lists of `rdkit.Chem.Mol` (or lists of fingerprints) and process them in parallel on one or more GPUs. nvMolKit links against RDKit at build time; inputs and outputs are real RDKit `Mol` objects.

## Where nvMolKit does well

Reach for nvMolKit when:

- The workload is **a large batch of molecules** processed together (typically thousands or more).
- The metric is **throughput / total wall time across the batch**, not per-molecule latency.
- The same operation is **repeated identically** across the batch (fingerprinting a library, embedding/minimizing many conformers, bulk pairwise similarity), so the GPU stays saturated.

Plain RDKit is usually the better choice for single-molecule one-offs or workflows that can't be expressed as a batch. nvMolKit is not meant to replace RDKit for those cases.

## Runtime requirements

- An NVIDIA GPU with compute capability 7.0 (V100) or higher
- A CUDA driver compatible with CUDA 12.6+.
- A working `torch` install with CUDA support (nvMolKit returns GPU tensors via `torch`'s CUDA array interface).

If CUDA is unavailable, nvMolKit calls raise. There is no CPU fallback - if the user needs one, use RDKit directly for that path.

## Verify the install before writing real code

Run this once to confirm nvMolKit is importable and a GPU op works end to end:

```python
import nvmolkit
import torch
from rdkit import Chem
from nvmolkit.fingerprints import MorganFingerprintGenerator

print("nvmolkit:", nvmolkit.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

mols = [Chem.MolFromSmiles(smi) for smi in ["CCO", "c1ccccc1", "CC(=O)O"]]
fpgen = MorganFingerprintGenerator(radius=2, fpSize=1024)
result = fpgen.GetFingerprints(mols)
torch.cuda.synchronize()
fps = result.torch()
print("fps shape:", tuple(fps.shape), "dtype:", fps.dtype)
# Expected: shape (3, 32), dtype torch.int32  (1024 bits packed into 32 int32s per row)
```

If this fails, point the user at the install guide on the docs site rather than guessing - see "Going deeper" below.

## Entry points

| Task | Module | Primary entry point |
|---|---|---|
| Morgan fingerprints | `nvmolkit.fingerprints` | `MorganFingerprintGenerator(radius, fpSize).GetFingerprints(mols)` |
| Bulk Tanimoto / cosine similarity | `nvmolkit.similarity` | `crossTanimotoSimilarity(...)`, `crossCosineSimilarity(...)`, plus `*MemoryConstrained` variants for results too large to fit in GPU memory |
| ETKDG conformer embedding | `nvmolkit.embedMolecules` | `EmbedMolecules(molecules, params, confsPerMolecule, ...)` |
| MMFF94 optimization | `nvmolkit.mmffOptimization` | `MMFFOptimizeMoleculesConfs(molecules, ...)` |
| UFF optimization | `nvmolkit.uffOptimization` | `UFFOptimizeMoleculesConfs(molecules, ...)` |
| Pairwise conformer RMSD | `nvmolkit.conformerRmsd` | `GetConformerRMSMatrix(mol)`, `GetConformerRMSMatrixBatch(mols)` |
| Torsion Fingerprint Deviation (TFD) | `nvmolkit.tfd` | `GetTFDMatrix(mol)`, `GetTFDMatrices(mols)` |
| Butina clustering | `nvmolkit.clustering` | `butina(distance_matrix, cutoff)` (precomputed matrix), `fused_butina(fingerprints, cutoff)` (memory-efficient, on-the-fly) |
| Substructure search | `nvmolkit.substructure` | `hasSubstructMatch`, `countSubstructMatches`, `getSubstructMatches` |
| Hardware tuning (batch size, GPU IDs) | `nvmolkit.types` | `HardwareOptions(...)` passed to ETKDG / MMFF / UFF |
| Optional autotuning of `HardwareOptions` | `nvmolkit.autotune` | `tune_embed_molecules`, `tune_mmff_optimize`, `tune_uff_optimize`, `tune_batched_forcefield`. Requires the `optuna` package |

## Async results: `AsyncGpuResult`

Operations that return GPU-resident data (fingerprints, similarity matrices, RMSD/TFD vectors) return an `AsyncGpuResult`. Key behaviors:

- It is asynchronous. The kernel may not have completed when the call returns.
- `result.torch()` returns a zero-copy `torch.Tensor` on the GPU. Caller is responsible for synchronizing before reading values on the host.
- `result.numpy()` synchronizes and returns a CPU numpy array.
- `AsyncGpuResult` exposes `__cuda_array_interface__`, so it can be passed directly into other nvMolKit functions (e.g. fingerprints -> similarity) with no host round-trip.

In-place APIs (ETKDG embedding, MMFF/UFF optimization) instead modify the input `Mol` objects directly: optimized coordinates land in each mol's conformer list. MMFF and UFF additionally return per-molecule lists of energies.

## Recipes

### Morgan fingerprints + bulk Tanimoto similarity

```python
import torch
from rdkit import Chem
from nvmolkit.fingerprints import MorganFingerprintGenerator
from nvmolkit.similarity import crossTanimotoSimilarity

smiles = ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "CCOCC"]
mols = [Chem.MolFromSmiles(smi) for smi in smiles]

fpgen = MorganFingerprintGenerator(radius=2, fpSize=1024)
fps = fpgen.GetFingerprints(mols)

sim = crossTanimotoSimilarity(fps)
torch.cuda.synchronize()
print(sim.torch())
```

Inputs are `list[Mol]`. Output of `GetFingerprints` is an `AsyncGpuResult` wrapping an `(n_mols, fpSize / 32)` int32 tensor of packed bits. Pass it straight into `crossTanimotoSimilarity` for an `(n, n)` similarity matrix; pass two fingerprint sets for an `(n, m)` cross-matrix. For sets too large to materialize on the GPU, use `crossTanimotoSimilarityMemoryConstrained` (chunked compute, returns numpy on CPU).

### ETKDG conformer embedding

```python
from rdkit.Chem import AddHs, MolFromSmiles
from rdkit.Chem.rdDistGeom import ETKDGv3
from nvmolkit.embedMolecules import EmbedMolecules

mols = [AddHs(MolFromSmiles(smi)) for smi in ["C1CCCCC1", "C1CCCCC2CCCCC12", "COO"]]
params = ETKDGv3()
params.useRandomCoords = True

EmbedMolecules(mols, params, confsPerMolecule=10, maxIterations=-1)

for mol in mols:
    print(mol.GetNumConformers())
```

Inputs are `list[Mol]`, sanitized and with hydrogens added (`AddHs`). Conformers are added in-place. `params.useRandomCoords` must be `True` - nvMolKit's ETKDG only supports random-coord initialization. A handful of niche `EmbedParameters` options are not supported (bounds matrices, custom CPCI, coord maps, separate-fragment embedding); the Features section of the docs site lists the full restrictions.

### MMFF94 minimization of a batch of conformers

```python
from rdkit.Chem import AddHs, MolFromSmiles
from rdkit.Chem.rdDistGeom import ETKDGv3
from nvmolkit.embedMolecules import EmbedMolecules
from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs

mols = [AddHs(MolFromSmiles(smi)) for smi in ["CCO", "CCN", "c1ccccc1"]]
params = ETKDGv3(); params.useRandomCoords = True
EmbedMolecules(mols, params, confsPerMolecule=5)

energies = MMFFOptimizeMoleculesConfs(mols, maxIters=500)
for mol, mol_energies in zip(mols, energies):
    print(mol.GetNumConformers(), mol_energies)
```

Inputs are `list[Mol]` with conformers already populated (typically by ETKDG, RDKit's `EmbedMultipleConfs`, or a prior nvMolKit call). Coordinates are updated in place; the return is `list[list[float]]` of optimized energies aligned with the input molecule order and conformer index. UFF is identical in shape: swap in `from nvmolkit.uffOptimization import UFFOptimizeMoleculesConfs`.

If any input molecule is `None` or lacks MMFF/UFF atom types, the call raises `ValueError`. The exception's `args[1]` is a dict with keys `"none"` and `"no_params"` listing the offending indices - useful for filtering a noisy input set.

### Tuning batch size on a single GPU

```python
from rdkit.Chem import AddHs, MolFromSmiles
from rdkit.Chem.rdDistGeom import ETKDGv3
from nvmolkit.types import HardwareOptions
from nvmolkit.embedMolecules import EmbedMolecules

mols = [AddHs(MolFromSmiles(smi)) for smi in ["CCO", "CCN", "c1ccccc1"]]
params = ETKDGv3(); params.useRandomCoords = True

opts = HardwareOptions(
    preprocessingThreads=12,
    batchSize=500,
    batchesPerGpu=4,
    gpuIds=[0],
)
EmbedMolecules(mols, params, confsPerMolecule=10, hardwareOptions=opts)
for mol in mols:
    print(mol.GetNumConformers())
```

For multi-GPU, scale `gpuIds` to the actually-visible device IDs (or pass an empty list to let nvMolKit use every visible GPU). See the next section for the full `HardwareOptions` surface.

## Hardware tuning surfaces

Two configuration objects expose the GPU/CPU knobs. They are reference data, not recipes.

### `HardwareOptions` (ETKDG, MMFF, UFF)

`from nvmolkit.types import HardwareOptions`. Passed via `hardwareOptions=` to `EmbedMolecules`, `MMFFOptimizeMoleculesConfs`, `UFFOptimizeMoleculesConfs`. Every field has an "auto" sentinel; the defaults are usually fine.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `preprocessingThreads` | int | `-1` (all visible CPUs) | CPU threads for preprocessing |
| `batchSize` | int | `-1` (auto-tuned) | Number of conformers per GPU batch |
| `batchesPerGpu` | int | `-1` (auto) | Concurrent batches per GPU; must be `>0` or `-1` |
| `gpuIds` | `list[int]` | `[]` (all visible GPUs) | Specific device ordinals to target |

Passing a `gpuIds` entry for a device that isn't visible raises `RuntimeError: invalid device ordinal`. For finding good values automatically across a representative sample, see `nvmolkit.autotune` (requires the `optuna` extra); each `tune_*` function returns a `TuneResult` whose `best_config` is a fully-populated `HardwareOptions` ready to pass back into the real call.

`HardwareOptions` round-trips through `to_dict()` / `from_dict()` for persisting tuned configs to disk.

### `SubstructSearchConfig` (substructure search)

`from nvmolkit.substructure import SubstructSearchConfig`. Passed via `config=` to `hasSubstructMatch`, `countSubstructMatches`, and `getSubstructMatches`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `batchSize` | int | `1024` | (target, query) pairs per GPU batch |
| `workerThreads` | int | `-1` (auto) | GPU runner threads per GPU |
| `preprocessingThreads` | int | `-1` (auto) | CPU threads for preprocessing |
| `maxMatches` | int | `0` (unlimited) | Max matches returned per (target, query) pair |
| `uniquify` | bool | `False` | Drop duplicate matches that differ only in atom enumeration order |
| `gpuIds` | `list[int] \| None` | `None` (current device only) | Specific device ordinals to target |

Substructure search currently does not support chirality-aware matching, enhanced stereochemistry, or other advanced RDKit `SubstructMatchParameters` options.

## Going deeper

- Full feature list, API reference, and guides: <https://nvidia-digital-bio.github.io/nvMolKit/>
- What changed in each release: <https://nvidia-digital-bio.github.io/nvMolKit/changelog.html>
- Worked examples (Jupyter notebooks): the `examples/` directory in the GitHub repo
