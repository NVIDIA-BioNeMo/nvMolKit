# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import numpy as np
import pytest
import torch
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import BulkCosineSimilarity, BulkTanimotoSimilarity, ExplicitBitVect

from nvmolkit.fingerprints import MorganFingerprintGenerator, pack_fingerprint
from nvmolkit.similarity import (
    crossCosineSimilarity,
    crossCosineSimilarityMemoryConstrained,
    crossTanimotoSimilarity,
    crossTanimotoSimilarityMemoryConstrained,
)
from nvmolkit.types import AsyncGpuResult

# --------------------------------
# Test helpers
# --------------------------------


def replicate_mols(mols, target_len):
    base = list(mols)
    reps = (target_len + len(base) - 1) // len(base)
    return (base * reps)[:target_len]


def make_rdkit_and_gpu_fps(mols, target_len, fp_size=1024):
    """Replicate molecules to target_len and compute RDKit and nvMolKit fingerprints.

    Returns (rdkit_fps_list, torch_gpu_packed_bit_ints)
    """
    mols_rep = replicate_mols(mols, target_len)
    fpgen_rd = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=fp_size)
    rdkit_fps = [fpgen_rd.GetFingerprint(mol) for mol in mols_rep]

    nvmolkit_fpgen = MorganFingerprintGenerator(radius=3, fpSize=fp_size)
    nvmolkit_fps_cu = nvmolkit_fpgen.GetFingerprints(mols_rep, num_threads=1)
    t = nvmolkit_fps_cu.torch()
    assert t.shape == (target_len, fp_size // 32)
    assert t.device.type == "cuda"
    return rdkit_fps, t


# --------------------------------
# Edge cases and failure tests.
# --------------------------------
@pytest.mark.parametrize("similarity", [crossTanimotoSimilarity, crossCosineSimilarity])
def test_cross_similarity_fp_mismatch(similarity, size_limited_mols):
    nvmolkit_fpgen = MorganFingerprintGenerator(radius=3, fpSize=128)
    nvmolkit_fpgen2 = MorganFingerprintGenerator(radius=3, fpSize=256)
    nvmolkit_fps_cu = nvmolkit_fpgen.GetFingerprints(size_limited_mols, num_threads=1)
    nvmolkit_fps_cu2 = nvmolkit_fpgen2.GetFingerprints(size_limited_mols, num_threads=1)
    with pytest.raises(ValueError):
        similarity(nvmolkit_fps_cu, nvmolkit_fps_cu2)


def test_cross_similarity_zero_fingerprint():
    fingerprint = torch.zeros((1, 8), dtype=torch.int32, device="cuda")

    assert crossTanimotoSimilarity(fingerprint).torch().item() == 1.0
    assert crossCosineSimilarity(fingerprint).torch().item() == 0.0


@pytest.mark.parametrize("similarity", [crossTanimotoSimilarity, crossCosineSimilarity])
def test_cross_similarity_rejects_zero_word_fingerprints(similarity):
    fingerprints = torch.empty((2, 0), dtype=torch.uint32, device="cuda")

    with pytest.raises(ValueError, match="at least one fingerprint word"):
        similarity(fingerprints)


# --------------------------------
# Tanimoto similarity tests
# --------------------------------


def test_nvmolkit_cross_tanimoto_similarity_from_nvmolkit_fp(size_limited_mols):
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
    nvmolkit_fpgen = MorganFingerprintGenerator(radius=3, fpSize=1024)

    fps = [fpgen.GetFingerprint(mol) for mol in size_limited_mols]
    nvmolkit_fps_cu = nvmolkit_fpgen.GetFingerprints(size_limited_mols, num_threads=1)
    nvmolkit_fps_torch = nvmolkit_fps_cu.torch()
    assert nvmolkit_fps_torch.shape == (len(size_limited_mols), 1024 // 32)
    assert nvmolkit_fps_torch.device.type == "cuda"
    ref_sims = torch.empty(len(fps), len(fps), dtype=torch.float64)
    for i in range(len(fps)):
        ref_sims[i] = torch.tensor(BulkTanimotoSimilarity(fps[i], fps))
    ref_sims = ref_sims.to("cuda")
    nvmolkit_sims = crossTanimotoSimilarity(nvmolkit_fps_torch).torch()

    torch.testing.assert_close(nvmolkit_sims, ref_sims)

    # Test that the same API can be used with the AsyncGpuResult directly.
    nvmolkit_sims_direct = crossTanimotoSimilarity(nvmolkit_fps_cu)
    torch.testing.assert_close(nvmolkit_sims_direct.torch(), ref_sims)


@pytest.mark.parametrize("nxmdims", ((1, 20), (20, 1), (20, 2), (29, 29)))
def test_nxm_cross_tanimoto_similarity_from_nvmolkit_fp(size_limited_mols, nxmdims):
    d1, d2 = nxmdims
    fps1, nvmolkit_fps_torch1 = make_rdkit_and_gpu_fps(size_limited_mols, d1, fp_size=1024)
    fps2, nvmolkit_fps_torch2 = make_rdkit_and_gpu_fps(size_limited_mols, d2, fp_size=1024)

    ref_sims = torch.empty(len(fps1), len(fps2), dtype=torch.float64)
    for i in range(len(fps1)):
        ref_sims[i, :] = torch.tensor(BulkTanimotoSimilarity(fps1[i], fps2))
    ref_sims = ref_sims.to("cuda")
    nvmolkit_sims = crossTanimotoSimilarity(nvmolkit_fps_torch1, nvmolkit_fps_torch2).torch()
    torch.testing.assert_close(nvmolkit_sims, ref_sims)


@pytest.mark.parametrize("nxmdims", ((1, 20), (2, 10), (50, 50)))
def test_nxm_cross_tanimoto_similarity_from_packing(nxmdims):
    d1, d2 = nxmdims

    fps_1 = torch.randint(0, 2, (d1, 2048), dtype=torch.bool).to("cuda")
    fps_2 = torch.randint(0, 2, (d2, 2048), dtype=torch.bool).to("cuda")
    nvmolkit_fps_torch1 = pack_fingerprint(fps_1)
    nvmolkit_fps_torch2 = pack_fingerprint(fps_2)

    bitvects_a = []
    bitvects_b = []
    for i in range(d1):
        bv = ExplicitBitVect(2048)
        for bit in range(2048):
            if fps_1[i, bit].item():
                bv.SetBit(bit)
        bitvects_a.append(bv)
    for i in range(d2):
        bv = ExplicitBitVect(2048)
        for bit in range(2048):
            if fps_2[i, bit].item():
                bv.SetBit(bit)
        bitvects_b.append(bv)

    ref_sims = torch.zeros(d1, d2, dtype=torch.float64)
    for i in range(d1):
        ref_sims[i, :] = torch.tensor(BulkTanimotoSimilarity(bitvects_a[i], bitvects_b))
    ref_sims = ref_sims.to("cuda")
    nvmolkit_sims = crossTanimotoSimilarity(nvmolkit_fps_torch1, nvmolkit_fps_torch2).torch()
    torch.testing.assert_close(nvmolkit_sims, ref_sims)


@pytest.mark.parametrize("similarity", (crossTanimotoSimilarity, crossCosineSimilarity))
@pytest.mark.parametrize("input_kind", ("async", "cpu_tensor", "numpy"))
def test_cross_similarity_accepts_array_input_types(similarity, input_kind):
    fps_1 = torch.randint(0, 2, (6, 256), dtype=torch.bool, device="cuda")
    fps_2 = torch.randint(0, 2, (5, 256), dtype=torch.bool, device="cuda")
    packed_1 = pack_fingerprint(fps_1)
    packed_2 = pack_fingerprint(fps_2)

    expected = similarity(packed_1, packed_2).torch()

    if input_kind == "async":
        inp_1 = AsyncGpuResult(packed_1)
        inp_2 = AsyncGpuResult(packed_2)
    elif input_kind == "cpu_tensor":
        inp_1 = packed_1.cpu()
        inp_2 = packed_2.cpu()
    else:
        inp_1 = packed_1.cpu().numpy()
        inp_2 = packed_2.cpu().numpy()

    got = similarity(inp_1, inp_2).torch()

    torch.testing.assert_close(got, expected)


# --------------------------------
# Cosine similarity tests
# --------------------------------


def test_nvmolkit_cross_cosine_similarity_from_nvmolkit_fp(size_limited_mols):
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
    nvmolkit_fpgen = MorganFingerprintGenerator(radius=3, fpSize=1024)

    fps = [fpgen.GetFingerprint(mol) for mol in size_limited_mols]
    nvmolkit_fps_cu = nvmolkit_fpgen.GetFingerprints(size_limited_mols, num_threads=1)
    nvmolkit_fps_torch = nvmolkit_fps_cu.torch()
    assert nvmolkit_fps_torch.shape == (len(size_limited_mols), 1024 // 32)
    assert nvmolkit_fps_torch.device.type == "cuda"
    ref_sims = torch.empty(len(fps), len(fps), dtype=torch.float64)
    for i in range(len(fps)):
        ref_sims[i] = torch.tensor(BulkCosineSimilarity(fps[i], fps))
    ref_sims = ref_sims.to("cuda")
    nvmolkit_sims = crossCosineSimilarity(nvmolkit_fps_torch).torch()

    torch.testing.assert_close(nvmolkit_sims, ref_sims)

    # Test that the same API can be used with the AsyncGpuResult directly.
    nvmolkit_sims_direct = crossCosineSimilarity(nvmolkit_fps_cu)
    torch.testing.assert_close(nvmolkit_sims_direct.torch(), ref_sims)


@pytest.mark.parametrize("nxmdims", ((1, 20), (2, 10), (4, 5), (5, 4), (10, 2), (20, 1)))
def test_nxm_cross_cosine_similarity_from_nvmolkit_fp(size_limited_mols, nxmdims):
    d1, d2 = nxmdims
    fps1, nvmolkit_fps_torch1 = make_rdkit_and_gpu_fps(size_limited_mols, d1, fp_size=1024)
    fps2, nvmolkit_fps_torch2 = make_rdkit_and_gpu_fps(size_limited_mols, d2, fp_size=1024)

    ref_sims = torch.empty(len(fps1), len(fps2), dtype=torch.float64)
    for i in range(len(fps1)):
        ref_sims[i, :] = torch.tensor(BulkCosineSimilarity(fps1[i], fps2))
    ref_sims = ref_sims.to("cuda")
    nvmolkit_sims = crossCosineSimilarity(nvmolkit_fps_torch1, nvmolkit_fps_torch2).torch()

    torch.testing.assert_close(nvmolkit_sims, ref_sims)


# --------------------------------
# Memory-constrained CPU result tests
# --------------------------------


def test_memory_constrained_tanimoto_self(size_limited_mols):
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
    nvmolkit_fpgen = MorganFingerprintGenerator(radius=3, fpSize=1024)

    fps = [fpgen.GetFingerprint(mol) for mol in size_limited_mols]
    ref = torch.empty(len(fps), len(fps), dtype=torch.float64)
    for i in range(len(fps)):
        ref[i] = torch.tensor(BulkTanimotoSimilarity(fps[i], fps))

    # GPU-packed fingerprints (torch CUDA tensor)
    nvmolkit_fps_cu = nvmolkit_fpgen.GetFingerprints(size_limited_mols, num_threads=1)
    nvmolkit_fps_torch = nvmolkit_fps_cu.torch()
    got = crossTanimotoSimilarityMemoryConstrained(nvmolkit_fps_torch)
    # Compare as numpy
    np.testing.assert_allclose(got, ref.cpu().numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("nxmdims", ((1, 20), (20, 1), (50, 50)))
def test_memory_constrained_tanimoto_cross(size_limited_mols, nxmdims):
    d1, d2 = nxmdims
    fps1, t1 = make_rdkit_and_gpu_fps(size_limited_mols, d1, fp_size=1024)
    fps2, t2 = make_rdkit_and_gpu_fps(size_limited_mols, d2, fp_size=1024)
    ref = torch.empty(d1, d2, dtype=torch.float64)
    for i in range(d1):
        ref[i] = torch.tensor(BulkTanimotoSimilarity(fps1[i], fps2))
    got = crossTanimotoSimilarityMemoryConstrained(t1, t2)
    np.testing.assert_allclose(got, ref.cpu().numpy(), rtol=1e-5, atol=1e-5)


def test_memory_constrained_cosine_self(size_limited_mols):
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
    nvmolkit_fpgen = MorganFingerprintGenerator(radius=3, fpSize=1024)

    fps = [fpgen.GetFingerprint(mol) for mol in size_limited_mols]
    ref = torch.empty(len(fps), len(fps), dtype=torch.float64)
    for i in range(len(fps)):
        ref[i] = torch.tensor(BulkCosineSimilarity(fps[i], fps))

    nvmolkit_fps_cu = nvmolkit_fpgen.GetFingerprints(size_limited_mols, num_threads=1)
    nvmolkit_fps_torch = nvmolkit_fps_cu.torch()
    got = crossCosineSimilarityMemoryConstrained(nvmolkit_fps_torch)
    np.testing.assert_allclose(got, ref.cpu().numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("nxmdims", ((1, 20), (20, 1), (50, 50)))
def test_memory_constrained_cosine_cross(size_limited_mols, nxmdims):
    d1, d2 = nxmdims
    fps1, t1 = make_rdkit_and_gpu_fps(size_limited_mols, d1, fp_size=1024)
    fps2, t2 = make_rdkit_and_gpu_fps(size_limited_mols, d2, fp_size=1024)
    ref = torch.empty(d1, d2, dtype=torch.float64)
    for i in range(d1):
        ref[i] = torch.tensor(BulkCosineSimilarity(fps1[i], fps2))
    got = crossCosineSimilarityMemoryConstrained(t1, t2)
    np.testing.assert_allclose(got, ref.cpu().numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "similarity",
    (crossTanimotoSimilarityMemoryConstrained, crossCosineSimilarityMemoryConstrained),
)
@pytest.mark.parametrize("input_kind", ("async", "cpu_tensor", "numpy"))
def test_memory_constrained_similarity_accepts_array_input_types(similarity, input_kind):
    fps_1 = torch.randint(0, 2, (6, 256), dtype=torch.bool, device="cuda")
    fps_2 = torch.randint(0, 2, (5, 256), dtype=torch.bool, device="cuda")
    packed_1 = pack_fingerprint(fps_1)
    packed_2 = pack_fingerprint(fps_2)

    expected = similarity(packed_1, packed_2)

    if input_kind == "async":
        inp_1 = AsyncGpuResult(packed_1)
        inp_2 = AsyncGpuResult(packed_2)
    elif input_kind == "cpu_tensor":
        inp_1 = packed_1.cpu()
        inp_2 = packed_2.cpu()
    else:
        inp_1 = packed_1.cpu().numpy()
        inp_2 = packed_2.cpu().numpy()

    got = similarity(inp_1, inp_2)

    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


# --------------------------------
# Stream tests
# --------------------------------


@pytest.mark.parametrize(
    ("similarity", "bulk_similarity"),
    (
        (crossTanimotoSimilarity, BulkTanimotoSimilarity),
        (crossCosineSimilarity, BulkCosineSimilarity),
    ),
)
def test_similarity_on_explicit_stream(similarity, bulk_similarity, size_limited_mols):
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
    nvmolkit_fpgen = MorganFingerprintGenerator(radius=3, fpSize=1024)

    fps = [fpgen.GetFingerprint(mol) for mol in size_limited_mols]
    nvmolkit_fps_torch = nvmolkit_fpgen.GetFingerprints(size_limited_mols, num_threads=1).torch()
    torch.cuda.synchronize()

    ref_sims = torch.empty(len(fps), len(fps), dtype=torch.float64)
    for i in range(len(fps)):
        ref_sims[i] = torch.tensor(bulk_similarity(fps[i], fps))
    ref_sims = ref_sims.to("cuda")

    s = torch.cuda.Stream()
    nvmolkit_sims = similarity(nvmolkit_fps_torch, stream=s).torch()
    s.synchronize()

    torch.testing.assert_close(nvmolkit_sims, ref_sims)


@pytest.mark.parametrize("similarity", (crossTanimotoSimilarity, crossCosineSimilarity))
def test_similarity_invalid_stream_type(similarity):
    fps = torch.zeros((1, 8), dtype=torch.uint32, device="cuda")
    with pytest.raises(TypeError):
        similarity(fps, stream=42)
