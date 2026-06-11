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

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from nvmolkit.fingerprints import MorganFingerprintGenerator, pack_fingerprint, unpack_fingerprint


def test_roundtrip_pack_unpack():
    # Create a test boolean tensor
    n_fps = 10
    fp_size = 128
    test_fp = torch.randint(0, 2, (n_fps, fp_size), dtype=torch.bool, device="cuda")

    # Pack it
    packed = pack_fingerprint(test_fp)
    assert packed.shape == (n_fps, fp_size // 32)
    assert packed.device.type == "cuda"

    # Unpack it
    unpacked = unpack_fingerprint(packed)
    # Verify roundtrip
    torch.testing.assert_close(test_fp, unpacked)


def test_pack_unpack_uneven_size():
    fp_size = 127
    n_fps = 10
    test_fp = torch.randint(0, 2, (n_fps, fp_size), dtype=torch.bool, device="cpu")
    packed = pack_fingerprint(test_fp)
    assert packed.shape == (n_fps, 4)
    unpacked = unpack_fingerprint(packed)
    assert unpacked.shape == (n_fps, 128)
    torch.testing.assert_close(test_fp, unpacked[:, :fp_size])


def test_unpack_invalid_dtype():
    with pytest.raises(ValueError):
        unpack_fingerprint(torch.randint(0, 2, (10, 32), device="cuda", dtype=torch.int64))


@pytest.mark.parametrize("fpSize", (17, 8192))
def test_nvmolkit_fingerprint_throws_on_invalid_fpsize(fpSize, size_limited_mols):
    fpgen = MorganFingerprintGenerator(radius=3, fpSize=fpSize)
    with pytest.raises(Exception):
        fpgen.GetFingerprints(size_limited_mols)


def test_empty_input():
    fpgen = MorganFingerprintGenerator(radius=3, fpSize=2048)
    fps = fpgen.GetFingerprints([]).torch()
    torch.cuda.synchronize()
    assert fps.shape == (0, 2048 // 32)


def test_invalid_input():
    fpgen = MorganFingerprintGenerator(radius=3, fpSize=2048)
    with pytest.raises(Exception):
        fpgen.GetFingerprints([None])


@pytest.mark.parametrize("fpSize", (128, 1024, 2048))
@pytest.mark.parametrize("radius", (0, 1, 3, 5))
def test_nvmolkit_morgan_fingerprint(size_limited_mols, fpSize, radius):
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)
    fps = [fpgen.GetFingerprint(mol) for mol in size_limited_mols]

    nvmolkit_fpgen = MorganFingerprintGenerator(radius=radius, fpSize=fpSize)
    nvmolkit_fps_torch = nvmolkit_fpgen.GetFingerprints(size_limited_mols).torch()
    torch.cuda.synchronize()
    assert nvmolkit_fps_torch.device.type == "cuda"
    want_n_rows = len(size_limited_mols)
    want_n_cols = fpSize / 32
    assert nvmolkit_fps_torch.shape == (want_n_rows, want_n_cols)
    for i in range(want_n_rows):
        ref_fp = fps[i]
        got_fp_row = nvmolkit_fps_torch[i, :]
        for j in range(fpSize):
            want_bit = ref_fp.GetBit(j)

            column = j // 32
            mask = 1 << (j % 32)

            got_bit = got_fp_row[column].item() & mask != 0
            assert got_bit == want_bit
        # Now test that the unpacked fingerprint matches the original
        unpacked = unpack_fingerprint(got_fp_row.unsqueeze(0))
        assert unpacked.shape == (
            1,
            fpSize,
        )
        assert unpacked.device.type == "cuda"
        assert unpacked.dtype == torch.bool
        torch.testing.assert_close(ref_fp.ToList(), unpacked.to(int).tolist()[0])


def test_fingerprints_on_explicit_stream(size_limited_mols):
    """Verify fingerprints computed with an explicit stream parameter produce correct results."""
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    ref_fps = [fpgen.GetFingerprint(mol) for mol in size_limited_mols]

    s = torch.cuda.Stream()
    gen = MorganFingerprintGenerator(radius=3, fpSize=2048)
    result = gen.GetFingerprints(size_limited_mols, stream=s).torch()
    s.synchronize()

    assert result.shape == (len(size_limited_mols), 2048 // 32)
    for i in range(len(size_limited_mols)):
        for j in range(2048):
            column = j // 32
            mask = 1 << (j % 32)
            got_bit = result[i, column].item() & mask != 0
            assert got_bit == ref_fps[i].GetBit(j)


def test_fingerprints_invalid_stream_type(size_limited_mols):
    gen = MorganFingerprintGenerator(radius=3, fpSize=2048)
    with pytest.raises(TypeError):
        gen.GetFingerprints(size_limited_mols, stream=42)


def test_gh_issue_84():
    """Regression test for https://github.com/NVIDIA/nvMolKit/issues/84."""
    mol = Chem.MolFromSmiles("CC1(C)C2=C(C=CC(=C2)P(C3=CC=CC=C3)C4=CC=CC=C4)OC5=C1C=CC(=C5)P(C6=CC=CC=C6)C7=CC=CC=C7")
    assert mol is not None

    configs = [(2, 512), (2, 1024), (3, 512), (3, 1024)]
    for i in range(256):
        radius, fp_size = configs[i % len(configs)]
        gen = MorganFingerprintGenerator(radius=radius, fpSize=fp_size)
        bits = unpack_fingerprint(gen.GetFingerprints([mol]).torch()).sum().item()
        assert bits > 0, f"Got empty fingerprint for BINAP on attempt {i}"


def test_gh_issue_195():
    """Regression test for https://github.com/NVIDIA-BioNeMo/nvMolKit/issues/195.

    A batch containing only molecules larger than the 128 atom/bond GPU buckets
    used to produce empty fingerprints.
    """
    radius = 3
    fp_size = 2048

    large_mol = Chem.MolFromSmiles("NCC(=O)" * 40)
    assert large_mol is not None
    assert large_mol.GetNumAtoms() >= 128 or large_mol.GetNumBonds() >= 128

    rdkit_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    ref_bits = rdkit_gen.GetFingerprint(large_mol).ToList()
    assert sum(ref_bits) > 0

    nvmolkit_gen = MorganFingerprintGenerator(radius=radius, fpSize=fp_size)

    for batch in ([large_mol], [large_mol, large_mol]):
        unpacked = unpack_fingerprint(nvmolkit_gen.GetFingerprints(batch).torch())
        torch.cuda.synchronize()
        assert unpacked.shape == (len(batch), fp_size)
        for row in range(len(batch)):
            assert unpacked[row].sum().item() > 0, "Large-only batch produced an empty fingerprint"
            torch.testing.assert_close(ref_bits, unpacked[row].to(int).tolist())
