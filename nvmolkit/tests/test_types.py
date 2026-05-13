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

import gc
import math

import pytest
import torch

from rdkit.Chem import MolFromSmiles

from nvmolkit.fingerprints import MorganFingerprintGenerator
from nvmolkit.types import (
    AsyncGpuResult,
    CoordinateOutput,
    Dense3DResult,
    Device3DResult,
    HardwareOptions,
)


def _get_fps(num_mols):
    generator = MorganFingerprintGenerator(radius=0, fpSize=2048)
    template = MolFromSmiles("CC")
    mols = [template] * num_mols

    result = generator.GetFingerprints(mols)
    torch.cuda.synchronize()
    return result


def test_async_gpu_result_release_frees_memory():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    base_free, _ = torch.cuda.mem_get_info()

    num_mols = 210_000
    expected_bytes = num_mols * 2048 // 8
    fps = _get_fps(num_mols)
    torch.cuda.synchronize()

    free_after_alloc, _ = torch.cuda.mem_get_info()
    assert free_after_alloc < base_free
    assert free_after_alloc + expected_bytes <= base_free

    del fps
    gc.collect()
    torch.cuda.synchronize()

    free_post, _ = torch.cuda.mem_get_info()

    assert (free_post - free_after_alloc) >= expected_bytes


@pytest.mark.parametrize("invalid_value", [0, -2, -99])
def test_hardware_options_invalid_batches_per_gpu(invalid_value):
    """Test that invalid batchesPerGpu values are rejected at construction time and via setter."""
    with pytest.raises(ValueError, match="batchesPerGpu must be greater than 0 or -1"):
        HardwareOptions(batchesPerGpu=invalid_value)

    hw = HardwareOptions()
    with pytest.raises(ValueError, match="batchesPerGpu must be greater than 0 or -1"):
        hw.batchesPerGpu = invalid_value


def _wrap(tensor):
    """Wrap a CUDA tensor as an AsyncGpuResult."""
    return AsyncGpuResult(tensor, gpu_id=tensor.device.index)


def _make_device_3d_result(
    atom_counts_per_conformer,
    mol_indices,
    n_mols,
    *,
    conf_indices=None,
    energies=None,
    converged=None,
):
    """Build a Device3DResult from a list of per-conformer atom counts.

    Values encode (conformer, atom, axis) so views can be checked without needing to run
    any GPU computation: the value is ``conformer * 1000 + atom * 10 + axis``.
    """
    assert len(atom_counts_per_conformer) == len(mol_indices)
    starts = [0]
    for atom_count in atom_counts_per_conformer:
        starts.append(starts[-1] + atom_count)
    total_atoms = starts[-1]

    values = torch.empty((total_atoms, 3), dtype=torch.float64, device="cuda")
    for conf_idx, atom_count in enumerate(atom_counts_per_conformer):
        for atom_idx in range(atom_count):
            row = starts[conf_idx] + atom_idx
            for axis in range(3):
                values[row, axis] = conf_idx * 1000 + atom_idx * 10 + axis

    atom_starts = torch.tensor(starts, dtype=torch.int32, device="cuda")
    mol_indices_t = torch.tensor(mol_indices, dtype=torch.int32, device="cuda")
    if conf_indices is None:
        per_mol_counter = {}
        conf_indices = []
        for mol_idx in mol_indices:
            slot = per_mol_counter.get(mol_idx, 0)
            conf_indices.append(slot)
            per_mol_counter[mol_idx] = slot + 1
    assert len(conf_indices) == len(mol_indices)
    conf_indices_t = torch.tensor(conf_indices, dtype=torch.int32, device="cuda")

    energies_wrapped = _wrap(energies) if energies is not None else None
    converged_wrapped = _wrap(converged) if converged is not None else None

    return Device3DResult(
        values=_wrap(values),
        atom_starts=_wrap(atom_starts),
        mol_indices=_wrap(mol_indices_t),
        conf_indices=_wrap(conf_indices_t),
        gpu_id=values.device.index,
        n_mols=n_mols,
        energies=energies_wrapped,
        converged=converged_wrapped,
    )


def test_coordinate_output_enum_values():
    assert CoordinateOutput.RDKIT_CONFORMERS.value == "rdkit"
    assert CoordinateOutput.DEVICE.value == "device"


def test_device_3d_result_num_conformers_matches_atom_starts():
    result = _make_device_3d_result(
        atom_counts_per_conformer=[3, 5, 2],
        mol_indices=[0, 0, 1],
        n_mols=2,
    )
    assert result.num_conformers == 3


def test_device_3d_result_per_molecule_groups_by_mol_index():
    """per_molecule routes each conformer slice to its mol_indices bucket."""
    result = _make_device_3d_result(
        atom_counts_per_conformer=[2, 3, 4],
        mol_indices=[1, 0, 1],
        n_mols=2,
    )
    nested = result.per_molecule()
    assert len(nested) == 2
    assert len(nested[0]) == 1
    assert len(nested[1]) == 2

    torch.cuda.synchronize()
    assert nested[0][0].shape == (3, 3)
    assert nested[1][0].shape == (2, 3)
    assert nested[1][1].shape == (4, 3)

    # Conformer 0 (mol 1) atom 0 axis 0 -> 0*1000+0*10+0 = 0
    # Conformer 1 (mol 0) atom 0 axis 0 -> 1*1000+0*10+0 = 1000
    # Conformer 2 (mol 1) atom 0 axis 0 -> 2*1000+0*10+0 = 2000
    assert nested[1][0][0, 0].item() == 0
    assert nested[0][0][0, 0].item() == 1000
    assert nested[1][1][0, 0].item() == 2000

    # Final atom of conformer 1 (mol 0): atom 2, axis 2 -> 1000+20+2 = 1022
    assert nested[0][0][2, 2].item() == 1022


def test_device_3d_result_per_molecule_handles_empty_molecules():
    """n_mols larger than max(mol_indices)+1 leaves trailing empty buckets."""
    result = _make_device_3d_result(
        atom_counts_per_conformer=[2],
        mol_indices=[0],
        n_mols=4,
    )
    nested = result.per_molecule()
    assert len(nested) == 4
    assert len(nested[0]) == 1
    assert nested[1] == []
    assert nested[2] == []
    assert nested[3] == []


def test_device_3d_result_per_molecule_views_share_storage():
    """per_molecule returns views, not copies; mutating the view mutates values."""
    result = _make_device_3d_result(
        atom_counts_per_conformer=[3],
        mol_indices=[0],
        n_mols=1,
    )
    nested = result.per_molecule()
    nested[0][0][0, 0] = -7.0
    torch.cuda.synchronize()
    assert result.values.torch()[0, 0].item() == -7.0


def test_device_3d_result_dense_returns_dense_3d_result():
    """dense() returns a Dense3DResult NamedTuple with values and both masks."""
    result = _make_device_3d_result(
        atom_counts_per_conformer=[2, 4, 3],
        mol_indices=[0, 1, 2],
        n_mols=3,
    )
    out = result.dense()
    assert isinstance(out, Dense3DResult)
    assert out.values.shape == (3, 1, 4, 3)
    assert out.conf_mask.shape == (3, 1)
    assert out.atom_mask.shape == (3, 1, 4)
    assert out.values.dtype == torch.float64
    assert out.conf_mask.dtype == torch.bool
    assert out.atom_mask.dtype == torch.bool


def test_device_3d_result_dense_distinguishes_per_mol_conformer_counts():
    """5/3/2 conformers across three molecules round-trip through dense() shape and masks."""
    counts = [5, 3, 2]
    n_mols = len(counts)
    atom_counts = []
    mol_indices = []
    conf_indices = []
    for mol_idx, conf_count in enumerate(counts):
        for conf_slot in range(conf_count):
            atom_counts.append(2 + (conf_slot % 3))
            mol_indices.append(mol_idx)
            conf_indices.append(conf_slot)

    result = _make_device_3d_result(
        atom_counts_per_conformer=atom_counts,
        mol_indices=mol_indices,
        n_mols=n_mols,
        conf_indices=conf_indices,
    )
    out = result.dense()
    torch.cuda.synchronize()

    max_confs = max(counts)
    max_atoms = max(atom_counts)
    assert out.values.shape == (n_mols, max_confs, max_atoms, 3)
    assert out.conf_mask.shape == (n_mols, max_confs)
    assert out.atom_mask.shape == (n_mols, max_confs, max_atoms)

    assert out.conf_mask.sum(dim=1).tolist() == counts

    expected_atoms_per_mol = [0, 0, 0]
    cursor = 0
    for mol_idx, conf_count in enumerate(counts):
        expected_atoms_per_mol[mol_idx] = sum(atom_counts[cursor : cursor + conf_count])
        cursor += conf_count
    assert out.atom_mask.sum(dim=(1, 2)).tolist() == expected_atoms_per_mol

    # mol 0 conf 4 atom 0 axis 0: this is flat conformer 4, encoded value = 4*1000 + 0*10 + 0 = 4000.
    assert out.values[0, 4, 0, 0].item() == 4000
    # mol 1 conf 2 atom 0 axis 0: flat conformer 5+2 = 7, encoded 7000.
    assert out.values[1, 2, 0, 0].item() == 7000
    # mol 2 conf 1 atom 0 axis 0: flat conformer 5+3+1 = 9, encoded 9000.
    assert out.values[2, 1, 0, 0].item() == 9000
    # mol 1 has only 3 confs, so slot 3,4 are padded NaN at every atom.
    assert math.isnan(out.values[1, 4, 0, 0].item())
    # mol 2 has only 2 confs, so slot 2,3,4 padded.
    assert math.isnan(out.values[2, 2, 0, 0].item())


def test_device_3d_result_dense_conf_mask_marks_failed_molecules():
    """A molecule with zero conformers gets a fully-False conf_mask row."""
    result = _make_device_3d_result(
        atom_counts_per_conformer=[3, 2],
        mol_indices=[0, 2],
        n_mols=3,
    )
    out = result.dense()
    torch.cuda.synchronize()
    assert out.conf_mask[0].any().item()
    assert not out.conf_mask[1].any().item()
    assert out.conf_mask[2].any().item()
    assert torch.isnan(out.values[1]).all().item()
    assert not out.atom_mask[1].any().item()


def test_device_3d_result_dense_atom_mask_marks_short_conformers():
    """atom_mask has False past each conformer's true atom count, even when conf_mask is True."""
    result = _make_device_3d_result(
        atom_counts_per_conformer=[2, 4],
        mol_indices=[0, 0],
        n_mols=1,
    )
    out = result.dense()
    torch.cuda.synchronize()
    assert out.values.shape == (1, 2, 4, 3)
    assert out.conf_mask[0].tolist() == [True, True]
    assert out.atom_mask[0, 0].tolist() == [True, True, False, False]
    assert out.atom_mask[0, 1].tolist() == [True, True, True, True]
    assert math.isnan(out.values[0, 0, 2, 0].item())
    assert not math.isnan(out.values[0, 0, 1, 0].item())


def test_device_3d_result_dense_custom_pad_value():
    result = _make_device_3d_result(
        atom_counts_per_conformer=[1, 3],
        mol_indices=[0, 1],
        n_mols=2,
    )
    out = result.dense(pad_value=-1.0)
    torch.cuda.synchronize()
    assert out.values.shape == (2, 1, 3, 3)
    assert out.values[0, 0, 1, 0].item() == -1.0
    assert out.values[0, 0, 2, 2].item() == -1.0
    assert out.values[1, 0, 2, 2].item() == 1000 + 20 + 2


def test_device_3d_result_dense_empty_returns_zero_shape():
    result = _make_device_3d_result(
        atom_counts_per_conformer=[],
        mol_indices=[],
        n_mols=3,
    )
    out = result.dense()
    assert out.values.shape == (3, 0, 0, 3)
    assert out.conf_mask.shape == (3, 0)
    assert out.atom_mask.shape == (3, 0, 0)
    assert out.values.dtype == torch.float64
