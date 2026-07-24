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

"""GPU-accelerated fingerprint generation."""

import torch

from nvmolkit._Fingerprints import MorganFingerprintGenerator as InternalFPGen
from nvmolkit.types import AsyncGpuResult, _resolve_cuda_stream


def unpack_fingerprint(fp: torch.Tensor) -> torch.Tensor:
    """Unpack a 32-bit integer-encoded fingerprint into a 2D boolean tensor of shape (len(fp), fingerprint_size).

    Args:
        fp: A tensor of shape `(n_fps, fp_size / 32)`, containing packed fingerprints with dtype uint32 or int32.

    Returns:
        A boolean tensor of shape `(n_fps, fp_size)`
    """
    if fp.ndim != 2:
        raise ValueError(f"Input tensor must be 2D, got shape={tuple(fp.shape)}")
    if fp.dtype not in (torch.int32, torch.uint32):
        raise ValueError("Input tensor must have dtype int32 or uint32")
    fp = fp.contiguous()
    if fp.dtype == torch.int32:
        fp = fp.view(torch.uint32)

    return (
        ((fp.to(torch.int64).unsqueeze(2) >> torch.arange(32, device=fp.device, dtype=torch.int64)) & 1)
        .bool()
        .flatten(1)
    )


def pack_fingerprint(fp: torch.Tensor) -> torch.Tensor:
    """Pack a 2D boolean tensor of shape `(n_fps, fingerprint_size)` into a 32-bit integer-encoded fingerprint.

    Args:
        fp: A boolean tensor of shape `(n_fps, fp_size)`

    Returns:
        A tensor of shape `(n_fps, fp_size / 32)` containing packed fingerprints (rounded up to the nearest multiple of 32)
    """
    if fp.ndim != 2:
        raise ValueError(f"Input tensor must be 2D, got shape={tuple(fp.shape)}")
    if fp.dtype != torch.bool:
        raise ValueError("Input tensor must have dtype bool")

    n_fps, fp_size = fp.shape
    padding = -fp_size % 32
    n_ints = (fp_size + padding) // 32

    # Pad to next multiple of 32 if needed
    if padding:
        padded = torch.zeros((n_fps, fp_size + padding), dtype=torch.bool, device=fp.device)
        padded[:, :fp_size] = fp
        fp = padded

    # Reshape to group bits into 32-bit chunks
    fp_reshaped = fp.reshape(n_fps, n_ints, 32)

    # Create powers of 2 for each bit position, using 0 to 31 instead of 31 to 0 to fix endianness
    powers = 1 << torch.arange(32, device=fp.device, dtype=torch.int64)

    # Multiply and sum to create packed integers
    return (fp_reshaped * powers.unsqueeze(0)).sum(dim=2).to(torch.uint32)


class MorganFingerprintGenerator:
    """Morgan fingerprint generator."""

    def __init__(self, radius: int, fpSize: int):
        """Initialize the Morgan fingerprint generator.

        Args:
            radius: The radius of the Morgan fingerprint.
            fpSize: The size of the fingerprint. Must be one of {128, 256, 512, 1024, 2048}.
        """
        self._internal = InternalFPGen(radius, fpSize)

    def GetFingerprints(
        self, mols: list, num_threads: int = 0, stream: torch.cuda.Stream | None = None
    ) -> AsyncGpuResult:
        """Compute Morgan fingerprints for a list of molecules.

        Preprocessing of fingerprinting features is done on the CPU, and is parallelized with the `num_threads` argument.
        The resulting tensor has dtype torch.uint32 and contains a packed fingerprint for each molecule, one row per
        molecule.

        Packed fingerprints can be passed directly to nvMolKit similarity calculations, or unpacked
        via `unpack_fingerprint`.

        Args:
            mols: List of RDKit molecules to generate fingerprints for
            num_threads: Number of CPU threads to use for fingerprint generation. If 0, uses all available threads.
            stream: CUDA stream to use. If None, uses the current stream.

        Returns:
            AsyncGpuResult wrapping a torch.Tensor of shape (len(mols), fpSize / 32) containing the fingerprints.
            Each row is a fingerprint for the corresponding molecule.
        """
        active_stream = _resolve_cuda_stream(stream)
        return AsyncGpuResult(self._internal.GetFingerprintsDevice(mols, num_threads, active_stream.cuda_stream))
