# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared preparation for APIs that consume packed fingerprint matrices."""

import torch

from nvmolkit.types import ArrayInput, _as_cuda_tensor, _resolve_cuda_stream


def _prepare_packed_fingerprints(
    *named_inputs: tuple[str, ArrayInput],
    stream: torch.cuda.Stream | None,
) -> tuple[tuple[torch.Tensor, ...], torch.cuda.Stream]:
    """Move packed fingerprints to one CUDA stream and validate their common layout."""
    if not named_inputs:
        raise ValueError("At least one packed fingerprint input is required")

    active_stream = _resolve_cuda_stream(stream, *(value for _, value in named_inputs))
    prepared = []
    with torch.cuda.stream(active_stream):
        for name, value in named_inputs:
            tensor = _as_cuda_tensor(name, value, stream=active_stream)
            if tensor.ndim != 2:
                raise ValueError(f"{name} must be 2D, got shape={tuple(tensor.shape)}")
            if tensor.dtype not in (torch.int32, torch.uint32):
                raise ValueError(f"{name} must have dtype int32 or uint32")
            if tensor.shape[1] == 0:
                raise ValueError(f"{name} must contain at least one fingerprint word")
            tensor = tensor.contiguous()
            if tensor.dtype == torch.int32:
                tensor = tensor.view(torch.uint32)
            prepared.append(tensor)

    return tuple(prepared), active_stream
