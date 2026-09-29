# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared preparation of distance-matrix and fused-metric inputs for clustering and pickers."""

import torch

from nvmolkit.types import ArrayInput, _as_cuda_tensor, _resolve_cuda_stream


def _prepare_distance_matrix(distance_matrix: ArrayInput, stream: torch.cuda.Stream | None):
    active_stream = _resolve_cuda_stream(stream, distance_matrix)
    with torch.cuda.stream(active_stream):
        tensor = _as_cuda_tensor("distance_matrix", distance_matrix, stream=active_stream)
        if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
            raise ValueError(f"distance_matrix must be a square 2D matrix, got shape={tuple(tensor.shape)}")
        if tensor.dtype not in (torch.float32, torch.float64):
            raise ValueError("distance_matrix must have dtype float32 or float64")
        tensor = tensor.contiguous()
    return tensor, active_stream
