# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
import triton
import triton.language as tl

# CUDA caps gridDim.y and gridDim.z at 65535, while gridDim.x allows up to 2**31-1.
# Tiles are linearized across a 2D launch grid with the bounded axis kept here.
MAX_GRID_DIM = 65535


def get_cuda_autotune_config():
    return [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32}, num_stages=5, num_warps=4),
    ]


@triton.jit
def _popcount32(x):
    x = x.to(tl.uint32)
    return tl.inline_asm_elementwise(
        asm="popc.b32 $0, $1;",
        constraints="=r,r",
        args=[x],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.int32)


def _check_fingerprint_matrix(name: str, x: torch.Tensor) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if x.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype int32")
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={tuple(x.shape)}")


def _check_bool_vector(
    name: str,
    x: torch.Tensor,
    expected_len: int,
) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if x.dtype != torch.bool:
        raise ValueError(f"{name} must have dtype bool")
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={tuple(x.shape)}")
    if x.numel() != expected_len:
        raise ValueError(f"{name} must have length {expected_len}, got {x.numel()}")


def _check_int32_vector(
    name: str,
    x: torch.Tensor,
    expected_len: int,
    *,
    allow_larger: bool = False,
) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if x.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype int32")
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={tuple(x.shape)}")
    if allow_larger:
        if x.numel() < expected_len:
            raise ValueError(f"{name} must have length >= {expected_len}, got {x.numel()}")
    else:
        if x.numel() != expected_len:
            raise ValueError(f"{name} must have length {expected_len}, got {x.numel()}")


# pyright: reportUnreachable=false
@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=["K"],
)
@triton.jit
def _update_neighbor_count_kernel(
    x_ptr,
    y_ptr,
    neighbors_ptr,
    M,
    N,
    K,
    x_stride_n,
    x_stride_k,
    y_stride_n,
    y_stride_k,
    threshold,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SUBTRACT: tl.constexpr,
    METRIC: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr = 8,
):
    """Compute pairwise similarity between blocks of x and y using bit-packed fingerprints.

    Atomically adds (SUBTRACT=False) or subtracts (SUBTRACT=True) the per-row
    neighbor counts into ``neighbors_ptr``.
    """
    pid = tl.program_id(axis=0).to(tl.int64) + tl.program_id(axis=1).to(tl.int64) * tl.num_programs(axis=0).to(
        tl.int64
    )
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    if pid >= num_pid_m.to(tl.int64) * num_pid_n.to(tl.int64):
        return
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    norm_x = tl.zeros((BLOCK_M,), dtype=tl.int32)
    norm_y = tl.zeros((BLOCK_N,), dtype=tl.int32)
    dots = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k_block * BLOCK_K
        for kk in tl.static_range(0, BLOCK_K):
            k_idx = k_offset + kk
            k_mask = k_idx < K
            xk = tl.load(
                x_ptr + offs_m * x_stride_n + k_idx * x_stride_k,
                mask=mask_m & k_mask,
                other=0,
            )
            yk = tl.load(
                y_ptr + offs_n * y_stride_n + k_idx * y_stride_k,
                mask=mask_n & k_mask,
                other=0,
            )
            norm_x += _popcount32(xk)
            norm_y += _popcount32(yk)
            dots += _popcount32(xk[:, None] & yk[None, :])

    if METRIC == "tanimoto":
        denom = norm_x[:, None] + norm_y[None, :] - dots
    elif METRIC == "cosine":
        denom = tl.sqrt(norm_x[:, None].to(tl.float32) * norm_y[None, :].to(tl.float32))
    else:
        raise ValueError(f"Invalid metric: {METRIC}")

    valid = mask_m[:, None] & mask_n[None, :] & (denom > 0)

    similarity = tl.where(valid, dots.to(tl.float32) / denom.to(tl.float32), 0.0)
    is_neighbor = valid & (similarity >= threshold)

    row_counts = tl.sum(is_neighbor.to(tl.int32), axis=1)
    if SUBTRACT:
        tl.atomic_add(neighbors_ptr + offs_m, -row_counts, mask=mask_m)
    else:
        tl.atomic_add(neighbors_ptr + offs_m, row_counts, mask=mask_m)


@triton.jit
def _extract_cluster_singleton_kernel(
    x_ptr,
    center_id,
    is_free_ptr,
    neighbors_ptr,
    cluster_count_ptr,
    cluster_indices_ptr,
    threshold,
    indices_ptr,
    M,
    K,
    x_stride_n,
    x_stride_k,
    BLOCK_K: tl.constexpr,
    METRIC: tl.constexpr,
):
    """For each free row, compute similarity to the cluster center.

    Neighbors (similarity >= threshold) are assigned to the cluster from the
    front of ``cluster_indices_ptr``; remaining rows whose neighbor degree is 1
    are collected as singletons from the back.
    """
    row = tl.program_id(axis=0)
    row_mask = row < M

    pa = tl.zeros((), dtype=tl.int32)
    pb = tl.zeros((), dtype=tl.int32)
    dot = tl.zeros((), dtype=tl.int32)

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k_block * BLOCK_K
        for kk in tl.static_range(0, BLOCK_K):
            k_idx = k_offset + kk
            k_mask = k_idx < K
            center_k = tl.load(x_ptr + center_id * x_stride_n + k_idx * x_stride_k, mask=k_mask, other=0)
            row_k = tl.load(
                x_ptr + row * x_stride_n + k_idx * x_stride_k,
                mask=row_mask & k_mask,
                other=0,
            )
            pa += _popcount32(center_k)
            pb += _popcount32(row_k)
            dot += _popcount32(row_k & center_k)

    if METRIC == "tanimoto":
        union = pa + pb - dot
    elif METRIC == "cosine":
        union = tl.sqrt(pa.to(tl.float32) * pb.to(tl.float32))
    else:
        raise ValueError(f"Invalid metric: {METRIC}")

    valid = row_mask & (union > 0)
    similarity = tl.where(valid, dot.to(tl.float32) / union.to(tl.float32), 0.0)
    is_neighbor = valid & (similarity >= threshold)

    orig_idx = tl.load(indices_ptr + row, mask=row_mask, other=0)
    neighbor_slot = tl.atomic_add(cluster_count_ptr + 0, 1, mask=is_neighbor)
    tl.store(cluster_indices_ptr + neighbor_slot, orig_idx, mask=is_neighbor)

    degree = tl.load(neighbors_ptr + row, mask=row_mask, other=0)
    is_singleton = row_mask & (~is_neighbor) & (degree == 1)
    singleton_slot = tl.atomic_add(cluster_count_ptr + 1, -1, mask=is_singleton)
    tl.store(cluster_indices_ptr + singleton_slot, orig_idx, mask=is_singleton)
    tl.store(is_free_ptr + row, False, mask=is_singleton | is_neighbor)


def update_neighbor_counts(
    x: torch.Tensor,
    y: torch.Tensor,
    neighbors: torch.Tensor,
    threshold: float,
    subtract: bool = False,
    metric: str = "tanimoto",
) -> None:
    """Update per-row neighbor counts for fingerprints in ``x`` against ``y``.

    For each row *i* in ``x``, counts how many rows in ``y`` have similarity
    >= ``threshold`` and atomically adds (or subtracts when ``subtract=True``)
    that count into ``neighbors[i]``.
    """
    _check_fingerprint_matrix("x", x)
    _check_fingerprint_matrix("y", y)
    _check_int32_vector("neighbors", neighbors, x.shape[0])
    if x.device != y.device or x.device != neighbors.device:
        raise ValueError("x, y, and neighbors must be on the same CUDA device")
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same feature dimension")

    M = x.shape[0]
    N = y.shape[0]
    K = x.shape[1]

    def grid(meta):
        num_tiles = triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"])
        grid_y = min(num_tiles, MAX_GRID_DIM)
        grid_x = triton.cdiv(num_tiles, grid_y)
        return (grid_x, grid_y)

    _update_neighbor_count_kernel[grid](
        x,
        y,
        neighbors,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        float(threshold),
        SUBTRACT=subtract,
        METRIC=metric,
    )


def extract_cluster_and_singletons(
    x: torch.Tensor,
    id: int,
    is_free: torch.Tensor,
    neighbors: torch.Tensor,
    cluster_count: torch.Tensor,
    cluster_indices: torch.Tensor,
    threshold: float,
    indices: torch.Tensor,
    metric: str = "tanimoto",
) -> None:
    """Extract the cluster around center ``id`` and collect singletons.

    Every free row similar to the center (>= ``threshold``) is written into
    ``cluster_indices`` from the front; free rows that are not neighbors but
    have a neighbor degree of 1 are collected as singletons from the back.
    Both groups are marked as non-free in ``is_free``.
    """
    _check_fingerprint_matrix("x", x)
    M = x.shape[0]
    K = x.shape[1]
    _check_bool_vector("is_free", is_free, M)
    _check_int32_vector("neighbors", neighbors, M)
    _check_int32_vector("cluster_indices", cluster_indices, M, allow_larger=True)
    _check_int32_vector("indices", indices, M)
    _check_int32_vector("cluster_count", cluster_count, 2)
    if not (0 <= id < M):
        raise ValueError(f"id must be in [0, {M}), got {id}")
    if (
        x.device != is_free.device
        or x.device != neighbors.device
        or x.device != cluster_count.device
        or x.device != cluster_indices.device
        or x.device != indices.device
    ):
        raise ValueError("all tensors must be on the same CUDA device")

    grid = (M,)
    _extract_cluster_singleton_kernel[grid](
        x,
        id,
        is_free,
        neighbors,
        cluster_count,
        cluster_indices,
        float(threshold),
        indices,
        M,
        K,
        x.stride(0),
        x.stride(1),
        BLOCK_K=32,
        num_warps=1,
        METRIC=metric,
    )
