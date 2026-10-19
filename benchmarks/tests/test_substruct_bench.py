# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from substruct_bench import _auto_preprocessing_threads


def test_auto_preprocessing_threads_reserves_explicit_workers_for_every_gpu():
    assert _auto_preprocessing_threads(worker_threads=8, num_gpus=8, physical_cores=128) == 64


def test_auto_preprocessing_threads_reserves_auto_workers_for_every_gpu():
    assert _auto_preprocessing_threads(worker_threads=-1, num_gpus=8, physical_cores=128) == 96


def test_auto_preprocessing_threads_keeps_one_thread_when_workers_exhaust_budget():
    assert _auto_preprocessing_threads(worker_threads=8, num_gpus=8, physical_cores=32) == 1
