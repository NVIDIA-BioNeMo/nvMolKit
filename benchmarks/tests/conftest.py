# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest setup for benchmark unit tests.

Bench scripts import the sibling ``bench_utils`` package without any
``sys.path`` manipulation because they run with ``benchmarks/`` as the
working directory. The same trick is needed for these tests, which live
one level deeper.
"""

import sys
from pathlib import Path

_BENCHMARKS_DIR = Path(__file__).resolve().parent.parent
if str(_BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_DIR))
