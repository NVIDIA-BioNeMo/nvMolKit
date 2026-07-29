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

"""Shared command-line argument helpers for nvMolKit benchmarks."""

import argparse


def add_smiles_sanitization_args(parser: argparse.ArgumentParser) -> None:
    """Add safe-by-default SMILES sanitization flags to ``parser``."""
    sanitize_group = parser.add_mutually_exclusive_group()
    sanitize_group.add_argument(
        "--sanitize",
        action="store_true",
        dest="sanitize",
        default=True,
        help="Sanitize SMILES during parsing (default)",
    )
    sanitize_group.add_argument(
        "--no_sanitize",
        action="store_false",
        dest="sanitize",
        help="Skip sanitization (preprocessed SMILES only)",
    )
