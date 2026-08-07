# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common CLI arguments for nvMolKit benchmark drivers."""

import argparse
from collections.abc import Sequence


def add_backend_selection_args(
    parser: argparse.ArgumentParser,
    *,
    rdkit_dest: str = "no_rdkit",
    nvmolkit_dest: str = "no_nvmolkit",
    rdkit_aliases: Sequence[str] = (),
    nvmolkit_aliases: Sequence[str] = (),
) -> None:
    """Add compatible hyphenated/underscored backend-disable flags."""
    parser.add_argument(
        "--no-rdkit",
        "--no_rdkit",
        *rdkit_aliases,
        dest=rdkit_dest,
        action="store_true",
        help="Skip the RDKit benchmark",
    )
    parser.add_argument(
        "--no-nvmolkit",
        "--no_nvmolkit",
        *nvmolkit_aliases,
        dest=nvmolkit_dest,
        action="store_true",
        help="Skip the nvMolKit benchmark",
    )
