# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precision selection for benchmark collectors."""

import argparse

PRECISION_CHOICES = ("full", "single")


def add_precision_arg(parser: argparse.ArgumentParser) -> None:
    """Add ``--precision`` selecting nvMolKit's device precision mode."""
    parser.add_argument(
        "--precision",
        choices=PRECISION_CHOICES,
        default="full",
        help="nvMolKit precision mode: 'full' (double, the library default) or 'single' (default: full)",
    )


def precision_mode(name: str):
    """Return the ``nvmolkit.types.PrecisionMode`` for a ``--precision`` value."""
    from nvmolkit.types import PrecisionMode

    return {"full": PrecisionMode.FULL, "single": PrecisionMode.SINGLE}[name]
