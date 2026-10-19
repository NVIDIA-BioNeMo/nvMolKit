# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precision selection and search-space settings shared by autotuned benchmarks."""

import argparse
import importlib
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial

PRECISION_CHOICES = ("full", "single")

# Search spaces used for publication runs. The library defaults cap the batch size at 1024 for
# force fields and ETKDG; larger batches help on data-center GPUs, so the benchmarks widen the
# range explicitly rather than depending on the installed release's defaults.
FF_SEARCH_SPACE_OVERRIDES = {"batchSize": (64, 4096, 64)}
EMBED_SEARCH_SPACE_OVERRIDES = {"batchSize": (64, 4096, 64)}
MCS_SEARCH_SPACE_OVERRIDES = {"batchSize": [128, 256, 512, 1024, 2048, 4096, 8192]}


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


@contextmanager
def autotune_at_precision(autotune_module: str, function_name: str, precision: str) -> Iterator[None]:
    """Make an autotune wrapper time its trials at ``precision``.

    This benchmark-only context assumes sequential tuning calls in a process.

    The ``nvmolkit.autotune.tune_*`` wrappers call the underlying optimize/embed function with its
    default (full) precision and expose no precision argument, so tuning a single-precision run
    would otherwise pick batch sizes for double precision. Inside this context the wrapper's
    module-level reference to that function is bound to the requested precision; the original is
    restored on exit.

    Args:
        autotune_module: Module that owns the reference, e.g. ``"nvmolkit.autotune.tune_mmff_optimize"``.
        function_name: Name of the function it calls, e.g. ``"MMFFOptimizeMoleculesConfs"``.
        precision: ``"full"`` or ``"single"``.

    Raises:
        AttributeError: if the module no longer references ``function_name``. Failing is deliberate:
            silently tuning at the wrong precision would invalidate the run.
    """
    module = importlib.import_module(autotune_module)
    original = getattr(module, function_name)
    setattr(module, function_name, partial(original, precision=precision_mode(precision)))
    try:
        yield
    finally:
        setattr(module, function_name, original)
