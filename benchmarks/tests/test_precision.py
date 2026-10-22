# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precision option utilities."""

import argparse

import pytest
from bench_utils import precision

from nvmolkit.types import PrecisionMode


def test_add_precision_arg_defaults_to_full():
    parser = argparse.ArgumentParser()
    precision.add_precision_arg(parser)

    assert parser.parse_args([]).precision == "full"
    assert parser.parse_args(["--precision", "single"]).precision == "single"
    with pytest.raises(SystemExit):
        parser.parse_args(["--precision", "half"])


def test_precision_mode_maps_names():
    assert precision.precision_mode("full") == PrecisionMode.FULL
    assert precision.precision_mode("single") == PrecisionMode.SINGLE
