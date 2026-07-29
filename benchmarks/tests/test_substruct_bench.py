# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the substructure benchmark CLI."""

import argparse

import pytest
from bench_utils.cli import add_smiles_sanitization_args

def test_smiles_sanitization_is_enabled_by_default():
    parser = argparse.ArgumentParser()
    add_smiles_sanitization_args(parser)

    assert parser.parse_args([]).sanitize is True
    assert parser.parse_args(["--sanitize"]).sanitize is True
    assert parser.parse_args(["--no_sanitize"]).sanitize is False


def test_smiles_sanitization_flags_are_mutually_exclusive():
    parser = argparse.ArgumentParser()
    add_smiles_sanitization_args(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--sanitize", "--no_sanitize"])
