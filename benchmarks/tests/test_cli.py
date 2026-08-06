# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest
from bench_utils.cli import add_backend_selection_args


@pytest.mark.parametrize("flag", ["--no-rdkit", "--no_rdkit", "--skip-rdkit"])
def test_backend_selection_accepts_common_and_legacy_aliases(flag):
    parser = argparse.ArgumentParser()
    add_backend_selection_args(parser, rdkit_dest="skip_rdkit", rdkit_aliases=("--skip-rdkit",))
    args = parser.parse_args([flag])
    assert args.skip_rdkit
    assert not args.no_nvmolkit


@pytest.mark.parametrize("flag", ["--no-nvmolkit", "--no_nvmolkit"])
def test_backend_selection_accepts_nvmolkit_spellings(flag):
    parser = argparse.ArgumentParser()
    add_backend_selection_args(parser)
    args = parser.parse_args([flag])
    assert args.no_nvmolkit
    assert not args.no_rdkit
