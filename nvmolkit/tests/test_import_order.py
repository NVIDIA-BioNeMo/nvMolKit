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

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "nvmolkit.mmffOptimization",
        "nvmolkit.uffOptimization",
    ],
)
def test_module_imports_as_first_nvmolkit_import(module_name):
    """Importing an optimization module first must not require nvmolkit.types to be imported first.

    The native extensions declare BatchHardwareOptions default arguments whose to-Python converter
    is registered by nvmolkit._embedMolecules. Each import runs in a fresh interpreter so the
    converter is not already registered by an earlier import in this test session.
    """
    result = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Importing {module_name} as the first nvMolKit import failed:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
