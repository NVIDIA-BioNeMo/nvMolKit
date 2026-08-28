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

"""Import an installed wheel and run a small Morgan fingerprint kernel."""

from __future__ import annotations

import sys

from rdkit import Chem
import torch

import nvmolkit
from nvmolkit.fingerprints import MorganFingerprintGenerator


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False", file=sys.stderr)
        return 2

    device_name = torch.cuda.get_device_name(0)
    nvmolkit_version = getattr(nvmolkit, "__version__", "unknown")
    rdkit_version_str = Chem.rdBase.rdkitVersion

    mol = Chem.MolFromSmiles("CCO")
    if mol is None:
        print("FAIL: rdkit could not parse 'CCO'", file=sys.stderr)
        return 3

    gen = MorganFingerprintGenerator(radius=2, fpSize=2048)
    handle = gen.GetFingerprints([mol])
    torch.cuda.synchronize()
    tensor = handle.torch()

    expected_shape = (1, 2048 // 32)
    if tuple(tensor.shape) != expected_shape:
        print(
            f"FAIL: fingerprint shape {tuple(tensor.shape)} != expected {expected_shape}",
            file=sys.stderr,
        )
        return 4

    nonzero_bits = int((tensor != 0).sum().item())
    if nonzero_bits == 0:
        print("FAIL: all-zero fingerprint for ethanol", file=sys.stderr)
        return 5

    print(
        f"OK nvmolkit={nvmolkit_version} rdkit={rdkit_version_str} "
        f"device={device_name} fp_shape={tuple(tensor.shape)} "
        f"nonzero_ints={nonzero_bits}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
