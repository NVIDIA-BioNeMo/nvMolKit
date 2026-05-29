# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys

import pyperf
import torch
from bench_utils import load_smiles
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import BulkCosineSimilarity, BulkTanimotoSimilarity

from nvmolkit.fingerprints import MorganFingerprintGenerator
from nvmolkit.similarity import crossCosineSimilarity, crossTanimotoSimilarity


SIZES = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 20000, 24000, 28000, 32000]
CPU_SINGLE_VALUE_ABOVE = 6000


def rdkit_sim(fps, sim_type):
    if sim_type.lower() == "tanimoto":
        [BulkTanimotoSimilarity(fps[i], fps) for i in range(len(fps))]
    elif sim_type.lower() == "cosine":
        [BulkCosineSimilarity(fps[i], fps) for i in range(len(fps))]


def nvmolkit_sim_gpu_only(fps, sim_type):
    if sim_type.lower() == "tanimoto":
        crossTanimotoSimilarity(fps)
    elif sim_type.lower() == "cosine":
        crossCosineSimilarity(fps)
    torch.cuda.synchronize()


# --no-rdkit / --no-nvmolkit gate the module-level fingerprint setup below,
# which runs at import time before pyperf's Runner parses args. Read them
# directly from argv and strip them so pyperf's argparser doesn't reject the
# unknown flags.
NO_RDKIT = "--no-rdkit" in sys.argv
if NO_RDKIT:
    sys.argv = [a for a in sys.argv if a != "--no-rdkit"]
NO_NVMOLKIT = "--no-nvmolkit" in sys.argv
if NO_NVMOLKIT:
    sys.argv = [a for a in sys.argv if a != "--no-nvmolkit"]
if NO_RDKIT and NO_NVMOLKIT:
    raise SystemExit("cross_similarity_bench: cannot pass both --no-rdkit and --no-nvmolkit")

runner = pyperf.Runner(min_time=0.01, values=3, processes=1, loops=3)
runner.metadata["description"] = "Cross Similarity benchmark"
runner.argparser.add_argument(
    "--input", type=str, default="data/benchmark_smiles.csv", help="Path to input SMILES file (.smi/.csv/.cxsmiles)"
)
runner.argparser.add_argument("--cosine", action="store_true", help="Include cosine similarity benchmarks")
runner.argparser.add_argument("--seed", type=int, default=42, help="Random seed for sampling SMILES (default: 42)")
args = runner.parse_args()

sim_types = ("tanimoto", "cosine") if args.cosine else ("tanimoto",)
fpsize = 1024
max_size = max(SIZES)
default_values = runner.args.values

mols = load_smiles(args.input, max_count=max_size, seed=args.seed)
if not mols:
    raise ValueError(f"No molecules parsed from {args.input}")
while len(mols) < max_size:
    mols += mols
mols = mols[:max_size]

if not NO_RDKIT:
    rdkit_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=fpsize)
    rdkit_fps_all = [rdkit_fpgen.GetFingerprint(mol) for mol in mols]
if not NO_NVMOLKIT:
    nvmolkit_fpgen = MorganFingerprintGenerator(radius=3, fpSize=fpsize)
    nvmolkit_fps_all = torch.as_tensor(nvmolkit_fpgen.GetFingerprints(mols), device="cuda")

for sim_type in sim_types:
    for molNum in SIZES:
        if not NO_RDKIT:
            fps = rdkit_fps_all[:molNum]
            runner.args.values = 1 if molNum > CPU_SINGLE_VALUE_ABOVE else default_values
            name = f"rdkit_{sim_type}sim_fpsize_{fpsize}_{molNum}mols"
            runner.bench_func(name, rdkit_sim, fps, sim_type, metadata={"name": name})

        if not NO_NVMOLKIT:
            nvmolkit_fps_cu = nvmolkit_fps_all[:molNum].contiguous()
            runner.args.values = default_values
            name2 = f"nvmolkit_gpu-only_{sim_type}sim_fpsize_{fpsize}_{molNum}mols_gpu_only"
            runner.bench_func(name2, nvmolkit_sim_gpu_only, nvmolkit_fps_cu, sim_type, metadata={"name": name2})
