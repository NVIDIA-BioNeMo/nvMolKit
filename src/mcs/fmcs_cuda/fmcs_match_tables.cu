// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fmcs_cuda/fmcs_match_tables.cuh"

namespace mcs {
namespace fmcs {

UploadedPairMatchTables uploadPairMatchTables(const std::vector<PairMatchTablesHost>& host, cudaStream_t stream) {
  size_t totalWords = 0;
  for (const auto& p : host) {
    totalWords += p.atoms.data.size();
    totalWords += p.bonds.data.size();
  }

  UploadedPairMatchTables out;
  out.storage = nvMolKit::AsyncDeviceVector<uint32_t>(totalWords, stream);
  out.tables.resize(host.size());

  std::vector<uint32_t> packed;
  packed.reserve(totalWords);
  uint32_t* base = out.storage.data();
  for (size_t i = 0; i < host.size(); ++i) {
    const auto& ph = host[i];
    auto&       pd = out.tables[i];

    if (!ph.atoms.data.empty()) {
      const size_t offset = packed.size();
      packed.insert(packed.end(), ph.atoms.data.begin(), ph.atoms.data.end());
      pd.atoms = {base + offset, ph.atoms.nRows, ph.atoms.nCols, ph.atoms.wordsPerRow};
    }

    if (!ph.bonds.data.empty()) {
      const size_t offset = packed.size();
      packed.insert(packed.end(), ph.bonds.data.begin(), ph.bonds.data.end());
      pd.bonds = {base + offset, ph.bonds.nRows, ph.bonds.nCols, ph.bonds.wordsPerRow};
    }
  }

  if (!packed.empty()) {
    out.storage.copyFromHost(packed);
  }
  return out;
}

}  // namespace fmcs
}  // namespace mcs
