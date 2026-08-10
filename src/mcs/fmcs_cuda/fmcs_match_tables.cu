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

#include <algorithm>
#include <stdexcept>

#include "src/mcs/fmcs_cuda/fmcs_match_tables.cuh"

namespace mcs {
namespace fmcs {

size_t computePairMatchTableWords(const std::vector<PairMatchTablesHost>& host) {
  size_t totalWords = 0;
  for (const auto& p : host) {
    totalWords += p.atoms.data.size();
    totalWords += p.bonds.data.size();
  }
  return totalWords;
}

UploadedPairMatchTables uploadPairMatchTables(const std::vector<PairMatchTablesHost>& host,
                                              std::span<uint32_t>                     packedHost,
                                              cudaStream_t                            stream) {
  const size_t totalWords = computePairMatchTableWords(host);
  if (packedHost.size() < totalWords) {
    throw std::out_of_range("Pinned match-table staging buffer is too small");
  }

  UploadedPairMatchTables out;
  out.storage = nvMolKit::AsyncDeviceVector<uint32_t>(totalWords, stream);
  out.tables.resize(host.size());

  uint32_t* base   = out.storage.data();
  size_t    cursor = 0;
  for (size_t i = 0; i < host.size(); ++i) {
    const auto& ph = host[i];
    auto&       pd = out.tables[i];

    if (!ph.atoms.data.empty()) {
      std::copy(ph.atoms.data.begin(), ph.atoms.data.end(), packedHost.begin() + cursor);
      pd.atoms = {base + cursor, ph.atoms.nRows, ph.atoms.nCols, ph.atoms.wordsPerRow};
      cursor += ph.atoms.data.size();
    }

    if (!ph.bonds.data.empty()) {
      std::copy(ph.bonds.data.begin(), ph.bonds.data.end(), packedHost.begin() + cursor);
      pd.bonds = {base + cursor, ph.bonds.nRows, ph.bonds.nCols, ph.bonds.wordsPerRow};
      cursor += ph.bonds.data.size();
    }
  }

  if (totalWords > 0) {
    out.storage.copyFromHost(packedHost.data(), totalWords);
  }
  return out;
}

}  // namespace fmcs
}  // namespace mcs
