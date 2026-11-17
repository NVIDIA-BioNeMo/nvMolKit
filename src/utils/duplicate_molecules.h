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

#ifndef NVMOLKIT_DUPLICATE_MOLECULES_H
#define NVMOLKIT_DUPLICATE_MOLECULES_H

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace nvMolKit {

//! Throws std::invalid_argument if the same molecule object is listed more than once.
//! \param remedy Appended to the message; says what the caller should do instead.
template <typename MolPtr> void throwIfDuplicateMolecules(const std::vector<MolPtr>& mols, const std::string& remedy) {
  std::unordered_map<MolPtr, size_t> firstIndex;
  firstIndex.reserve(mols.size());
  for (size_t i = 0; i < mols.size(); ++i) {
    const auto [it, inserted] = firstIndex.try_emplace(mols[i], i);
    if (!inserted) {
      throw std::invalid_argument("The same molecule object is listed at indices " + std::to_string(it->second) +
                                  " and " + std::to_string(i) + ". " + remedy);
    }
  }
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_DUPLICATE_MOLECULES_H
