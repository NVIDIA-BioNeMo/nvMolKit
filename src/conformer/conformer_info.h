// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVMOLKIT_CONFORMER_INFO_H
#define NVMOLKIT_CONFORMER_INFO_H

#include <cstddef>

namespace RDKit {
class ROMol;
class Conformer;
}  // namespace RDKit

namespace nvMolKit {

struct ConformerInfo {
  RDKit::ROMol*     mol;
  size_t            molIdx;
  RDKit::Conformer* conformer;
  int               conformerId;
  size_t            confIdx;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_CONFORMER_INFO_H
