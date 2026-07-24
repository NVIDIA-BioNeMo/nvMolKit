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

#ifndef FMCS_CUDA_FMCS_MATCH_TABLES_CUH
#define FMCS_CUDA_FMCS_MATCH_TABLES_CUH

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/mcs/mcs_common/mcs_types.cuh"
#include "src/utils/device_vector.h"

namespace mcs {
namespace fmcs {

/// Packed query-vs-target compatibility bitmap: row @c i covers query item
/// @c i, column @c j covers target item @c j.  Row stride is @c wordsPerRow
/// uint32 words.  Used for both atoms and bonds so every per-seed
/// compatibility check on the device is a single word load.
struct MatchTableHost {
  int                   nRows       = 0;
  int                   nCols       = 0;
  int                   wordsPerRow = 0;
  std::vector<uint32_t> data;

  static int computeWordsPerRow(int nCols) { return (nCols + 31) / 32; }

  void resize(int rows, int cols) {
    nRows       = rows;
    nCols       = cols;
    wordsPerRow = computeWordsPerRow(cols);
    data.assign(static_cast<size_t>(rows) * wordsPerRow, 0u);
  }

  void setBit(int row, int col) { data[static_cast<size_t>(row) * wordsPerRow + col / 32] |= (1u << (col % 32)); }

  bool testBit(int row, int col) const {
    return (data[static_cast<size_t>(row) * wordsPerRow + col / 32] >> (col % 32)) & 1u;
  }
};

struct MatchTableDevice {
  const uint32_t* data        = nullptr;
  int             nRows       = 0;
  int             nCols       = 0;
  int             wordsPerRow = 0;

  __host__ __device__ __forceinline__ bool testBit(int row, int col) const {
    return (data[static_cast<size_t>(row) * wordsPerRow + col / 32] >> (col % 32)) & 1u;
  }
};

struct PairMatchTablesHost {
  MatchTableHost atoms;
  MatchTableHost bonds;
};

struct PairMatchTablesDevice {
  MatchTableDevice atoms;
  MatchTableDevice bonds;
};

struct UploadedPairMatchTables {
  nvMolKit::AsyncDeviceVector<uint32_t> storage;
  std::vector<PairMatchTablesDevice>    tables;
};

/// Upload a batch of per-pair host match tables into one contiguous device
/// allocation.  The returned object owns that allocation and keeps the
/// per-pair device views valid for its lifetime.
UploadedPairMatchTables uploadPairMatchTables(const std::vector<PairMatchTablesHost>& host, cudaStream_t stream);

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_MATCH_TABLES_CUH
