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

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>

#include "fmcs_cuda/fmcs_match_tables.cuh"

namespace mcs {
namespace fmcs {

namespace {

void checkCuda(cudaError_t err, const char* context) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("fMCS match-tables CUDA error at ") + context + ": " +
                             cudaGetErrorString(err));
  }
}

}  // namespace

std::vector<PairMatchTablesDevice> uploadPairMatchTables(const std::vector<PairMatchTablesHost>& host,
                                                         cudaStream_t                            stream,
                                                         void**                                  outDeviceBuffer,
                                                         size_t*                                 outDeviceBufferBytes) {
  std::vector<PairMatchTablesDevice> out(host.size());

  size_t totalWords = 0;
  for (const auto& p : host) {
    totalWords += p.atoms.data.size();
    totalWords += p.bonds.data.size();
  }

  void* buf = nullptr;
  if (totalWords > 0) {
    checkCuda(cudaMallocAsync(&buf, totalWords * sizeof(uint32_t), stream), "cudaMallocAsync (match tables)");
  }

  size_t    cursor = 0;
  uint32_t* base   = reinterpret_cast<uint32_t*>(buf);
  for (size_t i = 0; i < host.size(); ++i) {
    const auto& ph = host[i];
    auto&       pd = out[i];

    if (!ph.atoms.data.empty()) {
      uint32_t* dst = base + cursor;
      checkCuda(cudaMemcpyAsync(dst,
                                ph.atoms.data.data(),
                                ph.atoms.data.size() * sizeof(uint32_t),
                                cudaMemcpyHostToDevice,
                                stream),
                "cudaMemcpyAsync (atoms)");
      pd.atoms = {dst, ph.atoms.nRows, ph.atoms.nCols, ph.atoms.wordsPerRow};
      cursor += ph.atoms.data.size();
    }

    if (!ph.bonds.data.empty()) {
      uint32_t* dst = base + cursor;
      checkCuda(cudaMemcpyAsync(dst,
                                ph.bonds.data.data(),
                                ph.bonds.data.size() * sizeof(uint32_t),
                                cudaMemcpyHostToDevice,
                                stream),
                "cudaMemcpyAsync (bonds)");
      pd.bonds = {dst, ph.bonds.nRows, ph.bonds.nCols, ph.bonds.wordsPerRow};
      cursor += ph.bonds.data.size();
    }
  }

  if (outDeviceBuffer)
    *outDeviceBuffer = buf;
  if (outDeviceBufferBytes)
    *outDeviceBufferBytes = totalWords * sizeof(uint32_t);
  return out;
}

void freePairMatchTablesBuffer(void* deviceBuffer, cudaStream_t stream) {
  if (deviceBuffer) {
    cudaFreeAsync(deviceBuffer, stream);
  }
}

}  // namespace fmcs
}  // namespace mcs
