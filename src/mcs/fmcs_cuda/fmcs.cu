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

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "src/mcs/fmcs_cuda/fmcs.cuh"
#include "src/mcs/fmcs_cuda/fmcs_kernel.cuh"
#include "src/mcs/fmcs_cuda/fmcs_match_tables.cuh"
#include "src/mcs/fmcs_cuda/fmcs_policy.cuh"
#include "src/utils/device.h"
#include "src/utils/device_vector.h"
#include "src/utils/gpu_executor_ring.h"
#include "src/utils/pinned_host_allocator.h"

namespace mcs {
namespace fmcs {

namespace {

void checkCuda(cudaError_t err, const char* context) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("fMCS dispatch CUDA error at ") + context + ": " + cudaGetErrorString(err));
  }
}

/// uint32 host-side mirror of a CSR @ref Graph plus packed bond metadata
/// matching @ref enumerateBonds ordering.
struct PackedGraphHost {
  std::vector<uint32_t> rowOffsets;
  std::vector<uint32_t> colIndices;
  /// Per-adjacency entry bond id, parallel to @c colIndices.
  std::vector<uint32_t> bondIndices;
  std::vector<uint32_t> bondEndpoints;  // (u << 16) | v, u < v
};

PackedGraphHost packGraph(const Graph& g) {
  constexpr uint32_t kUnsetBondIndex = 0xFFFFFFFFu;
  PackedGraphHost    out;
  out.rowOffsets.reserve(g.rowOffsets.size());
  for (size_t v : g.rowOffsets)
    out.rowOffsets.push_back(static_cast<uint32_t>(v));
  out.colIndices.reserve(g.colIndices.size());
  for (size_t v : g.colIndices)
    out.colIndices.push_back(static_cast<uint32_t>(v));
  out.bondIndices.assign(g.colIndices.size(), kUnsetBondIndex);
  out.bondEndpoints.reserve(static_cast<size_t>(g.numEdges));
  for (int u = 0; u < g.numVertices; ++u) {
    const size_t begin  = g.rowOffsets[u];
    const size_t end    = g.rowOffsets[u + 1];
    const size_t degree = end - begin;
    if (degree > static_cast<size_t>(kMaxNeighborsPerAtom)) {
      throw std::invalid_argument("fMCS graph vertex " + std::to_string(u) + " has degree " + std::to_string(degree) +
                                  "; maximum supported degree is " + std::to_string(kMaxNeighborsPerAtom));
    }
    for (size_t k = begin; k < end; ++k) {
      const int v = static_cast<int>(g.colIndices[k]);
      if (u < v) {
        const auto bondIdx = static_cast<uint32_t>(out.bondEndpoints.size());
        out.bondEndpoints.push_back((static_cast<uint32_t>(u) << 16) | static_cast<uint32_t>(v));
        out.bondIndices[k] = bondIdx;

        bool foundReverse = false;
        for (size_t rk = g.rowOffsets[v]; rk < g.rowOffsets[v + 1]; ++rk) {
          if (static_cast<int>(g.colIndices[rk]) == u && out.bondIndices[rk] == kUnsetBondIndex) {
            out.bondIndices[rk] = bondIdx;
            foundReverse        = true;
            break;
          }
        }
        if (!foundReverse) {
          throw std::runtime_error("fMCS CSR graph is missing reverse edge");
        }
      }
    }
  }
  if (out.bondEndpoints.size() != static_cast<size_t>(g.numEdges)) {
    throw std::runtime_error("fMCS CSR graph edge count is inconsistent");
  }
  return out;
}

/// Host-side descriptor assembled per pair.  @c queryGraph points at the
/// smaller input (fMCS enumerates subgraphs of the query), and
/// @c swapped records whether sides were flipped so host-side expansion
/// can un-swap the result mappings.
struct HostPairDescriptor {
  bool swapped    = false;
  bool overflowed = false;

  const Graph* queryGraph  = nullptr;
  const Graph* targetGraph = nullptr;

  PackedGraphHost     packedQuery;
  PackedGraphHost     packedTarget;
  PairMatchTablesHost tables;

  /// Maps to @ref pickMaxSizeTier: 0=16, 1=32, 2=64, 3=128; -1 on overflow.
  int tier = -1;
};

template <class Policy, class InputT>
HostPairDescriptor buildPairDescriptor(const InputT& sideA, const InputT& sideB, const Parameters& params) {
  HostPairDescriptor desc;
  const Graph&       gA = [&]() -> const Graph& {
    if constexpr (std::is_same_v<InputT, Graph>) {
      return sideA;
    } else {
      return sideA.graph;
    }
  }();
  const Graph& gB = [&]() -> const Graph& {
    if constexpr (std::is_same_v<InputT, Graph>) {
      return sideB;
    } else {
      return sideB.graph;
    }
  }();

  const bool pickA = gA.numEdges <= gB.numEdges;
  desc.swapped     = !pickA;
  const InputT& q  = pickA ? sideA : sideB;
  const InputT& t  = pickA ? sideB : sideA;
  const Graph&  qG = pickA ? gA : gB;
  const Graph&  tG = pickA ? gB : gA;

  desc.queryGraph   = &qG;
  desc.targetGraph  = &tG;
  desc.packedQuery  = packGraph(qG);
  desc.packedTarget = packGraph(tG);

  Policy::buildAtomMatchTable(q, t, desc.tables.atoms, params.matchVertexLabels);
  Policy::buildBondMatchTable(q, t, desc.tables.bonds, params.matchEdgeLabels);

  const int tier  = pickMaxSizeTier(std::max(qG.numVertices, tG.numVertices), std::max(qG.numEdges, tG.numEdges));
  desc.tier       = tier;
  desc.overflowed = (tier < 0);
  return desc;
}

/// Owning device allocations for one tier sub-batch. Results, queue, and
/// scratch element types depend on the selected tier, so their storage is
/// type-erased as bytes. The queue storage is one contiguous global-memory slab of
/// @c kFmcsQueueCapacity * numPairs QueuedSeed entries; per-block slices
/// are taken inside the kernel via @c blockIdx.x.
struct BatchDeviceBuffers {
  nvMolKit::AsyncDeviceVector<uint32_t>           csrStorage;
  nvMolKit::AsyncDeviceVector<DevicePerPairInput> pairInputs;
  nvMolKit::AsyncDeviceVector<std::byte>          resultsStorage;
  nvMolKit::AsyncDeviceVector<std::byte>          queueStorage;
  nvMolKit::AsyncDeviceVector<std::byte>          scratchStorage;
};

constexpr int    kMaxFmcsExecutorsPerRunner = 8;
constexpr size_t kPinnedHostAlignment       = 256;

struct PinnedHostRequirements {
  size_t matchTableWords = 0;
  size_t csrWords        = 0;
  size_t pairInputCount  = 0;
  size_t resultBytes     = 0;
};

struct DeviceRequirements {
  size_t matchTableWords = 0;
  size_t csrWords        = 0;
  size_t pairInputCount  = 0;
  size_t resultBytes     = 0;
  size_t queueBytes      = 0;
  size_t scratchBytes    = 0;
};

struct FmcsPinnedHostBuffer {
  nvMolKit::PinnedHostView<uint32_t>           matchTables;
  nvMolKit::PinnedHostView<uint32_t>           csr;
  nvMolKit::PinnedHostView<DevicePerPairInput> pairInputs;
  nvMolKit::PinnedHostView<std::byte>          results;
};

size_t alignPinnedOffset(const size_t offset) {
  return (offset + kPinnedHostAlignment - 1) & ~(kPinnedHostAlignment - 1);
}

FmcsPinnedHostBuffer createPinnedHostBuffer(const PinnedHostRequirements& requirements) {
  size_t totalBytes = 0;
  auto   addBlock   = [&](const size_t bytes) {
    if (bytes > 0) {
      totalBytes = alignPinnedOffset(totalBytes);
      totalBytes += bytes;
    }
  };
  addBlock(requirements.matchTableWords * sizeof(uint32_t));
  addBlock(requirements.csrWords * sizeof(uint32_t));
  addBlock(requirements.pairInputCount * sizeof(DevicePerPairInput));
  addBlock(requirements.resultBytes);

  nvMolKit::PinnedHostAllocator allocator(totalBytes);
  FmcsPinnedHostBuffer          buffer;
  if (requirements.matchTableWords > 0) {
    buffer.matchTables = allocator.allocate<uint32_t>(requirements.matchTableWords);
  }
  if (requirements.csrWords > 0) {
    buffer.csr = allocator.allocate<uint32_t>(requirements.csrWords);
  }
  if (requirements.pairInputCount > 0) {
    buffer.pairInputs = allocator.allocate<DevicePerPairInput>(requirements.pairInputCount);
  }
  if (requirements.resultBytes > 0) {
    buffer.results = allocator.allocate<std::byte>(requirements.resultBytes);
  }
  return buffer;
}

struct FmcsExecutor {
  std::unique_ptr<nvMolKit::ScopedStream> ownedStream;
  cudaStream_t                            stream = nullptr;
  nvMolKit::ScopedCudaEvent               completionEvent;
  FmcsPinnedHostBuffer                    pinnedHost;
  nvMolKit::AsyncDeviceVector<uint32_t>   matchTableStorage;
  BatchDeviceBuffers                      deviceBuffers;

  FmcsExecutor(int                           executorIdx,
               cudaStream_t                  externalStream,
               bool                          useExternalStream,
               const PinnedHostRequirements& pinnedRequirements,
               const DeviceRequirements&     deviceRequirements)
      : pinnedHost(createPinnedHostBuffer(pinnedRequirements)) {
    if (useExternalStream) {
      stream = externalStream;
    } else {
      const std::string streamName = "fmcs_executor_" + std::to_string(executorIdx);
      ownedStream                  = std::make_unique<nvMolKit::ScopedStream>(streamName.c_str());
      stream                       = ownedStream->stream();
    }

    matchTableStorage        = nvMolKit::AsyncDeviceVector<uint32_t>(deviceRequirements.matchTableWords, stream);
    deviceBuffers.csrStorage = nvMolKit::AsyncDeviceVector<uint32_t>(deviceRequirements.csrWords, stream);
    deviceBuffers.pairInputs =
      nvMolKit::AsyncDeviceVector<DevicePerPairInput>(deviceRequirements.pairInputCount, stream);
    deviceBuffers.resultsStorage = nvMolKit::AsyncDeviceVector<std::byte>(deviceRequirements.resultBytes, stream);
    deviceBuffers.queueStorage   = nvMolKit::AsyncDeviceVector<std::byte>(deviceRequirements.queueBytes, stream);
    deviceBuffers.scratchStorage = nvMolKit::AsyncDeviceVector<std::byte>(deviceRequirements.scratchBytes, stream);
  }
};

int validateRequestedExecutorCount(const Parameters& params) {
  if (params.executorsPerRunner < 1 || params.executorsPerRunner > kMaxFmcsExecutorsPerRunner) {
    throw std::invalid_argument("fMCS executorsPerRunner must be between 1 and " +
                                std::to_string(kMaxFmcsExecutorsPerRunner));
  }
  return params.executorsPerRunner;
}

int validateRequestedBlockSize(const Parameters& params) {
  if (params.blockSize == 64 || params.blockSize == 128 || params.blockSize == 256 || params.blockSize == 512) {
    return params.blockSize;
  }
  throw std::invalid_argument("fMCS blockSize must be one of 64, 128, 256, or 512");
}

std::vector<PairMatchTablesDevice> uploadPairMatchTablesToStorage(const std::vector<PairMatchTablesHost>& host,
                                                                  std::span<uint32_t>                     packedHost,
                                                                  cudaStream_t                            stream,
                                                                  nvMolKit::AsyncDeviceVector<uint32_t>&  storage) {
  const size_t totalWords = computePairMatchTableWords(host);
  if (packedHost.size() < totalWords || storage.size() < totalWords) {
    throw std::out_of_range("Persistent match-table buffer is too small");
  }

  std::vector<PairMatchTablesDevice> tables(host.size());
  uint32_t*                          base   = storage.data();
  size_t                             cursor = 0;
  for (size_t i = 0; i < host.size(); ++i) {
    const auto& source = host[i];
    auto&       dest   = tables[i];
    if (!source.atoms.data.empty()) {
      std::copy(source.atoms.data.begin(), source.atoms.data.end(), packedHost.begin() + cursor);
      dest.atoms = {base + cursor, source.atoms.nRows, source.atoms.nCols, source.atoms.wordsPerRow};
      cursor += source.atoms.data.size();
    }
    if (!source.bonds.data.empty()) {
      std::copy(source.bonds.data.begin(), source.bonds.data.end(), packedHost.begin() + cursor);
      dest.bonds = {base + cursor, source.bonds.nRows, source.bonds.nCols, source.bonds.wordsPerRow};
      cursor += source.bonds.data.size();
    }
  }

  if (totalWords > 0) {
    checkCuda(
      cudaMemcpyAsync(storage.data(), packedHost.data(), totalWords * sizeof(uint32_t), cudaMemcpyHostToDevice, stream),
      "cudaMemcpyAsync (match tables)");
  }
  return tables;
}

size_t computeCsrWords(const std::vector<HostPairDescriptor*>& descs) {
  size_t totalWords = 0;
  for (auto* d : descs) {
    totalWords += d->packedQuery.rowOffsets.size();
    totalWords += d->packedQuery.colIndices.size();
    totalWords += d->packedQuery.bondIndices.size();
    totalWords += d->packedQuery.bondEndpoints.size();
    totalWords += d->packedTarget.rowOffsets.size();
    totalWords += d->packedTarget.colIndices.size();
    totalWords += d->packedTarget.bondIndices.size();
    totalWords += d->packedTarget.bondEndpoints.size();
  }
  return totalWords;
}

/// Pack every pair's CSR and bond-endpoint arrays into one pinned host buffer,
/// upload them with one H2D copy, and assemble pinned per-pair descriptors whose
/// pointers refer into the contiguous device allocation.
void uploadCsrAndAssemblePairInputs(const std::vector<HostPairDescriptor*>&   descs,
                                    const std::vector<PairMatchTablesDevice>& tablesDev,
                                    std::span<uint32_t>                       packedHost,
                                    std::span<DevicePerPairInput>             pairInputsHost,
                                    cudaStream_t                              stream,
                                    nvMolKit::AsyncDeviceVector<uint32_t>&    storage) {
  const size_t totalWords = computeCsrWords(descs);
  if (packedHost.size() < totalWords) {
    throw std::out_of_range("Pinned CSR staging buffer is too small");
  }
  if (pairInputsHost.size() < descs.size() || tablesDev.size() != descs.size()) {
    throw std::out_of_range("Pinned pair-input staging buffer is too small");
  }
  if (storage.size() < totalWords) {
    throw std::out_of_range("Device CSR buffer is too small");
  }

  uint32_t* base = storage.data();

  size_t cursor  = 0;
  auto   packVec = [&](const std::vector<uint32_t>& v) -> uint32_t* {
    if (v.empty())
      return nullptr;
    uint32_t* dst = base + cursor;
    std::copy(v.begin(), v.end(), packedHost.begin() + cursor);
    cursor += v.size();
    return dst;
  };

  for (size_t i = 0; i < descs.size(); ++i) {
    const auto&         d = *descs[i];
    DevicePerPairInput& p = pairInputsHost[i];
    p.queryNumAtoms       = d.queryGraph->numVertices;
    p.queryNumBonds       = d.queryGraph->numEdges;
    p.targetNumAtoms      = d.targetGraph->numVertices;
    p.targetNumBonds      = d.targetGraph->numEdges;
    p.queryRowOffsets     = packVec(d.packedQuery.rowOffsets);
    p.queryColIndices     = packVec(d.packedQuery.colIndices);
    p.queryBondIndices    = packVec(d.packedQuery.bondIndices);
    p.queryBondEndpoints  = packVec(d.packedQuery.bondEndpoints);
    p.targetRowOffsets    = packVec(d.packedTarget.rowOffsets);
    p.targetColIndices    = packVec(d.packedTarget.colIndices);
    p.targetBondIndices   = packVec(d.packedTarget.bondIndices);
    p.targetBondEndpoints = packVec(d.packedTarget.bondEndpoints);
    p.tables              = tablesDev[i];
    p.swapped             = d.swapped;
  }

  checkCuda(
    cudaMemcpyAsync(storage.data(), packedHost.data(), totalWords * sizeof(uint32_t), cudaMemcpyHostToDevice, stream),
    "cudaMemcpyAsync (packed CSR)");
}

template <int blockThreads, int maxAtoms, int maxBonds, class Policy>
void launchTierAsync(std::span<const DevicePerPairInput> hostPairInputs,
                     const Parameters&                   params,
                     cudaStream_t                        stream,
                     BatchDeviceBuffers&                 bufs) {
  const int numPairs = static_cast<int>(hostPairInputs.size());
  if (numPairs == 0)
    return;

  if (bufs.pairInputs.size() < static_cast<size_t>(numPairs)) {
    throw std::out_of_range("Device pair-input buffer is too small");
  }
  checkCuda(cudaMemcpyAsync(bufs.pairInputs.data(),
                            hostPairInputs.data(),
                            numPairs * sizeof(DevicePerPairInput),
                            cudaMemcpyHostToDevice,
                            stream),
            "cudaMemcpyAsync (pair inputs)");

  const size_t resultsBytes = static_cast<size_t>(numPairs) * sizeof(DeviceMCSResult<maxAtoms, maxBonds>);
  if (bufs.resultsStorage.size() < resultsBytes) {
    throw std::out_of_range("Device result buffer is too small");
  }
  auto* dResults = reinterpret_cast<DeviceMCSResult<maxAtoms, maxBonds>*>(bufs.resultsStorage.data());

  using QueuedT           = QueuedSeed<maxAtoms, maxBonds, maxAtoms, maxBonds>;
  const size_t queueBytes = static_cast<size_t>(numPairs) * kFmcsQueueCapacity * sizeof(QueuedT);
  if (bufs.queueStorage.size() < queueBytes) {
    throw std::out_of_range("Device queue buffer is too small");
  }
  auto* dQueue = reinterpret_cast<QueuedT*>(bufs.queueStorage.data());

  using ScratchT     = FmcsSubstructureScratch<maxAtoms, maxAtoms>;
  ScratchT* dScratch = nullptr;
  if constexpr (kUseGlobalSubstructureScratch<blockThreads, maxAtoms>) {
    const size_t scratchBytes =
      static_cast<size_t>(numPairs) * static_cast<size_t>(FmcsBlockConfig<blockThreads>::numGroups) * sizeof(ScratchT);
    if (bufs.scratchStorage.size() < scratchBytes) {
      throw std::out_of_range("Device scratch buffer is too small");
    }
    dScratch = reinterpret_cast<ScratchT*>(bufs.scratchStorage.data());
  }

  unsigned long long timeoutClocks = 0;
  if (params.timeoutMs > 0.0f) {
    int device = 0;
    checkCuda(cudaGetDevice(&device), "cudaGetDevice (timeout)");
    int clockRateKHz = 0;
    checkCuda(cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, device),
              "cudaDeviceGetAttribute (clock rate)");
    timeoutClocks = static_cast<unsigned long long>(
      std::max(1.0, static_cast<double>(params.timeoutMs) * static_cast<double>(clockRateKHz)));
  }

  // Block size is selected from compile-time kernel specializations so the
  // kernel's per-group state arrays stay statically sized.
  dim3 grid(static_cast<unsigned>(numPairs));
  dim3 block(static_cast<unsigned>(blockThreads));
  fmcsKernel<maxAtoms, maxBonds, blockThreads, Policy, kUseGlobalSubstructureScratch<blockThreads, maxAtoms>>
    <<<grid, block, 0, stream>>>(bufs.pairInputs.data(),
                                 dResults,
                                 dQueue,
                                 dScratch,
                                 kFmcsQueueCapacity,
                                 numPairs,
                                 timeoutClocks);
  checkCuda(cudaGetLastError(), "fmcsKernel launch");
}

/// Translate fixed-size DeviceMCSResult into MCSResult, un-swapping the
/// mappings if @p swapped is set.  The device result stores edges as
/// (queryBondIdx, targetBondIdx) pairs; we recover (u, v) endpoints
/// here by indexing the per-pair @c bondEndpoints arrays the host
/// already has.
template <int maxAtoms, int maxBonds>
MCSResult expandDeviceResult(const DeviceMCSResult<maxAtoms, maxBonds>& dr,
                             const std::vector<std::uint32_t>&          queryBondEndpoints,
                             const std::vector<std::uint32_t>&          targetBondEndpoints,
                             bool                                       swapped) {
  MCSResult r;
  r.numCommonVertices = dr.numCommonVertices;
  r.numCommonEdges    = dr.numCommonEdges;
  r.timedOut          = dr.timedOut;
  r.overflowed        = dr.overflowed;
  r.mappingA.reserve(static_cast<size_t>(dr.numCommonVertices));
  r.mappingB.reserve(static_cast<size_t>(dr.numCommonVertices));
  for (int i = 0; i < dr.numCommonVertices; ++i) {
    const size_t a = dr.mappingA[i];
    const size_t b = dr.mappingB[i];
    if (swapped) {
      r.mappingA.push_back(b);
      r.mappingB.push_back(a);
    } else {
      r.mappingA.push_back(a);
      r.mappingB.push_back(b);
    }
  }
  r.edgeMappingA.reserve(static_cast<size_t>(dr.numCommonEdges));
  r.edgeMappingB.reserve(static_cast<size_t>(dr.numCommonEdges));
  for (int i = 0; i < dr.numCommonEdges; ++i) {
    const std::uint32_t       qPacked = queryBondEndpoints[dr.bondMapA[i]];
    const std::uint32_t       tPacked = targetBondEndpoints[dr.bondMapB[i]];
    std::pair<size_t, size_t> eA{static_cast<size_t>(qPacked >> 16), static_cast<size_t>(qPacked & 0xFFFFu)};
    std::pair<size_t, size_t> eB{static_cast<size_t>(tPacked >> 16), static_cast<size_t>(tPacked & 0xFFFFu)};
    if (swapped)
      std::swap(eA, eB);
    r.edgeMappingA.push_back(eA);
    r.edgeMappingB.push_back(eB);
  }
  return r;
}

template <int maxAtoms, int maxBonds, class Policy> struct TierChunk {
  std::vector<int>                   resultIndices;
  std::vector<HostPairDescriptor*>   tierDescs;
  std::vector<PairMatchTablesHost>   hostTables;
  std::vector<PairMatchTablesDevice> deviceTables;
};

template <class Policy>
using TierChunkVariant = std::variant<std::unique_ptr<TierChunk<16, 16, Policy>>,
                                      std::unique_ptr<TierChunk<32, 32, Policy>>,
                                      std::unique_ptr<TierChunk<64, 64, Policy>>,
                                      std::unique_ptr<TierChunk<128, 128, Policy>>>;

template <int maxAtoms, int maxBonds, class Policy>
std::unique_ptr<TierChunk<maxAtoms, maxBonds, Policy>> makeTierChunk(const std::vector<HostPairDescriptor*>& descs,
                                                                     const std::vector<int>&                 indices,
                                                                     size_t                                  begin,
                                                                     size_t                                  end) {
  auto chunk = std::make_unique<TierChunk<maxAtoms, maxBonds, Policy>>();
  chunk->resultIndices.reserve(end - begin);
  chunk->tierDescs.reserve(end - begin);
  chunk->hostTables.reserve(end - begin);
  for (size_t pos = begin; pos < end; ++pos) {
    const int idx = indices[pos];
    chunk->resultIndices.push_back(idx);
    chunk->tierDescs.push_back(descs[idx]);
    chunk->hostTables.push_back(descs[idx]->tables);
  }
  return chunk;
}

template <int blockThreads, int maxAtoms, int maxBonds, class Policy>
void enqueueTierChunks(const std::vector<HostPairDescriptor*>&              descs,
                       const std::vector<int>&                              indices,
                       size_t                                               chunkSize,
                       nvMolKit::ThreadSafeQueue<TierChunkVariant<Policy>>& chunkQueue,
                       PinnedHostRequirements&                              pinnedRequirements,
                       DeviceRequirements&                                  deviceRequirements) {
  if (indices.empty())
    return;

  for (size_t begin = 0; begin < indices.size(); begin += chunkSize) {
    const size_t end   = std::min(begin + chunkSize, indices.size());
    auto         chunk = makeTierChunk<maxAtoms, maxBonds, Policy>(descs, indices, begin, end);
    pinnedRequirements.matchTableWords =
      std::max(pinnedRequirements.matchTableWords, computePairMatchTableWords(chunk->hostTables));
    pinnedRequirements.csrWords       = std::max(pinnedRequirements.csrWords, computeCsrWords(chunk->tierDescs));
    pinnedRequirements.pairInputCount = std::max(pinnedRequirements.pairInputCount, end - begin);
    pinnedRequirements.resultBytes =
      std::max(pinnedRequirements.resultBytes, (end - begin) * sizeof(DeviceMCSResult<maxAtoms, maxBonds>));

    deviceRequirements.matchTableWords =
      std::max(deviceRequirements.matchTableWords, computePairMatchTableWords(chunk->hostTables));
    deviceRequirements.csrWords       = std::max(deviceRequirements.csrWords, computeCsrWords(chunk->tierDescs));
    deviceRequirements.pairInputCount = std::max(deviceRequirements.pairInputCount, end - begin);
    deviceRequirements.resultBytes =
      std::max(deviceRequirements.resultBytes, (end - begin) * sizeof(DeviceMCSResult<maxAtoms, maxBonds>));
    using QueuedT = QueuedSeed<maxAtoms, maxBonds, maxAtoms, maxBonds>;
    deviceRequirements.queueBytes =
      std::max(deviceRequirements.queueBytes, (end - begin) * kFmcsQueueCapacity * sizeof(QueuedT));
    if constexpr (kUseGlobalSubstructureScratch<blockThreads, maxAtoms>) {
      using ScratchT = FmcsSubstructureScratch<maxAtoms, maxAtoms>;
      deviceRequirements.scratchBytes =
        std::max(deviceRequirements.scratchBytes,
                 (end - begin) * static_cast<size_t>(FmcsBlockConfig<blockThreads>::numGroups) * sizeof(ScratchT));
    }

    TierChunkVariant<Policy> item = std::move(chunk);
    chunkQueue.push(std::move(item));
  }
}

template <int blockThreads, int maxAtoms, int maxBonds, class Policy>
void launchTierChunk(FmcsExecutor&                                           executor,
                     std::unique_ptr<TierChunk<maxAtoms, maxBonds, Policy>>& chunk,
                     const Parameters&                                       params) {
  cudaStream_t executorStream  = executor.stream;
  const size_t matchTableWords = computePairMatchTableWords(chunk->hostTables);
  chunk->deviceTables =
    uploadPairMatchTablesToStorage(chunk->hostTables,
                                   std::span<uint32_t>(executor.pinnedHost.matchTables.data(), matchTableWords),
                                   executorStream,
                                   executor.matchTableStorage);

  const size_t numPairs = chunk->tierDescs.size();
  uploadCsrAndAssemblePairInputs(chunk->tierDescs,
                                 chunk->deviceTables,
                                 std::span<uint32_t>(executor.pinnedHost.csr.data(), computeCsrWords(chunk->tierDescs)),
                                 std::span<DevicePerPairInput>(executor.pinnedHost.pairInputs.data(), numPairs),
                                 executorStream,
                                 executor.deviceBuffers.csrStorage);

  launchTierAsync<blockThreads, maxAtoms, maxBonds, Policy>(
    std::span<const DevicePerPairInput>(executor.pinnedHost.pairInputs.data(), numPairs),
    params,
    executorStream,
    executor.deviceBuffers);

  checkCuda(cudaEventRecord(executor.completionEvent.event(), executorStream), "cudaEventRecord (fMCS kernel done)");
}

template <int maxAtoms, int maxBonds, class Policy>
void drainTierChunk(FmcsExecutor&                                           executor,
                    std::unique_ptr<TierChunk<maxAtoms, maxBonds, Policy>>& chunk,
                    std::vector<MCSResult>&                                 outResults) {
  // Do not enqueue D2H copies behind kernels that have not completed yet. With
  // multiple workers, those blocked D2H copies can hold up later H2D work in
  // the copy-engine queue and create an artificial wave-wide barrier.
  checkCuda(cudaEventSynchronize(executor.completionEvent.event()), "cudaEventSynchronize (fMCS kernel done)");

  const size_t resultBytes = chunk->resultIndices.size() * sizeof(DeviceMCSResult<maxAtoms, maxBonds>);
  if (executor.pinnedHost.results.size() < resultBytes) {
    throw std::out_of_range("Pinned result staging buffer is too small");
  }
  checkCuda(cudaMemcpyAsync(executor.pinnedHost.results.data(),
                            executor.deviceBuffers.resultsStorage.data(),
                            resultBytes,
                            cudaMemcpyDeviceToHost,
                            executor.stream),
            "cudaMemcpyAsync (results)");
  checkCuda(cudaEventRecord(executor.completionEvent.event(), executor.stream),
            "cudaEventRecord (fMCS result copy done)");
  checkCuda(cudaEventSynchronize(executor.completionEvent.event()), "cudaEventSynchronize (fMCS result copy done)");

  const auto* hostResults =
    reinterpret_cast<const DeviceMCSResult<maxAtoms, maxBonds>*>(executor.pinnedHost.results.data());
  for (size_t k = 0; k < chunk->resultIndices.size(); ++k) {
    outResults[chunk->resultIndices[k]] =
      expandDeviceResult<maxAtoms, maxBonds>(hostResults[k],
                                             chunk->tierDescs[k]->packedQuery.bondEndpoints,
                                             chunk->tierDescs[k]->packedTarget.bondEndpoints,
                                             chunk->tierDescs[k]->swapped);
  }
}

template <int blockThreads, class Policy>
void runTierChunks(nvMolKit::ThreadSafeQueue<TierChunkVariant<Policy>>& chunkQueue,
                   size_t                                               numChunks,
                   const PinnedHostRequirements&                        pinnedRequirements,
                   const DeviceRequirements&                            deviceRequirements,
                   const Parameters&                                    params,
                   cudaStream_t                                         stream,
                   std::vector<MCSResult>&                              outResults) {
  if (numChunks == 0)
    return;

  const int executorCount =
    static_cast<int>(std::min<size_t>(static_cast<size_t>(validateRequestedExecutorCount(params)), numChunks));
  if (executorCount > 1 && stream != nullptr) {
    throw std::invalid_argument("fMCS multi-executor dispatch does not support an external CUDA stream");
  }

  std::vector<std::unique_ptr<FmcsExecutor>> executorStorage;
  executorStorage.reserve(static_cast<size_t>(executorCount));
  std::vector<FmcsExecutor*> executors;
  executors.reserve(static_cast<size_t>(executorCount));
  const bool useExternalStream = executorCount == 1;
  for (int i = 0; i < executorCount; ++i) {
    auto executor =
      std::make_unique<FmcsExecutor>(i, stream, useExternalStream, pinnedRequirements, deviceRequirements);
    executors.push_back(executor.get());
    executorStorage.push_back(std::move(executor));
  }

  auto launchChunk = [&](FmcsExecutor& executor, TierChunkVariant<Policy>& chunk) {
    std::visit(
      [&](auto& typedChunk) {
        if (typedChunk) {
          launchTierChunk<blockThreads>(executor, typedChunk, params);
        }
      },
      chunk);
  };

  auto drainChunk = [&](FmcsExecutor& executor, TierChunkVariant<Policy>& chunk) {
    std::visit(
      [&](auto& typedChunk) {
        if (typedChunk) {
          drainTierChunk(executor, typedChunk, outResults);
        }
      },
      chunk);
  };

  nvMolKit::runQueuedExecutorRing(executors, chunkQueue, launchChunk, drainChunk);
}

template <int blockThreads, class Policy, class InputT>
std::vector<MCSResult> runBatchWithBlockSize(const std::vector<InputT>& a,
                                             const std::vector<InputT>& b,
                                             Parameters                 params,
                                             cudaStream_t               stream) {
  if (a.size() != b.size()) {
    throw std::runtime_error("fMCS batch: graphsA and graphsB must have equal length");
  }
  const size_t           N = a.size();
  std::vector<MCSResult> results(N);
  if (N == 0)
    return results;

  std::vector<HostPairDescriptor> descs(N);
  std::array<std::vector<int>, 4> tierIndices;
  for (size_t i = 0; i < N; ++i) {
    descs[i] = buildPairDescriptor<Policy, InputT>(a[i], b[i], params);
    if (descs[i].overflowed) {
      MCSResult r;
      r.overflowed = true;
      results[i]   = r;
      continue;
    }
    tierIndices[descs[i].tier].push_back(static_cast<int>(i));
  }

  std::vector<HostPairDescriptor*> descPtrs(N);
  for (size_t i = 0; i < N; ++i)
    descPtrs[i] = &descs[i];

  const size_t chunkSize = params.batchSize > 0 ? static_cast<size_t>(params.batchSize) : N;
  size_t       numChunks = 0;
  for (const auto& indices : tierIndices) {
    if (!indices.empty()) {
      numChunks += (indices.size() + chunkSize - 1) / chunkSize;
    }
  }

  nvMolKit::ThreadSafeQueue<TierChunkVariant<Policy>> chunkQueue;
  PinnedHostRequirements                              pinnedRequirements;
  DeviceRequirements                                  deviceRequirements;
  enqueueTierChunks<blockThreads, 16, 16, Policy>(descPtrs,
                                                  tierIndices[0],
                                                  chunkSize,
                                                  chunkQueue,
                                                  pinnedRequirements,
                                                  deviceRequirements);
  enqueueTierChunks<blockThreads, 32, 32, Policy>(descPtrs,
                                                  tierIndices[1],
                                                  chunkSize,
                                                  chunkQueue,
                                                  pinnedRequirements,
                                                  deviceRequirements);
  enqueueTierChunks<blockThreads, 64, 64, Policy>(descPtrs,
                                                  tierIndices[2],
                                                  chunkSize,
                                                  chunkQueue,
                                                  pinnedRequirements,
                                                  deviceRequirements);
  enqueueTierChunks<blockThreads, 128, 128, Policy>(descPtrs,
                                                    tierIndices[3],
                                                    chunkSize,
                                                    chunkQueue,
                                                    pinnedRequirements,
                                                    deviceRequirements);
  chunkQueue.close();
  runTierChunks<blockThreads, Policy>(chunkQueue,
                                      numChunks,
                                      pinnedRequirements,
                                      deviceRequirements,
                                      params,
                                      stream,
                                      results);

  return results;
}

template <class Policy, class InputT>
std::vector<MCSResult> runBatch(const std::vector<InputT>& a,
                                const std::vector<InputT>& b,
                                Parameters                 params,
                                cudaStream_t               stream) {
  switch (validateRequestedBlockSize(params)) {
    case 64:
      return runBatchWithBlockSize<64, Policy, InputT>(a, b, params, stream);
    case 128:
      return runBatchWithBlockSize<128, Policy, InputT>(a, b, params, stream);
    case 256:
      return runBatchWithBlockSize<256, Policy, InputT>(a, b, params, stream);
    case 512:
      return runBatchWithBlockSize<512, Policy, InputT>(a, b, params, stream);
  }
  throw std::logic_error("unreachable fMCS blockSize dispatch");
}

}  // namespace

std::vector<MCSResult> findMCESfMCSBatch(const std::vector<Graph>& graphsA,
                                         const std::vector<Graph>& graphsB,
                                         Parameters                params,
                                         cudaStream_t              stream) {
  return runBatch<NullFMCSPolicy, Graph>(graphsA, graphsB, params, stream);
}

std::vector<MCSResult> findMCESfMCSBatchLabeled(const std::vector<LabeledGraph>& graphsA,
                                                const std::vector<LabeledGraph>& graphsB,
                                                Parameters                       params,
                                                cudaStream_t                     stream) {
  return runBatch<LabeledFMCSPolicy, LabeledGraph>(graphsA, graphsB, params, stream);
}

}  // namespace fmcs
}  // namespace mcs
