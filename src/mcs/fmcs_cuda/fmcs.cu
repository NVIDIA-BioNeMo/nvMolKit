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

constexpr int kMaxFmcsExecutorsPerRunner = 8;

struct FmcsExecutor {
  std::unique_ptr<nvMolKit::ScopedStream> ownedStream;
  cudaStream_t                            stream = nullptr;
  nvMolKit::ScopedCudaEvent               copyDoneEvent;

  FmcsExecutor(int executorIdx, cudaStream_t externalStream, bool useExternalStream) {
    if (useExternalStream) {
      stream = externalStream;
      return;
    }

    const std::string streamName = "fmcs_executor_" + std::to_string(executorIdx);
    ownedStream                  = std::make_unique<nvMolKit::ScopedStream>(streamName.c_str());
    stream                       = ownedStream->stream();
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

/// Upload every pair's CSR and bond-endpoint arrays into one contiguous
/// device buffer, returning per-pair descriptors whose pointers refer into
/// that buffer.
std::vector<DevicePerPairInput> uploadCsrAndAssemblePairInputs(const std::vector<HostPairDescriptor*>&   descs,
                                                               const std::vector<PairMatchTablesDevice>& tablesDev,
                                                               cudaStream_t                              stream,
                                                               nvMolKit::AsyncDeviceVector<uint32_t>&    storage) {
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

  storage        = nvMolKit::AsyncDeviceVector<uint32_t>(totalWords, stream);
  uint32_t* base = storage.data();

  std::vector<DevicePerPairInput> out(descs.size());
  size_t                          cursor    = 0;
  auto                            uploadVec = [&](const std::vector<uint32_t>& v) -> uint32_t* {
    if (v.empty())
      return nullptr;
    uint32_t* dst = base + cursor;
    checkCuda(cudaMemcpyAsync(dst, v.data(), v.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync (CSR)");
    cursor += v.size();
    return dst;
  };

  for (size_t i = 0; i < descs.size(); ++i) {
    const auto&         d = *descs[i];
    DevicePerPairInput& p = out[i];
    p.queryNumAtoms       = d.queryGraph->numVertices;
    p.queryNumBonds       = d.queryGraph->numEdges;
    p.targetNumAtoms      = d.targetGraph->numVertices;
    p.targetNumBonds      = d.targetGraph->numEdges;
    p.queryRowOffsets     = uploadVec(d.packedQuery.rowOffsets);
    p.queryColIndices     = uploadVec(d.packedQuery.colIndices);
    p.queryBondIndices    = uploadVec(d.packedQuery.bondIndices);
    p.queryBondEndpoints  = uploadVec(d.packedQuery.bondEndpoints);
    p.targetRowOffsets    = uploadVec(d.packedTarget.rowOffsets);
    p.targetColIndices    = uploadVec(d.packedTarget.colIndices);
    p.targetBondIndices   = uploadVec(d.packedTarget.bondIndices);
    p.targetBondEndpoints = uploadVec(d.packedTarget.bondEndpoints);
    p.tables              = tablesDev[i];
    p.swapped             = d.swapped;
  }

  return out;
}

template <int blockThreads, int maxAtoms, int maxBonds, class Policy>
void launchTierAsync(const std::vector<DevicePerPairInput>&            hostPairInputs,
                     const Parameters&                                 params,
                     cudaStream_t                                      stream,
                     BatchDeviceBuffers&                               bufs,
                     std::vector<DeviceMCSResult<maxAtoms, maxBonds>>& hostResults) {
  const int numPairs = static_cast<int>(hostPairInputs.size());
  if (numPairs == 0)
    return;

  bufs.pairInputs = nvMolKit::AsyncDeviceVector<DevicePerPairInput>(static_cast<size_t>(numPairs), stream);
  checkCuda(cudaMemcpyAsync(bufs.pairInputs.data(),
                            hostPairInputs.data(),
                            numPairs * sizeof(DevicePerPairInput),
                            cudaMemcpyHostToDevice,
                            stream),
            "cudaMemcpyAsync (pair inputs)");

  const size_t resultsBytes = static_cast<size_t>(numPairs) * sizeof(DeviceMCSResult<maxAtoms, maxBonds>);
  bufs.resultsStorage       = nvMolKit::AsyncDeviceVector<std::byte>(resultsBytes, stream);
  auto* dResults            = reinterpret_cast<DeviceMCSResult<maxAtoms, maxBonds>*>(bufs.resultsStorage.data());

  using QueuedT           = QueuedSeed<maxAtoms, maxBonds, maxAtoms, maxBonds>;
  const size_t queueBytes = static_cast<size_t>(numPairs) * kFmcsQueueCapacity * sizeof(QueuedT);
  bufs.queueStorage       = nvMolKit::AsyncDeviceVector<std::byte>(queueBytes, stream);
  auto* dQueue            = reinterpret_cast<QueuedT*>(bufs.queueStorage.data());

  using ScratchT     = FmcsSubstructureScratch<maxAtoms, maxAtoms>;
  ScratchT* dScratch = nullptr;
  if constexpr (kUseGlobalSubstructureScratch<blockThreads, maxAtoms>) {
    const size_t scratchBytes =
      static_cast<size_t>(numPairs) * static_cast<size_t>(FmcsBlockConfig<blockThreads>::numGroups) * sizeof(ScratchT);
    bufs.scratchStorage = nvMolKit::AsyncDeviceVector<std::byte>(scratchBytes, stream);
    dScratch            = reinterpret_cast<ScratchT*>(bufs.scratchStorage.data());
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

  hostResults.resize(static_cast<size_t>(numPairs));
  checkCuda(cudaMemcpyAsync(hostResults.data(), dResults, resultsBytes, cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync (results)");
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
  std::vector<int>                                 resultIndices;
  std::vector<HostPairDescriptor*>                 tierDescs;
  std::vector<PairMatchTablesHost>                 hostTables;
  UploadedPairMatchTables                          deviceTables;
  BatchDeviceBuffers                               deviceBuffers;
  std::vector<DevicePerPairInput>                  hostPairInputs;
  std::vector<DeviceMCSResult<maxAtoms, maxBonds>> hostResults;
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

template <int maxAtoms, int maxBonds, class Policy>
void enqueueTierChunks(const std::vector<HostPairDescriptor*>&              descs,
                       const std::vector<int>&                              indices,
                       size_t                                               chunkSize,
                       nvMolKit::ThreadSafeQueue<TierChunkVariant<Policy>>& chunkQueue) {
  if (indices.empty())
    return;

  for (size_t begin = 0; begin < indices.size(); begin += chunkSize) {
    const size_t             end  = std::min(begin + chunkSize, indices.size());
    TierChunkVariant<Policy> item = makeTierChunk<maxAtoms, maxBonds, Policy>(descs, indices, begin, end);
    chunkQueue.push(std::move(item));
  }
}

template <int blockThreads, int maxAtoms, int maxBonds, class Policy>
void launchTierChunk(FmcsExecutor&                                           executor,
                     std::unique_ptr<TierChunk<maxAtoms, maxBonds, Policy>>& chunk,
                     const Parameters&                                       params) {
  cudaStream_t executorStream = executor.stream;
  chunk->deviceTables         = uploadPairMatchTables(chunk->hostTables, executorStream);
  chunk->hostPairInputs       = uploadCsrAndAssemblePairInputs(chunk->tierDescs,
                                                         chunk->deviceTables.tables,
                                                         executorStream,
                                                         chunk->deviceBuffers.csrStorage);

  launchTierAsync<blockThreads, maxAtoms, maxBonds, Policy>(chunk->hostPairInputs,
                                                            params,
                                                            executorStream,
                                                            chunk->deviceBuffers,
                                                            chunk->hostResults);

  checkCuda(cudaEventRecord(executor.copyDoneEvent.event(), executorStream), "cudaEventRecord (fMCS chunk copy done)");
}

template <int maxAtoms, int maxBonds, class Policy>
void drainTierChunk(FmcsExecutor&                                           executor,
                    std::unique_ptr<TierChunk<maxAtoms, maxBonds, Policy>>& chunk,
                    std::vector<MCSResult>&                                 outResults) {
  checkCuda(cudaEventSynchronize(executor.copyDoneEvent.event()), "cudaEventSynchronize (fMCS chunk copy done)");
  for (size_t k = 0; k < chunk->hostResults.size(); ++k) {
    outResults[chunk->resultIndices[k]] =
      expandDeviceResult<maxAtoms, maxBonds>(chunk->hostResults[k],
                                             chunk->tierDescs[k]->packedQuery.bondEndpoints,
                                             chunk->tierDescs[k]->packedTarget.bondEndpoints,
                                             chunk->tierDescs[k]->swapped);
  }
}

template <int blockThreads, class Policy>
void runTierChunks(nvMolKit::ThreadSafeQueue<TierChunkVariant<Policy>>& chunkQueue,
                   size_t                                               numChunks,
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
    auto executor = std::make_unique<FmcsExecutor>(i, stream, useExternalStream);
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
  enqueueTierChunks<16, 16, Policy>(descPtrs, tierIndices[0], chunkSize, chunkQueue);
  enqueueTierChunks<32, 32, Policy>(descPtrs, tierIndices[1], chunkSize, chunkQueue);
  enqueueTierChunks<64, 64, Policy>(descPtrs, tierIndices[2], chunkSize, chunkQueue);
  enqueueTierChunks<128, 128, Policy>(descPtrs, tierIndices[3], chunkSize, chunkQueue);
  chunkQueue.close();
  runTierChunks<blockThreads, Policy>(chunkQueue, numChunks, params, stream, results);

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
