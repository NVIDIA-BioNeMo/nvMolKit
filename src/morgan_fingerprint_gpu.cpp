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

#include "src/morgan_fingerprint_gpu.h"

#include <DataStructs/ExplicitBitVect.h>
#include <GraphMol/ROMol.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <vector>

#include "src/data_structures/flat_bit_vect.h"
#include "src/morgan_fingerprint_common.h"
#include "src/morgan_fingerprint_cpu.h"
#include "src/morgan_fingerprint_kernels.h"
#include "src/utils/host_vector.h"
#include "src/utils/nvtx.h"
#include "src/utils/openmp_helpers.h"
namespace nvMolKit {

namespace {

constexpr int kDefaultGpuBatchSize = 2048;

// A simple thread-safe bag of work items (not FIFO), supporting batched retrieval.
// Important: Once the get_n function is called, do not add any more items to the bag. This is a drain-only class once
// begun, This is to optimize for the case of repeated calls to empty bags not needing to fully lock.
class WorkBag {
 public:
  WorkBag() = default;
  explicit WorkBag(std::vector<int> initial) : items_(std::move(initial)) {}

  // Returns up to n items from the end, removing them from the bag. Returns empty vector if no items are available.
  std::vector<int> get_n(const size_t n) {
    if (drained_.load(std::memory_order_relaxed)) {
      return {};
    }
    const std::lock_guard<std::mutex> lock(mtx_);
    const size_t                      available = items_.size();
    if (available == 0 || n == 0) {
      if (available == 0) {
        drained_.store(true, std::memory_order_release);
      }
      return {};
    }
    const size_t     take = std::min(n, available);
    std::vector<int> out;
    out.reserve(take);
    auto startIt = items_.end() - static_cast<std::ptrdiff_t>(take);
    out.insert(out.end(), startIt, items_.end());
    items_.erase(startIt, items_.end());
    if (items_.empty()) {
      drained_.store(true, std::memory_order_relaxed);
    }
    return out;
  }

  // Copies up to n items from the end into a pinned host vector, removing them from the bag.
  // Returns the number of items copied. Retains shrinking/drain behavior.
  size_t copy_n(PinnedHostVector<int>& outPinned, size_t n) {
    if (drained_.load(std::memory_order_relaxed)) {
      return 0;
    }
    const std::lock_guard<std::mutex> lock(mtx_);
    const size_t                      available = items_.size();
    if (available == 0 || n == 0 || outPinned.empty()) {
      if (available == 0) {
        drained_.store(true, std::memory_order_release);
      }
      return 0;
    }
    const size_t take    = std::min({n, available, outPinned.size()});
    auto         startIt = items_.end() - static_cast<std::ptrdiff_t>(take);
    // Copy into pinned host memory
    std::copy(startIt, items_.end(), outPinned.begin());
    items_.erase(startIt, items_.end());
    if (items_.empty()) {
      drained_.store(true, std::memory_order_relaxed);
    }
    return take;
  }

  // Add an item to the bag. Not thread-safe.
  void push_back(int item) { items_.push_back(item); }

  // Returns the number of items in the bag. Not thread-safe.
  size_t size() const { return items_.size(); }

 private:
  mutable std::mutex mtx_;
  std::vector<int>   items_;
  std::atomic<bool>  drained_{false};
};

void allocateGpuBatch(MorganGPUBuffersBatch& buffers,
                      cudaStream_t           stream,
                      const size_t           numMols,
                      const int              radius,
                      const int              maxAtoms) {
  buffers.atomInvariants       = AsyncDeviceVector<std::uint32_t>(maxAtoms * numMols, stream);
  buffers.bondInvariants       = AsyncDeviceVector<std::uint32_t>(maxAtoms * numMols, stream);
  buffers.bondIndices          = AsyncDeviceVector<std::int16_t>(maxAtoms * numMols * kMaxBondsPerAtom, stream);
  buffers.bondOtherAtomIndices = AsyncDeviceVector<std::int16_t>(maxAtoms * numMols * kMaxBondsPerAtom, stream);
  buffers.nAtomsPerMol         = AsyncDeviceVector<std::int16_t>(numMols, stream);
  buffers.outputIndices        = AsyncDeviceVector<int>(numMols, stream);

  switch (maxAtoms) {
    case 32:
      buffers.allSeenNeighborhoods32 = AsyncDeviceVector<FlatBitVect<32>>(numMols * 32 * (radius + 1), stream);
      buffers.allSeenNeighborhoods32.zero();
      break;
    case 64:
      buffers.allSeenNeighborhoods64 = AsyncDeviceVector<FlatBitVect<64>>(numMols * 64 * (radius + 1), stream);
      buffers.allSeenNeighborhoods64.zero();
      break;
    case 128:
      buffers.allSeenNeighborhoods128 = AsyncDeviceVector<FlatBitVect<128>>(numMols * 128 * (radius + 1), stream);
      buffers.allSeenNeighborhoods128.zero();
      break;
    default:
      throw std::runtime_error("Unsupported max atoms for Morgan fingerprint GPU: " + std::to_string(maxAtoms));
  }
}

template <int fpSize>
void solveOnGPUBatch(const MorganGPUBuffersBatch&            buffers,
                     AsyncDeviceVector<FlatBitVect<fpSize>>& outputAccumulator,
                     const size_t                            radius,
                     const int                               maxAtoms,
                     const int                               scopedChunkSize,
                     cudaStream_t                            stream) {
  launchMorganFingerprintKernelBatch<fpSize>(buffers, outputAccumulator, radius, maxAtoms, scopedChunkSize, stream);
}

template <int fpSize>
void extractResultsFromGPUBatch(const AsyncDeviceVector<FlatBitVect<fpSize>>& outputBuffer,
                                std::vector<FlatBitVect<fpSize>>&             result) {
  outputBuffer.copyToHost(result);
}

template <int nBits>
void populateResults(const std::vector<FlatBitVect<nBits>>&         resultsGpuVec,
                     std::vector<std::unique_ptr<ExplicitBitVect>>& results,
                     const int                                      numThreads) {
  detail::OpenMPExceptionRegistry exceptionRegistry;
#pragma omp parallel for default(none) num_threads(numThreads) shared(resultsGpuVec, results, exceptionRegistry)
  for (size_t vecIndex = 0; vecIndex < resultsGpuVec.size(); ++vecIndex) {
    try {
      const auto& tempResult    = resultsGpuVec[vecIndex];
      auto        resultBitVect = std::make_unique<ExplicitBitVect>(nBits);

      for (int i = 0; i < nBits; i++) {
        if (tempResult[i]) {
          resultBitVect->setBit(i);
        }
      }
      results[vecIndex] = std::move(resultBitVect);
    } catch (...) {
      exceptionRegistry.store(std::current_exception());
    }
  }
  exceptionRegistry.rethrow();
}

template <int fpSize>
FlatBitVect<fpSize> processSingleLargeMolecule(const RDKit::ROMol& mol, const std::uint32_t maxRadius) {
  auto                fingerprint = internal::getFingerprintImpl(mol, maxRadius, std::uint32_t(fpSize));
  FlatBitVect<fpSize> flatBitVect(false);
  for (int bitId = 0; bitId < fpSize; bitId++) {
    flatBitVect.setBit(bitId, fingerprint->getBit(bitId));
  }
  return flatBitVect;
}

template <int fpSize>
void sendLargeResultsToOutput(const std::vector<std::pair<FlatBitVect<fpSize>, int>>& largeResults,
                              AsyncDeviceVector<FlatBitVect<fpSize>>&                 outputAccumulator) {
  for (const auto& resultPair : largeResults) {
    const auto& result = resultPair.first;
    const int   molIdx = resultPair.second;
    outputAccumulator.copyFromHost(&result, 1, 0, molIdx);
  }
}

template <int fpSize>
AsyncDeviceVector<FlatBitVect<fpSize>> computeFingerprintsCuImpl(const std::vector<const RDKit::ROMol*>& mols,
                                                                 const int                               maxRadius,
                                                                 const size_t dispatchChunkSizeInit,
                                                                 const int    nThreads,
                                                                 std::vector<MorganPerThreadBuffers>& threadBuffers,
                                                                 cudaStream_t stream = nullptr) {
  nvMolKit::ScopedNvtxRange range1("MorganFPBatchAllocation");
  const size_t              numMols           = mols.size();
  auto                      outputAccumulator = AsyncDeviceVector<FlatBitVect<fpSize>>(numMols, stream);
  cudaCheckError(cudaMemsetAsync(outputAccumulator.data(), 0, numMols * sizeof(FlatBitVect<fpSize>), stream));
  const int nThreadsActual = nThreads == 0 ? omp_get_max_threads() : nThreads;
  threadBuffers.resize(nThreadsActual);

  const size_t dispatchChunkSize = std::min(dispatchChunkSizeInit, numMols);

  for (auto& perThreadBuffer : threadBuffers) {
    perThreadBuffer.nAtomsPerMol.resize(dispatchChunkSize);
    allocateGpuBatch(*perThreadBuffer.gpuBuffers32, perThreadBuffer.stream.stream(), dispatchChunkSize, maxRadius, 32);
    allocateGpuBatch(*perThreadBuffer.gpuBuffers64, perThreadBuffer.stream.stream(), dispatchChunkSize, maxRadius, 64);
    allocateGpuBatch(*perThreadBuffer.gpuBuffers128,
                     perThreadBuffer.stream.stream(),
                     dispatchChunkSize,
                     maxRadius,
                     128);
    // Pre-allocate pinned host buffers (fixed-size, reused across batches)
    perThreadBuffer.h_atomInvariants32.resize(dispatchChunkSize * 32);
    perThreadBuffer.h_bondInvariants32.resize(dispatchChunkSize * 32);
    perThreadBuffer.h_bondIndices32.resize(dispatchChunkSize * 32 * kMaxBondsPerAtom);
    perThreadBuffer.h_bondOtherAtomIndices32.resize(dispatchChunkSize * 32 * kMaxBondsPerAtom);

    perThreadBuffer.h_atomInvariants64.resize(dispatchChunkSize * 64);
    perThreadBuffer.h_bondInvariants64.resize(dispatchChunkSize * 64);
    perThreadBuffer.h_bondIndices64.resize(dispatchChunkSize * 64 * kMaxBondsPerAtom);
    perThreadBuffer.h_bondOtherAtomIndices64.resize(dispatchChunkSize * 64 * kMaxBondsPerAtom);

    perThreadBuffer.h_atomInvariants128.resize(dispatchChunkSize * 128);
    perThreadBuffer.h_bondInvariants128.resize(dispatchChunkSize * 128);
    perThreadBuffer.h_bondIndices128.resize(dispatchChunkSize * 128 * kMaxBondsPerAtom);
    perThreadBuffer.h_bondOtherAtomIndices128.resize(dispatchChunkSize * 128 * kMaxBondsPerAtom);

    perThreadBuffer.h_outputIndices.resize(dispatchChunkSize);
    cudaCheckError(cudaEventRecord(perThreadBuffer.prevMemcpyDoneEvent.event(), perThreadBuffer.stream.stream()));
  }

  // Ensure output alloc+memset on external stream is visible to per-thread streams
  ScopedCudaEvent outputAllocDone;
  cudaCheckError(cudaEventRecord(outputAllocDone.event(), stream));
  for (const auto& perThreadBuffer : threadBuffers) {
    cudaCheckError(cudaStreamWaitEvent(perThreadBuffer.stream.stream(), outputAllocDone.event(), 0));
  }
  range1.pop();

  WorkBag work32;
  WorkBag work64;
  WorkBag work128;
  WorkBag workLarge;
  for (int i = 0; i < mols.size(); i++) {
    const auto& mol = *mols[i];
    if (mol.getNumAtoms() < 32 && mol.getNumBonds() < 32) {
      work32.push_back(i);
    } else if (mol.getNumAtoms() < 64 && mol.getNumBonds() < 64) {
      work64.push_back(i);
    } else if (mol.getNumAtoms() < 128 && mol.getNumBonds() < 128) {
      work128.push_back(i);
    } else {
      workLarge.push_back(i);
    }
  }
  const size_t numThreads32    = (work32.size() + dispatchChunkSize - 1) / dispatchChunkSize;
  const size_t numThreads64    = (work64.size() + dispatchChunkSize - 1) / dispatchChunkSize;
  const size_t numThreads128   = (work128.size() + dispatchChunkSize - 1) / dispatchChunkSize;
  size_t       numThreadsTotal = numThreads32 + numThreads64 + numThreads128;
  // Large molecules are drained from the shared workLarge queue inside the worker
  // loop below, which only runs numThreadsTotal iterations. When every molecule is
  // large there is no small/medium work, so without dedicated iterations the queue
  // would never be drained and those molecules would get empty fingerprints. Spread
  // the drain across the available threads in that case.
  if (numThreadsTotal == 0) {
    numThreadsTotal = std::min(workLarge.size(), static_cast<size_t>(nThreadsActual));
  }
  detail::OpenMPExceptionRegistry exceptionRegistry;

#pragma omp parallel for num_threads(nThreadsActual) default(none) shared(numThreadsTotal,     \
                                                                            threadBuffers,     \
                                                                            maxRadius,         \
                                                                            dispatchChunkSize, \
                                                                            mols,              \
                                                                            outputAccumulator, \
                                                                            workLarge,         \
                                                                            work32,            \
                                                                            work64,            \
                                                                            work128,           \
                                                                            exceptionRegistry)
  for (size_t i = 0; i < numThreadsTotal; i++) {
    try {
      std::vector<std::pair<FlatBitVect<fpSize>, int>> largeResults;

      auto&             threadCpuBuffers = threadBuffers[omp_get_thread_num()];
      const WithDevice  dev(0);
      // Do large molecules while we're waiting for the previous cycle, if possible
      const std::string rangeName =
        "LargeMolecules processing during downtime in main run thread " + std::to_string(omp_get_thread_num());
      ScopedNvtxRange range3(rangeName.c_str());
      while (cudaEventQuery(threadCpuBuffers.prevMemcpyDoneEvent.event()) == cudaErrorNotReady) {
        auto batch = workLarge.get_n(1);
        if (batch.empty()) {
          break;  // No more large work to steal during wait
        }
        const int molIdxLarge = batch[0];
        auto      fingerprint = processSingleLargeMolecule<fpSize>(*mols[molIdxLarge], maxRadius);
        largeResults.emplace_back(std::make_pair(std::move(fingerprint), molIdxLarge));
      }
      range3.pop();

      const std::string rangeName2 =
        "Wait for previous memcpy after large mol processing Main run thread " + std::to_string(omp_get_thread_num());
      ScopedNvtxRange range2(rangeName2.c_str());
      cudaCheckError(cudaEventSynchronize(threadCpuBuffers.prevMemcpyDoneEvent.event()));
      range2.pop();

      const std::string rangemainName = "Main run processing mols: thread " + std::to_string(omp_get_thread_num());
      ScopedNvtxRange   rangeMain(rangemainName.c_str());

      std::fill(threadCpuBuffers.nAtomsPerMol.begin(), threadCpuBuffers.nAtomsPerMol.end(), 0);

      ScopedNvtxRange rangeGetDispatch("Get mol ids from dispatcher");

      int thisRoundNumAtoms = 0;
      int scopedChunkSize   = 0;
      {
        if (const size_t taken = work32.copy_n(threadCpuBuffers.h_outputIndices, dispatchChunkSize); taken > 0) {
          thisRoundNumAtoms = 32;
          scopedChunkSize   = static_cast<int>(taken);
        }
      }
      if (thisRoundNumAtoms == 0) {
        // Try with 64
        if (const size_t taken = work64.copy_n(threadCpuBuffers.h_outputIndices, dispatchChunkSize); taken > 0) {
          thisRoundNumAtoms = 64;
          scopedChunkSize   = static_cast<int>(taken);
        }
      }
      if (thisRoundNumAtoms == 0) {
        // Try with 128
        if (const size_t taken = work128.copy_n(threadCpuBuffers.h_outputIndices, dispatchChunkSize); taken > 0) {
          thisRoundNumAtoms = 128;
          scopedChunkSize   = static_cast<int>(taken);
        }
      }
      rangeGetDispatch.pop();
      if (scopedChunkSize > 0) {
        std::vector<const RDKit::ROMol*> molsView;
        int                              relIdx = 0;
        ScopedNvtxRange                  rangeComputeInvars("Compute invariants");
        for (int j = 0; j < scopedChunkSize; j++) {
          const int idx = threadCpuBuffers.h_outputIndices[j];
          molsView.push_back(mols[idx]);
          const RDKit::ROMol& mol               = *mols[idx];
          threadCpuBuffers.nAtomsPerMol[relIdx] = std::int16_t(mol.getNumAtoms());
          relIdx++;
        }
        // Compute invariants directly into pinned host buffers to avoid copies
        if (thisRoundNumAtoms == 32) {
          MorganInvariantsGenerator::ComputeInvariantsInto(molsView,
                                                           thisRoundNumAtoms,
                                                           threadCpuBuffers.h_atomInvariants32.data(),
                                                           threadCpuBuffers.h_bondInvariants32.data(),
                                                           threadCpuBuffers.h_bondIndices32.data(),
                                                           threadCpuBuffers.h_bondOtherAtomIndices32.data());
        } else if (thisRoundNumAtoms == 64) {
          MorganInvariantsGenerator::ComputeInvariantsInto(molsView,
                                                           thisRoundNumAtoms,
                                                           threadCpuBuffers.h_atomInvariants64.data(),
                                                           threadCpuBuffers.h_bondInvariants64.data(),
                                                           threadCpuBuffers.h_bondIndices64.data(),
                                                           threadCpuBuffers.h_bondOtherAtomIndices64.data());
        } else {  // 128
          MorganInvariantsGenerator::ComputeInvariantsInto(molsView,
                                                           thisRoundNumAtoms,
                                                           threadCpuBuffers.h_atomInvariants128.data(),
                                                           threadCpuBuffers.h_bondInvariants128.data(),
                                                           threadCpuBuffers.h_bondIndices128.data(),
                                                           threadCpuBuffers.h_bondOtherAtomIndices128.data());
        }
        rangeComputeInvars.pop();
        ScopedNvtxRange rangeMemcpy("Memcpy to GPU");
        auto&           buffersToUse = thisRoundNumAtoms == 32 ? threadCpuBuffers.gpuBuffers32 :
                                                                 (thisRoundNumAtoms == 64 ? threadCpuBuffers.gpuBuffers64 :
                                                                                            threadCpuBuffers.gpuBuffers128);
        cudaStream_t    stream       = threadCpuBuffers.stream.stream();

        // Send using pinned host buffers
        if (thisRoundNumAtoms == 32) {
          buffersToUse->atomInvariants.copyFromHost(threadCpuBuffers.h_atomInvariants32.data(),
                                                    thisRoundNumAtoms * scopedChunkSize);
          buffersToUse->bondInvariants.copyFromHost(threadCpuBuffers.h_bondInvariants32.data(),
                                                    thisRoundNumAtoms * scopedChunkSize);
          buffersToUse->bondIndices.copyFromHost(threadCpuBuffers.h_bondIndices32.data(),
                                                 thisRoundNumAtoms * scopedChunkSize * kMaxBondsPerAtom);
          buffersToUse->bondOtherAtomIndices.copyFromHost(threadCpuBuffers.h_bondOtherAtomIndices32.data(),
                                                          thisRoundNumAtoms * scopedChunkSize * kMaxBondsPerAtom);
        } else if (thisRoundNumAtoms == 64) {
          buffersToUse->atomInvariants.copyFromHost(threadCpuBuffers.h_atomInvariants64.data(),
                                                    thisRoundNumAtoms * scopedChunkSize);
          buffersToUse->bondInvariants.copyFromHost(threadCpuBuffers.h_bondInvariants64.data(),
                                                    thisRoundNumAtoms * scopedChunkSize);
          buffersToUse->bondIndices.copyFromHost(threadCpuBuffers.h_bondIndices64.data(),
                                                 thisRoundNumAtoms * scopedChunkSize * kMaxBondsPerAtom);
          buffersToUse->bondOtherAtomIndices.copyFromHost(threadCpuBuffers.h_bondOtherAtomIndices64.data(),
                                                          thisRoundNumAtoms * scopedChunkSize * kMaxBondsPerAtom);
        } else {
          buffersToUse->atomInvariants.copyFromHost(threadCpuBuffers.h_atomInvariants128.data(),
                                                    thisRoundNumAtoms * scopedChunkSize);
          buffersToUse->bondInvariants.copyFromHost(threadCpuBuffers.h_bondInvariants128.data(),
                                                    thisRoundNumAtoms * scopedChunkSize);
          buffersToUse->bondIndices.copyFromHost(threadCpuBuffers.h_bondIndices128.data(),
                                                 thisRoundNumAtoms * scopedChunkSize * kMaxBondsPerAtom);
          buffersToUse->bondOtherAtomIndices.copyFromHost(threadCpuBuffers.h_bondOtherAtomIndices128.data(),
                                                          thisRoundNumAtoms * scopedChunkSize * kMaxBondsPerAtom);
        }

        // nAtomsPerMol and output indices
        buffersToUse->nAtomsPerMol.copyFromHost(threadCpuBuffers.nAtomsPerMol.data(), dispatchChunkSize);
        buffersToUse->outputIndices.copyFromHost(threadCpuBuffers.h_outputIndices.data(), scopedChunkSize);
        cudaCheckError(cudaEventRecord(threadCpuBuffers.prevMemcpyDoneEvent.event(), stream));
        rangeMemcpy.pop();
        solveOnGPUBatch<fpSize>(*buffersToUse,
                                outputAccumulator,
                                maxRadius,
                                thisRoundNumAtoms,
                                scopedChunkSize,
                                stream);
      }
      rangeMain.pop();
      // Take any remaining large molecules
      const std::string name =
        "LargeMolecules processing after main run thread " + std::to_string(omp_get_thread_num());
      ScopedNvtxRange range(name.c_str());
      while (true) {
        auto batch = workLarge.get_n(1);
        if (batch.empty()) {
          break;
        }
        const int molIdxLarge = batch[0];
        auto      fingerprint = processSingleLargeMolecule<fpSize>(*mols[molIdxLarge], maxRadius);
        largeResults.emplace_back(std::make_pair(std::move(fingerprint), molIdxLarge));
      }
      range.pop();

      sendLargeResultsToOutput<fpSize>(largeResults, outputAccumulator);
    } catch (...) {
      exceptionRegistry.store(std::current_exception());
    }
  }
  exceptionRegistry.rethrow();

  // Make external stream wait on all per-thread work completion
  for (const auto& buf : threadBuffers) {
    ScopedCudaEvent workDone;
    cudaCheckError(cudaEventRecord(workDone.event(), buf.stream.stream()));
    cudaCheckError(cudaStreamWaitEvent(stream, workDone.event(), 0));
  }

  return outputAccumulator;
}

template <int fpSize>
std::vector<std::unique_ptr<ExplicitBitVect>> getGpuFpAsRdkitBitVect(
  const AsyncDeviceVector<FlatBitVect<fpSize>>& outputBufferGpu,
  const int                                     numThreads) {
  std::vector<FlatBitVect<fpSize>>              resultsGpuVec(outputBufferGpu.size());
  std::vector<std::unique_ptr<ExplicitBitVect>> results;
  results.resize(outputBufferGpu.size());
  cudaCheckError(cudaDeviceSynchronize());
  extractResultsFromGPUBatch<fpSize>(outputBufferGpu, resultsGpuVec);
  cudaCheckError(cudaDeviceSynchronize());
  nvMolKit::ScopedNvtxRange range2("Populate results");
  populateResults<fpSize>(resultsGpuVec, results, numThreads);
  range2.pop();
  return results;
}

std::vector<std::unique_ptr<ExplicitBitVect>> getFingerprintsCu(const std::vector<const RDKit::ROMol*>& mols,
                                                                const std::uint32_t                     maxRadius,
                                                                const std::uint64_t                     fpSize,
                                                                const size_t                            batchSize,
                                                                const int                               numThreads,
                                                                std::vector<MorganPerThreadBuffers>&    threadBuffers) {
  if (mols.empty()) {
    return {};
  }
  // NOLINTBEGIN (cppcoreguidelines-avoid-magic-numbers)
  switch (fpSize) {
    case 4096: {
      auto gpuResult = computeFingerprintsCuImpl<4096>(mols, maxRadius, batchSize, numThreads, threadBuffers);
      return getGpuFpAsRdkitBitVect<4096>(gpuResult, numThreads);
    }

    case 2048: {
      auto gpuResult = computeFingerprintsCuImpl<2048>(mols, maxRadius, batchSize, numThreads, threadBuffers);
      return getGpuFpAsRdkitBitVect<2048>(gpuResult, numThreads);
    }
    case 1024: {
      auto gpuResult = computeFingerprintsCuImpl<1024>(mols, maxRadius, batchSize, numThreads, threadBuffers);
      return getGpuFpAsRdkitBitVect<1024>(gpuResult, numThreads);
    }
    case 512: {
      auto gpuResult = computeFingerprintsCuImpl<512>(mols, maxRadius, batchSize, numThreads, threadBuffers);
      return getGpuFpAsRdkitBitVect<512>(gpuResult, numThreads);
    }
    case 256: {
      auto gpuResult = computeFingerprintsCuImpl<256>(mols, maxRadius, batchSize, numThreads, threadBuffers);
      return getGpuFpAsRdkitBitVect<256>(gpuResult, numThreads);
    }
    case 128: {
      auto gpuResult = computeFingerprintsCuImpl<128>(mols, maxRadius, batchSize, numThreads, threadBuffers);
      return getGpuFpAsRdkitBitVect<128>(gpuResult, numThreads);
    }
    default:
      throw std::runtime_error("Unsupported fingerprint size" + std::to_string(fpSize) +
                               ", must be multiple of 2 between 128 and 4096");
  }
  // NOLINTEND
}

}  // namespace

MorganFingerprintGpuGenerator::MorganFingerprintGpuGenerator(std::uint32_t radius, std::uint32_t fpSize)
    : radius_(radius),
      fpSize_(fpSize) {}

MorganFingerprintGpuGenerator::~MorganFingerprintGpuGenerator() = default;

std::unique_ptr<ExplicitBitVect> MorganFingerprintGpuGenerator::GetFingerprint(
  const RDKit::ROMol&                      mol,
  std::optional<FingerprintComputeOptions> computeOptions) {
  std::vector<const RDKit::ROMol*> molView;
  molView.push_back(&mol);
  return std::move(GetFingerprints(molView, computeOptions)[0]);
}
std::vector<std::unique_ptr<ExplicitBitVect>> MorganFingerprintGpuGenerator::GetFingerprints(
  const std::vector<const RDKit::ROMol*>&  mols,
  std::optional<FingerprintComputeOptions> computeOptions) {
  const FingerprintComputeOptions options = computeOptions.value_or(FingerprintComputeOptions());
  return getFingerprintsCu(mols,
                           radius_,
                           fpSize_,
                           options.gpuBatchSize.value_or(kDefaultGpuBatchSize),
                           options.numCpuThreads.value_or(omp_get_max_threads()),
                           perThreadCpuBuffers_);
}

template <int nBits>
AsyncDeviceVector<FlatBitVect<nBits>> MorganFingerprintGpuGenerator::GetFingerprintsGpuBuffer(
  const std::vector<const RDKit::ROMol*>&  mols,
  cudaStream_t                             stream,
  std::optional<FingerprintComputeOptions> computeOptions) {
  const FingerprintComputeOptions options = computeOptions.value_or(FingerprintComputeOptions());
  if (options.backend != FingerprintComputeBackend::GPU) {
    throw std::runtime_error("GPU results requested but GPU backend is not selected");
  }
  if (mols.empty()) {
    return AsyncDeviceVector<FlatBitVect<nBits>>();
  }
  const size_t batchSize = options.gpuBatchSize.value_or(kDefaultGpuBatchSize);
  return computeFingerprintsCuImpl<nBits>(mols,
                                          radius_,
                                          batchSize,
                                          options.numCpuThreads.value_or(omp_get_max_threads()),
                                          perThreadCpuBuffers_,
                                          stream);
}

#define DEFINE_TEMPLATE(fpSize)                                                                                      \
  template AsyncDeviceVector<FlatBitVect<(fpSize)>> MorganFingerprintGpuGenerator::GetFingerprintsGpuBuffer<fpSize>( \
    const std::vector<const RDKit::ROMol*>&  mols,                                                                   \
    cudaStream_t                             stream,                                                                 \
    std::optional<FingerprintComputeOptions> options);
DEFINE_TEMPLATE(128)
DEFINE_TEMPLATE(256)
DEFINE_TEMPLATE(512)
DEFINE_TEMPLATE(1024)
DEFINE_TEMPLATE(2048)
DEFINE_TEMPLATE(4096)
#undef DEFINE_TEMPLATE

}  // namespace nvMolKit
