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

#include "src/similarity.h"

#include <omp.h>

#include "src/similarity_kernels.h"
#include "src/utils/device.h"
#include "src/utils/device_vector.h"
#include "src/utils/host_vector.h"
#include "src/utils/nvtx.h"
#include "src/utils/openmp_helpers.h"
namespace nvMolKit {

using internal::kBlockType;

namespace {
constexpr int kBitsPerByte = 8;
}  // namespace

// --------------------------------
// Tanimoto similarity wrapper functions
// --------------------------------

AsyncDeviceVector<double> crossTanimotoSimilarityGpuResult(const cuda::std::span<const std::uint32_t> bits,
                                                           int                                        fpSize,
                                                           cudaStream_t                               stream) {
  const size_t nElementsPerFp = fpSize / (kBitsPerByte * sizeof(std::uint32_t));
  const size_t nFps           = bits.size() / nElementsPerFp;
  auto         similarities_d = AsyncDeviceVector<double>(nFps * nFps, stream);
  launchCrossTanimotoSimilarity(bits, bits, nElementsPerFp, toSpan(similarities_d), 0, stream);
  return similarities_d;
}

AsyncDeviceVector<double> crossTanimotoSimilarityGpuResult(const cuda::std::span<const std::uint32_t> bitsOneBuffer,
                                                           const cuda::std::span<const std::uint32_t> bitsTwoBuffer,
                                                           int                                        fpSize,
                                                           cudaStream_t                               stream) {
  const size_t nElementsPerFp = fpSize / (kBitsPerByte * sizeof(std::uint32_t));
  const size_t nFps1          = bitsOneBuffer.size() / nElementsPerFp;
  const size_t nFps2          = bitsTwoBuffer.size() / nElementsPerFp;
  auto         similarities_d = AsyncDeviceVector<double>(nFps1 * nFps2, stream);
  launchCrossTanimotoSimilarity(bitsOneBuffer, bitsTwoBuffer, nElementsPerFp, toSpan(similarities_d), 0, stream);
  return similarities_d;
}

struct SimilaritiesRotBuffers {
  // Declare streams before buffers so buffers are destroyed before streams.
  ScopedStream              streamA;
  ScopedStream              streamB;
  AsyncDeviceVector<double> bufferA;
  AsyncDeviceVector<double> bufferB;

  // 100MB pinned host transfer buffers, double-buffered per thread index
  PinnedHostVector<double> pinnedA0;
  PinnedHostVector<double> pinnedA1;
  PinnedHostVector<double> pinnedB0;
  PinnedHostVector<double> pinnedB1;

  // CUDA events per pinned buffer to signal when D2H completes
  ScopedCudaEvent eventA0;
  ScopedCudaEvent eventA1;
  ScopedCudaEvent eventB0;
  ScopedCudaEvent eventB1;

  [[nodiscard]] cudaStream_t streamForThread(int tid) const {
    return (tid % 2 == 0) ? streamA.stream() : streamB.stream();
  }
  AsyncDeviceVector<double>& bufferForThread(int tid) { return (tid % 2 == 0) ? bufferA : bufferB; }
  PinnedHostVector<double>&  pinnedForThreadBuffer(int tid, int bufIdx) {
    const bool isA = (tid % 2 == 0);
    if (isA) {
      return (bufIdx == 0) ? pinnedA0 : pinnedA1;
    }
    return (bufIdx == 0) ? pinnedB0 : pinnedB1;
  }
  [[nodiscard]] cudaEvent_t eventForThreadBuffer(int tid, unsigned int bufIdx) const {
    const bool isA = (tid % 2 == 0);
    if (isA) {
      return (bufIdx == 0) ? eventA0.event() : eventA1.event();
    }
    return (bufIdx == 0) ? eventB0.event() : eventB1.event();
  }
};

namespace {
// Shared implementation for cross similarity on CPU with GPU offload and chunked D2H.
// The provided launchKernel lambda is responsible for dispatching the correct GPU kernel
// (e.g., Tanimoto or Cosine) with the signature:
//   launchKernel(bitsA, bitsB, nElementsPerFp, outSpan, /*tile*/0, stream)
template <typename LaunchKernelFn>
std::vector<double> crossSimilarityImpl(const cuda::std::span<const std::uint32_t> bitsOneBuffer,
                                        const cuda::std::span<const std::uint32_t> bitsTwoBuffer,
                                        const int                                  fpSize,
                                        const CrossSimilarityOptions&              options,
                                        LaunchKernelFn                             launchKernel,
                                        cudaStream_t                               inputStream) {
  const size_t nElementsPerFp = fpSize / (kBitsPerByte * sizeof(std::uint32_t));
  const size_t nFps1          = bitsOneBuffer.size() / nElementsPerFp;
  const size_t nFps2          = bitsTwoBuffer.size() / nElementsPerFp;
  const size_t freeBytes      = options.maxDeviceMemoryBytes.value_or(getDeviceFreeMemory());

  ScopedCudaEvent inputsReady;
  cudaCheckError(cudaEventRecord(inputsReady.event(), inputStream));

  // If there is enough memory, compute in one shot on the device then copy to host
  if (freeBytes >= nFps1 * nFps2 * sizeof(double)) {
    ScopedStream stream;
    cudaCheckError(cudaStreamWaitEvent(stream.stream(), inputsReady.event()));
    AsyncDeviceVector<double> similarities_d(nFps1 * nFps2, stream.stream());
    launchKernel(bitsOneBuffer, bitsTwoBuffer, nElementsPerFp, toSpan(similarities_d), 0, stream.stream());
    std::vector<double> res(similarities_d.size());
    similarities_d.copyToHost(res);
    cudaStreamSynchronize(stream.stream());
    return res;
  }

  // Otherwise, compute in batches with two streams and double-buffered pinned D2H per stream
  const size_t     freeBytesPerBuffer = freeBytes / 2;  // We need space for two device buffers
  const size_t     maxDoublesInBuffer = (freeBytesPerBuffer / sizeof(double) * 9) / 10;  // Leave ~10% headroom
  constexpr size_t minBatchSizeA      = 32;
  const size_t     blockSizeIncrement = minBatchSizeA * nFps2;
  if (blockSizeIncrement > maxDoublesInBuffer) {
    throw std::runtime_error("Not enough memory to compute cross similarity");
  }
  const size_t batchSizeA = std::max(minBatchSizeA, (maxDoublesInBuffer / blockSizeIncrement) * minBatchSizeA);

  // Prepare per-thread resources
  SimilaritiesRotBuffers rotBuffers;
  cudaCheckError(cudaStreamWaitEvent(rotBuffers.streamA.stream(), inputsReady.event()));
  cudaCheckError(cudaStreamWaitEvent(rotBuffers.streamB.stream(), inputsReady.event()));
  rotBuffers.bufferA = AsyncDeviceVector<double>(batchSizeA * nFps2, rotBuffers.streamA.stream());
  rotBuffers.bufferB = AsyncDeviceVector<double>(batchSizeA * nFps2, rotBuffers.streamB.stream());

  // 100 MB pinned buffers per thread (double-buffered)
  constexpr size_t kPinnedBytes          = 100ULL * 1024ULL * 1024ULL;
  constexpr size_t pinnedCapacityDoubles = kPinnedBytes / sizeof(double);
  rotBuffers.pinnedA0                    = PinnedHostVector<double>(pinnedCapacityDoubles);
  rotBuffers.pinnedA1                    = PinnedHostVector<double>(pinnedCapacityDoubles);
  rotBuffers.pinnedB0                    = PinnedHostVector<double>(pinnedCapacityDoubles);
  rotBuffers.pinnedB1                    = PinnedHostVector<double>(pinnedCapacityDoubles);

  std::vector<double> res(nFps1 * nFps2);

  const size_t                    nBatches = (nFps1 + batchSizeA - 1) / batchSizeA;
  detail::OpenMPExceptionRegistry exceptionHandler;
#pragma omp parallel for num_threads(2) schedule(dynamic) default(shared)
  for (size_t batchIdx = 0; batchIdx < nBatches; ++batchIdx) {
    try {
      const ScopedNvtxRange      batchRange("CrossSimilarity: batch");
      const int                  tid           = omp_get_thread_num();
      cudaStream_t               currentStream = rotBuffers.streamForThread(tid);
      AsyncDeviceVector<double>& currentBuffer = rotBuffers.bufferForThread(tid);
      // Double-buffered pinned buffers for overlap
      PinnedHostVector<double>&  pinnedBuffer0 = rotBuffers.pinnedForThreadBuffer(tid, 0);
      PinnedHostVector<double>&  pinnedBuffer1 = rotBuffers.pinnedForThreadBuffer(tid, 1);

      const size_t startIdx          = batchIdx * batchSizeA;
      const size_t currentBatchSizeA = std::min(batchSizeA, nFps1 - startIdx);

      // Launch compute for this batch
      ScopedNvtxRange launchRange("CrossSimilarity: launch kernel");
      launchKernel(bitsOneBuffer.subspan(startIdx * nElementsPerFp, currentBatchSizeA * nElementsPerFp),
                   bitsTwoBuffer,
                   nElementsPerFp,
                   toSpan(currentBuffer),
                   0,
                   currentStream);
      launchRange.pop();

      // Double-buffered chunked D2H into pinned buffers with overlap of CPU memcpy
      const size_t batchElemsTotal = currentBatchSizeA * nFps2;
      unsigned int bufIdx          = 0;  // 0 -> use pinnedBuffer0, 1 -> pinnedBuffer1
      bool         hasPrev         = false;
      size_t       prevOffset      = 0;
      size_t       prevSize        = 0;
      int          prevBufIdx      = 0;
      for (size_t chunkOffset = 0; chunkOffset < batchElemsTotal; chunkOffset += pinnedCapacityDoubles, bufIdx ^= 1U) {
        const size_t chunkSize  = std::min(pinnedCapacityDoubles, batchElemsTotal - chunkOffset);
        auto&        pinnedBuf  = (bufIdx == 0) ? pinnedBuffer0 : pinnedBuffer1;
        auto&        prevPinned = (bufIdx == 0) ? pinnedBuffer1 : pinnedBuffer0;
        cudaEvent_t  bufEvent   = rotBuffers.eventForThreadBuffer(tid, bufIdx);
        cudaEvent_t  prevEvent  = rotBuffers.eventForThreadBuffer(tid, prevBufIdx);

        // Enqueue D2H into the selected pinned buffer and record event
        ScopedNvtxRange d2hRange("CrossSimilarity: D2H chunk");
        const size_t    deviceOffset = chunkOffset;
        currentBuffer.copyToHost(pinnedBuf.data(), chunkSize, 0, deviceOffset);
        cudaEventRecord(bufEvent, currentStream);
        d2hRange.pop();

        // While this D2H is running, memcpy from the previous buffer (if not first iteration)
        if (hasPrev) {
          ScopedNvtxRange memcpyPrevRange("CrossSimilarity: memcpy prev chunk");
          const size_t    resOffset = (startIdx * nFps2) + prevOffset;
          cudaEventSynchronize(prevEvent);
          std::memcpy(res.data() + resOffset, prevPinned.data(), sizeof(double) * prevSize);
          memcpyPrevRange.pop();
        }

        // Set current as previous for next iteration
        hasPrev    = true;
        prevOffset = chunkOffset;
        prevSize   = chunkSize;
        prevBufIdx = bufIdx;
      }

      // Copy the last in-flight chunk
      if (hasPrev) {
        ScopedNvtxRange memcpyLastRange("CrossSimilarity: memcpy last chunk");
        auto&           lastPinned = (prevBufIdx == 0) ? pinnedBuffer0 : pinnedBuffer1;
        cudaEvent_t     lastEvent  = rotBuffers.eventForThreadBuffer(tid, prevBufIdx);
        cudaEventSynchronize(lastEvent);
        const size_t resOffset = (startIdx * nFps2) + prevOffset;
        std::memcpy(res.data() + resOffset, lastPinned.data(), sizeof(double) * prevSize);
        memcpyLastRange.pop();
      }

      // Ensure all D2H copies are complete before proceeding
      ScopedNvtxRange syncRange("CrossSimilarity: stream sync");
      cudaStreamSynchronize(currentStream);
      syncRange.pop();
    } catch (...) {
      exceptionHandler.store(std::current_exception());
    }
  }
  exceptionHandler.rethrow();

  return res;
}
}  // namespace

std::vector<double> crossTanimotoSimilarityCPUResult(const cuda::std::span<const std::uint32_t> bitsOneBuffer,
                                                     const cuda::std::span<const std::uint32_t> bitsTwoBuffer,
                                                     const int                                  fpSize,
                                                     const CrossSimilarityOptions&              options,
                                                     cudaStream_t                               stream) {
  return crossSimilarityImpl(
    bitsOneBuffer,
    bitsTwoBuffer,
    fpSize,
    options,
    [](const cuda::std::span<const std::uint32_t> bufA,
       const cuda::std::span<const std::uint32_t> bufB,
       size_t                                     nElementsPerFp,
       const cuda::std::span<double>              out,
       int /*tile*/,
       cudaStream_t stream) { launchCrossTanimotoSimilarity(bufA, bufB, nElementsPerFp, out, 0, stream); },
    stream);
}

// --------------------------------
// Cosine similarity wrapper functions
// --------------------------------

AsyncDeviceVector<double> crossCosineSimilarityGpuResult(const cuda::std::span<const std::uint32_t> bits,
                                                         int                                        fpSize,
                                                         cudaStream_t                               stream) {
  const size_t nElementsPerFp = fpSize / (kBitsPerByte * sizeof(std::uint32_t));
  const size_t nFps           = bits.size() / nElementsPerFp;
  auto         similarities_d = AsyncDeviceVector<double>(nFps * nFps, stream);
  launchCrossCosineSimilarity(bits, bits, nElementsPerFp, toSpan(similarities_d), 0, stream);
  return similarities_d;
}

AsyncDeviceVector<double> crossCosineSimilarityGpuResult(const cuda::std::span<const std::uint32_t> bitsOneBuffer,
                                                         const cuda::std::span<const std::uint32_t> bitsTwoBuffer,
                                                         int                                        fpSize,
                                                         cudaStream_t                               stream) {
  const size_t nElementsPerFp = fpSize / (kBitsPerByte * sizeof(std::uint32_t));
  const size_t nFps1          = bitsOneBuffer.size() / nElementsPerFp;
  const size_t nFps2          = bitsTwoBuffer.size() / nElementsPerFp;
  auto         similarities_d = AsyncDeviceVector<double>(nFps1 * nFps2, stream);
  launchCrossCosineSimilarity(bitsOneBuffer, bitsTwoBuffer, nElementsPerFp, toSpan(similarities_d), 0, stream);
  return similarities_d;
}

std::vector<double> crossCosineSimilarityCPUResult(const cuda::std::span<const std::uint32_t> bitsOneBuffer,
                                                   const cuda::std::span<const std::uint32_t> bitsTwoBuffer,
                                                   const int                                  fpSize,
                                                   const CrossSimilarityOptions&              options,
                                                   cudaStream_t                               stream) {
  return crossSimilarityImpl(
    bitsOneBuffer,
    bitsTwoBuffer,
    fpSize,
    options,
    [](const cuda::std::span<const std::uint32_t> bufA,
       const cuda::std::span<const std::uint32_t> bufB,
       size_t                                     nElementsPerFp,
       const cuda::std::span<double>              out,
       int /*tile*/,
       cudaStream_t stream) { launchCrossCosineSimilarity(bufA, bufB, nElementsPerFp, out, 0, stream); },
    stream);
}

}  // namespace nvMolKit
