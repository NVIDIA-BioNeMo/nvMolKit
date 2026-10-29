// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cub/warp/warp_reduce.cuh>
#include <stdexcept>

#include "src/diversity_picker_algorithms.cuh"
#include "src/kmeans.h"
#include "src/kmeans_kernels.cuh"
#include "src/utils/cuda_error_check.h"

namespace nvMolKit {
namespace detail {
namespace {

constexpr int kBlockSize = 256;

int gridFor(const std::size_t count, const int blockSize = kBlockSize) {
  return static_cast<int>((count + blockSize - 1) / blockSize);
}

// ---------------------------------------------------------------------------------------------------------------------
// Row statistics and center preparation
// ---------------------------------------------------------------------------------------------------------------------

//! Squared norms of the fp16-rounded centers, which are the values the assignment kernels multiply.
__global__ void centerNormsKernel(const float* centers, float* norms, const int numCenters, const int numBits) {
  const int center = static_cast<int>(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
  const int lane   = static_cast<int>(threadIdx.x % 32);
  if (center >= numCenters) {
    return;
  }
  using WarpReduce = cub::WarpReduce<float>;
  __shared__ typename WarpReduce::TempStorage storage[kBlockSize / 32];
  float                                       sum = 0.0F;
  for (int bit = lane; bit < numBits; bit += 32) {
    const float value = __half2float(__float2half_rn(centers[static_cast<std::size_t>(center) * numBits + bit]));
    sum               = fmaf(value, value, sum);
  }
  sum = WarpReduce(storage[threadIdx.x / 32]).Sum(sum);
  if (lane == 0) {
    norms[center] = sum;
  }
}

//! Packs float centers into the MMA layout: row-major [padded center][padded bit] halves.
__global__ void packMmaCentersKernel(const float* centers,
                                     __half*      packed,
                                     const int    numCenters,
                                     const int    numBits,
                                     const int    paddedCenters,
                                     const int    paddedBits) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<std::size_t>(paddedCenters) * paddedBits) {
    return;
  }
  const int   center = static_cast<int>(index / paddedBits);
  const int   bit    = static_cast<int>(index % paddedBits);
  const float value =
    center < numCenters && bit < numBits ? centers[static_cast<std::size_t>(center) * numBits + bit] : 0.0F;
  packed[index] = __float2half_rn(value);
}

void validateFingerprints(const cuda::std::span<const std::uint32_t> fingerprints,
                          const int                                  numRows,
                          const int                                  numWords) {
  if (numRows <= 0 || numWords <= 0 ||
      fingerprints.size() != static_cast<std::size_t>(numRows) * static_cast<std::size_t>(numWords)) {
    throw std::invalid_argument("Fingerprint buffer size does not match its shape");
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Assignment
// ---------------------------------------------------------------------------------------------------------------------

//! Device copies of the centers in the layout the assignment kernel reads.
class PreparedCenters {
 public:
  PreparedCenters(const int numClusters, const int numWords, cudaStream_t stream)
      : numClusters_(numClusters),
        numBits_(numWords * 32),
        paddedCenters_((numClusters + kMmaBlockCenters - 1) / kMmaBlockCenters * kMmaBlockCenters),
        paddedBits_((numBits_ + kMmaStageBits - 1) / kMmaStageBits * kMmaStageBits),
        packed_(static_cast<std::size_t>(paddedCenters_) * paddedBits_, stream),
        norms_(numClusters, stream) {}

  void update(const float* centers, cudaStream_t stream) {
    centerNormsKernel<<<gridFor(static_cast<std::size_t>(numClusters_) * 32), kBlockSize, 0, stream>>>(centers,
                                                                                                       norms_.data(),
                                                                                                       numClusters_,
                                                                                                       numBits_);
    cudaCheckError(cudaGetLastError());
    packMmaCentersKernel<<<gridFor(packed_.size()), kBlockSize, 0, stream>>>(centers,
                                                                             packed_.data(),
                                                                             numClusters_,
                                                                             numBits_,
                                                                             paddedCenters_,
                                                                             paddedBits_);
    cudaCheckError(cudaGetLastError());
  }

  const __half* packed() const { return packed_.data(); }
  const float*  norms() const { return norms_.data(); }
  int           paddedBits() const { return paddedBits_; }

 private:
  int                       numClusters_;
  int                       numBits_;
  int                       paddedCenters_;
  int                       paddedBits_;
  AsyncDeviceVector<__half> packed_;
  AsyncDeviceVector<float>  norms_;
};

template <KMeansMetric Metric>
void launchAssign(const std::uint32_t*   fingerprints,
                  const int*             rowCounts,
                  const int              numRows,
                  const int              numWords,
                  const PreparedCenters& centers,
                  const int              numClusters,
                  int*                   labels,
                  float*                 distances,
                  int*                   changed,
                  cudaStream_t           stream) {
  const AssignArgs args{fingerprints,
                        rowCounts,
                        numRows,
                        numWords,
                        centers.packed(),
                        centers.norms(),
                        numClusters,
                        labels,
                        distances,
                        changed};
  launchMmaAssign<Metric>(args, centers.paddedBits(), stream);
}

void assign(const KMeansMetric     metric,
            const std::uint32_t*   fingerprints,
            const int*             rowCounts,
            const int              numRows,
            const int              numWords,
            const PreparedCenters& centers,
            const int              numClusters,
            int*                   labels,
            float*                 distances,
            int*                   changed,
            cudaStream_t           stream) {
  switch (metric) {
    case KMeansMetric::Cosine:
      launchAssign<KMeansMetric::Cosine>(fingerprints,
                                         rowCounts,
                                         numRows,
                                         numWords,
                                         centers,
                                         numClusters,
                                         labels,
                                         distances,
                                         changed,
                                         stream);
      return;
    case KMeansMetric::Tanimoto:
      launchAssign<KMeansMetric::Tanimoto>(fingerprints,
                                           rowCounts,
                                           numRows,
                                           numWords,
                                           centers,
                                           numClusters,
                                           labels,
                                           distances,
                                           changed,
                                           stream);
      return;
  }
  throw std::invalid_argument("Unsupported k-means metric");
}

}  // namespace
}  // namespace detail

void kmeansAssignGpu(const cuda::std::span<const std::uint32_t> fingerprints,
                     const int                                  numFingerprints,
                     const int                                  numWords,
                     const float*                               centers,
                     const int                                  numClusters,
                     const KMeansMetric                         metric,
                     int*                                       labels,
                     float*                                     distances,
                     cudaStream_t                               stream) {
  detail::validateFingerprints(fingerprints, numFingerprints, numWords);
  if (numClusters < 1) {
    throw std::invalid_argument("n_clusters must be positive");
  }
  AsyncDeviceVector<int>  rowCounts(numFingerprints, stream);
  AsyncDevicePtr<int>     changed(0, stream);
  detail::PreparedCenters prepared(numClusters, numWords, stream);
  detail::launchFingerprintBitCounts(fingerprints.data(), rowCounts.data(), numFingerprints, numWords, stream);
  prepared.update(centers, stream);
  cudaCheckError(cudaMemsetAsync(labels, 0xFF, sizeof(int) * numFingerprints, stream));
  detail::assign(metric,
                 fingerprints.data(),
                 rowCounts.data(),
                 numFingerprints,
                 numWords,
                 prepared,
                 numClusters,
                 labels,
                 distances,
                 changed.data(),
                 stream);
  cudaCheckError(cudaStreamSynchronize(stream));
}

}  // namespace nvMolKit
