// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "src/diversity_picker_algorithms.cuh"
#include "src/diversity_pickers.h"
#include "src/fingerprint_similarity_device.cuh"
#include "src/kmeans.h"
#include "src/kmeans_kernels.cuh"
#include "src/utils/cub_helpers.cuh"
#include "src/utils/cuda_error_check.h"
#include "src/utils/nvtx.h"

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

//! One warp per center: copies fingerprint row rows[center] into float centers, L2-normalized for cosine.
__global__ void centersFromRowsKernel(const std::uint32_t* fingerprints,
                                      const int*           rows,
                                      float*               centers,
                                      const int            numCenters,
                                      const int            numWords,
                                      const bool           normalize) {
  const int center = static_cast<int>(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
  const int lane   = static_cast<int>(threadIdx.x % 32);
  if (center >= numCenters) {
    return;
  }
  const std::uint32_t* row   = fingerprints + static_cast<std::size_t>(rows[center]) * numWords;
  int                  count = 0;
  for (int word = lane; word < numWords; word += 32) {
    count += __popc(row[word]);
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    count += __shfl_xor_sync(0xFFFFFFFFU, count, offset);
  }
  const float value = normalize ? (count > 0 ? __frsqrt_rn(static_cast<float>(count)) : 0.0F) : 1.0F;
  float*      out   = centers + static_cast<std::size_t>(center) * numWords * 32;
  for (int bit = lane; bit < numWords * 32; bit += 32) {
    out[bit] = (row[bit / 32] >> (bit % 32) & 1U) != 0U ? value : 0.0F;
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Update step
// ---------------------------------------------------------------------------------------------------------------------

//! Fixed-point scale for cosine row weights, so weighted sums are exact integer additions and deterministic.
constexpr double kWeightScale = 4294967296.0;

template <bool Weighted> using SumType = std::conditional_t<Weighted, unsigned long long, unsigned int>;

template <bool Weighted> __device__ __forceinline__ SumType<Weighted> rowIncrement(const int rowCount) {
  if constexpr (Weighted) {
    return rowCount > 0 ? static_cast<unsigned long long>(llrint(kWeightScale / sqrt(static_cast<double>(rowCount)))) :
                          0ULL;
  } else {
    return 1U;
  }
}

//! One thread per row: atomically adds the row's (weighted) bits to its cluster's sums.
template <bool Weighted>
__global__ void accumulateGlobalKernel(const std::uint32_t* fingerprints,
                                       const int*           rowCounts,
                                       const int*           labels,
                                       const int            numRows,
                                       const int            numWords,
                                       SumType<Weighted>*   sums,
                                       unsigned int*        sizes) {
  const auto row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= static_cast<std::size_t>(numRows)) {
    return;
  }
  const int  label     = labels[row];
  const auto increment = rowIncrement<Weighted>(rowCounts[row]);
  atomicAdd(&sizes[label], 1U);
  auto* clusterSums = sums + static_cast<std::size_t>(label) * numWords * 32;
  for (int word = 0; word < numWords; ++word) {
    std::uint32_t bits = fingerprints[row * numWords + word];
    while (bits != 0U) {
      const int bit = __ffs(static_cast<int>(bits)) - 1;
      bits &= bits - 1U;
      atomicAdd(&clusterSums[word * 32 + bit], increment);
    }
  }
}

//! Centers become the means of their members. Clusters without members are set to zero here and reseeded after.
//! sums may alias centers (unweighted counts are accumulated in place).
template <bool Weighted>
__global__ void finalizeCentersKernel(const SumType<Weighted>* sums,
                                      const unsigned int*      sizes,
                                      float*                   centers,
                                      const int                numClusters,
                                      const int                numBits) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<std::size_t>(numClusters) * numBits) {
    return;
  }
  const unsigned int size = sizes[index / numBits];
  if (size == 0U) {
    centers[index] = 0.0F;
    return;
  }
  if constexpr (Weighted) {
    centers[index] = static_cast<float>(static_cast<double>(sums[index]) / kWeightScale / size);
  } else {
    centers[index] = static_cast<float>(static_cast<double>(sums[index]) / size);
  }
}

__global__ void scatterRowIdsKernel(int* rowIds, const int numRows) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row < numRows) {
    rowIds[row] = row;
  }
}

__global__ void histogramKernel(const int* labels, const int numRows, std::int64_t* sizes) {
  const auto row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < static_cast<std::size_t>(numRows)) {
    atomicAdd(reinterpret_cast<unsigned long long*>(&sizes[labels[row]]), 1ULL);
  }
}

//! Widens the float row distances so the objective is summed in double precision.
struct FloatToDouble {
  __device__ double operator()(const float value) const { return value; }
};

//! Largest number of empty clusters reseeded farthest-first; beyond it the farthest rows are taken in order.
constexpr int kMaxSpreadReseeds = 256;
constexpr int kSpreadThreads    = 1024;

//! The metric's k-means distance between two fingerprint rows, in the units of the assignment distances.
template <KMeansMetric Metric>
__device__ __forceinline__ float rowPairDistance(const int intersection, const int lhsCount, const int rhsCount) {
  if constexpr (Metric == KMeansMetric::Cosine) {
    // Squared Euclidean distance between the L2-normalized rows; an empty row is the zero vector.
    if (lhsCount == 0 || rhsCount == 0) {
      return lhsCount == rhsCount ? 0.0F : 1.0F;
    }
    return 2.0F * fingerprintDistance<FingerprintSimilarityMetric::Cosine>(intersection, lhsCount, rhsCount);
  } else {
    return fingerprintDistance<FingerprintSimilarityMetric::Tanimoto>(intersection, lhsCount, rhsCount);
  }
}

/**
 * Farthest-first choice of count rows among the pool of candidates (the rows farthest from their centers, with those
 * distances): each pick maximizes the smaller of its distance to its own center and to every earlier pick, so several
 * empty clusters do not land in one far region. Ties go to the earlier candidate. One block runs the whole selection;
 * distances is overwritten.
 */
template <KMeansMetric Metric>
__global__ void __launch_bounds__(kSpreadThreads) spreadReseedsKernel(const std::uint32_t* fingerprints,
                                                                      const int*           bitCounts,
                                                                      const int            numWords,
                                                                      const int*           candidates,
                                                                      float*               distances,
                                                                      const int            pool,
                                                                      const int            count,
                                                                      int*                 picks) {
  using Pair        = cub::KeyValuePair<int, float>;
  using BlockArgMax = cub::BlockReduce<Pair, kSpreadThreads>;
  __shared__ typename BlockArgMax::TempStorage storage;
  __shared__ int                               chosen;
  for (int pick = 0; pick < count; ++pick) {
    Pair best(pool, -INFINITY);
    for (int index = static_cast<int>(threadIdx.x); index < pool; index += kSpreadThreads) {
      if (distances[index] > best.value) {
        best = Pair(index, distances[index]);
      }
    }
    best = BlockArgMax(storage).Reduce(best, cub::ArgMax());
    if (threadIdx.x == 0) {
      chosen      = best.key;
      picks[pick] = candidates[best.key];
    }
    __syncthreads();
    const int            pickRow   = candidates[chosen];
    const std::uint32_t* pickWords = fingerprints + static_cast<std::size_t>(pickRow) * numWords;
    for (int index = static_cast<int>(threadIdx.x); index < pool; index += kSpreadThreads) {
      const int            row          = candidates[index];
      const std::uint32_t* words        = fingerprints + static_cast<std::size_t>(row) * numWords;
      int                  intersection = 0;
      for (int word = 0; word < numWords; ++word) {
        intersection += __popc(words[word] & pickWords[word]);
      }
      const float distance = rowPairDistance<Metric>(intersection, bitCounts[row], bitCounts[pickRow]);
      distances[index]     = index == chosen ? -1.0F : fminf(distances[index], distance);
    }
    __syncthreads();
  }
}

std::vector<int> sampleDistinctRows(const int numRows, const int numClusters, const std::uint64_t seed) {
  // Floyd's algorithm: numClusters distinct values in [0, numRows) in O(numClusters).
  std::mt19937_64         generator(seed);
  std::unordered_set<int> chosen;
  chosen.reserve(static_cast<std::size_t>(numClusters) * 2);
  std::vector<int> rows;
  rows.reserve(numClusters);
  for (int upper = numRows - numClusters; upper < numRows; ++upper) {
    const int candidate = static_cast<int>(generator() % (static_cast<std::uint64_t>(upper) + 1));
    const int value     = chosen.insert(candidate).second ? candidate : upper;
    if (value == upper) {
      chosen.insert(upper);
    }
    rows.push_back(value);
  }
  std::shuffle(rows.begin(), rows.end(), generator);
  return rows;
}

void validateFingerprints(const cuda::std::span<const std::uint32_t> fingerprints,
                          const int                                  numRows,
                          const int                                  numWords) {
  if (numRows <= 0 || numWords <= 0 ||
      fingerprints.size() != static_cast<std::size_t>(numRows) * static_cast<std::size_t>(numWords)) {
    throw std::invalid_argument("Fingerprint buffer size does not match its shape");
  }
}

void validateClusterCount(const int numClusters, const int numRows) {
  if (numClusters < 1 || numClusters > numRows) {
    throw std::invalid_argument("n_clusters must be between 1 and the number of fingerprints");
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

// ---------------------------------------------------------------------------------------------------------------------
// Lloyd driver
// ---------------------------------------------------------------------------------------------------------------------

class LloydState {
 public:
  LloydState(const cuda::std::span<const std::uint32_t> fingerprints,
             const int                                  numRows,
             const int                                  numWords,
             const KMeansOptions&                       options,
             AsyncDeviceVector<float>&&                 centers,
             cudaStream_t                               stream)
      : fingerprints_(fingerprints.data()),
        numRows_(numRows),
        numWords_(numWords),
        numBits_(numWords * 32),
        options_(options),
        stream_(stream),
        rowCounts_(numRows, stream),
        labels_(numRows, stream),
        distances_(numRows, stream),
        centers_(std::move(centers)),
        prepared_(options.numClusters, numWords, stream),
        sizes_(options.numClusters, stream),
        inertia_(0.0, stream),
        changed_(0, stream) {
    launchFingerprintBitCounts(fingerprints_, rowCounts_.data(), numRows, numWords, stream);
    std::size_t reduceBytes = 0;
    cudaCheckError(cub::DeviceReduce::TransformReduce(nullptr,
                                                      reduceBytes,
                                                      distances_.data(),
                                                      inertia_.data(),
                                                      numRows,
                                                      cubSum(),
                                                      FloatToDouble(),
                                                      0.0,
                                                      stream));
    reduceTemp_ = AsyncDeviceVector<std::uint8_t>(reduceBytes, stream);
    cudaCheckError(cudaMemsetAsync(labels_.data(), 0xFF, sizeof(int) * numRows, stream));
    // Unweighted bit counts are accumulated in place in the centers buffer (uint32 then converted to float), so only
    // the cosine metric's 64-bit fixed-point sums need a separate buffer.
    if (options.metric == KMeansMetric::Cosine) {
      sums_ = AsyncDeviceVector<std::uint8_t>(centers_.size() * sizeof(unsigned long long), stream);
    }
  }

  //! Assigns rows to the current centers. Returns the objective and the number of changed labels.
  std::pair<double, int> assignPass() {
    prepared_.update(centers_.data(), stream_);
    changed_.memSet(0);
    assign(options_.metric,
           fingerprints_,
           rowCounts_.data(),
           numRows_,
           numWords_,
           prepared_,
           options_.numClusters,
           labels_.data(),
           distances_.data(),
           changed_.data(),
           stream_);
    std::size_t reduceBytes = reduceTemp_.size();
    cudaCheckError(cub::DeviceReduce::TransformReduce(reduceTemp_.data(),
                                                      reduceBytes,
                                                      distances_.data(),
                                                      inertia_.data(),
                                                      numRows_,
                                                      cubSum(),
                                                      FloatToDouble(),
                                                      0.0,
                                                      stream_));
    double inertia = 0.0;
    int    changed = 0;
    inertia_.get(inertia);
    changed_.get(changed);
    cudaCheckError(cudaStreamSynchronize(stream_));
    return {inertia, changed};
  }

  //! Recomputes centers as member means, then reseeds empty clusters. Returns the number reseeded.
  int updatePass() {
    if (options_.metric == KMeansMetric::Cosine) {
      accumulate<true>();
    } else {
      accumulate<false>();
    }
    return reseedEmpty();
  }

  KMeansFitResult finish(const double inertia, const int numIterations) {
    KMeansFitResult result;
    result.inertia       = inertia;
    result.numIterations = numIterations;
    result.clusterSizes  = AsyncDeviceVector<std::int64_t>(options_.numClusters, stream_);
    result.clusterSizes.zero();
    histogramKernel<<<gridFor(numRows_), kBlockSize, 0, stream_>>>(labels_.data(),
                                                                   numRows_,
                                                                   result.clusterSizes.data());
    cudaCheckError(cudaGetLastError());
    result.labels  = std::move(labels_);
    result.centers = std::move(centers_);
    return result;
  }

 private:
  template <bool Weighted> void accumulate() {
    using Sum  = SumType<Weighted>;
    auto* sums = Weighted ? reinterpret_cast<Sum*>(sums_.data()) : reinterpret_cast<Sum*>(centers_.data());
    cudaCheckError(cudaMemsetAsync(sums, 0, centers_.size() * sizeof(Sum), stream_));
    sizes_.zero();
    accumulateGlobalKernel<Weighted><<<gridFor(numRows_), kBlockSize, 0, stream_>>>(fingerprints_,
                                                                                    rowCounts_.data(),
                                                                                    labels_.data(),
                                                                                    numRows_,
                                                                                    numWords_,
                                                                                    sums,
                                                                                    sizes_.data());
    cudaCheckError(cudaGetLastError());
    finalizeCentersKernel<Weighted><<<gridFor(centers_.size()), kBlockSize, 0, stream_>>>(sums,
                                                                                          sizes_.data(),
                                                                                          centers_.data(),
                                                                                          options_.numClusters,
                                                                                          numBits_);
    cudaCheckError(cudaGetLastError());
  }

  //! Writes count reseed rows to picks: farthest-first among a pool of the farthest rows, or, above
  //! kMaxSpreadReseeds empty clusters, simply the count farthest rows.
  void chooseReseeds(const int* sortedIds, float* sortedDistances, const int count, int* picks) const {
    const int pool = count <= kMaxSpreadReseeds ? std::min(numRows_, 16 * count + 64) : count;
    if (pool == count) {
      cudaCheckError(cudaMemcpyAsync(picks, sortedIds, sizeof(int) * count, cudaMemcpyDeviceToDevice, stream_));
      return;
    }
    auto kernel = options_.metric == KMeansMetric::Cosine ? spreadReseedsKernel<KMeansMetric::Cosine> :
                                                            spreadReseedsKernel<KMeansMetric::Tanimoto>;
    kernel<<<1, kSpreadThreads, 0, stream_>>>(fingerprints_,
                                              rowCounts_.data(),
                                              numWords_,
                                              sortedIds,
                                              sortedDistances,
                                              pool,
                                              count,
                                              picks);
    cudaCheckError(cudaGetLastError());
  }

  /**
   * Moves each empty cluster's center onto one of the rows farthest from their current center, farthest first (as
   * scikit-learn does). The next assignment pass then gives the cluster at least that row unless it duplicates
   * another center.
   */
  int reseedEmpty() {
    std::vector<unsigned int> sizes(options_.numClusters);
    sizes_.copyToHost(sizes);
    cudaCheckError(cudaStreamSynchronize(stream_));
    std::vector<int> empty;
    for (int cluster = 0; cluster < options_.numClusters; ++cluster) {
      if (sizes[cluster] == 0U) {
        empty.push_back(cluster);
      }
    }
    if (empty.empty()) {
      return 0;
    }
    AsyncDeviceVector<int>   rowIds(numRows_, stream_);
    AsyncDeviceVector<int>   sortedIds(numRows_, stream_);
    AsyncDeviceVector<float> sortedDistances(numRows_, stream_);
    scatterRowIdsKernel<<<gridFor(numRows_), kBlockSize, 0, stream_>>>(rowIds.data(), numRows_);
    cudaCheckError(cudaGetLastError());
    std::size_t tempBytes = 0;
    cudaCheckError(cub::DeviceRadixSort::SortPairsDescending(nullptr,
                                                             tempBytes,
                                                             distances_.data(),
                                                             sortedDistances.data(),
                                                             rowIds.data(),
                                                             sortedIds.data(),
                                                             numRows_,
                                                             0,
                                                             32,
                                                             stream_));
    AsyncDeviceVector<std::uint8_t> temp(tempBytes, stream_);
    cudaCheckError(cub::DeviceRadixSort::SortPairsDescending(temp.data(),
                                                             tempBytes,
                                                             distances_.data(),
                                                             sortedDistances.data(),
                                                             rowIds.data(),
                                                             sortedIds.data(),
                                                             numRows_,
                                                             0,
                                                             32,
                                                             stream_));
    const int              count = static_cast<int>(empty.size());
    AsyncDeviceVector<int> rows(count, stream_);
    chooseReseeds(sortedIds.data(), sortedDistances.data(), count, rows.data());
    // Rows sharing the same centers buffer row: scatter one center row per reseeded cluster.
    AsyncDeviceVector<float> fresh(static_cast<std::size_t>(count) * numBits_, stream_);
    centersFromRowsKernel<<<gridFor(static_cast<std::size_t>(count) * 32), kBlockSize, 0, stream_>>>(
      fingerprints_,
      rows.data(),
      fresh.data(),
      count,
      numWords_,
      options_.metric == KMeansMetric::Cosine);
    cudaCheckError(cudaGetLastError());
    for (int index = 0; index < count; ++index) {
      cudaCheckError(cudaMemcpyAsync(centers_.data() + static_cast<std::size_t>(empty[index]) * numBits_,
                                     fresh.data() + static_cast<std::size_t>(index) * numBits_,
                                     sizeof(float) * numBits_,
                                     cudaMemcpyDeviceToDevice,
                                     stream_));
    }
    return count;
  }

  const std::uint32_t*            fingerprints_;
  int                             numRows_;
  int                             numWords_;
  int                             numBits_;
  KMeansOptions                   options_;
  cudaStream_t                    stream_;
  AsyncDeviceVector<int>          rowCounts_;
  AsyncDeviceVector<int>          labels_;
  AsyncDeviceVector<float>        distances_;
  AsyncDeviceVector<float>        centers_;
  PreparedCenters                 prepared_;
  AsyncDeviceVector<std::uint8_t> sums_;
  AsyncDeviceVector<unsigned int> sizes_;
  AsyncDeviceVector<std::uint8_t> reduceTemp_;
  AsyncDevicePtr<double>          inertia_;
  AsyncDevicePtr<int>             changed_;
};

}  // namespace
}  // namespace detail

AsyncDeviceVector<int> kmeansInitGpu(const cuda::std::span<const std::uint32_t> fingerprints,
                                     const int                                  numFingerprints,
                                     const int                                  numWords,
                                     const int                                  numClusters,
                                     const KMeansMetric                         metric,
                                     const KMeansInit                           init,
                                     const std::uint64_t                        seed,
                                     cudaStream_t                               stream) {
  detail::validateFingerprints(fingerprints, numFingerprints, numWords);
  detail::validateClusterCount(numClusters, numFingerprints);
  const ScopedNvtxRange range("kmeansInit", NvtxColor::kYellow);
  switch (init) {
    case KMeansInit::Random: {
      const auto             rows = detail::sampleDistinctRows(numFingerprints, numClusters, seed);
      AsyncDeviceVector<int> result(numClusters, stream);
      result.copyFromHost(rows);
      cudaCheckError(cudaStreamSynchronize(stream));
      return result;
    }
    case KMeansInit::MaxMin: {
      const auto similarity =
        metric == KMeansMetric::Cosine ? FingerprintSimilarityMetric::Cosine : FingerprintSimilarityMetric::Tanimoto;
      auto picks = fusedMaxMinGpu(fingerprints,
                                  numFingerprints,
                                  numWords,
                                  numClusters,
                                  similarity,
                                  {},
                                  static_cast<int>(seed & 0x7FFFFFFFU),
                                  -1.0,
                                  stream);
      if (static_cast<int>(picks.indices.size()) != numClusters) {
        throw std::runtime_error("MaxMin returned fewer picks than n_clusters");
      }
      return std::move(picks.indices);
    }
  }
  throw std::invalid_argument("Unsupported k-means initialization");
}

AsyncDeviceVector<float> kmeansCentersFromRows(const cuda::std::span<const std::uint32_t> fingerprints,
                                               const int                                  numWords,
                                               const AsyncDeviceVector<int>&              rows,
                                               const KMeansMetric                         metric,
                                               cudaStream_t                               stream) {
  const int                numCenters = static_cast<int>(rows.size());
  AsyncDeviceVector<float> centers(static_cast<std::size_t>(numCenters) * numWords * 32, stream);
  if (numCenters > 0) {
    detail::centersFromRowsKernel<<<detail::gridFor(static_cast<std::size_t>(numCenters) * 32),
                                    detail::kBlockSize,
                                    0,
                                    stream>>>(fingerprints.data(),
                                              rows.data(),
                                              centers.data(),
                                              numCenters,
                                              numWords,
                                              metric == KMeansMetric::Cosine);
    cudaCheckError(cudaGetLastError());
  }
  return centers;
}

KMeansFitResult kmeansFitGpu(const cuda::std::span<const std::uint32_t> fingerprints,
                             const int                                  numFingerprints,
                             const int                                  numWords,
                             const float*                               initialCenters,
                             const KMeansOptions&                       options,
                             cudaStream_t                               stream) {
  AsyncDeviceVector<float> centers(static_cast<std::size_t>(std::max(options.numClusters, 0)) * numWords * 32, stream);
  if (centers.size() > 0) {
    cudaCheckError(cudaMemcpyAsync(centers.data(),
                                   initialCenters,
                                   sizeof(float) * centers.size(),
                                   cudaMemcpyDeviceToDevice,
                                   stream));
  }
  return kmeansFitGpu(fingerprints, numFingerprints, numWords, std::move(centers), options, stream);
}

KMeansFitResult kmeansFitGpu(const cuda::std::span<const std::uint32_t> fingerprints,
                             const int                                  numFingerprints,
                             const int                                  numWords,
                             AsyncDeviceVector<float>                   initialCenters,
                             const KMeansOptions&                       options,
                             cudaStream_t                               stream) {
  detail::validateFingerprints(fingerprints, numFingerprints, numWords);
  detail::validateClusterCount(options.numClusters, numFingerprints);
  if (options.maxIter < 0) {
    throw std::invalid_argument("max_iter must be non-negative");
  }
  if (!(options.tol >= 0.0)) {
    throw std::invalid_argument("tol must be non-negative");
  }
  if (initialCenters.size() != static_cast<std::size_t>(options.numClusters) * numWords * 32) {
    throw std::invalid_argument("initial centers must have shape (n_clusters, num_words * 32)");
  }
  const ScopedNvtxRange range("kmeansFit", NvtxColor::kGreen);
  detail::LloydState    state(fingerprints, numFingerprints, numWords, options, std::move(initialCenters), stream);

  std::int64_t reseeded   = 0;
  auto [inertia, changed] = state.assignPass();
  int numIterations       = 0;
  for (int iteration = 1; iteration <= options.maxIter; ++iteration) {
    reseeded += state.updatePass();
    const auto [current, moved] = state.assignPass();
    numIterations               = iteration;
    const double previous       = inertia;
    inertia                     = current;
    // tol == 0 runs exactly maxIter iterations (fixed-cost timing); otherwise stop at a fixed point or when the
    // objective improves by at most tol relative to its value.
    if (options.tol > 0.0 && (moved == 0 || previous - current <= options.tol * current)) {
      break;
    }
  }
  auto result     = state.finish(inertia, numIterations);
  result.reseeded = reseeded;
  return result;
}

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
