// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/ROMol.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/aap.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"
#include "src/utils/host_vector.h"

namespace nvMolKit {
namespace {

// TODO: Support molecules with more than 64 atoms.
constexpr int           kMaxAtoms        = 64;
constexpr int           kSharedStride    = kMaxAtoms + 1;
constexpr int           kThreads         = 256;
constexpr std::uint64_t kFnvOffset       = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime        = 1099511628211ULL;
constexpr std::uint64_t kSignedInt64Mask = (1ULL << 63) - 1;

struct AapHostDescriptors {
  std::vector<std::int64_t> moleculeAtomOffsets{0};
  std::vector<std::int64_t> atomBinOffsets{0};
  std::vector<std::int64_t> atomPathLengths;
  std::vector<std::int16_t> atomNumbers;
  std::vector<std::uint8_t> aromatic;
  std::vector<std::int16_t> binIds;
  std::vector<std::int32_t> binCounts;
};

struct AapDeviceDescriptors {
  explicit AapDeviceDescriptors(const AapHostDescriptors& host, cudaStream_t stream)
      : moleculeAtomOffsets(host.moleculeAtomOffsets.size(), stream),
        atomBinOffsets(host.atomBinOffsets.size(), stream),
        atomPathLengths(host.atomPathLengths.size(), stream),
        atomNumbers(host.atomNumbers.size(), stream),
        aromatic(host.aromatic.size(), stream),
        binIds(host.binIds.size(), stream),
        binCounts(host.binCounts.size(), stream) {
    moleculeAtomOffsets.copyFromHost(host.moleculeAtomOffsets);
    atomBinOffsets.copyFromHost(host.atomBinOffsets);
    atomPathLengths.copyFromHost(host.atomPathLengths);
    atomNumbers.copyFromHost(host.atomNumbers);
    aromatic.copyFromHost(host.aromatic);
    binIds.copyFromHost(host.binIds);
    binCounts.copyFromHost(host.binCounts);
  }

  AsyncDeviceVector<std::int64_t> moleculeAtomOffsets;
  AsyncDeviceVector<std::int64_t> atomBinOffsets;
  AsyncDeviceVector<std::int64_t> atomPathLengths;
  AsyncDeviceVector<std::int16_t> atomNumbers;
  AsyncDeviceVector<std::uint8_t> aromatic;
  AsyncDeviceVector<std::int16_t> binIds;
  AsyncDeviceVector<std::int32_t> binCounts;
};

int aapBondCode(const RDKit::Bond& bond) {
  switch (bond.getBondType()) {
    case RDKit::Bond::SINGLE:
      return 1;
    case RDKit::Bond::DOUBLE:
      return 2;
    case RDKit::Bond::TRIPLE:
      return 3;
    case RDKit::Bond::AROMATIC:
      return 4;
    default:
      throw std::invalid_argument("AAP supports only single, double, triple, and aromatic bonds");
  }
}

void collectRootedPathBins(const RDKit::ROMol&                   mol,
                           const unsigned int                    currentAtom,
                           const int                             depth,
                           const int                             maxDepth,
                           const int                             histogramBins,
                           const std::uint64_t                   pathCode,
                           std::vector<std::uint8_t>&            visited,
                           std::map<std::int16_t, std::int32_t>& counts,
                           std::int64_t&                         pathCount) {
  if (depth == maxDepth) {
    return;
  }

  const auto* atom       = mol.getAtomWithIdx(currentAtom);
  auto [bondIt, bondEnd] = mol.getAtomBonds(atom);
  while (bondIt != bondEnd) {
    const auto* bond      = mol[*bondIt];
    const auto  otherAtom = bond->getOtherAtomIdx(currentAtom);
    ++bondIt;
    if (visited[otherAtom]) {
      continue;
    }

    const auto* neighbor = mol.getAtomWithIdx(otherAtom);
    const auto  atomCode = neighbor->getAtomicNum() + (neighbor->getIsAromatic() ? 108 : 0);
    const auto  token    = static_cast<std::uint64_t>(aapBondCode(*bond) * 256 + atomCode);
    const auto  nextCode = ((pathCode ^ token) * kFnvPrime) & kSignedInt64Mask;
    const auto  bin      = static_cast<std::int16_t>(nextCode % static_cast<std::uint64_t>(histogramBins));
    ++counts[bin];
    ++pathCount;

    visited[otherAtom] = 1;
    collectRootedPathBins(mol, otherAtom, depth + 1, maxDepth, histogramBins, nextCode, visited, counts, pathCount);
    visited[otherAtom] = 0;
  }
}

void validateOptions(const AapOptions& options) {
  if (options.maxPathLength <= 0) {
    throw std::invalid_argument("maxPathLength must be positive");
  }
  if (options.histogramBins <= 0 || options.histogramBins > std::numeric_limits<std::int16_t>::max()) {
    throw std::invalid_argument("histogramBins must be between 1 and 32767");
  }
  if (options.sinkhornIterations <= 0) {
    throw std::invalid_argument("sinkhornIterations must be positive");
  }
  if (!(options.sinkhornTemperature > 0.0F) || !std::isfinite(options.sinkhornTemperature)) {
    throw std::invalid_argument("sinkhornTemperature must be finite and positive");
  }
  if (options.sinkhornTemperature < std::numeric_limits<float>::min()) {
    throw std::invalid_argument("sinkhornTemperature must be at least the smallest positive normal float");
  }
}

AapHostDescriptors buildDescriptors(const std::vector<const RDKit::ROMol*>& molecules, const AapOptions& options) {
  AapHostDescriptors result;
  for (std::size_t moleculeIdx = 0; moleculeIdx < molecules.size(); ++moleculeIdx) {
    const auto* mol = molecules[moleculeIdx];
    if (mol == nullptr) {
      throw std::invalid_argument("Invalid molecule at index " + std::to_string(moleculeIdx));
    }
    const auto numAtoms = mol->getNumAtoms();
    if (numAtoms == 0) {
      throw std::invalid_argument("AAP does not support empty molecules");
    }
    if (numAtoms > kMaxAtoms) {
      throw std::invalid_argument("AAP currently supports at most " + std::to_string(kMaxAtoms) +
                                  " atoms per molecule");
    }

    std::vector<std::uint8_t> visited(numAtoms, 0);
    for (unsigned int atomIdx = 0; atomIdx < numAtoms; ++atomIdx) {
      const auto* atom = mol->getAtomWithIdx(atomIdx);
      result.atomNumbers.push_back(static_cast<std::int16_t>(atom->getAtomicNum()));
      result.aromatic.push_back(static_cast<std::uint8_t>(atom->getIsAromatic()));

      std::map<std::int16_t, std::int32_t> counts;
      std::int64_t                         pathCount = 0;
      visited[atomIdx]                               = 1;
      collectRootedPathBins(*mol,
                            atomIdx,
                            0,
                            options.maxPathLength,
                            options.histogramBins,
                            kFnvOffset,
                            visited,
                            counts,
                            pathCount);
      visited[atomIdx] = 0;

      result.atomPathLengths.push_back(pathCount);
      for (const auto& [bin, count] : counts) {
        result.binIds.push_back(bin);
        result.binCounts.push_back(count);
      }
      result.atomBinOffsets.push_back(static_cast<std::int64_t>(result.binIds.size()));
    }
    result.moleculeAtomOffsets.push_back(static_cast<std::int64_t>(result.atomNumbers.size()));
  }
  return result;
}

bool sameAtomDescriptor(const AapHostDescriptors& descriptors, const std::int64_t left, const std::int64_t right) {
  if (descriptors.atomNumbers[left] != descriptors.atomNumbers[right] ||
      descriptors.aromatic[left] != descriptors.aromatic[right] ||
      descriptors.atomPathLengths[left] != descriptors.atomPathLengths[right]) {
    return false;
  }
  const auto leftBinStart  = descriptors.atomBinOffsets[left];
  const auto leftBinEnd    = descriptors.atomBinOffsets[left + 1];
  const auto rightBinStart = descriptors.atomBinOffsets[right];
  const auto rightBinEnd   = descriptors.atomBinOffsets[right + 1];
  if (leftBinEnd - leftBinStart != rightBinEnd - rightBinStart) {
    return false;
  }
  for (std::int64_t binOffset = 0; binOffset < leftBinEnd - leftBinStart; ++binOffset) {
    if (descriptors.binIds[leftBinStart + binOffset] != descriptors.binIds[rightBinStart + binOffset] ||
        descriptors.binCounts[leftBinStart + binOffset] != descriptors.binCounts[rightBinStart + binOffset]) {
      return false;
    }
  }
  return true;
}

bool sameMoleculeDescriptor(const AapHostDescriptors& descriptors, const int left, const int right) {
  const auto leftAtomStart  = descriptors.moleculeAtomOffsets[left];
  const auto leftAtomEnd    = descriptors.moleculeAtomOffsets[left + 1];
  const auto rightAtomStart = descriptors.moleculeAtomOffsets[right];
  const auto rightAtomEnd   = descriptors.moleculeAtomOffsets[right + 1];
  if (leftAtomEnd - leftAtomStart != rightAtomEnd - rightAtomStart) {
    return false;
  }

  std::vector<std::uint8_t> matched(rightAtomEnd - rightAtomStart, 0);
  for (std::int64_t atomOffset = 0; atomOffset < leftAtomEnd - leftAtomStart; ++atomOffset) {
    bool found = false;
    for (std::int64_t rightOffset = 0; rightOffset < rightAtomEnd - rightAtomStart; ++rightOffset) {
      if (!matched[rightOffset] &&
          sameAtomDescriptor(descriptors, leftAtomStart + atomOffset, rightAtomStart + rightOffset)) {
        matched[rightOffset] = 1;
        found                = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
  }
  return true;
}

__device__ __forceinline__ float rowLogSumExp(const float* values, const int row, const int size) {
  float maximum = -INFINITY;
  for (int column = 0; column < size; ++column) {
    maximum = fmaxf(maximum, values[row * kSharedStride + column]);
  }
  float sum = 0.0F;
  for (int column = 0; column < size; ++column) {
    sum += expf(values[row * kSharedStride + column] - maximum);
  }
  return maximum + logf(sum);
}

__device__ __forceinline__ float columnLogSumExp(const float* values, const int column, const int size) {
  float maximum = -INFINITY;
  for (int row = 0; row < size; ++row) {
    maximum = fmaxf(maximum, values[row * kSharedStride + column]);
  }
  float sum = 0.0F;
  for (int row = 0; row < size; ++row) {
    sum += expf(values[row * kSharedStride + column] - maximum);
  }
  return maximum + logf(sum);
}

__global__ void aapSimilarityKernel(const std::int64_t* moleculeAtomOffsets,
                                    const std::int64_t* atomBinOffsets,
                                    const std::int64_t* atomPathLengths,
                                    const std::int16_t* atomNumbers,
                                    const std::uint8_t* aromatic,
                                    const std::int16_t* binIds,
                                    const std::int32_t* binCounts,
                                    const int*          candidates,
                                    const int           candidateCount,
                                    const int           centroid,
                                    const float         temperature,
                                    const int           iterations,
                                    float*              output) {
  const int candidatePosition = static_cast<int>(blockIdx.x);
  if (candidatePosition >= candidateCount) {
    return;
  }

  __shared__ float affinity[kMaxAtoms * kSharedStride];
  __shared__ float transport[kMaxAtoms * kSharedStride];

  const int candidate      = candidates[candidatePosition];
  const int leftStart      = static_cast<int>(moleculeAtomOffsets[centroid]);
  const int rightStart     = static_cast<int>(moleculeAtomOffsets[candidate]);
  const int leftCount      = static_cast<int>(moleculeAtomOffsets[centroid + 1]) - leftStart;
  const int candidateAtoms = static_cast<int>(moleculeAtomOffsets[candidate + 1]) - rightStart;
  const int squareSize     = max(leftCount, candidateAtoms);

  for (int linear = static_cast<int>(threadIdx.x); linear < squareSize * squareSize; linear += blockDim.x) {
    const int row    = linear / squareSize;
    const int column = linear % squareSize;
    float     score  = 0.0F;
    if (row < leftCount && column < candidateAtoms &&
        atomNumbers[leftStart + row] == atomNumbers[rightStart + column] &&
        aromatic[leftStart + row] == aromatic[rightStart + column]) {
      std::int64_t leftPosition  = atomBinOffsets[leftStart + row];
      const auto   leftEnd       = atomBinOffsets[leftStart + row + 1];
      std::int64_t rightPosition = atomBinOffsets[rightStart + column];
      const auto   rightEnd      = atomBinOffsets[rightStart + column + 1];
      std::int64_t overlap       = 0;
      while (leftPosition < leftEnd && rightPosition < rightEnd) {
        const auto leftBin  = binIds[leftPosition];
        const auto rightBin = binIds[rightPosition];
        if (leftBin == rightBin) {
          overlap += min(binCounts[leftPosition], binCounts[rightPosition]);
          ++leftPosition;
          ++rightPosition;
        } else if (leftBin < rightBin) {
          ++leftPosition;
        } else {
          ++rightPosition;
        }
      }
      const auto largestPathCount = max(atomPathLengths[leftStart + row], atomPathLengths[rightStart + column]);
      score = static_cast<float>(overlap + 1) / static_cast<float>(2 * largestPathCount - overlap + 1);
    }
    affinity[row * kSharedStride + column]  = score;
    transport[row * kSharedStride + column] = score / temperature;
  }
  __syncthreads();

  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (threadIdx.x < squareSize) {
      const int   row        = static_cast<int>(threadIdx.x);
      const float normalizer = rowLogSumExp(transport, row, squareSize);
      for (int column = 0; column < squareSize; ++column) {
        transport[row * kSharedStride + column] -= normalizer;
      }
    }
    __syncthreads();
    if (threadIdx.x < squareSize) {
      const int   column     = static_cast<int>(threadIdx.x);
      const float normalizer = columnLogSumExp(transport, column, squareSize);
      for (int row = 0; row < squareSize; ++row) {
        transport[row * kSharedStride + column] -= normalizer;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float matched = 0.0F;
    for (int row = 0; row < leftCount; ++row) {
      for (int column = 0; column < candidateAtoms; ++column) {
        matched += expf(transport[row * kSharedStride + column]) * affinity[row * kSharedStride + column];
      }
    }
    output[candidatePosition] =
      candidate == centroid ? 1.0F : matched / (2.0F * static_cast<float>(leftCount) - matched + 1e-10F);
  }
}

void launchSimilarity(const AapDeviceDescriptors&   descriptors,
                      const AsyncDeviceVector<int>& candidates,
                      const int                     candidateCount,
                      const int                     centroid,
                      const AapOptions&             options,
                      AsyncDeviceVector<float>&     output,
                      cudaStream_t                  stream) {
  if (candidateCount == 0) {
    return;
  }
  aapSimilarityKernel<<<candidateCount, kThreads, 0, stream>>>(descriptors.moleculeAtomOffsets.data(),
                                                               descriptors.atomBinOffsets.data(),
                                                               descriptors.atomPathLengths.data(),
                                                               descriptors.atomNumbers.data(),
                                                               descriptors.aromatic.data(),
                                                               descriptors.binIds.data(),
                                                               descriptors.binCounts.data(),
                                                               candidates.data(),
                                                               candidateCount,
                                                               centroid,
                                                               options.sinkhornTemperature,
                                                               options.sinkhornIterations,
                                                               output.data());
  cudaCheckError(cudaGetLastError());
}

struct CentroidSelection {
  std::vector<int> labels;
  std::vector<int> centroids;
};

CentroidSelection selectCentroids(const AapHostDescriptors&   hostDescriptors,
                                  const AapDeviceDescriptors& descriptors,
                                  const int                   numMolecules,
                                  const float                 threshold,
                                  const AapOptions&           options,
                                  AsyncDeviceVector<int>&     candidates,
                                  AsyncDeviceVector<float>&   output,
                                  PinnedHostVector<int>&      candidateHost,
                                  PinnedHostVector<float>&    outputHost,
                                  cudaStream_t                stream) {
  CentroidSelection result{std::vector<int>(numMolecules, -1), {}};

  for (int centroid = 0; centroid < numMolecules; ++centroid) {
    if (result.labels[centroid] >= 0) {
      continue;
    }
    const int clusterId     = static_cast<int>(result.centroids.size());
    result.labels[centroid] = clusterId;
    result.centroids.push_back(centroid);

    int candidateCount = 0;
    for (int moleculeIdx = 0; moleculeIdx < numMolecules; ++moleculeIdx) {
      if (result.labels[moleculeIdx] < 0) {
        if (sameMoleculeDescriptor(hostDescriptors, centroid, moleculeIdx)) {
          result.labels[moleculeIdx] = clusterId;
        } else {
          candidateHost[candidateCount++] = moleculeIdx;
        }
      }
    }
    if (candidateCount == 0) {
      continue;
    }

    candidates.copyFromHost(candidateHost.data(), candidateCount);
    launchSimilarity(descriptors, candidates, candidateCount, centroid, options, output, stream);
    output.copyToHost(outputHost.data(), candidateCount);
    cudaCheckError(cudaStreamSynchronize(stream));
    for (int position = 0; position < candidateCount; ++position) {
      if (outputHost[position] >= threshold) {
        result.labels[candidateHost[position]] = clusterId;
      }
    }
  }
  return result;
}

ClusteringResult buildClusteringResult(const std::vector<int>& labels, const std::vector<int>& centroids) {
  const int        numClusters = static_cast<int>(centroids.size());
  std::vector<int> sizes(numClusters, 0);
  for (const int label : labels) {
    ++sizes[label];
  }
  std::vector<int> order(numClusters);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&sizes](const int left, const int right) {
    return sizes[left] > sizes[right];
  });
  std::vector<int> remap(numClusters);
  for (int newId = 0; newId < numClusters; ++newId) {
    remap[order[newId]] = newId;
  }
  ClusteringResult result;
  result.clusterIds.resize(labels.size());
  result.centroids.resize(numClusters);
  result.clusterSizes.resize(numClusters);
  std::transform(labels.begin(), labels.end(), result.clusterIds.begin(), [&remap](const int label) {
    return remap[label];
  });
  for (int newId = 0; newId < numClusters; ++newId) {
    const int oldId            = order[newId];
    result.centroids[newId]    = centroids[oldId];
    result.clusterSizes[newId] = sizes[oldId];
  }
  return result;
}

}  // namespace

float aapSimilarityGpu(const RDKit::ROMol& left,
                       const RDKit::ROMol& right,
                       const AapOptions&   options,
                       cudaStream_t        stream) {
  validateOptions(options);
  const std::vector<const RDKit::ROMol*> molecules{&left, &right};
  const auto                             hostDescriptors = buildDescriptors(molecules, options);
  if (sameMoleculeDescriptor(hostDescriptors, 0, 1)) {
    return 1.0F;
  }
  const AapDeviceDescriptors descriptors(hostDescriptors, stream);
  AsyncDeviceVector<int>     candidates(1, stream);
  AsyncDeviceVector<float>   output(1, stream);
  PinnedHostVector<int>      candidateHost(1);
  PinnedHostVector<float>    outputHost(1);
  candidateHost[0] = 1;
  candidates.copyFromHost(candidateHost.data(), 1);
  launchSimilarity(descriptors, candidates, 1, 0, options, output, stream);
  outputHost.copyFromDevice(output, stream);
  cudaCheckError(cudaStreamSynchronize(stream));
  return outputHost[0];
}

ClusteringResult aapSimilarityClustering(const std::vector<const RDKit::ROMol*>& molecules,
                                         const float                             threshold,
                                         const AapOptions&                       options,
                                         cudaStream_t                            stream) {
  validateOptions(options);
  if (!(threshold >= 0.0F && threshold <= 1.0F)) {
    throw std::invalid_argument("threshold must be between 0 and 1");
  }
  if (molecules.empty()) {
    return {};
  }

  const auto                 hostDescriptors = buildDescriptors(molecules, options);
  const AapDeviceDescriptors descriptors(hostDescriptors, stream);
  const int                  numMolecules = static_cast<int>(molecules.size());
  AsyncDeviceVector<int>     candidates(numMolecules, stream);
  AsyncDeviceVector<float>   output(numMolecules, stream);
  PinnedHostVector<int>      candidateHost(numMolecules);
  PinnedHostVector<float>    outputHost(numMolecules);

  const auto selection = selectCentroids(hostDescriptors,
                                         descriptors,
                                         numMolecules,
                                         threshold,
                                         options,
                                         candidates,
                                         output,
                                         candidateHost,
                                         outputHost,
                                         stream);
  return buildClusteringResult(selection.labels, selection.centroids);
}

ClusteringResult aapDiseClustering(const std::vector<const RDKit::ROMol*>& molecules,
                                   const float                             threshold,
                                   const AapOptions&                       options,
                                   cudaStream_t                            stream) {
  validateOptions(options);
  if (!(threshold >= 0.0F && threshold <= 1.0F)) {
    throw std::invalid_argument("threshold must be between 0 and 1");
  }
  if (molecules.empty()) {
    return {};
  }

  const auto                 hostDescriptors = buildDescriptors(molecules, options);
  const AapDeviceDescriptors descriptors(hostDescriptors, stream);
  const int                  numMolecules = static_cast<int>(molecules.size());
  AsyncDeviceVector<int>     candidates(numMolecules, stream);
  AsyncDeviceVector<float>   output(numMolecules, stream);
  PinnedHostVector<int>      candidateHost(numMolecules);
  PinnedHostVector<float>    outputHost(numMolecules);

  const auto  selection = selectCentroids(hostDescriptors,
                                         descriptors,
                                         numMolecules,
                                         threshold,
                                         options,
                                         candidates,
                                         output,
                                         candidateHost,
                                         outputHost,
                                         stream);
  const auto& centroids = selection.centroids;

  std::vector<std::uint8_t> isCentroid(numMolecules, 0);
  std::vector<int>          labels(numMolecules, -1);
  std::vector<float>        bestSimilarity(numMolecules, -1.0F);
  for (int clusterId = 0; clusterId < static_cast<int>(centroids.size()); ++clusterId) {
    isCentroid[centroids[clusterId]]     = 1;
    labels[centroids[clusterId]]         = clusterId;
    bestSimilarity[centroids[clusterId]] = 1.0F;
  }

  for (int clusterId = 0; clusterId < static_cast<int>(centroids.size()); ++clusterId) {
    const int centroid       = centroids[clusterId];
    int       candidateCount = 0;
    for (int moleculeIdx = 0; moleculeIdx < numMolecules; ++moleculeIdx) {
      if (isCentroid[moleculeIdx]) {
        continue;
      }
      if (sameMoleculeDescriptor(hostDescriptors, centroid, moleculeIdx)) {
        if (1.0F > bestSimilarity[moleculeIdx]) {
          bestSimilarity[moleculeIdx] = 1.0F;
          labels[moleculeIdx]         = clusterId;
        }
      } else {
        candidateHost[candidateCount++] = moleculeIdx;
      }
    }
    if (candidateCount == 0) {
      continue;
    }
    candidates.copyFromHost(candidateHost.data(), candidateCount);
    launchSimilarity(descriptors, candidates, candidateCount, centroid, options, output, stream);
    output.copyToHost(outputHost.data(), candidateCount);
    cudaCheckError(cudaStreamSynchronize(stream));
    for (int position = 0; position < candidateCount; ++position) {
      const int moleculeIdx = candidateHost[position];
      if (outputHost[position] > bestSimilarity[moleculeIdx]) {
        bestSimilarity[moleculeIdx] = outputHost[position];
        labels[moleculeIdx]         = clusterId;
      }
    }
  }

  return buildClusteringResult(labels, centroids);
}

}  // namespace nvMolKit
