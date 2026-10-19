// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/ROMol.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "src/aap.h"
#include "src/diversity_picker_algorithms.cuh"
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

bool hasUnsupportedBond(const RDKit::ROMol& mol) {
  for (const auto* bond : mol.bonds()) {
    const auto type = bond->getBondType();
    if (type != RDKit::Bond::SINGLE && type != RDKit::Bond::DOUBLE && type != RDKit::Bond::TRIPLE &&
        type != RDKit::Bond::AROMATIC) {
      return true;
    }
  }
  return false;
}

void appendIndices(std::ostringstream& message, const char* reason, const std::vector<int>& indices) {
  constexpr std::size_t kShown = 10;
  if (indices.empty()) {
    return;
  }
  message << (message.tellp() > 0 ? "; " : "") << reason << " at indices [";
  for (std::size_t position = 0; position < std::min(indices.size(), kShown); ++position) {
    message << (position > 0 ? ", " : "") << indices[position];
  }
  if (indices.size() > kShown) {
    message << ", ... (" << indices.size() << " total)";
  }
  message << "]";
}

void validateMolecules(const std::vector<const RDKit::ROMol*>& molecules) {
  std::vector<int> none;
  std::vector<int> empty;
  std::vector<int> tooManyAtoms;
  std::vector<int> unsupportedBonds;
  for (std::size_t index = 0; index < molecules.size(); ++index) {
    const auto* mol      = molecules[index];
    const int   position = static_cast<int>(index);
    if (mol == nullptr) {
      none.push_back(position);
    } else if (mol->getNumAtoms() == 0) {
      empty.push_back(position);
    } else if (mol->getNumAtoms() > kMaxAtoms) {
      tooManyAtoms.push_back(position);
    } else if (hasUnsupportedBond(*mol)) {
      unsupportedBonds.push_back(position);
    }
  }
  if (none.empty() && empty.empty() && tooManyAtoms.empty() && unsupportedBonds.empty()) {
    return;
  }
  std::ostringstream message;
  appendIndices(message, "None molecules", none);
  appendIndices(message, "empty molecules", empty);
  appendIndices(message, ("molecules with more than " + std::to_string(kMaxAtoms) + " atoms").c_str(), tooManyAtoms);
  appendIndices(message, "bonds other than single, double, triple, or aromatic", unsupportedBonds);
  throw AapInvalidMoleculesError("AAP cannot process " + message.str(),
                                 std::move(none),
                                 std::move(empty),
                                 std::move(tooManyAtoms),
                                 std::move(unsupportedBonds));
}

struct MoleculeDescriptor {
  std::vector<std::int16_t> atomNumbers;
  std::vector<std::uint8_t> aromatic;
  std::vector<std::int64_t> atomPathLengths;
  std::vector<std::int64_t> atomBinCounts;
  std::vector<std::int16_t> binIds;
  std::vector<std::int32_t> binCounts;
};

MoleculeDescriptor describeMolecule(const RDKit::ROMol& mol, const AapOptions& options) {
  MoleculeDescriptor        result;
  const auto                numAtoms = mol.getNumAtoms();
  std::vector<std::uint8_t> visited(numAtoms, 0);
  for (unsigned int atomIdx = 0; atomIdx < numAtoms; ++atomIdx) {
    const auto* atom = mol.getAtomWithIdx(atomIdx);
    result.atomNumbers.push_back(static_cast<std::int16_t>(atom->getAtomicNum()));
    result.aromatic.push_back(static_cast<std::uint8_t>(atom->getIsAromatic()));

    std::map<std::int16_t, std::int32_t> counts;
    std::int64_t                         pathCount = 0;
    visited[atomIdx]                               = 1;
    collectRootedPathBins(mol,
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
    result.atomBinCounts.push_back(static_cast<std::int64_t>(counts.size()));
    for (const auto& [bin, count] : counts) {
      result.binIds.push_back(bin);
      result.binCounts.push_back(count);
    }
  }
  return result;
}

AapHostDescriptors buildDescriptors(const std::vector<const RDKit::ROMol*>& molecules, const AapOptions& options) {
  validateMolecules(molecules);

  // Rooted-path enumeration dominates host time and is independent per molecule.
  constexpr std::size_t           kMoleculesPerThread = 64;
  const std::size_t               numMolecules        = molecules.size();
  std::vector<MoleculeDescriptor> perMolecule(numMolecules);
  const std::size_t               numThreads =
    std::clamp<std::size_t>(numMolecules / kMoleculesPerThread, 1, std::max(1U, std::thread::hardware_concurrency()));
  std::vector<std::exception_ptr> failures(numThreads);
  const auto                      describeShare = [&](const std::size_t worker) {
    try {
      for (std::size_t index = worker; index < numMolecules; index += numThreads) {
        perMolecule[index] = describeMolecule(*molecules[index], options);
      }
    } catch (...) {
      failures[worker] = std::current_exception();
    }
  };
  // The calling thread takes the first share, so small inputs start no threads.
  std::vector<std::thread> workers;
  workers.reserve(numThreads - 1);
  try {
    for (std::size_t worker = 1; worker < numThreads; ++worker) {
      workers.emplace_back(describeShare, worker);
    }
  } catch (...) {
    // A joinable thread must not be destroyed, so join the workers that started before rethrowing.
    for (auto& thread : workers) {
      thread.join();
    }
    throw;
  }
  describeShare(0);
  for (auto& thread : workers) {
    thread.join();
  }
  for (const auto& failure : failures) {
    if (failure) {
      std::rethrow_exception(failure);
    }
  }

  AapHostDescriptors result;
  for (const auto& molecule : perMolecule) {
    for (std::size_t atom = 0; atom < molecule.atomNumbers.size(); ++atom) {
      result.atomBinOffsets.push_back(result.atomBinOffsets.back() + molecule.atomBinCounts[atom]);
    }
    result.atomNumbers.insert(result.atomNumbers.end(), molecule.atomNumbers.begin(), molecule.atomNumbers.end());
    result.aromatic.insert(result.aromatic.end(), molecule.aromatic.begin(), molecule.aromatic.end());
    result.atomPathLengths.insert(result.atomPathLengths.end(),
                                  molecule.atomPathLengths.begin(),
                                  molecule.atomPathLengths.end());
    result.binIds.insert(result.binIds.end(), molecule.binIds.begin(), molecule.binIds.end());
    result.binCounts.insert(result.binCounts.end(), molecule.binCounts.begin(), molecule.binCounts.end());
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

//! Identical descriptor multisets score exactly 1 regardless of Sinkhorn convergence. Groups are keyed by their first
//! member so every pair lookup on the device is a single comparison.
std::vector<int> descriptorGroups(const AapHostDescriptors& descriptors) {
  const auto mix = [](std::uint64_t hash, const std::uint64_t value) { return (hash ^ value) * kFnvPrime; };

  const int        numMolecules = static_cast<int>(descriptors.moleculeAtomOffsets.size()) - 1;
  std::vector<int> groups(numMolecules);
  std::unordered_map<std::uint64_t, std::vector<int>> representatives;
  std::vector<std::uint64_t>                          atomHashes;
  for (int molecule = 0; molecule < numMolecules; ++molecule) {
    atomHashes.clear();
    for (auto atom = descriptors.moleculeAtomOffsets[molecule]; atom < descriptors.moleculeAtomOffsets[molecule + 1];
         ++atom) {
      std::uint64_t hash = kFnvOffset;
      hash               = mix(hash, static_cast<std::uint64_t>(descriptors.atomNumbers[atom]));
      hash               = mix(hash, descriptors.aromatic[atom]);
      hash               = mix(hash, static_cast<std::uint64_t>(descriptors.atomPathLengths[atom]));
      for (auto bin = descriptors.atomBinOffsets[atom]; bin < descriptors.atomBinOffsets[atom + 1]; ++bin) {
        hash = mix(hash, static_cast<std::uint64_t>(descriptors.binIds[bin]));
        hash = mix(hash, static_cast<std::uint64_t>(descriptors.binCounts[bin]));
      }
      atomHashes.push_back(hash);
    }
    std::sort(atomHashes.begin(), atomHashes.end());
    std::uint64_t moleculeHash = kFnvOffset;
    for (const auto atomHash : atomHashes) {
      moleculeHash = mix(moleculeHash, atomHash);
    }

    auto& candidates = representatives[moleculeHash];
    groups[molecule] = molecule;
    const auto match = std::find_if(candidates.begin(), candidates.end(), [&](const int representative) {
      return sameMoleculeDescriptor(descriptors, representative, molecule);
    });
    if (match != candidates.end()) {
      groups[molecule] = *match;
    } else {
      candidates.push_back(molecule);
    }
  }
  return groups;
}

struct AapDeviceView {
  const std::int64_t* moleculeAtomOffsets;
  const std::int64_t* atomBinOffsets;
  const std::int64_t* atomPathLengths;
  const std::int16_t* atomNumbers;
  const std::uint8_t* aromatic;
  const std::int16_t* binIds;
  const std::int32_t* binCounts;
  const int*          groups;
};

template <typename Op>
__global__ void compactCandidatesKernel(const int numItems, const Op op, int* candidates, int* count) {
  const int candidate = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (candidate >= numItems || op.skip(candidate)) {
    return;
  }
  candidates[atomicAdd(count, 1)] = candidate;
}

__device__ __forceinline__ float warpMax(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffU, value, offset));
  }
  return value;
}

__device__ __forceinline__ float warpSum(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(0xffffffffU, value, offset);
  }
  return value;
}

//! Subtracts the log-sum-exp of each row (or column) of the square transport plan, one warp per line.
template <bool Rows> __device__ void normalizeLines(float* transport, const int squareSize) {
  constexpr int kWarps = kThreads / 32;
  const int     lane   = static_cast<int>(threadIdx.x) % 32;
  for (int line = static_cast<int>(threadIdx.x) / 32; line < squareSize; line += kWarps) {
    const auto at = [&](const int offset) -> float& {
      return Rows ? transport[line * kSharedStride + offset] : transport[offset * kSharedStride + line];
    };
    float maximum = -INFINITY;
    for (int offset = lane; offset < squareSize; offset += 32) {
      maximum = fmaxf(maximum, at(offset));
    }
    maximum   = warpMax(maximum);
    float sum = 0.0F;
    for (int offset = lane; offset < squareSize; offset += 32) {
      sum += expf(at(offset) - maximum);
    }
    const float normalizer = maximum + logf(warpSum(sum));
    for (int offset = lane; offset < squareSize; offset += 32) {
      at(offset) -= normalizer;
    }
  }
}

//! Block-wide AAP similarity of one ordered pair. Every thread must call this; the result is valid in all threads.
__device__ float aapPairSimilarity(const AapDeviceView& view,
                                   const int            source,
                                   const int            candidate,
                                   const float          temperature,
                                   const int            iterations,
                                   float*               affinity,
                                   float*               transport,
                                   float*               partials) {
  const int leftStart      = static_cast<int>(view.moleculeAtomOffsets[source]);
  const int rightStart     = static_cast<int>(view.moleculeAtomOffsets[candidate]);
  const int leftCount      = static_cast<int>(view.moleculeAtomOffsets[source + 1]) - leftStart;
  const int candidateAtoms = static_cast<int>(view.moleculeAtomOffsets[candidate + 1]) - rightStart;
  const int squareSize     = max(leftCount, candidateAtoms);

  for (int linear = static_cast<int>(threadIdx.x); linear < squareSize * squareSize; linear += kThreads) {
    const int row    = linear / squareSize;
    const int column = linear % squareSize;
    float     score  = 0.0F;
    if (row < leftCount && column < candidateAtoms &&
        view.atomNumbers[leftStart + row] == view.atomNumbers[rightStart + column] &&
        view.aromatic[leftStart + row] == view.aromatic[rightStart + column]) {
      std::int64_t leftPosition  = view.atomBinOffsets[leftStart + row];
      const auto   leftEnd       = view.atomBinOffsets[leftStart + row + 1];
      std::int64_t rightPosition = view.atomBinOffsets[rightStart + column];
      const auto   rightEnd      = view.atomBinOffsets[rightStart + column + 1];
      std::int64_t overlap       = 0;
      while (leftPosition < leftEnd && rightPosition < rightEnd) {
        const auto leftBin  = view.binIds[leftPosition];
        const auto rightBin = view.binIds[rightPosition];
        if (leftBin == rightBin) {
          overlap += min(view.binCounts[leftPosition], view.binCounts[rightPosition]);
          ++leftPosition;
          ++rightPosition;
        } else if (leftBin < rightBin) {
          ++leftPosition;
        } else {
          ++rightPosition;
        }
      }
      const auto largestPathCount =
        max(view.atomPathLengths[leftStart + row], view.atomPathLengths[rightStart + column]);
      score = static_cast<float>(overlap + 1) / static_cast<float>(2 * largestPathCount - overlap + 1);
    }
    affinity[row * kSharedStride + column]  = score;
    transport[row * kSharedStride + column] = score / temperature;
  }
  __syncthreads();

  for (int iteration = 0; iteration < iterations; ++iteration) {
    normalizeLines<true>(transport, squareSize);
    __syncthreads();
    normalizeLines<false>(transport, squareSize);
    __syncthreads();
  }

  float matched = 0.0F;
  for (int linear = static_cast<int>(threadIdx.x); linear < leftCount * candidateAtoms; linear += kThreads) {
    const int row    = linear / candidateAtoms;
    const int column = linear % candidateAtoms;
    matched += expf(transport[row * kSharedStride + column]) * affinity[row * kSharedStride + column];
  }
  matched = warpSum(matched);
  if (threadIdx.x % 32 == 0) {
    partials[threadIdx.x / 32] = matched;
  }
  __syncthreads();
  float total = 0.0F;
  for (int warp = 0; warp < kThreads / 32; ++warp) {
    total += partials[warp];
  }
  // The next pair overwrites the shared buffers read above.
  __syncthreads();
  return total / (2.0F * static_cast<float>(leftCount) - total + 1e-10F);
}

template <typename Op>
__global__ void aapDistanceKernel(const AapDeviceView view,
                                  const int*          candidates,
                                  const int*          candidateCount,
                                  const int*          sources,
                                  const int           numSources,
                                  const int           numItems,
                                  const float         temperature,
                                  const int           iterations,
                                  const Op            op) {
  __shared__ float affinity[kMaxAtoms * kSharedStride];
  __shared__ float transport[kMaxAtoms * kSharedStride];
  __shared__ float partials[kThreads / 32];

  const int count = *candidateCount;
  for (int position = static_cast<int>(blockIdx.x); position < count; position += static_cast<int>(gridDim.x)) {
    const int candidate = candidates[position];
    auto      state     = op.start(candidate);
    for (int ordinal = 0; ordinal < numSources; ++ordinal) {
      const int source = sources[ordinal];
      if (source < 0 || source >= numItems) {
        continue;
      }
      // Identical descriptors score exactly 1 regardless of Sinkhorn convergence.
      const float distance =
        view.groups[candidate] == view.groups[source] ?
          0.0F :
          1.0F - aapPairSimilarity(view, source, candidate, temperature, iterations, affinity, transport, partials);
      if (threadIdx.x == 0) {
        op.visit(state, ordinal, source == candidate, distance);
      }
    }
    if (threadIdx.x == 0) {
      op.finish(candidate, state);
    }
  }
}

class AapDistanceProvider {
 public:
  AapDistanceProvider(const std::vector<const RDKit::ROMol*>& molecules, const AapOptions& options, cudaStream_t stream)
      : options_(options),
        numItems_(static_cast<int>(molecules.size())),
        hostDescriptors_(buildDescriptors(molecules, options)),
        descriptors_(hostDescriptors_, stream),
        groups_(numItems_, stream),
        candidates_(numItems_, stream),
        candidateCount_(0, stream) {
    groups_.copyFromHost(descriptorGroups(hostDescriptors_));
    int device = 0;
    int numSms = 0;
    cudaCheckError(cudaGetDevice(&device));
    cudaCheckError(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device));
    gridSize_ = std::max(1, std::min(numItems_, numSms * kBlocksPerSm));
  }

  int size() const { return numItems_; }
  //! Sinkhorn pairs are expensive, so Leader resolves one candidate at a time rather than scoring rejected ones.
  int leaderWindow() const { return 1; }

  template <typename Op>
  void accumulateDistances(const int* sources, const int numSources, const Op& op, cudaStream_t stream) {
    if (numItems_ == 0 || numSources == 0) {
      return;
    }
    cudaCheckError(cudaMemsetAsync(candidateCount_.data(), 0, sizeof(int), stream));
    compactCandidatesKernel<<<(numItems_ + kThreads - 1) / kThreads, kThreads, 0, stream>>>(numItems_,
                                                                                            op,
                                                                                            candidates_.data(),
                                                                                            candidateCount_.data());
    cudaCheckError(cudaGetLastError());
    const AapDeviceView view{descriptors_.moleculeAtomOffsets.data(),
                             descriptors_.atomBinOffsets.data(),
                             descriptors_.atomPathLengths.data(),
                             descriptors_.atomNumbers.data(),
                             descriptors_.aromatic.data(),
                             descriptors_.binIds.data(),
                             descriptors_.binCounts.data(),
                             groups_.data()};
    aapDistanceKernel<<<gridSize_, kThreads, 0, stream>>>(view,
                                                          candidates_.data(),
                                                          candidateCount_.data(),
                                                          sources,
                                                          numSources,
                                                          numItems_,
                                                          options_.sinkhornTemperature,
                                                          options_.sinkhornIterations,
                                                          op);
    cudaCheckError(cudaGetLastError());
  }

 private:
  static constexpr int kBlocksPerSm = 4;

  AapOptions             options_;
  int                    numItems_;
  AapHostDescriptors     hostDescriptors_;
  AapDeviceDescriptors   descriptors_;
  AsyncDeviceVector<int> groups_;
  AsyncDeviceVector<int> candidates_;
  AsyncDevicePtr<int>    candidateCount_;
  int                    gridSize_ = 1;
};

//! Stores one candidate's similarity to the source.
struct PairSimilarityOp {
  float* similarity;
  int    target;

  using State = float;
  __device__ bool  skip(const int candidate) const { return candidate != target; }
  __device__ State start(const int /*candidate*/) const { return 0.0F; }
  __device__ void  visit(State& state, const int /*ordinal*/, const bool /*self*/, const float distance) const {
    state = 1.0F - distance;
  }
  __device__ void finish(const int /*candidate*/, const State state) const { *similarity = state; }
};

}  // namespace

float aapSimilarityGpu(const RDKit::ROMol& left,
                       const RDKit::ROMol& right,
                       const AapOptions&   options,
                       cudaStream_t        stream) {
  validateOptions(options);
  AapDistanceProvider     provider({&left, &right}, options, stream);
  AsyncDevicePtr<int>     source(0, stream);
  AsyncDevicePtr<float>   similarity(0.0F, stream);
  PinnedHostVector<float> similarityHost(1);
  provider.accumulateDistances(source.data(), 1, PairSimilarityOp{similarity.data(), 1}, stream);
  cudaCheckError(
    cudaMemcpyAsync(similarityHost.data(), similarity.data(), sizeof(float), cudaMemcpyDeviceToHost, stream));
  cudaCheckError(cudaStreamSynchronize(stream));
  return similarityHost[0];
}

PickerResult aapLeader(const std::vector<const RDKit::ROMol*>& molecules,
                       const double                            cutoff,
                       const AapOptions&                       options,
                       const int                               pickSize,
                       const std::vector<int>&                 firstPicks,
                       cudaStream_t                            stream) {
  validateOptions(options);
  detail::validateUnitCutoff(cutoff);
  AapDistanceProvider provider(molecules, options, stream);
  return detail::leaderPick(provider, static_cast<float>(cutoff), pickSize, firstPicks, nullptr, stream);
}

PickerResult aapMaxMin(const std::vector<const RDKit::ROMol*>& molecules,
                       const int                               pickSize,
                       const AapOptions&                       options,
                       const std::vector<int>&                 firstPicks,
                       const int                               seed,
                       const double                            threshold,
                       cudaStream_t                            stream) {
  validateOptions(options);
  detail::validateMaxMinThreshold(threshold, 1.0);
  AapDistanceProvider provider(molecules, options, stream);
  return detail::maxMinPick(provider, pickSize, firstPicks, seed, static_cast<float>(threshold), stream);
}

ClusteringResult aapDise(const std::vector<const RDKit::ROMol*>& molecules,
                         const double                            cutoff,
                         const AapOptions&                       options,
                         const bool                              nearestAssignment,
                         cudaStream_t                            stream) {
  validateOptions(options);
  detail::validateUnitCutoff(cutoff);
  AapDistanceProvider provider(molecules, options, stream);
  return detail::diseCluster(provider, static_cast<float>(cutoff), nearestAssignment, stream);
}

}  // namespace nvMolKit
