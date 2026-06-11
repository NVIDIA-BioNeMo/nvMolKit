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

#include <cooperative_groups.h>

#include <cassert>
#include <cub/cub.cuh>
#include <cuda/std/span>
#include <cuda/std/tuple>
namespace cg = cooperative_groups;

#include "src/data_structures/flat_bit_vect.h"
#include "src/morgan_fingerprint_kernels.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"

namespace nvMolKit {

namespace {
constexpr int bondStride = 8;

}  // namespace

using neighborHoodInvariantType = cuda::std::pair<std::int32_t, std::uint32_t>;

__device__ void insertionSort(neighborHoodInvariantType* arr, int N) {
  for (int i = 1; i < N; i++) {
    neighborHoodInvariantType key = arr[i];
    int                       j   = i - 1;

    // Move elements of arr[0..i-1], that are greater than key,
    // to one position ahead of their current position
    while (j >= 0 && arr[j] > key) {
      arr[j + 1] = arr[j];
      j -= 1;
    }
    arr[j + 1] = key;
  }
}

template <typename T> __device__ __forceinline__ void hashCombine(std::uint32_t& seed, const T& value) {
  seed ^= static_cast<std::uint32_t>(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);  // Taken from boost hash.hpp
}

__device__ __forceinline__ void hashCombinePair(std::uint32_t& seed, const neighborHoodInvariantType& value) {
  std::uint32_t pairSeed = 0;
  hashCombine(pairSeed, value.first);
  hashCombine(pairSeed, value.second);
  hashCombine(seed, pairSeed);
}

struct AccumTupleLess {
  template <std::size_t maxAtoms>
  __device__ bool operator()(
    const cuda::std::tuple<nvMolKit::FlatBitVect<maxAtoms>, std::uint32_t, std::uint32_t>& lhs,
    const cuda::std::tuple<nvMolKit::FlatBitVect<maxAtoms>, std::uint32_t, std::uint32_t>& rhs) {
    if (cuda::std::get<0>(lhs) != cuda::std::get<0>(rhs)) {
      return cuda::std::get<0>(lhs) < cuda::std::get<0>(rhs);
    }
    if (cuda::std::get<1>(lhs) != cuda::std::get<1>(rhs)) {
      return cuda::std::get<1>(lhs) < cuda::std::get<1>(rhs);
    }
    return cuda::std::get<2>(lhs) < cuda::std::get<2>(rhs);
  }
};

template <std::size_t bitFieldSize>
__device__ __forceinline__ void atomicSetBit(FlatBitVect<bitFieldSize>* vec, int bit) {
  const int wordIdx = bit / vec->kStorageBits;
  const int bitIdx  = bit % vec->kStorageBits;
  assert(vec->kStorageBits == 32);
  std::uint32_t* word = vec->begin() + wordIdx;

  atomicOr(word, 1 << bitIdx);
}

// Simple wrapper to not do accidental bad math bools vs ints.
struct DeadAtomHolder {
  int value;
};

__forceinline__ __device__ bool isDeadAtom(const DeadAtomHolder* deadAtomsArray, const int atomIdx) {
  return deadAtomsArray[atomIdx].value == 1;
}

__forceinline__ __device__ void setDeadAtom(DeadAtomHolder* deadAtomsArray, const int atomIdx, const bool isDead) {
  deadAtomsArray[atomIdx].value = int(isDead);
}

template <size_t maxAtoms>
__forceinline__ __device__ int populateThisRoundNeighborhoods(
  nvMolKit::FlatBitVect<maxAtoms>&                             roundAtomNeighborhoods,  // output
  cuda::std::array<neighborHoodInvariantType, bondStride>&     neighborhoodInvariants,  // output
  const cuda::std::span<const nvMolKit::FlatBitVect<maxAtoms>> atomNeighborhoodsArray,
  const cuda::std::span<const std::uint32_t>                   bondInvariants,
  const cuda::std::span<const std::int16_t>                    atomBondIndices,
  const cuda::std::span<const std::int16_t>                    atomBondOtherAtomIndices,
  const std::uint32_t                                          currentInvariantsArray[maxAtoms]) {
  roundAtomNeighborhoods.clear();
  int numberOfBondsThisAtom = 0;
  for (size_t localBondAccessorIdx = 0; localBondAccessorIdx < bondStride; localBondAccessorIdx++) {
    const int bondIdx      = atomBondIndices[localBondAccessorIdx];
    const int otherAtomIdx = atomBondOtherAtomIndices[localBondAccessorIdx];
    if (bondIdx == -1) {
      break;
    }
    roundAtomNeighborhoods.setBit(bondIdx, true);
    roundAtomNeighborhoods |= atomNeighborhoodsArray[otherAtomIdx];

    const auto bondType                          = bondInvariants[bondIdx];
    neighborhoodInvariants[localBondAccessorIdx] = {static_cast<std::int32_t>(bondType),
                                                    currentInvariantsArray[otherAtomIdx]};
    numberOfBondsThisAtom++;
  }
  return numberOfBondsThisAtom;
}

template <size_t nBits>
__forceinline__ __device__ bool findMatchingNeighborhood(
  const nvMolKit::FlatBitVect<nBits>&                       thisSortedNeighborhood,
  const cuda::std::span<const nvMolKit::FlatBitVect<nBits>> searchArray,
  const cuda::std::span<const int>                          sortOrderings) {
  const int searchSize = static_cast<int>(searchArray.size());
  const int orderSize  = static_cast<int>(sortOrderings.size());
  const int loopEnd    = (searchSize < orderSize) ? searchSize : orderSize;
  for (int i = 0; i < loopEnd; i++) {
    const int idx = sortOrderings[i];
    if (idx < 0 || idx >= searchSize) {
      continue;  // skip invalid lanes (e.g., filler entries)
    }
    const auto& searchNeighborhood = searchArray[idx];
    if (searchNeighborhood == thisSortedNeighborhood) {
      return true;
    }
  }
  return false;
}

constexpr int kBlockSize = 128;
template <std::size_t maxAtoms, int fpSize>
__global__ void morganFingerprintKernelBatch(const cuda::std::span<std::uint32_t> atomInvariants,  // mutable
                                             [[maybe_unused]] const cuda::std::span<const std::uint32_t> bondInvariants,
                                             const cuda::std::span<const std::int16_t>                   bondIndices,
                                             const cuda::std::span<const std::int16_t> bondOtherAtomIndices,
                                             const cuda::std::span<const std::int16_t> nAtomsPerMolArray,

                                             cuda::std::span<FlatBitVect<maxAtoms>>     allSeenNeighborhoods,
                                             const cuda::std::span<FlatBitVect<fpSize>> outputAccumulator,
                                             const size_t                               maxRadius,
                                             const cuda::std::span<const int>           outputIndices) {
  static_assert(maxAtoms == 32 || maxAtoms == 64 || maxAtoms == 128, "maxAtoms must be 32, 64, or 128");
  using AccumTuple     = cuda::std::tuple<nvMolKit::FlatBitVect<maxAtoms>, std::uint32_t, std::uint32_t>;
  using BlockMergeSort = cub::BlockMergeSort<AccumTuple, kBlockSize, 1>;
  // Sorting strategy by specialization:
  // - maxAtoms == 128: use BlockMergeSort once (one tile per block)
  // - maxAtoms == 64 : use WarpMergeSort with 2 items per lane (first warp in tile)
  // - maxAtoms == 32 : use WarpMergeSort with 1 item per lane (one warp per tile)

  const AccumTuple outOfBoundsFillerMaxValue =
    cuda::std::make_tuple(FlatBitVect<maxAtoms>(true), 0xFFFFFFFF, 0xFFFFFFFF);

  // If we're running on cc < 8.0, we need to manually allocate shared memory for the cooperative group if it's
  // larger than a warp.
  __shared__ cg::block_tile_memory<maxAtoms> shared;
  cg::thread_block                           block = this_thread_block(shared);
  auto                                       tile  = cg::tiled_partition<maxAtoms>(block);

  // Each block is split into tiles of size maxAtoms. Each tile processes one molecule.
  // TODO: For maxAtoms == 32 (one warp), consider warp-level shuffles and reductions for additional speedups.
  constexpr int tilesPerBlock    = kBlockSize / static_cast<int>(maxAtoms);
  const int     tileId           = tile.meta_group_rank();
  const int     atomIdx          = tile.thread_rank();
  const int     nMolsInBatch     = nAtomsPerMolArray.size();
  const int     molIdx           = static_cast<int>(blockIdx.x) * tilesPerBlock + tileId;
  const bool    validTile        = molIdx < nMolsInBatch;
  const int     nAtomsInMolecule = validTile ? nAtomsPerMolArray[molIdx] : 0;

  if (!validTile || nAtomsInMolecule == 0) {
    return;
  }
  const int outputIdx = outputIndices[molIdx];

  assert(atomIdx < maxAtoms);
  assert(nAtomsInMolecule <= static_cast<int>(maxAtoms));
  assert(atomInvariants.size() == maxAtoms * nMolsInBatch);
  assert(bondInvariants.size() == maxAtoms * nMolsInBatch);
  assert(bondIndices.size() == maxAtoms * nMolsInBatch * bondStride);
  assert(bondOtherAtomIndices.size() == maxAtoms * nMolsInBatch * bondStride);
  assert(allSeenNeighborhoods.size() == maxAtoms * nMolsInBatch * (maxRadius + 1));
  assert(outputAccumulator.size() > outputIdx);
  // Active in the sense that we have an atom to process.
  // All threads must take part in CUB sort - so no breaking out of the function/main loop, and
  // be careful about untaken code paths with syncs to avoid hangs.
  const bool activeThread = atomIdx < nAtomsInMolecule;

  // Grab the local view into data we need.
  const size_t                              bondIndexingOffset = molIdx * maxAtoms * bondStride + atomIdx * bondStride;
  const cuda::std::span<const std::int16_t> atomBondIndices    = bondIndices.subspan(bondIndexingOffset, bondStride);
  const cuda::std::span<const std::int16_t> atomBondOtherAtomIndices =
    bondOtherAtomIndices.subspan(bondIndexingOffset, bondStride);

  const cuda::std::span<const std::uint32_t> atomInvariantsThisMol =
    atomInvariants.subspan(molIdx * maxAtoms, maxAtoms);
  const cuda::std::span<const std::uint32_t> bondInvariantsThisMol =
    bondInvariants.subspan(molIdx * maxAtoms, maxAtoms);

  const cuda::std::span<FlatBitVect<maxAtoms>> allSeenNeighborhoodsThisMol =
    allSeenNeighborhoods.subspan(molIdx * maxAtoms * (maxRadius + 1), maxAtoms * (maxRadius + 1));
  // ------------------------------------------------------------
  // Construct local shared arrays - these are one per atom, that will require reads AND writes between threads
  // ------------------------------------------------------------
  // Shared memory is allocated for the whole block; index with tileOffset to get the per-tile slice
  constexpr int             tileSliceSize = static_cast<int>(maxAtoms);
  __shared__ DeadAtomHolder deadAtomsArray[kBlockSize];  // kBlockSize ints
  // Note these next two should be sized to the max number of bonds, not the maxAtoms, but we're making a clean round.
  __shared__                nvMolKit::FlatBitVect<maxAtoms>
                            roundAtomNeighborhoodsArray[kBlockSize];              // kBlockSize * (maxAtoms / 8) bytes
  __shared__ nvMolKit::FlatBitVect<maxAtoms> atomNeighborhoodsArray[kBlockSize];  // kBlockSize * (maxAtoms / 8) bytes
  __shared__ std::uint32_t currentInvariantsArray[kBlockSize];                    // kBlockSize ints
  __shared__ std::uint32_t nextInvariantsArray[kBlockSize];                       // kBlockSize ints
  __shared__ int           sortOrderings[kBlockSize];                             // kBlockSize ints

  __shared__ FlatBitVect<fpSize> localUpdateAccumulator[tilesPerBlock];  // one per tile
  __shared__ AccumTuple          sharedAccums[kBlockSize];               // one per thread, used for tile sort

  const int tileOffset = tileId * tileSliceSize;
  const int sharedIdx  = tileOffset + atomIdx;

  // ------------------------------------------------
  // Set up initial values
  // ------------------------------------------------
  if (tile.thread_rank() == 0) {
    localUpdateAccumulator[tileId].clear();
  }

  atomNeighborhoodsArray[sharedIdx].clear();

  // This is equivalent to atom->getDegree() == 0
  // It also takes care of the case where we're > the number of atoms in the molecule.
  const bool thisThreadIsDead = !activeThread || atomBondIndices[0] == -1;
  setDeadAtom(deadAtomsArray, sharedIdx, thisThreadIsDead);

  currentInvariantsArray[sharedIdx] = activeThread ? atomInvariantsThisMol[atomIdx] : 0u;
  nextInvariantsArray[sharedIdx]    = 0u;

  // ------------------------------------------------------------
  // Each thread gets one atom. So allocate local storage for it.
  // ------------------------------------------------------------
  cuda::std::array<neighborHoodInvariantType, bondStride> neighborhoodInvariants;
  AccumTuple                                              accum[1];

  // Do loop 0
  auto bit = atomInvariantsThisMol[atomIdx] % fpSize;
  tile.sync();
  if (activeThread) {
    atomicSetBit(&localUpdateAccumulator[tileId], bit);
  }

  for (size_t radius = 0; radius < maxRadius; radius++) {
    tile.sync();

    if (isDeadAtom(deadAtomsArray, sharedIdx)) {
      // sort to the back
      accum[0] = cuda::std::make_tuple(roundAtomNeighborhoodsArray[sharedIdx], 0xFFFFFFFF, atomIdx);
    } else {
      // ------------------------------------------------------------
      // Compute the new neighborhood of this atom (max 8 ops). == roundAtomNeighborhoods,
      // and Compute neighborhood invariants (max 8 ops).
      // ------------------------------------------------------------
      const int numberOfBondsThisAtom =
        populateThisRoundNeighborhoods<maxAtoms>(roundAtomNeighborhoodsArray[sharedIdx],
                                                 neighborhoodInvariants,
                                                 cuda::std::span(atomNeighborhoodsArray + tileOffset, tileSliceSize),
                                                 bondInvariantsThisMol,
                                                 atomBondIndices,
                                                 atomBondOtherAtomIndices,
                                                 currentInvariantsArray + tileOffset);

      // ------------------------------------------------------------
      // Sort neighborhood invariants (size 8). This is in-thread
      // TODO - consider bubble sort, there's rarely more than 1 swap (or even one swap) here.
      // ------------------------------------------------------------

      insertionSort(neighborhoodInvariants.data(), numberOfBondsThisAtom);

      // ------------------------------------------------------------
      // Compute the next layer invariant (loop over max 8). Be careful about setting currentInvariant, either at end of
      // prev loop or index from the global array.
      // ------------------------------------------------------------
      std::uint32_t invar = radius;
      hashCombine(invar, currentInvariantsArray[sharedIdx]);
      for (int i = 0; i < numberOfBondsThisAtom; i++) {
        hashCombinePair(invar, neighborhoodInvariants[i]);
      }
      nextInvariantsArray[sharedIdx] = invar;

      // ------------------------------------------------------------
      // Combine invariant with metadata for sorting (allNeighborhoodsThisRound equiv, needs (new neighborhood, invar,
      // atomidx)) Then sort, size nAtoms max, so cooperative.
      // ------------------------------------------------------------
      accum[0] = cuda::std::make_tuple(roundAtomNeighborhoodsArray[sharedIdx], invar, atomIdx);
    }  // endif deadatoms

    // Stage per-thread tuple into shared
    sharedAccums[sharedIdx] = accum[0];
    // Ensure shared staging visible within the tile
    tile.sync();

    // Specialized sorting per maxAtoms
    if constexpr (maxAtoms == 128) {
      __shared__ typename BlockMergeSort::TempStorage temp_storage_shuffle;  // block-wide temp storage for sort
      // One tile per block; safe to use block-wide sort
      AccumTuple                                      accumSeq[1];
      // Use valid count to push invalid lanes to the end
      accumSeq[0] = sharedAccums[sharedIdx];
      block.sync();
      BlockMergeSort(temp_storage_shuffle)
        .Sort(accumSeq, AccumTupleLess(), nAtomsInMolecule, outOfBoundsFillerMaxValue);
      block.sync();
      sharedAccums[sharedIdx] = accumSeq[0];
      block.sync();
    } else if constexpr (maxAtoms == 64) {
      __shared__ typename cub::WarpMergeSort<AccumTuple, 2, 32>::TempStorage warp_temp_64[tilesPerBlock];
      // First warp in the tile sorts the 64 items with 2 items per lane
      if (tile.thread_rank() < 32) {
        AccumTuple laneItems[2];
        const int  lane = tile.thread_rank();
        const int  idx0 = tileOffset + lane;
        const int  idx1 = tileOffset + 32 + lane;
        laneItems[0]    = (lane < nAtomsInMolecule) ? sharedAccums[idx0] : outOfBoundsFillerMaxValue;
        laneItems[1]    = ((32 + lane) < nAtomsInMolecule) ? sharedAccums[idx1] : outOfBoundsFillerMaxValue;
        cub::WarpMergeSort<AccumTuple, 2, 32>(warp_temp_64[tileId]).Sort(laneItems, AccumTupleLess());
        const int outIdx0     = tileOffset + 2 * lane;
        const int outIdx1     = tileOffset + 2 * lane + 1;
        sharedAccums[outIdx0] = laneItems[0];
        sharedAccums[outIdx1] = laneItems[1];
      }
      tile.sync();
    } else {  // maxAtoms == 32
      __shared__ typename cub::WarpMergeSort<AccumTuple, 1, 32>::TempStorage warp_temp_32[tilesPerBlock];
      // One warp per tile; each lane sorts one item
      AccumTuple                                                             laneItem[1];
      const int                                                              lane = tile.thread_rank();
      const int                                                              idx0 = tileOffset + lane;
      laneItem[0] = (lane < nAtomsInMolecule) ? sharedAccums[idx0] : outOfBoundsFillerMaxValue;
      cub::WarpMergeSort<AccumTuple, 1, 32>(warp_temp_32[tileId]).Sort(laneItem, AccumTupleLess());
      sharedAccums[idx0] = laneItem[0];
      tile.sync();
    }

    // Load back our tile's sorted tuple
    accum[0] = sharedAccums[sharedIdx];

    // ------------------------------------------------------------
    // Given above target index, check all prev indices for this neighborhood, and add to next layers outputs if not
    // found. Else mark dead.
    // ------------------------------------------------------------
    const AccumTuple& thisSortedAccumTuple     = accum[0];
    const auto&       thisSortedNeighborhood   = cuda::std::get<0>(thisSortedAccumTuple);
    const int         envAndNeighborhoodOffset = radius + 1;  // Our loop starts at 0, but that corresponds to radius 1.
    const size_t      origIndexSz              = cuda::std::get<2>(thisSortedAccumTuple);
    const int         origIndex                = static_cast<int>(origIndexSz);
    sortOrderings[sharedIdx]                   = origIndex;

    bool foundInPrevious = true;  // default true for inactive/invalid to keep them "dead"
    if (activeThread && origIndex >= 0 && origIndex < nAtomsInMolecule) {
      foundInPrevious = isDeadAtom(deadAtomsArray, tileOffset + origIndex);
    }

    // First check the current round for duplicates, ordered smaller than the current index.
    // This is a bit of a mess, but we need to check all the previous atoms in the current round
    tile.sync();  // To make sure sortOrderings is populated.

    bool foundInThisRound = findMatchingNeighborhood<maxAtoms>(
      thisSortedNeighborhood,
      cuda::std::span(roundAtomNeighborhoodsArray + tileOffset, nAtomsInMolecule),  // full original-indexed view
      cuda::std::span(sortOrderings + tileOffset, atomIdx));  // only check earlier sorted positions

    tile.sync();
    if (activeThread && foundInThisRound) {
      setDeadAtom(deadAtomsArray, tileOffset + static_cast<int>(origIndex), true);
      foundInPrevious = true;
    }
    tile.sync();

    for (int prevRadius = 0; activeThread && prevRadius < envAndNeighborhoodOffset; prevRadius++) {
      if (findMatchingNeighborhood<maxAtoms>(
            thisSortedNeighborhood,
            cuda::std::span(allSeenNeighborhoodsThisMol.subspan(prevRadius * maxAtoms, nAtomsInMolecule)),
            cuda::std::span(sortOrderings + tileOffset, nAtomsInMolecule))) {
        setDeadAtom(deadAtomsArray, tileOffset + static_cast<int>(origIndex), true);

        foundInPrevious = true;
        break;
      }
    }

    if (activeThread && !foundInPrevious) {
      // Note we're writing in thread order here, not using the original ordering.
      // TODO maybe use bloom filter since we don't need the actual values, just need to know if it's in there.
      allSeenNeighborhoodsThisMol[envAndNeighborhoodOffset * maxAtoms + atomIdx] = thisSortedNeighborhood;

      bit = nextInvariantsArray[tileOffset + static_cast<int>(origIndex)] % fpSize;
      // TODO maybe a block reduce instead with thread 0 writeout.
      atomicSetBit(&localUpdateAccumulator[tileId], bit);
    }

    // ------------------------------------------------------------
    // Swap current invariants with next layer invariants, or just index.
    // ------------------------------------------------------------
    tile.sync();
    currentInvariantsArray[sharedIdx] = nextInvariantsArray[sharedIdx];
    nextInvariantsArray[sharedIdx]    = 0;
    atomNeighborhoodsArray[sharedIdx] = roundAtomNeighborhoodsArray[sharedIdx];
  }
  tile.sync();
  if (tile.thread_rank() == 0 && validTile) {
    outputAccumulator[outputIdx] = localUpdateAccumulator[tileId];
  }
}

template <int fpSize>
void launchMorganFingerprintKernelBatch(const MorganGPUBuffersBatch&            buffers,
                                        AsyncDeviceVector<FlatBitVect<fpSize>>& outputAccumulator,
                                        const size_t                            maxRadius,
                                        const int                               maxAtoms,
                                        const int                               nMolecules,
                                        cudaStream_t                            stream) {
  const int tilesPerBlock = kBlockSize / maxAtoms;
  const int numBlocks     = (nMolecules + tilesPerBlock - 1) / tilesPerBlock;  // ceil division

  switch (maxAtoms) {
    case 32:
      morganFingerprintKernelBatch<32, fpSize>
        <<<numBlocks, kBlockSize, 0, stream>>>(toSpan(buffers.atomInvariants),
                                               toSpan(buffers.bondInvariants),
                                               toSpan(buffers.bondIndices),
                                               toSpan(buffers.bondOtherAtomIndices),
                                               toSpan(buffers.nAtomsPerMol),
                                               toSpan(buffers.allSeenNeighborhoods32),
                                               toSpan(outputAccumulator),
                                               maxRadius,
                                               toSpan(buffers.outputIndices));
      break;
    case 64:
      morganFingerprintKernelBatch<64, fpSize>
        <<<numBlocks, kBlockSize, 0, stream>>>(toSpan(buffers.atomInvariants),
                                               toSpan(buffers.bondInvariants),
                                               toSpan(buffers.bondIndices),
                                               toSpan(buffers.bondOtherAtomIndices),
                                               toSpan(buffers.nAtomsPerMol),
                                               toSpan(buffers.allSeenNeighborhoods64),
                                               toSpan(outputAccumulator),
                                               maxRadius,
                                               toSpan(buffers.outputIndices));
      break;
    case 128:
      morganFingerprintKernelBatch<128, fpSize>
        <<<numBlocks, kBlockSize, 0, stream>>>(toSpan(buffers.atomInvariants),
                                               toSpan(buffers.bondInvariants),
                                               toSpan(buffers.bondIndices),
                                               toSpan(buffers.bondOtherAtomIndices),
                                               toSpan(buffers.nAtomsPerMol),
                                               toSpan(buffers.allSeenNeighborhoods128),
                                               toSpan(outputAccumulator),
                                               maxRadius,
                                               toSpan(buffers.outputIndices));
      break;
    default:
      throw std::runtime_error("maxAtoms must be 32, 64, or 128");
  }
  cudaCheckError(cudaGetLastError());
}

}  // namespace nvMolKit

#define DEFINE_TEMPLATE(fpSize)                                       \
  template void nvMolKit::launchMorganFingerprintKernelBatch<fpSize>( \
    const nvMolKit::MorganGPUBuffersBatch&  buffers,                  \
    AsyncDeviceVector<FlatBitVect<fpSize>>& outputAccumulator,        \
    size_t                                  maxRadius,                \
    int                                     maxAtoms,                 \
    int                                     nMolecules,               \
    cudaStream_t                            stream);
DEFINE_TEMPLATE(128)
DEFINE_TEMPLATE(256)
DEFINE_TEMPLATE(512)
DEFINE_TEMPLATE(1024)
DEFINE_TEMPLATE(2048)
DEFINE_TEMPLATE(4096)
#undef DEFINE_TEMPLATE
