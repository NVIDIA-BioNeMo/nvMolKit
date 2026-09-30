// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_DESCRIPTORS3D_PAIRWISE_CUH
#define NVMOLKIT_DESCRIPTORS3D_PAIRWISE_CUH

#include <cmath>
#include <type_traits>

#include "src/descriptors3d.h"
#include "src/descriptors3d_kernel.cuh"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"

namespace nvMolKit::descriptors3d_detail {

//! Unweighted channel plus the atom-property channels, in RDKit's output order (u, m, v, e, p, i, s).
constexpr int kNumPairwiseChannels = kNumAtomPropertyChannels + 1;
constexpr int kNumRdfRadii         = kNumRdfProperties / kNumPairwiseChannels;
constexpr int kNumMorseScatterings = kNumMorseProperties / kNumPairwiseChannels;
//! Bins per property owned by each lane of a bin group: lane l owns bins l, l + kGroupSize, ...
constexpr int kPairwiseBinsPerLane = (kNumMorseScatterings + kGroupSize - 1) / kGroupSize;
//! One warp per conformer: its kGroupSize-lane groups walk interleaved rows of the atom-pair triangle.
constexpr int kPairStreams         = kWarpSize / kGroupSize;
static_assert(kNumRdfRadii <= kPairwiseBinsPerLane * kGroupSize);
//! AUTOCORR3D adds RDKit's relative covalent radius after the pairwise channels (u, m, v, e, p, i, s, r).
constexpr int kNumAutocorrChannels = kNumPairwiseChannels + 1;
constexpr int kNumAutocorrLags     = kNumAutocorr3DProperties / kNumAutocorrChannels;
constexpr int kAutocorrLagsPerLane = (kNumAutocorrLags + kGroupSize - 1) / kGroupSize;

//! Pairwise properties served by one kernel instantiation; unrequested ones compile out.
enum PairwiseSet : unsigned {
  kPairwiseRdf        = 1u,
  kPairwiseMorse      = 2u,
  kPairwiseAutocorr3D = 4u,
};

//! Per-atom scratch row written once per conformer: centered x, y, z, the atom-property channels (the
//! last one MORSE's I-state), RDF's I-state, then AUTOCORR3D's covalent radius, all as the compute type.
constexpr int kPairwiseScratchStride = 3 + kNumAtomPropertyChannels + 2;
constexpr int kScratchIStateDrag     = kPairwiseScratchStride - 2;
constexpr int kScratchCovalentRadius = kPairwiseScratchStride - 1;

//! Per-atom channel weights; channel 0 is the unit weight.
template <typename Real> struct PairwiseAtom {
  Real x, y, z;
  Real weights[kNumPairwiseChannels];
};

template <typename Real> __device__ __forceinline__ PairwiseAtom<Real> loadPairwiseAtom(const Real* row) {
  PairwiseAtom<Real> atom;
  atom.x          = row[0];
  atom.y          = row[1];
  atom.z          = row[2];
  atom.weights[0] = Real(1);
  for (int channel = 1; channel < kNumPairwiseChannels; ++channel) {
    atom.weights[channel] = row[2 + channel];
  }
  return atom;
}

template <typename Real> __device__ __forceinline__ void sinCos(const Real x, Real& sine, Real& cosine) {
  if constexpr (std::is_same_v<Real, float>) {
    sincosf(x, &sine, &cosine);
  } else {
    sincos(x, &sine, &cosine);
  }
}

//! Accumulates one term into every channel of one bin: acc[c] += w_j[c] * w_k[c] * term.
template <typename Real>
__device__ __forceinline__ void accumulatePairTerm(Real (&acc)[kNumPairwiseChannels],
                                                   const Real (&products)[kNumPairwiseChannels],
                                                   const Real term) {
  for (int channel = 0; channel < kNumPairwiseChannels; ++channel) {
    acc[channel] += products[channel] * term;
  }
}

//! Writes one property's owned bins, rounded like RDKit, or NaN for an invalid row.
template <typename Real, typename OutputReal>
__device__ __forceinline__ void writePairwiseBins(const Real (&acc)[kPairwiseBinsPerLane][kNumPairwiseChannels],
                                                  const int   numBins,
                                                  const int   laneInGroup,
                                                  const bool  valid,
                                                  OutputReal* row) {
  for (int slot = 0; slot < kPairwiseBinsPerLane; ++slot) {
    const int bin = laneInGroup + slot * kGroupSize;
    if (bin < numBins) {
      for (int channel = 0; channel < kNumPairwiseChannels; ++channel) {
        row[channel * numBins + bin] =
          valid ? static_cast<OutputReal>(roundThousandths(acc[slot][channel])) : static_cast<OutputReal>(nan(""));
      }
    }
  }
}

/**
 * @brief Warp-collective. MORSE bin 0, sum over pairs j < k of w_j * w_k per channel, as
 *        ((sum w)^2 - sum w^2) / 2 in float64; the unweighted channel is n (n - 1) / 2.
 *
 * FP64 required: every bin-0 term is a positive weight product, so the sum grows to ~1e5 for
 * drug-sized molecules and float32 accumulation over the pair loop drifted by up to 19 (measured on ChEMBL
 * conformers). The closed form costs O(atoms) float64 work instead of O(pairs). Other bins oscillate in
 * sign, stay small, and are float32-accurate.
 */
__device__ __forceinline__ void computeMorseZeroBin(const ConformerAtoms& atoms,
                                                    const double*         propertyWeights,
                                                    const int             totalAtoms,
                                                    const int             lane,
                                                    double (&zeroBin)[kNumPairwiseChannels]) {
  const double numAtoms = atoms.numAtoms;
  zeroBin[0]            = 0.5 * numAtoms * (numAtoms - 1.0);
  for (int channel = 1; channel < kNumPairwiseChannels; ++channel) {
    const double* weights = propertyWeights + static_cast<size_t>(channel - 1) * totalAtoms;
    double        sum     = 0;
    double        squares = 0;
    for (int atomIdx = lane; atomIdx < atoms.numAtoms; atomIdx += kWarpSize) {
      sum += weights[atomIdx];
      squares += weights[atomIdx] * weights[atomIdx];
    }
    sum              = groupAllReduceSum<kWarpSize>(sum);
    zeroBin[channel] = 0.5 * (sum * sum - groupAllReduceSum<kWarpSize>(squares));
  }
}

/**
 * @brief The pairwise properties in @p kSet (PairwiseSet bits) for every conformer; one warp per conformer.
 *
 * The warp's kPairStreams groups of kGroupSize lanes take interleaved rows j of the atom-pair triangle
 * (pairs j < k), so every group of a warp works on the same conformer and molecules of different sizes do
 * not idle each other's lanes. Within a group, each lane owns a strided subset of each property's bins; the
 * groups' partial sums are combined with shuffles at the end. RDF bin i adds exp(-100 (R_i - r)^2) with
 * R_i = 1 + 0.5 i; MORSE bin i adds sin(i r) / (i r), or 1 for i = 0. Each term is scaled by the pair
 * product of the channel's atom weights. RDF's I-state channel uses `inputs.iStateDragWeights`; MORSE's uses
 * the I-state block of `inputs.atomPropertyWeights`, matching RDKit.
 *
 * AUTOCORR3D lag l (1-10) adds r * w_j * w_k over pairs whose bond-count distance is l, for the MORSE
 * channels plus covalent radius; the lane owning the lag accumulates it. RDKit evaluates each sum as
 * w^T (B_l .* D) w, so a NaN weight anywhere in a channel makes every lag of that channel NaN, which RDKit
 * then reports as 0; the kernel reproduces that per channel. Sums count both orders of each pair and are
 * divided by n (n - 1).
 *
 * The warp first writes one @p scratch row per atom (kPairwiseScratchStride values: coordinates centered in
 * float64, see centeredPosition(), then the channel weights), so the pair loop reads only @p Real values.
 *
 * Each lane's RDF radii are 4 A apart, so only the one nearest r can contribute more than exp(-400) and
 * the others are skipped. MORSE's sines for the lane's scattering values s, s + 8, s + 16, s + 24 come from
 * sincos(s r) and sincos(8 r) by angle addition.
 */
template <typename Real, unsigned kSet>
__global__ void pairwise3DKernel(const DeviceCoordView        coordinates,
                                 const Property3DDeviceInputs inputs,
                                 Real* __restrict__ scratch,
                                 Real* __restrict__ rdfOutput,
                                 Real* __restrict__ morseOutput,
                                 Real* __restrict__ autocorrOutput) {
  constexpr bool kRdf         = (kSet & kPairwiseRdf) != 0;
  constexpr bool kMorse       = (kSet & kPairwiseMorse) != 0;
  constexpr bool kAutocorr    = (kSet & kPairwiseAutocorr3D) != 0;
  const int      lane         = static_cast<int>(threadIdx.x) % kWarpSize;
  const int      laneInGroup  = lane % kGroupSize;
  const int      stream       = lane / kGroupSize;
  const int      conformerIdx = blockIdx.x * kWarpsPerBlock + static_cast<int>(threadIdx.x) / kWarpSize;
  if (conformerIdx >= coordinates.numConformers) {
    return;  // Uniform across the warp.
  }
  const ConformerAtoms atoms = loadConformer(coordinates, nullptr, inputs.moleculeAtomStarts, conformerIdx);

  Real       rdfAcc[kPairwiseBinsPerLane][kNumPairwiseChannels]      = {};
  Real       morseAcc[kPairwiseBinsPerLane][kNumPairwiseChannels]    = {};
  double     morseZeroBin[kNumPairwiseChannels]                      = {};
  Real       autocorrAcc[kAutocorrLagsPerLane][kNumAutocorrChannels] = {};
  bool       autocorrChannelNaN[kNumAutocorrChannels]                = {};
  // First RDF radius and MORSE scattering value owned by this lane; later slots add kGroupSize bins.
  const Real firstRadius     = Real(1) + Real(0.5) * static_cast<Real>(laneInGroup);
  const Real firstScattering = static_cast<Real>(laneInGroup);
  Real       inverseScatterings[kPairwiseBinsPerLane];
  for (int slot = 0; slot < kPairwiseBinsPerLane; ++slot) {
    const int bin            = laneInGroup + slot * kGroupSize;
    // Bin 0 comes from computeMorseZeroBin(); 0 keeps its unused accumulator finite.
    inverseScatterings[slot] = bin == 0 ? Real(0) : Real(1) / static_cast<Real>(bin);
  }

  if (atoms.valid) {
    double sumX = 0;
    double sumY = 0;
    double sumZ = 0;
    for (int atomIdx = lane; atomIdx < atoms.numAtoms; atomIdx += kWarpSize) {
      sumX += atoms.positions[atomIdx * 3 + 0];
      sumY += atoms.positions[atomIdx * 3 + 1];
      sumZ += atoms.positions[atomIdx * 3 + 2];
    }
    const double  inverseAtoms    = 1.0 / static_cast<double>(atoms.numAtoms);
    const double  centroidX       = groupAllReduceSum<kWarpSize>(sumX) * inverseAtoms;
    const double  centroidY       = groupAllReduceSum<kWarpSize>(sumY) * inverseAtoms;
    const double  centroidZ       = groupAllReduceSum<kWarpSize>(sumZ) * inverseAtoms;
    const int     moleculeStart   = inputs.moleculeAtomStarts[coordinates.molIndices[conformerIdx]];
    const int     totalAtoms      = inputs.moleculeAtomStarts[coordinates.nMols];
    const double* propertyWeights = inputs.atomPropertyWeights + moleculeStart;
    const double* iStateDrag      = kRdf ? inputs.iStateDragWeights + moleculeStart : nullptr;
    const double* radius          = kAutocorr ? inputs.covalentRadiusWeights + moleculeStart : nullptr;
    Real* const   rows            = scratch + (atoms.positions - coordinates.positions) / 3 * kPairwiseScratchStride;
    for (int atomIdx = lane; atomIdx < atoms.numAtoms; atomIdx += kWarpSize) {
      Real* const row = rows + atomIdx * kPairwiseScratchStride;
      centeredPosition(atoms.positions, atomIdx, centroidX, centroidY, centroidZ, row[0], row[1], row[2]);
      for (int channel = 0; channel < kNumAtomPropertyChannels; ++channel) {
        row[3 + channel] = static_cast<Real>(propertyWeights[static_cast<size_t>(channel) * totalAtoms + atomIdx]);
      }
      row[kScratchIStateDrag]     = kRdf ? static_cast<Real>(iStateDrag[atomIdx]) : Real(0);
      row[kScratchCovalentRadius] = kAutocorr ? static_cast<Real>(radius[atomIdx]) : Real(0);
    }
    // Makes the scratch rows visible to every lane.
    __syncwarp();

    const uint8_t* topology = nullptr;
    if constexpr (kAutocorr) {
      topology = inputs.topologicalDistances + inputs.topologicalDistanceStarts[coordinates.molIndices[conformerIdx]];
      for (int channel = 1; channel < kNumAutocorrChannels; ++channel) {
        const double* weights =
          channel < kNumPairwiseChannels ? propertyWeights + static_cast<size_t>(channel - 1) * totalAtoms : radius;
        int nanCount = 0;
        for (int atomIdx = lane; atomIdx < atoms.numAtoms; atomIdx += kWarpSize) {
          nanCount += isnan(weights[atomIdx]) ? 1 : 0;
        }
        autocorrChannelNaN[channel] = groupAllReduceSum<kWarpSize>(nanCount) != 0;
      }
    }

    for (int j = stream; j < atoms.numAtoms - 1; j += kPairStreams) {
      const Real* const        rowJ    = rows + j * kPairwiseScratchStride;
      const PairwiseAtom<Real> first   = loadPairwiseAtom(rowJ);
      const Real               dragJ   = rowJ[kScratchIStateDrag];
      const Real               radiusJ = rowJ[kScratchCovalentRadius];
      // Packed upper-triangle index of pair (j, j + 1) in the topology rows.
      const int64_t rowPairStart = static_cast<int64_t>(j) * atoms.numAtoms - static_cast<int64_t>(j) * (j + 1) / 2;
      for (int k = j + 1; k < atoms.numAtoms; ++k) {
        const Real* const        rowK   = rows + k * kPairwiseScratchStride;
        const PairwiseAtom<Real> second = loadPairwiseAtom(rowK);
        const Real               dx     = first.x - second.x;
        const Real               dy     = first.y - second.y;
        const Real               dz     = first.z - second.z;
        const Real               r      = sqrt(dx * dx + dy * dy + dz * dz);
        Real                     products[kNumPairwiseChannels];
        for (int channel = 0; channel < kNumPairwiseChannels; ++channel) {
          products[channel] = first.weights[channel] * second.weights[channel];
        }
        if constexpr (kMorse) {
          Real sine;
          Real cosine;
          Real stepSine;
          Real stepCosine;
          sinCos(firstScattering * r, sine, cosine);
          sinCos(static_cast<Real>(kGroupSize) * r, stepSine, stepCosine);
          const Real inverseR = Real(1) / r;
          for (int slot = 0; slot < kPairwiseBinsPerLane; ++slot) {
            accumulatePairTerm(morseAcc[slot], products, sine * inverseScatterings[slot] * inverseR);
            const Real nextSine = sine * stepCosine + cosine * stepSine;
            cosine              = cosine * stepCosine - sine * stepSine;
            sine                = nextSine;
          }
        }
        if constexpr (kAutocorr) {
          const int lag = topology[rowPairStart + (k - j - 1)];
          if (lag != 0 && (lag - 1) % kGroupSize == laneInGroup) {
            // A select on the slot, not an index, keeps autocorrAcc in registers.
            const int  lagSlot       = (lag - 1) / kGroupSize;
            const Real radiusProduct = radiusJ * rowK[kScratchCovalentRadius];
            for (int slot = 0; slot < kAutocorrLagsPerLane; ++slot) {
              const Real distance = slot == lagSlot ? r : Real(0);
              for (int channel = 0; channel < kNumPairwiseChannels; ++channel) {
                autocorrAcc[slot][channel] += products[channel] * distance;
              }
              autocorrAcc[slot][kNumPairwiseChannels] += radiusProduct * distance;
            }
          }
        }
        // RDF replaces the I-state product, so it runs after every property that uses it.
        if constexpr (kRdf) {
          products[kNumPairwiseChannels - 1] = dragJ * rowK[kScratchIStateDrag];
          const int nearest =
            min(max(static_cast<int>(rint((r - firstRadius) * Real(0.25))), 0), kPairwiseBinsPerLane - 1);
          const Real offset = firstRadius + Real(4) * static_cast<Real>(nearest) - r;
          const Real term   = exp(Real(-100) * offset * offset);
          // A select, not a branch on the slot, keeps rdfAcc in registers.
          for (int slot = 0; slot < kPairwiseBinsPerLane; ++slot) {
            accumulatePairTerm(rdfAcc[slot], products, slot == nearest ? term : Real(0));
          }
        }
      }
    }
    if constexpr (kMorse) {
      computeMorseZeroBin(atoms, propertyWeights, totalAtoms, lane, morseZeroBin);
    }
  }
  // Combine the pair streams' partial sums; lanes with equal laneInGroup own the same bins.
  for (int slot = 0; slot < kPairwiseBinsPerLane; ++slot) {
    for (int channel = 0; channel < kNumPairwiseChannels; ++channel) {
      for (int offset = kGroupSize; offset < kWarpSize; offset <<= 1) {
        if constexpr (kRdf) {
          rdfAcc[slot][channel] += __shfl_xor_sync(0xffffffffu, rdfAcc[slot][channel], offset);
        }
        if constexpr (kMorse) {
          morseAcc[slot][channel] += __shfl_xor_sync(0xffffffffu, morseAcc[slot][channel], offset);
        }
      }
    }
  }
  if constexpr (kAutocorr) {
    for (int slot = 0; slot < kAutocorrLagsPerLane; ++slot) {
      for (int channel = 0; channel < kNumAutocorrChannels; ++channel) {
        for (int offset = kGroupSize; offset < kWarpSize; offset <<= 1) {
          autocorrAcc[slot][channel] += __shfl_xor_sync(0xffffffffu, autocorrAcc[slot][channel], offset);
        }
      }
    }
  }
  if (stream != 0) {
    return;
  }

  if constexpr (kRdf) {
    writePairwiseBins(rdfAcc,
                      kNumRdfRadii,
                      laneInGroup,
                      atoms.valid,
                      rdfOutput + static_cast<size_t>(conformerIdx) * kNumRdfProperties);
  }
  if constexpr (kMorse) {
    Real* const row = morseOutput + static_cast<size_t>(conformerIdx) * kNumMorseProperties;
    writePairwiseBins(morseAcc, kNumMorseScatterings, laneInGroup, atoms.valid, row);
    if (laneInGroup == 0 && atoms.valid) {
      for (int channel = 0; channel < kNumPairwiseChannels; ++channel) {
        row[channel * kNumMorseScatterings] = static_cast<Real>(roundThousandths(morseZeroBin[channel]));
      }
    }
  }
  if constexpr (kAutocorr) {
    Real* const  row          = autocorrOutput + static_cast<size_t>(conformerIdx) * kNumAutocorr3DProperties;
    const double orderedPairs = static_cast<double>(atoms.numAtoms) * (atoms.numAtoms - 1);
    for (int slot = 0; slot < kAutocorrLagsPerLane; ++slot) {
      const int lagIdx = laneInGroup + slot * kGroupSize;
      if (lagIdx < kNumAutocorrLags) {
        for (int channel = 0; channel < kNumAutocorrChannels; ++channel) {
          // One atom gives 0 / 0 = NaN, as in RDKit.
          const double sum = autocorrChannelNaN[channel] ? 0.0 : 2.0 * static_cast<double>(autocorrAcc[slot][channel]);
          row[channel * kNumAutocorrLags + lagIdx] =
            atoms.valid ? static_cast<Real>(roundThousandths(sum / orderedPairs)) : static_cast<Real>(nan(""));
        }
      }
    }
  }
}

//! Output rows of the pairwise properties; null for properties that were not requested.
template <typename Real> struct PairwiseOutputs {
  Real* rdf        = nullptr;
  Real* morse      = nullptr;
  Real* autocorr3D = nullptr;
};

template <typename Real, unsigned kSet>
void launchPairwiseSet(const DeviceCoordView&        coordinates,
                       const Property3DDeviceInputs& inputs,
                       Real*                         scratch,
                       const PairwiseOutputs<Real>&  outputs,
                       const int                     numBlocks,
                       const cudaStream_t            stream) {
  pairwise3DKernel<Real, kSet><<<numBlocks, kBlockSize, 0, stream>>>(coordinates,
                                                                     inputs,
                                                                     scratch,
                                                                     outputs.rdf,
                                                                     outputs.morse,
                                                                     outputs.autocorr3D);
}

//! Launches the pairwise kernel when RDF, MORSE or AUTOCORR3D is requested; one pass over atom pairs serves
//! all of them.
template <typename Real>
void launchPairwiseProperties(const DeviceCoordView&        coordinates,
                              const Property3DDeviceInputs& inputs,
                              const PairwiseOutputs<Real>&  outputs,
                              const cudaStream_t            stream) {
  const unsigned set = (outputs.rdf != nullptr ? kPairwiseRdf : 0u) | (outputs.morse != nullptr ? kPairwiseMorse : 0u) |
                       (outputs.autocorr3D != nullptr ? kPairwiseAutocorr3D : 0u);
  const int numConformers = coordinates.numConformers;
  if (numConformers == 0 || set == 0) {
    return;
  }
  const int               numBlocks = (numConformers + kWarpsPerBlock - 1) / kWarpsPerBlock;
  AsyncDeviceVector<Real> scratch(static_cast<size_t>(coordinates.numAtoms) * kPairwiseScratchStride, stream);
  Real* const             rows = scratch.data();
  switch (set) {
    case kPairwiseRdf:
      launchPairwiseSet<Real, kPairwiseRdf>(coordinates, inputs, rows, outputs, numBlocks, stream);
      break;
    case kPairwiseMorse:
      launchPairwiseSet<Real, kPairwiseMorse>(coordinates, inputs, rows, outputs, numBlocks, stream);
      break;
    case kPairwiseRdf | kPairwiseMorse:
      launchPairwiseSet<Real, kPairwiseRdf | kPairwiseMorse>(coordinates, inputs, rows, outputs, numBlocks, stream);
      break;
    case kPairwiseAutocorr3D:
      launchPairwiseSet<Real, kPairwiseAutocorr3D>(coordinates, inputs, rows, outputs, numBlocks, stream);
      break;
    case kPairwiseRdf | kPairwiseAutocorr3D:
      launchPairwiseSet<Real, kPairwiseRdf | kPairwiseAutocorr3D>(coordinates,
                                                                  inputs,
                                                                  rows,
                                                                  outputs,
                                                                  numBlocks,
                                                                  stream);
      break;
    case kPairwiseMorse | kPairwiseAutocorr3D:
      launchPairwiseSet<Real, kPairwiseMorse | kPairwiseAutocorr3D>(coordinates,
                                                                    inputs,
                                                                    rows,
                                                                    outputs,
                                                                    numBlocks,
                                                                    stream);
      break;
    default:
      launchPairwiseSet<Real, kPairwiseRdf | kPairwiseMorse | kPairwiseAutocorr3D>(coordinates,
                                                                                   inputs,
                                                                                   rows,
                                                                                   outputs,
                                                                                   numBlocks,
                                                                                   stream);
      break;
  }
  cudaCheckError(cudaGetLastError());
}

}  // namespace nvMolKit::descriptors3d_detail

#endif  // NVMOLKIT_DESCRIPTORS3D_PAIRWISE_CUH
