// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_DESCRIPTORS3D_PAIRWISE_CUH
#define NVMOLKIT_DESCRIPTORS3D_PAIRWISE_CUH

#include <cmath>

#include "src/descriptors3d.h"
#include "src/descriptors3d_kernel.cuh"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"

namespace nvMolKit::descriptors3d_detail {

//! Unweighted channel plus the atom-property channels, in RDKit's output order (u, m, v, e, p, i, s).
constexpr int kNumPairwiseChannels = kNumAtomPropertyChannels + 1;
constexpr int kNumRdfRadii         = kNumRdfProperties / kNumPairwiseChannels;
constexpr int kNumMorseScatterings = kNumMorseProperties / kNumPairwiseChannels;
//! Bins per property owned by each lane of a conformer group: lane l owns bins l, l + kGroupSize, ...
constexpr int kPairwiseBinsPerLane = (kNumMorseScatterings + kGroupSize - 1) / kGroupSize;
static_assert(kNumRdfRadii <= kPairwiseBinsPerLane * kGroupSize);

//! Per-atom channel weights; channel 0 is the unit weight.
template <typename Real> struct PairwiseAtom {
  Real x, y, z;
  Real weights[kNumPairwiseChannels];
};

/**
 * @brief Load atom @p atomIdx's centered coordinates and weights. Channels 1-5 come from
 *        @p atomPropertyWeights; channel 6 from @p iStateWeights, which points at the I-state variant the
 *        property uses.
 */
template <typename Real>
__device__ __forceinline__ PairwiseAtom<Real> loadPairwiseAtom(const Real*   centered,
                                                               const double* atomPropertyWeights,
                                                               const double* iStateWeights,
                                                               const int     totalAtoms,
                                                               const int     atomIdx) {
  PairwiseAtom<Real> atom;
  atom.x          = centered[atomIdx * 3 + 0];
  atom.y          = centered[atomIdx * 3 + 1];
  atom.z          = centered[atomIdx * 3 + 2];
  atom.weights[0] = Real(1);
  for (int channel = 1; channel < kNumPairwiseChannels - 1; ++channel) {
    atom.weights[channel] =
      static_cast<Real>(atomPropertyWeights[static_cast<size_t>(channel - 1) * totalAtoms + atomIdx]);
  }
  atom.weights[kNumPairwiseChannels - 1] = static_cast<Real>(iStateWeights[atomIdx]);
  return atom;
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
 * @brief Group-collective. MORSE bin 0, sum over pairs j < k of w_j * w_k per channel, as
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
                                                    const int             laneInGroup,
                                                    double (&zeroBin)[kNumPairwiseChannels]) {
  const double numAtoms = atoms.numAtoms;
  zeroBin[0]            = 0.5 * numAtoms * (numAtoms - 1.0);
  for (int channel = 1; channel < kNumPairwiseChannels; ++channel) {
    const double* weights = propertyWeights + static_cast<size_t>(channel - 1) * totalAtoms;
    double        sum     = 0;
    double        squares = 0;
    for (int atomIdx = laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
      sum += weights[atomIdx];
      squares += weights[atomIdx] * weights[atomIdx];
    }
    sum              = groupAllReduceSum(sum);
    zeroBin[channel] = 0.5 * (sum * sum - groupAllReduceSum(squares));
  }
}

/**
 * @brief RDF and/or MORSE for every conformer; one group of kGroupSize lanes per conformer.
 *
 * Every lane of a group walks all atom pairs j < k in the same order and owns a strided subset of each
 * property's bins, so bins accumulate without reductions. RDF bin i adds exp(-100 (R_i - r)^2) with
 * R_i = 1 + 0.5 i; MORSE bin i adds sin(i r) / (i r), or 1 for i = 0. Each term is scaled by the pair
 * product of the channel's atom weights. RDF's I-state channel uses `inputs.iStateDragWeights`; MORSE's uses
 * the I-state block of `inputs.atomPropertyWeights`, matching RDKit.
 *
 * Each group first writes its conformer's coordinates, centered in float64 (see centeredPosition()), to
 * @p centered, laid out like `coordinates.positions`; the pair loop then reads only @p Real values.
 */
template <typename Real, bool kRdf, bool kMorse>
__global__ void pairwise3DKernel(const DeviceCoordView        coordinates,
                                 const Property3DDeviceInputs inputs,
                                 Real* __restrict__ centered,
                                 Real* __restrict__ rdfOutput,
                                 Real* __restrict__ morseOutput) {
  const int lane        = static_cast<int>(threadIdx.x) % kWarpSize;
  const int laneInGroup = lane % kGroupSize;
  const int conformerIdx =
    (blockIdx.x * kWarpsPerBlock + static_cast<int>(threadIdx.x) / kWarpSize) * kGroupsPerWarp + lane / kGroupSize;
  if (conformerIdx >= coordinates.numConformers) {
    return;
  }
  const ConformerAtoms atoms = loadConformer(coordinates, nullptr, inputs.moleculeAtomStarts, conformerIdx);

  Real   rdfAcc[kPairwiseBinsPerLane][kNumPairwiseChannels]   = {};
  Real   morseAcc[kPairwiseBinsPerLane][kNumPairwiseChannels] = {};
  double morseZeroBin[kNumPairwiseChannels]                   = {};
  Real   rdfRadii[kPairwiseBinsPerLane];
  Real   morseScatterings[kPairwiseBinsPerLane];
  for (int slot = 0; slot < kPairwiseBinsPerLane; ++slot) {
    const int bin          = laneInGroup + slot * kGroupSize;
    rdfRadii[slot]         = Real(1) + Real(0.5) * static_cast<Real>(bin);
    morseScatterings[slot] = static_cast<Real>(bin);
  }

  if (atoms.valid) {
    double sumX = 0;
    double sumY = 0;
    double sumZ = 0;
    for (int atomIdx = laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
      sumX += atoms.positions[atomIdx * 3 + 0];
      sumY += atoms.positions[atomIdx * 3 + 1];
      sumZ += atoms.positions[atomIdx * 3 + 2];
    }
    const double inverseAtoms      = 1.0 / static_cast<double>(atoms.numAtoms);
    const double centroidX         = groupAllReduceSum(sumX) * inverseAtoms;
    const double centroidY         = groupAllReduceSum(sumY) * inverseAtoms;
    const double centroidZ         = groupAllReduceSum(sumZ) * inverseAtoms;
    Real* const  conformerCentered = centered + (atoms.positions - coordinates.positions);
    for (int atomIdx = laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
      centeredPosition(atoms.positions,
                       atomIdx,
                       centroidX,
                       centroidY,
                       centroidZ,
                       conformerCentered[atomIdx * 3 + 0],
                       conformerCentered[atomIdx * 3 + 1],
                       conformerCentered[atomIdx * 3 + 2]);
    }
    // Makes the group's centered coordinates visible to all of its lanes.
    __syncwarp(0xffu << (lane - laneInGroup));

    const int     moleculeStart   = inputs.moleculeAtomStarts[coordinates.molIndices[conformerIdx]];
    const int     totalAtoms      = inputs.moleculeAtomStarts[coordinates.nMols];
    const double* propertyWeights = inputs.atomPropertyWeights + moleculeStart;
    // The I-state block is the last atom-property channel.
    const double* iState =
      inputs.atomPropertyWeights + static_cast<size_t>(kNumAtomPropertyChannels - 1) * totalAtoms + moleculeStart;
    const double* iStateDrag = kRdf ? inputs.iStateDragWeights + moleculeStart : nullptr;

    for (int j = 0; j < atoms.numAtoms - 1; ++j) {
      const PairwiseAtom<Real> first = loadPairwiseAtom(conformerCentered, propertyWeights, iState, totalAtoms, j);
      const Real               dragJ = kRdf ? static_cast<Real>(iStateDrag[j]) : Real(0);
      for (int k = j + 1; k < atoms.numAtoms; ++k) {
        const PairwiseAtom<Real> second = loadPairwiseAtom(conformerCentered, propertyWeights, iState, totalAtoms, k);
        const Real               dx     = first.x - second.x;
        const Real               dy     = first.y - second.y;
        const Real               dz     = first.z - second.z;
        const Real               r      = sqrt(dx * dx + dy * dy + dz * dz);
        Real                     products[kNumPairwiseChannels];
        for (int channel = 0; channel < kNumPairwiseChannels; ++channel) {
          products[channel] = first.weights[channel] * second.weights[channel];
        }
        if constexpr (kMorse) {
          for (int slot = 0; slot < kPairwiseBinsPerLane; ++slot) {
            // Bin 0 (lane 0, slot 0) comes from computeMorseZeroBin().
            const Real scattering = morseScatterings[slot] * r;
            accumulatePairTerm(morseAcc[slot], products, sin(scattering) / scattering);
          }
        }
        if constexpr (kRdf) {
          products[kNumPairwiseChannels - 1] = dragJ * static_cast<Real>(iStateDrag[k]);
          for (int slot = 0; slot < kPairwiseBinsPerLane; ++slot) {
            const Real offset = rdfRadii[slot] - r;
            accumulatePairTerm(rdfAcc[slot], products, exp(Real(-100) * offset * offset));
          }
        }
      }
    }
    if constexpr (kMorse) {
      computeMorseZeroBin(atoms, propertyWeights, totalAtoms, laneInGroup, morseZeroBin);
    }
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
}

//! Launches the pairwise kernel when RDF or MORSE is requested; one pass over atom pairs serves both.
template <typename Real>
void launchPairwiseProperties(const DeviceCoordView&        coordinates,
                              const Property3DDeviceInputs& inputs,
                              Real*                         rdfOutput,
                              Real*                         morseOutput,
                              const cudaStream_t            stream) {
  const int numConformers = coordinates.numConformers;
  if (numConformers == 0 || (rdfOutput == nullptr && morseOutput == nullptr)) {
    return;
  }
  const int               numBlocks = (numConformers + kConformersPerBlock - 1) / kConformersPerBlock;
  AsyncDeviceVector<Real> centered(static_cast<size_t>(coordinates.numAtoms) * 3, stream);
  if (rdfOutput != nullptr && morseOutput != nullptr) {
    pairwise3DKernel<Real, true, true>
      <<<numBlocks, kBlockSize, 0, stream>>>(coordinates, inputs, centered.data(), rdfOutput, morseOutput);
  } else if (rdfOutput != nullptr) {
    pairwise3DKernel<Real, true, false>
      <<<numBlocks, kBlockSize, 0, stream>>>(coordinates, inputs, centered.data(), rdfOutput, nullptr);
  } else {
    pairwise3DKernel<Real, false, true>
      <<<numBlocks, kBlockSize, 0, stream>>>(coordinates, inputs, centered.data(), nullptr, morseOutput);
  }
  cudaCheckError(cudaGetLastError());
}

}  // namespace nvMolKit::descriptors3d_detail

#endif  // NVMOLKIT_DESCRIPTORS3D_PAIRWISE_CUH
