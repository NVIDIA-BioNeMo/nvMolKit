// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_DESCRIPTORS3D_PROJECTION_CUH
#define NVMOLKIT_DESCRIPTORS3D_PROJECTION_CUH

#include <algorithm>
#include <cmath>

#include "src/descriptors3d.h"
#include "src/descriptors3d_kernel.cuh"
#include "src/utils/cuda_error_check.h"
#include "src/utils/symmetric_eigenvalues_3x3.cuh"

namespace nvMolKit::descriptors3d_detail {

//! Coordinate PCA of one conformer. Every lane of a group holds an identical copy.
template <typename Real> struct ProjectionState {
  Real centroidX;
  Real centroidY;
  Real centroidZ;
  //! Covariance eigenvalue magnitudes in descending order.
  Real eigenvalues[3];
  //! Row-major; column i is the unit axis of eigenvalues[i].
  Real eigenvectors[9];
};

//! Group-collective. Unweighted centroid, shared by every PCA channel.
template <typename Real>
__device__ __forceinline__ void computeProjectionCentroid(const ConformerAtoms&  atoms,
                                                          const int              laneInGroup,
                                                          ProjectionState<Real>& state) {
  Real sumX = 0;
  Real sumY = 0;
  Real sumZ = 0;
  for (int atomIdx = laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
    sumX += static_cast<Real>(atoms.positions[atomIdx * 3 + 0]);
    sumY += static_cast<Real>(atoms.positions[atomIdx * 3 + 1]);
    sumZ += static_cast<Real>(atoms.positions[atomIdx * 3 + 2]);
  }
  const Real inverseAtoms = Real(1) / static_cast<Real>(atoms.numAtoms > 0 ? atoms.numAtoms : 1);
  state.centroidX         = groupAllReduceSum(sumX) * inverseAtoms;
  state.centroidY         = groupAllReduceSum(sumY) * inverseAtoms;
  state.centroidZ         = groupAllReduceSum(sumZ) * inverseAtoms;
}

template <typename Real>
__device__ __forceinline__ void centeredCoordinates(const ConformerAtoms&        atoms,
                                                    const int                    atomIdx,
                                                    const ProjectionState<Real>& state,
                                                    Real&                        x,
                                                    Real&                        y,
                                                    Real&                        z) {
  x = static_cast<Real>(atoms.positions[atomIdx * 3 + 0]) - state.centroidX;
  y = static_cast<Real>(atoms.positions[atomIdx * 3 + 1]) - state.centroidY;
  z = static_cast<Real>(atoms.positions[atomIdx * 3 + 2]) - state.centroidZ;
}

template <typename Real>
__device__ __forceinline__ void swapEigenpairs(ProjectionState<Real>& state, const int first, const int second) {
  const Real value          = state.eigenvalues[first];
  state.eigenvalues[first]  = state.eigenvalues[second];
  state.eigenvalues[second] = value;
  for (int row = 0; row < 3; ++row) {
    const Real component                 = state.eigenvectors[row * 3 + first];
    state.eigenvectors[row * 3 + first]  = state.eigenvectors[row * 3 + second];
    state.eigenvectors[row * 3 + second] = component;
  }
}

/**
 * @brief Group-collective. Covariance of the centered coordinates about ProjectionState's centroid and
 *        its eigensystem.
 *
 * @p weights (one per atom) scales each atom's contribution and normalizes by their sum, falling back to
 * a plain sum when that total is near zero. A null @p weights gives the unweighted covariance normalized
 * by the atom count. Every lane diagonalizes its own identical copy of the reduced covariance, so the
 * eigensystem is available group-wide without shared memory or synchronization.
 */
template <typename Real>
__device__ __forceinline__ void computeProjectionPca(const ConformerAtoms&  atoms,
                                                     const double*          weights,
                                                     const int              laneInGroup,
                                                     ProjectionState<Real>& state) {
  Real totalWeight = 0;
  Real xx          = 0;
  Real xy          = 0;
  Real xz          = 0;
  Real yy          = 0;
  Real yz          = 0;
  Real zz          = 0;
  for (int atomIdx = laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
    const Real weight = weights == nullptr ? Real(1) : static_cast<Real>(weights[atomIdx]);
    Real       x;
    Real       y;
    Real       z;
    centeredCoordinates(atoms, atomIdx, state, x, y, z);
    totalWeight += weight;
    xx += weight * x * x;
    xy += weight * x * y;
    xz += weight * x * z;
    yy += weight * y * y;
    yz += weight * y * z;
    zz += weight * z * z;
  }
  totalWeight = groupAllReduceSum(totalWeight);
  Real inverseWeight;
  if (weights == nullptr) {
    inverseWeight = Real(1) / static_cast<Real>(atoms.numAtoms > 0 ? atoms.numAtoms : 1);
  } else {
    inverseWeight = fabs(totalWeight) < Real(1e-4) ? Real(1) : Real(1) / totalWeight;
  }
  symmetricEigensystemJacobi3x3(groupAllReduceSum(xx) * inverseWeight,
                                groupAllReduceSum(xy) * inverseWeight,
                                groupAllReduceSum(xz) * inverseWeight,
                                groupAllReduceSum(yy) * inverseWeight,
                                groupAllReduceSum(yz) * inverseWeight,
                                groupAllReduceSum(zz) * inverseWeight,
                                state.eigenvalues,
                                state.eigenvectors);
  for (int axis = 0; axis < 3; ++axis) {
    state.eigenvalues[axis] = fabs(state.eigenvalues[axis]);
  }
  if (state.eigenvalues[1] > state.eigenvalues[0]) {
    swapEigenpairs(state, 0, 1);
  }
  if (state.eigenvalues[2] > state.eigenvalues[0]) {
    swapEigenpairs(state, 0, 2);
  }
  if (state.eigenvalues[2] > state.eigenvalues[1]) {
    swapEigenpairs(state, 1, 2);
  }
}

//! Centered coordinate of atom @p atomIdx along principal axis @p axis.
template <typename Real>
__device__ __forceinline__ Real
projectionScore(const ConformerAtoms& atoms, const int atomIdx, const ProjectionState<Real>& state, const int axis) {
  Real x;
  Real y;
  Real z;
  centeredCoordinates(atoms, atomIdx, state, x, y, z);
  return x * state.eigenvectors[axis] + y * state.eigenvectors[3 + axis] + z * state.eigenvectors[6 + axis];
}

//! Group-collective. Mean absolute distance of the atoms from the plane of best fit, i.e. their mean
//! absolute score along the smallest principal axis of the unweighted PCA.
template <typename Real>
__device__ __forceinline__ Real computePbf(const ConformerAtoms&        atoms,
                                           const int                    laneInGroup,
                                           const ProjectionState<Real>& state) {
  Real distanceSum = 0;
  for (int atomIdx = laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
    distanceSum += fabs(projectionScore(atoms, atomIdx, state, 2));
  }
  return groupAllReduceSum(distanceSum) / static_cast<Real>(atoms.numAtoms > 0 ? atoms.numAtoms : 1);
}

//! RDKit rounds every WHIM value and intermediate projection to three decimals.
template <typename Real> __device__ __forceinline__ Real roundWhim(const Real value) {
  return round(value * Real(1000)) / Real(1000);
}

/**
 * @brief RDKit's symmetry comparisons on projections rounded to thousandths.
 *
 * RDKit compares `fabs(a + b) <= threshold` and `fabs(a) < threshold` on float64 values `k / 1000.0`.
 * On integer thousandths the outcome is decided exactly unless the value lies on the threshold.
 *
 * FP64 required: the threshold and the on-threshold evaluation. Rounded projections live on the same
 * 0.001 grid as the default threshold, so many pair sums land exactly on it, and RDKit's result there
 * depends on how `k / 1000.0` is represented in float64. Evaluating those cases in float32 flips them.
 */
struct WhimThreshold {
  double value;        //!< Threshold in the projection's units.
  double thousandths;  //!< value * 1000.

  __device__ __forceinline__ bool nearBoundary(const double magnitude, const double scale) const {
    return fabs(magnitude - thousandths) <= 1e-9 * (scale + 1.0);
  }

  //! `fabs(a / 1000.0 + b / 1000.0) <= value` for rounded projections @p a and @p b.
  __device__ __forceinline__ bool mirrored(const int32_t a, const int32_t b) const {
    const double sum = fabs(static_cast<double>(a) + static_cast<double>(b));
    if (!nearBoundary(sum, fabs(static_cast<double>(a)) + fabs(static_cast<double>(b)))) {
      return sum <= thousandths;
    }
    return fabs(static_cast<double>(a) / 1000.0 + static_cast<double>(b) / 1000.0) <= value;
  }

  //! `fabs(a / 1000.0) < value` for rounded projection @p a.
  __device__ __forceinline__ bool centered(const int32_t a) const {
    const double magnitude = fabs(static_cast<double>(a));
    if (!nearBoundary(magnitude, magnitude)) {
      return magnitude < thousandths;
    }
    return fabs(static_cast<double>(a) / 1000.0) < value;
  }
};

/**
 * @brief Group-collective. WHIM symmetry term along principal axis @p axis: atoms whose rounded
 *        projection is centered, or mirrored by another atom within the threshold, count as symmetric.
 *
 * The pairwise search compares every atom with every other, so each projection is rounded once to
 * integer thousandths in @p scores (this conformer's slots, at least `atoms.numAtoms` long; @p groupMask
 * selects the group's lanes).
 */
template <typename Real>
__device__ __forceinline__ Real computeWhimGamma(const ConformerAtoms&        atoms,
                                                 const int                    laneInGroup,
                                                 const unsigned               groupMask,
                                                 const ProjectionState<Real>& state,
                                                 const int                    axis,
                                                 const WhimThreshold&         threshold,
                                                 int32_t*                     scores) {
  for (int atomIdx = laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
    scores[atomIdx] = static_cast<int32_t>(round(projectionScore(atoms, atomIdx, state, axis) * Real(1000)));
  }
  __syncwarp(groupMask);

  Real symmetricCount  = 0;
  Real asymmetricCount = 0;
  for (int atomIdx = laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
    const int32_t score       = scores[atomIdx];
    bool          hasOpposite = false;
    for (int otherIdx = 0; otherIdx < atoms.numAtoms; ++otherIdx) {
      if (otherIdx != atomIdx && threshold.mirrored(score, scores[otherIdx])) {
        hasOpposite = true;
        break;
      }
    }
    if (hasOpposite || threshold.centered(score)) {
      symmetricCount += Real(1);
    } else {
      asymmetricCount += Real(1);
    }
  }
  __syncwarp(groupMask);  // Every lane is done reading before the next axis overwrites the scores.
  symmetricCount      = groupAllReduceSum(symmetricCount);
  asymmetricCount     = groupAllReduceSum(asymmetricCount);
  const Real numAtoms = static_cast<Real>(atoms.numAtoms);
  Real       inverseGamma;
  if (symmetricCount == Real(0)) {
    inverseGamma = Real(1) - (asymmetricCount / numAtoms) * log(Real(1) / numAtoms) / log(Real(2));
  } else {
    inverseGamma = Real(1) - ((symmetricCount / numAtoms) * log(symmetricCount / numAtoms) / log(Real(2)) +
                              (asymmetricCount / numAtoms) * log(Real(1) / numAtoms) / log(Real(2)));
  }
  return Real(1) / inverseGamma;
}

/**
 * @brief Group-collective. Writes WHIM channel @p channel of one conformer's 114-value row.
 *
 * Layout matches RDKit's CalcWHIM: 11 directional values per channel (eigenvalues, two eigenvalue ratios,
 * three symmetry terms, three inverse kurtoses), then 7 totals, 7 pairwise products, 2 symmetry means,
 * 7 anisotropies, 7 kurtosis means and 7 combined terms. Only @p writeRow lanes store. @p groupMask and
 * @p scores are forwarded to computeWhimGamma().
 */
template <typename Real, typename OutputReal>
__device__ __forceinline__ void writeWhimChannel(const ConformerAtoms&        atoms,
                                                 const int                    laneInGroup,
                                                 const unsigned               groupMask,
                                                 const int                    channel,
                                                 const WhimThreshold&         threshold,
                                                 const ProjectionState<Real>& state,
                                                 const bool                   writeRow,
                                                 OutputReal*                  row,
                                                 int32_t*                     scores) {
  const Real first  = state.eigenvalues[0];
  const Real second = state.eigenvalues[1];
  const Real third  = state.eigenvalues[2];
  const Real total  = first + second + third;

  Real fourthMomentFirst  = 0;
  Real fourthMomentSecond = 0;
  Real fourthMomentThird  = 0;
  for (int atomIdx = laneInGroup; atomIdx < atoms.numAtoms; atomIdx += kGroupSize) {
    const Real scoreFirst  = projectionScore(atoms, atomIdx, state, 0);
    const Real scoreSecond = projectionScore(atoms, atomIdx, state, 1);
    const Real scoreThird  = projectionScore(atoms, atomIdx, state, 2);
    fourthMomentFirst += scoreFirst * scoreFirst * scoreFirst * scoreFirst;
    fourthMomentSecond += scoreSecond * scoreSecond * scoreSecond * scoreSecond;
    fourthMomentThird += scoreThird * scoreThird * scoreThird * scoreThird;
  }
  fourthMomentFirst   = groupAllReduceSum(fourthMomentFirst);
  fourthMomentSecond  = groupAllReduceSum(fourthMomentSecond);
  fourthMomentThird   = groupAllReduceSum(fourthMomentThird);
  const Real numAtoms = static_cast<Real>(atoms.numAtoms);
  const Real e1       = fourthMomentFirst > Real(0) ? numAtoms * first * first / fourthMomentFirst : Real(0);
  const Real e2       = fourthMomentSecond > Real(0) ? numAtoms * second * second / fourthMomentSecond : Real(0);
  const Real e3       = fourthMomentThird > Real(0) ? numAtoms * third * third / fourthMomentThird : Real(0);

  const int channelStart = channel * 11;
  if (writeRow) {
    row[channelStart + 0]  = roundWhim(first);
    row[channelStart + 1]  = roundWhim(second);
    row[channelStart + 2]  = roundWhim(third);
    row[channelStart + 3]  = roundWhim(first / total);
    row[channelStart + 4]  = roundWhim(second / total);
    row[channelStart + 8]  = roundWhim(e1);
    row[channelStart + 9]  = roundWhim(e2);
    row[channelStart + 10] = roundWhim(e3);
    row[77 + channel]      = roundWhim(total);
    row[84 + channel]      = roundWhim(first * second + first * third + second * third);
    const Real anisotropy =
      Real(0.75) * (fabs(first / total - Real(1) / Real(3)) + fabs(second / total - Real(1) / Real(3)) +
                    fabs(third / total - Real(1) / Real(3)));
    row[93 + channel]  = roundWhim(anisotropy);
    row[100 + channel] = roundWhim((e1 + e2 + e3) / Real(3));
    row[107 + channel] = roundWhim(total + first * second + first * third + second * third + first * second * third);
  }

  Real gammaProduct = Real(1);
  for (int axis = 0; axis < 3; ++axis) {
    const Real gamma = computeWhimGamma(atoms, laneInGroup, groupMask, state, axis, threshold, scores);
    gammaProduct *= gamma;
    if (writeRow) {
      row[channelStart + 5 + axis] = roundWhim(gamma);
    }
  }
  if (writeRow && channel < 2) {
    row[91 + channel] = roundWhim(pow(gammaProduct, Real(1) / Real(3)));
  }
}

//! Number of WHIM atom-property channels after the unweighted one (mass, van der Waals volume,
//! electronegativity, polarizability, ionization potential, I-state).
constexpr int kNumWhimWeightChannels = 6;

/**
 * @brief Arithmetic type of WHIM's PCA and projections, in every precision mode.
 *
 * FP64 required: WHIM's inverse kurtosis, `n * lambda^2 / sum(score^4)` per axis, is scale-invariant.
 * On the near-zero axis of a planar or near-planar conformer it depends only on out-of-plane deviations
 * far below float32 resolution relative to the molecule's extent, so a float32 covariance, eigensystem or
 * projection turns it (and the kurtosis means built from it) into rounding noise. Every other WHIM term is
 * accurate in float32.
 */
using WhimReal = double;

/**
 * @brief PBF and WHIM for conformers [@p conformerBegin, @p conformerEnd); one group of lanes per conformer.
 *
 * PBF uses an unweighted PCA in @p OutputReal. WHIM runs its own PCA in WhimReal for the unweighted channel
 * and each atom-property channel, whose weights are stored channel-major in `inputs.whimWeights`
 * (`kNumWhimWeightChannels` blocks of one value per molecule atom), and uses @p whimScores as
 * symmetry-search scratch: @p whimScoreStride slots per conformer of the range. Rows longer than
 * @p whimScoreStride are treated as invalid. Outputs are @p OutputReal.
 */
template <typename OutputReal, bool kComputeWhim>
__global__ void projection3DKernel(const DeviceCoordView        coordinates,
                                   const Property3DDeviceInputs inputs,
                                   const int                    conformerBegin,
                                   const int                    conformerEnd,
                                   OutputReal* __restrict__ pbfOutput,
                                   OutputReal* __restrict__ whimOutput,
                                   int32_t* __restrict__ whimScores,
                                   const int    whimScoreStride,
                                   const double whimThreshold) {
  const int lane        = static_cast<int>(threadIdx.x) % kWarpSize;
  const int laneInGroup = lane % kGroupSize;
  const int warpStart =
    conformerBegin + (blockIdx.x * kWarpsPerBlock + static_cast<int>(threadIdx.x) / kWarpSize) * kGroupsPerWarp;
  const int conformerIdx = warpStart + lane / kGroupSize;
  if (warpStart >= conformerEnd) {
    return;  // Uniform across the warp; partially filled warps keep every lane for the shuffles.
  }
  const bool inRange  = conformerIdx < conformerEnd;
  const bool writeRow = inRange && laneInGroup == 0;

  ConformerAtoms atoms =
    loadConformer(coordinates, nullptr, inputs.moleculeAtomStarts, inRange ? conformerIdx : coordinates.numConformers);
  if constexpr (kComputeWhim) {
    if (atoms.numAtoms > whimScoreStride) {
      atoms = ConformerAtoms{};
    }
  }

  if (pbfOutput != nullptr) {
    ProjectionState<OutputReal> pbfState;
    computeProjectionCentroid(atoms, laneInGroup, pbfState);
    computeProjectionPca(atoms, nullptr, laneInGroup, pbfState);
    const OutputReal pbf = computePbf(atoms, laneInGroup, pbfState);
    if (writeRow) {
      const bool is3D         = inputs.conformerIs3D == nullptr || inputs.conformerIs3D[conformerIdx] != 0;
      pbfOutput[conformerIdx] = !atoms.valid                ? static_cast<OutputReal>(nan("")) :
                                atoms.numAtoms < 4 || !is3D ? OutputReal(0) :
                                                              pbf;
    }
  }

  if constexpr (kComputeWhim) {
    const unsigned      groupMask = 0xffu << (lane - laneInGroup);
    const WhimThreshold threshold{whimThreshold, whimThreshold * 1000.0};
    int32_t*            scores =
      inRange ? whimScores + static_cast<size_t>(conformerIdx - conformerBegin) * whimScoreStride : nullptr;
    OutputReal* row = inRange ? whimOutput + static_cast<size_t>(conformerIdx) * kNumWhimProperties : nullptr;
    ProjectionState<WhimReal> state;
    computeProjectionCentroid(atoms, laneInGroup, state);
    computeProjectionPca(atoms, nullptr, laneInGroup, state);
    writeWhimChannel(atoms, laneInGroup, groupMask, 0, threshold, state, writeRow && atoms.valid, row, scores);
    const int moleculeIdx       = atoms.valid ? coordinates.molIndices[conformerIdx] : 0;
    const int moleculeAtomStart = atoms.valid ? inputs.moleculeAtomStarts[moleculeIdx] : 0;
    const int totalAtoms        = inputs.moleculeAtomStarts[coordinates.nMols];
    for (int channel = 1; channel <= kNumWhimWeightChannels; ++channel) {
      const double* weights = inputs.whimWeights + static_cast<size_t>(channel - 1) * totalAtoms + moleculeAtomStart;
      computeProjectionPca(atoms, weights, laneInGroup, state);
      writeWhimChannel(atoms, laneInGroup, groupMask, channel, threshold, state, writeRow && atoms.valid, row, scores);
    }
    if (inRange && !atoms.valid) {
      for (int valueIdx = laneInGroup; valueIdx < kNumWhimProperties; valueIdx += kGroupSize) {
        row[valueIdx] = static_cast<OutputReal>(nan(""));
      }
    }
  }
}

//! Upper bound on the WHIM symmetry-search scratch; larger batches run in conformer chunks.
constexpr size_t kWhimScratchBudgetBytes = size_t{256} << 20;

/**
 * @brief Launches the projection kernel when PBF or WHIM is requested. PBF computes in @p Real; WHIM's
 *        PCA computes in WhimReal. Both write @p Real outputs.
 */
template <typename Real>
void launchProjectionProperties(const DeviceCoordView&        coordinates,
                                const Property3DDeviceInputs& inputs,
                                const WhimOptions&            whim,
                                Real*                         pbfOutput,
                                Real*                         whimOutput,
                                const cudaStream_t            stream) {
  const int numConformers = coordinates.numConformers;
  if (numConformers == 0 || (pbfOutput == nullptr && whimOutput == nullptr)) {
    return;
  }
  if (whimOutput == nullptr) {
    const int numBlocks = (numConformers + kConformersPerBlock - 1) / kConformersPerBlock;
    projection3DKernel<Real, false><<<numBlocks, kBlockSize, 0, stream>>>(coordinates,
                                                                          inputs,
                                                                          0,
                                                                          numConformers,
                                                                          pbfOutput,
                                                                          nullptr,
                                                                          nullptr,
                                                                          0,
                                                                          whim.threshold);
    cudaCheckError(cudaGetLastError());
    return;
  }

  const int                  scoreStride = std::max(inputs.maxMoleculeAtoms, 1);
  const size_t               chunkSize   = std::clamp<size_t>(kWhimScratchBudgetBytes / (sizeof(int32_t) * scoreStride),
                                              1,
                                              static_cast<size_t>(numConformers));
  AsyncDeviceVector<int32_t> scores(chunkSize * scoreStride, stream);
  for (int begin = 0; begin < numConformers; begin += static_cast<int>(chunkSize)) {
    const int end       = std::min(numConformers, begin + static_cast<int>(chunkSize));
    const int numBlocks = (end - begin + kConformersPerBlock - 1) / kConformersPerBlock;
    projection3DKernel<Real, true><<<numBlocks, kBlockSize, 0, stream>>>(coordinates,
                                                                         inputs,
                                                                         begin,
                                                                         end,
                                                                         pbfOutput,
                                                                         whimOutput,
                                                                         scores.data(),
                                                                         scoreStride,
                                                                         whim.threshold);
    cudaCheckError(cudaGetLastError());
  }
}

}  // namespace nvMolKit::descriptors3d_detail

#endif  // NVMOLKIT_DESCRIPTORS3D_PROJECTION_CUH
