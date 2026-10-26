// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_DESCRIPTORS3D_GETAWAY_CUH
#define NVMOLKIT_DESCRIPTORS3D_GETAWAY_CUH

#include <cmath>
#include <cstdint>

#include "src/descriptors3d.h"
#include "src/descriptors3d_kernel.cuh"
#include "src/descriptors3d_projection.cuh"
#include "src/descriptors3d_topology.cuh"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"
#include "src/utils/symmetric_eigenvalues_3x3.cuh"

namespace nvMolKit::descriptors3d_detail {

//! Unweighted channel plus the atom-property channels, in RDKit's output order (u, m, v, e, p, i, s).
constexpr int kNumGetawayChannels = kNumAtomPropertyChannels + 1;
//! Topological lags with their own H, HATS and R values (1-8); farther or disconnected pairs only enter the
//! totals. One lane of each kGroupSize-lane group owns each lag.
constexpr int kNumGetawayLags     = 8;
static_assert(kNumGetawayLags == kGroupSize);
//! Pair streams per warp: each kGroupSize-lane group walks interleaved rows of the atom-pair triangle.
constexpr int kGetawayPairStreams = kWarpSize / kGroupSize;

//! Values per channel in the H/HATS block: H0..H8, HT, HATS0..HATS8, HATST.
constexpr int kGetawayHBlock    = 2 * (kNumGetawayLags + 2);
//! Values per channel in the R block: R1..R8, RT, R1+..R8+, RT+.
constexpr int kGetawayRBlock    = 2 * (kNumGetawayLags + 1);
constexpr int kGetawayHStart    = 4;
constexpr int kGetawayRconIndex = kGetawayHStart + kNumGetawayChannels * kGetawayHBlock;
constexpr int kGetawayRStart    = kGetawayRconIndex + 3;
static_assert(kGetawayRStart + kNumGetawayChannels * kGetawayRBlock == kNumGetawayProperties);

//! Singular values of X^T X at or below this are dropped from its pseudo-inverse, as in RDKit's GetPinv.
constexpr double kGetawayPinvTolerance = 1e-3;
//! PBF below this makes RDKit treat the molecule as two-dimensional for HIC.
constexpr double kGetawayPlanarPbf     = 1e-5;
//! Upper bound on power-iteration steps for REIG.
constexpr int    kGetawayMaxIterations = 1000;

/**
 * @brief Per-atom scratch row, written once per conformer as the compute type: scaled projections y (three
 *        values, H(j, k) = y_j . y_k), leverage h = H(j, j), sqrt(h), the atom-property channels, then the
 *        centered coordinates.
 */
constexpr int kGetawayRowY        = 0;
constexpr int kGetawayRowLeverage = 3;
constexpr int kGetawayRowRootLev  = 4;
constexpr int kGetawayRowWeights  = 5;
constexpr int kGetawayRowCentered = kGetawayRowWeights + kNumAtomPropertyChannels;
constexpr int kGetawayRowStride   = kGetawayRowCentered + 3;

// ITH and ISH group the heavy-atom leverages by their decimal digits, not their numeric values: RDKit rounds
// each leverage to `precision` significant digits (round_to_n_digits_), prints it with C++ stream output, and
// its IsClose2 compares substrings of that printed text. Reproducing RDKit's groups exactly therefore means
// reproducing that rounding and that text, which the helpers below do for the values leverages can take.

//! Exact powers of ten representable in float64.
__device__ __forceinline__ double exactPowerOfTen(const int exponent) {
  // Built from its binary digits rather than a table (a local array would live in local memory). Every partial
  // product is itself a power of ten up to 1e22 = 2^22 * 5^22 with 5^22 < 2^53, so each product is exact.
  double power = 1.0;
  if (exponent & 1) {
    power *= 1e1;
  }
  if (exponent & 2) {
    power *= 1e2;
  }
  if (exponent & 4) {
    power *= 1e4;
  }
  if (exponent & 8) {
    power *= 1e8;
  }
  if (exponent & 16) {
    power *= 1e16;
  }
  return power;
}

//! @p value * 10^@p exponent; exact scaling factors within float64's exact powers of ten.
__device__ __forceinline__ double scaleByPowerOfTen(double value, int exponent) {
  while (exponent > 22) {
    value *= 1e22;
    exponent -= 22;
  }
  while (exponent < -22) {
    value /= 1e22;
    exponent += 22;
  }
  return exponent >= 0 ? value * exactPowerOfTen(exponent) : value / exactPowerOfTen(-exponent);
}

/**
 * @brief RDKit's round_to_n_digits_: `atof(sprintf("%.*g", digits, value))` for a non-negative @p value.
 *
 * Also returns the rounded value's significand (@p mantissa, exactly @p digits digits unless the value is 0)
 * and decimal exponent, from which clusterKey() derives its printed text. Exact ties round to even, as glibc
 * does for exactly representable halves; values within float64 rounding of a tie may differ from printf.
 */
__device__ __forceinline__ double roundToSignificantDigits(const double value,
                                                           const int    digits,
                                                           int64_t&     mantissa,
                                                           int&         exponent) {
  mantissa = 0;
  exponent = 0;
  if (!(value > 0.0) || !isfinite(value)) {
    return value;  // 0 stays 0; NaN and infinity stay as they are.
  }
  exponent             = static_cast<int>(floor(log10(value)));
  const double lowest  = exactPowerOfTen(digits - 1);
  const double highest = exactPowerOfTen(digits);
  double       scaled  = scaleByPowerOfTen(value, digits - 1 - exponent);
  if (scaled < lowest) {
    --exponent;
    scaled = scaleByPowerOfTen(value, digits - 1 - exponent);
  } else if (scaled >= highest) {
    ++exponent;
    scaled = scaleByPowerOfTen(value, digits - 1 - exponent);
  }
  mantissa = llrint(scaled);
  if (mantissa == static_cast<int64_t>(highest)) {
    mantissa /= 10;
    ++exponent;
  }
  return scaleByPowerOfTen(static_cast<double>(mantissa), exponent - (digits - 1));
}

//! What RDKit's IsClose2 reads from a rounded leverage's printed fraction text (the text after the first '.', or
//! all of it when there is none): its leading '0' count, its length, its leading integer (C atoi) and whether it
//! is all digits, in which case appending '0' multiplies that integer by 10.
struct ClusterKey {
  int     zeros;
  int     length;
  int64_t leading;
  bool    allDigits;
};

//! Packs a ClusterKey into 64 bits: leading integer (below 2^40) in the low bits, then length, zeros and the
//! all-digits flag.
__device__ __forceinline__ int64_t packClusterKey(const ClusterKey& key) {
  return key.leading | (int64_t{key.length} << 40) | (int64_t{key.zeros} << 48) |
         (int64_t{key.allDigits ? 1 : 0} << 56);
}

__device__ __forceinline__ ClusterKey unpackClusterKey(const int64_t packed) {
  return {static_cast<int>((packed >> 48) & 0xff),
          static_cast<int>((packed >> 40) & 0xff),
          packed & ((int64_t{1} << 40) - 1),
          ((packed >> 56) & 1) != 0};
}

//! Rounds @p leverage to @p digits significant digits (into @p rounded) and returns the ClusterKey of its
//! printed text.
__device__ __forceinline__ ClusterKey clusterKey(const double leverage, const int digits, double& rounded) {
  int64_t mantissa;
  int     exponent;
  rounded = roundToSignificantDigits(leverage, digits, mantissa, exponent);
  // IsClose2 prints with default C++ stream formatting: up to 6 significant digits with trailing zeros removed,
  // fixed notation for decimal exponents -4 to 5 (e.g. "0.034", "1") and scientific notation otherwise (e.g.
  // "1.8e-06"); precision is at most 6, so every significant digit is printed. The key of that text's fraction is
  // derived from the digits directly rather than by building the
  // text (character arrays would live in local memory): the significand without trailing zeros, its length, and
  // where the printed '.' falls.
  if (mantissa == 0) {
    return {1, 1, 0, true};  // "0": no point, so the whole text.
  }
  // Significands of up to 9 digits (every precision below 10) fit 32 bits, whose division is far cheaper than
  // the 64-bit one; the quotients and remainders are the same.
  int64_t    significand = mantissa;
  int        length      = digits;
  const bool narrow      = significand <= int64_t{0xffffffff};
  if (narrow) {
    uint32_t value = static_cast<uint32_t>(significand);
    while (length > 1 && value % 10u == 0u) {
      value /= 10u;
      --length;
    }
    significand = value;
  } else {
    while (length > 1 && significand % 10 == 0) {
      significand /= 10;
      --length;
    }
  }
  // Fraction digits: the trailing @p count digits of the significand, whose last digit is not zero.
  const auto trailing = [&](const int count, int64_t& value) {
    int64_t scale = 1;
    for (int position = 0; position < count; ++position) {
      scale *= 10;
    }
    if (narrow) {
      value = scale <= int64_t{0xffffffff} ?
                static_cast<int64_t>(static_cast<uint32_t>(significand) % static_cast<uint32_t>(scale)) :
                significand;  // Below 2^32 <= scale.
    } else {
      value = significand % scale;
    }
    int valueDigits = 1;
    for (int64_t bound = 10; bound <= value; bound *= 10) {
      ++valueDigits;
    }
    return count - valueDigits;  // Leading '0' characters.
  };
  if (exponent < -4 || exponent >= 6) {
    // "d[.ddd]e±XX[X]": the fraction text keeps the exponent suffix, so it is not all digits.
    const int exponentDigits = (exponent <= -100 || exponent >= 100) ? 3 : 2;
    if (length > 1) {
      int64_t   value;
      const int zeros = trailing(length - 1, value);
      return {zeros, length - 1 + 2 + exponentDigits, value, false};
    }
    return {0, 1 + 2 + exponentDigits, significand, false};
  }
  if (exponent >= 0) {
    if (length > exponent + 1) {
      int64_t   value;
      const int zeros = trailing(length - exponent - 1, value);
      return {zeros, length - exponent - 1, value, true};
    }
    int64_t value = significand;
    for (int position = length; position <= exponent; ++position) {
      value *= 10;
    }
    return {0, exponent + 1, value, true};  // Integer text without a point.
  }
  // "0.00ddd": -exponent - 1 zeros, then every significant digit.
  return {-exponent - 1, -exponent - 1 + length, significand, true};
}

//! The effect of appending '0' to the text a ClusterKey describes.
__device__ __forceinline__ void appendZero(ClusterKey& key) {
  ++key.length;
  if (key.allDigits) {
    key.leading *= 10;
  }
}

/**
 * @brief RDKit's IsClose2 on the keys of two printed values: equal leading-zero counts pad the shorter text
 *        once with '0'; texts of equal length are close when their leading integers differ by less than 2.
 */
__device__ __forceinline__ bool rdkitIsClose(ClusterKey a, ClusterKey b) {
  if (a.zeros == b.zeros) {
    if (a.length > b.length) {
      appendZero(b);
    }
    if (a.length < b.length) {
      appendZero(a);
    }
  }
  if (a.length != b.length) {
    return false;
  }
  const int64_t difference = a.leading - b.leading;
  return difference > -2 && difference < 2;
}

/**
 * @brief Warp-collective. ITH and ISH from the heavy-atom leverages, porting RDKit's getGETAWAYDesc: each
 *        leverage rounded to @p digits significant digits, sorted in descending order and grouped by RDKit's
 *        clusterArray2 (a greedy scan with IsClose2 comparisons of the printed values), then
 *        ITH = n log2 n - sum(c log2 c) over the group sizes c and ISH = ITH / (n log2 n).
 *
 * The lanes round, key and rank the heavy atoms (equal values print identically, so their relative order
 * does not matter); the warp then runs the greedy scan on the sorted keys a cluster at a time, and lane 0 alone
 * receives @p ith and @p ish.
 * Both are NaN when any heavy-atom leverage is not finite. @p heavyValues and @p unsortedKeys hold one entry and @p
 * sortedKeys one entry per heavy atom.
 */
__device__ __forceinline__ void computeInformationIndices(const double*  leverages,
                                                          const uint8_t* heavyAtomFlags,
                                                          const int      numAtoms,
                                                          const int      digits,
                                                          const int      lane,
                                                          double*        heavyValues,
                                                          int64_t*       unsortedKeys,
                                                          int64_t*       sortedKeys,
                                                          double&        ith,
                                                          double&        ish) {
  const auto warp      = laneTile<kWarpSize>();
  int        numHeavy  = 0;
  bool       nonFinite = false;
  for (int base = 0; base < numAtoms; base += kWarpSize) {
    const int      atomIdx = base + lane;
    const bool     isHeavy = atomIdx < numAtoms && heavyAtomFlags[atomIdx] != 0;
    const unsigned heavy   = warp.ballot(isHeavy);
    nonFinite              = nonFinite || (isHeavy && !isfinite(leverages[atomIdx]));
    if (isHeavy) {
      const int  slot = numHeavy + __popc(heavy & ((1u << lane) - 1u));
      double     rounded;
      const auto key     = clusterKey(leverages[atomIdx], digits, rounded);
      heavyValues[slot]  = rounded;
      unsortedKeys[slot] = packClusterKey(key);
    }
    numHeavy += __popc(heavy);
  }
  // Non-finite leverages (from non-finite coordinates) have no rank or printed digits to cluster.
  if (warp.any(nonFinite)) {
    ith = nan("");
    ish = nan("");
    return;
  }
  warp.sync();
  for (int slot = lane; slot < numHeavy; slot += kWarpSize) {
    const double value = heavyValues[slot];
    int          rank  = 0;
#pragma unroll 4
    for (int other = 0; other < numHeavy; ++other) {
      const double otherValue = heavyValues[other];
      rank += otherValue > value || (otherValue == value && other < slot) ? 1 : 0;
    }
    sortedKeys[rank] = unsortedKeys[slot];
  }
  warp.sync();
  // clusterArray2's greedy scan, a cluster per step: the lanes test 32 followers of the cluster's front at once
  // and the first one that is not close ends it. Cluster sizes go to heavyValues (free after the ranking).
  int clusters = 0;
  int head     = 0;
  while (head < numHeavy) {
    const ClusterKey front     = unpackClusterKey(sortedKeys[head]);
    const int        remaining = numHeavy - head - 1;
    int              stop      = -1;  // offset of the first follower that is not close to the front
    for (int base = 0; base < remaining && stop < 0; base += kWarpSize) {
      const int      offset = base + lane;
      const bool     far  = offset < remaining && !rdkitIsClose(front, unpackClusterKey(sortedKeys[head + 1 + offset]));
      const unsigned ends = warp.ballot(far);
      stop                = ends != 0u ? base + __ffs(static_cast<int>(ends)) - 1 : stop;
    }
    int size;
    if (remaining == 0) {
      size = 1;
      head = numHeavy;
    } else if (stop >= 0) {
      size = 1 + stop;
      head += 1 + stop;
    } else {
      // Every follower is close: RDKit counts them all, then resumes from the second element of the cluster.
      size = 1 + remaining;
      head += 2;
    }
    if (lane == 0) {
      heavyValues[clusters] = size;
    }
    ++clusters;
  }
  warp.sync();
  // A group of one contributes 1 * log(1) = 0 exactly and is skipped; the others are subtracted in scan order.
  for (int cluster = lane; cluster < clusters; cluster += kWarpSize) {
    const double size    = heavyValues[cluster];
    heavyValues[cluster] = size > 1.0 ? size * log(size) / log(2.0) : 0.0;
  }
  warp.sync();
  if (lane != 0) {
    return;
  }
  const double heavyCount = numHeavy;
  const double total0     = heavyCount * log(heavyCount) / log(2.0);
  double       total      = total0;
  for (int cluster = 0; cluster < clusters; ++cluster) {
    const double term = heavyValues[cluster];
    if (term != 0.0) {
      total -= term;
    }
  }
  ith = total;
  ish = total / total0;
}

/**
 * @brief Warp-collective. Leverages of the conformer's atoms: the diagonal of H = X (X^T X)^+ X^T for the
 *        centered coordinates X, with RDKit's pseudo-inverse (singular values at or below
 *        kGetawayPinvTolerance dropped). Writes each atom's scratch row and float64 leverage.
 *
 * FP64 required: leverages decide ITH and ISH through their rounding to `precision` significant digits and
 * RDKit's digit-string clustering, so float32 leverages would move values across rounding boundaries; the
 * pseudo-inverse cutoff decides which axes count for (near-)planar conformers. This is O(atoms) work.
 */
template <typename Real>
__device__ __forceinline__ void computeLeverages(const ConformerAtoms& atoms,
                                                 const double*         propertyWeights,
                                                 const int             totalAtoms,
                                                 const int             lane,
                                                 Real*                 rows,
                                                 double*               leverages) {
  double sumX = 0;
  double sumY = 0;
  double sumZ = 0;
  for (int atomIdx = lane; atomIdx < atoms.numAtoms; atomIdx += kWarpSize) {
    sumX += atoms.positions[atomIdx * 3 + 0];
    sumY += atoms.positions[atomIdx * 3 + 1];
    sumZ += atoms.positions[atomIdx * 3 + 2];
  }
  const double inverseAtoms = 1.0 / static_cast<double>(atoms.numAtoms);
  const double centroidX    = groupAllReduceSum<kWarpSize>(sumX) * inverseAtoms;
  const double centroidY    = groupAllReduceSum<kWarpSize>(sumY) * inverseAtoms;
  const double centroidZ    = groupAllReduceSum<kWarpSize>(sumZ) * inverseAtoms;

  double moments[6] = {};
  for (int atomIdx = lane; atomIdx < atoms.numAtoms; atomIdx += kWarpSize) {
    const double x = atoms.positions[atomIdx * 3 + 0] - centroidX;
    const double y = atoms.positions[atomIdx * 3 + 1] - centroidY;
    const double z = atoms.positions[atomIdx * 3 + 2] - centroidZ;
    moments[0] += x * x;
    moments[1] += x * y;
    moments[2] += x * z;
    moments[3] += y * y;
    moments[4] += y * z;
    moments[5] += z * z;
  }
  for (double& moment : moments) {
    moment = groupAllReduceSum<kWarpSize>(moment);
  }
  double eigenvalues[3];
  double eigenvectors[9];
  symmetricEigensystemJacobi3x3(moments[0],
                                moments[1],
                                moments[2],
                                moments[3],
                                moments[4],
                                moments[5],
                                eigenvalues,
                                eigenvectors);
  double inverseRoots[3];
  for (int axis = 0; axis < 3; ++axis) {
    const double singular = fabs(eigenvalues[axis]);
    inverseRoots[axis]    = singular > kGetawayPinvTolerance ? 1.0 / sqrt(singular) : 0.0;
  }

  for (int atomIdx = lane; atomIdx < atoms.numAtoms; atomIdx += kWarpSize) {
    const double x        = atoms.positions[atomIdx * 3 + 0] - centroidX;
    const double y        = atoms.positions[atomIdx * 3 + 1] - centroidY;
    const double z        = atoms.positions[atomIdx * 3 + 2] - centroidZ;
    Real* const  row      = rows + atomIdx * kGetawayRowStride;
    double       leverage = 0;
    for (int axis = 0; axis < 3; ++axis) {
      const double scaled =
        (x * eigenvectors[axis] + y * eigenvectors[3 + axis] + z * eigenvectors[6 + axis]) * inverseRoots[axis];
      row[kGetawayRowY + axis] = static_cast<Real>(scaled);
      leverage += scaled * scaled;
    }
    leverages[atomIdx]       = leverage;
    row[kGetawayRowLeverage] = static_cast<Real>(leverage);
    row[kGetawayRowRootLev]  = static_cast<Real>(sqrt(leverage));
    for (int channel = 0; channel < kNumAtomPropertyChannels; ++channel) {
      row[kGetawayRowWeights + channel] =
        static_cast<Real>(propertyWeights[static_cast<size_t>(channel) * totalAtoms + atomIdx]);
    }
    row[kGetawayRowCentered + 0] = static_cast<Real>(x);
    row[kGetawayRowCentered + 1] = static_cast<Real>(y);
    row[kGetawayRowCentered + 2] = static_cast<Real>(z);
  }
  __syncwarp();
}

//! Atom-property weight of channel @p channel (0 is the unweighted channel) from a scratch row.
template <typename Real> __device__ __forceinline__ Real getawayWeight(const Real* row, const int channel) {
  return channel == 0 ? Real(1) : row[kGetawayRowWeights + channel - 1];
}

//! Distance between two atoms from their scratch rows' centered coordinates.
template <typename Real> __device__ __forceinline__ Real getawayDistance(const Real* first, const Real* second) {
  const Real dx = first[kGetawayRowCentered + 0] - second[kGetawayRowCentered + 0];
  const Real dy = first[kGetawayRowCentered + 1] - second[kGetawayRowCentered + 1];
  const Real dz = first[kGetawayRowCentered + 2] - second[kGetawayRowCentered + 2];
  return sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * @brief Warp-collective. (R v)_i = sqrt(h_i) * sum_{k != i} sqrt(h_k) v_k / r_ik for every atom i into
 *        @p product; returns v^T R v and |R v|^2, reduced over the warp. Rows are computed on the fly.
 */
template <typename Real>
__device__ __forceinline__ void multiplyByInfluenceMatrix(const Real* rows,
                                                          const int   numAtoms,
                                                          const int   lane,
                                                          const Real* vector,
                                                          Real*       product,
                                                          Real&       rayleighNumerator,
                                                          Real&       productNormSquared) {
  Real numerator = 0;
  Real norm      = 0;
  for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
    const Real* rowI = rows + atomIdx * kGetawayRowStride;
    Real        sum  = 0;
    for (int other = 0; other < numAtoms; ++other) {
      if (other != atomIdx) {
        const Real* rowK = rows + other * kGetawayRowStride;
        sum += rowK[kGetawayRowRootLev] * vector[other] / getawayDistance(rowI, rowK);
      }
    }
    const Real value = rowI[kGetawayRowRootLev] * sum;
    product[atomIdx] = value;
    numerator += vector[atomIdx] * value;
    norm += value * value;
  }
  rayleighNumerator  = groupAllReduceSum<kWarpSize>(numerator);
  productNormSquared = groupAllReduceSum<kWarpSize>(norm);
  __syncwarp();
}

/**
 * @brief RDKit's GETAWAY for every conformer; one warp per conformer.
 *
 * Per conformer: leverages and scratch rows (computeLeverages()); per-atom H0/HATS0 sums, HIC (dimension from
 * the PBF of the molecule's default conformer, as RDKit's PBF(mol) call does), HGM (sequential product, so
 * underflow matches RDKit) and ITH/ISH (computeInformationIndices()); then the pair loop: each kGroupSize-lane
 * group walks interleaved rows j of the pair triangle, finds row j's bond distances (capped at
 * kNumGetawayLags) with searchBondDepths() into its @p bondDepths array, and the lane owning a pair's lag
 * accumulates its H, HATS and R terms; pairs farther apart or disconnected enter the totals only. REIG, the largest
 * singular value of the nonnegative symmetric R, is its Perron eigenvalue, found by power iteration whose first product
 * (with ones) gives the row sums for RARS and RCON.
 */
template <typename Real>
__global__ void getaway3DKernel(const DeviceCoordView        coordinates,
                                const Property3DDeviceInputs inputs,
                                const int                    precisionDigits,
                                Real* __restrict__ rows,
                                double* __restrict__ leverageScratch,
                                double* __restrict__ sortScratch,
                                int64_t* __restrict__ keyScratch,
                                Real* __restrict__ iterateScratch,
                                uint8_t* __restrict__ bondDepths,
                                const BondDistanceTable bondTable,
                                Real* __restrict__ output) {
  const int lane         = static_cast<int>(threadIdx.x) % kWarpSize;
  const int laneInGroup  = lane % kGroupSize;
  const int stream       = lane / kGroupSize;
  const int conformerIdx = blockIdx.x * kWarpsPerBlock + static_cast<int>(threadIdx.x) / kWarpSize;
  if (conformerIdx >= coordinates.numConformers) {
    return;  // Uniform across the warp.
  }
  Real* const          row   = output + static_cast<size_t>(conformerIdx) * kNumGetawayProperties;
  const ConformerAtoms atoms = loadConformer(coordinates, nullptr, inputs.moleculeAtomStarts, conformerIdx);
  if (!atoms.valid) {
    for (int valueIdx = lane; valueIdx < kNumGetawayProperties; valueIdx += kWarpSize) {
      row[valueIdx] = static_cast<Real>(nan(""));
    }
    return;
  }
  const int     numAtoms      = atoms.numAtoms;
  const int     moleculeIdx   = coordinates.molIndices[conformerIdx];
  const int     moleculeStart = inputs.moleculeAtomStarts[moleculeIdx];
  const int     totalAtoms    = inputs.moleculeAtomStarts[coordinates.nMols];
  const size_t  atomOffset    = static_cast<size_t>(atoms.positions - coordinates.positions) / 3;
  Real* const   atomRows      = rows + atomOffset * kGetawayRowStride;
  double* const leverages     = leverageScratch + atomOffset;
  computeLeverages(atoms, inputs.atomPropertyWeights + moleculeStart, totalAtoms, lane, atomRows, leverages);

  // HIC's dimension: RDKit's PBF of the molecule's default conformer below kGetawayPlanarPbf means 2D.
  int            defaultRow = inputs.defaultConformerRows != nullptr ? inputs.defaultConformerRows[moleculeIdx] : -1;
  ConformerAtoms defaultAtoms =
    defaultRow >= 0 ? loadConformer(coordinates, nullptr, inputs.moleculeAtomStarts, defaultRow) : ConformerAtoms{};
  if (!defaultAtoms.valid) {
    defaultRow   = conformerIdx;
    defaultAtoms = atoms;
  }
  ProjectionState<double> plane;
  computeProjectionCentroid(defaultAtoms, laneInGroup, plane);
  computeProjectionPca(defaultAtoms, nullptr, laneInGroup, plane);
  const double pbf       = computePbf(defaultAtoms, laneInGroup, plane);
  const bool   is3D      = inputs.conformerIs3D == nullptr || inputs.conformerIs3D[defaultRow] != 0;
  const double dimension = (defaultAtoms.numAtoms < 4 || !is3D ? 0.0 : pbf) < kGetawayPlanarPbf ? 2.0 : 3.0;

  // Pair loop: lane l owns lag l + 1; pairs beyond kNumGetawayLags (or disconnected) go to lane k % kGroupSize.
  Real lagH[kNumGetawayChannels]    = {};
  Real lagHats[kNumGetawayChannels] = {};
  Real lagR[kNumGetawayChannels]    = {};
  Real lagRMax[kNumGetawayChannels] = {};
  Real farH[kNumGetawayChannels]    = {};
  Real farHats[kNumGetawayChannels] = {};
  Real farR[kNumGetawayChannels]    = {};
  {
    const int32_t* neighborStarts = inputs.bondNeighborStarts + moleculeStart;
    uint8_t*       depth     = bondDepths + atomOffset * kGetawayPairStreams + static_cast<size_t>(stream) * numAtoms;
    const unsigned groupMask = 0xffu << (stream * kGroupSize);
    const uint8_t* moleculeTable = moleculeBondDistances(bondTable, moleculeIdx);
    for (int j = stream; j < numAtoms - 1; j += kGetawayPairStreams) {
      // Row j's bond distances, indexed by k - lagBase: the molecule's table row, or a search into depth.
      const uint8_t* lags    = depth;
      int            lagBase = 0;
      if (moleculeTable != nullptr) {
        lags    = moleculeTable + bondTableRowStart(j, numAtoms);
        lagBase = j + 1;
      } else {
        searchBondDepths<kNumGetawayLags>(neighborStarts,
                                          inputs.bondNeighbors,
                                          numAtoms,
                                          j,
                                          laneInGroup,
                                          groupMask,
                                          depth);
      }
      const Real* rowJ = atomRows + j * kGetawayRowStride;
      for (int k = j + 1; k < numAtoms; ++k) {
        const int  lag   = lags[k - lagBase];
        const bool near  = lag <= kNumGetawayLags;
        const bool owner = near ? lag - 1 == laneInGroup : k % kGroupSize == laneInGroup;
        if (!owner) {
          continue;
        }
        const Real* rowK      = atomRows + k * kGetawayRowStride;
        const Real  hatsTerm  = rowJ[kGetawayRowLeverage] * rowK[kGetawayRowLeverage];
        const Real  influence = rowJ[kGetawayRowY + 0] * rowK[kGetawayRowY + 0] +
                               rowJ[kGetawayRowY + 1] * rowK[kGetawayRowY + 1] +
                               rowJ[kGetawayRowY + 2] * rowK[kGetawayRowY + 2];
        const Real rTerm = rowJ[kGetawayRowRootLev] * rowK[kGetawayRowRootLev] / getawayDistance(rowJ, rowK);
        for (int channel = 0; channel < kNumGetawayChannels; ++channel) {
          const Real product = getawayWeight(rowJ, channel) * getawayWeight(rowK, channel);
          if (near) {
            lagHats[channel] += product * hatsTerm;
            if (influence > Real(0)) {
              lagH[channel] += product * influence;
            }
            const Real weighted = product * rTerm;
            lagR[channel] += weighted;
            if (weighted > lagRMax[channel]) {
              lagRMax[channel] = weighted;
            }
          } else {
            farHats[channel] += product * hatsTerm;
            if (influence > Real(0)) {
              farH[channel] += product * influence;
            }
            farR[channel] += product * rTerm;
          }
        }
      }
    }
  }
  for (int channel = 0; channel < kNumGetawayChannels; ++channel) {
    lagH[channel]    = sumAcrossGroups(lagH[channel]);
    lagHats[channel] = sumAcrossGroups(lagHats[channel]);
    lagR[channel]    = sumAcrossGroups(lagR[channel]);
    lagRMax[channel] = maxAcrossGroups(lagRMax[channel]);
    farH[channel]    = sumAcrossGroups(farH[channel]);
    farHats[channel] = sumAcrossGroups(farHats[channel]);
    farR[channel]    = sumAcrossGroups(farR[channel]);
  }

  // Lag 0: H0 = sum w^2 h and HATS0 = sum w^2 h^2 over atoms with h > 0; HIC from every leverage. Computed
  // after the pair loop, and the H and R blocks written before REIG, so neither set of sums holds registers
  // through the other's loop (168 -> 128 registers in float32).
  Real   lagZeroH[kNumGetawayChannels]    = {};
  Real   lagZeroHats[kNumGetawayChannels] = {};
  double hic                              = 0;
  for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
    const Real*  atomRow  = atomRows + atomIdx * kGetawayRowStride;
    const Real   leverage = atomRow[kGetawayRowLeverage];
    const double exact    = leverages[atomIdx];
    hic -= exact / dimension * log(exact / dimension) / log(2.0);
    if (exact > 0.0) {
      for (int channel = 0; channel < kNumGetawayChannels; ++channel) {
        const Real weight = getawayWeight(atomRow, channel);
        lagZeroH[channel] += weight * weight * leverage;
        lagZeroHats[channel] += weight * weight * leverage * leverage;
      }
    }
  }
  hic = groupAllReduceSum<kWarpSize>(hic);
  for (int channel = 0; channel < kNumGetawayChannels; ++channel) {
    lagZeroH[channel]    = groupAllReduceSum<kWarpSize>(lagZeroH[channel]);
    lagZeroHats[channel] = groupAllReduceSum<kWarpSize>(lagZeroHats[channel]);
  }

  if (stream == 0) {
    for (int channel = 0; channel < kNumGetawayChannels; ++channel) {
      Real* const hBlock     = row + kGetawayHStart + channel * kGetawayHBlock;
      Real* const rBlock     = row + kGetawayRStart + channel * kGetawayRBlock;
      const Real  lagHSum    = groupAllReduceSum(lagH[channel]);
      const Real  lagHatsSum = groupAllReduceSum(lagHats[channel]);
      const Real  lagRSum    = groupAllReduceSum(lagR[channel]);
      const Real  farHSum    = groupAllReduceSum(farH[channel]);
      const Real  farHatsSum = groupAllReduceSum(farHats[channel]);
      const Real  farRSum    = groupAllReduceSum(farR[channel]);
      const Real  rMax =
        cooperative_groups::reduce(laneTile<kGroupSize>(), lagRMax[channel], cooperative_groups::greater<Real>());
      const int lagIdx                     = laneInGroup + 1;
      hBlock[lagIdx]                       = roundThousandths(lagH[channel]);
      hBlock[kNumGetawayLags + 2 + lagIdx] = roundThousandths(lagHats[channel]);
      rBlock[lagIdx - 1]                   = roundThousandths(lagR[channel]);
      rBlock[kNumGetawayLags + lagIdx]     = roundThousandths(lagRMax[channel]);
      if (laneInGroup == 0) {
        hBlock[0]                   = roundThousandths(lagZeroH[channel]);
        hBlock[kNumGetawayLags + 1] = roundThousandths(lagZeroH[channel] + Real(2) * (lagHSum + farHSum));
        hBlock[kNumGetawayLags + 2] = roundThousandths(lagZeroHats[channel]);
        hBlock[kGetawayHBlock - 1]  = roundThousandths(lagZeroHats[channel] + Real(2) * (lagHatsSum + farHatsSum));
        rBlock[kNumGetawayLags]     = roundThousandths(Real(2) * (lagRSum + farRSum));
        rBlock[kGetawayRBlock - 1]  = roundThousandths(rMax);
      }
    }
  }

  // REIG by power iteration on R; the first product with ones gives R's row sums for RARS and RCON.
  Real* const current = iterateScratch + atomOffset * 2;
  Real* const next    = current + numAtoms;
  for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
    current[atomIdx] = Real(1);
  }
  __syncwarp();
  Real numerator;
  Real normSquared;
  multiplyByInfluenceMatrix(atomRows, numAtoms, lane, current, next, numerator, normSquared);
  Real rowSum = 0;
  Real rcon   = 0;
  for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
    rowSum += next[atomIdx];
    for (int edge = inputs.bondNeighborStarts[moleculeStart + atomIdx];
         edge < inputs.bondNeighborStarts[moleculeStart + atomIdx + 1];
         ++edge) {
      const int neighbor = inputs.bondNeighbors[edge];
      if (neighbor > atomIdx) {
        rcon += sqrt(next[atomIdx] * next[neighbor]);
      }
    }
  }
  const Real rars           = groupAllReduceSum<kWarpSize>(rowSum) / static_cast<Real>(numAtoms);
  rcon                      = groupAllReduceSum<kWarpSize>(rcon);
  Real           eigenvalue = numerator / static_cast<Real>(numAtoms);
  constexpr Real kTolerance = sizeof(Real) == sizeof(double) ? Real(1e-13) : Real(2e-7);
  for (int iteration = 0; iteration < kGetawayMaxIterations && normSquared > Real(0); ++iteration) {
    const Real inverseNorm = Real(1) / sqrt(normSquared);
    for (int atomIdx = lane; atomIdx < numAtoms; atomIdx += kWarpSize) {
      current[atomIdx] = next[atomIdx] * inverseNorm;
    }
    __syncwarp();
    multiplyByInfluenceMatrix(atomRows, numAtoms, lane, current, next, numerator, normSquared);
    const Real previous = eigenvalue;
    eigenvalue          = numerator;
    if (fabs(eigenvalue - previous) <= kTolerance * fabs(eigenvalue)) {
      break;
    }
  }
  if (!(normSquared > Real(0))) {
    eigenvalue = Real(0);
  }

  double         ith  = 0;
  double         ish  = 0;
  int64_t* const keys = keyScratch + atomOffset * 2;
  computeInformationIndices(leverages,
                            inputs.heavyAtomFlags + moleculeStart,
                            numAtoms,
                            precisionDigits,
                            lane,
                            sortScratch + atomOffset,
                            keys,
                            keys + numAtoms,
                            ith,
                            ish);
  if (lane == 0) {
    double leverageProduct = 1.0;
    for (int atomIdx = 0; atomIdx < numAtoms; ++atomIdx) {
      leverageProduct *= leverages[atomIdx];
    }
    const double hgm           = 100.0 * pow(leverageProduct, 1.0 / numAtoms);
    row[0]                     = static_cast<Real>(roundThousandths(ith));
    row[1]                     = static_cast<Real>(roundThousandths(ish));
    row[2]                     = static_cast<Real>(roundThousandths(hic));
    row[3]                     = static_cast<Real>(roundThousandths(hgm));
    row[kGetawayRconIndex + 0] = roundThousandths(rcon);
    row[kGetawayRconIndex + 1] = roundThousandths(rars);
    row[kGetawayRconIndex + 2] = roundThousandths(eigenvalue);
  }
}

/**
 * @brief Launches the GETAWAY kernel. Scratch per atom: one kGetawayRowStride row of @p Real, a float64
 *        leverage and rounded-leverage slot, two ITH/ISH sort keys, two power-iteration values and one
 *        bond-depth byte per pair stream.
 */
template <typename Real>
void launchGetawayProperties(const DeviceCoordView&        coordinates,
                             const Property3DDeviceInputs& inputs,
                             const GetawayOptions&         options,
                             const BondDistanceTable&      bondTable,
                             Real*                         output,
                             const cudaStream_t            stream) {
  const int numConformers = coordinates.numConformers;
  if (numConformers == 0 || output == nullptr) {
    return;
  }
  const size_t               numAtoms = static_cast<size_t>(coordinates.numAtoms);
  AsyncDeviceVector<Real>    rows(numAtoms * kGetawayRowStride, stream);
  AsyncDeviceVector<double>  leverages(numAtoms, stream);
  AsyncDeviceVector<double>  sorted(numAtoms, stream);
  AsyncDeviceVector<int64_t> keys(numAtoms * 2, stream);
  AsyncDeviceVector<Real>    iterates(numAtoms * 2, stream);
  AsyncDeviceVector<uint8_t> depths(numAtoms * kGetawayPairStreams, stream);
  const int                  numBlocks = (numConformers + kWarpsPerBlock - 1) / kWarpsPerBlock;
  getaway3DKernel<Real><<<numBlocks, kBlockSize, 0, stream>>>(coordinates,
                                                              inputs,
                                                              static_cast<int>(options.precision),
                                                              rows.data(),
                                                              leverages.data(),
                                                              sorted.data(),
                                                              keys.data(),
                                                              iterates.data(),
                                                              depths.data(),
                                                              bondTable,
                                                              output);
  cudaCheckError(cudaGetLastError());
}

}  // namespace nvMolKit::descriptors3d_detail

#endif  // NVMOLKIT_DESCRIPTORS3D_GETAWAY_CUH
