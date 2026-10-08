// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_DESCRIPTORS3D_EEM_CUH
#define NVMOLKIT_DESCRIPTORS3D_EEM_CUH

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include "src/descriptors3d.h"
#include "src/descriptors3d_kernel.cuh"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"

namespace nvMolKit::descriptors3d_detail {

//! RDKit's EEM kappa (Code/GraphMol/Descriptors/EEM.cpp).
constexpr double kEemKappa = 0.5125;

constexpr int kEemBlockSize   = 128;
//! Dynamic shared memory for one conformer's workspace; larger workspaces use global scratch. Leaves room for
//! the solver's static shared memory below the 48 KiB launch limit.
constexpr int kEemSharedBytes = 47 * 1024;

__host__ __device__ constexpr int64_t alignTo8(const int64_t bytes) {
  return (bytes + 7) / 8 * 8;
}

/**
 * @brief Per-conformer workspace of an @p numAtoms -atom EEM solve: the augmented (n + 1) x (n + 2) system of
 *        @p Real, then the float64 first solution and the pivot rows used by float32 refinement.
 */
template <typename Real> struct EemWorkspace {
  Real*    system;
  double*  firstSolution;
  int32_t* pivots;

  __host__ __device__ static constexpr int64_t systemBytes(const int64_t numAtoms) {
    return alignTo8((numAtoms + 1) * (numAtoms + 2) * static_cast<int64_t>(sizeof(Real)));
  }
  __host__ __device__ static constexpr int64_t bytes(const int64_t numAtoms) {
    return systemBytes(numAtoms) + alignTo8((numAtoms + 1) * static_cast<int64_t>(sizeof(double) + sizeof(int32_t)));
  }
  __device__ static EemWorkspace at(unsigned char* base, const int numAtoms) {
    EemWorkspace workspace;
    workspace.system        = reinterpret_cast<Real*>(base);
    workspace.firstSolution = reinterpret_cast<double*>(base + systemBytes(numAtoms));
    workspace.pivots        = reinterpret_cast<int32_t*>(workspace.firstSolution + numAtoms + 1);
    return workspace;
  }
};

/**
 * @brief Block-collective LU factorization with partial pivoting of the augmented system @p system (rows of
 *        `size + 1` entries, the last column the right-hand side). Rows are swapped whole, so on return the
 *        strictly lower part holds the unit-lower factor's multipliers, the upper part the upper factor, the
 *        last column the forward-eliminated right-hand side, and `pivots[k]` the row swapped into row k. Ties
 *        between pivot candidates go to the lowest row.
 */
template <typename Real> __device__ void factorAugmentedSystem(Real* system, int32_t* pivots, const int size) {
  __shared__ int pivotRow;
  const int      stride = size + 1;
  const int      lane   = static_cast<int>(threadIdx.x) % kWarpSize;
  for (int k = 0; k < size; ++k) {
    if (threadIdx.x < kWarpSize) {
      Real bestValue = -1;
      int  bestRow   = k;
      for (int row = k + lane; row < size; row += kWarpSize) {
        const Real value = fabs(system[row * stride + k]);
        if (value > bestValue) {
          bestValue = value;
          bestRow   = row;
        }
      }
      for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        const Real otherValue = __shfl_xor_sync(0xffffffffu, bestValue, offset);
        const int  otherRow   = __shfl_xor_sync(0xffffffffu, bestRow, offset);
        if (otherValue > bestValue || (otherValue == bestValue && otherRow < bestRow)) {
          bestValue = otherValue;
          bestRow   = otherRow;
        }
      }
      if (lane == 0) {
        pivotRow  = bestRow;
        pivots[k] = bestRow;
      }
    }
    __syncthreads();
    const int pivot = pivotRow;
    if (pivot != k) {
      for (int col = static_cast<int>(threadIdx.x); col < stride; col += blockDim.x) {
        const Real swapped           = system[k * stride + col];
        system[k * stride + col]     = system[pivot * stride + col];
        system[pivot * stride + col] = swapped;
      }
      __syncthreads();
    }
    const Real inversePivot = Real(1) / system[k * stride + k];
    for (int row = k + 1 + static_cast<int>(threadIdx.x); row < size; row += blockDim.x) {
      system[row * stride + k] *= inversePivot;
    }
    __syncthreads();
    const int trailingCols = stride - k - 1;
    const int trailing     = (size - k - 1) * trailingCols;
    for (int idx = static_cast<int>(threadIdx.x); idx < trailing; idx += blockDim.x) {
      const int row = k + 1 + idx / trailingCols;
      const int col = k + 1 + idx % trailingCols;
      system[row * stride + col] -= system[row * stride + k] * system[k * stride + col];
    }
    __syncthreads();
  }
}

//! Block-collective. Column-oriented forward substitution with the unit-lower factor, in place on the
//! right-hand-side column.
template <typename Real> __device__ void forwardSubstitute(Real* system, const int size) {
  const int stride = size + 1;
  for (int k = 0; k < size - 1; ++k) {
    const Real solved = system[k * stride + size];
    for (int row = k + 1 + static_cast<int>(threadIdx.x); row < size; row += blockDim.x) {
      system[row * stride + size] -= system[row * stride + k] * solved;
    }
    __syncthreads();
  }
}

//! Block-collective. Column-oriented back substitution with the upper factor, leaving the solution in the
//! right-hand-side column.
template <typename Real> __device__ void backSubstitute(Real* system, const int size) {
  const int stride = size + 1;
  for (int k = size - 1; k >= 0; --k) {
    const Real solution = system[k * stride + size] / system[k * stride + k];
    for (int row = static_cast<int>(threadIdx.x); row < k; row += blockDim.x) {
      system[row * stride + size] -= system[row * stride + k] * solution;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      system[k * stride + size] = solution;
    }
  }
  __syncthreads();
}

//! Atoms, parameters and charge of one conformer's EEM system.
struct EemConformer {
  const double* positions;
  const double* electronegativity;
  const double* hardness;
  double        formalCharge;
  int           numAtoms;
};

/**
 * @brief Entry (@p row, @p col) of RDKit's EEM system as it solves it, the right-hand side in column
 *        `numAtoms + 1`. Sets @p coincident when an atom pair's squared distance is not positive.
 *
 * RDKit fills the system row-major and hands it to Eigen as column-major, so it solves the transpose of the
 * matrix its code describes: row i < n is `sum_j J_ij q_j + chi = -A_i` with `J_ii = B_i` and
 * `J_ij = kappa / r_ij`, and the last row is `-sum_j q_j = Q`. The charges therefore sum to the negated
 * formal charge, which this reproduces.
 *
 * FP64 required: the coordinate differences, before the conversion to @p T (see centeredPosition()).
 */
template <typename T>
__device__ __forceinline__ T
eemSystemEntry(const EemConformer& conformer, const int row, const int col, bool& coincident) {
  const int numAtoms = conformer.numAtoms;
  if (col == numAtoms + 1) {
    return row < numAtoms ? static_cast<T>(-conformer.electronegativity[row]) : static_cast<T>(conformer.formalCharge);
  }
  if (row == numAtoms) {
    return col < numAtoms ? T(-1) : T(0);
  }
  if (col == numAtoms) {
    return T(1);
  }
  if (row == col) {
    return static_cast<T>(conformer.hardness[row]);
  }
  const double* positions = conformer.positions;
  const T       dx        = static_cast<T>(positions[row * 3 + 0] - positions[col * 3 + 0]);
  const T       dy        = static_cast<T>(positions[row * 3 + 1] - positions[col * 3 + 1]);
  const T       dz        = static_cast<T>(positions[row * 3 + 2] - positions[col * 3 + 2]);
  const T       squared   = dx * dx + dy * dy + dz * dz;
  coincident |= !(squared > T(0));
  return static_cast<T>(kEemKappa) / sqrt(squared);
}

/**
 * @brief Block-collective. Replaces the float32 solution in the factored @p workspace system's right-hand-side
 *        column by the correction of one iterative-refinement step, keeping the solution in
 *        `workspace.firstSolution`.
 *
 * FP64 required: the residual. float32 elimination of the bordered system accumulates rounding that grows with
 * the atom count, ~3e-4 absolute on 160-340-atom conformers against ~5e-7 from rounding the entries alone
 * (measured); one refinement step with the float32 factors and a float64 residual recovers ~5e-10.
 */
__device__ inline void refineEemSolution(const EemConformer& conformer, const EemWorkspace<float>& workspace) {
  const int size   = conformer.numAtoms + 1;
  const int stride = size + 1;
  float*    system = workspace.system;
  for (int row = static_cast<int>(threadIdx.x); row < size; row += blockDim.x) {
    workspace.firstSolution[row] = system[row * stride + size];
  }
  __syncthreads();
  // Residual b - A x in the original row order, written to each row's right-hand side.
  for (int row = static_cast<int>(threadIdx.x); row < size; row += blockDim.x) {
    bool   ignored  = false;
    double residual = eemSystemEntry<double>(conformer, row, size, ignored);
    for (int col = 0; col < size; ++col) {
      residual -= eemSystemEntry<double>(conformer, row, col, ignored) * workspace.firstSolution[col];
    }
    system[row * stride + size] = static_cast<float>(residual);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int k = 0; k < size; ++k) {
      const int pivot = workspace.pivots[k];
      if (pivot != k) {
        const float swapped           = system[k * stride + size];
        system[k * stride + size]     = system[pivot * stride + size];
        system[pivot * stride + size] = swapped;
      }
    }
  }
  __syncthreads();
  forwardSubstitute(system, size);
  backSubstitute(system, size);
}

/**
 * @brief One block per conformer (grid-stride). Builds RDKit's EEM system (see eemSystemEntry()) and writes
 *        one partial charge per atom to @p output, laid out like the coordinate positions.
 */
template <typename Real>
__global__ void __launch_bounds__(kEemBlockSize) eemChargesKernel(const DeviceCoordView        coordinates,
                                                                  const Property3DDeviceInputs inputs,
                                                                  const int64_t                sharedCapacity,
                                                                  unsigned char*               globalScratch,
                                                                  const int64_t                scratchStride,
                                                                  Real*                        output) {
  extern __shared__ __align__(16) unsigned char eemSharedBytes[];
  for (int conformerIdx = blockIdx.x; conformerIdx < coordinates.numConformers; conformerIdx += gridDim.x) {
    const int64_t atomStart = coordinates.atomStarts[conformerIdx];
    const int64_t atomStop  = coordinates.atomStarts[conformerIdx + 1];
    if (atomStart < 0 || atomStop <= atomStart || atomStop > coordinates.numAtoms) {
      continue;  // No rows to write; the output was initialized to NaN.
    }
    const int numAtoms    = static_cast<int>(atomStop - atomStart);
    const int moleculeIdx = coordinates.molIndices[conformerIdx];
    bool      valid       = moleculeIdx >= 0 && moleculeIdx < coordinates.nMols && numAtoms <= inputs.maxMoleculeAtoms;
    int       paramStart  = 0;
    if (valid) {
      paramStart = inputs.moleculeAtomStarts[moleculeIdx];
      valid      = inputs.moleculeAtomStarts[moleculeIdx + 1] - paramStart == numAtoms;
    }
    bool missingParameters = false;
    for (int atomIdx = static_cast<int>(threadIdx.x); valid && atomIdx < numAtoms; atomIdx += blockDim.x) {
      missingParameters |= !isfinite(inputs.eemElectronegativity[paramStart + atomIdx]);
    }
    valid = valid && !__syncthreads_or(missingParameters);

    bool               coincident = false;
    EemConformer       conformer{};
    EemWorkspace<Real> workspace{};
    const int          size   = numAtoms + 1;
    const int          stride = size + 1;
    if (valid) {
      conformer = EemConformer{coordinates.positions + atomStart * 3,
                               inputs.eemElectronegativity + paramStart,
                               inputs.eemHardness + paramStart,
                               inputs.moleculeFormalCharges[moleculeIdx],
                               numAtoms};
      workspace = EemWorkspace<Real>::at(EemWorkspace<Real>::bytes(numAtoms) <= sharedCapacity ?
                                           eemSharedBytes :
                                           globalScratch + blockIdx.x * scratchStride,
                                         numAtoms);
      for (int64_t idx = threadIdx.x; idx < static_cast<int64_t>(size) * stride; idx += blockDim.x) {
        workspace.system[idx] =
          eemSystemEntry<Real>(conformer, static_cast<int>(idx / stride), static_cast<int>(idx % stride), coincident);
      }
    }
    // Coincident atoms (or non-finite coordinates) put an infinite kappa / r in the system, whose solution is
    // meaningless; RDKit's full-pivot LU returns all-zero charges for it.
    valid = valid && !__syncthreads_or(coincident);
    if (!valid) {
      for (int atomIdx = static_cast<int>(threadIdx.x); atomIdx < numAtoms; atomIdx += blockDim.x) {
        output[atomStart + atomIdx] = static_cast<Real>(nan(""));
      }
      __syncthreads();
      continue;
    }

    factorAugmentedSystem(workspace.system, workspace.pivots, size);
    backSubstitute(workspace.system, size);
    if constexpr (std::is_same_v<Real, float>) {
      refineEemSolution(conformer, workspace);
    }
    for (int atomIdx = static_cast<int>(threadIdx.x); atomIdx < numAtoms; atomIdx += blockDim.x) {
      if constexpr (std::is_same_v<Real, float>) {
        output[atomStart + atomIdx] = static_cast<float>(
          workspace.firstSolution[atomIdx] + static_cast<double>(workspace.system[atomIdx * stride + size]));
      } else {
        output[atomStart + atomIdx] = workspace.system[atomIdx * stride + size];
      }
    }
    // The next conformer reuses the workspace.
    __syncthreads();
  }
}

template <typename Real>
void launchEemCharges(const DeviceCoordView&        coordinates,
                      const Property3DDeviceInputs& inputs,
                      Real*                         output,
                      const cudaStream_t            stream) {
  if (output == nullptr) {
    return;
  }
  // Bytes of 0xFF are NaN for float and double: rows outside every conformer's atom range stay NaN.
  cudaCheckError(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(coordinates.numAtoms) * sizeof(Real), stream));
  if (coordinates.numConformers == 0) {
    return;
  }

  int device = 0;
  int numSms = 0;
  cudaCheckError(cudaGetDevice(&device));
  cudaCheckError(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device));
  const int                        numBlocks        = std::min(coordinates.numConformers, numSms * 16);
  const int64_t                    largestWorkspace = EemWorkspace<Real>::bytes(inputs.maxMoleculeAtoms);
  const int64_t                    sharedCapacity   = std::min<int64_t>(largestWorkspace, kEemSharedBytes);
  const int64_t                    scratchStride    = largestWorkspace > sharedCapacity ? largestWorkspace : 0;
  AsyncDeviceVector<unsigned char> scratch(static_cast<size_t>(scratchStride * numBlocks), stream);
  eemChargesKernel<Real><<<numBlocks, kEemBlockSize, sharedCapacity, stream>>>(coordinates,
                                                                               inputs,
                                                                               sharedCapacity,
                                                                               scratch.data(),
                                                                               scratchStride,
                                                                               output);
  cudaCheckError(cudaGetLastError());
}

}  // namespace nvMolKit::descriptors3d_detail

#endif  // NVMOLKIT_DESCRIPTORS3D_EEM_CUH
