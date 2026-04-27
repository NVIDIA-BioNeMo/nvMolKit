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
#include <cooperative_groups/reduce.h>

#include <cub/cub.cuh>
#include <vector>

#include "src/minimizer/bfgs_hessian.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"
#include "src/utils/host_vector.h"
namespace cg = cooperative_groups;

namespace nvMolKit {

namespace {

constexpr int warpSize  = 32;
constexpr int blockSize = 512;
constexpr int numWarp   = blockSize / warpSize;
constexpr int maxAtom   = 256;

__device__ __forceinline__ void computeBfgsSums(const double*                    dGrad,
                                                const double*                    xi,
                                                const double*                    hessDGrad,
                                                double*                          facShared,
                                                double*                          faeShared,
                                                double*                          sumDGradShared,
                                                double*                          sumXiShared,
                                                cg::thread_block_tile<warpSize>& warp,
                                                int                              warpIdx,
                                                int                              laneIdx,
                                                int                              dim) {
  double sumTerm = 0.0;
  if (warpIdx == 0) {
    for (int i = laneIdx; i < dim; i += warpSize) {
      sumTerm += dGrad[i] * xi[i];
    }
    cg::reduce_store_async(warp, &facShared[0], sumTerm, cg::plus<double>{});
  } else if (warpIdx == 1) {
    for (int i = laneIdx; i < dim; i += warpSize) {
      sumTerm += dGrad[i] * hessDGrad[i];
    }
    cg::reduce_store_async(warp, &faeShared[0], sumTerm, cg::plus<double>{});
  } else if (warpIdx == 2) {
    for (int i = laneIdx; i < dim; i += warpSize) {
      sumTerm += dGrad[i] * dGrad[i];
    }
    cg::reduce_store_async(warp, &sumDGradShared[0], sumTerm, cg::plus<double>{});
  } else if (warpIdx == 3) {
    for (int i = laneIdx; i < dim; i += warpSize) {
      sumTerm += xi[i] * xi[i];
    }
    cg::reduce_store_async(warp, &sumXiShared[0], sumTerm, cg::plus<double>{});
  }
}

__device__ __forceinline__ void computeUpdateFlag(int     idxWithinSystem,
                                                  double* facShared,
                                                  double* faeShared,
                                                  double* fadShared,
                                                  double* sumDGradShared,
                                                  double* sumXiShared,
                                                  bool*   needUpdateInverseHessian) {
  if (idxWithinSystem == 0) {
    constexpr double EPS                  = 3e-8;
    const double     sumXi                = sumXiShared[0];
    const double     sumDGrad             = sumDGradShared[0];
    const double     fac                  = facShared[0];
    const double     fae                  = faeShared[0];
    bool             updateInverseHessian = fac > sqrt(EPS * sumDGrad * sumXi);
    if (updateInverseHessian) {
      facShared[0] = 1.0 / fac;
      fadShared[0] = 1.0 / fae;
    }
    needUpdateInverseHessian[0] = updateInverseHessian;
  }
}

// Shared-memory optimized kernel used when all systems have <= maxAtom atoms
template <int dataDim>
__global__ void updateInverseHessianBFGSBatchKernelShared(const int16_t* statuses,
                                                          const int*     atomStarts,
                                                          const int*     hessianStarts,
                                                          double*        invHessians,
                                                          double*        dGrads,
                                                          double*        xis,
                                                          double*        hessDGrads,
                                                          const double*  grads,
                                                          const int*     activeSystemIndices) {
  __shared__ double facShared[1];
  __shared__ double faeShared[1];
  __shared__ double fadShared[1];
  __shared__ double sumDGradShared[1];
  __shared__ double sumXiShared[1];
  __shared__ bool   needUpdateInverseHessian[1];

  __shared__ double cachedDGrads[dataDim * maxAtom];
  __shared__ double cachedHessDGrads[dataDim * maxAtom];
  __shared__ double cachedXis[dataDim * maxAtom];
  __shared__ double cachedGrads[dataDim * maxAtom];

  const int sysIdx = activeSystemIndices[blockIdx.x];
  if (statuses != nullptr && statuses[sysIdx] == 0) {
    return;
  }

  cg::thread_block                block           = cg::this_thread_block();
  cg::thread_block_tile<warpSize> warp            = cg::tiled_partition<warpSize>(block);
  const int                       idxWithinSystem = threadIdx.x;
  const int                       warpIdx         = idxWithinSystem / warpSize;
  const int                       laneIdx         = idxWithinSystem % warpSize;

  // Get local pointers. Note that inverse hessian is dim indexed but the atomStart-based ones are * dataDim
  const int           atomOffset      = atomStarts[sysIdx];
  const int           dim             = dataDim * (atomStarts[sysIdx + 1] - atomOffset);
  double* const       invHessianLocal = &invHessians[hessianStarts[sysIdx]];
  const int           absAtomOffset   = atomOffset * dataDim;
  double* const       localDGrad      = &dGrads[absAtomOffset];
  double* const       localHessDGrad  = &hessDGrads[absAtomOffset];
  double* const       localXi         = &xis[absAtomOffset];
  const double* const localGrad       = &grads[absAtomOffset];

  // Load dGrads, Xi, grads into shared memory
  for (int i = idxWithinSystem; i < dim; i += blockSize) {
    cachedDGrads[i] = localDGrad[i];
    cachedXis[i]    = localXi[i];
    cachedGrads[i]  = localGrad[i];
  }

  block.sync();

  // Update hessDGrads
  // Update hessDGrads: Each warp processes different rows
  for (int row = warpIdx; row < dim; row += numWarp) {
    double dotProduct = 0.0;

    // Update hessDGrads: Each thread in warp processes different columns
    for (int col = laneIdx; col < dim; col += warpSize) {
      dotProduct += invHessianLocal[row * dim + col] * cachedDGrads[col];
    }

    cg::reduce_store_async(warp, &cachedHessDGrads[row], dotProduct, cg::plus<double>{});
  }

  block.sync();

  // Compute BFGS sums: four dot products using four warps
  computeBfgsSums(cachedDGrads,
                  cachedXis,
                  cachedHessDGrads,
                  facShared,
                  faeShared,
                  sumDGradShared,
                  sumXiShared,
                  warp,
                  warpIdx,
                  laneIdx,
                  dim);

  block.sync();

  // Compute BFGS sums: compute the update flag
  computeUpdateFlag(idxWithinSystem,
                    facShared,
                    faeShared,
                    fadShared,
                    sumDGradShared,
                    sumXiShared,
                    needUpdateInverseHessian);

  block.sync();

  if (needUpdateInverseHessian[0]) {
    // Update dGrads, Inverse Hessian, and Xi
    const double fac = facShared[0];
    const double fae = faeShared[0];
    const double fad = fadShared[0];

    for (int i = idxWithinSystem; i < dim; i += blockSize) {
      cachedDGrads[i] = fac * cachedXis[i] - fad * cachedHessDGrads[i];
    }

    block.sync();

    for (int i = idxWithinSystem; i < dim; i += blockSize) {
      localDGrad[i]     = cachedDGrads[i];
      localHessDGrad[i] = cachedHessDGrads[i];
    }

    for (int row = warpIdx; row < dim; row += numWarp) {
      const double pxi        = fac * cachedXis[row];
      const double hdgi       = fad * cachedHessDGrads[row];
      const double dgi        = fae * cachedDGrads[row];
      double       dotProduct = 0.0;

      for (int col = laneIdx; col < dim; col += warpSize) {
        const double pxj       = cachedXis[col];
        const double hdgj      = cachedHessDGrads[col];
        const double dgj       = cachedDGrads[col];
        double       new_value = pxi * pxj - hdgi * hdgj + dgi * dgj;
        new_value += invHessianLocal[row * dim + col];
        dotProduct -= new_value * cachedGrads[col];
        invHessianLocal[row * dim + col] = new_value;
      }

      cg::reduce_store_async(warp, &localXi[row], dotProduct, cg::plus<double>{});
    }
  } else {
    // Update Xi Only
    if (idxWithinSystem < dim) {
      localHessDGrad[idxWithinSystem] = cachedHessDGrads[idxWithinSystem];
    }

    for (int row = warpIdx; row < dim; row += numWarp) {
      double dotProduct = 0.0;

      for (int col = laneIdx; col < dim; col += warpSize) {
        dotProduct -= invHessianLocal[row * dim + col] * cachedGrads[col];
      }

      cg::reduce_store_async(warp, &localXi[row], dotProduct, cg::plus<double>{});
    }
  }
}

// Global-memory variant that avoids fixed-size shared arrays; safe for large molecules
template <int dataDim>
__global__ void updateInverseHessianBFGSBatchKernelGlobal(const int16_t* statuses,
                                                          const int*     atomStarts,
                                                          const int*     hessianStarts,
                                                          double*        invHessians,
                                                          double*        dGrads,
                                                          double*        xis,
                                                          double*        hessDGrads,
                                                          const double*  grads,
                                                          const int*     activeSystemIndices) {
  __shared__ double facShared[1];
  __shared__ double faeShared[1];
  __shared__ double fadShared[1];
  __shared__ double sumDGradShared[1];
  __shared__ double sumXiShared[1];
  __shared__ bool   needUpdateInverseHessian[1];

  const int sysIdx = activeSystemIndices[blockIdx.x];
  if (statuses != nullptr && statuses[sysIdx] == 0) {
    return;
  }

  cg::thread_block                block           = cg::this_thread_block();
  cg::thread_block_tile<warpSize> warp            = cg::tiled_partition<warpSize>(block);
  const int                       idxWithinSystem = threadIdx.x;
  const int                       warpIdx         = idxWithinSystem / warpSize;
  const int                       laneIdx         = idxWithinSystem % warpSize;

  // Get local pointers. Note that inverse hessian is dim indexed but the atomStart-based ones are * dataDim
  const int           atomOffset      = atomStarts[sysIdx];
  const int           dim             = dataDim * (atomStarts[sysIdx + 1] - atomOffset);
  double* const       invHessianLocal = &invHessians[hessianStarts[sysIdx]];
  const int           absAtomOffset   = atomOffset * dataDim;
  double* const       localDGrad      = &dGrads[absAtomOffset];
  double* const       localHessDGrad  = &hessDGrads[absAtomOffset];
  double* const       localXi         = &xis[absAtomOffset];
  const double* const localGrad       = &grads[absAtomOffset];

  // Compute hessDGrads directly into global memory
  for (int row = warpIdx; row < dim; row += numWarp) {
    double dotProduct = 0.0;

    for (int col = laneIdx; col < dim; col += warpSize) {
      dotProduct += invHessianLocal[row * dim + col] * localDGrad[col];
    }

    cg::reduce_store_async(warp, &localHessDGrad[row], dotProduct, cg::plus<double>{});
  }

  block.sync();

  // Compute BFGS sums using global memory
  computeBfgsSums(localDGrad,
                  localXi,
                  localHessDGrad,
                  facShared,
                  faeShared,
                  sumDGradShared,
                  sumXiShared,
                  warp,
                  warpIdx,
                  laneIdx,
                  dim);

  block.sync();

  computeUpdateFlag(idxWithinSystem,
                    facShared,
                    faeShared,
                    fadShared,
                    sumDGradShared,
                    sumXiShared,
                    needUpdateInverseHessian);

  block.sync();

  if (needUpdateInverseHessian[0]) {
    const double fac = facShared[0];
    const double fae = faeShared[0];
    const double fad = fadShared[0];

    // Update dGrad with snapshot values; do not touch Xi yet.
    for (int i = idxWithinSystem; i < dim; i += blockSize) {
      const double dval = fac * localXi[i] - fad * localHessDGrad[i];
      localDGrad[i]     = dval;
    }

    block.sync();

    // Update inverse Hessian using OLD Xi snapshot (still in localXi)
    for (int row = warpIdx; row < dim; row += numWarp) {
      const double pxi  = fac * localXi[row];
      const double hdgi = fad * localHessDGrad[row];
      const double dgi  = fae * localDGrad[row];

      for (int col = laneIdx; col < dim; col += warpSize) {
        const double pxj       = localXi[col];
        const double hdgj      = localHessDGrad[col];
        const double dgj       = localDGrad[col];
        double       new_value = pxi * pxj - hdgi * hdgj + dgi * dgj;
        new_value += invHessianLocal[row * dim + col];
        invHessianLocal[row * dim + col] = new_value;
      }
    }

    block.sync();

    // Now compute Xi = -H_new * grad using the updated inverse Hessian
    for (int row = warpIdx; row < dim; row += numWarp) {
      double dotProduct = 0.0;
      for (int col = laneIdx; col < dim; col += warpSize) {
        dotProduct -= invHessianLocal[row * dim + col] * localGrad[col];
      }
      cg::reduce_store_async(warp, &localXi[row], dotProduct, cg::plus<double>{});
    }
  } else {
    // Xi update only
    for (int row = warpIdx; row < dim; row += numWarp) {
      double dotProduct = 0.0;

      for (int col = laneIdx; col < dim; col += warpSize) {
        dotProduct -= invHessianLocal[row * dim + col] * localGrad[col];
      }

      cg::reduce_store_async(warp, &localXi[row], dotProduct, cg::plus<double>{});
    }
  }
}

}  // namespace

void updateInverseHessianBFGSBatch(int            numActiveSystems,
                                   const int16_t* statuses,
                                   const int*     hessianStarts,
                                   const int*     atomStarts,
                                   double*        invHessians,
                                   double*        dGrads,
                                   double*        xis,
                                   double*        hessDGrads,
                                   const double*  grads,
                                   int            dataDim,
                                   bool           hasLargeMolecule,
                                   const int*     activeSystemIndices,
                                   cudaStream_t   stream) {
  // Row mapping parameters are computed and passed but not used yet
  // They will be used when we implement true row-based processing

  if (dataDim == 3) {
    if (hasLargeMolecule) {
      updateInverseHessianBFGSBatchKernelGlobal<3><<<numActiveSystems, blockSize, 0, stream>>>(statuses,
                                                                                               atomStarts,
                                                                                               hessianStarts,
                                                                                               invHessians,
                                                                                               dGrads,
                                                                                               xis,
                                                                                               hessDGrads,
                                                                                               grads,
                                                                                               activeSystemIndices);
    } else {
      updateInverseHessianBFGSBatchKernelShared<3><<<numActiveSystems, blockSize, 0, stream>>>(statuses,
                                                                                               atomStarts,
                                                                                               hessianStarts,
                                                                                               invHessians,
                                                                                               dGrads,
                                                                                               xis,
                                                                                               hessDGrads,
                                                                                               grads,
                                                                                               activeSystemIndices);
    }
  } else if (dataDim == 4) {
    if (hasLargeMolecule) {
      updateInverseHessianBFGSBatchKernelGlobal<4><<<numActiveSystems, blockSize, 0, stream>>>(statuses,
                                                                                               atomStarts,
                                                                                               hessianStarts,
                                                                                               invHessians,
                                                                                               dGrads,
                                                                                               xis,
                                                                                               hessDGrads,
                                                                                               grads,
                                                                                               activeSystemIndices);
    } else {
      updateInverseHessianBFGSBatchKernelShared<4><<<numActiveSystems, blockSize, 0, stream>>>(statuses,
                                                                                               atomStarts,
                                                                                               hessianStarts,
                                                                                               invHessians,
                                                                                               dGrads,
                                                                                               xis,
                                                                                               hessDGrads,
                                                                                               grads,
                                                                                               activeSystemIndices);
    }
  } else {
    throw std::runtime_error("Unsupported data dimension: " + std::to_string(dataDim));
  }

  cudaCheckError(cudaGetLastError());
}

}  // namespace nvMolKit
