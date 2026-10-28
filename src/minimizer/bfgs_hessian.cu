// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

template <typename reduceT, typename outputT>
__device__ __forceinline__ void warpReduceStore(cg::thread_block_tile<warpSize>& warp, outputT* output, reduceT value) {
  const reduceT sum = cg::reduce(warp, value, cg::plus<reduceT>{});
  if (warp.thread_rank() == 0)
    *output = static_cast<outputT>(sum);
}

template <typename real> __device__ __forceinline__ real hessianSqrt(real value) {
  if constexpr (cuda::std::is_same_v<real, float>)
    return sqrtf(value);
  else
    return sqrt(value);
}

template <typename real, typename reduceT, typename inputT>
__device__ __forceinline__ void computeBfgsSums(const inputT*                    dGrad,
                                                const inputT*                    xi,
                                                const inputT*                    hessDGrad,
                                                reduceT*                         facShared,
                                                reduceT*                         faeShared,
                                                reduceT*                         sumDGradShared,
                                                reduceT*                         sumXiShared,
                                                cg::thread_block_tile<warpSize>& warp,
                                                int                              warpIdx,
                                                int                              laneIdx,
                                                int                              dim) {
  reduceT sumTerm = 0;
  if (warpIdx == 0) {
    for (int i = laneIdx; i < dim; i += warpSize) {
      sumTerm += static_cast<reduceT>(static_cast<real>(dGrad[i]) * static_cast<real>(xi[i]));
    }
    warpReduceStore(warp, &facShared[0], sumTerm);
  } else if (warpIdx == 1) {
    for (int i = laneIdx; i < dim; i += warpSize) {
      sumTerm += static_cast<reduceT>(static_cast<real>(dGrad[i]) * static_cast<real>(hessDGrad[i]));
    }
    warpReduceStore(warp, &faeShared[0], sumTerm);
  } else if (warpIdx == 2) {
    for (int i = laneIdx; i < dim; i += warpSize) {
      const real value = static_cast<real>(dGrad[i]);
      sumTerm += static_cast<reduceT>(value * value);
    }
    warpReduceStore(warp, &sumDGradShared[0], sumTerm);
  } else if (warpIdx == 3) {
    for (int i = laneIdx; i < dim; i += warpSize) {
      const real value = static_cast<real>(xi[i]);
      sumTerm += static_cast<reduceT>(value * value);
    }
    warpReduceStore(warp, &sumXiShared[0], sumTerm);
  }
}

template <typename real, typename reduceT>
__device__ __forceinline__ void computeUpdateFlag(int      idxWithinSystem,
                                                  reduceT* facShared,
                                                  reduceT* faeShared,
                                                  reduceT* fadShared,
                                                  reduceT* sumDGradShared,
                                                  reduceT* sumXiShared,
                                                  bool*    needUpdateInverseHessian) {
  if (idxWithinSystem == 0) {
    constexpr reduceT EPS   = static_cast<reduceT>(3e-8);
    const reduceT     sumXi = sumXiShared[0], sumDGrad = sumDGradShared[0];
    const reduceT     fac = facShared[0], fae = faeShared[0];
    const reduceT     threshold            = hessianSqrt(EPS * sumDGrad * sumXi);
    bool              updateInverseHessian = fac > threshold;
    if (updateInverseHessian) {
      facShared[0] = reduceT{1} / fac;
      fadShared[0] = reduceT{1} / fae;
    }
    needUpdateInverseHessian[0] = updateInverseHessian;
  }
}

// Shared-memory optimized kernel used when all systems have <= maxAtom atoms
template <int dataDim, typename real, typename reduceT, typename storageT>
__global__ void updateInverseHessianBFGSBatchKernelShared(const int16_t*  statuses,
                                                          const int*      atomStarts,
                                                          const int*      hessianStarts,
                                                          storageT*       invHessians,
                                                          storageT*       dGrads,
                                                          storageT*       xis,
                                                          storageT*       hessDGrads,
                                                          const storageT* grads,
                                                          const int*      activeSystemIndices) {
  __shared__ reduceT facShared[1];
  __shared__ reduceT faeShared[1];
  __shared__ reduceT fadShared[1];
  __shared__ reduceT sumDGradShared[1];
  __shared__ reduceT sumXiShared[1];
  __shared__ bool    needUpdateInverseHessian[1];

  __shared__ real cachedDGrads[dataDim * maxAtom];
  __shared__ real cachedHessDGrads[dataDim * maxAtom];
  __shared__ real cachedXis[dataDim * maxAtom];
  __shared__ real cachedGrads[dataDim * maxAtom];

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
  const int             atomOffset      = atomStarts[sysIdx];
  const int             dim             = dataDim * (atomStarts[sysIdx + 1] - atomOffset);
  storageT* const       invHessianLocal = &invHessians[hessianStarts[sysIdx]];
  const int             absAtomOffset   = atomOffset * dataDim;
  storageT* const       localDGrad      = &dGrads[absAtomOffset];
  storageT* const       localHessDGrad  = &hessDGrads[absAtomOffset];
  storageT* const       localXi         = &xis[absAtomOffset];
  const storageT* const localGrad       = &grads[absAtomOffset];

  // Load dGrads, Xi, grads into shared memory
  for (int i = idxWithinSystem; i < dim; i += blockSize) {
    cachedDGrads[i] = static_cast<real>(localDGrad[i]);
    cachedXis[i]    = static_cast<real>(localXi[i]);
    cachedGrads[i]  = static_cast<real>(localGrad[i]);
  }

  block.sync();

  // Update hessDGrads
  // Update hessDGrads: Each warp processes different rows
  for (int row = warpIdx; row < dim; row += numWarp) {
    reduceT dotProduct = 0;

    // Update hessDGrads: Each thread in warp processes different columns
    for (int col = laneIdx; col < dim; col += warpSize) {
      dotProduct += static_cast<reduceT>(static_cast<real>(invHessianLocal[row * dim + col]) * cachedDGrads[col]);
    }

    warpReduceStore(warp, &cachedHessDGrads[row], dotProduct);
  }

  block.sync();

  // Compute BFGS sums: four dot products using four warps
  computeBfgsSums<real, reduceT>(cachedDGrads,
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
  computeUpdateFlag<real, reduceT>(idxWithinSystem,
                                   facShared,
                                   faeShared,
                                   fadShared,
                                   sumDGradShared,
                                   sumXiShared,
                                   needUpdateInverseHessian);

  block.sync();

  if (needUpdateInverseHessian[0]) {
    // Update dGrads, Inverse Hessian, and Xi
    const real fac = static_cast<real>(facShared[0]);
    const real fae = static_cast<real>(faeShared[0]);
    const real fad = static_cast<real>(fadShared[0]);

    for (int i = idxWithinSystem; i < dim; i += blockSize) {
      cachedDGrads[i] = fac * cachedXis[i] - fad * cachedHessDGrads[i];
    }

    block.sync();

    for (int i = idxWithinSystem; i < dim; i += blockSize) {
      localDGrad[i]     = cachedDGrads[i];
      localHessDGrad[i] = cachedHessDGrads[i];
    }

    for (int row = warpIdx; row < dim; row += numWarp) {
      const real pxi        = fac * cachedXis[row];
      const real hdgi       = fad * cachedHessDGrads[row];
      const real dgi        = fae * cachedDGrads[row];
      reduceT    dotProduct = 0;

      for (int col = laneIdx; col < dim; col += warpSize) {
        const real pxj = cachedXis[col], hdgj = cachedHessDGrads[col], dgj = cachedDGrads[col];
        real       new_value = pxi * pxj - hdgi * hdgj + dgi * dgj;
        new_value += static_cast<real>(invHessianLocal[row * dim + col]);
        dotProduct -= new_value * cachedGrads[col];
        invHessianLocal[row * dim + col] = static_cast<storageT>(new_value);
      }

      warpReduceStore(warp, &localXi[row], dotProduct);
    }
  } else {
    // Update Xi Only
    if (idxWithinSystem < dim) {
      localHessDGrad[idxWithinSystem] = cachedHessDGrads[idxWithinSystem];
    }

    for (int row = warpIdx; row < dim; row += numWarp) {
      reduceT dotProduct = 0;

      for (int col = laneIdx; col < dim; col += warpSize) {
        dotProduct -= static_cast<reduceT>(static_cast<real>(invHessianLocal[row * dim + col]) * cachedGrads[col]);
      }

      warpReduceStore(warp, &localXi[row], dotProduct);
    }
  }
}

// Global-memory variant that avoids fixed-size shared arrays; safe for large molecules
template <int dataDim, typename real, typename reduceT, typename storageT>
__global__ void updateInverseHessianBFGSBatchKernelGlobal(const int16_t*  statuses,
                                                          const int*      atomStarts,
                                                          const int*      hessianStarts,
                                                          storageT*       invHessians,
                                                          storageT*       dGrads,
                                                          storageT*       xis,
                                                          storageT*       hessDGrads,
                                                          const storageT* grads,
                                                          const int*      activeSystemIndices) {
  __shared__ reduceT facShared[1];
  __shared__ reduceT faeShared[1];
  __shared__ reduceT fadShared[1];
  __shared__ reduceT sumDGradShared[1];
  __shared__ reduceT sumXiShared[1];
  __shared__ bool    needUpdateInverseHessian[1];

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
  const int             atomOffset      = atomStarts[sysIdx];
  const int             dim             = dataDim * (atomStarts[sysIdx + 1] - atomOffset);
  storageT* const       invHessianLocal = &invHessians[hessianStarts[sysIdx]];
  const int             absAtomOffset   = atomOffset * dataDim;
  storageT* const       localDGrad      = &dGrads[absAtomOffset];
  storageT* const       localHessDGrad  = &hessDGrads[absAtomOffset];
  storageT* const       localXi         = &xis[absAtomOffset];
  const storageT* const localGrad       = &grads[absAtomOffset];

  // Compute hessDGrads directly into global memory
  for (int row = warpIdx; row < dim; row += numWarp) {
    reduceT dotProduct = 0;

    for (int col = laneIdx; col < dim; col += warpSize) {
      dotProduct +=
        static_cast<reduceT>(static_cast<real>(invHessianLocal[row * dim + col]) * static_cast<real>(localDGrad[col]));
    }

    warpReduceStore(warp, &localHessDGrad[row], dotProduct);
  }

  block.sync();

  // Compute BFGS sums using global memory
  computeBfgsSums<real, reduceT>(localDGrad,
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

  computeUpdateFlag<real, reduceT>(idxWithinSystem,
                                   facShared,
                                   faeShared,
                                   fadShared,
                                   sumDGradShared,
                                   sumXiShared,
                                   needUpdateInverseHessian);

  block.sync();

  if (needUpdateInverseHessian[0]) {
    const real fac = static_cast<real>(facShared[0]);
    const real fae = static_cast<real>(faeShared[0]);
    const real fad = static_cast<real>(fadShared[0]);

    // Update dGrad with snapshot values; do not touch Xi yet.
    for (int i = idxWithinSystem; i < dim; i += blockSize) {
      const real dval = fac * static_cast<real>(localXi[i]) - fad * static_cast<real>(localHessDGrad[i]);
      localDGrad[i]   = dval;
    }

    block.sync();

    // Update inverse Hessian using OLD Xi snapshot (still in localXi)
    for (int row = warpIdx; row < dim; row += numWarp) {
      const real pxi  = fac * static_cast<real>(localXi[row]);
      const real hdgi = fad * static_cast<real>(localHessDGrad[row]);
      const real dgi  = fae * static_cast<real>(localDGrad[row]);

      for (int col = laneIdx; col < dim; col += warpSize) {
        const real pxj       = static_cast<real>(localXi[col]);
        const real hdgj      = static_cast<real>(localHessDGrad[col]);
        const real dgj       = static_cast<real>(localDGrad[col]);
        real       new_value = pxi * pxj - hdgi * hdgj + dgi * dgj;
        new_value += static_cast<real>(invHessianLocal[row * dim + col]);
        invHessianLocal[row * dim + col] = static_cast<storageT>(new_value);
      }
    }

    block.sync();

    // Now compute Xi = -H_new * grad using the updated inverse Hessian
    for (int row = warpIdx; row < dim; row += numWarp) {
      reduceT dotProduct = 0;
      for (int col = laneIdx; col < dim; col += warpSize) {
        dotProduct -=
          static_cast<reduceT>(static_cast<real>(invHessianLocal[row * dim + col]) * static_cast<real>(localGrad[col]));
      }
      warpReduceStore(warp, &localXi[row], dotProduct);
    }
  } else {
    // Xi update only
    for (int row = warpIdx; row < dim; row += numWarp) {
      reduceT dotProduct = 0;

      for (int col = laneIdx; col < dim; col += warpSize) {
        dotProduct -=
          static_cast<reduceT>(static_cast<real>(invHessianLocal[row * dim + col]) * static_cast<real>(localGrad[col]));
      }

      warpReduceStore(warp, &localXi[row], dotProduct);
    }
  }
}

}  // namespace

template <typename real, typename reduceT, typename storageT>
void updateInverseHessianBFGSBatchImpl(int             numActiveSystems,
                                       const int16_t*  statuses,
                                       const int*      hessianStarts,
                                       const int*      atomStarts,
                                       storageT*       invHessians,
                                       storageT*       dGrads,
                                       storageT*       xis,
                                       storageT*       hessDGrads,
                                       const storageT* grads,
                                       int             dataDim,
                                       bool            hasLargeMolecule,
                                       const int*      activeSystemIndices,
                                       cudaStream_t    stream) {
  // Row mapping parameters are computed and passed but not used yet
  // They will be used when we implement true row-based processing

  if (dataDim == 3) {
    if (hasLargeMolecule) {
      updateInverseHessianBFGSBatchKernelGlobal<3, real, reduceT, storageT>
        <<<numActiveSystems, blockSize, 0, stream>>>(statuses,
                                                     atomStarts,
                                                     hessianStarts,
                                                     invHessians,
                                                     dGrads,
                                                     xis,
                                                     hessDGrads,
                                                     grads,
                                                     activeSystemIndices);
    } else {
      updateInverseHessianBFGSBatchKernelShared<3, real, reduceT, storageT>
        <<<numActiveSystems, blockSize, 0, stream>>>(statuses,
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
      updateInverseHessianBFGSBatchKernelGlobal<4, real, reduceT, storageT>
        <<<numActiveSystems, blockSize, 0, stream>>>(statuses,
                                                     atomStarts,
                                                     hessianStarts,
                                                     invHessians,
                                                     dGrads,
                                                     xis,
                                                     hessDGrads,
                                                     grads,
                                                     activeSystemIndices);
    } else {
      updateInverseHessianBFGSBatchKernelShared<4, real, reduceT, storageT>
        <<<numActiveSystems, blockSize, 0, stream>>>(statuses,
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

#define NVMOLKIT_DEFINE_BFGS_HESSIAN_OVERLOAD(real, reduceT, storageT)    \
  void updateInverseHessianBFGSBatch(int             numActiveSystems,    \
                                     const int16_t*  statuses,            \
                                     const int*      hessianStarts,       \
                                     const int*      atomStarts,          \
                                     storageT*       invHessians,         \
                                     storageT*       dGrads,              \
                                     storageT*       xis,                 \
                                     storageT*       hessDGrads,          \
                                     const storageT* grads,               \
                                     int             dataDim,             \
                                     bool            hasLargeMolecule,    \
                                     const int*      activeSystemIndices, \
                                     cudaStream_t    stream) {               \
    updateInverseHessianBFGSBatchImpl<real, reduceT>(numActiveSystems,    \
                                                     statuses,            \
                                                     hessianStarts,       \
                                                     atomStarts,          \
                                                     invHessians,         \
                                                     dGrads,              \
                                                     xis,                 \
                                                     hessDGrads,          \
                                                     grads,               \
                                                     dataDim,             \
                                                     hasLargeMolecule,    \
                                                     activeSystemIndices, \
                                                     stream);             \
  }

NVMOLKIT_DEFINE_BFGS_HESSIAN_OVERLOAD(double, double, double)
NVMOLKIT_DEFINE_BFGS_HESSIAN_OVERLOAD(float, float, float)
#undef NVMOLKIT_DEFINE_BFGS_HESSIAN_OVERLOAD

}  // namespace nvMolKit
