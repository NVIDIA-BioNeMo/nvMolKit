// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "src/tfd/tfd_detail.h"
#include "src/tfd/tfd_kernels.h"
#include "src/utils/cuda_error_check.h"

namespace nvMolKit {

using detail::circularDifference;
using detail::computeDihedralAngle;

namespace {

//! Binary search: find m such that starts[m] <= idx < starts[m+1]
__device__ __forceinline__ int findMolecule(const int* starts, int numMolecules, int idx) {
  int lo = 0, hi = numMolecules;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (starts[mid + 1] <= idx) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

//! Kernel to compute dihedral angles for all conformers.
//! One thread per (conformer, quartet) work item.
//! Uses binary search on dihedralWorkStarts to find the molecule,
//! then computes (confIdx, quartetIdx, outIdx) arithmetically.
__global__ void dihedralKernel(const int totalWorkItems,
                               const float* __restrict__ positions,
                               const int* __restrict__ confPositionStarts,
                               const int* __restrict__ torsionAtoms,
                               const MolDescriptor* __restrict__ molDescriptors,
                               const int* __restrict__ dihedralWorkStarts,
                               const int numMolecules,
                               float* __restrict__ dihedralAngles) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= totalWorkItems) {
    return;
  }

  int m = findMolecule(dihedralWorkStarts, numMolecules, idx);

  const MolDescriptor& desc         = molDescriptors[m];
  int                  localIdx     = idx - dihedralWorkStarts[m];
  int                  localConfIdx = localIdx / desc.numQuartets;
  int                  quartetIdx   = localIdx % desc.numQuartets;

  int confIdx = desc.confStart + localConfIdx;
  int torsIdx = desc.quartetStart + quartetIdx;
  int outIdx  = desc.dihedStart + localIdx;

  int atomA = torsionAtoms[torsIdx * 4 + 0];
  int atomB = torsionAtoms[torsIdx * 4 + 1];
  int atomC = torsionAtoms[torsIdx * 4 + 2];
  int atomD = torsionAtoms[torsIdx * 4 + 3];

  const float* posBase = positions + confPositionStarts[confIdx];

  dihedralAngles[outIdx] =
    computeDihedralAngle(posBase + atomA * 3, posBase + atomB * 3, posBase + atomC * 3, posBase + atomD * 3);
}

//! Kernel to compute TFD matrix values.
//! One block per molecule. Threads within a block cooperatively process
//! all C*(C-1)/2 conformer pairs via grid-stride loop.
//! All threads in a block share the same torsion types — no cross-molecule divergence.
__global__ void tfdMatrixKernel(const int numMolecules,
                                const float* __restrict__ dihedralAngles,
                                const float* __restrict__ torsionWeights,
                                const float* __restrict__ torsionMaxDevs,
                                const int* __restrict__ quartetStarts,
                                const uint8_t* __restrict__ torsionTypes,
                                const MolDescriptor* __restrict__ molDescriptors,
                                float* __restrict__ tfdOutput) {
  int m = blockIdx.x;
  if (m >= numMolecules) {
    return;
  }

  const MolDescriptor& desc     = molDescriptors[m];
  int                  numConf  = desc.numConformers;
  int                  numPairs = numConf * (numConf - 1) / 2;
  int                  numTors  = desc.numTorsions;
  int                  qBase    = quartetStarts[desc.torsStart];

  if (numPairs == 0 || numTors == 0) {
    return;
  }

  // Each thread processes conformer pairs via grid-stride within this block
  for (int pairIdx = threadIdx.x; pairIdx < numPairs; pairIdx += blockDim.x) {
    // Decode lower-triangular (i, j) from flat pairIdx
    // i*(i-1)/2 + j = pairIdx, i > j >= 0
    // i = floor((1 + sqrt(1 + 8*pairIdx)) / 2)
    // Use double precision: float32 loses bits when 8*pairIdx > 2^23 (~1M pairs,
    // ~1415 conformers), the same threshold documented in conformer_rmsd.cu.
    int i = static_cast<int>((1.0 + sqrt(1.0 + 8.0 * static_cast<double>(pairIdx))) * 0.5);
    // Clamp guards for any residual rounding at boundary values.
    if (i * (i - 1) / 2 > pairIdx)
      i--;
    else if ((i + 1) * i / 2 <= pairIdx)
      i++;
    int j = pairIdx - i * (i - 1) / 2;

    int aI = desc.dihedStart + i * desc.numQuartets;
    int aJ = desc.dihedStart + j * desc.numQuartets;

    float sumWeightedDev = 0.0f;
    float sumWeights     = 0.0f;

    for (int t = 0; t < numTors; ++t) {
      int     globalT     = desc.torsStart + t;
      int     qLocalStart = quartetStarts[globalT] - qBase;
      int     numQ        = quartetStarts[globalT + 1] - quartetStarts[globalT];
      uint8_t type        = torsionTypes[globalT];

      float deviation;
      if (type == 0) {  // Single
        deviation = circularDifference(dihedralAngles[aI + qLocalStart], dihedralAngles[aJ + qLocalStart]) /
                    torsionMaxDevs[globalT];
      } else if (type == 1) {  // Ring
        if (numQ == 0) {
          deviation = 0.0f;
        } else {
          float avgI = 0.0f;
          float avgJ = 0.0f;
          for (int qq = 0; qq < numQ; ++qq) {
            avgI += fabsf(dihedralAngles[aI + qLocalStart + qq] - 180.0f);
            avgJ += fabsf(dihedralAngles[aJ + qLocalStart + qq] - 180.0f);
          }
          avgI /= numQ;
          avgJ /= numQ;
          deviation = fabsf(avgI - avgJ) / torsionMaxDevs[globalT];
        }
      } else {  // Symmetric
        float minDiff = 180.0f;
        for (int qi = 0; qi < numQ; ++qi) {
          for (int qj = 0; qj < numQ; ++qj) {
            float diff =
              circularDifference(dihedralAngles[aI + qLocalStart + qi], dihedralAngles[aJ + qLocalStart + qj]);
            minDiff = fminf(minDiff, diff);
          }
        }
        deviation = minDiff / torsionMaxDevs[globalT];
      }

      float weight = torsionWeights[globalT];
      sumWeightedDev += deviation * weight;
      sumWeights += weight;
    }

    tfdOutput[desc.tfdOutStart + pairIdx] = (sumWeights > 1e-10f) ? (sumWeightedDev / sumWeights) : 0.0f;
  }
}

}  // namespace

void launchDihedralKernel(int                  totalWorkItems,
                          const float*         positions,
                          const int*           confPositionStarts,
                          const int*           torsionAtoms,
                          const MolDescriptor* molDescriptors,
                          const int*           dihedralWorkStarts,
                          int                  numMolecules,
                          float*               dihedralAngles,
                          cudaStream_t         stream) {
  if (totalWorkItems == 0) {
    return;
  }

  int gridSize = (totalWorkItems + kTFDBlockSize - 1) / kTFDBlockSize;

  dihedralKernel<<<gridSize, kTFDBlockSize, 0, stream>>>(totalWorkItems,
                                                         positions,
                                                         confPositionStarts,
                                                         torsionAtoms,
                                                         molDescriptors,
                                                         dihedralWorkStarts,
                                                         numMolecules,
                                                         dihedralAngles);

  cudaCheckError(cudaGetLastError());
}

void launchTFDMatrixKernel(int                  numMolecules,
                           const float*         dihedralAngles,
                           const float*         torsionWeights,
                           const float*         torsionMaxDevs,
                           const int*           quartetStarts,
                           const uint8_t*       torsionTypes,
                           const MolDescriptor* molDescriptors,
                           float*               tfdOutput,
                           cudaStream_t         stream) {
  if (numMolecules == 0) {
    return;
  }

  // One block per molecule; threads within block process pairs via grid-stride
  tfdMatrixKernel<<<numMolecules, kTFDBlockSize, 0, stream>>>(numMolecules,
                                                              dihedralAngles,
                                                              torsionWeights,
                                                              torsionMaxDevs,
                                                              quartetStarts,
                                                              torsionTypes,
                                                              molDescriptors,
                                                              tfdOutput);

  cudaCheckError(cudaGetLastError());
}

}  // namespace nvMolKit
