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

#include "src/tfd/tfd_cpu.h"

#include <stdexcept>

#include "src/tfd/tfd_detail.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "src/utils/nvtx.h"

namespace nvMolKit {

double TFDCpuGenerator::computeTFDPair(const float*              anglesI,
                                       const float*              anglesJ,
                                       const TorsionList&        torsionList,
                                       const std::vector<float>& weights) {
  double sumWeightedDev = 0.0;
  double sumWeights     = 0.0;
  int    qOffset        = 0;
  int    torsIdx        = 0;

  auto processTorsion = [&](const TorsionDef& torsion, bool isRing) {
    int numQ = static_cast<int>(torsion.atomQuartets.size());
    if (numQ == 0) {
      torsIdx++;
      return;
    }

    double deviation;
    if (numQ == 1) {
      float diff = detail::circularDifference(anglesI[qOffset], anglesJ[qOffset]);
      deviation  = diff / torsion.maxDev;
    } else if (isRing) {
      // Average abs(signed dihedral) for each conformer, then compare averages.
      // Our angles are in [0,360) with a 180 deg offset from RDKit's signed convention,
      // so abs(signed) = abs(angle - 180).
      double avgI = 0.0, avgJ = 0.0;
      for (int q = 0; q < numQ; ++q) {
        avgI += std::abs(static_cast<double>(anglesI[qOffset + q]) - 180.0);
        avgJ += std::abs(static_cast<double>(anglesJ[qOffset + q]) - 180.0);
      }
      avgI /= numQ;
      avgJ /= numQ;
      deviation = std::abs(avgI - avgJ) / torsion.maxDev;
    } else {
      // Symmetric: minimum circular difference across all (qi, qj) pairs
      double minDiff = 180.0;
      for (int qi = 0; qi < numQ; ++qi) {
        for (int qj = 0; qj < numQ; ++qj) {
          float diff = detail::circularDifference(anglesI[qOffset + qi], anglesJ[qOffset + qj]);
          minDiff    = std::min(minDiff, static_cast<double>(diff));
        }
      }
      deviation = minDiff / torsion.maxDev;
    }

    float w = (torsIdx < static_cast<int>(weights.size())) ? weights[torsIdx] : 1.0f;
    sumWeightedDev += deviation * w;
    sumWeights += w;
    qOffset += numQ;
    torsIdx++;
  };

  for (const auto& t : torsionList.nonRingTorsions)
    processTorsion(t, false);
  for (const auto& t : torsionList.ringTorsions)
    processTorsion(t, true);

  return (sumWeights > 1e-10) ? (sumWeightedDev / sumWeights) : 0.0;
}

std::vector<float> TFDCpuGenerator::computeDihedralAngles(const RDKit::ROMol& mol, const TorsionList& torsionList) {
  int numConformers = mol.getNumConformers();
  int totalQuartets = 0;
  for (const auto& t : torsionList.nonRingTorsions)
    totalQuartets += static_cast<int>(t.atomQuartets.size());
  for (const auto& t : torsionList.ringTorsions)
    totalQuartets += static_cast<int>(t.atomQuartets.size());

  if (totalQuartets == 0 || numConformers == 0) {
    return {};
  }

  std::vector<float> angles(numConformers * totalQuartets);

  // Collect all quartets into a flat list for indexed access
  std::vector<const std::array<int, 4>*> allQuartets;
  allQuartets.reserve(totalQuartets);
  for (const auto& t : torsionList.nonRingTorsions)
    for (const auto& q : t.atomQuartets)
      allQuartets.push_back(&q);
  for (const auto& t : torsionList.ringTorsions)
    for (const auto& q : t.atomQuartets)
      allQuartets.push_back(&q);

  int numAtoms = mol.getNumAtoms();
  int confIdx  = 0;
  for (auto confIt = mol.beginConformers(); confIt != mol.endConformers(); ++confIt, ++confIdx) {
    const auto&        conf = **confIt;
    std::vector<float> pos(numAtoms * 3);
    for (int a = 0; a < numAtoms; ++a) {
      const auto& p  = conf.getAtomPos(a);
      pos[a * 3]     = static_cast<float>(p.x);
      pos[a * 3 + 1] = static_cast<float>(p.y);
      pos[a * 3 + 2] = static_cast<float>(p.z);
    }

    for (int qIdx = 0; qIdx < totalQuartets; ++qIdx) {
      const auto& quartet                    = *allQuartets[qIdx];
      angles[confIdx * totalQuartets + qIdx] = detail::computeDihedralAngle(&pos[quartet[0] * 3],
                                                                            &pos[quartet[1] * 3],
                                                                            &pos[quartet[2] * 3],
                                                                            &pos[quartet[3] * 3]);
    }
  }

  return angles;
}

std::vector<double> TFDCpuGenerator::computeTFDMatrixFromAngles(const std::vector<float>& angles,
                                                                int                       numConformers,
                                                                int                       totalQuartets,
                                                                const TorsionList&        torsionList,
                                                                const std::vector<float>& weights) {
  int                 numPairs = numConformers * (numConformers - 1) / 2;
  std::vector<double> tfdMatrix(numPairs);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int i = 1; i < numConformers; ++i) {
    for (int j = 0; j < i; ++j) {
      int    pairIdx = i * (i - 1) / 2 + j;
      double tfd =
        computeTFDPair(angles.data() + i * totalQuartets, angles.data() + j * totalQuartets, torsionList, weights);
      tfdMatrix[pairIdx] = tfd;
    }
  }

  return tfdMatrix;
}

TFDCpuGenerator::TorsionData TFDCpuGenerator::extractTorsionData(const RDKit::ROMol&      mol,
                                                                 const TFDComputeOptions& options) {
  TorsionData data;
  data.torsionList = extractTorsionList(mol, options.maxDevMode, options.symmRadius, options.ignoreColinearBonds);
  if (options.useWeights) {
    data.weights = computeTorsionWeights(mol, data.torsionList, options.ignoreColinearBonds);
  }
  return data;
}

std::vector<double> TFDCpuGenerator::GetTFDMatrix(const RDKit::ROMol& mol, const TFDComputeOptions& options) {
  int numConformers = mol.getNumConformers();
  if (numConformers < 2) {
    return {};
  }

  TorsionData td;
  {
    ScopedNvtxRange range("CPU: extractTorsions", NvtxColor::kGreen);
    td = extractTorsionData(mol, options);
  }

  if (td.torsionList.totalCount() == 0) {
    int numPairs = numConformers * (numConformers - 1) / 2;
    return std::vector<double>(numPairs, 0.0);
  }

  std::vector<float> angles;
  int                totalQuartets;
  {
    ScopedNvtxRange range("CPU: computeDihedralAngles", NvtxColor::kGreen);
    angles        = computeDihedralAngles(mol, td.torsionList);
    totalQuartets = static_cast<int>(angles.size()) / numConformers;
  }

  ScopedNvtxRange range("CPU: computeTFDMatrix", NvtxColor::kGreen);
  return computeTFDMatrixFromAngles(angles, numConformers, totalQuartets, td.torsionList, td.weights);
}

std::vector<std::vector<double>> TFDCpuGenerator::GetTFDMatrices(const std::vector<const RDKit::ROMol*>& mols,
                                                                 const TFDComputeOptions&                options) {
  ScopedNvtxRange range("CPU: GetTFDMatrices (" + std::to_string(mols.size()) + " mols)", NvtxColor::kGreen);
  std::vector<std::vector<double>> results(mols.size());

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (size_t i = 0; i < mols.size(); ++i) {
    results[i] = GetTFDMatrix(*mols[i], options);
  }

  return results;
}

}  // namespace nvMolKit
