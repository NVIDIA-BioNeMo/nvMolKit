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

#ifndef NVMOLKIT_TFD_CPU_H
#define NVMOLKIT_TFD_CPU_H

#include <vector>

#include "src/tfd/tfd_common.h"

namespace nvMolKit {

//! CPU implementation of TFD computation
class TFDCpuGenerator {
 public:
  TFDCpuGenerator() = default;

  //! Compute TFD matrix for a single molecule
  //! @param mol Molecule with conformers
  //! @param options Computation options
  //! @return Lower triangular TFD matrix as flat vector [C*(C-1)/2 values]
  std::vector<double> GetTFDMatrix(const RDKit::ROMol& mol, const TFDComputeOptions& options = TFDComputeOptions{});

  //! Compute TFD matrices for multiple molecules
  //! @param mols Vector of molecules
  //! @param options Computation options
  //! @return Vector of TFD matrices, one per molecule
  std::vector<std::vector<double>> GetTFDMatrices(const std::vector<const RDKit::ROMol*>& mols,
                                                  const TFDComputeOptions& options = TFDComputeOptions{});

  //! Compute dihedral angles for all conformers of a molecule
  //! @param mol Molecule with conformers
  //! @param torsionList Previously extracted torsion list
  //! @return Angles array [numConformers][totalQuartets], flattened
  std::vector<float> computeDihedralAngles(const RDKit::ROMol& mol, const TorsionList& torsionList);

 private:
  //! Torsion list and weights for a single molecule
  struct TorsionData {
    TorsionList        torsionList;
    std::vector<float> weights;
  };

  //! Extract torsion list and weights from a molecule
  static TorsionData extractTorsionData(const RDKit::ROMol& mol, const TFDComputeOptions& options);

  //! Compute TFD matrix from precomputed angles
  std::vector<double> computeTFDMatrixFromAngles(const std::vector<float>& angles,
                                                 int                       numConformers,
                                                 int                       totalQuartets,
                                                 const TorsionList&        torsionList,
                                                 const std::vector<float>& weights);

  //! Compute TFD between two conformers
  static double computeTFDPair(const float*              anglesI,
                               const float*              anglesJ,
                               const TorsionList&        torsionList,
                               const std::vector<float>& weights);
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_TFD_CPU_H
