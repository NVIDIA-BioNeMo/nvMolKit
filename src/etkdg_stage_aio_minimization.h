// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_ETKDG_STAGE_AIO_MINIMIZATION_H
#define NVMOLKIT_ETKDG_STAGE_AIO_MINIMIZATION_H

#include <GraphMol/DistGeomHelpers/Embedder.h>

#include <unordered_map>

#include "src/etkdg_impl.h"
#include "src/forcefields/dist_geom.h"
#include "src/minimizer/bfgs_minimize.h"
#include "versions.h"

namespace nvMolKit::detail {

#if RDKIT_AIO_ETKDG_API
//! RDKit's all-in-one ETKDG minimization: the distance-geometry terms first, then the experimental torsion and
//! planarity terms on the same force field, followed by the all-in-one energy check.
template <typename Real> class ETKDGAllInOneMinimizationStageT final : public ETKDGStage {
 public:
  ETKDGAllInOneMinimizationStageT(
    const std::vector<const RDKit::ROMol*>&                                       mols,
    const std::vector<EmbedArgs>&                                                 eargs,
    const RDKit::DGeomHelpers::EmbedParameters&                                   embedParams,
    const ETKDGContext&                                                           ctx,
    BfgsBatchMinimizerT<Real>&                                                    minimizer,
    cudaStream_t                                                                  stream = nullptr,
    std::unordered_map<const RDKit::ROMol*, DistGeom::AllInOneForceContribsHost>* cache  = nullptr);

  void        execute(ETKDGContext& ctx) override;
  std::string name() const override { return "All-in-one ETKDG Minimization"; }

 private:
  DistGeom::BatchedAllInOneSystemHost systemHost_;
  DistGeom::AllInOneForceWeights      weights_{};
  BatchedForcefieldMetadata           metadata_;
  BfgsBatchMinimizerT<Real>&          minimizer_;
  AsyncDeviceVector<double>           gradients_;
  AsyncDeviceVector<double>           energies_;
  double                              optimizerForceTol_;
  //! RDKit runs the first phase for each molecule's heavy atom count.
  std::vector<int>                    firstPhaseIterationsHost_;
  AsyncDeviceVector<int>              firstPhaseIterations_;
  cudaStream_t                        stream_ = nullptr;
};

using ETKDGAllInOneMinimizationStage = ETKDGAllInOneMinimizationStageT<double>;
extern template class ETKDGAllInOneMinimizationStageT<double>;
extern template class ETKDGAllInOneMinimizationStageT<float>;
#endif

//! RDKit's all-in-one K-term check: each linear angle and improper term of DistGeom::construct3DImproperForceField
//! must stay under its own limit, and their sum under 0.7 per linear or improper center.
class ETKDGKTermCheckStage final : public ETKDGStage {
 public:
  ETKDGKTermCheckStage(const std::vector<EmbedArgs>& eargs, const ETKDGContext& ctx, cudaStream_t stream = nullptr);
  void        execute(ETKDGContext& ctx) override;
  std::string name() const override { return "K-term check"; }

 private:
  DistGeom::BatchedMolecular3DDeviceBuffers systemDevice_;
  AsyncDeviceVector<int>                    numCenters_;
  AsyncDeviceVector<double>                 systemEnergies_;
  cudaStream_t                              stream_ = nullptr;
};

}  // namespace nvMolKit::detail

#endif  // NVMOLKIT_ETKDG_STAGE_AIO_MINIMIZATION_H
