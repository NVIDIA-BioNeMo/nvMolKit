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

#ifndef NVMOLKIT_ETKDG_STAGE_DISTGEOM_MINIMIZE_H
#define NVMOLKIT_ETKDG_STAGE_DISTGEOM_MINIMIZE_H

#include <GraphMol/DistGeomHelpers/Embedder.h>

#include <unordered_map>

#include "src/etkdg_impl.h"
#include "src/forcefields/dist_geom.h"
#include "src/minimizer/bfgs_minimize.h"

using ::nvMolKit::detail::EmbedArgs;
using ::nvMolKit::detail::ETKDGContext;
using ::nvMolKit::detail::ETKDGStage;

namespace nvMolKit {
namespace detail {

constexpr double MAX_MINIMIZED_E_PER_ATOM = 0.05;  // Maximum energy per atom threshold

/// Generic distance geometry minimization stage with configurable weights and parameters
/// \tparam real Working precision of the BFGS minimizer driving this stage.
template <typename real> class DistGeomMinimizeStageT : public ETKDGStage {
 public:
  /**
   * @brief Construct a distance geometry minimization stage
   *
   * @param mols Vector of molecules to minimize
   * @param eargs Vector of embed arguments
   * @param embedParam Embedding parameters
   * @param ctx ETKDG context
   * @param minimizer BFGS minimizer
   * @param chiralWeight Weight for Chiral violation term
   * @param fourthDimWeight Weight for 4th dim minimization term
   * @param maxIters Maximum number of iterations per minimization cycle
   * @param checkEnergy Whether to check energy per atom after minimization
   * @param stageName Name of the stage for logging
   * @param stream CUDA stream
   * @param cache Optional cache for force field parameters
   */
  DistGeomMinimizeStageT(
    const std::vector<const RDKit::ROMol*>&                                               mols,
    const std::vector<EmbedArgs>&                                                         eargs,
    const RDKit::DGeomHelpers::EmbedParameters&                                           embedParam,
    ETKDGContext&                                                                         ctx,
    BfgsBatchMinimizerT<real>&                                                            minimizer,
    double                                                                                chiralWeight,
    double                                                                                fourthDimWeight,
    int                                                                                   maxIters,
    bool                                                                                  checkEnergy,
    const std::string&                                                                    stageName,
    cudaStream_t                                                                          stream = nullptr,
    std::unordered_map<const RDKit::ROMol*, nvMolKit::DistGeom::EnergyForceContribsHost>* cache  = nullptr);

  void executeImpl(ETKDGContext& ctx, double chiralWeight, double fourthDimWeight, int maxIters, bool checkEnergy);

  std::string name() const override { return stageName_; }

  void execute(ETKDGContext& ctx) override {
    executeImpl(ctx, chiralWeight_, fourthDimWeight_, maxIters_, checkEnergy_);
  }

  nvMolKit::DistGeom::BatchedMolecularSystemHost           molSystemHost;
  nvMolKit::DistGeom::BatchedMolecularDeviceBuffersT<real> molSystemDevice;
  BatchedForcefieldMetadata                                metadata_;
  AsyncDeviceVector<double>                                grad_;
  AsyncDeviceVector<double>                                energyOuts_;
  //! Working-precision copy of the context positions for the per-molecule path. Unused for double precision.
  AsyncDeviceVector<real>                                  positionsScratch_;
  const RDKit::DGeomHelpers::EmbedParameters&              embedParam_;
  BfgsBatchMinimizerT<real>&                               minimizer_;
  double                                                   chiralWeight_;
  double                                                   fourthDimWeight_;
  int                                                      maxIters_;
  bool                                                     checkEnergy_;
  std::string                                              stageName_;
  cudaStream_t                                             stream_;
};

using DistGeomMinimizeStage       = DistGeomMinimizeStageT<double>;
using DistGeomMinimizeStageSingle = DistGeomMinimizeStageT<float>;

extern template class DistGeomMinimizeStageT<double>;
extern template class DistGeomMinimizeStageT<float>;

//! Wrapper stage for distance geometry minimization
//! Uses same base parameter set as base stage, but overrides weights.
template <typename real> class DistGeomMinimizeWrapperStageT final : public ETKDGStage {
 public:
  DistGeomMinimizeWrapperStageT(DistGeomMinimizeStageT<real>& baseStage,
                                double                        chiralWeight,
                                double                        fourthDimWeight,
                                int                           maxIters,
                                bool                          checkEnergy,
                                const std::string&            stageName)
      : baseStage_(baseStage),
        chiralWeight_(chiralWeight),
        fourthDimWeight_(fourthDimWeight),
        maxIters_(maxIters),
        checkEnergy_(checkEnergy),
        stageName_(stageName) {}

  void execute(ETKDGContext& ctx) override {
    baseStage_.executeImpl(ctx, chiralWeight_, fourthDimWeight_, maxIters_, checkEnergy_);
  }
  std::string name() const override { return stageName_; }

 private:
  DistGeomMinimizeStageT<real>& baseStage_;
  double                        chiralWeight_;
  double                        fourthDimWeight_;
  int                           maxIters_;
  bool                          checkEnergy_;
  std::string                   stageName_;
};

using DistGeomMinimizeWrapperStage = DistGeomMinimizeWrapperStageT<double>;

}  // namespace detail
}  // namespace nvMolKit
#endif  // NVMOLKIT_ETKDG_STAGE_DISTGEOM_MINIMIZE_H
