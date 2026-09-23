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

#ifndef NVMOLKIT_BFGS_MINIMIZE_H
#define NVMOLKIT_BFGS_MINIMIZE_H

#include <memory>
#include <vector>

#include "src/minimizer/bfgs_types.h"
#include "src/precision/precision_mode.h"
#include "src/utils/device_vector.h"
#include "src/utils/host_vector.h"

namespace nvMolKit {

class BatchedForcefield;

// Forward declarations for forcefield types
namespace MMFF {
template <typename ParameterScalar, typename CoordinateScalar, typename TorsionScalar>
struct BatchedMolecularDeviceBuffersT;
using BatchedMolecularDeviceBuffers       = BatchedMolecularDeviceBuffersT<double, double, float>;
using BatchedMolecularDeviceBuffersSingle = BatchedMolecularDeviceBuffersT<float, float, float>;
}  // namespace MMFF

namespace DistGeom {
struct BatchedMolecularDeviceBuffers;
struct BatchedMolecular3DDeviceBuffers;
}  // namespace DistGeom

//! \brief Computes energies, optionally on an external set of positions.
//! \param positions Optional flattened coordinate buffer to evaluate.
//!        When null, the implementation uses its internal position storage.
//! Precision-dependent device state for the batched BFGS implementation.
template <typename real, typename reduceT, typename storageT> struct BfgsWorkspace {
  using ComputeScalar   = real;
  using ReductionScalar = reduceT;
  using StorageScalar   = storageT;

  AsyncDeviceVector<storageT>  scratchPositions;
  AsyncDeviceVector<storageT>  positions;
  AsyncDeviceVector<storageT>  grad;
  AsyncDeviceVector<storageT>  lineSearchDir;
  AsyncDeviceVector<storageT>  lineSearchLambdaMins;
  AsyncDeviceVector<storageT>  lineSearchLambdas;
  AsyncDeviceVector<storageT>  lineSearchLambdas2;
  AsyncDeviceVector<storageT>  lineSearchSlope;
  AsyncDeviceVector<storageT>  lineSearchMaxSteps;
  AsyncDeviceVector<storageT>  lineSearchStoredEnergy;
  AsyncDeviceVector<storageT>  lineSearchEnergyScratch;
  AsyncDeviceVector<storageT>  energy;
  AsyncDeviceVector<storageT>  scratchGrad;
  AsyncDeviceVector<storageT>  gradScales;
  AsyncDeviceVector<storageT>  inverseHessian;
  AsyncDeviceVector<storageT>  hessDGrad;
  AsyncDeviceVector<storageT*> scratchBufferPointers;
  PinnedHostVector<storageT*>  scratchBufferPointersHost;

  void setStream(cudaStream_t stream) {
    scratchPositions.setStream(stream);
    positions.setStream(stream);
    grad.setStream(stream);
    lineSearchDir.setStream(stream);
    lineSearchLambdaMins.setStream(stream);
    lineSearchLambdas.setStream(stream);
    lineSearchLambdas2.setStream(stream);
    lineSearchSlope.setStream(stream);
    lineSearchMaxSteps.setStream(stream);
    lineSearchStoredEnergy.setStream(stream);
    lineSearchEnergyScratch.setStream(stream);
    energy.setStream(stream);
    scratchGrad.setStream(stream);
    gradScales.setStream(stream);
    inverseHessian.setStream(stream);
    hessDGrad.setStream(stream);
    scratchBufferPointers.setStream(stream);
  }
};

using FullBfgsWorkspace   = BfgsWorkspace<double, double, double>;
using SingleBfgsWorkspace = BfgsWorkspace<float, float, float>;

//! BFGS Batch Minimizer
//!
//! This class implements a BFGS minimizer for batch systems, should be a 1:1 port of the RDKit BFGS minimizer.
//! \param dataDim Dimensionality of positions, default is 3 for 3D systems.
//! \param debugLevel Debug level, default is NONE. STEPWISE will collect stepwise data for debugging.
//! \param scaleGrads Whether to dynamically scale down gradients to match RDKit forcefield calculations, default is
//! true.
//!                   Note that when true, simple systems may not converge as well, but it is necessary for
//!                   compatibility with RDKit forcefield calculations.
//! TODO: Constructor should be parameter struct based, now that we have more parameters.
struct BfgsBatchMinimizer {
  explicit BfgsBatchMinimizer(int           dataDim    = 3,
                              DebugLevel    debugLevel = DebugLevel::NONE,
                              bool          scaleGrads = true,
                              cudaStream_t  stream     = nullptr,
                              BfgsBackend   backend    = BfgsBackend::BATCHED,
                              PrecisionMode precision  = PrecisionMode::FULL);
  ~BfgsBatchMinimizer();

  const PrecisionMode& precision() const { return precision_; }

  //! \brief Runs host-driven batched BFGS through the forcefield abstraction.
  //! \param numIters Maximum number of BFGS iterations to perform.
  //! \param gradTol Convergence tolerance applied to the scaled gradients.
  //! \param ff Forcefield adapter used to evaluate energies and gradients.
  //! \param positions Flattened coordinate buffer for the batch.
  //! \param grad Gradient output buffer matching `positions`.
  //! \param energyOuts Per-system energy output buffer.
  //! \param activeSystemMask Optional per-system activity mask for staged minimization.
  //! \return `false` when all systems converged and `true` when at least one system needs another cycle.
  //! \note This overload is only valid for the batched backend.
  bool minimize(int                        numIters,
                double                     gradTol,
                BatchedForcefield&         ff,
                AsyncDeviceVector<double>& positions,
                AsyncDeviceVector<double>& grad,
                AsyncDeviceVector<double>& energyOuts,
                const uint8_t*             activeSystemMask = nullptr);

  //! \brief Runs MMFF minimization through the per-molecule CUDA kernels.
  //! \param numIters Maximum number of BFGS iterations to perform.
  //! \param gradTol Convergence tolerance applied to the scaled gradients.
  //! \param atomStartsHost Host-side atom offsets for the flattened systems.
  //! \param systemDevice MMFF device buffers used by the per-molecule kernels.
  //! \param activeThisStage Optional per-system activity mask for staged minimization.
  //! \return `false` when all systems converged and `true` when at least one system needs another cycle.
  bool minimizeWithMMFF(int                                  numIters,
                        double                               gradTol,
                        const std::vector<int>&              atomStartsHost,
                        MMFF::BatchedMolecularDeviceBuffers& systemDevice,
                        const uint8_t*                       activeThisStage = nullptr);
  //! \brief Runs native single-precision MMFF minimization through the per-molecule CUDA kernels.
  //! \param numIters Maximum number of BFGS iterations to perform.
  //! \param gradTol Convergence tolerance applied to the scaled gradients.
  //! \param atomStartsHost Host-side atom offsets for the flattened systems.
  //! \param systemDevice Single-precision MMFF device buffers used by the per-molecule kernels.
  //! \param activeThisStage Optional per-system activity mask for staged minimization.
  //! \return `false` when all systems converged and `true` when at least one system needs another cycle.
  bool minimizeWithMMFF(int                                        numIters,
                        double                                     gradTol,
                        const std::vector<int>&                    atomStartsHost,
                        MMFF::BatchedMolecularDeviceBuffersSingle& systemDevice,
                        const uint8_t*                             activeThisStage = nullptr);
  //! \brief Runs ETK minimization through the per-molecule CUDA kernels.
  //! \param numIters Maximum number of BFGS iterations to perform.
  //! \param gradTol Convergence tolerance applied to the scaled gradients.
  //! \param atomStartsHost Host-side atom offsets for the flattened systems.
  //! \param atomStarts Device-side atom offsets for the flattened systems.
  //! \param positions Flattened coordinate buffer for the batch.
  //! \param systemDevice ETK device buffers used by the per-molecule kernels.
  //! \param activeThisStage Optional per-system activity mask for staged minimization.
  //! \return `false` when all systems converged and `true` when at least one system needs another cycle.
  bool minimizeWithETK(int                                        numIters,
                       double                                     gradTol,
                       const std::vector<int>&                    atomStartsHost,
                       const AsyncDeviceVector<int>&              atomStarts,
                       AsyncDeviceVector<double>&                 positions,
                       DistGeom::BatchedMolecular3DDeviceBuffers& systemDevice,
                       const uint8_t*                             activeThisStage = nullptr);

  //! \brief Runs DG minimization through the per-molecule CUDA kernels.
  //! \param numIters Maximum number of BFGS iterations to perform.
  //! \param gradTol Convergence tolerance applied to the scaled gradients.
  //! \param atomStartsHost Host-side atom offsets for the flattened systems.
  //! \param atomStarts Device-side atom offsets for the flattened systems.
  //! \param positions Flattened coordinate buffer for the batch.
  //! \param systemDevice DG device buffers used by the per-molecule kernels.
  //! \param chiralWeight Weight applied to the DG chirality term.
  //! \param fourthDimWeight Weight applied to the DG fourth-dimension term.
  //! \param activeThisStage Optional per-system activity mask for staged minimization.
  //! \return `false` when all systems converged and `true` when at least one system needs another cycle.
  bool minimizeWithDG(int                                      numIters,
                      double                                   gradTol,
                      const std::vector<int>&                  atomStartsHost,
                      const AsyncDeviceVector<int>&            atomStarts,
                      AsyncDeviceVector<double>&               positions,
                      DistGeom::BatchedMolecularDeviceBuffers& systemDevice,
                      double                                   chiralWeight,
                      double                                   fourthDimWeight,
                      const uint8_t*                           activeThisStage = nullptr);

  //! \brief Resolves the effective backend for the provided batch.
  //! \param atomStartsHost Host-side atom offsets for the systems under consideration.
  //! \return The effective backend after applying the HYBRID size heuristic.
  BfgsBackend resolveBackend(const std::vector<int>& atomStartsHost) const;

  //! \brief Initializes persistent buffers for a new batch of systems.
  //! \param atomStartsHost Host-side atom offsets for the batch.
  //! \param atomStarts Device-side atom offsets for the batch.
  //! \param positions Flattened coordinate buffer for the batch.
  //! \param grad Gradient buffer matching `positions`.
  //! \param energyOuts Per-system energy buffer.
  //! \param effectiveBackend Backend that will be used for this run.
  //! \param activeThisStage Optional per-system activity mask for staged minimization.
  void initialize(const std::vector<int>& atomStartsHost,
                  const int*              atomStarts,
                  double*                 positions,
                  double*                 grad,
                  double*                 energyOuts,
                  BfgsBackend             effectiveBackend,
                  const uint8_t*          activeThisStage = nullptr);

  //! \brief Sets the initial inverse Hessian approximation to the identity matrix.
  void setHessianToIdentity();
  //! \brief Determines the maximum line-search step for each active system.
  void setMaxStep();
  //! \brief Initializes line-search buffers from the current energies.
  void doLineSearchSetup(const double* srcEnergies);
  //! \brief Perturbs positions along the current search direction.
  void doLineSearchPerturb();
  //! \brief Updates line-search lambdas after evaluating the perturbed energies.
  void doLineSearchPostEnergy(int iter);
  //! \brief Finalizes line-search state before the next BFGS update.
  void doLineSearchPostLoop();
  //! \brief Counts the systems that have finished their current line search.
  int  lineSearchCountFinished() const;
  //! \brief Updates the search direction from the current inverse Hessian and gradient.
  void setDirection();
  //! \brief Scales gradients to match RDKit forcefield conventions.
  void scaleGrad(bool preLoop);
  //! \brief Updates the gradient-difference buffer and convergence statuses.
  void updateDGrad();
  //! \brief Compacts converged systems out of the active set and returns their count.
  int  compactAndCountConverged() const;
  //! \brief Applies the BFGS inverse-Hessian update to all active systems.
  void updateHessian();
  //! \brief Captures per-iteration debug data when stepwise debugging is enabled.
  void collectDebugData();

  AsyncDeviceVector<int> allSystemIndices_;
  AsyncDeviceVector<int> activeSystemIndices_;  // Indices of systems that are active in the current iteration.
  mutable int            numUnfinishedSystems_ = 0;

  // Precision-dependent working state. Public coordinates, gradients, and
  // energies remain double precision at the API boundary.
  std::unique_ptr<FullBfgsWorkspace>   fullWorkspace_;
  std::unique_ptr<SingleBfgsWorkspace> singleWorkspace_;
  AsyncDeviceVector<int16_t>           statuses_;

  // Precision-independent line-search status and public energy output.
  AsyncDeviceVector<int16_t>         lineSearchStatus_;
  // Temporary buffers for counting finished systems. Mutable to allow
  // for const counting methods.
  mutable AsyncDeviceVector<uint8_t> countTempStorage_;
  mutable AsyncDevicePtr<int>        countFinished_;
  mutable PinnedHostVector<int>      loopStatusHost_;

  // Hessian approximation and scratch buffers.
  AsyncDeviceVector<int> hessianStarts_;

  int  dataDim_        = 3;      // Dimensionality of positions.
  bool scaleGrads_     = true;   // Whether to scale gradients to match RDKit forcefield.
  bool hasLargeSystem_ = false;  // Whether any system exceeds shared-memory kernel limit

  // Tracking variables to determine if system needs initializing.
  int numAtomsTotal_ = 0;
  int numSystems_    = 0;

  double gradTol_ = 0.0;

  // The following are non-owning pointers to device, owned by
  // (e.g.) an MMFF system description.
  const int* atomStartsDevice = nullptr;
  double*    positionsDevice  = nullptr;
  double*    gradDevice       = nullptr;
  double*    energyOutsDevice = nullptr;

  DebugLevel                        debugLevel_ = DebugLevel::NONE;
  BfgsBackend                       backend_    = BfgsBackend::BATCHED;
  PrecisionMode                     precision_;
  std::vector<std::vector<int16_t>> stepwiseStatuses;
  std::vector<std::vector<double>>  stepwiseEnergies;

  // Per-molecule kernel data (used when backend_ == PER_MOLECULE)
  int                    maxAtomsInBatch_ = 0;  // Largest molecule in batch (for kernel dispatch)
  std::vector<int>       activeMolIds_;         // Active molecule IDs
  AsyncDeviceVector<int> activeMolIdsDevice_;   // Device copy of active molecule IDs

  // Pinned host buffers for async transfers (allocated lazily in initialize())
  PinnedHostVector<uint8_t> activeHost_;
  PinnedHostVector<int16_t> convergenceHost_;  // Changed to int16_t to match statuses_

  // Persistent host vectors for async copies (to avoid stack allocation issues)
  std::vector<int> systemIndicesHost_;
  std::vector<int> hessianStartsHost_;

  cudaStream_t stream_ = nullptr;

 private:
  FullBfgsWorkspace&         fullWorkspace() { return *fullWorkspace_; }
  const FullBfgsWorkspace&   fullWorkspace() const { return *fullWorkspace_; }
  SingleBfgsWorkspace&       singleWorkspace() { return *singleWorkspace_; }
  const SingleBfgsWorkspace& singleWorkspace() const { return *singleWorkspace_; }

  template <typename Workspace>
  void initializeImpl(const std::vector<int>& atomStartsHost,
                      const int*              atomStarts,
                      double*                 positions,
                      double*                 grad,
                      double*                 energyOuts,
                      BfgsBackend             effectiveBackend,
                      Workspace&              workspace,
                      const uint8_t*          activeThisStage);
  template <typename Workspace, typename storageT>
  void doLineSearchSetupImpl(const storageT* srcEnergies,
                             const storageT* positions,
                             const storageT* grads,
                             Workspace&      workspace);
  template <typename Workspace, typename storageT>
  void doLineSearchPerturbImpl(const storageT* positions, Workspace& workspace);
  template <typename Workspace, typename storageT>
  void doLineSearchPostEnergyImpl(int iter, const storageT* energies, Workspace& workspace);
  template <typename Workspace, typename storageT>
  void                               doLineSearchPostLoopImpl(const storageT* positions, Workspace& workspace);
  template <typename Workspace> void setHessianToIdentityImpl(Workspace& workspace);
  template <typename Workspace, typename storageT> void setMaxStepImpl(const storageT* positions, Workspace& workspace);
  template <typename Workspace, typename storageT>
  void setDirectionImpl(storageT* positions, storageT* grads, Workspace& workspace);
  template <typename Workspace, typename storageT>
  void scaleGradImpl(bool preLoop, storageT* grads, Workspace& workspace);
  template <typename Workspace, typename storageT>
  void                                                  updateDGradImpl(const storageT* energies,
                                                                        const storageT* grads,
                                                                        const storageT* positions,
                                                                        Workspace&      workspace);
  template <typename Workspace, typename storageT> void updateHessianImpl(const storageT* grads, Workspace& workspace);
  template <typename storageT> void                     collectDebugDataImpl(const storageT* energies);
  template <typename Workspace, typename storageT, typename EnergyEvaluator, typename GradientEvaluator>
  bool minimizeImpl(int                        numIters,
                    double                     gradTol,
                    AsyncDeviceVector<double>& publicPositions,
                    AsyncDeviceVector<double>& publicGrad,
                    AsyncDeviceVector<double>& publicEnergies,
                    storageT*                  positions,
                    storageT*                  grad,
                    storageT*                  energies,
                    Workspace&                 workspace,
                    EnergyEvaluator            evaluateEnergy,
                    GradientEvaluator          evaluateGradient);

  template <typename DeviceBuffers, typename Workspace>
  bool minimizeWithMMFFImpl(int                     numIters,
                            double                  gradTol,
                            const std::vector<int>& atomStartsHost,
                            DeviceBuffers&          systemDevice,
                            Workspace&              workspace,
                            const uint8_t*          activeThisStage);
};

void copyAndInvert(const AsyncDeviceVector<double>& src, AsyncDeviceVector<double>& dst);
void copyAndInvert(const AsyncDeviceVector<float>& src, AsyncDeviceVector<float>& dst);

}  // namespace nvMolKit

#endif  // NVMOLKIT_BFGS_MINIMIZE_H
