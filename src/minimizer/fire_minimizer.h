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

#ifndef NVMOLKIT_FIRE_MINIMIZER_H
#define NVMOLKIT_FIRE_MINIMIZER_H

#include <vector>

#include "src/minimizer/bfgs_types.h"
#include "src/minimizer/fire_options.h"
#include "src/minimizer/minimizer_api.h"
#include "src/utils/device_vector.h"
#include "src/utils/host_vector.h"

namespace nvMolKit {

class BatchedForcefield;

namespace MMFF {
struct BatchedMolecularDeviceBuffers;
}  // namespace MMFF

//! \brief Per-system per-iteration debug snapshot recorded when the minimizer is
//! constructed in debug mode.
struct FireDebugOutput {
  std::vector<double> alphas;
  std::vector<double> dt;
  std::vector<double> powers;
  std::vector<double> energies;
};

//! \brief Snapshot of internal per-system state, exposed for testing.
struct FireInternalState {
  std::vector<double>  velocities;
  std::vector<double>  dt;
  std::vector<double>  alpha;
  std::vector<int>     nStepsPositive;
  std::vector<uint8_t> statuses;
};

//! \brief Batched FIRE 2.0 minimizer.
//!
//! Implements the ASE FIRE 2.0 algorithm with a single semi-implicit Euler
//! integrator, an optional ABC-FIRE mixer correction, and a post-mixer
//! 2-norm displacement clip. Each system in the batch maintains its own
//! per-system state (dt, alpha, nStepsPositive, velocities) and converges
//! independently. Inactive systems (passed via @p activeThisStage at
//! construction or marked converged during minimization) are not touched.
class FireBatchMinimizer final : public BatchMinimizer {
 public:
  explicit FireBatchMinimizer(int                dataDim   = 3,
                              const FireOptions& options   = FireOptions(),
                              cudaStream_t       stream    = nullptr,
                              bool               debugMode = false,
                              FireBackend        backend   = FireBackend::BATCHED);
  ~FireBatchMinimizer() override = default;

  //! \brief Resolve the effective backend for the provided batch under HYBRID selection.
  FireBackend resolveBackend(const std::vector<int>& atomStartsHost) const;

  //! \brief Initialize internal buffers for a new batch.
  //! \param atomStartsHost Host offsets for the first atom of each system.
  //! \param masses Optional pointer to per-atom masses; nullptr means use any masses set via setMasses().
  //! \param activeThisStage Optional uint8_t mask (1 = active). When nullptr all systems start active.
  //! \param effectiveBackend Selects which backend's auxiliary buffers to materialize.
  //!        Pass the value returned by ::resolveBackend so HYBRID is collapsed first.
  void initialize(const std::vector<int>& atomStartsHost,
                  const double*           masses           = nullptr,
                  const uint8_t*          activeThisStage  = nullptr,
                  FireBackend             effectiveBackend = FireBackend::BATCHED);

  //! \brief Provide per-atom masses to be used on the next initialization
  //! when explicit masses are not supplied. Passing an empty vector clears
  //! previously stored masses.
  void setMasses(const std::vector<double>& masses);

  //! \brief Run a single FIRE step synchronously.
  //! \return True if all (initially-active) systems are now converged.
  bool step(double                        gradTol,
            const AsyncDeviceVector<int>& atomStarts,
            AsyncDeviceVector<double>&    positions,
            AsyncDeviceVector<double>&    grad,
            const GradFunctor&            gFunc);

  bool minimize(int                           numIters,
                double                        gradTol,
                const std::vector<int>&       atomStartsHost,
                const AsyncDeviceVector<int>& atomStarts,
                AsyncDeviceVector<double>&    positions,
                AsyncDeviceVector<double>&    grad,
                AsyncDeviceVector<double>&    energyOuts,
                AsyncDeviceVector<double>&    energyBuffer,
                EnergyFunctor                 eFunc,
                GradFunctor                   gFunc,
                const uint8_t*                activeThisStage = nullptr) override;

  //! \brief Minimize using a BatchedForcefield directly.
  bool minimize(int                        numIters,
                double                     gradTol,
                BatchedForcefield&         ff,
                AsyncDeviceVector<double>& positions,
                AsyncDeviceVector<double>& grad,
                AsyncDeviceVector<double>& energyOuts,
                const uint8_t*             activeSystemMask = nullptr);

  //! \brief Run MMFF FIRE minimization through the per-molecule kernel.
  //! \pre Backend must be ::FireBackend::PER_MOLECULE or ::FireBackend::HYBRID resolving
  //!      to PER_MOLECULE for this batch.
  //! \pre ::FireOptions::stuckDetectionEnabled must be false; the per-molecule path does
  //!      not support FIRE energy-plateau detection.
  bool minimizeWithMMFF(int                                  numIters,
                        double                               gradTol,
                        const std::vector<int>&              atomStartsHost,
                        MMFF::BatchedMolecularDeviceBuffers& systemDevice,
                        const uint8_t*                       activeThisStage = nullptr);

  const std::vector<FireDebugOutput>& debugOutputs() const { return debugOutputs_; }

  //! \brief Cadence (in iterations) at which the minimize() loop reads the
  //! still-running system count back to the host. Default 8.
  //! \note Only the BATCHED backend uses this; per-molecule kernels iterate
  //! entirely device-side and ignore the poll interval.
  void setConvergencePollInterval(int interval);

  //! \brief Read back internal per-system state for testing.
  FireInternalState snapshotInternalState() const;

  //! \brief Number of currently-active systems (host-side cached).
  int numActiveSystemsHost() const { return lastKnownNumUnfinished_; }

  //! \brief Per-system convergence status (0 = converged, 1 = not converged).
  const AsyncDeviceVector<uint8_t>& statuses() const { return statuses_; }

  //! \brief Forget any cached batch state so the next @c initialize() call resets all
  //! per-system convergence state (statuses, streak counters, etc.). Use before starting
  //! a new minimization session on the same minimizer instance when the active-mask
  //! contents may have changed (the address comparison alone cannot detect that).
  void resetContinuationCache();

 private:
  void launchPreKick(double                        gradTol,
                     const AsyncDeviceVector<int>& atomStarts,
                     AsyncDeviceVector<double>&    positions,
                     AsyncDeviceVector<double>&    grad,
                     int                           launchBlocks,
                     bool                          isFirstStep);
  void launchPostKick(double                        gradTol,
                      const AsyncDeviceVector<int>& atomStarts,
                      AsyncDeviceVector<double>&    positions,
                      AsyncDeviceVector<double>&    grad,
                      int                           launchBlocks);
  void compactActiveAsync();
  int  readbackNumUnfinished();

  //! \brief Copy per-molecule statuses to host and report whether all active systems converged.
  bool checkPerMolConvergence();

  int          dataDim_;
  FireOptions  fireOptions_;
  cudaStream_t stream_;
  int          step_                    = 0;
  bool         debugMode_               = false;
  int          numSystems_              = 0;
  int          convergencePollInterval_ = 8;
  int          lastKnownNumUnfinished_  = 0;
  FireBackend  backend_                 = FireBackend::BATCHED;

  AsyncDeviceVector<double> velocities_;
  AsyncDeviceVector<double> masses_;

  AsyncDeviceVector<double>  dt_;
  AsyncDeviceVector<double>  alpha_;
  AsyncDeviceVector<int>     numStepsWithPositivePower_;
  AsyncDeviceVector<uint8_t> statuses_;

  AsyncDeviceVector<uint8_t> countTempStorage_;
  AsyncDevicePtr<int>        countUnfinished_;
  PinnedHostVector<int>      loopStatusHost_;
  AsyncDeviceVector<int>     activeSystemIndices_;
  AsyncDeviceVector<int>     allSystemIndices_;

  std::vector<double> hostMasses_;

  AsyncDeviceVector<double>    debugPowers_;
  std::vector<FireDebugOutput> debugOutputs_;

  //! Cached last-call signature for detecting continuation calls with the same batch
  //! and active mask. When set, @c initialize() preserves
  //! per-system convergence state (statuses, streak counters, convergeReason) so that
  //! systems that already converged in the previous call are not re-run.
  bool           hasInitializedBatch_   = false;
  int            cachedNumSystems_      = -1;
  int            cachedTotalAtoms_      = -1;
  const uint8_t* cachedActiveThisStage_ = nullptr;
  const double*  cachedMasses_          = nullptr;

  //! Per-system state for energy-plateau stuck detection. ``energyMinStreak_`` and
  //! ``energyMaxStreak_`` track the windowed extrema while ``stuckStreak_`` counts
  //! consecutive plateau polls; all reset when the relative tolerance is violated.
  AsyncDeviceVector<double>  energyMinStreak_;
  AsyncDeviceVector<double>  energyMaxStreak_;
  AsyncDeviceVector<int32_t> stuckStreak_;
  int                        pollsSinceLastEnergyEval_ = 0;

  //! Per-system convergence reason for diagnostics: 0=active, 1=grad-tol, 2=stuck-plateau.
  AsyncDeviceVector<uint8_t> convergeReason_;

  // Per-molecule kernel data (used when backend_ == PER_MOLECULE / HYBRID resolves to it).
  int                       maxAtomsInBatch_ = 0;  //!< Largest molecule in batch (for kernel dispatch).
  std::vector<int>          activeMolIds_;         //!< Active molecule IDs (host).
  AsyncDeviceVector<int>    activeMolIdsDevice_;   //!< Device copy of @c activeMolIds_.
  PinnedHostVector<uint8_t> activeHost_;           //!< Pinned scratch for caller-supplied active mask.
  PinnedHostVector<uint8_t> convergenceHost_;      //!< Pinned scratch for status readback.
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_FIRE_MINIMIZER_H
