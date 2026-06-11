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

#ifndef NVMOLKIT_ETKDG_IMPL_H
#define NVMOLKIT_ETKDG_IMPL_H

#include <DistGeom/BoundsMatrix.h>
#include <DistGeom/ChiralSet.h>
#include <ForceField/ForceField.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/CrystalFF/TorsionPreferences.h>

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/embedder_utils.h"
#include "src/forcefields/dist_geom.h"
#include "src/utils/device_vector.h"
#include "src/utils/host_vector.h"

// forward declarations

namespace RDKit {
class ROMol;
namespace DGeomHelpers {
struct EmbedParameters;
}  // namespace DGeomHelpers
}  // namespace RDKit

namespace nvMolKit {

namespace detail {

struct ETKDGSystemHost {
  //! Size n_molecules + 1, defines the start and end of each molecule in the batch
  std::vector<int>    atomStarts = {0};
  //! Size total num positions of all molecules
  std::vector<double> positions;
};

struct ETKDGSystemDevice {
  //! Size n_molecules + 1, defines the start and end of each molecule in the batch
  AsyncDeviceVector<int>    atomStarts;
  //! Size total num positions of all molecules
  AsyncDeviceVector<double> positions;
};

//! All relevant CPU/GPU data.
struct ETKDGContext {
  explicit ETKDGContext(cudaStream_t stream = nullptr) : countFinishedThisIteration(0, stream) {}

  //! Host data
  ETKDGSystemHost                         systemHost;
  //! Device data
  ETKDGSystemDevice                       systemDevice;
  //! One per molecule conformer. Default to -1, set to iteration finished on.
  AsyncDeviceVector<int16_t>              finishedOnIteration;
  //! Reset to 0 each stage.
  AsyncDeviceVector<uint8_t>              failedThisStage;
  //! One per molecule conformer. Set to 0 if molecule finished in earlier iteration or has failed a stage this
  //! iteration.
  AsyncDeviceVector<uint8_t>              activeThisStage;
  //! Summed each iteration. element [i][j] is stage i, molecule j failure count. Where i may be coordinate gen,
  //! chirality check, etc.
  std::vector<AsyncDeviceVector<int16_t>> totalFailures;
  //! Buffer for summing finished runs.
  AsyncDevicePtr<int>                     countFinishedThisIteration;
  //! Molecules * confs per molecule, typically.
  int                                     nTotalSystems = 0;
};

void setStreams(ETKDGContext& ctx, cudaStream_t stream);

//! For the given stage, determine failed conformers, deactivate for future stages within the iteration,
//! and add to failure tracker counts.
void launchCollectAndFilterFailuresKernel(ETKDGContext& context, int stageId, cudaStream_t stream);

//! Checks for newly finished conformers, returns the number of remaining conformers.
//! Accepts pinned scratch buffer for finished counts.
int launchGetFinishedKernels(ETKDGContext& context, int iteration, int* finishedCountHost, cudaStream_t stream);

//! Set initial active filter, based on if a conformer has succeeded in the previous iteration.
void launchSetRunFilterForIterationKernel(ETKDGContext& context, cudaStream_t stream);

//! Base class for all stages of the ETKDG algorithm.
//! Each stage must:
//!   1. Read the `activeThisStage` buffer, and do nothing if not set for the given conformer.
//!   2. Set the `failedThisStage` buffer for any conformer that fails the stage.
//!
//! The two above requirements are not explicitly enforced in the API for performance reasons, as the reading/writing
//! of active/failure buffers may be done inside kernels.
//!
//! Note that the active bit should not be set by stages - it is collected after each stage by the driver, based on
//! the failure bit.
class ETKDGStage {
 public:
  virtual ~ETKDGStage()                          = default;
  //! Execute the stage.
  virtual void        execute(ETKDGContext& ctx) = 0;
  //! Get the name of the stage for debugging and timing purposes
  virtual std::string name() const               = 0;
};

// Structure to hold timing information for a stage
struct StageTiming {
  double totalTime = 0.0;  // Total time spent in this stage across all iterations
  double minTime   = std::numeric_limits<double>::max();
  double maxTime   = 0.0;
  int    callCount = 0;  // Number of times this stage was executed
};

//! ETKDG Runner class.
//! Can be reused across multiple batches by calling reset() with a new context and stages.
//!
//! The runner is designed to operate on batches of molecules. Conformers for the same molecule are considered
//! independent, and any shared setup upstream of this class should be finished by the time the context is passed to the
//! constructor or reset() method.
//!
//! At each iteration, an active subset of conformers are attempted to be generated. A conformer is inactive if a
//! previous iteration completed succesfully, OR if a previous stage in the current iteration has failed.
class ETKDGDriver {
 public:
  ETKDGDriver(std::unique_ptr<ETKDGContext>&&            context,
              std::vector<std::unique_ptr<ETKDGStage>>&& stages,
              bool                                       debugMode       = false,
              cudaStream_t                               stream          = nullptr,
              const std::atomic<bool>*                   earlyExitToggle = nullptr);

  ETKDGDriver() = default;

  //! Reset the driver with a new context and stages
  void reset(std::unique_ptr<ETKDGContext>&&            context,
             std::vector<std::unique_ptr<ETKDGStage>>&& stages,
             bool                                       debugMode       = false,
             cudaStream_t                               stream          = nullptr,
             const std::atomic<bool>*                   earlyExitToggle = nullptr);

  //! Run one iteration. The iteration ID is incremented.
  void                              iterate();
  //! Return the current number of finished iterations.
  int                               iterationsComplete() const { return iteration_; }
  //! Return the total number of conformers completed as of this iteration.
  int                               numConfsFinished() const { return numFinished_; }
  //! Returns a size(batch) array of iteration numbers on which the conformer was finished. -1 if not finished.
  std::vector<int16_t>              getFinishedOnIterations() const;
  //! Returns a size(batch) array of 1/0 indicating if the given batch index has a completed conformer.
  std::vector<int16_t>              completedConformers() const;
  //! Returns failure counts. The outer vector is per stage, the inner vector is per conformer,
  //! so getFailures()[i][j] is the number of times that conformer j has failed stage i.
  //! Uses single pinned memory buffer for intermediate D2H transfer (all stages concatenated).
  std::vector<std::vector<int16_t>> getFailures(PinnedHostVector<int16_t>& failuresScratch) const;
  const ETKDGContext&               context() const { return *context_; }

  //! Iterate until all conformers are finished or maxIterations is reached. Does not reset iterations,
  //! so iterate(5) followed by iterate(5) will not run additional iterations, as 5 has already been reached, whereas
  //! iterate(5) followed by iterate(8) will run up to 3 additional iterations. Will always end early at the current
  //! iteration if all conformers are finished.
  void run(int maxIterations);

 private:
  std::unique_ptr<ETKDGContext>                context_;
  std::vector<std::unique_ptr<ETKDGStage>>     stages_;
  int                                          totalConfs_  = 0;
  int                                          numFinished_ = 0;
  int                                          iteration_   = 0;
  cudaStream_t                                 stream_      = nullptr;
  const std::atomic<bool>*                     earlyExit_   = nullptr;  // Optional early exit flag for external control
  // Debug mode members
  bool                                         debugMode_   = false;
  std::unordered_map<std::string, StageTiming> stageTimings_;

  // Pinned memory buffer for D2H transfer of finished count
  PinnedHostVector<int> finishedCountHost_{1};

  // Helper method to initialize driver state from context and stages
  void initialize();

  // Helper method to record timing for a stage
  void recordStageTiming(const std::string& stageName, double duration);

  // Helper method to print timing statistics in a table format
  void printTimingStatistics() const;
};

void initETKDGContext(const std::vector<RDKit::ROMol*>& mols, ETKDGContext& context, int confsPerMol = 1);

/**
 * @brief Tracks conformer generation results and dispatches molecule IDs for processing
 *
 * Provides a batch of molecule IDs for the next round of conformer generation attempt. First, loops over all molecules,
 * and sequentially assigns molecules that are not yet at their target conformer count and have not exceeded the max
 * iterations threshold. Each molecule will be assigned up to `numConfsPerMol` conformers, either in one batch or
 * distributed over several. If a molecule has already reached its target conformer count, it will not be included in
 * the batch.
 *
 * Once all molecules have had at least n_conformers dispatched, the scheduler will loop back to the first molecule and
 * start oversubscribing, but will only add the number of unfinished remaining conformers to start. See unit tests for
 * examples of this behavior.
 *
 * Thread-safe operations.
 *
 * TODO:
 * For a given unique molecule, ETKDG attempts N conformers with M max iterations per conformer. We approximate this
 * with N X M total attempts per unique molecule, noting that this is not strictly the same as the above case, as if
 * a molecule attempt succeeds before max attempts, the extra attempts are added on at the end.
 */
class Scheduler {
 public:
  /**
   * @brief Constructor
   * @param numUniqueMols Number of unique molecules to track (must be > 0)
   * @param numConfsPerMol Target number of conformers per molecule (must be > 0)
   * @param maxIterations Maximum iterations allowed per conformer attempt (must be > 0)
   * @throws std::invalid_argument if any parameter is <= 0
   */
  Scheduler(int numUniqueMols, int numConfsPerMol, int maxIterations);

  /**
   * @brief Dispatch molecule IDs for the next batch of processing
   *
   * Returns a vector of molecule IDs that need conformer generation attempts.
   * Prioritizes molecules that haven't reached their target conformer count
   * and haven't exceeded the maximum attempt limit. Will return duplicates if not enough
   * unique molecules are available to fill the batch size.
   *
   * @param batchSize Maximum number of molecule IDs to return
   * @return Vector of molecule IDs to process (may be smaller than batchSize if insufficient work remains)
   */
  std::vector<int> dispatch(int batchSize);

  /**
   * @brief Record the results of conformer generation attempts
   *
   * Updates the completion status for molecules based on whether their
   * conformer generation attempts succeeded or failed.
   *
   * @param molIds Vector of molecule IDs that were processed
   * @param finishedOnIteration Vector indicating completion status: -1 for failed, >= 0 for successful
   * @throws std::invalid_argument if molIds and finishedOnIteration have different sizes
   * @throws std::out_of_range if any molId is outside the valid range [0, numUniqueMols)
   */
  void record(const std::vector<int>& molIds, const std::vector<int16_t>& finishedOnIteration);

  //! Returns true if sufficient conformers for each molecule have been generated.
  bool allFinished() const {
    std::lock_guard lock(mutex_);
    return std::all_of(completedConformers_.begin(), completedConformers_.end(), [this](const int confs) {
      return confs >= numConfsPerMol_;
    });
  }

 private:
  //!
  mutable std::mutex mutex_;

  int    numConfsPerMol_;
  int    maxIterations_;
  size_t numUniqueMolecules_;
  int    maxTriesPerMolecule_;
  int    roundRobinIter_ = 1;

  std::vector<int> completedConformers_;
  std::vector<int> totalAttempts_;
};

}  // namespace detail

}  // namespace nvMolKit

#endif  // NVMOLKIT_ETKDG_IMPL_H
