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

#include "src/etkdg_impl.h"

#include <DistGeom/BoundsMatrix.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/MolOps.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <iomanip>
#include <iostream>

#include "rdkit_extensions/dist_geom_flattened_builder.h"
#include "src/triangle_smooth.h"
#include "src/utils/device_vector.h"
#include "src/utils/nvtx.h"

namespace nvMolKit {

namespace detail {

void setStreams(ETKDGContext& ctx, cudaStream_t stream) {
  ctx.systemDevice.atomStarts.setStream(stream);
  ctx.systemDevice.positions.setStream(stream);
  ctx.activeThisStage.setStream(stream);
  ctx.failedThisStage.setStream(stream);
  ctx.finishedOnIteration.setStream(stream);
  for (auto& failures : ctx.totalFailures) {
    failures.setStream(stream);
  }
  ctx.countFinishedThisIteration.setStream(stream);
}

ETKDGDriver::ETKDGDriver(std::unique_ptr<ETKDGContext>&&            context,
                         std::vector<std::unique_ptr<ETKDGStage>>&& stages,
                         bool                                       debugMode,
                         cudaStream_t                               stream,
                         const std::atomic<bool>*                   earlyExitToggle)
    : context_(std::move(context)),
      stages_(std::move(stages)),
      stream_(stream),
      earlyExit_(earlyExitToggle),
      debugMode_(debugMode) {
  initialize();
}

void ETKDGDriver::reset(std::unique_ptr<ETKDGContext>&&            context,
                        std::vector<std::unique_ptr<ETKDGStage>>&& stages,
                        bool                                       debugMode,
                        cudaStream_t                               stream,
                        const std::atomic<bool>*                   earlyExitToggle) {
  context_   = std::move(context);
  stages_    = std::move(stages);
  stream_    = stream;
  earlyExit_ = earlyExitToggle;
  debugMode_ = debugMode;

  // Reset counters
  numFinished_ = 0;
  iteration_   = 0;
  stageTimings_.clear();

  initialize();
}

void ETKDGDriver::initialize() {
  if (context_->nTotalSystems == 0) {
    throw std::runtime_error("No conformers to process.");
  }
  if (stages_.empty()) {
    throw std::runtime_error("No stages to process.");
  }
  totalConfs_ = context_->nTotalSystems;
  context_->totalFailures.resize(stages_.size());
  for (size_t i = 0; i < stages_.size(); i++) {
    context_->totalFailures[i].setStream(stream_);
    context_->totalFailures[i].resize(totalConfs_);
    context_->totalFailures[i].zero();
  }
  context_->failedThisStage.resize(context_->nTotalSystems);
  // TODO: AsyncDeviceVector should have a constructor that takes a size and a value.
  const std::vector<int16_t> copyFrom(context_->nTotalSystems, -1);
  context_->finishedOnIteration.resize(context_->nTotalSystems);
  context_->finishedOnIteration.copyFromHost(copyFrom);
  context_->activeThisStage.resize(context_->nTotalSystems);
  cudaStreamSynchronize(stream_);  // Sync before copyFrom goes out of scope
}

void ETKDGDriver::recordStageTiming(const std::string& stageName, double duration) {
  auto& timing = stageTimings_[stageName];
  timing.totalTime += duration;
  timing.minTime = std::min(timing.minTime, duration);
  timing.maxTime = std::max(timing.maxTime, duration);
  timing.callCount++;
}

void ETKDGDriver::iterate() {
  const size_t numStages = stages_.size();
  assert(context_->totalFailures.size() == numStages);

  // Set up run filter for this iteration
  launchSetRunFilterForIterationKernel(*context_, stream_);

  // Execute each stage
  for (size_t i = 0; i < numStages; i++) {
    // Check for early exit condition, external.
    if (earlyExit_ != nullptr && earlyExit_->load()) {
      break;
    }
    const ScopedNvtxRange runRange("ETKDG stage " + std::to_string(i) + ": " + stages_[i]->name());
    context_->failedThisStage.zero();
    if (debugMode_) {
      // Record start time
      auto startTime = std::chrono::high_resolution_clock::now();

      // Execute the stage
      stages_[i]->execute(*context_);

      // Record end time and calculate duration
      auto endTime  = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration<double, std::milli>(endTime - startTime).count();

      // Record timing for this stage using its name
      recordStageTiming(stages_[i]->name(), duration);
    } else {
      stages_[i]->execute(*context_);
    }

    launchCollectAndFilterFailuresKernel(*context_, static_cast<int>(i), stream_);
  }

  numFinished_ += launchGetFinishedKernels(*context_, iteration_, finishedCountHost_.data(), stream_);
  iteration_++;
}

void ETKDGDriver::run(int maxIterations) {
  while (numFinished_ < totalConfs_ && iteration_ < maxIterations) {
    const ScopedNvtxRange runRange("Iteration" + std::to_string(iteration_));
    iterate();
  }

  if (debugMode_) {
    printTimingStatistics();
  }
}

void ETKDGDriver::printTimingStatistics() const {
  // Constants for table formatting
  constexpr int kTableWidth     = 90;
  constexpr int kStageNameWidth = 30;
  constexpr int kColumnWidth    = 12;
  constexpr int kPrecision      = 3;

  std::cout << "\nETKDG Pipeline Timing Statistics\n";
  std::cout << "================================\n\n";

  // Print pipeline configuration
  std::cout << "Pipeline Configuration:\n";
  std::cout << "Total conformers to process: " << totalConfs_ << "\n";
  std::cout << "Maximum iterations: " << iteration_ << "\n";
  std::cout << std::string(kTableWidth, '-') << "\n\n";

  // Print table header
  std::cout << std::left << std::setw(kStageNameWidth) << "Stage Name" << std::right << std::setw(kColumnWidth)
            << "Total (ms)" << std::setw(kColumnWidth) << "Avg (ms)" << std::setw(kColumnWidth) << "Min (ms)"
            << std::setw(kColumnWidth) << "Max (ms)" << std::setw(kColumnWidth) << "Calls"
            << "\n";

  // Print separator line
  std::cout << std::string(kTableWidth, '-') << "\n";

  // Print data for each stage
  for (const auto& [stageName, timing] : stageTimings_) {
    const double avgTime = timing.totalTime / timing.callCount;
    std::cout << std::left << std::setw(kStageNameWidth) << stageName << std::right << std::fixed
              << std::setprecision(kPrecision) << std::setw(kColumnWidth) << timing.totalTime << std::setw(kColumnWidth)
              << avgTime << std::setw(kColumnWidth) << timing.minTime << std::setw(kColumnWidth) << timing.maxTime
              << std::setw(kColumnWidth) << timing.callCount << "\n";
  }

  // Print summary
  std::cout << "\nPipeline Summary:\n";
  std::cout << "Total iterations: " << iteration_ << "\n";
  std::cout << "Conformers finished: " << numFinished_ << "/" << totalConfs_ << "\n";
  std::cout << std::string(kTableWidth, '=') << "\n";
}

std::vector<std::vector<int16_t>> ETKDGDriver::getFailures(PinnedHostVector<int16_t>& failuresScratch) const {
  const size_t numStages = context_->totalFailures.size();

  // Calculate total required size for all stages
  size_t              totalSize = 0;
  std::vector<size_t> stageSizes;
  stageSizes.reserve(numStages);
  for (const auto& stageFailures : context_->totalFailures) {
    const size_t stageSize = stageFailures.size();
    stageSizes.push_back(stageSize);
    totalSize += stageSize;
  }

  if (failuresScratch.size() < totalSize) {
    failuresScratch.resize(totalSize);
  }

  // Dispatch all device -> pinned memory copies at once and then only one sync.
  size_t offset = 0;
  for (size_t i = 0; i < numStages; ++i) {
    context_->totalFailures[i].copyToHost(failuresScratch.data() + offset, stageSizes[i]);
    offset += stageSizes[i];
  }
  cudaStreamSynchronize(stream_);

  // Split pinned memory into individual std::vectors
  std::vector<std::vector<int16_t>> res;
  res.reserve(numStages);
  offset = 0;
  for (size_t i = 0; i < numStages; ++i) {
    res.emplace_back(failuresScratch.begin() + offset, failuresScratch.begin() + offset + stageSizes[i]);
    offset += stageSizes[i];
  }
  return res;
}

std::vector<int16_t> ETKDGDriver::getFinishedOnIterations() const {
  std::vector<int16_t> res(totalConfs_);
  context_->finishedOnIteration.copyToHost(res);
  cudaStreamSynchronize(stream_);
  return res;
}

std::vector<int16_t> ETKDGDriver::completedConformers() const {
  std::vector<int16_t> res = getFinishedOnIterations();
  std::transform(res.begin(), res.end(), res.begin(), [](int16_t elem) { return elem == -1 ? 0 : 1; });
  return res;
}

void initETKDGContext(const std::vector<RDKit::ROMol*>& mols, ETKDGContext& context, const int confsPerMol) {
  // Handle atom offsets.
  for (const auto* mol : mols) {
    // Add to
    const size_t numAtoms = mol->getNumAtoms();
    for (int i = 0; i < confsPerMol; i++) {
      nvMolKit::DistGeom::addMoleculeToContext(4,
                                               static_cast<int>(numAtoms),
                                               context.nTotalSystems,
                                               context.systemHost.atomStarts,
                                               context.systemHost.positions);
    }
  }
  // Send off async GPU op before moving on to CPU preprocessing.
  // TODO: Are positions even valid at this point? Should it just be atomstarts?
  nvMolKit::DistGeom::sendContextToDevice(context.systemHost.positions,
                                          context.systemDevice.positions,
                                          context.systemHost.atomStarts,
                                          context.systemDevice.atomStarts);
}

Scheduler::Scheduler(const int numConfsPerMol, const std::vector<int>& maxIterationsPerMol)
    : numConfsPerMol_(numConfsPerMol),
      numUniqueMolecules_(maxIterationsPerMol.size()) {
  if (maxIterationsPerMol.empty() || numConfsPerMol <= 0 ||
      std::any_of(maxIterationsPerMol.begin(), maxIterationsPerMol.end(), [](const int it) { return it <= 0; })) {
    throw std::invalid_argument("All parameters must be greater than 0.");
  }

  maxTriesPerMolecule_.reserve(numUniqueMolecules_);
  for (const int maxIterations : maxIterationsPerMol) {
    maxTriesPerMolecule_.push_back(maxIterations * numConfsPerMol_);
  }
  completedConformers_.resize(numUniqueMolecules_, 0);
  totalAttempts_.resize(numUniqueMolecules_, 0);
  attemptsInFlightByMolecule_.resize(numUniqueMolecules_, 0);
}

std::vector<int> Scheduler::dispatch(const int batchSize, std::vector<int>* attemptIds) {
  std::unique_lock lock(mutex_);
  if (attemptIds != nullptr) {
    attemptIds->clear();
  }
  if (batchSize <= 0) {
    return {};
  }
  while (true) {
    if (canceled_) {
      return {};
    }
    auto molIds = dispatchAvailableLocked(batchSize, attemptIds);
    if (!molIds.empty()) {
      return molIds;
    }
    if (attemptsInFlight_ > 0) {
      progressCondition_.wait(lock);
    } else {
      return {};
    }
  }
}

void Scheduler::cancel() {
  {
    const std::lock_guard lock(mutex_);
    canceled_ = true;
  }
  progressCondition_.notify_all();
}

std::vector<int> Scheduler::dispatchAvailableLocked(const int batchSize, std::vector<int>* attemptIds) {
  std::vector<int> molIds;
  molIds.reserve(batchSize);
  if (attemptIds != nullptr) {
    attemptIds->clear();
    attemptIds->reserve(batchSize);
  }
  std::vector<size_t> eligible;
  for (size_t i = 0; i < numUniqueMolecules_; i++) {
    // Keep one attempt in flight for each conformer still needed. Recording a
    // failure immediately reopens its slot without waiting on other molecules.
    while (completedConformers_[i] + attemptsInFlightByMolecule_[i] < numConfsPerMol_ &&
           totalAttempts_[i] < maxTriesPerMolecule_[i]) {
      if (static_cast<int>(molIds.size()) >= batchSize) {
        return molIds;
      }
      pushAttemptLocked(i, molIds, attemptIds);
    }
    if (completedConformers_[i] < numConfsPerMol_ && totalAttempts_[i] < maxTriesPerMolecule_[i]) {
      eligible.push_back(i);
    }
  }

  // Fill spare capacity round-robin with extra attempts for unfinished molecules, ignoring the in-flight cap.
  // Surplus successes are discarded at writeback.
  while (!eligible.empty() && static_cast<int>(molIds.size()) < batchSize) {
    size_t numStillEligible = 0;
    for (const size_t i : eligible) {
      if (static_cast<int>(molIds.size()) < batchSize) {
        pushAttemptLocked(i, molIds, attemptIds);
      }
      if (totalAttempts_[i] < maxTriesPerMolecule_[i]) {
        eligible[numStillEligible++] = i;
      }
    }
    eligible.resize(numStillEligible);
  }
  return molIds;
}

void Scheduler::pushAttemptLocked(const size_t molIdx, std::vector<int>& molIds, std::vector<int>* attemptIds) {
  molIds.push_back(static_cast<int>(molIdx));
  if (attemptIds != nullptr) {
    attemptIds->push_back(totalAttempts_[molIdx]);
  }
  totalAttempts_[molIdx]++;
  attemptsInFlightByMolecule_[molIdx]++;
  attemptsInFlight_++;
}

void Scheduler::record(const std::vector<int>& molIds, const std::vector<int16_t>& finishedOnIteration) {
  if (molIds.size() != finishedOnIteration.size()) {
    throw std::invalid_argument("molIds and finishedOnIteration must have the same size.");
  }
  {
    const std::lock_guard lock(mutex_);
    for (const int molId : molIds) {
      if (molId < 0 || static_cast<size_t>(molId) >= numUniqueMolecules_) {
        throw std::out_of_range("molId is out of range: " + std::to_string(molId));
      }
    }
    for (size_t i = 0; i < molIds.size(); i++) {
      const int molId = molIds[i];
      completedConformers_[molId] += finishedOnIteration[i] == -1 ? 0 : 1;
      attemptsInFlightByMolecule_[molId]--;
    }
    attemptsInFlight_ -= static_cast<int>(molIds.size());
  }
  progressCondition_.notify_all();
}

}  // namespace detail

}  // namespace nvMolKit
