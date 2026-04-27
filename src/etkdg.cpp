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

#include "src/etkdg.h"

#include <omp.h>

#include <atomic>
#include <mutex>
#include <unordered_map>

#include "rdkit_extensions/conformer_pruning.h"
#include "src/utils/device.h"
#include "src/etkdg_stage_coordgen.h"
#include "src/etkdg_stage_distgeom_minimize.h"
#include "src/etkdg_stage_etk_minimization.h"
#include "src/etkdg_stage_stereochem_checks.h"
#include "src/etkdg_stage_update_conformers.h"
#include "src/utils/host_vector.h"
#include "src/utils/nvtx.h"
#include "src/utils/openmp_helpers.h"

namespace nvMolKit {
namespace {

// Helper function to calculate max iterations
unsigned int calculateMaxIterations(const std::vector<RDKit::ROMol*>& mols, unsigned int maxIterations) {
  // TODO: Support per-molecule maxIterations to match RDKit's implementation.
  // Current implementation uses a single maxIterations value for all molecules.
  // Consider adding a vector of maxIterations to the context for per-molecule control.
  if (maxIterations == 0) {
    // Find maximum number of atoms
    unsigned int maxAtoms = 0;
    for (const auto& mol : mols) {
      maxAtoms = std::max(maxAtoms, mol->getNumAtoms());
    }
    constexpr unsigned int kIterationsPerAtom = 10;
    maxIterations                             = kIterationsPerAtom * maxAtoms;
  }
  return maxIterations;
}
}  // anonymous namespace

// TODO: The useRDKitCoordGen parameter will be removed once ETKDGCoordGenStage is optimized, after which we will
// exclusively use ETKDGCoordGenStage.
void embedMolecules(const std::vector<RDKit::ROMol*>&           mols,
                    const RDKit::DGeomHelpers::EmbedParameters& params,
                    int                                         confsPerMolecule,
                    int                                         maxIterations,
                    bool                                        debugMode,
                    std::vector<std::vector<int16_t>>*          failures,
                    const BatchHardwareOptions&                 hardwareOptions,
                    BfgsBackend                                 backend) {
  const ScopedNvtxRange fullRange("EmbedMolecules");
  if (!params.useRandomCoords) {
    throw std::runtime_error("ETKDG requires useRandomCoords to be true. Please set it in the EmbedParameters.");
  }

  // Validate inputs
  for (size_t i = 0; i < mols.size(); ++i) {
    const auto* mol = mols[i];
    if (mol == nullptr) {
      throw std::invalid_argument("Invalid molecule pointer at index " + std::to_string(i));
    }
  }

  // Default to 4 batches per GPU and 500 conformers per batch
  const int batchesPerGpu = hardwareOptions.batchesPerGpu == -1 ? 4 : hardwareOptions.batchesPerGpu;
  const int numThreads =
    hardwareOptions.preprocessingThreads == -1 ? omp_get_max_threads() : hardwareOptions.preprocessingThreads;
  const int batchSize = hardwareOptions.batchSize == -1 ? 500 : hardwareOptions.batchSize;

  if (batchesPerGpu <= 0) {
    throw std::invalid_argument("batchesPerGpu must be greater than 0");
  }

  // Calculate actual batches needed and clamp batchesPerGpu if necessary
  const int actualBatchesNeeded =
    batchSize > 0 ? (static_cast<int>(mols.size()) * confsPerMolecule + batchSize - 1) / batchSize : 1;
  const int        effectivebatchesPerGpu  = std::min(batchesPerGpu, actualBatchesNeeded);
  auto             paramsCopy              = params;
  constexpr double randomCoordsBasinThresh = 1e8;
  constexpr int    dim                     = 4;  // ETKDG always uses 4D coordinates
  paramsCopy.basinThresh                   = randomCoordsBasinThresh;

  ScopedNvtxRange coordsRange("Init ETKDG");

  // Initialize context and arguments for unique molecules only
  std::vector<detail::EmbedArgs> eargs;
  eargs.resize(mols.size());

  // Pre-initialize RDKit internal data structures to avoid race conditions
  if (!mols.empty()) {
    detail::EmbedArgs dummyEarg;
    nvMolKit::DGeomHelpers::prepareEmbedderArgs(*mols[0], paramsCopy, dummyEarg);
  }

  std::vector<RDKit::ROMol*> sortedMols = mols;
  std::sort(sortedMols.begin(), sortedMols.end(), [](const RDKit::ROMol* mol1, const RDKit::ROMol* mol2) {
    return mol1->getNumAtoms() > mol2->getNumAtoms();
  });

  // Prepare embedder args for each unique molecule (without duplication)
  detail::OpenMPExceptionRegistry prepareExceptionRegistry;
#pragma omp parallel for num_threads(numThreads) default(none) \
  shared(sortedMols, eargs, paramsCopy, prepareExceptionRegistry)
  for (int i = 0; i < static_cast<int>(sortedMols.size()); i++) {
    try {
      RDKit::ROMol* mol  = sortedMols[i];
      auto&         earg = eargs[i];

      if (!nvMolKit::DGeomHelpers::prepareEmbedderArgs(*mol, paramsCopy, earg)) {
        throw std::runtime_error("Failed to prepare ETKDG parameters for molecule");
      }

      // Set dimensionality to 4D for ETKDG
      earg.dim = 4;
    } catch (...) {
      prepareExceptionRegistry.store(std::current_exception());
    }
  }
  prepareExceptionRegistry.rethrow();

  // Set max iterations if not specified
  if (maxIterations == -1) {
    maxIterations = static_cast<int>(calculateMaxIterations(sortedMols, paramsCopy.maxIterations));
  }
  coordsRange.pop();

  // Initialize failures structure if needed (outer vector is per stage, inner is per conformer)
  if (failures != nullptr) {
    failures->clear();
  }

  // Create mutex for thread-safe conformer updates and failure tracking
  std::mutex                conformer_mutex;
  std::mutex                failure_mutex;
  std::vector<ScopedStream> streamsPerThread;
  std::vector<int>          devicesPerThread;

  // Set up GPU IDs to use - default to all available GPUs if not specified
  std::vector<int> gpuIdsToUse = hardwareOptions.gpuIds;
  if (gpuIdsToUse.empty()) {
    const int numDevices = countCudaDevices();
    gpuIdsToUse.reserve(numDevices);
    for (int i = 0; i < numDevices; ++i) {
      gpuIdsToUse.push_back(i);
    }
  }

  // Assign streams to the specified GPU devices
  const int numThreadsGpuBatching = effectivebatchesPerGpu * static_cast<int>(gpuIdsToUse.size());
  for (int i = 0; i < numThreadsGpuBatching; ++i) {
    const int        deviceId = gpuIdsToUse[i % gpuIdsToUse.size()];
    const WithDevice dev(deviceId);
    streamsPerThread.emplace_back();
    devicesPerThread.push_back(deviceId);
  }

  // Work with original unique molecules, not the duplicated ones
  const size_t numUniqueMols      = sortedMols.size();
  const int    effectiveBatchSize = (batchSize <= 0) ? static_cast<int>(numUniqueMols) : batchSize;

  // Create result tracker for work dispatch
  detail::Scheduler Scheduler(static_cast<int>(numUniqueMols), confsPerMolecule, maxIterations);

  // Shared completion flag
  std::atomic<bool> workComplete{false};
  std::atomic<bool> allFinished{false};

  std::unordered_map<const RDKit::ROMol*, std::vector<std::unique_ptr<Conformer>>> conformers;

  // Process molecules using Scheduler dispatch in parallel
  detail::OpenMPExceptionRegistry dispatchExceptionRegistry;
#pragma omp parallel num_threads(numThreadsGpuBatching) default(shared)
  {
    try {
      cudaStream_t     streamPtr = streamsPerThread[omp_get_thread_num()].stream();
      const int        deviceId  = devicesPerThread[omp_get_thread_num()];
      const WithDevice dev(deviceId);
      auto             minimizer = std::make_unique<BfgsBatchMinimizer>(4,  // dataDim for ETKDG (4D distance geometry)
                                                            DebugLevel::NONE,
                                                            true,  // scaleGrads
                                                            streamPtr,
                                                            backend);
      std::unordered_map<const RDKit::ROMol*, nvMolKit::DistGeom::EnergyForceContribsHost>   dgCache;
      std::unordered_map<const RDKit::ROMol*, nvMolKit::DistGeom::Energy3DForceContribsHost> etkCache;
      // Pinned reusable buffers for common copies.
      PinnedHostVector<double>                                                               positionsScratch;
      PinnedHostVector<uint8_t>                                                              activeScratch;
      PinnedHostVector<int16_t>                                                              failuresScratch;
      detail::ETKDGDriver                                                                    driver;

      while (!workComplete.load()) {
        // Dispatch work for this thread
        std::vector<int> molIds = Scheduler.dispatch(effectiveBatchSize);

        if (molIds.empty()) {
          workComplete.store(true);
          // Work may be complete due to all passes, or if we've hit the maximum iteration count.
          // In the latter case, we still need to finish outstanding runs.
          if (Scheduler.allFinished()) {
            allFinished.store(true);
          }
          break;
        }

        // Create batch of molecules and eargs for the dispatched work
        std::vector<RDKit::ROMol*>     batchMolsWithConfs;
        std::vector<detail::EmbedArgs> batchEargs;

        batchMolsWithConfs.reserve(molIds.size());
        batchEargs.reserve(molIds.size());

        for (const int molId : molIds) {
          // Use the original unique molecules and their prepared eargs
          batchMolsWithConfs.push_back(sortedMols[molId]);
          batchEargs.push_back(eargs[molId]);
        }

        auto context = std::make_unique<detail::ETKDGContext>(streamPtr);
        detail::setStreams(*context, streamPtr);
        // Treat each conformer attempt as an individual molecule (confsPerMolecule = 1)
        detail::initETKDGContext(batchMolsWithConfs, *context, 1);

        ScopedNvtxRange                                  stageSetupRange("Setup ETKDG Stages");
        // Create stages in order
        std::vector<std::unique_ptr<detail::ETKDGStage>> stages;

        // Convert to const pointers for stages that require them
        const std::vector<const RDKit::ROMol*> constMolPtrs(batchMolsWithConfs.begin(), batchMolsWithConfs.end());

        // Create coordinate generation stage based on parameter
        // FIXME: arguments still involve useRDKitcoordgen.
        stages.push_back(std::make_unique<detail::ETKDGCoordGenRDKitStage>(paramsCopy,
                                                                           constMolPtrs,
                                                                           batchEargs,
                                                                           positionsScratch,
                                                                           activeScratch,
                                                                           streamPtr));

        // First minimize, then first round of chiral checks.
        auto                           firstMinStage    = std::make_unique<detail::DistGeomMinimizeStage>(constMolPtrs,
                                                                             batchEargs,
                                                                             paramsCopy,
                                                                             *context,
                                                                             *minimizer,
                                                                             1.0,
                                                                             0.1,
                                                                             400,
                                                                             true,
                                                                             "First Minimization",
                                                                             streamPtr,
                                                                             &dgCache);
        detail::DistGeomMinimizeStage* firstMinStagePtr = firstMinStage.get();
        stages.push_back(std::move(firstMinStage));
        stages.push_back(std::make_unique<detail::ETKDGTetrahedralCheckStage>(*context, batchEargs, dim, streamPtr));

        // Only add first chiral check if enforceChirality is enabled
        detail::ETKDGFirstChiralCenterCheckStage* chiralStagePtr = nullptr;
        if (paramsCopy.enforceChirality) {
          auto chiralStage =
            std::make_unique<detail::ETKDGFirstChiralCenterCheckStage>(*context, batchEargs, dim, streamPtr);
          chiralStagePtr = chiralStage.get();
          stages.push_back(std::move(chiralStage));
        }

        // Second + 3rd minimize, then double bond checks.
        stages.push_back(std::make_unique<detail::DistGeomMinimizeWrapperStage>(*firstMinStagePtr,
                                                                                0.2,
                                                                                1.0,
                                                                                200,
                                                                                false,
                                                                                "Fourth Dimension Minimization"));
        // (ET)(K)DG: Add experimental torsion minimization stage only if needed to match RDKit's logic.
        if (paramsCopy.useExpTorsionAnglePrefs || paramsCopy.useBasicKnowledge) {
          stages.push_back(std::make_unique<detail::ETKMinimizationStage>(constMolPtrs,
                                                                          batchEargs,
                                                                          paramsCopy,
                                                                          *context,
                                                                          *minimizer,
                                                                          streamPtr,
                                                                          &etkCache));
        }

        // Final chiral and stereochem checks
        stages.push_back(
          std::make_unique<detail::ETKDGDoubleBondGeometryCheckStage>(*context, batchEargs, dim, streamPtr));

        if (paramsCopy.enforceChirality) {
          // This is a pass-through, don't need to set the stream
          stages.push_back(std::make_unique<detail::ETKDGFinalChiralCenterCheckStage>(*chiralStagePtr));
          stages.push_back(
            std::make_unique<detail::ETKDGChiralDistMatrixCheckStage>(*context, batchEargs, dim, streamPtr));
          stages.push_back(
            std::make_unique<detail::ETKDGChiralCenterVolumeCheckStage>(*context, batchEargs, dim, streamPtr));
          stages.push_back(
            std::make_unique<detail::ETKDGDoubleBondStereoCheckStage>(*context, batchEargs, dim, streamPtr));
        }

        // Writeback
        stages.push_back(std::make_unique<detail::ETKDGUpdateConformersStage>(batchMolsWithConfs,
                                                                              batchEargs,
                                                                              conformers,
                                                                              positionsScratch,
                                                                              activeScratch,
                                                                              streamPtr,
                                                                              &conformer_mutex,
                                                                              confsPerMolecule));

        // Create and run driver
        driver.reset(std::move(context), std::move(stages), debugMode, streamPtr, &allFinished);
        stageSetupRange.pop();

        ScopedNvtxRange runRange("ETKDG execute");
        driver.run(1);
        runRange.pop();
        // Get results and record them back to Scheduler
        const std::vector<int16_t> finishedOnIteration = driver.getFinishedOnIterations();
        Scheduler.record(molIds, finishedOnIteration);

        // Handle failures if requested
        if (failures != nullptr) {
          auto batchFailures = driver.getFailures(failuresScratch);

          const std::lock_guard<std::mutex> failureLock(failure_mutex);
          // Initialize failures structure on first batch (outer vector is per stage, inner per conformer)
          if (failures->empty() && !batchFailures.empty()) {
            failures->resize(batchFailures.size());
            for (size_t stageIdx = 0; stageIdx < batchFailures.size(); ++stageIdx) {
              (*failures)[stageIdx].resize(numUniqueMols * confsPerMolecule, 0);
            }
          }

          // Merge batch failures into main failures vector
          // Note: This is a simplified approach - we're mapping each batch result to a global conformer index
          // In a more sophisticated implementation, we'd need to track which specific conformer each attempt represents
          for (size_t stageIdx = 0; stageIdx < batchFailures.size(); ++stageIdx) {
            for (size_t batchIdx = 0; batchIdx < batchFailures[stageIdx].size() && batchIdx < molIds.size();
                 ++batchIdx) {
              // For now, we'll use a simple mapping - this may need refinement based on exact requirements
              const size_t globalConformerIdx = molIds[batchIdx] * confsPerMolecule;  // Simplified mapping
              if (globalConformerIdx < (*failures)[stageIdx].size()) {
                (*failures)[stageIdx][globalConformerIdx] += batchFailures[stageIdx][batchIdx];
              }
            }
          }
        }
      }
    } catch (...) {
      dispatchExceptionRegistry.store(std::current_exception());
    }
  }

  detail::OpenMPExceptionRegistry updateExceptionRegistry;
#pragma omp parallel for num_threads(numThreads) default(none) schedule(dynamic) \
  shared(mols, conformers, params, updateExceptionRegistry)
  for (const auto& mol : mols) {
    try {
      auto iter = conformers.find(mol);
      if (iter != conformers.end()) {
        nvmolkit::addConformersToMoleculeWithPruning(*mol, iter->second, params);
      }
    } catch (...) {
      updateExceptionRegistry.store(std::current_exception());
    }
  }
  updateExceptionRegistry.rethrow();
}

}  // namespace nvMolKit
