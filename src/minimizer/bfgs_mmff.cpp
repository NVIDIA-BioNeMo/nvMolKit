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

#include "bfgs_mmff.h"

#include <GraphMol/ROMol.h>
#include <omp.h>

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

#include "bfgs_common.h"
#include "bfgs_minimize.h"
#include "ff_device_collect.h"
#include "ff_utils.h"
#include "mmff_batched_forcefield.h"
#include "mmff_flattened_builder.h"
#include "nvtx.h"
#include "openmp_helpers.h"

namespace nvMolKit::MMFF {

//! Cached molecule-specific preprocessing
struct CachedMoleculeData {
  EnergyForceContribsHost ffParams;
};

std::vector<std::vector<double>> MMFFOptimizeMoleculesConfsBfgs(std::vector<RDKit::ROMol*>& mols,
                                                                const int                   maxIters,
                                                                const MMFFProperties&       properties,
                                                                const BatchHardwareOptions& perfOptions,
                                                                const BfgsBackend           backend) {
  return MMFFOptimizeMoleculesConfsBfgs(mols,
                                        maxIters,
                                        std::vector<MMFFProperties>(mols.size(), properties),
                                        perfOptions,
                                        backend);
}

MMFFMinimizeResult MMFFMinimizeMoleculesConfs(std::vector<RDKit::ROMol*>&                                  mols,
                                              const int                                                    maxIters,
                                              const double                                                 gradTol,
                                              const std::vector<MMFFProperties>&                           properties,
                                              const std::vector<ForceFieldConstraints::PerMolConstraints>& constraints,
                                              const BatchHardwareOptions&                                  perfOptions,
                                              const BfgsBackend                                            backend,
                                              const CoordinateOutput                                       output,
                                              int                                                          targetGpu,
                                              const DeviceCoordResult* deviceInput) {
  ScopedNvtxRange fullRange("BFGS MMFF Minimize Molecules Confs");

  if (properties.size() != mols.size()) {
    throw std::invalid_argument("Expected one MMFFProperties entry per molecule");
  }
  if (!constraints.empty() && constraints.size() != mols.size()) {
    throw std::invalid_argument("Expected one PerMolConstraints entry per molecule");
  }
  if (deviceInput != nullptr) {
    const bool anyConstraint =
      std::any_of(constraints.begin(), constraints.end(), [](const auto& perMol) { return !perMol.empty(); });
    if (anyConstraint) {
      throw std::invalid_argument(
        "Device input coordinates not supported with custom constraints. "
        "Use the RDKit Mol + Conformer path to apply constraints "
        "(call MMFFMinimizeMoleculesConfs without deviceInput; the constraints anchor positions are "
        "read from each mol's RDKit conformer at force-field construction time).");
    }
  }

  const bool deviceOutput = output == CoordinateOutput::DEVICE;

  auto ctx = setupBatchExecution(perfOptions);

  if (deviceOutput) {
    if (targetGpu < 0) {
      targetGpu = ctx.devicesPerThread.empty() ? 0 : ctx.devicesPerThread.front();
    }
    if (std::find(ctx.devicesPerThread.begin(), ctx.devicesPerThread.end(), targetGpu) == ctx.devicesPerThread.end()) {
      throw std::invalid_argument(
        "targetGpu " + std::to_string(targetGpu) +
        " is not in the configured set of execution GPUs; pass it via perfOptions.gpuIds first.");
    }
  }

  std::vector<std::vector<double>> moleculeEnergies;
  const auto                       allConformers = flattenConformers(mols, moleculeEnergies);

  std::vector<std::vector<int8_t>> moleculeConverged(mols.size());
  for (size_t i = 0; i < mols.size(); ++i) {
    moleculeConverged[i].resize(moleculeEnergies[i].size(), 0);
  }

  // Build & validate the device-input index up front so per-batch broadcasts can do an O(1)
  // source lookup for each batch slot. The validation also catches mismatched conformer
  // counts/labels before we start any GPU work.
  detail::DeviceInputIndex deviceInputIndex;
  if (deviceInput != nullptr) {
    deviceInputIndex = detail::buildDeviceInputIndex(*deviceInput, allConformers);
  }
  const bool useDeviceInput = deviceInput != nullptr;

  const size_t totalConformers    = allConformers.size();
  const size_t effectiveBatchSize = (ctx.batchSize == 0) ? totalConformers : ctx.batchSize;

  if (totalConformers == 0) {
    if (deviceOutput) {
      std::vector<detail::DeviceCoordCollector> emptyCollectors;
      return {{}, {}, detail::finalizeOnTarget(emptyCollectors, targetGpu, static_cast<int>(mols.size()))};
    }
    return {moleculeEnergies, moleculeConverged, std::nullopt};
  }

  std::vector<ThreadLocalBuffers>           threadBuffers(ctx.numThreads);
  std::vector<detail::DeviceCoordCollector> deviceCollectors(deviceOutput ? ctx.numThreads : 0);
  if (deviceOutput) {
    for (int threadId = 0; threadId < ctx.numThreads; ++threadId) {
      auto& collector  = deviceCollectors[threadId];
      collector.gpuId  = ctx.devicesPerThread[threadId];
      collector.stream = ctx.streamPool[threadId].stream();
      collector.positions.setStream(collector.stream);
      collector.energies.setStream(collector.stream);
      collector.converged.setStream(collector.stream);
    }
  }
  detail::OpenMPExceptionRegistry exceptionHandler;
#pragma omp parallel for num_threads(ctx.numThreads) schedule(dynamic) default(none) shared(allConformers,        \
                                                                                              moleculeEnergies,   \
                                                                                              moleculeConverged,  \
                                                                                              totalConformers,    \
                                                                                              effectiveBatchSize, \
                                                                                              maxIters,           \
                                                                                              gradTol,            \
                                                                                              properties,         \
                                                                                              constraints,        \
                                                                                              ctx,                \
                                                                                              threadBuffers,      \
                                                                                              deviceCollectors,   \
                                                                                              deviceOutput,       \
                                                                                              useDeviceInput,     \
                                                                                              deviceInput,        \
                                                                                              deviceInputIndex,   \
                                                                                              backend,            \
                                                                                              exceptionHandler)
  for (size_t batchStart = 0; batchStart < totalConformers; batchStart += effectiveBatchSize) {
    try {
      std::unordered_map<RDKit::ROMol*, CachedMoleculeData> moleculeCache;
      ScopedNvtxRange                                       singleBatchRange("OpenMP loop thread");
      ScopedNvtxRange                                       setupBatchRange("OpenMP loop preprocessing");
      const int                                             threadId     = omp_get_thread_num();
      const int                                             executingGpu = ctx.devicesPerThread[threadId];
      const WithDevice                                      dev(executingGpu);
      const size_t batchEnd = std::min(batchStart + effectiveBatchSize, totalConformers);

      std::vector<nvMolKit::ConformerInfo> batchConformers(allConformers.begin() + batchStart,
                                                           allConformers.begin() + batchEnd);

      // Precompute per-batch source indices and atom counts when we'll broadcast device input.
      std::vector<int> batchSrcIndices;
      std::vector<int> batchAtomCounts;
      if (useDeviceInput) {
        batchSrcIndices.reserve(batchConformers.size());
        batchAtomCounts.reserve(batchConformers.size());
        for (size_t k = 0; k < batchConformers.size(); ++k) {
          batchSrcIndices.push_back(deviceInputIndex.conformerIndexBy[batchStart + k]);
          batchAtomCounts.push_back(static_cast<int>(batchConformers[k].mol->getNumAtoms()));
        }
      }

      cudaStream_t streamPtr = ctx.streamPool[threadId].stream();

      BatchedMolecularSystemHost    systemHost;
      BatchedMolecularDeviceBuffers systemDevice;
      BatchedForcefieldMetadata     metadata;
      std::vector<double>           pos;
      std::vector<uint32_t>         conformerAtomStarts;
      uint32_t                      currentAtomOffset = 0;

      for (const auto& confInfo : batchConformers) {
        auto*          mol      = confInfo.mol;
        const uint32_t numAtoms = mol->getNumAtoms();

        auto it = moleculeCache.find(mol);
        if (it == moleculeCache.end()) {
          ScopedNvtxRange    computeCacheRange("Preprocess single molecule");
          CachedMoleculeData cached;
          cached.ffParams = constructForcefieldContribs(*mol, properties[confInfo.molIdx]);
          it              = moleculeCache.insert({mol, std::move(cached)}).first;
        }

        ScopedNvtxRange addToBatchRange("Add conformer to batch data");
        conformerAtomStarts.push_back(currentAtomOffset);
        currentAtomOffset += numAtoms;

        nvMolKit::confPosToVect(*confInfo.conformer, pos);

        auto contribs = it->second.ffParams;
        if (!constraints.empty()) {
          constraints[confInfo.molIdx].applyTo(contribs, pos);
        }
        nvMolKit::MMFF::addMoleculeToBatch(contribs, pos, systemHost, &metadata, confInfo.molIdx, confInfo.confIdx);
      }

      auto& buffers = threadBuffers[threadId];
      buffers.ensureCapacity(systemHost.positions.size(), batchConformers.size());
      std::copy(systemHost.positions.begin(), systemHost.positions.end(), buffers.initialPositions.begin());

      nvMolKit::BfgsBatchMinimizer bfgsMinimizer(/*dataDim=*/3, nvMolKit::DebugLevel::NONE, true, streamPtr, backend);
      const auto                   effectiveBackend = bfgsMinimizer.resolveBackend(systemHost.indices.atomStarts);
      setupBatchRange.pop();

      const AsyncDeviceVector<double>* finalPositions = nullptr;
      const AsyncDeviceVector<double>* finalEnergies  = nullptr;
      AsyncDeviceVector<double>        positionsDevice;
      AsyncDeviceVector<double>        gradDevice;
      AsyncDeviceVector<double>        energyOutsDevice;

      if (effectiveBackend == BfgsBackend::BATCHED) {
        MMFFBatchedForcefield forcefield(systemHost, metadata, streamPtr);
        positionsDevice.setStream(streamPtr);
        gradDevice.setStream(streamPtr);
        energyOutsDevice.setStream(streamPtr);
        positionsDevice.resize(systemHost.positions.size());
        positionsDevice.copyFromHost(buffers.initialPositions.data(), systemHost.positions.size());
        if (useDeviceInput) {
          detail::broadcastDeviceInputBatch(*deviceInput,
                                            deviceInputIndex,
                                            batchSrcIndices,
                                            batchAtomCounts,
                                            executingGpu,
                                            streamPtr,
                                            positionsDevice);
        }
        gradDevice.resize(systemHost.positions.size());
        gradDevice.zero();
        energyOutsDevice.resize(batchConformers.size());
        energyOutsDevice.zero();

        bfgsMinimizer.minimize(maxIters, gradTol, forcefield, positionsDevice, gradDevice, energyOutsDevice);

        finalPositions = &positionsDevice;
        finalEnergies  = &energyOutsDevice;

        if (!deviceOutput) {
          ScopedNvtxRange finalizeBatchRange("OpenMP loop finalizing batch");
          positionsDevice.copyToHost(buffers.positions.data(), positionsDevice.size());
          energyOutsDevice.copyToHost(buffers.energies.data(), energyOutsDevice.size());
          cudaStreamSynchronize(streamPtr);
        }
      } else {
        nvMolKit::MMFF::setStreams(systemDevice, streamPtr);
        nvMolKit::MMFF::sendContribsAndIndicesToDevice(systemHost, systemDevice);
        nvMolKit::MMFF::allocateIntermediateBuffers(systemHost, systemDevice);
        systemDevice.positions.resize(systemHost.positions.size());
        systemDevice.positions.copyFromHost(buffers.initialPositions.data(), systemHost.positions.size());
        if (useDeviceInput) {
          detail::broadcastDeviceInputBatch(*deviceInput,
                                            deviceInputIndex,
                                            batchSrcIndices,
                                            batchAtomCounts,
                                            executingGpu,
                                            streamPtr,
                                            systemDevice.positions);
        }
        systemDevice.grad.resize(systemHost.positions.size());
        systemDevice.grad.zero();

        bfgsMinimizer.minimizeWithMMFF(maxIters, gradTol, systemHost.indices.atomStarts, systemDevice);

        finalPositions = &systemDevice.positions;
        finalEnergies  = &systemDevice.energyOuts;

        if (!deviceOutput) {
          ScopedNvtxRange finalizeBatchRange("OpenMP loop finalizing batch");
          systemDevice.positions.copyToHost(buffers.positions.data(), systemDevice.positions.size());
          systemDevice.energyOuts.copyToHost(buffers.energies.data(), systemDevice.energyOuts.size());
          cudaStreamSynchronize(streamPtr);
        }
      }

      if (deviceOutput) {
        detail::appendBatch(batchConformers,
                            *finalPositions,
                            *finalEnergies,
                            bfgsMinimizer.statuses_,
                            deviceCollectors[threadId]);
      } else {
        std::vector<int16_t> statusesHost(batchConformers.size());
        bfgsMinimizer.statuses_.copyToHost(statusesHost.data(), batchConformers.size());
        cudaStreamSynchronize(streamPtr);

        writeBackResults(batchConformers, conformerAtomStarts, buffers, moleculeEnergies);

        for (size_t i = 0; i < batchConformers.size(); ++i) {
          const auto& confInfo                                 = batchConformers[i];
          moleculeConverged[confInfo.molIdx][confInfo.confIdx] = static_cast<int8_t>(statusesHost[i] == 0);
        }
      }
    } catch (...) {
      exceptionHandler.store(std::current_exception());
    }
  }
  exceptionHandler.rethrow();
  if (deviceOutput) {
    return {{}, {}, detail::finalizeOnTarget(deviceCollectors, targetGpu, static_cast<int>(mols.size()))};
  }
  return {moleculeEnergies, moleculeConverged, std::nullopt};
}

std::vector<std::vector<double>> MMFFOptimizeMoleculesConfsBfgs(std::vector<RDKit::ROMol*>&        mols,
                                                                const int                          maxIters,
                                                                const std::vector<MMFFProperties>& properties,
                                                                const BatchHardwareOptions&        perfOptions,
                                                                const BfgsBackend                  backend) {
  return MMFFMinimizeMoleculesConfs(mols, maxIters, 1e-4, properties, {}, perfOptions, backend).energies;
}

}  // namespace nvMolKit::MMFF
