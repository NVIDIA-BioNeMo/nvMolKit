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

#include "src/minimizer/bfgs_mmff.h"

#include <GraphMol/ROMol.h>
#include <omp.h>

#include <unordered_map>

#include "src/minimizer/bfgs_common.h"
#include "src/minimizer/bfgs_minimize.h"
#include "src/forcefields/ff_utils.h"
#include "src/forcefields/mmff_batched_forcefield.h"
#include "rdkit_extensions/mmff_flattened_builder.h"
#include "src/utils/nvtx.h"
#include "src/utils/openmp_helpers.h"

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
                                              const BfgsBackend                                            backend) {
  ScopedNvtxRange fullRange("BFGS MMFF Minimize Molecules Confs");

  if (properties.size() != mols.size()) {
    throw std::invalid_argument("Expected one MMFFProperties entry per molecule");
  }
  if (!constraints.empty() && constraints.size() != mols.size()) {
    throw std::invalid_argument("Expected one PerMolConstraints entry per molecule");
  }

  auto                             ctx = setupBatchExecution(perfOptions);
  std::vector<std::vector<double>> moleculeEnergies;
  const auto                       allConformers = flattenConformers(mols, moleculeEnergies);

  std::vector<std::vector<int8_t>> moleculeConverged(mols.size());
  for (size_t i = 0; i < mols.size(); ++i) {
    moleculeConverged[i].resize(moleculeEnergies[i].size(), 0);
  }

  const size_t totalConformers    = allConformers.size();
  const size_t effectiveBatchSize = (ctx.batchSize == 0) ? totalConformers : ctx.batchSize;

  if (totalConformers == 0) {
    return {moleculeEnergies, moleculeConverged};
  }

  std::vector<ThreadLocalBuffers> threadBuffers(ctx.numThreads);
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
                                                                                              backend,            \
                                                                                              exceptionHandler)
  for (size_t batchStart = 0; batchStart < totalConformers; batchStart += effectiveBatchSize) {
    try {
      std::unordered_map<RDKit::ROMol*, CachedMoleculeData> moleculeCache;
      ScopedNvtxRange                                       singleBatchRange("OpenMP loop thread");
      ScopedNvtxRange                                       setupBatchRange("OpenMP loop preprocessing");
      const int                                             threadId = omp_get_thread_num();
      const WithDevice                                      dev(ctx.devicesPerThread[threadId]);
      const size_t batchEnd = std::min(batchStart + effectiveBatchSize, totalConformers);

      std::vector<nvMolKit::ConformerInfo> batchConformers(allConformers.begin() + batchStart,
                                                           allConformers.begin() + batchEnd);

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

      if (effectiveBackend == BfgsBackend::BATCHED) {
        MMFFBatchedForcefield     forcefield(systemHost, metadata, streamPtr);
        AsyncDeviceVector<double> positionsDevice;
        AsyncDeviceVector<double> gradDevice;
        AsyncDeviceVector<double> energyOutsDevice;
        positionsDevice.setStream(streamPtr);
        gradDevice.setStream(streamPtr);
        energyOutsDevice.setStream(streamPtr);
        positionsDevice.resize(systemHost.positions.size());
        positionsDevice.copyFromHost(buffers.initialPositions.data(), systemHost.positions.size());
        gradDevice.resize(systemHost.positions.size());
        gradDevice.zero();
        energyOutsDevice.resize(batchConformers.size());
        energyOutsDevice.zero();

        bfgsMinimizer.minimize(maxIters, gradTol, forcefield, positionsDevice, gradDevice, energyOutsDevice);

        ScopedNvtxRange finalizeBatchRange("OpenMP loop finalizing batch");
        positionsDevice.copyToHost(buffers.positions.data(), positionsDevice.size());
        energyOutsDevice.copyToHost(buffers.energies.data(), energyOutsDevice.size());
        cudaStreamSynchronize(streamPtr);
      } else {
        nvMolKit::MMFF::sendContribsAndIndicesToDevice(systemHost, systemDevice);
        nvMolKit::MMFF::setStreams(systemDevice, streamPtr);
        nvMolKit::MMFF::allocateIntermediateBuffers(systemHost, systemDevice);
        systemDevice.positions.resize(systemHost.positions.size());
        systemDevice.positions.copyFromHost(buffers.initialPositions.data(), systemHost.positions.size());
        systemDevice.grad.resize(systemHost.positions.size());
        systemDevice.grad.zero();

        bfgsMinimizer.minimizeWithMMFF(maxIters, gradTol, systemHost.indices.atomStarts, systemDevice);

        ScopedNvtxRange finalizeBatchRange("OpenMP loop finalizing batch");
        systemDevice.positions.copyToHost(buffers.positions.data(), systemDevice.positions.size());
        systemDevice.energyOuts.copyToHost(buffers.energies.data(), systemDevice.energyOuts.size());
        cudaStreamSynchronize(streamPtr);
      }

      std::vector<int16_t> statusesHost(batchConformers.size());
      bfgsMinimizer.statuses_.copyToHost(statusesHost.data(), batchConformers.size());
      cudaStreamSynchronize(streamPtr);

      writeBackResults(batchConformers, conformerAtomStarts, buffers, moleculeEnergies);

      for (size_t i = 0; i < batchConformers.size(); ++i) {
        const auto& confInfo                                 = batchConformers[i];
        moleculeConverged[confInfo.molIdx][confInfo.confIdx] = static_cast<int8_t>(statusesHost[i] == 0);
      }
    } catch (...) {
      exceptionHandler.store(std::current_exception());
    }
  }
  exceptionHandler.rethrow();
  return {moleculeEnergies, moleculeConverged};
}

std::vector<std::vector<double>> MMFFOptimizeMoleculesConfsBfgs(std::vector<RDKit::ROMol*>&        mols,
                                                                const int                          maxIters,
                                                                const std::vector<MMFFProperties>& properties,
                                                                const BatchHardwareOptions&        perfOptions,
                                                                const BfgsBackend                  backend) {
  return MMFFMinimizeMoleculesConfs(mols, maxIters, 1e-4, properties, {}, perfOptions, backend).energies;
}

}  // namespace nvMolKit::MMFF
