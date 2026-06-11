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

#include <optional>
#include <unordered_map>

#include "rdkit_extensions/dist_geom_flattened_builder.h"
#include "src/etkdg_stage_etk_minimization.h"
#include "src/forcefields/etk_batched_forcefield.h"
#include "src/minimizer/bfgs_minimize.h"

namespace nvMolKit {
namespace detail {

constexpr int dim = 4;

namespace {

// TODO: Only run on active systems.
__global__ void updateReferencePositionsKernel(const int      numTerms,
                                               const double*  refPos,
                                               const int*     idx1,
                                               const int*     idx2,
                                               double*        lowerBound,
                                               double*        upperBound,
                                               const uint8_t* isImproperConstrainedTerm = nullptr) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numTerms) {
    if (isImproperConstrainedTerm != nullptr && isImproperConstrainedTerm[idx]) {
      // Skip improper constraints.
      return;
    }
    const int    i1  = idx1[idx];
    const int    i2  = idx2[idx];
    const double p1x = refPos[dim * i1];
    const double p1y = refPos[dim * i1 + 1];
    const double p1z = refPos[dim * i1 + 2];
    const double p2x = refPos[dim * i2];
    const double p2y = refPos[dim * i2 + 1];
    const double p2z = refPos[dim * i2 + 2];

    // For long distance, the constraint can be tighter. Get it from the previous bounds rather than constants.
    const double lowerBoundValue = lowerBound[idx];
    const double upperBoundValue = upperBound[idx];
    const double boundDelta      = (upperBoundValue - lowerBoundValue) / 2.0;

    const double dist = sqrt((p1x - p2x) * (p1x - p2x) + (p1y - p2y) * (p1y - p2y) + (p1z - p2z) * (p1z - p2z));

    lowerBound[idx] = dist - boundDelta;
    upperBound[idx] = dist + boundDelta;
  }
}

__global__ void planarToleranceCheck(const int      numSystems,
                                     const double*  energies,
                                     const int*     numImpropers,
                                     const uint8_t* activeThisStage,
                                     uint8_t*       failedThisStage) {
  const int sysIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (sysIdx >= numSystems) {
    return;
  }
  if (!activeThisStage[sysIdx]) {
    return;  // Skip inactive systems.
  }
  const int        numTermsForSystem = numImpropers[sysIdx];
  constexpr double toleranceFactor   = 0.7;
  const double     tolerance         = toleranceFactor * numTermsForSystem;
  const double     e                 = energies[sysIdx];

  if (e > tolerance) {
    // If the energy is too high, mark the system as failed.
    failedThisStage[sysIdx] = 1;
  }
}

void runPlanarToleranceCheck(const AsyncDeviceVector<double>& planarEnergies,
                             const int*                       numImpropers,
                             const ETKDGContext&              ctx,
                             cudaStream_t                     stream) {
  const int numSystems = ctx.systemHost.atomStarts.size() - 1;
  planarToleranceCheck<<<(numSystems + 255) / 256, 256, 0, stream>>>(numSystems,
                                                                     planarEnergies.data(),
                                                                     numImpropers,
                                                                     ctx.activeThisStage.data(),
                                                                     ctx.failedThisStage.data());
  cudaCheckError(cudaGetLastError());
}

}  // namespace

ETKMinimizationStage::ETKMinimizationStage(
  const std::vector<const RDKit::ROMol*>&                                                 mols,
  const std::vector<EmbedArgs>&                                                           eargs,
  const RDKit::DGeomHelpers::EmbedParameters&                                             embedParam,
  const ETKDGContext&                                                                     ctx,
  BfgsBatchMinimizer&                                                                     minimizer,
  cudaStream_t                                                                            stream,
  std::unordered_map<const RDKit::ROMol*, nvMolKit::DistGeom::Energy3DForceContribsHost>* cache)
    : embedParam_(embedParam),
      minimizer_(minimizer),
      stream_(stream) {
  grad_.setStream(stream);
  energyOuts_.setStream(stream);

  const int totalNumAtoms = ctx.systemHost.atomStarts.back();

  std::vector<double> positions(totalNumAtoms * dim, 0.0);

  bool                                         preallocated = false;
  std::unordered_map<const RDKit::ROMol*, int> moleculeSlots;
  std::unordered_map<const RDKit::ROMol*, int> conformerCounts;
  for (size_t i = 0; i < mols.size(); ++i) {
    const auto& mol          = mols[i];
    const auto& etkdgDetails = eargs[i].etkdgDetails;
    const auto& mmat         = eargs[i].mmat;

    // Get or construct force field parameters
    const nvMolKit::DistGeom::Energy3DForceContribsHost* ffParams = nullptr;
    nvMolKit::DistGeom::Energy3DForceContribsHost        uncachedParams;

    if (cache != nullptr) {
      auto it = cache->find(mol);
      if (it != cache->end()) {
        ffParams = &it->second;
      } else {
        // Construct directly into cache
        auto result = cache->emplace(mol,
                                     nvMolKit::DistGeom::construct3DForceFieldContribs(*mmat,
                                                                                       etkdgDetails,
                                                                                       positions,
                                                                                       /*dim=*/3,
                                                                                       embedParam.useBasicKnowledge));
        ffParams    = &result.first->second;
      }
    } else {
      // No cache, construct locally
      uncachedParams = nvMolKit::DistGeom::construct3DForceFieldContribs(*mmat,
                                                                         etkdgDetails,
                                                                         positions,
                                                                         /*dim=*/3,
                                                                         embedParam.useBasicKnowledge);
      ffParams       = &uncachedParams;
    }

    // Preallocate once using the first molecule's parameters
    if (!preallocated) {
      nvMolKit::DistGeom::preallocateEstimatedBatch3D(*ffParams, molSystemHost, static_cast<int>(mols.size()));
      preallocated = true;
    }

    auto [slotIt, inserted] = moleculeSlots.emplace(mol, static_cast<int>(moleculeSlots.size()));
    const int moleculeIdx   = slotIt->second;
    const int conformerIdx  = conformerCounts[mol]++;
    addMoleculeToMolecularSystem3D(*ffParams,
                                   ctx.systemHost.atomStarts,
                                   molSystemHost,
                                   &metadata_,
                                   moleculeIdx,
                                   conformerIdx);
  }
}

void ETKMinimizationStage::setReferenceValues(const ETKDGContext&                          ctx,
                                              const DistGeom::Energy3DForceContribsDevice& contribs) {
  const int numTerms12 = contribs.dist12Terms.idx1.size();
  const int numTerms13 = contribs.dist13Terms.idx1.size();

  if (numTerms12 > 0) {
    updateReferencePositionsKernel<<<(numTerms12 + 255) / 256, 256, 0, stream_>>>(numTerms12,
                                                                                  ctx.systemDevice.positions.data(),
                                                                                  contribs.dist12Terms.idx1.data(),
                                                                                  contribs.dist12Terms.idx2.data(),
                                                                                  contribs.dist12Terms.minLen.data(),
                                                                                  contribs.dist12Terms.maxLen.data());
    cudaCheckError(cudaGetLastError());
  }
  if (numTerms13 > 0) {
    updateReferencePositionsKernel<<<(numTerms13 + 255) / 256, 256, 0, stream_>>>(
      numTerms13,
      ctx.systemDevice.positions.data(),
      contribs.dist13Terms.idx1.data(),
      contribs.dist13Terms.idx2.data(),
      contribs.dist13Terms.minLen.data(),
      contribs.dist13Terms.maxLen.data(),
      contribs.dist13Terms.isImproperConstrained.data());

    cudaCheckError(cudaGetLastError());
  }
}

void ETKMinimizationStage::execute(ETKDGContext& ctx) {
  const auto effectiveBackend = minimizer_.resolveBackend(ctx.systemHost.atomStarts);

  // 1. Update reference positions for start of loop.
  constexpr int                             maxIters = 300;  // Taken from hard-coded RDKit value.
  DistGeom::BatchedMolecular3DDeviceBuffers molSystemDevice;
  std::optional<ETKBatchedForcefield>       forcefield;
  AsyncDeviceVector<double>*                planarEnergies = nullptr;
  const int*                                numImpropers   = nullptr;

  if (effectiveBackend == BfgsBackend::BATCHED) {
    forcefield.emplace(molSystemHost, ctx.systemHost.atomStarts, embedParam_.useBasicKnowledge, metadata_, stream_);
    setReferenceValues(ctx, forcefield->contribs());
    grad_.resize(ctx.systemHost.positions.size());
    grad_.zero();
    energyOuts_.resize(ctx.systemHost.atomStarts.size() - 1);
    energyOuts_.zero();
    minimizer_.minimize(maxIters,
                        embedParam_.optimizerForceTol,
                        *forcefield,
                        ctx.systemDevice.positions,
                        grad_,
                        energyOuts_,
                        ctx.activeThisStage.data());
    planarEnergies = &energyOuts_;
    numImpropers   = forcefield->contribs().improperTorsionTerms.numImpropers.data();
    if (embedParam_.useBasicKnowledge) {
      planarEnergies->zero();
      forcefield->computePlanarEnergy(planarEnergies->data(),
                                      ctx.systemDevice.positions.data(),
                                      ctx.activeThisStage.data(),
                                      stream_);
    }
  } else {
    setStreams(molSystemDevice, stream_);
    std::vector<double> positions(ctx.systemHost.atomStarts.back() * dim, 0.0);
    setupDeviceBuffers3D(molSystemHost, molSystemDevice, positions, ctx.systemHost.atomStarts.size() - 1);
    DistGeom::sendContribsAndIndicesToDevice3D(molSystemHost, molSystemDevice);
    setReferenceValues(ctx, molSystemDevice.contribs);

    minimizer_.minimizeWithETK(maxIters,
                               embedParam_.optimizerForceTol,
                               ctx.systemHost.atomStarts,
                               ctx.systemDevice.atomStarts,
                               ctx.systemDevice.positions,
                               molSystemDevice,
                               ctx.activeThisStage.data());
    planarEnergies = &molSystemDevice.energyOuts;
    numImpropers   = molSystemDevice.contribs.improperTorsionTerms.numImpropers.data();
    if (embedParam_.useBasicKnowledge) {
      DistGeom::computePlanarEnergy(molSystemDevice,
                                    ctx.systemDevice.atomStarts,
                                    ctx.systemDevice.positions,
                                    ctx.activeThisStage.data(),
                                    nullptr,
                                    stream_);
    }
  }

  if (embedParam_.useBasicKnowledge) {
    runPlanarToleranceCheck(*planarEnergies, numImpropers, ctx, stream_);
  }
}

}  // namespace detail
}  // namespace nvMolKit
