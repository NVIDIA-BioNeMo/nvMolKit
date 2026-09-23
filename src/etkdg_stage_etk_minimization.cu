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

#include <type_traits>
#include <unordered_map>

#include "rdkit_extensions/dist_geom_flattened_builder.h"
#include "src/etkdg_stage_etk_minimization.h"
#include "src/forcefields/etk_batched_forcefield.h"
#include "src/minimizer/bfgs_minimize.h"
#include "src/utils/device_convert.cuh"

namespace nvMolKit {
namespace detail {

constexpr int dim = 4;

namespace {

// TODO: Only run on active systems.
template <typename Scalar>
__global__ void updateReferencePositionsKernel(const int      numTerms,
                                               const double*  refPos,
                                               const int*     idx1,
                                               const int*     idx2,
                                               Scalar*        lowerBound,
                                               Scalar*        upperBound,
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

    lowerBound[idx] = static_cast<Scalar>(dist - boundDelta);
    upperBound[idx] = static_cast<Scalar>(dist + boundDelta);
  }
}

template <typename Scalar>
__global__ void planarToleranceCheck(const int      numSystems,
                                     const Scalar*  energies,
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
  const Scalar     e                 = energies[sysIdx];

  if (e > tolerance) {
    // If the energy is too high, mark the system as failed.
    failedThisStage[sysIdx] = 1;
  }
}

template <typename Scalar>
void runPlanarToleranceCheck(const AsyncDeviceVector<Scalar>& planarEnergies,
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

template <typename Scalar>
void ETKMinimizationStage::setReferenceValues(const ETKDGContext&                                   ctx,
                                              const DistGeom::Energy3DForceContribsDeviceT<Scalar>& contribs) {
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
  constexpr int maxIters = 300;  // Taken from hard-coded RDKit value.

  if (effectiveBackend == BfgsBackend::BATCHED) {
    ETKBatchedForcefield forcefield(molSystemHost,
                                    ctx.systemHost.atomStarts,
                                    embedParam_.useBasicKnowledge,
                                    metadata_,
                                    stream_,
                                    minimizer_.precision());
    const int*           numImpropers = nullptr;
    if (usesSinglePrecision(minimizer_.precision())) {
      setReferenceValues(ctx, forcefield.singleContribs());
      numImpropers = forcefield.singleContribs().improperTorsionTerms.numImpropers.data();
    } else {
      setReferenceValues(ctx, forcefield.contribs());
      numImpropers = forcefield.contribs().improperTorsionTerms.numImpropers.data();
    }
    grad_.resize(ctx.systemHost.positions.size());
    grad_.zero();
    energyOuts_.resize(ctx.systemHost.atomStarts.size() - 1);
    energyOuts_.zero();
    minimizer_.minimize(maxIters,
                        embedParam_.optimizerForceTol,
                        forcefield,
                        ctx.systemDevice.positions,
                        grad_,
                        energyOuts_,
                        ctx.activeThisStage.data());
    if (embedParam_.useBasicKnowledge) {
      energyOuts_.zero();
      forcefield.computePlanarEnergy(energyOuts_.data(),
                                     ctx.systemDevice.positions.data(),
                                     ctx.activeThisStage.data(),
                                     stream_);
      runPlanarToleranceCheck(energyOuts_, numImpropers, ctx, stream_);
    }
  } else {
    auto minimizePerMolecule = [&](auto& device, auto& positions) {
      using Scalar = std::remove_pointer_t<decltype(positions.data())>;
      DistGeom::setStreams(device, stream_);
      DistGeom::sendContribsAndIndicesToDevice3D(molSystemHost, device);
      if constexpr (std::is_same_v<Scalar, double>) {
        DistGeom::allocateIntermediateBuffers3D(molSystemHost, device);
      } else {
        device.energyOuts.resize(ctx.systemHost.atomStarts.size() - 1);
        device.energyOuts.zero();
      }
      device.grad.resize(positions.size());
      device.grad.zero();
      setReferenceValues(ctx, device.contribs);

      minimizer_.minimizeWithETK(maxIters,
                                 embedParam_.optimizerForceTol,
                                 ctx.systemHost.atomStarts,
                                 ctx.systemDevice.atomStarts,
                                 positions,
                                 device,
                                 ctx.activeThisStage.data());
      if (embedParam_.useBasicKnowledge) {
        device.energyOuts.zero();
        if constexpr (std::is_same_v<Scalar, float>) {
          cudaCheckError(DistGeom::launchPlanarEnergyKernelETK(
            static_cast<int>(ctx.systemHost.atomStarts.size() - 1),
            DistGeom::toEnergy3DForceContribsDevicePtr(device),
            DistGeom::toBatchedIndices3DDevicePtr(device, ctx.systemDevice.atomStarts.data()),
            positions.data(),
            device.energyOuts.data(),
            ctx.activeThisStage.data(),
            stream_));
        } else {
          cudaCheckError(DistGeom::computePlanarEnergy(device,
                                                       device.energyOuts.data(),
                                                       ctx.systemDevice.atomStarts.data(),
                                                       positions.data(),
                                                       ctx.activeThisStage.data(),
                                                       positions.data(),
                                                       stream_));
        }
        runPlanarToleranceCheck(device.energyOuts,
                                device.contribs.improperTorsionTerms.numImpropers.data(),
                                ctx,
                                stream_);
      }
    };

    if (usesSinglePrecision(minimizer_.precision())) {
      DistGeom::BatchedMolecular3DDeviceBuffersSingle device;
      AsyncDeviceVector<float>                        positions;
      positions.setStream(stream_);
      positions.resize(ctx.systemDevice.positions.size());
      cudaCheckError(nvMolKit::detail::convertDeviceArray(positions.data(),
                                                          ctx.systemDevice.positions.data(),
                                                          positions.size(),
                                                          stream_));
      minimizePerMolecule(device, positions);
      cudaCheckError(nvMolKit::detail::convertDeviceArray(ctx.systemDevice.positions.data(),
                                                          positions.data(),
                                                          positions.size(),
                                                          stream_));
    } else {
      DistGeom::BatchedMolecular3DDeviceBuffers device;
      minimizePerMolecule(device, ctx.systemDevice.positions);
    }
  }
}

}  // namespace detail
}  // namespace nvMolKit
