// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "rdkit_extensions/dist_geom_flattened_builder.h"
#include "src/etkdg_stage_aio_minimization.h"
#include "src/forcefields/aio_etkdg_batched_forcefield.h"
#include "src/forcefields/dist_geom_kernels_device_dispatch.cuh"
#include "src/utils/cuda_error_check.h"

namespace nvMolKit::detail {
namespace {

constexpr int kBlockSize = 256;

//! Force constant of the linear angle terms in RDKit's construct3DImproperForceField.
constexpr double kLinearAngleCheckForceConstant = 10.0;
//! Per-term and per-center limits of RDKit's _checkKTerms.
constexpr double kMaxLinearAngleEnergy          = 0.05;
constexpr double kMaxImproperEnergy             = 5.0;
constexpr double kPlanarityTolerance            = 0.7;

__global__ void checkKAngleTermsKernel(const int                                            numTerms,
                                       const DistGeom::AngleConstraintContribTermsDevicePtr terms,
                                       const int*                                           atomIdxToBatchIdx,
                                       const double*                                        positions,
                                       const uint8_t*                                       activeSystems,
                                       double*                                              systemEnergies,
                                       uint8_t*                                             failedSystems) {
  const int termIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (termIdx >= numTerms) {
    return;
  }
  const int systemIdx = atomIdxToBatchIdx[terms.idx2[termIdx]];
  if (!activeSystems[systemIdx]) {
    return;
  }
  const double energy = DistGeom::angleConstraintEnergy(positions,
                                                        terms.idx1[termIdx],
                                                        terms.idx2[termIdx],
                                                        terms.idx3[termIdx],
                                                        terms.minAngle[termIdx],
                                                        terms.maxAngle[termIdx],
                                                        kLinearAngleCheckForceConstant);
  atomicAdd(&systemEnergies[systemIdx], energy);
  if (energy > kMaxLinearAngleEnergy) {
    failedSystems[systemIdx] = 1;
  }
}

__global__ void checkKImproperTermsKernel(const int                                      numTerms,
                                          const DistGeom::InversionContribTermsDevicePtr terms,
                                          const int*                                     atomIdxToBatchIdx,
                                          const double*                                  positions,
                                          const uint8_t*                                 activeSystems,
                                          double*                                        systemEnergies,
                                          uint8_t*                                       failedSystems) {
  const int termIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (termIdx >= numTerms) {
    return;
  }
  const int systemIdx = atomIdxToBatchIdx[terms.idx2[termIdx]];
  if (!activeSystems[systemIdx]) {
    return;
  }
  const double energy = DistGeom::inversionEnergy(positions,
                                                  terms.idx1[termIdx],
                                                  terms.idx2[termIdx],
                                                  terms.idx3[termIdx],
                                                  terms.idx4[termIdx],
                                                  terms.C0[termIdx],
                                                  terms.C1[termIdx],
                                                  terms.C2[termIdx],
                                                  terms.forceConstant[termIdx]);
  atomicAdd(&systemEnergies[systemIdx], energy);
  if (energy > kMaxImproperEnergy) {
    failedSystems[systemIdx] = 1;
  }
}

//! RDKit 2026.09 compares this total before summing it, so its check never fires; nvMolKit applies it as intended.
__global__ void checkKTotalEnergyKernel(const int      numSystems,
                                        const double*  energies,
                                        const int*     numCenters,
                                        const uint8_t* activeSystems,
                                        uint8_t*       failedSystems) {
  const int systemIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (systemIdx < numSystems && activeSystems[systemIdx] &&
      energies[systemIdx] > kPlanarityTolerance * numCenters[systemIdx]) {
    failedSystems[systemIdx] = 1;
  }
}

int numBlocks(const int numItems) {
  return (numItems + kBlockSize - 1) / kBlockSize;
}

}  // namespace

#if RDKIT_AIO_ETKDG_API
namespace {

//! RDKit's MAX_MINIMIZED_E_PER_ATOM_AIO.
constexpr double kMaxMinimizedEnergyPerAtom = 0.1;
//! RDKit repeats 100-iteration minimizations of the second phase at most this many times.
constexpr int    kMaxSecondPhaseCycles      = 8;

//! Fails systems whose energy per atom is not below the all-in-one limit, including non-finite energies.
__global__ void checkAllInOneEnergyKernel(const int      numSystems,
                                          const double*  energies,
                                          const int*     atomStarts,
                                          const uint8_t* activeSystems,
                                          uint8_t*       failedSystems) {
  const int systemIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (systemIdx >= numSystems || !activeSystems[systemIdx]) {
    return;
  }
  const int numAtoms = atomStarts[systemIdx + 1] - atomStarts[systemIdx];
  if (!(energies[systemIdx] / numAtoms < kMaxMinimizedEnergyPerAtom)) {
    failedSystems[systemIdx] = 1;
  }
}

}  // namespace

template <typename Real>
ETKDGAllInOneMinimizationStageT<Real>::ETKDGAllInOneMinimizationStageT(
  const std::vector<const RDKit::ROMol*>&                                       mols,
  const std::vector<EmbedArgs>&                                                 eargs,
  const RDKit::DGeomHelpers::EmbedParameters&                                   embedParams,
  const ETKDGContext&                                                           ctx,
  BfgsBatchMinimizerT<Real>&                                                    minimizer,
  const cudaStream_t                                                            stream,
  std::unordered_map<const RDKit::ROMol*, DistGeom::AllInOneForceContribsHost>* cache)
    : minimizer_(minimizer),
      optimizerForceTol_(embedParams.optimizerForceTol),
      stream_(stream) {
  if (mols.size() != eargs.size()) {
    throw std::invalid_argument("Number of molecules and embed args must be the same");
  }
  gradients_.setStream(stream);
  energies_.setStream(stream);
  firstPhaseIterations_.setStream(stream);
  if (eargs.empty()) {
    return;
  }
  weights_ = DistGeom::allInOneForceWeights(eargs.front().etkdgDetails);

  for (std::size_t systemIdx = 0; systemIdx < mols.size(); ++systemIdx) {
    const auto* mol           = mols[systemIdx];
    const auto& earg          = eargs[systemIdx];
    const auto  buildContribs = [&]() {
      return DistGeom::constructAllInOneForceFieldContribs(earg.dim,
                                                           *earg.mmat,
                                                           earg.chiralCenters,
                                                           earg.etkdgDetails,
                                                           embedParams.useExpTorsionAnglePrefs,
                                                           embedParams.useBasicKnowledge);
    };

    // Get or construct force field parameters
    const DistGeom::AllInOneForceContribsHost* contribs = nullptr;
    DistGeom::AllInOneForceContribsHost        uncachedContribs;
    if (cache != nullptr) {
      auto it = cache->find(mol);
      if (it == cache->end()) {
        it = cache->emplace(mol, buildContribs()).first;
      }
      contribs = &it->second;
    } else {
      uncachedContribs = buildContribs();
      contribs         = &uncachedContribs;
    }

    if (systemIdx == 0) {
      DistGeom::preallocateEstimatedBatch(contribs->distanceGeometry,
                                          systemHost_.distanceGeometry,
                                          static_cast<int>(mols.size()));
    }
    DistGeom::addMoleculeToAllInOneSystem(*contribs,
                                          static_cast<int>(mol->getNumAtoms()),
                                          ctx.systemHost.atomStarts,
                                          systemHost_,
                                          &metadata_,
                                          static_cast<int>(systemIdx),
                                          0);
    firstPhaseIterationsHost_.push_back(static_cast<int>(mol->getNumHeavyAtoms()));
  }
  firstPhaseIterations_.setFromVector(firstPhaseIterationsHost_);
}

template <typename Real> void ETKDGAllInOneMinimizationStageT<Real>::execute(ETKDGContext& ctx) {
  if (ctx.nTotalSystems == 0) {
    return;
  }
  AllInOneETKDGBatchedForcefield forcefield(systemHost_,
                                            ctx.systemHost.atomStarts,
                                            weights_,
                                            metadata_,
                                            stream_,
                                            BfgsBatchMinimizerT<Real>::kPrecision);
  gradients_.resize(ctx.systemDevice.positions.size());
  energies_.resize(ctx.systemHost.atomStarts.size() - 1);

  forcefield.setTorsionTermsEnabled(false);
  minimizer_.minimize(*std::max_element(firstPhaseIterationsHost_.begin(), firstPhaseIterationsHost_.end()),
                      optimizerForceTol_,
                      forcefield,
                      ctx.systemDevice.positions,
                      gradients_,
                      energies_,
                      ctx.activeThisStage.data(),
                      firstPhaseIterations_.data());

  forcefield.setTorsionTermsEnabled(true);
  for (int cycle = 0; cycle < kMaxSecondPhaseCycles; ++cycle) {
    const bool needsMore = minimizer_.minimize(100,
                                               optimizerForceTol_,
                                               forcefield,
                                               ctx.systemDevice.positions,
                                               gradients_,
                                               energies_,
                                               ctx.activeThisStage.data());
    if (!needsMore) {
      break;
    }
  }

  energies_.zero();
  cudaCheckError(
    forcefield.computeEnergy(energies_.data(), ctx.systemDevice.positions.data(), ctx.activeThisStage.data(), stream_));
  const int numSystems = energies_.size();
  checkAllInOneEnergyKernel<<<numBlocks(numSystems), kBlockSize, 0, stream_>>>(numSystems,
                                                                               energies_.data(),
                                                                               ctx.systemDevice.atomStarts.data(),
                                                                               ctx.activeThisStage.data(),
                                                                               ctx.failedThisStage.data());
  cudaCheckError(cudaGetLastError());
}

template class ETKDGAllInOneMinimizationStageT<double>;
template class ETKDGAllInOneMinimizationStageT<float>;
#endif

ETKDGKTermCheckStage::ETKDGKTermCheckStage(const std::vector<EmbedArgs>& eargs,
                                           const ETKDGContext&           ctx,
                                           const cudaStream_t            stream)
    : stream_(stream) {
  DistGeom::BatchedMolecularSystem3DHost systemHost;
  std::vector<int>                       numCenters;
  for (const auto& earg : eargs) {
    const auto& details = earg.etkdgDetails;
    DistGeom::addMoleculeToMolecularSystem3D(DistGeom::constructImproperCheckContribs(details, earg.mmat->numRows()),
                                             ctx.systemHost.atomStarts,
                                             systemHost);
    const auto numLinearCenters =
      std::count_if(details.angles.begin(), details.angles.end(), [](const auto& angle) { return angle[3] != 0; });
    numCenters.push_back(static_cast<int>(numLinearCenters + details.improperAtoms.size()));
  }
  DistGeom::setStreams(systemDevice_, stream);
  DistGeom::sendContribsAndIndicesToDevice3D(systemHost, systemDevice_);
  numCenters_.setStream(stream);
  numCenters_.setFromVector(numCenters);
  systemEnergies_.setStream(stream);
  cudaCheckError(cudaStreamSynchronize(stream));
}

void ETKDGKTermCheckStage::execute(ETKDGContext& ctx) {
  if (ctx.nTotalSystems == 0) {
    return;
  }
  systemEnergies_.resize(ctx.nTotalSystems);
  systemEnergies_.zero();
  const auto  terms             = DistGeom::toEnergy3DForceContribsDevicePtr(systemDevice_);
  const int*  atomIdxToBatchIdx = systemDevice_.indices.atomIdxToBatchIdx.data();
  const auto& contribs          = systemDevice_.contribs;

  const int numAngles = contribs.angle13Terms.idx1.size();
  if (numAngles > 0) {
    checkKAngleTermsKernel<<<numBlocks(numAngles), kBlockSize, 0, stream_>>>(numAngles,
                                                                             terms.angle13Terms,
                                                                             atomIdxToBatchIdx,
                                                                             ctx.systemDevice.positions.data(),
                                                                             ctx.activeThisStage.data(),
                                                                             systemEnergies_.data(),
                                                                             ctx.failedThisStage.data());
    cudaCheckError(cudaGetLastError());
  }
  const int numImpropers = contribs.improperTorsionTerms.idx1.size();
  if (numImpropers > 0) {
    checkKImproperTermsKernel<<<numBlocks(numImpropers), kBlockSize, 0, stream_>>>(numImpropers,
                                                                                   terms.improperTorsionTerms,
                                                                                   atomIdxToBatchIdx,
                                                                                   ctx.systemDevice.positions.data(),
                                                                                   ctx.activeThisStage.data(),
                                                                                   systemEnergies_.data(),
                                                                                   ctx.failedThisStage.data());
    cudaCheckError(cudaGetLastError());
  }
  checkKTotalEnergyKernel<<<numBlocks(ctx.nTotalSystems), kBlockSize, 0, stream_>>>(ctx.nTotalSystems,
                                                                                    systemEnergies_.data(),
                                                                                    numCenters_.data(),
                                                                                    ctx.activeThisStage.data(),
                                                                                    ctx.failedThisStage.data());
  cudaCheckError(cudaGetLastError());
}

}  // namespace nvMolKit::detail
