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

#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/FMCS/FMCS.h>
#include <GraphMol/RingInfo.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/Substruct/SubstructMatch.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "src/mcs/fmcs_cuda/fmcs.cuh"
#include "src/mcs/labeled_graph.h"
#include "src/mcs/mcs_search.h"
#include "src/utils/device.h"

namespace nvMolKit {
namespace {

using mcs::LabeledGraph;

constexpr int kMaxMCSExecutorsPerRunner = 8;

struct AtomLabelKey {
  int atomCompareValue = 0;
  int isotope          = 0;
  int totalValence     = 0;
  int formalCharge     = 0;
  int ringState        = 0;

  [[nodiscard]] auto tie() const { return std::tie(atomCompareValue, isotope, totalValence, formalCharge, ringState); }

  bool operator==(const AtomLabelKey& other) const { return tie() == other.tie(); }
  bool operator<(const AtomLabelKey& other) const { return tie() < other.tie(); }
};

struct PreparedGpuPair {
  size_t       resultIdx = 0;
  size_t       molIdxA   = 0;
  size_t       molIdxB   = 0;
  LabeledGraph graphA;
  LabeledGraph graphB;
};

void checkCuda(cudaError_t err, const char* context) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("MCS dispatch CUDA error at ") + context + ": " + cudaGetErrorString(err));
  }
}

std::vector<int> resolveGpuIds(const MCSParameters& params) {
  std::vector<int> gpuIds = params.gpuIds;
  if (gpuIds.empty()) {
    int currentDevice = 0;
    checkCuda(cudaGetDevice(&currentDevice), "cudaGetDevice");
    gpuIds.push_back(currentDevice);
  }
  return gpuIds;
}

void computeEffectiveThreadCounts(const MCSParameters& params,
                                  int                  numGpus,
                                  int&                 effectivePreprocessingThreads,
                                  int&                 effectiveWorkerThreads) {
  const int hwThreads        = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  const int effectiveNumGpus = std::max(1, numGpus);

  effectivePreprocessingThreads =
    (params.preprocessingThreads == -1) ? hwThreads : std::max(1, params.preprocessingThreads);
  effectiveWorkerThreads = (params.workerThreads == -1) ? std::min(4, std::max(1, hwThreads / effectiveNumGpus)) :
                                                          std::max(1, params.workerThreads);
}

int effectiveExecutorsPerRunner(const MCSParameters& params, int totalRunners, cudaStream_t stream) {
  if (params.executorsPerRunner == -1) {
    return stream != nullptr ? 1 : (totalRunners == 1 ? 3 : 2);
  }
  if (params.executorsPerRunner < 1 || params.executorsPerRunner > kMaxMCSExecutorsPerRunner) {
    throw std::invalid_argument("MCS executorsPerRunner must be -1 (auto) or between 1 and " +
                                std::to_string(kMaxMCSExecutorsPerRunner));
  }
  return params.executorsPerRunner;
}

struct BondLabelKey {
  int bondCompareValue = 0;
  int ringState        = 0;

  [[nodiscard]] auto tie() const { return std::tie(bondCompareValue, ringState); }

  bool operator<(const BondLabelKey& other) const { return tie() < other.tie(); }
};

template <typename Key> std::uint16_t internLabel(std::map<Key, std::uint16_t>& labels, const Key& key) {
  const auto found = labels.find(key);
  if (found != labels.end()) {
    return found->second;
  }
  if (labels.size() >= static_cast<size_t>(UINT16_MAX)) {
    throw std::runtime_error("MCS label table exceeded uint16_t capacity");
  }
  const auto label = static_cast<std::uint16_t>(labels.size() + 1);
  labels.emplace(key, label);
  return label;
}

bool usesAtomLabels(const MCSParameters& params) {
  return params.atomCompare != MCSAtomCompare::Any || params.atomCompareParameters.matchValences ||
         params.atomCompareParameters.matchFormalCharge || params.atomCompareParameters.ringMatchesRingOnly ||
         params.atomCompareParameters.matchIsotope;
}

bool usesBondLabels(const MCSParameters& params) {
  return params.bondCompare != MCSBondCompare::Any || params.bondCompareParameters.ringMatchesRingOnly;
}

AtomLabelKey makeAtomLabelKey(const RDKit::ROMol& mol, const RDKit::Atom& atom, const MCSParameters& params) {
  AtomLabelKey key;
  switch (params.atomCompare) {
    case MCSAtomCompare::Any:
      key.atomCompareValue = 0;
      break;
    case MCSAtomCompare::Elements:
      key.atomCompareValue = atom.getAtomicNum();
      break;
    case MCSAtomCompare::Isotopes:
      key.atomCompareValue = static_cast<int>(atom.getIsotope());
      break;
    case MCSAtomCompare::AnyHeavyAtom:
      key.atomCompareValue = atom.getAtomicNum() == 1 ? 1 : 2;
      break;
  }

  if (params.atomCompareParameters.matchIsotope) {
    key.isotope = static_cast<int>(atom.getIsotope());
  }
  if (params.atomCompareParameters.matchValences) {
    key.totalValence = static_cast<int>(atom.getTotalValence());
  }
  if (params.atomCompareParameters.matchFormalCharge) {
    key.formalCharge = atom.getFormalCharge();
  }
  if (params.atomCompareParameters.ringMatchesRingOnly) {
    key.ringState = mol.getRingInfo()->numAtomRings(atom.getIdx()) > 0 ? 1 : 2;
  }
  return key;
}

int bondOrderClass(const RDKit::Bond& bond, const MCSParameters& params) {
  switch (params.bondCompare) {
    case MCSBondCompare::Any:
      return 0;
    case MCSBondCompare::Order: {
      const auto bondType = static_cast<int>(bond.getBondType());
      if (bondType == static_cast<int>(RDKit::Bond::SINGLE) || bondType == static_cast<int>(RDKit::Bond::AROMATIC)) {
        return 1;
      }
      return bondType;
    }
    case MCSBondCompare::OrderExact: {
      const auto bondType = static_cast<int>(bond.getBondType());
      if (bondType == static_cast<int>(RDKit::Bond::ONEANDAHALF) ||
          bondType == static_cast<int>(RDKit::Bond::AROMATIC)) {
        return static_cast<int>(RDKit::Bond::AROMATIC);
      }
      return bondType;
    }
  }
  return 0;
}

BondLabelKey makeBondLabelKey(const RDKit::Bond& bond, const MCSParameters& params) {
  BondLabelKey key;
  key.bondCompareValue = bondOrderClass(bond, params);
  if (params.bondCompareParameters.ringMatchesRingOnly) {
    key.ringState = bond.getOwningMol().getRingInfo()->numBondRings(bond.getIdx()) > 0 ? 1 : 2;
  }
  return key;
}

LabeledGraph buildLabeledGraph(const RDKit::ROMol&               mol,
                               const MCSParameters&              params,
                               std::map<AtomLabelKey, uint16_t>& atomLabels,
                               std::map<BondLabelKey, uint16_t>& bondLabels) {
  LabeledGraph                           out;
  std::vector<std::pair<size_t, size_t>> edges;
  edges.reserve(mol.getNumBonds());

  for (const auto* bond : mol.bonds()) {
    edges.emplace_back(bond->getBeginAtomIdx(), bond->getEndAtomIdx());
  }
  out.graph = mcs::buildGraphFromEdges(mol.getNumAtoms(), edges);

  out.vertexLabels.reserve(mol.getNumAtoms());
  for (const auto* atom : mol.atoms()) {
    out.vertexLabels.push_back(internLabel(atomLabels, makeAtomLabelKey(mol, *atom, params)));
  }

  const auto numAtoms = static_cast<size_t>(mol.getNumAtoms());
  out.edgeLabels.assign(numAtoms * numAtoms, 0);
  for (const auto* bond : mol.bonds()) {
    const auto label                       = internLabel(bondLabels, makeBondLabelKey(*bond, params));
    const auto begin                       = static_cast<size_t>(bond->getBeginAtomIdx());
    const auto end                         = static_cast<size_t>(bond->getEndAtomIdx());
    out.edgeLabels[begin * numAtoms + end] = label;
    out.edgeLabels[end * numAtoms + begin] = label;
  }

  return out;
}

RDKit::MCSParameters buildRDKitParameters(const MCSParameters& params) {
  RDKit::MCSParameters rdParams;
  rdParams.MaximizeBonds = params.maximizeBonds;
  rdParams.Timeout       = params.timeoutSeconds;

  rdParams.AtomCompareParameters.MatchValences       = params.atomCompareParameters.matchValences;
  rdParams.AtomCompareParameters.MatchFormalCharge   = params.atomCompareParameters.matchFormalCharge;
  rdParams.AtomCompareParameters.RingMatchesRingOnly = params.atomCompareParameters.ringMatchesRingOnly;
  rdParams.AtomCompareParameters.MatchIsotope        = params.atomCompareParameters.matchIsotope;
  rdParams.BondCompareParameters.RingMatchesRingOnly = params.bondCompareParameters.ringMatchesRingOnly;

  switch (params.atomCompare) {
    case MCSAtomCompare::Any:
      rdParams.setMCSAtomTyperFromEnum(RDKit::AtomCompareAny);
      break;
    case MCSAtomCompare::Elements:
      rdParams.setMCSAtomTyperFromEnum(RDKit::AtomCompareElements);
      break;
    case MCSAtomCompare::Isotopes:
      rdParams.setMCSAtomTyperFromEnum(RDKit::AtomCompareIsotopes);
      break;
    case MCSAtomCompare::AnyHeavyAtom:
      rdParams.setMCSAtomTyperFromEnum(RDKit::AtomCompareAnyHeavyAtom);
      break;
  }

  switch (params.bondCompare) {
    case MCSBondCompare::Any:
      rdParams.setMCSBondTyperFromEnum(RDKit::BondCompareAny);
      break;
    case MCSBondCompare::Order:
      rdParams.setMCSBondTyperFromEnum(RDKit::BondCompareOrder);
      break;
    case MCSBondCompare::OrderExact:
      rdParams.setMCSBondTyperFromEnum(RDKit::BondCompareOrderExact);
      break;
  }

  return rdParams;
}

void fillResultBondMapping(const RDKit::ROMol&                     molA,
                           const RDKit::ROMol&                     molB,
                           const std::vector<std::pair<int, int>>& atomMapping,
                           std::vector<std::pair<int, int>>&       bondMapping) {
  std::vector<int> aToB(molA.getNumAtoms(), -1);
  for (const auto& [aIdx, bIdx] : atomMapping) {
    if (aIdx >= 0 && aIdx < static_cast<int>(aToB.size())) {
      aToB[static_cast<size_t>(aIdx)] = bIdx;
    }
  }

  bondMapping.clear();
  for (const auto* bondA : molA.bonds()) {
    const int aBegin = static_cast<int>(bondA->getBeginAtomIdx());
    const int aEnd   = static_cast<int>(bondA->getEndAtomIdx());
    if (aToB[static_cast<size_t>(aBegin)] < 0 || aToB[static_cast<size_t>(aEnd)] < 0) {
      continue;
    }
    const auto* bondB = molB.getBondBetweenAtoms(aToB[static_cast<size_t>(aBegin)], aToB[static_cast<size_t>(aEnd)]);
    if (bondB != nullptr) {
      bondMapping.emplace_back(static_cast<int>(bondA->getIdx()), static_cast<int>(bondB->getIdx()));
    }
  }
}

MCSResult runRDKitFallback(const RDKit::ROMol& molA, const RDKit::ROMol& molB, const MCSParameters& params) {
  MCSResult result;
  result.source = MCSResultSource::RDKitFallback;

  std::vector<RDKit::ROMOL_SPTR> mols;
  mols.emplace_back(new RDKit::ROMol(molA));
  mols.emplace_back(new RDKit::ROMol(molB));
  auto       rdParams = buildRDKitParameters(params);
  const auto rdResult = RDKit::findMCS(mols, &rdParams);

  result.numAtoms = rdResult.NumAtoms;
  result.numBonds = rdResult.NumBonds;
  result.canceled = rdResult.Canceled;

  if (rdResult.QueryMol == nullptr) {
    return result;
  }

  RDKit::SubstructMatchParameters matchParams;
  matchParams.uniquify   = false;
  matchParams.maxMatches = 1;
  const auto matchesA    = RDKit::SubstructMatch(molA, *rdResult.QueryMol, matchParams);
  const auto matchesB    = RDKit::SubstructMatch(molB, *rdResult.QueryMol, matchParams);
  if (matchesA.empty() || matchesB.empty()) {
    return result;
  }

  std::vector<int> queryToA(rdResult.QueryMol->getNumAtoms(), -1);
  std::vector<int> queryToB(rdResult.QueryMol->getNumAtoms(), -1);
  for (const auto& [queryAtomIdx, targetAtomIdx] : matchesA.front()) {
    queryToA[static_cast<size_t>(queryAtomIdx)] = targetAtomIdx;
  }
  for (const auto& [queryAtomIdx, targetAtomIdx] : matchesB.front()) {
    queryToB[static_cast<size_t>(queryAtomIdx)] = targetAtomIdx;
  }

  for (size_t queryIdx = 0; queryIdx < queryToA.size(); ++queryIdx) {
    if (queryToA[queryIdx] >= 0 && queryToB[queryIdx] >= 0) {
      result.atomMapping.emplace_back(queryToA[queryIdx], queryToB[queryIdx]);
    }
  }
  fillResultBondMapping(molA, molB, result.atomMapping, result.bondMapping);
  return result;
}

void validateSupportedParameters(const MCSParameters& params) {
  if (params.storeAll) {
    throw std::invalid_argument("GPU fMCS does not support StoreAll");
  }
  if (!params.connectedOnly) {
    throw std::invalid_argument("GPU fMCS supports connected MCS only");
  }
  if (!params.maximizeBonds) {
    throw std::invalid_argument("GPU fMCS supports MaximizeBonds only");
  }
  if (params.threshold != 1.0) {
    throw std::invalid_argument("GPU fMCS supports Threshold=1.0 only");
  }
  if (params.verbose) {
    throw std::invalid_argument("GPU fMCS does not support Verbose output");
  }
  if (!params.initialSeed.empty()) {
    throw std::invalid_argument("GPU fMCS does not support an initial SMARTS seed");
  }
  if (params.atomCompare == MCSAtomCompare::AnyHeavyAtom) {
    throw std::invalid_argument("GPU fMCS does not support AtomCompareAnyHeavyAtom");
  }
  if (params.atomCompareParameters.matchChiralTag) {
    throw std::invalid_argument("GPU fMCS does not support atom MatchChiralTag");
  }
  if (params.atomCompareParameters.completeRingsOnly) {
    throw std::invalid_argument("GPU fMCS does not support atom CompleteRingsOnly");
  }
  if (params.atomCompareParameters.matchIsotope) {
    throw std::invalid_argument(
      "GPU fMCS does not support atomCompareParameters.matchIsotope; use AtomCompareIsotopes instead");
  }
  if (params.atomCompareParameters.maxDistance != -1.0) {
    throw std::invalid_argument("GPU fMCS does not support atom MaxDistance");
  }
  if (params.bondCompareParameters.completeRingsOnly) {
    throw std::invalid_argument("GPU fMCS does not support bond CompleteRingsOnly");
  }
  if (params.bondCompareParameters.matchFusedRings) {
    throw std::invalid_argument("GPU fMCS does not support MatchFusedRings");
  }
  if (params.bondCompareParameters.matchFusedRingsStrict) {
    throw std::invalid_argument("GPU fMCS does not support MatchFusedRingsStrict");
  }
  if (params.bondCompareParameters.matchStereo) {
    throw std::invalid_argument("GPU fMCS does not support bond MatchStereo");
  }
}

bool shouldFallbackToRDKit(const RDKit::ROMol& molA, const RDKit::ROMol& molB, std::string& reason) {
  if (std::max(molA.getNumAtoms(), molB.getNumAtoms()) > 128 ||
      std::max(molA.getNumBonds(), molB.getNumBonds()) > 128) {
    reason = "molecule exceeds fMCS tier-128 limits";
    return true;
  }
  for (const auto* atom : molA.atoms()) {
    if (atom->getDegree() > static_cast<unsigned int>(mcs::fmcs::kMaxNeighborsPerAtom)) {
      reason = "molecule exceeds fMCS maximum atom degree of " + std::to_string(mcs::fmcs::kMaxNeighborsPerAtom);
      return true;
    }
  }
  for (const auto* atom : molB.atoms()) {
    if (atom->getDegree() > static_cast<unsigned int>(mcs::fmcs::kMaxNeighborsPerAtom)) {
      reason = "molecule exceeds fMCS maximum atom degree of " + std::to_string(mcs::fmcs::kMaxNeighborsPerAtom);
      return true;
    }
  }
  return false;
}

MCSResult convertGpuResult(const RDKit::ROMol&   molA,
                           const RDKit::ROMol&   molB,
                           const mcs::MCSResult& gpuResult,
                           const MCSParameters&  params) {
  MCSResult out;
  out.numAtoms = static_cast<unsigned int>(gpuResult.numCommonVertices);
  out.numBonds = static_cast<unsigned int>(gpuResult.numCommonEdges);
  out.canceled = gpuResult.timedOut || gpuResult.killed;
  out.source   = MCSResultSource::GPU;

  const size_t numMappedAtoms = std::min(gpuResult.mappingA.size(), gpuResult.mappingB.size());
  out.atomMapping.reserve(numMappedAtoms);
  for (size_t i = 0; i < numMappedAtoms; ++i) {
    out.atomMapping.emplace_back(static_cast<int>(gpuResult.mappingA[i]), static_cast<int>(gpuResult.mappingB[i]));
  }

  out.bondMapping.reserve(gpuResult.edgeMappingA.size());
  const size_t numMappedBonds = std::min(gpuResult.edgeMappingA.size(), gpuResult.edgeMappingB.size());
  for (size_t i = 0; i < numMappedBonds; ++i) {
    const auto& edgeA = gpuResult.edgeMappingA[i];
    const auto& edgeB = gpuResult.edgeMappingB[i];
    const auto* bondA =
      molA.getBondBetweenAtoms(static_cast<unsigned int>(edgeA.first), static_cast<unsigned int>(edgeA.second));
    const auto* bondB =
      molB.getBondBetweenAtoms(static_cast<unsigned int>(edgeB.first), static_cast<unsigned int>(edgeB.second));
    if (bondA != nullptr && bondB != nullptr) {
      out.bondMapping.emplace_back(static_cast<int>(bondA->getIdx()), static_cast<int>(bondB->getIdx()));
    }
  }

  // The fMCS engine searches the connected MCES lattice, whose seeds are
  // compatible bond pairs. RDKit MCS still returns a one-atom result when no
  // common bond exists, so complete that degenerate case in the public adapter.
  if (out.numAtoms == 0 && !out.canceled) {
    for (const auto* atomA : molA.atoms()) {
      const auto labelA = makeAtomLabelKey(molA, *atomA, params);
      for (const auto* atomB : molB.atoms()) {
        if (labelA == makeAtomLabelKey(molB, *atomB, params)) {
          out.numAtoms = 1;
          out.atomMapping.emplace_back(static_cast<int>(atomA->getIdx()), static_cast<int>(atomB->getIdx()));
          return out;
        }
      }
    }
  }
  return out;
}

std::vector<PreparedGpuPair> prepareGpuPairs(const std::vector<const RDKit::ROMol*>& mols,
                                             const std::vector<MCSPair>&             pairs,
                                             const MCSParameters&                    params,
                                             int                                     preprocessingThreads,
                                             std::vector<MCSResult>&                 results) {
  std::vector<std::unique_ptr<PreparedGpuPair>> prepared(pairs.size());
  if (pairs.empty()) {
    return {};
  }

  std::atomic<size_t> nextPair{0};
  std::atomic<bool>   abort{false};
  std::exception_ptr  firstException;
  std::mutex          exceptionMutex;

  auto setException = [&](std::exception_ptr ex) {
    std::lock_guard<std::mutex> lock(exceptionMutex);
    if (!firstException) {
      firstException = ex;
    }
    abort.store(true, std::memory_order_release);
  };

  auto prepareOne = [&](size_t i) {
    const auto [idxA, idxB] = pairs[i];
    if (idxA >= mols.size() || idxB >= mols.size()) {
      throw std::runtime_error("findMCSBatch pair index out of range");
    }
    const RDKit::ROMol* molA = mols[idxA];
    const RDKit::ROMol* molB = mols[idxB];
    if (molA == nullptr || molB == nullptr) {
      throw std::runtime_error("findMCSBatch pair references a null molecule pointer");
    }

    std::string fallbackReason;
    if (shouldFallbackToRDKit(*molA, *molB, fallbackReason)) {
      if (params.requireGpu) {
        throw std::runtime_error("GPU MCS path unavailable: " + fallbackReason);
      }
      results[i] = runRDKitFallback(*molA, *molB, params);
      return;
    }

    std::map<AtomLabelKey, uint16_t> atomLabels;
    std::map<BondLabelKey, uint16_t> bondLabels;

    auto item       = std::make_unique<PreparedGpuPair>();
    item->resultIdx = i;
    item->molIdxA   = idxA;
    item->molIdxB   = idxB;
    item->graphA    = buildLabeledGraph(*molA, params, atomLabels, bondLabels);
    item->graphB    = buildLabeledGraph(*molB, params, atomLabels, bondLabels);
    prepared[i]     = std::move(item);
  };

  const int threadCount = std::min<int>(std::max(1, preprocessingThreads), static_cast<int>(pairs.size()));
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(threadCount));
  for (int t = 0; t < threadCount; ++t) {
    workers.emplace_back([&]() {
      while (!abort.load(std::memory_order_acquire)) {
        const size_t i = nextPair.fetch_add(1, std::memory_order_relaxed);
        if (i >= pairs.size()) {
          break;
        }
        try {
          prepareOne(i);
        } catch (...) {
          setException(std::current_exception());
          break;
        }
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  if (firstException) {
    std::rethrow_exception(firstException);
  }

  std::vector<PreparedGpuPair> gpuPairs;
  gpuPairs.reserve(pairs.size());
  for (auto& item : prepared) {
    if (item) {
      gpuPairs.push_back(std::move(*item));
    }
  }
  return gpuPairs;
}

void runGpuPairs(std::vector<PreparedGpuPair>&           gpuPairs,
                 const std::vector<const RDKit::ROMol*>& mols,
                 const MCSParameters&                    params,
                 const std::vector<int>&                 gpuIds,
                 int                                     effectiveWorkerThreads,
                 int                                     effectiveExecutorsPerRunner,
                 cudaStream_t                            stream,
                 std::vector<MCSResult>&                 results) {
  if (gpuPairs.empty()) {
    return;
  }

  const int numGpus      = static_cast<int>(gpuIds.size());
  const int totalRunners = std::max(1, numGpus * std::max(1, effectiveWorkerThreads));
  if (stream != nullptr && totalRunners > 1) {
    throw std::invalid_argument("MCS multi-worker or multi-GPU dispatch does not support an external CUDA stream");
  }

  const size_t activeRunners = std::min<size_t>(static_cast<size_t>(totalRunners), gpuPairs.size());
  std::vector<std::vector<PreparedGpuPair>> runnerWork(activeRunners);
  for (size_t i = 0; i < gpuPairs.size(); ++i) {
    runnerWork[i % activeRunners].push_back(std::move(gpuPairs[i]));
  }

  auto runOneRunner = [&](size_t runnerIdx) {
    auto& work = runnerWork[runnerIdx];
    if (work.empty()) {
      return;
    }

    const int        deviceId = gpuIds[runnerIdx % static_cast<size_t>(numGpus)];
    const WithDevice setDevice(deviceId);

    std::vector<LabeledGraph> gpuGraphsA;
    std::vector<LabeledGraph> gpuGraphsB;
    std::vector<size_t>       resultIndices;
    std::vector<size_t>       molIndicesA;
    std::vector<size_t>       molIndicesB;
    gpuGraphsA.reserve(work.size());
    gpuGraphsB.reserve(work.size());
    resultIndices.reserve(work.size());
    molIndicesA.reserve(work.size());
    molIndicesB.reserve(work.size());
    for (auto& item : work) {
      resultIndices.push_back(item.resultIdx);
      molIndicesA.push_back(item.molIdxA);
      molIndicesB.push_back(item.molIdxB);
      gpuGraphsA.push_back(std::move(item.graphA));
      gpuGraphsB.push_back(std::move(item.graphB));
    }

    mcs::fmcs::Parameters fmcsParams;
    fmcsParams.batchSize          = params.batchSize;
    fmcsParams.blockSize          = params.blockSize;
    fmcsParams.executorsPerRunner = effectiveExecutorsPerRunner;
    fmcsParams.matchVertexLabels  = usesAtomLabels(params);
    fmcsParams.matchEdgeLabels    = usesBondLabels(params);
    fmcsParams.timeoutMs          = static_cast<float>(params.timeoutSeconds) * 1000.0f;

    auto gpuResults = mcs::fmcs::findMCESfMCSBatchLabeled(gpuGraphsA, gpuGraphsB, fmcsParams, stream);
    for (size_t gpuIdx = 0; gpuIdx < gpuResults.size(); ++gpuIdx) {
      const size_t resultIdx = resultIndices[gpuIdx];
      const size_t idxA      = molIndicesA[gpuIdx];
      const size_t idxB      = molIndicesB[gpuIdx];
      if (gpuResults[gpuIdx].overflowed) {
        if (params.requireGpu) {
          throw std::runtime_error("GPU MCS path overflowed");
        }
        results[resultIdx] = runRDKitFallback(*mols[idxA], *mols[idxB], params);
      } else {
        results[resultIdx] = convertGpuResult(*mols[idxA], *mols[idxB], gpuResults[gpuIdx], params);
      }
    }
  };

  if (activeRunners == 1) {
    runOneRunner(0);
    return;
  }

  std::atomic<bool>  abort{false};
  std::exception_ptr firstException;
  std::mutex         exceptionMutex;
  auto               setException = [&](std::exception_ptr ex) {
    std::lock_guard<std::mutex> lock(exceptionMutex);
    if (!firstException) {
      firstException = ex;
    }
    abort.store(true, std::memory_order_release);
  };

  std::vector<std::thread> runners;
  runners.reserve(activeRunners);
  for (size_t runnerIdx = 0; runnerIdx < activeRunners; ++runnerIdx) {
    runners.emplace_back([&, runnerIdx]() {
      if (abort.load(std::memory_order_acquire)) {
        return;
      }
      try {
        runOneRunner(runnerIdx);
      } catch (...) {
        setException(std::current_exception());
      }
    });
  }
  for (auto& runner : runners) {
    runner.join();
  }
  if (firstException) {
    std::rethrow_exception(firstException);
  }
}

}  // namespace

std::vector<MCSResult> findMCSBatch(const std::vector<const RDKit::ROMol*>& mols,
                                    const std::vector<MCSPair>&             pairs,
                                    cudaStream_t                            stream,
                                    const MCSParameters&                    params) {
  validateSupportedParameters(params);

  std::vector<MCSResult> results(pairs.size());
  if (pairs.empty())
    return results;

  const auto gpuIds                        = resolveGpuIds(params);
  int        effectivePreprocessingThreads = 1;
  int        effectiveWorkerThreads        = 1;
  computeEffectiveThreadCounts(params,
                               static_cast<int>(gpuIds.size()),
                               effectivePreprocessingThreads,
                               effectiveWorkerThreads);
  if (stream != nullptr) {
    if ((params.workerThreads != -1 && effectiveWorkerThreads > 1) || gpuIds.size() > 1) {
      throw std::invalid_argument("MCS multi-worker or multi-GPU dispatch does not support an external CUDA stream");
    }
    const WithDevice streamDevice(gpuIds.front());
    if (!acquireExternalStream(reinterpret_cast<std::uintptr_t>(stream)).has_value()) {
      throw std::invalid_argument("MCS external CUDA stream does not belong to the configured GPU");
    }
    effectiveWorkerThreads = 1;
  }

  const int totalRunners       = std::max(1, static_cast<int>(gpuIds.size()) * effectiveWorkerThreads);
  const int effectiveExecutors = effectiveExecutorsPerRunner(params, totalRunners, stream);

  auto gpuPairs = prepareGpuPairs(mols, pairs, params, effectivePreprocessingThreads, results);
  runGpuPairs(gpuPairs, mols, params, gpuIds, effectiveWorkerThreads, effectiveExecutors, stream, results);

  return results;
}

std::vector<MCSResult> findMCSBatch(const std::vector<const RDKit::ROMol*>& molsA,
                                    const std::vector<const RDKit::ROMol*>& molsB,
                                    cudaStream_t                            stream,
                                    const MCSParameters&                    params) {
  if (molsA.size() != molsB.size()) {
    throw std::runtime_error("findMCSBatch requires equal-sized molecule arrays");
  }

  std::vector<const RDKit::ROMol*> mols;
  mols.reserve(molsA.size() + molsB.size());
  mols.insert(mols.end(), molsA.begin(), molsA.end());
  mols.insert(mols.end(), molsB.begin(), molsB.end());

  std::vector<MCSPair> pairs;
  pairs.reserve(molsA.size());
  for (size_t i = 0; i < molsA.size(); ++i) {
    pairs.emplace_back(i, molsA.size() + i);
  }
  return findMCSBatch(mols, pairs, stream, params);
}

std::vector<MCSResult> findMCSAllPairs(const std::vector<const RDKit::ROMol*>& mols,
                                       MCSAllPairsOptions                      options,
                                       cudaStream_t                            stream,
                                       const MCSParameters&                    params) {
  std::vector<MCSPair> pairs;
  const size_t         n = mols.size();
  if (options.upperTriangle) {
    const size_t maxPairs = options.includeDiagonal ? (n * (n + 1)) / 2 : (n > 1 ? (n * (n - 1)) / 2 : 0);
    pairs.reserve(maxPairs);
    for (size_t i = 0; i < n; ++i) {
      const size_t begin = options.includeDiagonal ? i : i + 1;
      for (size_t j = begin; j < n; ++j) {
        pairs.emplace_back(i, j);
      }
    }
  } else {
    pairs.reserve(options.includeDiagonal ? n * n : (n > 1 ? n * (n - 1) : 0));
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        if (!options.includeDiagonal && i == j) {
          continue;
        }
        pairs.emplace_back(i, j);
      }
    }
  }
  return findMCSBatch(mols, pairs, stream, params);
}

}  // namespace nvMolKit
