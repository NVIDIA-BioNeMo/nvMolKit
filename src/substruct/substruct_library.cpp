// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "src/substruct/substruct_library.h"

#include <GraphMol/MolOps.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/Substruct/SubstructMatch.h>
#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "src/substruct/molecules.h"
#include "src/substruct/substruct_constants.h"
#include "src/substruct/substruct_search.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"
#include "src/utils/nvtx.h"
#include "src/utils/openmp_helpers.h"

namespace nvMolKit {

namespace {

bool rdkitHasMatch(const RDKit::ROMol& target, const RDKit::ROMol& query) {
  RDKit::SubstructMatchParameters params;
  params.uniquify             = false;
  params.maxMatches           = 1;
  params.useChirality         = false;
  params.useQueryQueryMatches = false;
  return !RDKit::SubstructMatch(target, query, params).empty();
}

//! A copy of molecule with the symmetrized SSSR ring info that GPU packing and RDKit ring queries need. RDKit's
//! SubstructLibrary computes it at search time; the library owns its copies, so it is computed once here.
std::unique_ptr<RDKit::ROMol> copyWithRingInfo(const RDKit::ROMol& molecule) {
  auto copy = std::make_unique<RDKit::ROMol>(molecule);
  if (!copy->getRingInfo()->isSymmSssr()) {
    RDKit::MolOps::symmetrizeSSSR(*copy);
  }
  return copy;
}

//! Whether the packed GPU target format can represent a molecule; the rest are matched with RDKit.
bool fitsGpu(const RDKit::ROMol& molecule) {
  try {
    return molecule.getNumAtoms() <= kMaxTargetAtoms && !requiresRDKitFallback(&molecule);
  } catch (const std::runtime_error&) {
    return false;
  }
}

//! The finalized molecules searched on one GPU, packed and uploaded as a single batch in ID order.
struct GpuTargets {
  std::unique_ptr<MoleculesHost>                 host;
  std::unique_ptr<MoleculesDevice>               device;
  std::vector<const RDKit::ROMol*>               molecules;
  std::vector<unsigned int>                      ids;
  std::shared_ptr<const PersistentDeviceTargets> searchTargets;
};

}  // namespace

class SubstructLibrary::Impl {
 public:
  explicit Impl(SubstructSearchConfig config) : config_(std::move(config)) {
    if (config_.algorithm == SubstructAlgorithm::VF2) {
      throw std::invalid_argument("Substructure libraries support the GSI and DFS production backends");
    }
    std::unordered_set<int> seenDevices;
    for (const int deviceId : config_.gpuIds) {
      if (deviceId < 0) {
        throw std::invalid_argument("Substructure library GPU ID must be nonnegative");
      }
      if (!seenDevices.insert(deviceId).second) {
        throw std::invalid_argument("Substructure library GPU IDs must be unique");
      }
    }
    deviceIds_ = config_.gpuIds;
  }

  ~Impl() noexcept {
    releaseOnDevices(
      gpus_.size(),
      [&](std::size_t index) { return gpus_[index] != nullptr; },
      [&](std::size_t index) { gpus_[index].reset(); });
    releaseOnDevices(
      workspaces_.size(),
      [&](std::size_t index) { return workspaces_[index] != nullptr; },
      [&](std::size_t index) { workspaces_[index].reset(); });
  }

  unsigned int addMol(const RDKit::ROMol& molecule) {
    const std::lock_guard lock(mutex_);
    requireIdsAvailable(1);
    molecules_.push_back(copyWithRingInfo(molecule));
    return static_cast<unsigned int>(molecules_.size() - 1);
  }

  std::vector<unsigned int> addMols(const std::vector<const RDKit::ROMol*>& molecules) {
    const std::lock_guard lock(mutex_);
    requireIdsAvailable(molecules.size());
    const std::size_t firstId = molecules_.size();
    molecules_.resize(firstId + molecules.size());
    detail::OpenMPExceptionRegistry exceptions;
#pragma omp parallel for num_threads(threads()) schedule(static)
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(molecules.size()); ++index) {
      try {
        const RDKit::ROMol* molecule = molecules[static_cast<std::size_t>(index)];
        if (molecule == nullptr) {
          throw std::invalid_argument("Substructure library molecule cannot be null");
        }
        molecules_[firstId + static_cast<std::size_t>(index)] = copyWithRingInfo(*molecule);
      } catch (...) {
        exceptions.store(std::current_exception());
      }
    }
    try {
      exceptions.rethrow();
    } catch (...) {
      molecules_.resize(firstId);
      throw;
    }
    std::vector<unsigned int> ids(molecules.size());
    for (std::size_t index = 0; index < ids.size(); ++index) {
      ids[index] = static_cast<unsigned int>(firstId + index);
    }
    return ids;
  }

  void finalize(cudaStream_t stream) {
    const std::lock_guard lock(mutex_);
    useCurrentDeviceIfNoneConfigured();
    if (deviceIds_.size() > 1 && stream != nullptr) {
      throw std::invalid_argument("A single external CUDA stream cannot be used with a multi-GPU substructure library");
    }
    int deviceCount = 0;
    cudaCheckError(cudaGetDeviceCount(&deviceCount));
    for (const int deviceId : deviceIds_) {
      if (deviceId >= deviceCount) {
        throw std::invalid_argument("Substructure library GPU ID is not available");
      }
    }
    gpus_.resize(deviceIds_.size());

    // Molecules added since the last finalize(): those the GPU format cannot represent are matched with RDKit,
    // and the rest go to the GPU searching the fewest molecules.
    const std::size_t               firstNew = numFinalized_;
    const std::size_t               numNew   = molecules_.size() - firstNew;
    std::vector<std::uint8_t>       onGpu(numNew);
    detail::OpenMPExceptionRegistry screenExceptions;
#pragma omp parallel for num_threads(threads()) schedule(static)
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(numNew); ++index) {
      try {
        onGpu[static_cast<std::size_t>(index)] = fitsGpu(*molecules_[firstNew + static_cast<std::size_t>(index)]);
      } catch (...) {
        screenExceptions.store(std::current_exception());
      }
    }
    screenExceptions.rethrow();
    std::vector<std::size_t> gpuLoad(deviceIds_.size(), 0);
    for (std::size_t gpu = 0; gpu < deviceIds_.size(); ++gpu) {
      gpuLoad[gpu] = gpus_[gpu] != nullptr ? gpus_[gpu]->ids.size() : 0;
    }
    std::vector<std::vector<unsigned int>> newGpuIds(deviceIds_.size());
    std::vector<unsigned int>              newRdkitIds;
    for (std::size_t index = 0; index < numNew; ++index) {
      const auto id = static_cast<unsigned int>(firstNew + index);
      if (onGpu[index] == 0) {
        newRdkitIds.push_back(id);
        continue;
      }
      const auto gpu = static_cast<std::size_t>(std::min_element(gpuLoad.begin(), gpuLoad.end()) - gpuLoad.begin());
      newGpuIds[gpu].push_back(id);
      ++gpuLoad[gpu];
    }

    // Pack one GPU at a time with every thread, rather than inside the per-GPU loop below, where OpenMP may not
    // give the packer's nested parallel region more than one thread.
    std::vector<MoleculesHost> packed(deviceIds_.size());
    for (std::size_t gpu = 0; gpu < deviceIds_.size(); ++gpu) {
      if (!newGpuIds[gpu].empty()) {
        buildTargetBatchParallelInto(packed[gpu], threads(), moleculesWithIds(newGpuIds[gpu]), {});
      }
    }

    // Build every GPU's replacement before touching the current ones, so a failure leaves the library as it was.
    std::vector<std::unique_ptr<GpuTargets>> replacements(deviceIds_.size());
    detail::OpenMPExceptionRegistry          exceptions;
#pragma omp parallel for num_threads(static_cast<int>(deviceIds_.size())) schedule(static)
    for (std::int64_t gpu = 0; gpu < static_cast<std::int64_t>(deviceIds_.size()); ++gpu) {
      try {
        const auto index = static_cast<std::size_t>(gpu);
        if (!newGpuIds[index].empty()) {
          const WithDevice device(deviceIds_[index]);
          replacements[index] =
            extendGpuTargets(gpus_[index].get(), newGpuIds[index], packed[index], validateStream(stream));
        }
      } catch (...) {
        // Clear a recoverable CUDA error such as out of memory, so this thread's next CUDA check does not report it.
        static_cast<void>(cudaGetLastError());
        exceptions.store(std::current_exception());
      }
    }
    try {
      exceptions.rethrow();
      createWorkspaces();
      rdkitIds_.reserve(rdkitIds_.size() + newRdkitIds.size());
    } catch (...) {
      static_cast<void>(cudaGetLastError());
      releaseOnDevices(
        replacements.size(),
        [&](std::size_t index) { return replacements[index] != nullptr; },
        [&](std::size_t index) { replacements[index].reset(); });
      throw;
    }

    releaseOnDevices(
      gpus_.size(),
      [&](std::size_t index) { return replacements[index] != nullptr; },
      [&](std::size_t index) { gpus_[index] = std::move(replacements[index]); });
    rdkitIds_.insert(rdkitIds_.end(), newRdkitIds.begin(), newRdkitIds.end());
    numFinalized_ = molecules_.size();
    finalized_    = true;
  }

  [[nodiscard]] std::size_t size() const {
    const std::lock_guard lock(mutex_);
    return numFinalized_;
  }

  [[nodiscard]] std::size_t pendingSize() const {
    const std::lock_guard lock(mutex_);
    return molecules_.size() - numFinalized_;
  }

  [[nodiscard]] std::vector<unsigned int> getMatches(const RDKit::ROMol& query,
                                                     int                 maxResults,
                                                     cudaStream_t        stream) const {
    const std::lock_guard lock(mutex_);
    requireFinalized();
    if (maxResults == 0) {
      return {};
    }
    return matches(query, stream, maxResults);
  }

  [[nodiscard]] std::size_t countMatches(const RDKit::ROMol& query, cudaStream_t stream) const {
    const std::lock_guard lock(mutex_);
    requireFinalized();
    return matches(query, stream, -1).size();
  }

  [[nodiscard]] bool hasMatch(const RDKit::ROMol& query, cudaStream_t stream) const {
    const std::lock_guard lock(mutex_);
    requireFinalized();
    return !matches(query, stream, 1).empty();
  }

 private:
  void requireIdsAvailable(std::size_t count) const {
    if (count > std::numeric_limits<unsigned int>::max() - molecules_.size()) {
      throw std::overflow_error("Substructure library molecule ID space exhausted");
    }
  }

  void requireFinalized() const {
    if (!finalized_) {
      throw std::logic_error("Substructure library must be finalized before querying");
    }
  }

  [[nodiscard]] int threads() const {
    return config_.preprocessingThreads == -1 ? omp_get_max_threads() : std::max(1, config_.preprocessingThreads);
  }

  void useCurrentDeviceIfNoneConfigured() {
    if (deviceIds_.empty()) {
      int currentDevice = 0;
      cudaCheckError(cudaGetDevice(&currentDevice));
      deviceIds_.push_back(currentDevice);
      config_.gpuIds = deviceIds_;
    }
  }

  [[nodiscard]] cudaStream_t validateStream(cudaStream_t stream) const {
    const auto validated = acquireExternalStream(reinterpret_cast<std::uintptr_t>(stream));
    if (!validated.has_value()) {
      throw std::invalid_argument("CUDA stream does not belong to the substructure library device");
    }
    return *validated;
  }

  [[nodiscard]] std::vector<const RDKit::ROMol*> moleculesWithIds(const std::vector<unsigned int>& ids) const {
    std::vector<const RDKit::ROMol*> result;
    result.reserve(ids.size());
    for (const unsigned int id : ids) {
      result.push_back(molecules_[id].get());
    }
    return result;
  }

  // The current device's targets plus newIds, whose packed form is packed, uploaded on stream and ready to search.
  // The whole batch is copied and uploaded again, so the current targets stay intact if this throws.
  [[nodiscard]] std::unique_ptr<GpuTargets> extendGpuTargets(const GpuTargets*                current,
                                                             const std::vector<unsigned int>& newIds,
                                                             const MoleculesHost&             packed,
                                                             cudaStream_t                     stream) const {
    const std::vector<const RDKit::ROMol*> newMolecules = moleculesWithIds(newIds);

    auto next = std::make_unique<GpuTargets>();
    next->host =
      current != nullptr ? std::make_unique<MoleculesHost>(*current->host) : std::make_unique<MoleculesHost>();
    mergeBatch(*next->host, packed);
    if (current != nullptr) {
      next->molecules = current->molecules;
      next->ids       = current->ids;
    }
    next->molecules.insert(next->molecules.end(), newMolecules.begin(), newMolecules.end());
    next->ids.insert(next->ids.end(), newIds.begin(), newIds.end());

    next->device = std::make_unique<MoleculesDevice>(stream);
    next->device->copyFromHost(*next->host, stream);
    cudaCheckError(cudaStreamSynchronize(stream));
    // The device copy outlives the caller's stream, so release it on the default stream.
    next->device->setStream(nullptr);
    next->searchTargets = makePersistentDeviceTargets(next->molecules, *next->host, *next->device);
    return next;
  }

  // Search state for each GPU, created on the first finalize() and reused by every query.
  void createWorkspaces() {
    if (workspaces_.size() == deviceIds_.size()) {
      return;
    }
    std::vector<std::shared_ptr<SubstructSearchWorkspace>> created(deviceIds_.size());
    for (std::size_t gpu = 0; gpu < deviceIds_.size(); ++gpu) {
      created[gpu] = makeSubstructSearchWorkspace(deviceIds_[gpu]);
    }
    workspaces_ = std::move(created);
    matchFlags_.resize(deviceIds_.size());
  }

  [[nodiscard]] SubstructSearchConfig gpuConfig(std::size_t gpu) const {
    SubstructSearchConfig result = config_;
    result.gpuIds                = {deviceIds_[gpu]};
    // Targets are already packed and each search runs one query, so a large preprocessing team does not help.
    result.preprocessingThreads  = config_.preprocessingThreads == -1 ?
                                     1 :
                                     std::max(1, config_.preprocessingThreads / static_cast<int>(deviceIds_.size()));
    if (result.workerThreads == -1) {
      result.workerThreads = 1;
    }
    return result;
  }

  // IDs of molecules matching query, ascending and truncated to maxResults when positive.
  [[nodiscard]] std::vector<unsigned int> matches(const RDKit::ROMol& query,
                                                  cudaStream_t        stream,
                                                  int                 maxResults) const {
    if (deviceIds_.size() > 1 && stream != nullptr) {
      throw std::invalid_argument("A single external CUDA stream cannot be used with a multi-GPU substructure library");
    }
    std::vector<std::vector<unsigned int>> perGpu(deviceIds_.size());
    detail::OpenMPExceptionRegistry        exceptions;
#pragma omp parallel for num_threads(static_cast<int>(deviceIds_.size())) schedule(static)
    for (std::int64_t gpu = 0; gpu < static_cast<std::int64_t>(deviceIds_.size()); ++gpu) {
      try {
        const auto index = static_cast<std::size_t>(gpu);
        if (gpus_[index] != nullptr) {
          const WithDevice device(deviceIds_[index]);
          perGpu[index] = gpuMatches(index, query, validateStream(stream));
        }
      } catch (...) {
        exceptions.store(std::current_exception());
      }
    }
    exceptions.rethrow();

    std::vector<unsigned int> result;
    for (const auto& gpuResult : perGpu) {
      result.insert(result.end(), gpuResult.begin(), gpuResult.end());
    }
    std::sort(result.begin(), result.end());

    // Molecules matched with RDKit. Once the result limit is filled, only IDs below the last kept match can
    // change the answer.
    const bool          limitFilled = maxResults > 0 && result.size() >= static_cast<std::size_t>(maxResults);
    const std::uint64_t rdkitIdLimit =
      limitFilled ? result[static_cast<std::size_t>(maxResults) - 1] : std::numeric_limits<std::uint64_t>::max();
    const std::size_t gpuMatchCount = result.size();
    ScopedNvtxRange   rdkitRange("SubstructLibrary RDKit matching");
    for (const unsigned int id : rdkitIds_) {
      if (id >= rdkitIdLimit) {
        break;
      }
      if (rdkitHasMatch(*molecules_[id], query)) {
        result.push_back(id);
      }
    }
    std::inplace_merge(result.begin(), result.begin() + static_cast<std::ptrdiff_t>(gpuMatchCount), result.end());
    if (maxResults > 0 && result.size() > static_cast<std::size_t>(maxResults)) {
      result.resize(static_cast<std::size_t>(maxResults));
    }
    return result;
  }

  // IDs of the molecules on one GPU that match query, ascending.
  [[nodiscard]] std::vector<unsigned int> gpuMatches(std::size_t         gpu,
                                                     const RDKit::ROMol& query,
                                                     cudaStream_t        stream) const {
    ScopedNvtxRange             searchRange("SubstructLibrary GPU search");
    const GpuTargets&           targets = *gpus_[gpu];
    std::vector<std::uint8_t>&  flags   = matchFlags_[gpu];
    const SubstructSearchConfig config  = gpuConfig(gpu);
    hasSubstructMatch(*targets.searchTargets, query, flags, config.algorithm, stream, config, workspaces_[gpu].get());
    std::vector<unsigned int> result;
    for (std::size_t target = 0; target < flags.size(); ++target) {
      if (flags[target] != 0) {
        result.push_back(targets.ids[target]);
      }
    }
    return result;
  }

  // For each of the first count configured devices i where holds(i), call release(i) with device i current, then
  // restore the caller's device. Devices without resources are never selected, so a device ID rejected by
  // finalize() leaves no CUDA error behind. Teardown cannot throw, so CUDA failures are logged and skipped.
  template <typename Holds, typename Release>
  void releaseOnDevices(std::size_t count, const Holds& holds, const Release& release) const noexcept {
    int originalDevice = -1;
    cudaCheckErrorNoThrow(cudaGetDevice(&originalDevice));
    for (std::size_t index = 0; index < std::min(count, deviceIds_.size()); ++index) {
      if (!holds(index)) {
        continue;
      }
      const cudaError_t selected = cudaSetDevice(deviceIds_[index]);
      cudaCheckErrorNoThrow(selected);
      if (selected == cudaSuccess) {
        release(index);
      }
    }
    if (originalDevice >= 0) {
      cudaCheckErrorNoThrow(cudaSetDevice(originalDevice));
    }
  }

  SubstructSearchConfig config_;
  std::vector<int>      deviceIds_;
  // Held by every call, so queries, additions, and finalize() run one at a time.
  mutable std::mutex    mutex_;

  //! Every added molecule, indexed by ID; the first numFinalized_ are searchable.
  std::vector<std::unique_ptr<RDKit::ROMol>> molecules_;
  std::size_t                                numFinalized_ = 0;
  bool                                       finalized_    = false;

  std::vector<std::unique_ptr<GpuTargets>>               gpus_;
  //! Finalized molecules the GPU format cannot represent, ascending; matched with RDKit.
  std::vector<unsigned int>                              rdkitIds_;
  std::vector<std::shared_ptr<SubstructSearchWorkspace>> workspaces_;
  //! Per-GPU match flags, reused across queries to avoid reallocating them.
  mutable std::vector<std::vector<std::uint8_t>>         matchFlags_;
};

SubstructLibrary::SubstructLibrary(SubstructSearchConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}

SubstructLibrary::~SubstructLibrary() = default;

unsigned int SubstructLibrary::addMol(const RDKit::ROMol& molecule) {
  return impl_->addMol(molecule);
}

std::vector<unsigned int> SubstructLibrary::addMols(const std::vector<const RDKit::ROMol*>& molecules) {
  return impl_->addMols(molecules);
}

void SubstructLibrary::finalize(cudaStream_t stream) {
  impl_->finalize(stream);
}

std::size_t SubstructLibrary::size() const {
  return impl_->size();
}

std::size_t SubstructLibrary::pendingSize() const {
  return impl_->pendingSize();
}

std::vector<unsigned int> SubstructLibrary::getMatches(const RDKit::ROMol& query,
                                                       int                 maxResults,
                                                       cudaStream_t        stream) const {
  return impl_->getMatches(query, maxResults, stream);
}

std::size_t SubstructLibrary::countMatches(const RDKit::ROMol& query, cudaStream_t stream) const {
  return impl_->countMatches(query, stream);
}

bool SubstructLibrary::hasMatch(const RDKit::ROMol& query, cudaStream_t stream) const {
  return impl_->hasMatch(query, stream);
}

}  // namespace nvMolKit
