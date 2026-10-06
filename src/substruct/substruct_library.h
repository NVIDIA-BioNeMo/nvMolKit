// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_SUBSTRUCT_LIBRARY_H
#define NVMOLKIT_SUBSTRUCT_LIBRARY_H

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "src/substruct/substruct_results.h"

namespace RDKit {
class ROMol;
}  // namespace RDKit

namespace nvMolKit {

/**
 * Target molecules kept on the GPU for repeated substructure queries, the GPU
 * counterpart of RDKit's SubstructLibrary. Molecules get IDs in the order they
 * are added and become searchable at the next finalize(). With several GPU IDs
 * configured, molecules are split across those GPUs and results are merged in
 * ID order. Molecules the GPU format cannot represent are matched with RDKit.
 *
 * Queries may run concurrently, as many as fit in GPU memory; additions and
 * finalize() wait for running queries and hold off new ones.
 */
class SubstructLibrary {
 public:
  explicit SubstructLibrary(SubstructSearchConfig config = SubstructSearchConfig{});
  ~SubstructLibrary();

  SubstructLibrary(const SubstructLibrary&)            = delete;
  SubstructLibrary& operator=(const SubstructLibrary&) = delete;
  SubstructLibrary(SubstructLibrary&&)                 = delete;
  SubstructLibrary& operator=(SubstructLibrary&&)      = delete;

  /** Copy a molecule into the library and return its ID. It is searchable after the next finalize(). */
  unsigned int addMol(const RDKit::ROMol& molecule);

  /**
   * Copy molecules into the library in parallel and return their IDs. They are searchable after the next
   * finalize(). Uses preprocessingThreads from the search configuration (-1 selects all OpenMP threads).
   */
  std::vector<unsigned int> addMols(const std::vector<const RDKit::ROMol*>& molecules);

  /**
   * Pack and upload every molecule added since the last finalize() and make them searchable. If it throws, the
   * library is unchanged and the molecules remain waiting for the next finalize(). A GPU that receives new
   * molecules has all of its molecules copied and uploaded again, so add molecules in large groups between calls.
   */
  void finalize(cudaStream_t stream = nullptr);

  /** Number of searchable molecules. */
  [[nodiscard]] std::size_t size() const;

  /** Number of molecules added since the last finalize(). */
  [[nodiscard]] std::size_t pendingSize() const;

  /** How many queries can run at once, set by finalize() from the GPU memory left after the molecules. */
  [[nodiscard]] std::size_t maxConcurrentQueries() const;

  /** IDs of matching molecules, ascending. maxResults limits the count; -1 returns all, 0 returns none. */
  [[nodiscard]] std::vector<unsigned int> getMatches(const RDKit::ROMol& query,
                                                     int                 maxResults = -1,
                                                     cudaStream_t        stream     = nullptr) const;

  [[nodiscard]] std::size_t countMatches(const RDKit::ROMol& query, cudaStream_t stream = nullptr) const;
  [[nodiscard]] bool        hasMatch(const RDKit::ROMol& query, cudaStream_t stream = nullptr) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_SUBSTRUCT_LIBRARY_H
