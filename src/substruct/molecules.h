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

#ifndef NVMOLKIT_MOLECULES_H
#define NVMOLKIT_MOLECULES_H

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/substruct/atom_data_packed.h"
#include "src/substruct/boolean_tree.cuh"
#include "src/substruct/packed_bonds.h"
#include "src/utils/device_vector.h"

namespace RDKit {
class ROMol;
}  // namespace RDKit

namespace nvMolKit {

/**
 * @brief Bitmask specifying which atom fields to compare for query matching.
 *
 * Multiple fields can be combined with bitwise OR for composite queries.
 * For example, 'C' in SMARTS checks both AtomicNum and IsAliphatic.
 */
enum AtomQueryFlags : uint32_t {
  AtomQueryNone                = 0,
  AtomQueryAtomicNum           = 1 << 0,
  AtomQueryNumExplicitHs       = 1 << 1,
  AtomQueryExplicitValence     = 1 << 2,
  AtomQueryImplicitValence     = 1 << 3,
  AtomQueryFormalCharge        = 1 << 4,
  AtomQueryChiralTag           = 1 << 5,
  AtomQueryNumRadicalElectrons = 1 << 6,
  AtomQueryHybridization       = 1 << 7,
  AtomQueryMinRingSize         = 1 << 8,
  AtomQueryNumRings            = 1 << 9,
  AtomQueryIsAromatic          = 1 << 10,
  AtomQueryIsAliphatic         = 1 << 11,
  AtomQueryTotalValence        = 1 << 12,
  AtomQueryIsInRing            = 1 << 13,  ///< For [R] and [r] any-ring queries
  AtomQueryIsotope             = 1 << 14,  ///< For isotope/mass queries like [13C]
  AtomQueryDegree              = 1 << 15,  ///< For [D] degree queries (explicit bond count)
  AtomQueryTotalConnectivity   = 1 << 16,  ///< For [X] total connectivity queries (degree + Hs)
  AtomQueryNeverMatches        = 1 << 17,  ///< Impossible constraint (e.g., [C;a] aromatic aliphatic)
  AtomQueryRingBondCount       = 1 << 18,  ///< For [x] ring connectivity queries (ring bond count)
  AtomQueryNumImplicitHs       = 1 << 19,  ///< For [h] implicit hydrogen count queries
  AtomQueryHasImplicitH        = 1 << 20,  ///< For [h] without number - has any implicit H
  AtomQueryNumHeteroNeighbors  = 1 << 21,  ///< For heteroatom neighbor count queries
};

using AtomQuery = uint32_t;

/**
 * @brief Bond query data for SMARTS bond queries.
 *
 * Stores the bond type to match and ring bond constraints.
 * For complex OR patterns (e.g., =,#,:), allowedBondTypes is a bitmask where
 * bit N is set if bond type N is allowed (types 0-15 supported).
 */
struct BondQueryData {
  uint8_t  bondType         = 0;  ///< 0 = any, 1 = single, 2 = double, 3 = triple, 12 = aromatic
  uint8_t  queryFlags       = 0;  ///< BondQueryFlags bitmask
  uint16_t allowedBondTypes = 0;  ///< Bitmask of allowed bond types when BondQueryUseBondMask is set
};

/**
 * @brief Information about a single recursive SMARTS pattern within a query.
 *
 * Each RecursivePatternEntry represents one $(...) pattern found in the SMARTS query.
 * The queryMol pointer is non-owning - the original query owns the pattern.
 *
 * For nested recursive patterns, patterns are extracted leaf-first and stored in
 * depth order (depth 0 = leaves, depth N = root-level patterns). This enables
 * level-by-level processing where child pattern results become input for parents.
 */
struct RecursivePatternEntry {
  const RDKit::ROMol* queryMol           = nullptr;  ///< The inner query molecule from $(...)
  int                 queryAtomIdx       = 0;        ///< Index of the query atom containing this pattern
  int                 patternId          = 0;        ///< Unique ID (0-31) for this pattern in the batch
  int                 depth              = 0;        ///< Nesting depth: 0=leaf (no children), 1+=has children
  int                 parentPatternIdx   = -1;       ///< Index into patterns array (before sorting), -1=root
  int                 parentPatternId    = -1;       ///< Parent's patternId (stable across sorting), -1=root
  int                 parentQueryAtomIdx = -1;       ///< Atom index in parent pattern needing this bit
  int                 localIdInParent    = 0;        ///< Bit position in parent's RecursiveMatch (0, 1, 2...)
};

/**
 * @brief Collection of recursive SMARTS patterns extracted from a query.
 *
 * Used to preprocess recursive patterns before main substructure matching.
 * Patterns are sorted by depth (leaves first) for level-by-level processing.
 */
struct RecursivePatternInfo {
  static constexpr int kMaxPatterns = 32;  ///< Maximum supported recursive SMARTS patterns
  static_assert(kMaxPatterns <= 32, "kMaxPatterns must fit in the 32-bit recursive match mask");

  std::vector<RecursivePatternEntry> patterns;                      ///< All recursive patterns found, sorted by depth
  bool                               hasRecursivePatterns = false;  ///< Quick check for any patterns
  int                                maxDepth             = 0;      ///< Maximum nesting depth (0=flat, 1+=nested)

  /**
   * @brief Check if the query has any recursive patterns.
   */
  [[nodiscard]] bool empty() const { return patterns.empty(); }

  /**
   * @brief Get the number of recursive patterns.
   */
  [[nodiscard]] size_t size() const { return patterns.size(); }

  /**
   * @brief Check if the query has nested recursive patterns.
   */
  [[nodiscard]] bool hasNestedPatterns() const { return maxDepth > 0; }
};

/**
 * @brief Host-side batched molecule storage.
 *
 * Stores multiple molecules in a flattened format optimized for GPU transfer.
 * Each molecule's atoms and bonds are stored contiguously, with offset arrays
 * to locate each molecule's data.
 */
struct MoleculesHost {
  // Batch-level offsets (size = numMolecules + 1)
  std::vector<int> batchAtomStarts;  ///< Start index into atomDataPacked for each molecule

  // GPU-optimized packed data (flattened across all molecules)
  std::vector<AtomDataPacked>  atomDataPacked;   ///< Packed atom properties for GPU matching
  std::vector<AtomQueryMask>   atomQueryMasks;   ///< Precomputed query masks (for query molecules only)
  std::vector<BondTypeCounts>  bondTypeCounts;   ///< Precomputed bond type counts per atom
  std::vector<TargetAtomBonds> targetAtomBonds;  ///< Packed bond adjacency for targets
  std::vector<QueryAtomBonds>  queryAtomBonds;   ///< Packed bond adjacency for queries

  // Boolean expression tree data for compound queries (OR/NOT support)
  std::vector<AtomQueryTree>   atomQueryTrees;       ///< Tree metadata per query atom
  std::vector<BoolInstruction> queryInstructions;    ///< Flattened instruction arrays
  std::vector<AtomQueryMask>   queryLeafMasks;       ///< Flattened leaf masks for compound queries
  std::vector<BondTypeCounts>  queryLeafBondCounts;  ///< Flattened leaf bond counts
  std::vector<int>             atomInstrStarts;      ///< Start index into queryInstructions per atom
  std::vector<int>             atomLeafMaskStarts;   ///< Start index into queryLeafMasks per atom

  // Recursive SMARTS patterns extracted from query molecules (one per molecule in batch)
  std::vector<RecursivePatternInfo> recursivePatterns;

  MoleculesHost();

  /**
   * @brief Pre-allocate storage for expected batch size.
   * @param numMols Expected number of molecules
   * @param numAtoms Expected total number of atoms across all molecules
   */
  void reserve(size_t numMols, size_t numAtoms);

  /**
   * @brief Clear all data while preserving allocated capacity.
   *
   * Resets the batch to empty state (just the initial 0 in batchAtomStarts)
   * but keeps vector capacities for reuse without reallocation.
   */
  void clear();

  [[nodiscard]] size_t numMolecules() const { return batchAtomStarts.empty() ? 0 : batchAtomStarts.size() - 1; }
  [[nodiscard]] size_t totalAtoms() const { return atomDataPacked.size(); }
};

/**
 * @brief Molecule type for device-side views.
 */
enum class MoleculeType : uint8_t {
  Target,
  Query
};

/**
 * @brief Device-side view into batched molecule data (templated by molecule type).
 *
 * This structure contains pointers to device memory for the full batch.
 * This is a POD struct that can be passed to CUDA kernels by value.
 * Use getMolecule() from molecules_device.cuh to get per-molecule views.
 *
 * @tparam Type MoleculeType::Target or MoleculeType::Query
 */
template <MoleculeType Type> struct MoleculesDeviceViewT;

/**
 * @brief Device-side view for target molecule batches.
 */
template <> struct MoleculesDeviceViewT<MoleculeType::Target> {
  int const* batchAtomStarts;
  int        numMolecules;

  // GPU-optimized packed data
  AtomDataPacked const*  atomDataPacked;   ///< Packed atom properties for GPU matching
  BondTypeCounts const*  bondTypeCounts;   ///< Precomputed bond type counts per atom
  TargetAtomBonds const* targetAtomBonds;  ///< Packed bond adjacency for targets
};

/**
 * @brief Device-side view for query molecule batches.
 */
template <> struct MoleculesDeviceViewT<MoleculeType::Query> {
  int const* batchAtomStarts;
  int        numMolecules;

  // GPU-optimized packed data
  AtomDataPacked const* atomDataPacked;  ///< Packed atom properties for GPU matching
  AtomQueryMask const*  atomQueryMasks;  ///< Precomputed query masks
  BondTypeCounts const* bondTypeCounts;  ///< Precomputed bond type counts per atom
  QueryAtomBonds const* queryAtomBonds;  ///< Packed bond adjacency for queries

  // Boolean expression tree data for compound queries
  AtomQueryTree const*   atomQueryTrees;       ///< Tree metadata per query atom
  BoolInstruction const* queryInstructions;    ///< Flattened instruction arrays
  AtomQueryMask const*   queryLeafMasks;       ///< Flattened leaf masks for compound queries
  BondTypeCounts const*  queryLeafBondCounts;  ///< Flattened leaf bond counts
  int const*             atomInstrStarts;      ///< Start index into queryInstructions per atom
  int const*             atomLeafMaskStarts;   ///< Start index into queryLeafMasks per atom
};

using TargetMoleculesDeviceView = MoleculesDeviceViewT<MoleculeType::Target>;
using QueryMoleculesDeviceView  = MoleculesDeviceViewT<MoleculeType::Query>;

/**
 * @brief Device-side storage for batched molecules using AsyncDeviceVector.
 *
 * Owns the device memory and provides a view for kernel access.
 */
class MoleculesDevice {
 public:
  MoleculesDevice() = default;
  explicit MoleculesDevice(cudaStream_t stream) { setStream(stream); }

  /**
   * @brief Copy molecule data from host to device.
   * @param host The host-side molecule batch to copy
   * @param stream CUDA stream for async operations (optional, uses stored stream if not provided)
   */
  void copyFromHost(const MoleculesHost& host, cudaStream_t stream);
  void copyFromHost(const MoleculesHost& host) { copyFromHost(host, stream_); }

  /**
   * @brief Get a view suitable for passing to CUDA kernels.
   */
  template <MoleculeType Type> [[nodiscard]] MoleculesDeviceViewT<Type> view() const;

  void setStream(cudaStream_t stream);

 private:
  cudaStream_t stream_       = nullptr;
  int          numMolecules_ = 0;

  AsyncDeviceVector<int> batchAtomStarts_;

  // GPU-optimized packed data
  AsyncDeviceVector<AtomDataPacked>  atomDataPacked_;
  AsyncDeviceVector<AtomQueryMask>   atomQueryMasks_;
  AsyncDeviceVector<BondTypeCounts>  bondTypeCounts_;
  AsyncDeviceVector<TargetAtomBonds> targetAtomBonds_;
  AsyncDeviceVector<QueryAtomBonds>  queryAtomBonds_;

  // Boolean expression tree data for compound queries
  AsyncDeviceVector<AtomQueryTree>   atomQueryTrees_;
  AsyncDeviceVector<BoolInstruction> queryInstructions_;
  AsyncDeviceVector<AtomQueryMask>   queryLeafMasks_;
  AsyncDeviceVector<BondTypeCounts>  queryLeafBondCounts_;
  AsyncDeviceVector<int>             atomInstrStarts_;
  AsyncDeviceVector<int>             atomLeafMaskStarts_;
};

/**
 * @brief Add a molecule to an existing batch.
 * @param mol Pointer to the RDKit molecule to add
 * @param batch The batch to add the molecule to
 */
void addToBatch(const RDKit::ROMol* mol, MoleculesHost& batch);

/**
 * @brief Add a query molecule (from SMARTS) to an existing batch.
 *
 * Extracts query information from QueryAtom objects and populates atomQueries.
 * Supports AND, OR, and NOT combinations of query types via boolean expression trees.
 * XOR queries and recursive SMARTS ($(...)) throw an exception.
 *
 * @param mol Pointer to the RDKit molecule (typically parsed from SMARTS)
 * @param batch The batch to add the query molecule to
 */
void addQueryToBatch(const RDKit::ROMol* mol, MoleculesHost& batch);

/**
 * @brief Build a target molecule batch in parallel into existing storage.
 *
 * Uses direct parallel writing - each thread writes directly to the result
 * buffer at computed offsets. Reuses result's existing capacity when possible.
 *
 * @param result Output batch (will be overwritten)
 * @param numThreads Number of OpenMP threads to use
 * @param molecules Vector of molecule pointers
 * @param sortOrder Optional sort order (empty = use sequential order)
 */
void buildTargetBatchParallelInto(MoleculesHost&                          result,
                                  int                                     numThreads,
                                  const std::vector<const RDKit::ROMol*>& molecules,
                                  const std::vector<int>&                 sortOrder);

/**
 * @brief Add a query molecule with explicit child pattern ID mapping.
 *
 * Used when adding cached recursive patterns. The childPatternIds vector maps
 * local RecursiveStructure indices (in DFS order) to global pattern IDs.
 *
 * @param mol Pointer to the RDKit molecule
 * @param batch The batch to add the query molecule to
 * @param childPatternIds Map from local index (0, 1, ...) to global patternId
 */
void addQueryToBatch(const RDKit::ROMol* mol, MoleculesHost& batch, const std::vector<int>& childPatternIds);

/**
 * @brief Build a query molecule batch in parallel using OpenMP.
 *
 * Processes molecules in parallel when numThreads > 1. The molecules are added
 * in the order specified by sortOrder (or sequential order if sortOrder is empty).
 *
 * @param molecules Vector of molecule pointers
 * @param sortOrder Optional sort order (empty = use sequential order)
 * @param numThreads Number of OpenMP threads (1 = serial)
 * @return Populated MoleculesHost batch
 */
MoleculesHost buildQueryBatchParallel(const std::vector<const RDKit::ROMol*>& molecules,
                                      const std::vector<int>&                 sortOrder,
                                      int                                     numThreads);

/**
 * @brief Convert RDKit query description string to AtomQuery flags.
 * @param description The query description from RDKit (e.g., "AtomAtomicNum")
 * @return The corresponding AtomQuery flag value, or AtomQueryNone if unsupported
 */
AtomQuery atomQueryFromDescription(const std::string& description);

/**
 * @brief Build a query mask from packed atom data and query flags.
 *
 * Creates a precomputed mask and expected value pair for branchless GPU matching.
 * For each field specified in queryFlags, sets the corresponding mask byte to 0xFF
 * and the expected byte to the query atom's value.
 *
 * @param queryAtom The packed query atom data
 * @param queryFlags Bitmask of AtomQueryFlags indicating which fields to compare
 * @return AtomQueryMask with precomputed mask and expected values
 */
AtomQueryMask        buildQueryMask(const AtomDataPacked& queryAtom, AtomQuery queryFlags);
/**
 * @brief Extract recursive SMARTS patterns from a query molecule.
 *
 * Walks the query tree looking for RecursiveStructure nodes and extracts the
 * inner query molecules, including nested patterns. Patterns are extracted
 * depth-first and sorted by depth (leaves first) for level-by-level processing.
 *
 * Validates constraints:
 * - Maximum 32 total recursive patterns per query (including nested)
 *
 * @param mol The query molecule (typically parsed from SMARTS)
 * @return RecursivePatternInfo containing all found patterns sorted by depth
 * @throws std::runtime_error if constraints are violated
 */
RecursivePatternInfo extractRecursivePatterns(const RDKit::ROMol* mol);

/**
 * @brief Check if a SMARTS query contains recursive patterns.
 *
 * Quick check without full extraction. Useful for batch sorting.
 *
 * @param mol The query molecule to check
 * @return true if the query contains any recursive SMARTS ($(...))
 */
bool hasRecursiveSmarts(const RDKit::ROMol* mol);

/**
 * @brief Check if a target molecule requires RDKit fallback processing.
 *
 * Detects molecules with properties that exceed GPU processing limits:
 * - Atom degree > 8 (hypervalent atoms)
 * - Atom ring count > 15 (e.g., buckyballs)
 * - Ring bond count > 15
 * - Implicit H count > 15
 * - Heteroatom neighbor count > 15
 *
 * @param mol The molecule to check
 * @return true if the molecule cannot be processed on GPU and needs RDKit fallback
 */
bool requiresRDKitFallback(const RDKit::ROMol* mol);

/**
 * @brief Get the pipeline scheduling depth for a query in a batch.
 *
 * This is used for scheduling match stages after recursive SMARTS painting:
 * - Non-recursive queries return 0 (can run immediately).
 * - Recursive queries return maxRecursiveDepth + 1 so at least one paint pass
 *   occurs before matching.
 *
 * @param queriesHost The host batch containing the query
 * @param queryIdx Index of the query in the batch
 * @return 0 if no recursive patterns, otherwise maxRecursiveDepth + 1
 */
int getQueryPipelineDepth(const MoleculesHost& queriesHost, int queryIdx);

}  // namespace nvMolKit

#endif  // NVMOLKIT_MOLECULES_H
