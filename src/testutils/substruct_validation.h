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

#pragma once

#include <cuda_runtime.h>
#include <GraphMol/ROMol.h>

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "src/substruct/substruct_types.h"

namespace nvMolKit {

/**
 * @brief Get substructure matches using RDKit.
 *
 * @param target Target molecule
 * @param query Query molecule (typically from SMARTS)
 * @param uniquify If true, return only unique matches
 * @return Vector of matches, where each match is a vector of target atom indices
 *         indexed by query atom position
 */
std::vector<std::vector<int>> getRDKitSubstructMatches(const RDKit::ROMol& target,
                                                       const RDKit::ROMol& query,
                                                       bool                uniquify = true);

/**
 * @brief Compare two sets of matches (order-independent).
 *
 * @param gpuMatches Matches from GPU results
 * @param rdkitMatches Matches from RDKit or another source
 * @return true if the match sets are identical
 */
bool matchSetsEqual(const std::vector<std::vector<int>>& gpuMatches, const std::vector<std::vector<int>>& rdkitMatches);

/**
 * @brief Get a string name for a substruct algorithm.
 * @param algo The algorithm enum value
 * @return Human-readable name for the algorithm
 */
std::string algorithmName(SubstructAlgorithm algo);

/**
 * @brief Validation results from comparing GPU matches to RDKit ground truth.
 */
struct SubstructValidationResult {
  int  totalPairs        = 0;
  int  matchingPairs     = 0;
  int  mismatchedPairs   = 0;
  int  wrongMappingPairs = 0;  ///< Pairs where count matches but mappings differ
  bool allMatch          = false;

  /// Details about count mismatches: (targetIdx, queryIdx, gpuCount, rdkitCount)
  std::vector<std::tuple<int, int, int, int>> mismatches;

  /// Details about mapping mismatches: (targetIdx, queryIdx)
  std::vector<std::pair<int, int>> mappingMismatches;
};

/**
 * @brief Compare GPU substructure match results against RDKit ground truth.
 *
 * @param results GPU results from getSubstructMatches
 * @param targetMols Target molecules (parallel to results.numTargets)
 * @param queryMols Query molecules (parallel to results.numQueries)
 * @param uniquify If true, compare against uniquified RDKit matches
 * @return Validation result with mismatch details
 */
SubstructValidationResult validateAgainstRDKit(const SubstructSearchResults&                     results,
                                               const std::vector<std::unique_ptr<RDKit::ROMol>>& targetMols,
                                               const std::vector<std::unique_ptr<RDKit::ROMol>>& queryMols,
                                               bool                                              uniquify = false);

/**
 * @brief Print validation results to stdout.
 * @param result The validation result to print
 * @param algorithmName Optional algorithm name to include in output
 */
void printValidationResult(const SubstructValidationResult& result, const std::string& algorithmName = "");

/**
 * @brief Print detailed validation results including SMILES/SMARTS strings and actual matches.
 * @param result The validation result to print
 * @param gpuResults GPU match results for extracting actual matches
 * @param targetMols Target molecules for RDKit matching
 * @param queryMols Query molecules for RDKit matching
 * @param targetSmiles Target SMILES strings
 * @param querySmarts Query SMARTS strings
 * @param algorithmName Optional algorithm name to include in output
 * @param maxDetails Maximum number of detailed mismatches to print (default 5)
 */
void printValidationResultDetailed(const SubstructValidationResult&                  result,
                                   const SubstructSearchResults&                     gpuResults,
                                   const std::vector<std::unique_ptr<RDKit::ROMol>>& targetMols,
                                   const std::vector<std::unique_ptr<RDKit::ROMol>>& queryMols,
                                   const std::vector<std::string>&                   targetSmiles,
                                   const std::vector<std::string>&                   querySmarts,
                                   const std::string&                                algorithmName = "",
                                   int                                               maxDetails    = 5,
                                   bool                                              uniquify      = false);

/**
 * @brief Compute the expected label matrix using RDKit atom matching.
 *
 * For each (targetAtom, queryAtom) pair, determines if the query atom's
 * constraints are satisfied by the target atom using RDKit's Match() method.
 *
 * @param targetMol Target molecule
 * @param queryMol Query molecule (typically from SMARTS)
 * @return 2D matrix of compatibility flags indexed [targetAtom][queryAtom]
 */
std::vector<std::vector<uint8_t>> computeRDKitLabelMatrix(const RDKit::ROMol& targetMol, const RDKit::ROMol& queryMol);

/**
 * @brief Result from comparing GPU vs RDKit label matrices.
 */
struct LabelMatrixComparisonResult {
  int  numTargetAtoms   = 0;
  int  numQueryAtoms    = 0;
  int  totalComparisons = 0;
  int  falsePositives   = 0;  ///< GPU says match, RDKit says no
  int  falseNegatives   = 0;  ///< GPU says no match, RDKit says yes
  bool allMatch         = false;

  /// Details: (targetAtomIdx, queryAtomIdx, gpuResult, rdkitResult)
  std::vector<std::tuple<int, int, bool, bool>> mismatches;
};

/**
 * @brief Compute a GPU label matrix for a single target/query pair.
 *
 * Handles recursive SMARTS preprocessing if the query contains recursive patterns.
 *
 * @param targetMol Target molecule
 * @param queryMol Query molecule (typically from SMARTS)
 * @param stream CUDA stream to use
 * @return 2D matrix of compatibility flags indexed [targetAtom][queryAtom]
 */
std::vector<std::vector<uint8_t>> computeGpuLabelMatrix(const RDKit::ROMol& targetMol,
                                                        const RDKit::ROMol& queryMol,
                                                        cudaStream_t        stream);

/**
 * @brief Compare GPU and RDKit label matrices for a single target/query pair.
 *
 * @param targetMol Target molecule
 * @param queryMol Query molecule
 * @param stream CUDA stream to use
 * @return Comparison result with mismatch details
 */
LabelMatrixComparisonResult compareLabelMatrices(const RDKit::ROMol& targetMol,
                                                 const RDKit::ROMol& queryMol,
                                                 cudaStream_t        stream);

}  // namespace nvMolKit
