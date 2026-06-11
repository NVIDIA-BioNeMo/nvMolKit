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

#ifndef NVMOLKIT_BOOLEAN_TREE_CUH
#define NVMOLKIT_BOOLEAN_TREE_CUH

#include <cstdint>

#include "src/substruct/atom_data_packed.h"

#ifdef __CUDACC__
#define HD_CALLABLE __host__ __device__ __forceinline__
#else
#define HD_CALLABLE inline
#endif

namespace nvMolKit {

/**
 * @brief Boolean operation type for query expression evaluation.
 */
enum class BoolOp : uint8_t {
  Leaf,            ///< Evaluate AtomQueryMask against target atom
  And,             ///< Binary AND of two operands
  Or,              ///< Binary OR of two operands
  Not,             ///< Unary NOT of single operand
  RecursiveMatch,  ///< Check if target atom has recursive pattern bit set
  GreaterThan,     ///< Compare: getField(target, fieldId) > value
  LessEqual,       ///< Compare: getField(target, fieldId) <= value
  GreaterEqual,    ///< Compare: getField(target, fieldId) >= value
  Range            ///< Compare: minVal <= getField(target, fieldId) <= maxVal
};

/**
 * @brief Field identifiers for comparison operations.
 *
 * Used by comparison BoolOps to specify which atom field to compare.
 */
enum class CompareField : uint8_t {
  MinRingSize = 0,
  NumRings,
  RingBondCount,
  NumImplicitHs,
  NumHeteroatomNeighbors,
  TotalValence,
  Degree,
  NumExplicitHs
};

/**
 * @brief Extract a field value from packed atom data for comparison.
 */
HD_CALLABLE uint8_t getAtomField(const AtomDataPacked& atom, CompareField field) {
  switch (field) {
    case CompareField::MinRingSize:
      return atom.minRingSize();
    case CompareField::NumRings:
      return atom.numRings();
    case CompareField::RingBondCount:
      return atom.ringBondCount();
    case CompareField::NumImplicitHs:
      return atom.numImplicitHs();
    case CompareField::NumHeteroatomNeighbors:
      return atom.numHeteroatomNeighbors();
    case CompareField::TotalValence:
      return atom.totalValence();
    case CompareField::Degree:
      return atom.degree();
    case CompareField::NumExplicitHs:
      return atom.numExplicitHs();
    default:
      return 0;
  }
}

/**
 * @brief A single instruction in the boolean expression evaluation sequence.
 *
 * Instructions are executed in post-order and write a boolean result into the
 * scratch array at index @c dst. Operand mapping by op:
 * - Leaf: @c scratch[dst] = atomMatchesPacked(target, leafMasks[auxArg])
 * - And/Or: @c scratch[dst] = scratch[src1] &/| scratch[src2]
 * - Not: @c scratch[dst] = !scratch[src1]
 * - RecursiveMatch: @c scratch[dst] = (recursiveMatchBits >> auxArg) & 1
 * - GreaterThan/LessEqual/GreaterEqual:
 *   @c field = static_cast<CompareField>(src1), @c rhs = src2, compare
 *   @c getAtomField(target, field) against @c rhs
 * - Range:
 *   @c field = static_cast<CompareField>(src1), @c min = src2, @c max = auxArg,
 *   @c scratch[dst] = (min <= getAtomField(target, field) && getAtomField(target, field) <= max)
 */
struct BoolInstruction {
  BoolOp  op;    ///< Operation type
  uint8_t dst;   ///< Destination index in scratch array
  uint8_t src1;  ///< Left operand index, or fieldId for comparisons
  uint8_t src2;  ///< Right operand index, or value/minVal for comparisons
  uint8_t
    auxArg;  ///< Aux operand: leaf mask index (Leaf), pattern id (RecursiveMatch), max (Range) depending on use case.

  HD_CALLABLE static BoolInstruction makeLeaf(uint8_t dst, uint8_t maskIdx) {
    return BoolInstruction{BoolOp::Leaf, dst, 0, 0, maskIdx};
  }

  HD_CALLABLE static BoolInstruction makeAnd(uint8_t dst, uint8_t src1, uint8_t src2) {
    return BoolInstruction{BoolOp::And, dst, src1, src2, 0};
  }

  HD_CALLABLE static BoolInstruction makeOr(uint8_t dst, uint8_t src1, uint8_t src2) {
    return BoolInstruction{BoolOp::Or, dst, src1, src2, 0};
  }

  HD_CALLABLE static BoolInstruction makeNot(uint8_t dst, uint8_t src) {
    return BoolInstruction{BoolOp::Not, dst, src, 0, 0};
  }

  HD_CALLABLE static BoolInstruction makeRecursiveMatch(uint8_t dst, uint8_t patternId) {
    return BoolInstruction{BoolOp::RecursiveMatch, dst, 0, 0, patternId};
  }

  HD_CALLABLE static BoolInstruction makeGreaterThan(uint8_t dst, CompareField field, uint8_t value) {
    return BoolInstruction{BoolOp::GreaterThan, dst, static_cast<uint8_t>(field), value, 0};
  }

  HD_CALLABLE static BoolInstruction makeLessEqual(uint8_t dst, CompareField field, uint8_t value) {
    return BoolInstruction{BoolOp::LessEqual, dst, static_cast<uint8_t>(field), value, 0};
  }

  HD_CALLABLE static BoolInstruction makeGreaterEqual(uint8_t dst, CompareField field, uint8_t value) {
    return BoolInstruction{BoolOp::GreaterEqual, dst, static_cast<uint8_t>(field), value, 0};
  }

  HD_CALLABLE static BoolInstruction makeRange(uint8_t dst, CompareField field, uint8_t minVal, uint8_t maxVal) {
    return BoolInstruction{BoolOp::Range, dst, static_cast<uint8_t>(field), minVal, maxVal};
  }
};

static_assert(sizeof(BoolInstruction) == 5, "BoolInstruction must be exactly 5 bytes");

/**
 * @brief Metadata for a single query atom's boolean expression tree.
 *
 * Each query atom has its own tree describing how to combine leaf mask checks.
 * For simple AND-only queries, numInstructions=1 and scratchSize=1.
 */
struct AtomQueryTree {
  uint8_t numLeaves       = 0;  ///< Number of AtomQueryMask entries for this atom
  uint8_t numInstructions = 0;  ///< Length of instruction sequence
  uint8_t scratchSize     = 0;  ///< Number of scratch slots needed for evaluation
  uint8_t resultIdx       = 0;  ///< Index in scratch where final result is stored
};

static_assert(sizeof(AtomQueryTree) == 4, "AtomQueryTree must be exactly 4 bytes");

constexpr int kMaxBoolScratchSize = 256;
/**
 * @brief Evaluate a boolean expression tree for atom matching.
 *
 * Executes the instruction sequence to determine if a target atom matches
 * a compound query expression (AND/OR/NOT combinations).
 *
 * @param targetPacked The target atom's packed data
 * @param targetBonds The target atom's bond type counts (may be nullptr if checkBonds=false)
 * @param leafMasks Pointer to first leaf mask for this query atom
 * @param leafBondCounts Pointer to first leaf bond count for this query atom (may be nullptr if checkBonds=false)
 * @param instructions Pointer to first instruction for this query atom
 * @param tree Tree metadata (num instructions, scratch size, result index)
 * @param recursiveMatchBits Per-pair recursive match bits for this target atom (32 bits for patterns 0-31)
 * @tparam checkBonds If true, also check bond count requirements (for substructure search).
 *                   If false, only check atom properties (for label matrix compatibility).
 * @return true if target atom matches the compound query
 */
template <bool checkBonds = true>
HD_CALLABLE bool evaluateBoolTree(const AtomDataPacked*  targetPacked,
                                  const BondTypeCounts*  targetBonds,
                                  const AtomQueryMask*   leafMasks,
                                  const BondTypeCounts*  leafBondCounts,
                                  const BoolInstruction* instructions,
                                  const AtomQueryTree&   tree,
                                  uint32_t               recursiveMatchBits = 0) {
  // Empty tree (e.g., wildcard atom *) - atom properties always match,
  // but still need to check bond counts if requested
  if (tree.numInstructions == 0) {
    if constexpr (checkBonds) {
      // For empty trees, leaf 0 still holds the bond count requirements
      return tree.numLeaves > 0 ? bondCountsMatchPacked(*targetBonds, leafBondCounts[0]) : true;
    }
    return true;
  }

  uint8_t scratch[kMaxBoolScratchSize];

  for (int i = 0; i < tree.numInstructions; ++i) {
    const BoolInstruction& instr = instructions[i];

    switch (instr.op) {
      case BoolOp::Leaf: {
        bool match = atomMatchesPacked(*targetPacked, leafMasks[instr.auxArg]);
        if constexpr (checkBonds) {
          match &= bondCountsMatchPacked(*targetBonds, leafBondCounts[instr.auxArg]);
        }
        scratch[instr.dst] = match ? 1 : 0;
        break;
      }
      case BoolOp::And:
        scratch[instr.dst] = scratch[instr.src1] & scratch[instr.src2];
        break;
      case BoolOp::Or:
        scratch[instr.dst] = scratch[instr.src1] | scratch[instr.src2];
        break;
      case BoolOp::Not:
        scratch[instr.dst] = scratch[instr.src1] ? 0 : 1;
        break;
      case BoolOp::RecursiveMatch:
        scratch[instr.dst] = ((recursiveMatchBits >> instr.auxArg) & 1u) ? 1 : 0;
        break;
      case BoolOp::GreaterThan: {
        const uint8_t fieldVal = getAtomField(*targetPacked, static_cast<CompareField>(instr.src1));
        scratch[instr.dst]     = (fieldVal > instr.src2) ? 1 : 0;
        break;
      }
      case BoolOp::LessEqual: {
        const uint8_t fieldVal = getAtomField(*targetPacked, static_cast<CompareField>(instr.src1));
        scratch[instr.dst]     = (fieldVal <= instr.src2) ? 1 : 0;
        break;
      }
      case BoolOp::GreaterEqual: {
        const uint8_t fieldVal = getAtomField(*targetPacked, static_cast<CompareField>(instr.src1));
        scratch[instr.dst]     = (fieldVal >= instr.src2) ? 1 : 0;
        break;
      }
      case BoolOp::Range: {
        const uint8_t fieldVal = getAtomField(*targetPacked, static_cast<CompareField>(instr.src1));
        scratch[instr.dst]     = (fieldVal >= instr.src2 && fieldVal <= instr.auxArg) ? 1 : 0;
        break;
      }
    }
  }

  return scratch[tree.resultIdx] != 0;
}

}  // namespace nvMolKit

#undef HD_CALLABLE

#endif  // NVMOLKIT_BOOLEAN_TREE_CUH
