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

#ifndef FMCS_CUDA_FMCS_GROW_CUH
#define FMCS_CUDA_FMCS_GROW_CUH

#include <cstdint>

#include "src/mcs/fmcs_cuda/fmcs_seed.cuh"
#include "src/mcs/fmcs_cuda/fmcs_topology.cuh"
#include "src/mcs/mcs_common/mcs_cooperative_copy.cuh"

namespace mcs {
namespace fmcs {

template <int maxAtoms, int maxBonds>
__device__ __forceinline__ void seedAddNewBondWithinThread(Seed<maxAtoms, maxBonds>& seed, const NewBond& bond) {
  seedAddBondWithinThread(seed, bond.bondIdx);
  if (bond.endAtomSeedIdx == NewBond::kNotInSeed) {
    seedAddAtomWithinThread(seed, bond.newAtomIdx);
  }
}

// Seed growth follows RDKit Seed::grow()'s three stage shape:
//
//   Stage 0 (fillNewBonds + biggest-child match): collect every outgoing
//   query bond and build the all-bonds-added child.
//   Stage 1: for each new bond build the singleton extension child; failed
//   singleton matches remove that NewBond from Stage 2.
//   Stage 2: enumerate the non-singleton subsets of the surviving NewBonds.
//
// The kernel owns the Stage 0/1/2 control flow because it has the queue,
// incumbent, and substructure scratch.  This file supplies the shared NewBond
// enumeration and seed-patching helpers used by those stages.
//
// Every helper is cooperative-group-parallel across bonds.

/// Cooperative: bond-centric scan that distributes query bonds across
/// @p group's lanes.  A query bond becomes a NewBond entry iff it is
/// (a) not in @c seed.excludedBonds and (b) incident to at least one
/// bit in @c seed.lastAddedAtoms (i.e., one endpoint was newly added in
/// the most recent grow step).  Each surviving bond is classified:
///
///   - Both endpoints already in @c seed.atoms -> ring-closing.
///     @c endAtomSeedIdx is set to a non-sentinel value to flag this
///     to the kernel's Stage 0 patch loop (which then skips
///     @c seedAddAtom for this bond).
///   - Otherwise -> atom-adding.  @c newAtomIdx holds the query atom
///     index that is NOT yet in the seed; @c endAtomSeedIdx is set to
///     @c NewBond::kNotInSeed.
///
/// Lanes append to @p outBonds via @c atomicAdd on @p outCount.  If
/// the count would exceed @p maxNewBonds the function returns false
/// and clamps @p outCount to @p maxNewBonds (slots beyond that bound
/// are not written).  No I/O ordering on @p outBonds is guaranteed --
/// the order depends on lane race outcomes.
template <int maxAtoms, int maxBonds, class GroupT>
__device__ __forceinline__ bool fillNewBondsCooperative(const GroupT&                   group,
                                                        const Seed<maxAtoms, maxBonds>& seed,
                                                        const DeviceCsrView&            queryTopology,
                                                        NewBond*                        outBonds,
                                                        int*                            outCount,
                                                        int                             maxNewBonds) {
  using SeedT                    = Seed<maxAtoms, maxBonds>;
  using AtomWord                 = typename SeedT::atom_word_type;
  using BondWord                 = typename SeedT::bond_word_type;
  constexpr int kAtomBitsPerWord = SeedT::kAtomBitsPerWord;
  constexpr int kBondBitsPerWord = SeedT::kBondBitsPerWord;

  const int laneRank  = static_cast<int>(group.thread_rank());
  const int laneCount = static_cast<int>(group.num_threads());

  if (laneRank == 0)
    *outCount = 0;
  group.sync();

  for (int q = laneRank; q < queryTopology.numBonds; q += laneCount) {
    // Excluded check first (cheapest).
    const BondWord excludedBondsWord = seed.excludedBonds[q / kBondBitsPerWord];
    if ((excludedBondsWord >> (q % kBondBitsPerWord)) & 1)
      continue;

    // Decode this query bond's endpoints.
    const std::uint32_t queryEndpoints = queryTopology.bondEndpoints[q];
    const int           queryEndpointU = static_cast<int>(queryEndpoints >> kBondEndpointShift);
    const int           queryEndpointV = static_cast<int>(queryEndpoints & kBondEndpointMask);

    // Bond is a "new boundary" candidate iff at least one endpoint
    // was added in the most recent grow step.
    const AtomWord lastAddedWordU = seed.lastAddedAtoms[queryEndpointU / kAtomBitsPerWord];
    const AtomWord lastAddedWordV = seed.lastAddedAtoms[queryEndpointV / kAtomBitsPerWord];
    const bool     uIsNewlyAdded  = (lastAddedWordU >> (queryEndpointU % kAtomBitsPerWord)) & 1;
    const bool     vIsNewlyAdded  = (lastAddedWordV >> (queryEndpointV % kAtomBitsPerWord)) & 1;
    if (!uIsNewlyAdded && !vIsNewlyAdded)
      continue;

    // Classify ring-closing vs atom-adding via seed.atoms membership.
    const AtomWord seedAtomsWordU = seed.atoms[queryEndpointU / kAtomBitsPerWord];
    const AtomWord seedAtomsWordV = seed.atoms[queryEndpointV / kAtomBitsPerWord];
    const bool     uInSeed        = (seedAtomsWordU >> (queryEndpointU % kAtomBitsPerWord)) & 1;
    const bool     vInSeed        = (seedAtomsWordV >> (queryEndpointV % kAtomBitsPerWord)) & 1;

    NewBond newBond;
    newBond.bondIdx = static_cast<uint16_t>(q);
    newBond.alive   = true;
    if (uInSeed && vInSeed) {
      // Ring-closing: both endpoints already in the seed.  newAtomIdx
      // is unused by the Stage 0 patch loop in this case; we just
      // need endAtomSeedIdx to be non-sentinel to flag ring-closing.
      newBond.newAtomIdx     = static_cast<uint16_t>(queryEndpointV);
      newBond.endAtomSeedIdx = static_cast<uint16_t>(queryEndpointU);
    } else {
      newBond.newAtomIdx     = static_cast<uint16_t>(uInSeed ? queryEndpointV : queryEndpointU);
      newBond.endAtomSeedIdx = NewBond::kNotInSeed;
    }

    const int slot = atomicAdd(outCount, 1);
    if (slot < maxNewBonds) {
      outBonds[slot] = newBond;
    }
    // Otherwise: no out-of-bounds write, but outCount has overrun
    // maxNewBonds and we'll clamp + return false below.
  }
  group.sync();

  // Read the pre-clamp count so all lanes can return a consistent
  // success/overflow signal even after lane 0 clamps below.
  const int totalAttempted = *outCount;
  group.sync();
  if (laneRank == 0 && totalAttempted > maxNewBonds) {
    *outCount = maxNewBonds;
  }
  group.sync();
  return totalAttempted <= maxNewBonds;
}

/// Stage 1 driver.  Despite the @c Cooperative suffix, the parallelism
/// here is shallow: the outer loop walks @p bonds serially (uniform
/// across all lanes of @p group), and within each iteration the only
/// truly group-parallel step is the @c warpCopy of @c parent into
/// @p childWorkspace.  Lane 0 then patches in the singleton bond /
/// atom; the per-bond match call is whatever @p matchFn does (when
/// the kernel passes @ref matchIncrementalFastCooperative the inner
/// target-bond scan IS lane-parallel); @p childSink is typically lane
/// 0 only (cache.insert + queue.push).
///
/// In other words: this function is cooperative only for the
/// parent->child copy and for delegating to its callbacks; it does
/// not itself parallelize across @p bonds.  Future refactors aiming
/// for cross-bond parallelism would need a different structure -- the
/// per-bond commits depend on shared visited-state from the previous
/// bond's match call, which is incompatible with running them
/// concurrently.
///
/// @p childWorkspace must be a per-group shared slot the caller owns
/// -- the kernel reuses the same buffer it already allocated for the
/// Stage 0 biggest-child build.
template <int maxAtoms, int maxBonds, int maxTA, int maxTB, class GroupT, class MatchFn, class ChildSink>
__device__ __forceinline__ void pruneIndividualBondsCooperative(
  const GroupT&                                       group,
  const QueuedSeed<maxAtoms, maxBonds, maxTA, maxTB>& parent,
  QueuedSeed<maxAtoms, maxBonds, maxTA, maxTB>&       childWorkspace,
  NewBond*                                            bonds,
  int                                                 nBonds,
  MatchFn&&                                           matchFn,
  ChildSink&&                                         childSink) {
  using QueuedT = QueuedSeed<maxAtoms, maxBonds, maxTA, maxTB>;
  for (int i = 0; i < nBonds; ++i) {
    if (!bonds[i].alive)
      continue;

    // Reset the workspace to a copy of the parent (group-cooperative).
    warpCopy(group, &childWorkspace, &parent, sizeof(QueuedT));
    group.sync();

    // Lane 0 patches in this single new bond / atom on top of the parent.
    if (group.thread_rank() == 0) {
      seedBeginGrowStepWithinThread(childWorkspace.seed);
      seedAddNewBondWithinThread(childWorkspace.seed, bonds[i]);
    }
    group.sync();

    // Try to extend the parent's recorded match by this single bond.
    if (matchFn(childWorkspace)) {
      childSink(childWorkspace);
    } else if (group.thread_rank() == 0) {
      bonds[i].alive = false;
    }
    group.sync();
  }
}

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_GROW_CUH
