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

#include "src/morgan_fingerprint_cpu.h"

#include <GraphMol/Fingerprints/FingerprintUtil.h>
#include <GraphMol/Fingerprints/MorganGenerator.h>
#include <omp.h>

#include <RDGeneral/hash/hash.hpp>

#include "src/utils/openmp_helpers.h"

namespace nvMolKit {

namespace {
constexpr int kMaxBondsPerAtom = 8;
using OutputType               = std::uint64_t;

//! Class to hold the atom environment
//! Note that without additional output features, this really just needs to be the hash value, but
//! we're keeping the structure for future work.
class MorganAtomEnv {
  OutputType d_code;

 public:
  [[nodiscard]] OutputType getBitId() const { return d_code; }

  /**
   \brief Construct a new MorganAtomEnv object

   \param code bit id generated from this environment
   \param atomId atom id of the atom at the center of this environment
   \param layer radius of this environment
   */
  MorganAtomEnv(const std::uint32_t                 code,
                [[maybe_unused]] const unsigned int atomId,
                [[maybe_unused]] const unsigned int layer)
      : d_code(code) {}
};

namespace {

using AccumTuple = std::tuple<boost::dynamic_bitset<>, uint32_t, unsigned int>;

}  // namespace

// Adapted from RDKit environment code.
std::vector<MorganAtomEnv> getEnvironments(const RDKit::ROMol&               mol,
                                           const std::uint32_t               radius,
                                           const bool                        includeChirality,
                                           const bool                        onlyNonzeroInvariants,
                                           const std::vector<std::uint32_t>* atomInvariants,
                                           const std::vector<std::uint32_t>* bondInvariants) {
  PRECONDITION(atomInvariants && (atomInvariants->size() >= mol.getNumAtoms()), "bad atom invariants size");
  PRECONDITION(bondInvariants && (bondInvariants->size() >= mol.getNumBonds()), "bad bond invariants size");
  const unsigned int nAtoms = mol.getNumAtoms();

  const std::uint32_t        maxNumResults = (radius + 1) * nAtoms;
  std::vector<MorganAtomEnv> result;
  result.reserve(maxNumResults);  // Duplicates will make this less
  std::vector<OutputType> currentInvariants(atomInvariants->size());
  std::vector<OutputType> nextLayerInvariants(nAtoms);

  std::copy(atomInvariants->begin(), atomInvariants->end(), currentInvariants.begin());

  boost::dynamic_bitset<> includeAtoms(nAtoms);
  includeAtoms.set();
  boost::dynamic_bitset<> chiralAtoms(nAtoms);

  // these are the neighborhoods that have already been added
  // to the fingerprint
  std::unordered_set<boost::dynamic_bitset<>> neighborhoods;
  neighborhoods.reserve(maxNumResults);  // More than we'll need due to clashes

  // these are the environments around each atom:
  std::vector<boost::dynamic_bitset<>> atomNeighborhoods(nAtoms, boost::dynamic_bitset<>(mol.getNumBonds()));
  std::vector<boost::dynamic_bitset<>> roundAtomNeighborhoods = atomNeighborhoods;

  boost::dynamic_bitset<> deadAtoms(nAtoms);

  // will hold up to date invariants of neighboring atoms with bond
  // types, these invariants hold information from atoms around radius
  // as big as current layer around the current atom
  std::vector<std::pair<int32_t, uint32_t>> neighborhoodInvariants(kMaxBondsPerAtom);

  // if df_onlyNonzeroInvariants is set order the atoms to make sure atoms
  // with zero invariants are processed last so that in case of duplicate
  // environments atoms with non-zero invariants are used
  std::vector<unsigned int> atomOrder(nAtoms);

  // holds atoms in the environment (neighborhood) for the current layer for
  // each atom, starts with the immediate neighbors of atoms and expands
  // with every iteration
  std::vector<AccumTuple> allNeighborhoodsThisRound;
  allNeighborhoodsThisRound.reserve(nAtoms);

  if (onlyNonzeroInvariants) {
    std::vector<std::pair<int32_t, uint32_t>> ordering;
    for (std::uint32_t i = 0; i < nAtoms; ++i) {
      if (currentInvariants[i] == 0) {
        ordering.emplace_back(1, i);
      } else {
        ordering.emplace_back(0, i);
      }
    }
    std::sort(ordering.begin(), ordering.end());
    for (unsigned int i = 0; i < nAtoms; ++i) {
      atomOrder[i] = ordering[i].second;
    }
  } else {
    for (unsigned int i = 0; i < nAtoms; ++i) {
      atomOrder[i] = i;
    }
  }

  // add the round 0 invariants to the result
  for (unsigned int i = 0; i < nAtoms; ++i) {
    if (includeAtoms[i]) {
      if (!onlyNonzeroInvariants || currentInvariants[i] != 0) {
        result.emplace_back(currentInvariants[i], i, 0);
      }
    }
  }

  // now do our subsequent rounds:
  for (unsigned int layer = 0; layer < radius; ++layer) {
    // will hold bit ids calculated this round to be used as invariants next
    // round

    for (auto atomIdx : atomOrder) {
      // skip atoms which will not generate unique environments
      // (neighborhoods) anymore
      if (!deadAtoms[atomIdx]) {
        const RDKit::Atom* tAtom = mol.getAtomWithIdx(atomIdx);
        if (tAtom->getDegree() == 0) {
          deadAtoms.set(atomIdx, true);
          continue;
        }

        RDKit::ROMol::OEDGE_ITER beg;
        RDKit::ROMol::OEDGE_ITER end;
        boost::tie(beg, end) = mol.getAtomBonds(tAtom);

        // add up to date invariants of neighbors
        // This should keep capacity, so reallocation only needed if we haven't seen a molecule of this size.
        // Fancier but buggier would be to overwrite rather than clear, but then we'd need logic to check the size.
        neighborhoodInvariants.clear();
        while (beg != end) {
          const RDKit::Bond* bond                         = mol[*beg];
          roundAtomNeighborhoods[atomIdx][bond->getIdx()] = true;

          const unsigned int oIdx = bond->getOtherAtomIdx(atomIdx);
          roundAtomNeighborhoods[atomIdx] |= atomNeighborhoods[oIdx];

          auto bondType = static_cast<int32_t>((*bondInvariants)[bond->getIdx()]);
          neighborhoodInvariants.emplace_back(bondType, currentInvariants[oIdx]);

          ++beg;
        }

        // sort the neighbor list:
        std::sort(neighborhoodInvariants.begin(), neighborhoodInvariants.end());
        // and now calculate the new invariant and test if the atom is newly
        // "chiral"
        std::uint32_t invar = layer;

        gboost::hash_combine(invar, currentInvariants[atomIdx]);

        bool looksChiral = (tAtom->getChiralTag() != RDKit::Atom::CHI_UNSPECIFIED);
        for (auto it = neighborhoodInvariants.begin(); it != neighborhoodInvariants.end(); ++it) {
          // add the contribution to the new invariant:
          gboost::hash_combine(invar, *it);

          // update our "chirality":
          // NOLINTBEGIN
          if (includeChirality && looksChiral && chiralAtoms[atomIdx]) {
            if (it->first != static_cast<int32_t>(RDKit::Bond::SINGLE)) {
              looksChiral = false;
            } else if (it != neighborhoodInvariants.begin() && it->second == (it - 1)->second) {
              looksChiral = false;
            }
          }
          // NOLINTEND
        }

        if (includeChirality && looksChiral) {
          chiralAtoms[atomIdx] = true;
          // add an extra value to the invariant to reflect chirality:
          std::string cip;
          tAtom->getPropIfPresent(RDKit::common_properties::_CIPCode, cip);
          if (cip == "R") {
            gboost::hash_combine(invar, 3);
          } else if (cip == "S") {
            gboost::hash_combine(invar, 2);
          } else {
            gboost::hash_combine(invar, 1);
          }
        }

        // this rounds bit id will be next rounds atom invariant, so we save
        // it here
        nextLayerInvariants[atomIdx] = static_cast<OutputType>(invar);

        // store the environment that generated this bit id along with the bit
        // id and the atom id
        allNeighborhoodsThisRound.emplace_back(roundAtomNeighborhoods[atomIdx],
                                               static_cast<OutputType>(invar),
                                               atomIdx);
      }
    }

    std::sort(allNeighborhoodsThisRound.begin(), allNeighborhoodsThisRound.end());

    for (const auto& iter : allNeighborhoodsThisRound) {
      // if we haven't seen this exact environment before, add it to the
      // result
      if (neighborhoods.count(std::get<0>(iter)) == 0) {
        if (!onlyNonzeroInvariants || (*atomInvariants)[std::get<2>(iter)] != 0) {
          if (includeAtoms[std::get<2>(iter)]) {
            result.emplace_back(std::get<1>(iter), std::get<2>(iter), layer + 1);

            neighborhoods.insert(std::get<0>(iter));
          }
        }
      } else {
        // we have seen this exact environment before, this atom
        // is now out of consideration:
        deadAtoms[std::get<2>(iter)] = true;
      }
    }
    allNeighborhoodsThisRound.clear();

    // the invariants from this round become the next round invariants:
    currentInvariants.swap(nextLayerInvariants);
    std::fill(nextLayerInvariants.begin(), nextLayerInvariants.end(), 0);
    // this rounds calculated neighbors will be next rounds initial neighbors,
    // so the radius can grow every iteration
    atomNeighborhoods = roundAtomNeighborhoods;
  }

  return result;
}

std::unique_ptr<RDKit::SparseIntVect<OutputType>> computeFpFromEnvironments(
  const std::vector<MorganAtomEnv>& atomEnvironments,
  const std::uint64_t               fpSize) {
  auto res = std::make_unique<RDKit::SparseIntVect<OutputType>>(fpSize);
  // iterate over every atom environment and generate bit-ids that will make up
  // the fingerprint
  for (const auto& env : atomEnvironments) {
    const OutputType seed = env.getBitId();

    auto bitId = seed;
    if (fpSize != 0) {
      bitId %= fpSize;
    }

    res->setVal(bitId, res->getVal(bitId) + 1);
  }

  return res;
}

}  // namespace

namespace internal {

std::unique_ptr<ExplicitBitVect> getFingerprintImpl(const RDKit::ROMol& mol,
                                                    const std::uint32_t radius,
                                                    const std::uint32_t fpSize) {
  auto atomInvariantsGenerator =
    std::make_unique<RDKit::MorganFingerprint::MorganAtomInvGenerator>(/*includeRingMembership=*/true);
  auto bondInvariantsGenerator =
    std::make_unique<RDKit::MorganFingerprint::MorganBondInvGenerator>(/*useBondTypes=*/true, /*useChirality=*/false);

  const std::unique_ptr<std::vector<std::uint32_t>> atomInvariants(atomInvariantsGenerator->getAtomInvariants(mol));
  const std::unique_ptr<std::vector<std::uint32_t>> bondInvariants(bondInvariantsGenerator->getBondInvariants(mol));

  auto atomEnvironments = getEnvironments(mol,
                                          radius,
                                          /*includeChirality=*/false,
                                          /*onlyNonZeroInvariants=*/false,
                                          atomInvariants.get(),
                                          bondInvariants.get());

  auto tempResult = computeFpFromEnvironments(atomEnvironments, fpSize);

  auto result = std::make_unique<ExplicitBitVect>(fpSize);  // NOLINT(cppcoreguidelines-owning-memory)
  for (auto val : tempResult->getNonzeroElements()) {
    result->setBit(val.first);
  }

  return result;
}

}  // namespace internal

MorganFingerprintCpuGenerator::MorganFingerprintCpuGenerator(std::uint32_t radius, std::uint32_t fpSize)
    : radius_(radius),
      fpSize_(fpSize) {}

std::unique_ptr<ExplicitBitVect> MorganFingerprintCpuGenerator::GetFingerprint(
  const RDKit::ROMol&                      mol,
  std::optional<FingerprintComputeOptions> computeOptions) const {
  std::vector<const RDKit::ROMol*> molView;
  molView.push_back(&mol);
  return std::move(GetFingerprints(molView, computeOptions)[0]);
}
std::vector<std::unique_ptr<ExplicitBitVect>> MorganFingerprintCpuGenerator::GetFingerprints(
  const std::vector<const RDKit::ROMol*>&  mols,
  std::optional<FingerprintComputeOptions> computeOptions) const {
  // cppcheck-suppress-begin unreadVariable
  // NOLINTNEXTLINE (clang-analyzer-deadcode.DeadStores)
  const size_t numCpuThreads =
    computeOptions.value_or(FingerprintComputeOptions()).numCpuThreads.value_or(omp_get_max_threads());
  // cppcheck-suppress-end unreadVariable

  std::vector<std::unique_ptr<ExplicitBitVect>> fingerprints(mols.size());
  detail::OpenMPExceptionRegistry               exceptionRegistry;
#pragma omp parallel for default(none) shared(fingerprints, mols, exceptionRegistry) num_threads(numCpuThreads)
  for (size_t i = 0; i < mols.size(); i++) {
    try {
      const RDKit::ROMol* mol = mols[i];
      fingerprints[i]         = internal::getFingerprintImpl(*mol, radius_, fpSize_);
    } catch (...) {
      exceptionRegistry.store(std::current_exception());
    }
  }
  exceptionRegistry.rethrow();
  return fingerprints;
}

}  // namespace nvMolKit
