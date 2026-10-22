// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "versions.h"

#if RDKIT_AIO_ETKDG_API

#include <DistGeom/ChiralViolationContribs.h>
#include <DistGeom/DistGeomUtils.h>
#include <DistGeom/DistViolationContribs.h>
#include <DistGeom/FourthDimContribs.h>
#include <ForceField/AngleConstraints.h>
#include <ForceField/DistanceConstraints.h>
#include <ForceField/ForceField.h>
#include <GraphMol/ForceFieldHelpers/CrystalFF/PlanarityContribs.h>
#include <GraphMol/ForceFieldHelpers/CrystalFF/TorsionAngleContribs.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include <functional>
#include <random>

#include "rdkit_extensions/dist_geom_flattened_builder.h"
#include "src/embedder_utils.h"
#include "src/forcefields/aio_etkdg_batched_forcefield.h"
#include "src/forcefields/dist_geom.h"
#include "src/forcefields/dist_geom_kernels.h"
#include "tests/test_utils.h"

namespace {

using nvMolKit::AsyncDeviceVector;
using nvMolKit::PrecisionMode;
namespace DG = nvMolKit::DistGeom;

//! Molecules with linear, planar and chiral centers in addition to the SDF test sets.
const std::vector<std::string> kCoverageSmiles =
  {"CC#N", "C#CC(=O)NC", "c1ccccc1C(=O)O", "C[C@H](O)F", "F/C=C/C#CC", "CC(C)(C)C(=O)N[C@@H](C)C(=O)O"};

//! One molecule prepared for the all-in-one force field: RDKit embedder state and perturbed 4D coordinates.
struct AllInOneMolecule {
  nvMolKit::detail::EmbedArgs                   eargs;
  std::vector<std::unique_ptr<RDGeom::PointND>> pointOwners;
  RDGeom::PointPtrVect                          points;
  std::vector<double>                           positions;
};

std::vector<std::unique_ptr<RDKit::ROMol>> coverageMolecules() {
  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  for (const auto& smiles : kCoverageSmiles) {
    std::unique_ptr<RDKit::ROMol> mol(RDKit::SmilesToMol(smiles));
    mols.emplace_back(RDKit::MolOps::addHs(*mol));
  }
  return mols;
}

std::vector<std::unique_ptr<RDKit::ROMol>> sdfMolecules(const std::string& fileName, const int count) {
  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  getMols(getTestDataFolderPath() + "/" + fileName, mols, count);
  return mols;
}

//! Prepares each molecule with the all-in-one force constants. Coordinates start from the input conformer when there
//! is one, are perturbed, and get a nonzero fourth dimension.
std::vector<AllInOneMolecule> prepareMolecules(const std::vector<std::unique_ptr<RDKit::ROMol>>& mols,
                                               const RDKit::DGeomHelpers::EmbedParameters&       params) {
  std::mt19937                           generator(1729);
  std::uniform_real_distribution<double> perturbation(-0.3, 0.3);
  std::uniform_real_distribution<double> randomCoordinate(-1.5, 1.5);

  std::vector<AllInOneMolecule> prepared(mols.size());
  for (std::size_t molIdx = 0; molIdx < mols.size(); ++molIdx) {
    auto& mol          = *mols[molIdx];
    auto& molecule     = prepared[molIdx];
    molecule.eargs.dim = 4;
    EXPECT_TRUE(nvMolKit::DGeomHelpers::prepareEmbedderArgs(mol, params, molecule.eargs));
    molecule.eargs.etkdgDetails.forceConsts = ForceFields::CrystalFF::ETKDGForceConsts::AIO::Cosine;
    molecule.eargs.etkdgDetails.distMat     = RDKit::MolOps::getDistanceMat(mol);

    for (unsigned int atomIdx = 0; atomIdx < mol.getNumAtoms(); ++atomIdx) {
      auto& point = *molecule.pointOwners.emplace_back(std::make_unique<RDGeom::PointND>(4));
      for (int dim = 0; dim < 3; ++dim) {
        point[dim] = mol.getNumConformers() > 0 ?
                       mol.getConformer().getAtomPos(atomIdx)[dim] + perturbation(generator) :
                       randomCoordinate(generator);
      }
      point[3] = perturbation(generator);
      molecule.points.push_back(&point);
      for (int dim = 0; dim < 4; ++dim) {
        molecule.positions.push_back(point[dim]);
      }
    }
  }
  return prepared;
}

DG::AllInOneForceContribsHost buildContribs(const AllInOneMolecule&                     molecule,
                                            const RDKit::DGeomHelpers::EmbedParameters& params) {
  return DG::constructAllInOneForceFieldContribs(molecule.eargs.dim,
                                                 *molecule.eargs.mmat,
                                                 molecule.eargs.chiralCenters,
                                                 molecule.eargs.etkdgDetails,
                                                 params.useExpTorsionAnglePrefs,
                                                 params.useBasicKnowledge);
}

//! RDKit's all-in-one force field for the given minimization phase, built as in RDKit's minimizeAllInOne.
std::unique_ptr<ForceFields::ForceField> referenceForcefield(AllInOneMolecule&                           molecule,
                                                             const RDKit::DGeomHelpers::EmbedParameters& params,
                                                             const bool torsionTermsEnabled) {
  std::unique_ptr<ForceFields::ForceField> field(
    ::DistGeom::constructAllInOneForceField(*molecule.eargs.mmat,
                                            molecule.points,
                                            molecule.eargs.etkdgDetails,
                                            &molecule.eargs.chiralCenters));
  if (torsionTermsEnabled) {
    ::DistGeom::addTorsionTerms(field.get(),
                                molecule.eargs.etkdgDetails,
                                params.useBasicKnowledge,
                                params.useExpTorsionAnglePrefs);
  }
  field->initialize();
  return field;
}

//! RDKit 2026.09's TorsionAngleContribs::getGrad mis-evaluates the six-fold term (fixed upstream in rdkit/rdkit#9699),
//! so the reference torsion gradient is a central difference of RDKit's torsion energy instead.
void replaceTorsionGradient(const ForceFields::ForceField& field,
                            std::vector<double>&           positions,
                            std::vector<double>&           gradient) {
  constexpr double step = 1e-6;
  for (const auto& contrib : field.contribs()) {
    const auto* torsions = dynamic_cast<const ForceFields::CrystalFF::TorsionAngleContribs*>(contrib.get());
    if (torsions == nullptr) {
      continue;
    }
    std::vector<double> rdkitGradient(gradient.size(), 0.0);
    torsions->getGrad(positions.data(), rdkitGradient.data());
    for (std::size_t coord = 0; coord < positions.size(); ++coord) {
      const double original = positions[coord];
      positions[coord]      = original + step;
      const double plus     = torsions->getEnergy(positions.data());
      positions[coord]      = original - step;
      const double minus    = torsions->getEnergy(positions.data());
      positions[coord]      = original;
      gradient[coord] += (plus - minus) / (2.0 * step) - rdkitGradient[coord];
    }
  }
}

struct Evaluation {
  std::vector<double>              energies;
  std::vector<std::vector<double>> gradients;
};

Evaluation evaluateReference(std::vector<AllInOneMolecule>&                       molecules,
                             const RDKit::DGeomHelpers::EmbedParameters&          params,
                             const bool                                           torsionTermsEnabled,
                             const std::vector<uint8_t>&                          active,
                             const std::function<void(ForceFields::ForceField&)>& filterContribs = {}) {
  Evaluation reference;
  for (std::size_t molIdx = 0; molIdx < molecules.size(); ++molIdx) {
    auto&               molecule = molecules[molIdx];
    std::vector<double> gradient(molecule.positions.size(), 0.0);
    double              energy = 0.0;
    if (active[molIdx]) {
      auto field = referenceForcefield(molecule, params, torsionTermsEnabled);
      if (filterContribs) {
        filterContribs(*field);
      }
      energy = field->calcEnergy(molecule.positions.data());
      field->calcGrad(molecule.positions.data(), gradient.data());
      replaceTorsionGradient(*field, molecule.positions, gradient);
    }
    reference.energies.push_back(energy);
    reference.gradients.push_back(std::move(gradient));
  }
  return reference;
}

struct Batch {
  DG::BatchedAllInOneSystemHost system;
  std::vector<int>              atomStarts = {0};
  std::vector<double>           positions;
};

Batch buildBatch(const std::vector<AllInOneMolecule>&              molecules,
                 const std::vector<DG::AllInOneForceContribsHost>& contribs) {
  Batch batch;
  for (std::size_t molIdx = 0; molIdx < molecules.size(); ++molIdx) {
    const auto& positions = molecules[molIdx].positions;
    const int   numAtoms  = static_cast<int>(positions.size() / 4);
    batch.atomStarts.push_back(batch.atomStarts.back() + numAtoms);
    batch.positions.insert(batch.positions.end(), positions.begin(), positions.end());
    DG::addMoleculeToAllInOneSystem(contribs[molIdx], numAtoms, batch.atomStarts, batch.system);
  }
  return batch;
}

Evaluation splitByMolecule(std::vector<double> energies, const std::vector<double>& gradient, const Batch& batch) {
  Evaluation result;
  result.energies = std::move(energies);
  for (std::size_t molIdx = 0; molIdx + 1 < batch.atomStarts.size(); ++molIdx) {
    result.gradients.emplace_back(gradient.begin() + 4 * batch.atomStarts[molIdx],
                                  gradient.begin() + 4 * batch.atomStarts[molIdx + 1]);
  }
  return result;
}

constexpr double kGradientFill = 7.0;

//! Calls the block-per-molecule kernels directly. Gradients start at kGradientFill so untouched systems are visible.
template <typename real>
Evaluation evaluateKernels(const Batch&                    batch,
                           const DG::AllInOneForceWeights& weights,
                           const bool                      torsionTermsEnabled,
                           const std::vector<uint8_t>&     active) {
  using Buffers = std::conditional_t<std::is_same_v<real, double>,
                                     DG::BatchedAllInOneDeviceBuffers,
                                     DG::BatchedAllInOneDeviceBuffersSingle>;
  Buffers buffers;
  DG::sendAllInOneSystemToDevice(batch.system, buffers, nullptr);
  const int                  numMols = static_cast<int>(batch.atomStarts.size()) - 1;
  AsyncDeviceVector<int>     atomStartsDevice;
  AsyncDeviceVector<real>    positionsDevice;
  AsyncDeviceVector<real>    energiesDevice(numMols);
  AsyncDeviceVector<real>    gradientDevice;
  AsyncDeviceVector<uint8_t> activeDevice;
  atomStartsDevice.setFromVector(batch.atomStarts);
  positionsDevice.setFromVector(std::vector<real>(batch.positions.begin(), batch.positions.end()));
  gradientDevice.setFromVector(std::vector<real>(batch.positions.size(), static_cast<real>(kGradientFill)));
  activeDevice.setFromVector(active);

  const auto terms = DG::toAllInOneContribsDevicePtr(buffers, atomStartsDevice.data());
  CHECK_CUDA_RETURN(DG::launchBlockPerMolEnergyKernelAIO(numMols,
                                                         terms,
                                                         positionsDevice.data(),
                                                         energiesDevice.data(),
                                                         weights,
                                                         torsionTermsEnabled,
                                                         activeDevice.data()));
  CHECK_CUDA_RETURN(DG::launchBlockPerMolGradKernelAIO(numMols,
                                                       terms,
                                                       positionsDevice.data(),
                                                       gradientDevice.data(),
                                                       weights,
                                                       torsionTermsEnabled,
                                                       activeDevice.data()));
  std::vector<real> energies(numMols);
  std::vector<real> gradient(batch.positions.size());
  energiesDevice.copyToHost(energies);
  gradientDevice.copyToHost(gradient);
  CHECK_CUDA_RETURN(cudaDeviceSynchronize());
  return splitByMolecule(std::vector<double>(energies.begin(), energies.end()),
                         std::vector<double>(gradient.begin(), gradient.end()),
                         batch);
}

//! Calls the batched forcefield through its double-precision interface.
Evaluation evaluateForcefield(nvMolKit::AllInOneETKDGBatchedForcefield& forcefield,
                              const Batch&                              batch,
                              const std::vector<uint8_t>&               active) {
  const int                  numMols = static_cast<int>(batch.atomStarts.size()) - 1;
  AsyncDeviceVector<double>  positionsDevice;
  AsyncDeviceVector<double>  energiesDevice(numMols);
  AsyncDeviceVector<double>  gradientDevice(batch.positions.size());
  AsyncDeviceVector<uint8_t> activeDevice;
  positionsDevice.setFromVector(batch.positions);
  activeDevice.setFromVector(active);
  energiesDevice.zero();
  gradientDevice.zero();
  CHECK_CUDA_RETURN(forcefield.computeEnergy(energiesDevice.data(), positionsDevice.data(), activeDevice.data()));
  CHECK_CUDA_RETURN(forcefield.computeGradients(gradientDevice.data(), positionsDevice.data(), activeDevice.data()));
  std::vector<double> energies(numMols);
  std::vector<double> gradient(batch.positions.size());
  energiesDevice.copyToHost(energies);
  gradientDevice.copyToHost(gradient);
  CHECK_CUDA_RETURN(cudaDeviceSynchronize());
  return splitByMolecule(std::move(energies), gradient, batch);
}

//! Tolerances relative to max(1, |expected|).
struct Tolerance {
  double energy;
  double gradient;
};

Tolerance toleranceFor(const PrecisionMode precision) {
  return precision == PrecisionMode::FULL ? Tolerance{1e-5, 1e-4} : Tolerance{2e-3, 2e-3};
}

bool withinTolerance(const double got, const double want, const double tolerance) {
  return std::abs(got - want) <= tolerance * std::max(1.0, std::abs(want));
}

void expectMatches(const Evaluation&           got,
                   const Evaluation&           want,
                   const std::vector<uint8_t>& active,
                   const Tolerance&            tolerance,
                   const double                inactiveGradient) {
  ASSERT_EQ(got.energies.size(), want.energies.size());
  int energyFailures   = 0;
  int gradientFailures = 0;
  for (std::size_t molIdx = 0; molIdx < want.energies.size(); ++molIdx) {
    if (!withinTolerance(got.energies[molIdx], want.energies[molIdx], tolerance.energy)) {
      ADD_FAILURE() << "molecule " << molIdx << " energy " << got.energies[molIdx] << " vs " << want.energies[molIdx];
      if (++energyFailures > 5) {
        return;
      }
    }
    for (std::size_t coord = 0; coord < want.gradients[molIdx].size(); ++coord) {
      const double expected = active[molIdx] ? want.gradients[molIdx][coord] : inactiveGradient;
      if (!withinTolerance(got.gradients[molIdx][coord], expected, tolerance.gradient)) {
        ADD_FAILURE() << "molecule " << molIdx << " coordinate " << coord << " gradient "
                      << got.gradients[molIdx][coord] << " vs " << expected;
        if (++gradientFailures > 5) {
          return;
        }
        break;
      }
    }
  }
}

std::vector<uint8_t> randomActiveMask(const std::size_t size) {
  std::mt19937                    generator(42);
  std::uniform_int_distribution<> coin(0, 1);
  std::vector<uint8_t>            active(size);
  for (auto& flag : active) {
    flag = static_cast<uint8_t>(coin(generator));
  }
  active.front() = 1;
  return active;
}

// ---------------------------------------
// Kernel tests: one term family at a time
// ---------------------------------------

enum class AllInOneTerm {
  DistViolation,
  ChiralViolation,
  FourthDim,
  HarmonicDistance,
  LinearAngle,
  ExperimentalTorsion,
  Planarity
};

std::string termName(const AllInOneTerm term) {
  switch (term) {
    case AllInOneTerm::DistViolation:
      return "DistViolation";
    case AllInOneTerm::ChiralViolation:
      return "ChiralViolation";
    case AllInOneTerm::FourthDim:
      return "FourthDim";
    case AllInOneTerm::HarmonicDistance:
      return "HarmonicDistance";
    case AllInOneTerm::LinearAngle:
      return "LinearAngle";
    case AllInOneTerm::ExperimentalTorsion:
      return "ExperimentalTorsion";
    case AllInOneTerm::Planarity:
      return "Planarity";
  }
  return "Unknown";
}

bool isTermContrib(const ForceFields::ForceFieldContrib* contrib, const AllInOneTerm term) {
  switch (term) {
    case AllInOneTerm::DistViolation:
      return dynamic_cast<const ::DistGeom::DistViolationContribs*>(contrib) != nullptr;
    case AllInOneTerm::ChiralViolation:
      return dynamic_cast<const ::DistGeom::ChiralViolationContribs*>(contrib) != nullptr;
    case AllInOneTerm::FourthDim:
      return dynamic_cast<const ::DistGeom::FourthDimContribs*>(contrib) != nullptr;
    case AllInOneTerm::HarmonicDistance:
      return dynamic_cast<const ForceFields::DistanceConstraintContribs*>(contrib) != nullptr;
    case AllInOneTerm::LinearAngle:
      return dynamic_cast<const ForceFields::AngleConstraintContribs*>(contrib) != nullptr;
    case AllInOneTerm::ExperimentalTorsion:
      return dynamic_cast<const ForceFields::CrystalFF::TorsionAngleContribs*>(contrib) != nullptr;
    case AllInOneTerm::Planarity:
      return dynamic_cast<const ForceFields::CrystalFF::PlanarityContribs*>(contrib) != nullptr;
  }
  return false;
}

//! Clears every term family except one, returning how many terms of that family remain.
std::size_t keepOnlyTerm(DG::AllInOneForceContribsHost& contribs, const AllInOneTerm term) {
  auto& dg = contribs.distanceGeometry;
  if (term != AllInOneTerm::DistViolation) {
    dg.distTerms = {};
  }
  if (term != AllInOneTerm::ChiralViolation) {
    dg.chiralTerms = {};
  }
  if (term != AllInOneTerm::FourthDim) {
    dg.fourthTerms = {};
  }
  if (term != AllInOneTerm::HarmonicDistance) {
    contribs.harmonicDistanceTerms = {};
  }
  if (term != AllInOneTerm::LinearAngle) {
    contribs.linearAngleTerms = {};
  }
  if (term != AllInOneTerm::ExperimentalTorsion) {
    contribs.experimentalTorsionTerms = {};
  }
  if (term != AllInOneTerm::Planarity) {
    contribs.planarityTerms = {};
  }
  return dg.distTerms.idx1.size() + dg.chiralTerms.idx1.size() + dg.fourthTerms.idx.size() +
         contribs.harmonicDistanceTerms.idx1.size() + contribs.linearAngleTerms.idx1.size() +
         contribs.experimentalTorsionTerms.idx1.size() + contribs.planarityTerms.idx1.size();
}

using KernelTestParams = std::tuple<AllInOneTerm, PrecisionMode>;

class AllInOneKernelTermTest : public ::testing::TestWithParam<KernelTestParams> {};

TEST_P(AllInOneKernelTermTest, MatchesRdkitInBothPhases) {
  const auto [term, precision] = GetParam();
  const auto params            = RDKit::DGeomHelpers::ETKDGv3;

  auto mols = sdfMolecules("MMFF94_dative.sdf", 50);
  for (auto& mol : sdfMolecules("MMFF94_hypervalent.sdf", 50)) {
    mols.push_back(std::move(mol));
  }
  for (auto& mol : coverageMolecules()) {
    mols.push_back(std::move(mol));
  }
  auto molecules = prepareMolecules(mols, params);

  std::vector<DG::AllInOneForceContribsHost> contribs;
  std::size_t                                numTerms = 0;
  for (const auto& molecule : molecules) {
    numTerms += keepOnlyTerm(contribs.emplace_back(buildContribs(molecule, params)), term);
  }
  ASSERT_GT(numTerms, 0u) << "test molecules have no " << termName(term) << " terms";

  const auto batch   = buildBatch(molecules, contribs);
  const auto weights = DG::allInOneForceWeights(molecules.front().eargs.etkdgDetails);
  const auto active  = randomActiveMask(molecules.size());
  for (const bool torsionTermsEnabled : {false, true}) {
    SCOPED_TRACE("torsionTermsEnabled=" + std::to_string(torsionTermsEnabled));
    const auto want = evaluateReference(molecules, params, torsionTermsEnabled, active, [term](auto& field) {
      std::erase_if(field.contribs(), [term](const auto& contrib) { return !isTermContrib(contrib.get(), term); });
    });
    const auto got  = precision == PrecisionMode::FULL ?
                        evaluateKernels<double>(batch, weights, torsionTermsEnabled, active) :
                        evaluateKernels<float>(batch, weights, torsionTermsEnabled, active);
    expectMatches(got, want, active, toleranceFor(precision), kGradientFill);
  }
}

INSTANTIATE_TEST_SUITE_P(AllTerms,
                         AllInOneKernelTermTest,
                         ::testing::Combine(::testing::Values(AllInOneTerm::DistViolation,
                                                              AllInOneTerm::ChiralViolation,
                                                              AllInOneTerm::FourthDim,
                                                              AllInOneTerm::HarmonicDistance,
                                                              AllInOneTerm::LinearAngle,
                                                              AllInOneTerm::ExperimentalTorsion,
                                                              AllInOneTerm::Planarity),
                                            ::testing::Values(PrecisionMode::FULL, PrecisionMode::SINGLE)),
                         [](const ::testing::TestParamInfo<KernelTestParams>& info) {
                           return termName(std::get<0>(info.param)) +
                                  (std::get<1>(info.param) == PrecisionMode::FULL ? "_Double" : "_Float");
                         });

TEST(AllInOneKernel, MoleculesTooLargeForSharedMemoryAccumulateInGlobalMemory) {
  const auto                                 params = RDKit::DGeomHelpers::ETKDGv3;
  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  for (const auto& smiles : {std::string(100, 'C') + "C#N", std::string("CC#N")}) {
    std::unique_ptr<RDKit::ROMol> mol(RDKit::SmilesToMol(smiles));
    mols.emplace_back(RDKit::MolOps::addHs(*mol));
  }
  ASSERT_GT(mols.front()->getNumAtoms(), 256u);
  auto                                       molecules = prepareMolecules(mols, params);
  std::vector<DG::AllInOneForceContribsHost> contribs;
  for (const auto& molecule : molecules) {
    contribs.push_back(buildContribs(molecule, params));
  }
  const auto                 batch   = buildBatch(molecules, contribs);
  const auto                 weights = DG::allInOneForceWeights(molecules.front().eargs.etkdgDetails);
  const std::vector<uint8_t> active(molecules.size(), 1);
  const auto                 want = evaluateReference(molecules, params, true, active);
  expectMatches(evaluateKernels<double>(batch, weights, true, active),
                want,
                active,
                toleranceFor(PrecisionMode::FULL),
                0.0);
  expectMatches(evaluateKernels<float>(batch, weights, true, active),
                want,
                active,
                toleranceFor(PrecisionMode::SINGLE),
                0.0);
}

// ---------------------------------------------
// Batched forcefield tests: all terms together
// ---------------------------------------------

using ForcefieldTestParams = std::tuple<ETKDGOption, std::string, PrecisionMode>;

class AllInOneForcefieldTest : public ::testing::TestWithParam<ForcefieldTestParams> {};

TEST_P(AllInOneForcefieldTest, BatchMatchesRdkitInBothPhases) {
  const auto [option, fileName, precision] = GetParam();
  const auto params                        = getETKDGOption(option);
  const auto mols                          = sdfMolecules(fileName, 100);
  auto       molecules                     = prepareMolecules(mols, params);

  std::vector<DG::AllInOneForceContribsHost> contribs;
  for (const auto& molecule : molecules) {
    contribs.push_back(buildContribs(molecule, params));
  }
  const auto                               batch  = buildBatch(molecules, contribs);
  const auto                               active = randomActiveMask(molecules.size());
  nvMolKit::AllInOneETKDGBatchedForcefield forcefield(batch.system,
                                                      batch.atomStarts,
                                                      DG::allInOneForceWeights(molecules.front().eargs.etkdgDetails),
                                                      {},
                                                      nullptr,
                                                      precision);
  EXPECT_EQ(forcefield.precision(), precision);
  EXPECT_EQ(forcefield.dataDim(), 4);
  for (const bool torsionTermsEnabled : {false, true}) {
    SCOPED_TRACE("torsionTermsEnabled=" + std::to_string(torsionTermsEnabled));
    forcefield.setTorsionTermsEnabled(torsionTermsEnabled);
    expectMatches(evaluateForcefield(forcefield, batch, active),
                  evaluateReference(molecules, params, torsionTermsEnabled, active),
                  active,
                  toleranceFor(precision),
                  0.0);
  }
}

INSTANTIATE_TEST_SUITE_P(
  ETKDGOptions,
  AllInOneForcefieldTest,
  ::testing::Combine(::testing::Values(ETKDGOption::ETKDGv3, ETKDGOption::ETKDGv2, ETKDGOption::ETDG, ETKDGOption::KDG),
                     ::testing::Values("MMFF94_dative.sdf", "MMFF94_hypervalent.sdf"),
                     ::testing::Values(PrecisionMode::FULL, PrecisionMode::SINGLE)),
  [](const ::testing::TestParamInfo<ForcefieldTestParams>& info) {
    const auto& fileName = std::get<1>(info.param);
    return getETKDGOptionName(std::get<0>(info.param)) + "_" + fileName.substr(0, fileName.find('.')) +
           (std::get<2>(info.param) == PrecisionMode::FULL ? "_Double" : "_Float");
  });

struct EdgeTestCase {
  std::vector<std::string> smiles;
  std::vector<uint8_t>     active;
};

class AllInOneForcefieldEdgeTest : public ::testing::TestWithParam<std::tuple<EdgeTestCase, PrecisionMode>> {};

TEST_P(AllInOneForcefieldEdgeTest, OneAndTwoAtomMolecules) {
  const auto& [testCase, precision]                 = GetParam();
  const auto                                 params = RDKit::DGeomHelpers::ETKDGv3;
  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  for (const auto& smiles : testCase.smiles) {
    mols.emplace_back(RDKit::SmilesToMol(smiles));
  }
  auto                                       molecules = prepareMolecules(mols, params);
  std::vector<DG::AllInOneForceContribsHost> contribs;
  for (const auto& molecule : molecules) {
    contribs.push_back(buildContribs(molecule, params));
  }
  const auto                               batch = buildBatch(molecules, contribs);
  nvMolKit::AllInOneETKDGBatchedForcefield forcefield(batch.system,
                                                      batch.atomStarts,
                                                      DG::allInOneForceWeights(molecules.front().eargs.etkdgDetails),
                                                      {},
                                                      nullptr,
                                                      precision);
  forcefield.setTorsionTermsEnabled(true);
  expectMatches(evaluateForcefield(forcefield, batch, testCase.active),
                evaluateReference(molecules, params, true, testCase.active),
                testCase.active,
                toleranceFor(precision),
                0.0);
}

INSTANTIATE_TEST_SUITE_P(OneTwoAtoms,
                         AllInOneForcefieldEdgeTest,
                         ::testing::Combine(::testing::Values(
                                              EdgeTestCase{
                                                {"C", "O"},
                                                {  1,   1}
},
                                              EdgeTestCase{{"C", "CC"}, {0, 1}},
                                              EdgeTestCase{{"CO", "N"}, {1, 0}},
                                              EdgeTestCase{{"CC", "CN"}, {1, 1}}),
                                            ::testing::Values(PrecisionMode::FULL, PrecisionMode::SINGLE)));

TEST(AllInOneForcefield, SinglePrecisionInterfaceEvaluatesNativelyAndRejectsFullPrecisionBuffers) {
  const auto params    = RDKit::DGeomHelpers::ETKDGv3;
  auto       mols      = coverageMolecules();
  auto       molecules = prepareMolecules(mols, params);

  std::vector<DG::AllInOneForceContribsHost> contribs;
  for (const auto& molecule : molecules) {
    contribs.push_back(buildContribs(molecule, params));
  }
  const auto batch   = buildBatch(molecules, contribs);
  const auto weights = DG::allInOneForceWeights(molecules.front().eargs.etkdgDetails);
  const int  numMols = static_cast<int>(molecules.size());

  AsyncDeviceVector<float> positionsDevice;
  AsyncDeviceVector<float> energiesDevice(numMols);
  AsyncDeviceVector<float> gradientDevice(batch.positions.size());
  positionsDevice.setFromVector(std::vector<float>(batch.positions.begin(), batch.positions.end()));

  nvMolKit::AllInOneETKDGBatchedForcefield full(batch.system, batch.atomStarts, weights);
  EXPECT_EQ(full.computeEnergy(energiesDevice.data(), positionsDevice.data()), cudaErrorInvalidValue);
  EXPECT_EQ(full.computeGradients(gradientDevice.data(), positionsDevice.data()), cudaErrorInvalidValue);

  nvMolKit::AllInOneETKDGBatchedForcefield single(batch.system,
                                                  batch.atomStarts,
                                                  weights,
                                                  {},
                                                  nullptr,
                                                  PrecisionMode::SINGLE);
  single.setTorsionTermsEnabled(true);
  gradientDevice.zero();
  CHECK_CUDA_RETURN(single.computeEnergy(energiesDevice.data(), positionsDevice.data()));
  CHECK_CUDA_RETURN(single.computeGradients(gradientDevice.data(), positionsDevice.data()));
  std::vector<float> energies(numMols);
  std::vector<float> gradient(batch.positions.size());
  energiesDevice.copyToHost(energies);
  gradientDevice.copyToHost(gradient);
  CHECK_CUDA_RETURN(cudaDeviceSynchronize());

  const std::vector<uint8_t> active(numMols, 1);
  expectMatches(splitByMolecule(std::vector<double>(energies.begin(), energies.end()),
                                std::vector<double>(gradient.begin(), gradient.end()),
                                batch),
                evaluateReference(molecules, params, true, active),
                active,
                toleranceFor(PrecisionMode::SINGLE),
                0.0);
}

}  // namespace

#endif  // RDKIT_AIO_ETKDG_API
