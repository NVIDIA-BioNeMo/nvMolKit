// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "versions.h"

#if RDKIT_ETKDG_2026_09_API

#include <DistGeom/DistGeomUtils.h>
#include <ForceField/ForceField.h>
#include <GraphMol/ForceFieldHelpers/CrystalFF/GaussianTorsionAngleContribs.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include <cmath>
#include <limits>
#include <numbers>
#include <random>
#include <set>

#include "rdkit_extensions/dist_geom_flattened_builder.h"
#include "src/embedder_utils.h"
#include "src/forcefields/dist_geom.h"
#include "src/forcefields/dist_geom_kernels.h"
#include "src/forcefields/etk_batched_forcefield.h"
#include "tests/test_utils.h"

namespace {

using nvMolKit::AsyncDeviceVector;
using nvMolKit::PrecisionMode;
namespace DG        = nvMolKit::DistGeom;
namespace CrystalFF = ForceFields::CrystalFF;

//! ETKDGv4 with the legacy refinement, which minimizes the experimental torsions in the ETK force field.
RDKit::DGeomHelpers::EmbedParameters legacyETKDGv4() {
  auto params                    = RDKit::DGeomHelpers::ETKDGv4;
  params.useLegacyImplementation = true;
  return params;
}

//! Amide (forced trans torsion), small ring, macrocycle, aromatic ester and chiral peptide torsions, plus a chain long
//! enough that its torsions span many warps.
const std::vector<std::string> kCoverageSmiles = {"CC(=O)NC",
                                                  "CCCC",
                                                  "C1CCCCC1",
                                                  "C1CCCCCCCCCCC1",
                                                  "c1ccccc1CC(=O)OC",
                                                  "CC(C)(C)C(=O)N[C@@H](C)C(=O)O",
                                                  std::string(100, 'C')};

std::vector<std::unique_ptr<RDKit::ROMol>> testMolecules(const int sdfCount) {
  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  for (const std::string fileName : {"MMFF94_dative.sdf", "MMFF94_hypervalent.sdf"}) {
    std::vector<std::unique_ptr<RDKit::ROMol>> fileMols;
    getMols(getTestDataFolderPath() + "/" + fileName, fileMols, sdfCount);
    for (auto& mol : fileMols) {
      mols.push_back(std::move(mol));
    }
  }
  for (const auto& smiles : kCoverageSmiles) {
    std::unique_ptr<RDKit::ROMol> mol(RDKit::SmilesToMol(smiles));
    mols.emplace_back(RDKit::MolOps::addHs(*mol));
  }
  return mols;
}

//! One molecule's embedder state and perturbed coordinates: 3D points for RDKit and the same coordinates four values
//! per atom for nvMolKit, whose ETK kernels ignore the fourth.
struct ETKMolecule {
  nvMolKit::detail::EmbedArgs  eargs;
  std::vector<RDGeom::Point3D> points;
  std::vector<double>          positions;
};

//! Prepares each molecule's embedder state with perturbed or random coordinates.
std::vector<ETKMolecule> prepareMolecules(const std::vector<std::unique_ptr<RDKit::ROMol>>& mols,
                                          const RDKit::DGeomHelpers::EmbedParameters&       params) {
  std::mt19937                           generator(1729);
  std::uniform_real_distribution<double> perturbation(-0.3, 0.3);
  std::uniform_real_distribution<double> randomCoordinate(-2.5, 2.5);

  std::vector<ETKMolecule> prepared(mols.size());
  for (std::size_t molIdx = 0; molIdx < mols.size(); ++molIdx) {
    auto& mol      = *mols[molIdx];
    auto& molecule = prepared[molIdx];
    EXPECT_TRUE(nvMolKit::DGeomHelpers::prepareEmbedderArgs(mol, params, molecule.eargs));
    for (unsigned int atomIdx = 0; atomIdx < mol.getNumAtoms(); ++atomIdx) {
      RDGeom::Point3D point;
      for (int dim = 0; dim < 3; ++dim) {
        point[dim] = mol.getNumConformers() > 0 ?
                       mol.getConformer().getAtomPos(atomIdx)[dim] + perturbation(generator) :
                       randomCoordinate(generator);
      }
      molecule.points.push_back(point);
      molecule.positions.insert(molecule.positions.end(), {point.x, point.y, point.z, 100.0});
    }
  }
  return prepared;
}

std::vector<double> positions3D(const ETKMolecule& molecule) {
  std::vector<double> result;
  for (const auto& point : molecule.points) {
    result.insert(result.end(), {point.x, point.y, point.z});
  }
  return result;
}

DG::Energy3DForceContribsHost buildContribs(const ETKMolecule&                          molecule,
                                            const RDKit::DGeomHelpers::EmbedParameters& params) {
  return DG::construct3DForceFieldContribs(*molecule.eargs.mmat,
                                           molecule.eargs.etkdgDetails,
                                           positions3D(molecule),
                                           3,
                                           params.useBasicKnowledge);
}

struct Batch {
  DG::BatchedMolecularSystem3DHost system;
  std::vector<int>                 atomStarts = {0};
  std::vector<double>              positions;
};

Batch buildBatch(const std::vector<ETKMolecule>&                   molecules,
                 const std::vector<DG::Energy3DForceContribsHost>& contribs) {
  Batch batch;
  for (std::size_t molIdx = 0; molIdx < molecules.size(); ++molIdx) {
    batch.atomStarts.push_back(batch.atomStarts.back() + static_cast<int>(molecules[molIdx].points.size()));
    batch.positions.insert(batch.positions.end(),
                           molecules[molIdx].positions.begin(),
                           molecules[molIdx].positions.end());
    DG::addMoleculeToMolecularSystem3D(contribs[molIdx], batch.atomStarts, batch.system);
  }
  return batch;
}

struct Evaluation {
  std::vector<double>              energies;
  std::vector<std::vector<double>> gradients;  // four values per atom
};

//! RDKit's ETK force field for the molecule (DistGeom::construct3DForceField), optionally only its Gaussian torsions.
Evaluation evaluateReference(std::vector<ETKMolecule>&   molecules,
                             const std::vector<uint8_t>& active,
                             const bool                  gaussianTorsionsOnly) {
  Evaluation reference;
  for (std::size_t molIdx = 0; molIdx < molecules.size(); ++molIdx) {
    auto&               molecule = molecules[molIdx];
    std::vector<double> gradient(molecule.positions.size(), 0.0);
    double              energy = 0.0;
    if (active[molIdx]) {
      RDGeom::Point3DPtrVect pointers;
      for (auto& point : molecule.points) {
        pointers.push_back(&point);
      }
      std::unique_ptr<ForceFields::ForceField> field(
        ::DistGeom::construct3DForceField(*molecule.eargs.mmat, pointers, molecule.eargs.etkdgDetails));
      if (gaussianTorsionsOnly) {
        std::erase_if(field->contribs(), [](const auto& contrib) {
          return dynamic_cast<const CrystalFF::GaussianTorsionAngleContribs*>(contrib.get()) == nullptr;
        });
      }
      field->initialize();
      auto                coordinates = positions3D(molecule);
      std::vector<double> gradient3D(coordinates.size(), 0.0);
      energy = field->calcEnergy(coordinates.data());
      field->calcGrad(coordinates.data(), gradient3D.data());
      for (std::size_t atom = 0; atom < molecule.points.size(); ++atom) {
        for (int dim = 0; dim < 3; ++dim) {
          gradient[4 * atom + dim] = gradient3D[3 * atom + dim];
        }
      }
    }
    reference.energies.push_back(energy);
    reference.gradients.push_back(std::move(gradient));
  }
  return reference;
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

//! Tolerances relative to max(1, |expected|).
struct Tolerance {
  double energy;
  double gradient;
};

//! Gaussian torsions alone. Double precision matches RDKit to rounding. Single precision is limited by rounding the
//! coordinates to float: against RDKit at the rounded coordinates it agrees to 1e-6 (energy) and 3e-5 (gradient).
Tolerance toleranceFor(const PrecisionMode precision) {
  return precision == PrecisionMode::FULL ? Tolerance{1e-10, 1e-7} : Tolerance{5e-5, 2e-3};
}

//! Every ETK term. The other ETK terms evaluate parts of their arithmetic in float even in full precision.
Tolerance etkToleranceFor(const PrecisionMode precision) {
  return precision == PrecisionMode::FULL ? Tolerance{5e-6, 2e-4} : Tolerance{1e-4, 3e-3};
}

void expectMatches(const Evaluation& got, const Evaluation& want, const Tolerance& tolerance) {
  ASSERT_EQ(got.energies.size(), want.energies.size());
  int failures = 0;
  for (std::size_t molIdx = 0; molIdx < want.energies.size() && failures <= 5; ++molIdx) {
    const double energyScale = std::max(1.0, std::abs(want.energies[molIdx]));
    if (!(std::abs(got.energies[molIdx] - want.energies[molIdx]) <= tolerance.energy * energyScale)) {
      ADD_FAILURE() << "molecule " << molIdx << " energy " << got.energies[molIdx] << " vs " << want.energies[molIdx];
      ++failures;
    }
    ASSERT_EQ(got.gradients[molIdx].size(), want.gradients[molIdx].size());
    for (std::size_t coord = 0; coord < want.gradients[molIdx].size(); ++coord) {
      const double expected = want.gradients[molIdx][coord];
      if (!(std::abs(got.gradients[molIdx][coord] - expected) <=
            tolerance.gradient * std::max(1.0, std::abs(expected)))) {
        ADD_FAILURE() << "molecule " << molIdx << " coordinate " << coord << " gradient "
                      << got.gradients[molIdx][coord] << " vs " << expected;
        ++failures;
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
  active.back()  = 1;
  return active;
}

//! Evaluates through ETKBatchedForcefield's double-precision interface, as the batched BFGS minimizer does.
Evaluation evaluateForcefield(const Batch& batch, const std::vector<uint8_t>& active, const PrecisionMode precision) {
  nvMolKit::ETKBatchedForcefield forcefield(batch.system, batch.atomStarts, true, {}, nullptr, precision);
  EXPECT_EQ(forcefield.precision(), precision);
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

//! Calls the block-per-molecule ETK kernels, whose per-molecule energy and gradient functions the per-molecule BFGS
//! kernel also uses. Gradients start at kGradientFill so untouched inactive systems are visible.
constexpr double kGradientFill = 7.0;

template <typename real> Evaluation evaluateBlockPerMolKernels(const Batch& batch, const std::vector<uint8_t>& active) {
  using Buffers = std::conditional_t<std::is_same_v<real, double>,
                                     DG::BatchedMolecular3DDeviceBuffers,
                                     DG::BatchedMolecular3DDeviceBuffersSingle>;
  Buffers buffers;
  DG::sendContribsAndIndicesToDevice3D(batch.system, buffers);
  const int                  numMols = static_cast<int>(batch.atomStarts.size()) - 1;
  AsyncDeviceVector<int>     atomStartsDevice;
  AsyncDeviceVector<real>    positionsDevice;
  AsyncDeviceVector<real>    energiesDevice(numMols);
  AsyncDeviceVector<real>    gradientDevice;
  AsyncDeviceVector<uint8_t> activeDevice;
  atomStartsDevice.setFromVector(batch.atomStarts);
  positionsDevice.setFromVector(std::vector<real>(batch.positions.begin(), batch.positions.end()));
  activeDevice.setFromVector(active);
  // The kernels accumulate into the gradient, so active systems start from zero.
  std::vector<real> initialGradient(batch.positions.size(), static_cast<real>(kGradientFill));
  for (std::size_t molIdx = 0; molIdx < active.size(); ++molIdx) {
    if (active[molIdx]) {
      std::fill(initialGradient.begin() + 4 * batch.atomStarts[molIdx],
                initialGradient.begin() + 4 * batch.atomStarts[molIdx + 1],
                real(0));
    }
  }
  gradientDevice.setFromVector(initialGradient);

  const auto terms   = DG::toEnergy3DForceContribsDevicePtr(buffers);
  const auto indices = DG::toBatchedIndices3DDevicePtr(buffers, atomStartsDevice.data());
  CHECK_CUDA_RETURN(DG::launchBlockPerMolEnergyKernelETK(numMols,
                                                         terms,
                                                         indices,
                                                         positionsDevice.data(),
                                                         energiesDevice.data(),
                                                         activeDevice.data()));
  CHECK_CUDA_RETURN(DG::launchBlockPerMolGradKernelETK(numMols,
                                                       terms,
                                                       indices,
                                                       positionsDevice.data(),
                                                       gradientDevice.data(),
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

//! Inactive systems: zero energy, gradient left at fill.
Evaluation withInactiveExpectations(Evaluation want, const std::vector<uint8_t>& active, const double gradientFill) {
  for (std::size_t molIdx = 0; molIdx < active.size(); ++molIdx) {
    if (!active[molIdx]) {
      want.energies[molIdx] = 0.0;
      std::fill(want.gradients[molIdx].begin(), want.gradients[molIdx].end(), gradientFill);
    }
  }
  return want;
}

//! ETKDGv4 SDF and coverage molecules batched with every ETK term, and with only their Gaussian torsions.
class GaussianTorsionETKFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    params_    = legacyETKDGv4();
    mols_      = testMolecules(50);
    molecules_ = prepareMolecules(mols_, params_);
    std::vector<DG::Energy3DForceContribsHost> contribs;
    for (const auto& molecule : molecules_) {
      contribs.push_back(buildContribs(molecule, params_));
    }
    batch_ = buildBatch(molecules_, contribs);
    for (auto& molecule : contribs) {
      DG::Energy3DForceContribsHost gaussianOnly;
      gaussianOnly.gaussianTorsionTerms = std::move(molecule.gaussianTorsionTerms);
      molecule                          = std::move(gaussianOnly);
    }
    gaussianBatch_ = buildBatch(molecules_, contribs);
    active_        = randomActiveMask(molecules_.size());
    ASSERT_TRUE(batch_.system.contribs.experimentalTorsionTerms.idx1.empty());
    ASSERT_GT(batch_.system.contribs.gaussianTorsionTerms.idx1.size(), molecules_.size());
  }

  RDKit::DGeomHelpers::EmbedParameters       params_;
  std::vector<std::unique_ptr<RDKit::ROMol>> mols_;
  std::vector<ETKMolecule>                   molecules_;
  //! Every ETK term.
  Batch                                      batch_;
  //! Only the Gaussian torsions, to check them at their own precision.
  Batch                                      gaussianBatch_;
  std::vector<uint8_t>                       active_;
};

class GaussianTorsionETKTest : public GaussianTorsionETKFixture, public ::testing::WithParamInterface<PrecisionMode> {};

TEST_P(GaussianTorsionETKTest, GaussianTermsThroughForcefieldMatchRdkit) {
  const auto want = withInactiveExpectations(evaluateReference(molecules_, active_, true), active_, 0.0);
  expectMatches(evaluateForcefield(gaussianBatch_, active_, GetParam()), want, toleranceFor(GetParam()));
}

TEST_P(GaussianTorsionETKTest, GaussianTermsThroughBlockPerMoleculeKernelsMatchRdkit) {
  const auto want = withInactiveExpectations(evaluateReference(molecules_, active_, true), active_, kGradientFill);
  const auto got  = GetParam() == PrecisionMode::FULL ? evaluateBlockPerMolKernels<double>(gaussianBatch_, active_) :
                                                        evaluateBlockPerMolKernels<float>(gaussianBatch_, active_);
  expectMatches(got, want, toleranceFor(GetParam()));
}

TEST_P(GaussianTorsionETKTest, ETKForcefieldMatchesRdkit) {
  const auto want = withInactiveExpectations(evaluateReference(molecules_, active_, false), active_, 0.0);
  expectMatches(evaluateForcefield(batch_, active_, GetParam()), want, etkToleranceFor(GetParam()));
}

TEST_P(GaussianTorsionETKTest, ETKBlockPerMoleculeKernelsMatchRdkit) {
  const auto want = withInactiveExpectations(evaluateReference(molecules_, active_, false), active_, kGradientFill);
  const auto got  = GetParam() == PrecisionMode::FULL ? evaluateBlockPerMolKernels<double>(batch_, active_) :
                                                        evaluateBlockPerMolKernels<float>(batch_, active_);
  expectMatches(got, want, etkToleranceFor(GetParam()));
}

//! Calls ETKBatchedForcefield's float interface, as the single-precision batched BFGS minimizer does.
Evaluation evaluateFloatInterface(const Batch& batch, const std::vector<uint8_t>& active) {
  nvMolKit::ETKBatchedForcefield forcefield(batch.system, batch.atomStarts, true, {}, nullptr, PrecisionMode::SINGLE);
  const int                      numMols = static_cast<int>(batch.atomStarts.size()) - 1;
  AsyncDeviceVector<float>       positionsDevice;
  AsyncDeviceVector<float>       energiesDevice(numMols);
  AsyncDeviceVector<float>       gradientDevice(batch.positions.size());
  AsyncDeviceVector<uint8_t>     activeDevice;
  positionsDevice.setFromVector(std::vector<float>(batch.positions.begin(), batch.positions.end()));
  activeDevice.setFromVector(active);
  energiesDevice.zero();
  gradientDevice.zero();
  CHECK_CUDA_RETURN(forcefield.computeEnergy(energiesDevice.data(), positionsDevice.data(), activeDevice.data()));
  CHECK_CUDA_RETURN(forcefield.computeGradients(gradientDevice.data(), positionsDevice.data(), activeDevice.data()));
  std::vector<float> energies(numMols);
  std::vector<float> gradient(batch.positions.size());
  energiesDevice.copyToHost(energies);
  gradientDevice.copyToHost(gradient);
  CHECK_CUDA_RETURN(cudaDeviceSynchronize());
  return splitByMolecule(std::vector<double>(energies.begin(), energies.end()),
                         std::vector<double>(gradient.begin(), gradient.end()),
                         batch);
}

TEST_F(GaussianTorsionETKFixture, SinglePrecisionInterfaceMatchesRdkit) {
  expectMatches(evaluateFloatInterface(gaussianBatch_, active_),
                withInactiveExpectations(evaluateReference(molecules_, active_, true), active_, 0.0),
                toleranceFor(PrecisionMode::SINGLE));
  expectMatches(evaluateFloatInterface(batch_, active_),
                withInactiveExpectations(evaluateReference(molecules_, active_, false), active_, 0.0),
                etkToleranceFor(PrecisionMode::SINGLE));
}

//! Full precision ETK gradients and per-term energies use the term-parallel kernels.
TEST_F(GaussianTorsionETKFixture, TermParallelKernelsMatchRdkitGaussianTerms) {
  DG::BatchedMolecular3DDeviceBuffers buffers;
  DG::sendContribsAndIndicesToDevice3D(batch_.system, buffers);
  DG::allocateIntermediateBuffers3D(batch_.system, buffers);
  const int                  numMols = static_cast<int>(molecules_.size());
  AsyncDeviceVector<int>     atomStartsDevice;
  AsyncDeviceVector<double>  positionsDevice;
  AsyncDeviceVector<double>  energiesDevice(numMols);
  AsyncDeviceVector<double>  gradientDevice(batch_.positions.size());
  AsyncDeviceVector<uint8_t> activeDevice;
  atomStartsDevice.setFromVector(batch_.atomStarts);
  positionsDevice.setFromVector(batch_.positions);
  activeDevice.setFromVector(active_);
  energiesDevice.zero();
  gradientDevice.zero();
  CHECK_CUDA_RETURN(DG::computeEnergyETK(buffers,
                                         energiesDevice.data(),
                                         atomStartsDevice.data(),
                                         positionsDevice.data(),
                                         activeDevice.data(),
                                         nullptr,
                                         DG::ETKTerm::EXPERIMANTAL_TORSION));
  CHECK_CUDA_RETURN(DG::computeGradientsETK(buffers,
                                            gradientDevice.data(),
                                            atomStartsDevice.data(),
                                            positionsDevice.data(),
                                            activeDevice.data(),
                                            DG::ETKTerm::EXPERIMANTAL_TORSION));
  std::vector<double> energies(numMols);
  std::vector<double> gradient(batch_.positions.size());
  energiesDevice.copyToHost(energies);
  gradientDevice.copyToHost(gradient);
  CHECK_CUDA_RETURN(cudaDeviceSynchronize());
  expectMatches(splitByMolecule(std::move(energies), gradient, batch_),
                withInactiveExpectations(evaluateReference(molecules_, active_, true), active_, 0.0),
                toleranceFor(PrecisionMode::FULL));
}

INSTANTIATE_TEST_SUITE_P(Precisions,
                         GaussianTorsionETKTest,
                         ::testing::Values(PrecisionMode::FULL, PrecisionMode::SINGLE),
                         [](const ::testing::TestParamInfo<PrecisionMode>& info) {
                           return info.param == PrecisionMode::FULL ? "Double" : "Float";
                         });

TEST(GaussianTorsionBuilder, PacksOneTermPerCentralBondWithTheMoleculesTables) {
  const auto params    = legacyETKDGv4();
  auto       mols      = testMolecules(50);
  auto       molecules = prepareMolecules(mols, params);
  for (std::size_t molIdx = 0; molIdx < molecules.size(); ++molIdx) {
    SCOPED_TRACE(molIdx);
    const auto& details  = molecules[molIdx].eargs.etkdgDetails;
    const auto  contribs = buildContribs(molecules[molIdx], params);
    const auto& terms    = contribs.gaussianTorsionTerms;
    EXPECT_TRUE(contribs.experimentalTorsionTerms.idx1.empty());
    ASSERT_EQ(terms.idx2.size(), terms.idx1.size());
    ASSERT_EQ(terms.tableStarts.size(), terms.idx1.size());
    ASSERT_EQ(terms.tableSizes.size(), terms.idx1.size());
    ASSERT_EQ(terms.scaling.size(), terms.idx1.size());
    ASSERT_EQ(terms.derivatives.size(), terms.energies.size());

    // Each table is copied once and every term reads a table RDKit holds for the molecule.
    std::set<std::pair<int, int>> centralBonds;
    std::set<int>                 tableStarts;
    for (std::size_t term = 0; term < terms.idx1.size(); ++term) {
      EXPECT_TRUE(centralBonds.insert(std::minmax(terms.idx2[term], terms.idx3[term])).second)
        << "two terms about bond " << terms.idx2[term] << "-" << terms.idx3[term];
      ASSERT_GE(terms.tableSizes[term], 2);
      ASSERT_LE(static_cast<std::size_t>(terms.tableStarts[term] + terms.tableSizes[term]), terms.energies.size());
      tableStarts.insert(terms.tableStarts[term]);
      const std::vector<double> energies(terms.energies.begin() + terms.tableStarts[term],
                                         terms.energies.begin() + terms.tableStarts[term] + terms.tableSizes[term]);
      EXPECT_NE(std::find(details.phiToEnergy.begin(), details.phiToEnergy.end(), energies), details.phiToEnergy.end());
    }
    std::size_t tableEntries = 0;
    for (const int start : tableStarts) {
      const auto term =
        std::distance(terms.tableStarts.begin(), std::find(terms.tableStarts.begin(), terms.tableStarts.end(), start));
      tableEntries += terms.tableSizes[term];
    }
    EXPECT_EQ(tableEntries, terms.energies.size());
  }

  // A batch stores each distinct table once: repeated conformers of one molecule add no table entries, and molecules
  // sharing torsion patterns share tables.
  std::vector<DG::Energy3DForceContribsHost> contribs;
  std::size_t                                unbatchedEntries = 0;
  for (const auto& molecule : molecules) {
    contribs.push_back(buildContribs(molecule, params));
    unbatchedEntries += contribs.back().gaussianTorsionTerms.energies.size();
  }
  const auto batch = buildBatch(molecules, contribs);
  EXPECT_LT(batch.system.contribs.gaussianTorsionTerms.energies.size(), unbatchedEntries);
  const std::vector<DG::Energy3DForceContribsHost> repeated(3, contribs.front());
  DG::BatchedMolecularSystem3DHost                 repeatedSystem;
  const int                                        numAtoms = static_cast<int>(molecules.front().points.size());
  const std::vector<int>                           atomStarts{0, numAtoms, 2 * numAtoms, 3 * numAtoms};
  for (const auto& molecule : repeated) {
    DG::addMoleculeToMolecularSystem3D(molecule, atomStarts, repeatedSystem);
  }
  EXPECT_EQ(repeatedSystem.contribs.gaussianTorsionTerms.energies.size(),
            contribs.front().gaussianTorsionTerms.energies.size());
  EXPECT_EQ(repeatedSystem.contribs.gaussianTorsionTerms.idx1.size(),
            3 * contribs.front().gaussianTorsionTerms.idx1.size());

  // Cosine ETversions keep producing only cosine terms.
  const auto cosineParams    = RDKit::DGeomHelpers::ETKDGv3;
  auto       cosineMolecules = prepareMolecules(mols, cosineParams);
  for (const auto& molecule : cosineMolecules) {
    EXPECT_TRUE(buildContribs(molecule, cosineParams).gaussianTorsionTerms.idx1.empty());
  }
}

//! Butane-like four-atom system with one Gaussian torsion that uses the given RDKit lookup tables.
struct SingleTorsion {
  DG::BatchedMolecularSystem3DHost system;
  std::vector<int>                 atomStarts = {0, 4};
  std::vector<double>              energyTable;
  std::vector<double>              gradientTable;
  double                           scaling = 0.0;
};

SingleTorsion singleTorsion(const std::vector<double>& energies, const std::vector<double>& gradients, double scaling) {
  SingleTorsion torsion;
  torsion.energyTable   = energies;
  torsion.gradientTable = gradients;
  torsion.scaling       = scaling;
  DG::Energy3DForceContribsHost contribs;
  auto&                         terms = contribs.gaussianTorsionTerms;
  terms.idx1                          = {0};
  terms.idx2                          = {1};
  terms.idx3                          = {2};
  terms.idx4                          = {3};
  terms.tableStarts                   = {0};
  terms.tableSizes                    = {static_cast<int>(energies.size())};
  terms.scaling                       = {scaling};
  terms.energies                      = energies;
  terms.derivatives                   = gradients;
  DG::addMoleculeToMolecularSystem3D(contribs, torsion.atomStarts, torsion.system);
  return torsion;
}

//! Positions with the dihedral 1-2-3-4 at the given angle; collinear puts atom 1 on the 2-3 axis.
std::vector<double> torsionPositions(const double angle, const bool collinear = false) {
  return {collinear ? 0.0 : 1.0,
          collinear ? -1.0 : 0.3,
          0.0,
          100.0,
          0.0,
          0.0,
          0.0,
          100.0,
          0.0,
          1.5,
          0.0,
          100.0,
          std::cos(angle),
          1.8,
          std::sin(angle),
          100.0};
}

class GaussianTorsionGeometryTest : public ::testing::TestWithParam<PrecisionMode> {};

TEST_P(GaussianTorsionGeometryTest, MatchesRdkitAcrossTheTableIncludingEndpointsAndDegenerateTorsions) {
  // A real ETKDGv4 table: the first Gaussian torsion of butane.
  std::unique_ptr<RDKit::ROMol> butane(
    RDKit::MolOps::addHs(*std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol("CCCC"))));
  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  mols.push_back(std::move(butane));
  const auto  params    = legacyETKDGv4();
  auto        molecules = prepareMolecules(mols, params);
  const auto  contribs  = buildContribs(molecules.front(), params);
  const auto& terms     = contribs.gaussianTorsionTerms;
  ASSERT_FALSE(terms.idx1.empty());
  const std::vector<double> energies(terms.energies.begin() + terms.tableStarts[0],
                                     terms.energies.begin() + terms.tableStarts[0] + terms.tableSizes[0]);
  const std::vector<double> gradients(terms.derivatives.begin() + terms.tableStarts[0],
                                      terms.derivatives.begin() + terms.tableStarts[0] + terms.tableSizes[0]);
  const auto                torsion = singleTorsion(energies, gradients, terms.scaling[0]);

  std::vector<RDGeom::Point3D> points(4);
  ForceFields::ForceField      reference(3);
  for (auto& point : points) {
    reference.positions().push_back(&point);
  }
  auto contribution = std::make_unique<CrystalFF::GaussianTorsionAngleContribs>(&reference);
  contribution->addContrib(0, 1, 2, 3, energies, gradients, torsion.scaling);
  reference.contribs().push_back(std::move(contribution));
  reference.initialize();

  const double pi = std::numbers::pi;
  const std::vector<double>
    angles{0.0, 1e-7, 1e-3, 0.01, 0.4, pi / 2, 2.0, pi - 0.01, pi - 1e-3, pi - 1e-7, pi, -pi / 3, -2.9};
  for (std::size_t geometry = 0; geometry <= angles.size(); ++geometry) {
    const bool   collinear = geometry == angles.size();
    const double angle     = collinear ? 0.7 : angles[geometry];
    SCOPED_TRACE(collinear ? std::string("collinear") : "angle " + std::to_string(angle));
    Batch               batch{.system     = torsion.system,
                              .atomStarts = torsion.atomStarts,
                              .positions  = torsionPositions(angle, collinear)};
    std::vector<double> coordinates;
    for (int atom = 0; atom < 4; ++atom) {
      coordinates.insert(coordinates.end(), batch.positions.begin() + 4 * atom, batch.positions.begin() + 4 * atom + 3);
    }
    std::vector<double> gradient3D(12, 0.0);
    reference.calcGrad(coordinates.data(), gradient3D.data());
    Evaluation want{.energies  = {reference.calcEnergy(coordinates.data())},
                    .gradients = {std::vector<double>(16, 0.0)}};
    for (int atom = 0; atom < 4; ++atom) {
      for (int dim = 0; dim < 3; ++dim) {
        want.gradients[0][4 * atom + dim] = gradient3D[3 * atom + dim];
      }
    }
    const std::vector<uint8_t> active{1};
    auto                       viaForcefield = evaluateForcefield(batch, active, GetParam());
    auto viaBlocks = GetParam() == PrecisionMode::FULL ? evaluateBlockPerMolKernels<double>(batch, active) :
                                                         evaluateBlockPerMolKernels<float>(batch, active);
    // Within 1e-5 of 0 or pi, RDKit's acos(cos(phi)) is off by up to sqrt(machine epsilon) where nvMolKit's atan2 is
    // exact, and RDKit divides dE/dphi by its 1e-8 floor on sin(phi), so its gradient is rounding noise there. Those
    // geometries check energies against RDKit and the single-precision gradient against nvMolKit's double precision.
    auto tolerance = toleranceFor(GetParam());
    if (!collinear && std::abs(std::sin(angle)) < 1e-5) {
      tolerance.energy = std::max(tolerance.energy, 1e-6);
      want.gradients   = evaluateForcefield(batch, active, PrecisionMode::FULL).gradients;
    }
    expectMatches(viaForcefield, want, tolerance);
    expectMatches(viaBlocks, want, tolerance);
  }
}

//! RDKit treats a torsion as degenerate only when a plane normal is shorter than 1e-10. A nearly collinear first atom,
//! whose normal is about 1e-7 long, still has a well-defined dihedral that both evaluate.
TEST(GaussianTorsionDegeneracy, NearlyCollinearTorsionIsEvaluatedLikeRdkit) {
  const std::vector<double> energies{0.0, 1.0, 3.0};
  const auto                torsion = singleTorsion(energies, {0.0, 1.0, 2.0}, 1.0);
  Batch batch{.system = torsion.system, .atomStarts = torsion.atomStarts, .positions = torsionPositions(0.0, true)};
  batch.positions[0] = 1e-7;  // atom 1 slightly off the 2-3 axis

  std::vector<RDGeom::Point3D> points(4);
  ForceFields::ForceField      reference(3);
  for (auto& point : points) {
    reference.positions().push_back(&point);
  }
  auto contribution = std::make_unique<CrystalFF::GaussianTorsionAngleContribs>(&reference);
  contribution->addContrib(0, 1, 2, 3, energies, {0.0, 1.0, 2.0}, 1.0);
  reference.contribs().push_back(std::move(contribution));
  reference.initialize();
  std::vector<double> coordinates;
  for (int atom = 0; atom < 4; ++atom) {
    coordinates.insert(coordinates.end(), batch.positions.begin() + 4 * atom, batch.positions.begin() + 4 * atom + 3);
  }
  const double want = reference.calcEnergy(coordinates.data());
  const auto   got  = evaluateForcefield(batch, {1}, PrecisionMode::FULL);
  // The degenerate fallback would give the table's value at pi/2, 1.0.
  EXPECT_GT(std::abs(want - 1.0), 0.1);
  EXPECT_NEAR(got.energies[0], want, 1e-6);
}

TEST_P(GaussianTorsionGeometryTest, NonFiniteCoordinatesGiveNonFiniteEnergies) {
  const auto torsion = singleTorsion({1.0, 0.5, 2.0}, {0.0, 1.0, -1.0}, 1.0);
  for (const double invalid : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    Batch batch{.system = torsion.system, .atomStarts = torsion.atomStarts, .positions = torsionPositions(1.0)};
    batch.positions[1] = invalid;
    const auto got     = evaluateForcefield(batch, {1}, GetParam());
    EXPECT_FALSE(std::isfinite(got.energies[0]));
  }
}

INSTANTIATE_TEST_SUITE_P(Precisions,
                         GaussianTorsionGeometryTest,
                         ::testing::Values(PrecisionMode::FULL, PrecisionMode::SINGLE),
                         [](const ::testing::TestParamInfo<PrecisionMode>& info) {
                           return info.param == PrecisionMode::FULL ? "Double" : "Float";
                         });

}  // namespace

#endif  // RDKIT_ETKDG_2026_09_API
