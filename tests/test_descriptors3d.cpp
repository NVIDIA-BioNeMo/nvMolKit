// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <GraphMol/Conformer.h>
#include <GraphMol/RWMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "src/conformer/conformer_coord_upload.h"
#include "src/descriptors3d_mol.h"

using nvMolKit::Property3D;

namespace {

using Point = std::array<double, 3>;

std::unique_ptr<RDKit::RWMol> molWithConformers(const char* smiles, const std::vector<std::vector<Point>>& confs) {
  std::unique_ptr<RDKit::RWMol> mol(RDKit::SmilesToMol(smiles));
  for (const auto& points : confs) {
    auto conf = std::make_unique<RDKit::Conformer>(mol->getNumAtoms());
    for (size_t atomIdx = 0; atomIdx < points.size(); ++atomIdx) {
      conf->setAtomPos(atomIdx, RDGeom::Point3D(points[atomIdx][0], points[atomIdx][1], points[atomIdx][2]));
    }
    mol->addConformer(conf.release(), true);
  }
  return mol;
}

template <typename T> std::vector<T> toHost(const nvMolKit::AsyncDeviceVector<T>& values) {
  std::vector<T> host(values.size());
  values.copyToHost(host);
  EXPECT_EQ(cudaStreamSynchronize(values.stream()), cudaSuccess);
  return host;
}

//! Bitwise NaN check: the project's -ffast-math host flags fold std::isnan to false.
bool isNanBits(const double value) {
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  constexpr uint64_t kExponentMask = 0x7ff0000000000000ULL;
  constexpr uint64_t kMantissaMask = 0x000fffffffffffffULL;
  return (bits & kExponentMask) == kExponentMask && (bits & kMantissaMask) != 0;
}

class Descriptors3DTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Unit-weight geometries with closed-form principal moments.
    // Line along x at -1, 0, 1: I = diag(0, 2, 2), Rg^2 = 2/3.
    line_         = molWithConformers("CCC",
                                      {
                                {{-1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}
    });
    // Square (+-1, +-1, 0), translated to check centering: I = diag(4, 4, 8), Rg^2 = 2.
    square_       = molWithConformers("C1CCC1",
                                      {
                                  {  {1.0, 1.0, 0.0}, {-1.0, 1.0, 0.0}, {-1.0, -1.0, 0.0},  {1.0, -1.0, 0.0}},
                                  {{11.0, -4.0, 3.0}, {9.0, -4.0, 3.0},  {9.0, -6.0, 3.0}, {11.0, -6.0, 3.0}}
    });
    noConformers_ = molWithConformers("CC", {});
    mols_         = {line_.get(), noConformers_.get(), square_.get()};
  }

  std::unique_ptr<RDKit::RWMol>    line_;
  std::unique_ptr<RDKit::RWMol>    square_;
  std::unique_ptr<RDKit::RWMol>    noConformers_;
  std::vector<const RDKit::ROMol*> mols_;
};

}  // namespace

TEST(Property3DNames, RoundTripAndRejectUnknown) {
  for (const Property3D property : nvMolKit::kAllProperty3D) {
    EXPECT_EQ(nvMolKit::property3DFromName(nvMolKit::property3DName(property)), property);
  }
  EXPECT_EQ(nvMolKit::property3DName(Property3D::RadiusOfGyration), "RadiusOfGyration");
  EXPECT_THROW(nvMolKit::property3DFromName("PMI4"), std::invalid_argument);
}

TEST_F(Descriptors3DTest, UnitWeightMomentsMatchClosedForm) {
  const std::vector<Property3D> properties = {Property3D::RadiusOfGyration,
                                              Property3D::PMI3,
                                              Property3D::PMI1,
                                              Property3D::PMI2};
  auto results = nvMolKit::calc3DProperties<double>(mols_, properties, /*useAtomicMasses=*/false, nullptr);
  ASSERT_EQ(results.properties.size(), properties.size());

  const auto pmi1 = toHost(results.properties.at(Property3D::PMI1));
  const auto pmi2 = toHost(results.properties.at(Property3D::PMI2));
  const auto pmi3 = toHost(results.properties.at(Property3D::PMI3));
  const auto rg   = toHost(results.properties.at(Property3D::RadiusOfGyration));
  ASSERT_EQ(pmi1.size(), 3u);  // line + two square conformers; the conformer-less mol adds no rows

  // Rows are labeled by input molecule and per-molecule conformer position.
  EXPECT_EQ(toHost(results.molIndices), (std::vector<int32_t>{0, 2, 2}));
  EXPECT_EQ(toHost(results.confIndices), (std::vector<int32_t>{0, 0, 1}));

  constexpr double kTol = 1e-12;
  EXPECT_NEAR(pmi1[0], 0.0, kTol);
  EXPECT_NEAR(pmi2[0], 2.0, kTol);
  EXPECT_NEAR(pmi3[0], 2.0, kTol);
  EXPECT_NEAR(rg[0], std::sqrt(2.0 / 3.0), kTol);
  for (int row = 1; row < 3; ++row) {
    EXPECT_NEAR(pmi1[row], 4.0, kTol);
    EXPECT_NEAR(pmi2[row], 4.0, kTol);
    EXPECT_NEAR(pmi3[row], 8.0, kTol);
    EXPECT_NEAR(rg[row], std::sqrt(2.0), kTol);
  }
}

TEST_F(Descriptors3DTest, SinglePrecisionMatchesClosedForm) {
  const std::vector<Property3D> properties = {Property3D::PMI1, Property3D::PMI3, Property3D::RadiusOfGyration};
  auto results = nvMolKit::calc3DProperties<float>(mols_, properties, /*useAtomicMasses=*/false, nullptr);

  const std::vector<float> pmi1 = toHost(results.properties.at(Property3D::PMI1));
  const std::vector<float> pmi3 = toHost(results.properties.at(Property3D::PMI3));
  const std::vector<float> rg   = toHost(results.properties.at(Property3D::RadiusOfGyration));
  ASSERT_EQ(pmi1.size(), 3u);
  constexpr float kTol = 1e-5f;
  EXPECT_NEAR(pmi1[0], 0.0f, kTol);
  EXPECT_NEAR(pmi3[0], 2.0f, kTol);
  EXPECT_NEAR(rg[0], std::sqrt(2.0f / 3.0f), kTol);
  for (int row = 1; row < 3; ++row) {
    EXPECT_NEAR(pmi1[row], 4.0f, 4.0f * kTol);
    EXPECT_NEAR(pmi3[row], 8.0f, 8.0f * kTol);
    EXPECT_NEAR(rg[row], std::sqrt(2.0f), kTol);
  }
}

TEST(Descriptors3DMass, MassWeightedDiatomicUsesReducedMass) {
  constexpr double kBond   = 1.2;
  auto             mol     = molWithConformers("CO",
                                               {
                                 {{0.0, 0.0, 0.0}, {0.0, 0.0, kBond}}
  });
  const double     m0      = mol->getAtomWithIdx(0)->getMass();
  const double     m1      = mol->getAtomWithIdx(1)->getMass();
  const double     inertia = m0 * m1 / (m0 + m1) * kBond * kBond;

  const std::vector<const RDKit::ROMol*> mols = {mol.get()};
  auto                                   results =
    nvMolKit::calc3DProperties<double>(mols, {Property3D::PMI3, Property3D::RadiusOfGyration}, true, nullptr);
  EXPECT_EQ(results.properties.count(Property3D::PMI1), 0u);
  EXPECT_NEAR(toHost(results.properties.at(Property3D::PMI3))[0], inertia, 1e-12);
  EXPECT_NEAR(toHost(results.properties.at(Property3D::RadiusOfGyration))[0], std::sqrt(inertia / (m0 + m1)), 1e-12);
}

TEST_F(Descriptors3DTest, DeviceCoordinatesMatchMoleculeCoordinates) {
  const auto uploaded = nvMolKit::uploadConformerCoordinates(mols_, nullptr);
  const auto view     = nvMolKit::makeDeviceCoordView(uploaded);
  ASSERT_EQ(view.numConformers, 3);
  ASSERT_EQ(view.nMols, 3);

  auto fromMols   = nvMolKit::calc3DProperties<double>(mols_, {Property3D::PMI2}, true, nullptr);
  auto fromDevice = nvMolKit::calc3DProperties<double>(mols_, {Property3D::PMI2}, true, nullptr, &view);
  EXPECT_EQ(toHost(fromMols.properties.at(Property3D::PMI2)), toHost(fromDevice.properties.at(Property3D::PMI2)));
  // Rows follow the caller's coordinates, so no labels are produced.
  EXPECT_EQ(fromDevice.molIndices.size(), 0u);
  EXPECT_EQ(fromDevice.confIndices.size(), 0u);
}

TEST_F(Descriptors3DTest, AtomCountMismatchProducesNaN) {
  // Coordinates uploaded for [square, line] but interpreted against [line, square] molecules.
  const std::vector<const RDKit::ROMol*> swapped  = {square_.get(), line_.get()};
  const auto                             uploaded = nvMolKit::uploadConformerCoordinates(swapped, nullptr);
  const auto                             view     = nvMolKit::makeDeviceCoordView(uploaded);

  const std::vector<const RDKit::ROMol*> mols = {line_.get(), square_.get()};
  auto results = nvMolKit::calc3DProperties<double>(mols, {Property3D::RadiusOfGyration}, false, nullptr, &view);
  for (const double value : toHost(results.properties.at(Property3D::RadiusOfGyration))) {
    EXPECT_TRUE(isNanBits(value)) << value;
  }
}

TEST_F(Descriptors3DTest, AtomRangesOutsideCoordinatesProduceNaN) {
  // Two 4-atom square conformers occupy coordinate rows [0, 8). Each case keeps every row's atom count
  // equal to the molecule's, so only the bounds check can reject a row.
  const std::vector<const RDKit::ROMol*> mols     = {square_.get()};
  const auto                             uploaded = nvMolKit::uploadConformerCoordinates(mols, nullptr);
  ASSERT_EQ(nvMolKit::makeDeviceCoordView(uploaded).numAtoms, 8);

  const std::vector<std::vector<int32_t>> startsCases = {
    {-4, 0,  4},
    { 4, 8, 12}
  };
  const std::vector<int> badRows = {0, 1};
  for (size_t caseIdx = 0; caseIdx < startsCases.size(); ++caseIdx) {
    nvMolKit::AsyncDeviceVector<int32_t> atomStarts(startsCases[caseIdx].size());
    atomStarts.copyFromHost(startsCases[caseIdx]);
    auto view       = nvMolKit::makeDeviceCoordView(uploaded);
    view.atomStarts = atomStarts.data();

    auto results      = nvMolKit::calc3DProperties<double>(mols, {Property3D::RadiusOfGyration}, false, nullptr, &view);
    const auto rg     = toHost(results.properties.at(Property3D::RadiusOfGyration));
    const int  badRow = badRows[caseIdx];
    EXPECT_TRUE(isNanBits(rg[badRow])) << rg[badRow];
    EXPECT_NEAR(rg[1 - badRow], std::sqrt(2.0), 1e-12);
  }
}

TEST_F(Descriptors3DTest, RejectsUnknownPropertyValue) {
  const std::vector<Property3D> properties = {Property3D::PMI1,
                                              Property3D::PMI2,
                                              Property3D::PMI3,
                                              Property3D::RadiusOfGyration,
                                              static_cast<Property3D>(7)};
  EXPECT_THROW(nvMolKit::calc3DProperties<double>(mols_, properties, true, nullptr), std::invalid_argument);
  EXPECT_THROW(nvMolKit::calc3DProperties<float>(mols_, {static_cast<Property3D>(-1)}, true, nullptr),
               std::invalid_argument);
}

TEST_F(Descriptors3DTest, RejectsInvalidSelectionAndBatchMismatch) {
  EXPECT_THROW(nvMolKit::calc3DProperties<double>(mols_, {}, true, nullptr), std::invalid_argument);
  EXPECT_THROW(nvMolKit::calc3DProperties<double>(mols_, {Property3D::PMI1, Property3D::PMI1}, true, nullptr),
               std::invalid_argument);

  const auto                             uploaded  = nvMolKit::uploadConformerCoordinates(mols_, nullptr);
  const auto                             view      = nvMolKit::makeDeviceCoordView(uploaded);
  const std::vector<const RDKit::ROMol*> fewerMols = {line_.get()};
  EXPECT_THROW(nvMolKit::calc3DProperties<double>(fewerMols, {Property3D::PMI1}, true, nullptr, &view),
               std::invalid_argument);
}
