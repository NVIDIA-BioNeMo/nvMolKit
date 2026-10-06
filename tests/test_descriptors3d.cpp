// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <GraphMol/Conformer.h>
#include <GraphMol/Descriptors/AUTOCORR3D.h>
#include <GraphMol/Descriptors/GETAWAY.h>
#include <GraphMol/Descriptors/MORSE.h>
#include <GraphMol/Descriptors/PBF.h>
#include <GraphMol/Descriptors/RDF.h>
#include <GraphMol/Descriptors/USRDescriptor.h>
#include <GraphMol/Descriptors/WHIM.h>
#include <GraphMol/RWMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "src/conformer/conformer_coord_upload.h"
#include "src/descriptors3d_mol.h"
#include "src/utils/device.h"

using nvMolKit::Property3D;

namespace {

using Point = std::array<double, 3>;

constexpr std::array<Property3D, 10> kMomentProperties = {
  Property3D::PMI1,
  Property3D::PMI2,
  Property3D::PMI3,
  Property3D::RadiusOfGyration,
  Property3D::NPR1,
  Property3D::NPR2,
  Property3D::InertialShapeFactor,
  Property3D::Eccentricity,
  Property3D::Asphericity,
  Property3D::SpherocityIndex,
};

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

nvMolKit::Property3DOptions unitWeightOptions() {
  nvMolKit::Property3DOptions options;
  options.moments.useAtomicMasses = false;
  return options;
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
  EXPECT_EQ(nvMolKit::property3DWidth(Property3D::PBF), 1);
  EXPECT_EQ(nvMolKit::property3DWidth(Property3D::WHIM), 114);
  EXPECT_THROW(nvMolKit::property3DFromName("PMI4"), std::invalid_argument);
}

TEST_F(Descriptors3DTest, UnitWeightMomentsMatchClosedForm) {
  const std::vector<Property3D> properties(kMomentProperties.begin(), kMomentProperties.end());
  auto results = nvMolKit::calc3DProperties<double>(mols_, properties, unitWeightOptions(), nullptr);
  ASSERT_EQ(results.properties.size(), properties.size());

  const auto pmi1 = toHost(results.properties.at(Property3D::PMI1));
  const auto pmi2 = toHost(results.properties.at(Property3D::PMI2));
  const auto pmi3 = toHost(results.properties.at(Property3D::PMI3));
  const auto rg   = toHost(results.properties.at(Property3D::RadiusOfGyration));
  const auto npr1 = toHost(results.properties.at(Property3D::NPR1));
  const auto npr2 = toHost(results.properties.at(Property3D::NPR2));
  const auto isf  = toHost(results.properties.at(Property3D::InertialShapeFactor));
  const auto ecc  = toHost(results.properties.at(Property3D::Eccentricity));
  const auto asph = toHost(results.properties.at(Property3D::Asphericity));
  const auto sph  = toHost(results.properties.at(Property3D::SpherocityIndex));
  ASSERT_EQ(pmi1.size(), 3u);  // line + two square conformers; the conformer-less mol adds no rows

  // Rows are labeled by input molecule and per-molecule conformer position.
  EXPECT_EQ(toHost(results.molIndices), (std::vector<int32_t>{0, 2, 2}));
  EXPECT_EQ(toHost(results.confIndices), (std::vector<int32_t>{0, 0, 1}));

  constexpr double kTol = 1e-12;
  EXPECT_NEAR(pmi1[0], 0.0, kTol);
  EXPECT_NEAR(pmi2[0], 2.0, kTol);
  EXPECT_NEAR(pmi3[0], 2.0, kTol);
  EXPECT_NEAR(rg[0], std::sqrt(2.0 / 3.0), kTol);
  EXPECT_NEAR(npr1[0], 0.0, kTol);
  EXPECT_NEAR(npr2[0], 1.0, kTol);
  EXPECT_NEAR(isf[0], 0.0, kTol);
  EXPECT_NEAR(ecc[0], 1.0, kTol);
  EXPECT_NEAR(asph[0], 1.0, kTol);
  EXPECT_NEAR(sph[0], 0.0, kTol);
  for (int row = 1; row < 3; ++row) {
    EXPECT_NEAR(pmi1[row], 4.0, kTol);
    EXPECT_NEAR(pmi2[row], 4.0, kTol);
    EXPECT_NEAR(pmi3[row], 8.0, kTol);
    EXPECT_NEAR(rg[row], std::sqrt(2.0), kTol);
    EXPECT_NEAR(npr1[row], 0.5, kTol);
    EXPECT_NEAR(npr2[row], 0.5, kTol);
    EXPECT_NEAR(isf[row], 0.125, kTol);
    EXPECT_NEAR(ecc[row], std::sqrt(3.0) / 2.0, kTol);
    EXPECT_NEAR(asph[row], 0.25, kTol);
    EXPECT_NEAR(sph[row], 0.0, kTol);
  }
}

TEST_F(Descriptors3DTest, SinglePrecisionMatchesClosedForm) {
  const std::vector<Property3D> properties = {Property3D::PMI1, Property3D::PMI3, Property3D::RadiusOfGyration};
  auto results = nvMolKit::calc3DProperties<float>(mols_, properties, unitWeightOptions(), nullptr);

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
    nvMolKit::calc3DProperties<double>(mols,
                                       {Property3D::PMI3, Property3D::RadiusOfGyration, Property3D::SpherocityIndex},
                                       nvMolKit::Property3DOptions{},
                                       nullptr);
  EXPECT_EQ(results.properties.count(Property3D::PMI1), 0u);
  EXPECT_NEAR(toHost(results.properties.at(Property3D::PMI3))[0], inertia, 1e-12);
  EXPECT_NEAR(toHost(results.properties.at(Property3D::RadiusOfGyration))[0], std::sqrt(inertia / (m0 + m1)), 1e-12);
  EXPECT_NEAR(toHost(results.properties.at(Property3D::SpherocityIndex))[0], 0.0, 1e-12);
}

TEST_F(Descriptors3DTest, DeviceCoordinatesMatchMoleculeCoordinates) {
  const auto uploaded = nvMolKit::uploadConformerCoordinates(mols_, nullptr);
  const auto view     = nvMolKit::makeDeviceCoordView(uploaded);
  ASSERT_EQ(view.numConformers, 3);
  ASSERT_EQ(view.nMols, 3);

  auto fromMols = nvMolKit::calc3DProperties<double>(mols_, {Property3D::PMI2}, nvMolKit::Property3DOptions{}, nullptr);
  auto fromDevice =
    nvMolKit::calc3DProperties<double>(mols_, {Property3D::PMI2}, nvMolKit::Property3DOptions{}, nullptr, &view);
  EXPECT_EQ(toHost(fromMols.properties.at(Property3D::PMI2)), toHost(fromDevice.properties.at(Property3D::PMI2)));
  // Rows follow the caller's coordinates, so no labels are produced.
  EXPECT_EQ(fromDevice.molIndices.size(), 0u);
  EXPECT_EQ(fromDevice.confIndices.size(), 0u);
}

TEST(Descriptors3DProjection, MatchesRdkitPbfAndWhim) {
  auto                                   mol       = molWithConformers("CCCO",
                                                                       {
                                 {{-1.3, 0.2, 0.7}, {-0.2, -0.8, 0.1}, {0.9, 0.4, -0.6},  {1.7, 1.1, 0.9}},
                                 {{2.0, -1.0, 0.5},  {2.7, 0.3, -0.4}, {3.9, -0.2, 0.8}, {4.6, 1.0, -0.7}},
  });
  const std::vector<const RDKit::ROMol*> mols      = {mol.get()};
  constexpr double                       threshold = 0.01;
  nvMolKit::Property3DOptions            options;
  options.whim.threshold = threshold;
  auto results = nvMolKit::calc3DProperties<double>(mols, {Property3D::PBF, Property3D::WHIM}, options, nullptr);

  // PBF shares the unweighted PCA with WHIM; requesting it alone must not change its value.
  auto       pbfAlone       = nvMolKit::calc3DProperties<double>(mols, {Property3D::PBF}, options, nullptr);
  const auto pbfAloneValues = toHost(pbfAlone.properties.at(Property3D::PBF));
  const auto pbfWithWhim    = toHost(results.properties.at(Property3D::PBF));
  for (size_t confIdx = 0; confIdx < pbfAloneValues.size(); ++confIdx) {
    EXPECT_NEAR(pbfAloneValues[confIdx], pbfWithWhim[confIdx], 1e-12);
  }

  const auto pbf  = toHost(results.properties.at(Property3D::PBF));
  const auto whim = toHost(results.properties.at(Property3D::WHIM));
  ASSERT_EQ(pbf.size(), 2u);
  ASSERT_EQ(whim.size(), 2u * nvMolKit::kNumWhimProperties);
  for (int confIdx = 0; confIdx < 2; ++confIdx) {
    RDKit::RWMol referenceMol(*mol);
    referenceMol.clearComputedProps();
    EXPECT_NEAR(pbf[confIdx], RDKit::Descriptors::PBF(referenceMol, confIdx), 2e-10);
    std::vector<double> expectedWhim;
    RDKit::Descriptors::WHIM(*mol, expectedWhim, confIdx, threshold);
    ASSERT_EQ(expectedWhim.size(), static_cast<size_t>(nvMolKit::kNumWhimProperties));
    for (int valueIdx = 0; valueIdx < nvMolKit::kNumWhimProperties; ++valueIdx) {
      EXPECT_NEAR(whim[confIdx * nvMolKit::kNumWhimProperties + valueIdx], expectedWhim[valueIdx], 1.1e-3)
        << "conformer " << confIdx << ", WHIM value " << valueIdx;
    }
  }
}

TEST(Descriptors3DProjection, WhimChannelBroadcastMatchesRdkitAcrossMixedGeometries) {
  // Unequal atom weights give channel lanes different covariance matrices and Jacobi paths.
  // Cross group/warp/block boundaries, including a partially occupied final warp.
  std::mt19937                               rng(346);
  std::uniform_real_distribution<double>     coordinate(-3.0, 3.0);
  std::vector<std::unique_ptr<RDKit::RWMol>> owned;
  std::vector<const RDKit::ROMol*>           mols;
  nvMolKit::ScopedStream                     stream;
  for (const char* smiles : {"CCCO", "CCNCCOCCF", "CCOC(=O)NCCSCCCl"}) {
    std::unique_ptr<RDKit::RWMol>   shape(RDKit::SmilesToMol(smiles));
    std::vector<std::vector<Point>> conformers;
    for (int confIdx = 0; confIdx < 23; ++confIdx) {
      std::vector<Point> points;
      for (unsigned atomIdx = 0; atomIdx < shape->getNumAtoms(); ++atomIdx) {
        const double x = coordinate(rng);
        const double y = coordinate(rng);
        // Include planar and nearly planar matrices alongside fully 3D matrices.
        const double z = confIdx % 3 == 0 ? 0.0 : coordinate(rng) * (confIdx % 3 == 1 ? 1e-5 : 1.0);
        points.push_back({x, y, z});
      }
      conformers.push_back(std::move(points));
    }
    owned.push_back(molWithConformers(smiles, conformers));
    mols.push_back(owned.back().get());
  }
  for (const double threshold : {0.001, 0.01, 0.0105}) {
    SCOPED_TRACE(threshold);
    nvMolKit::Property3DOptions options;
    options.whim.threshold = threshold;
    std::vector<double> expected;
    for (const auto* mol : mols) {
      for (unsigned confIdx = 0; confIdx < mol->getNumConformers(); ++confIdx) {
        std::vector<double> row;
        RDKit::Descriptors::WHIM(*mol, row, confIdx, threshold);
        expected.insert(expected.end(), row.begin(), row.end());
      }
    }
    const auto results = nvMolKit::calc3DProperties<double>(mols, {Property3D::WHIM}, options, nullptr);
    const auto actual  = toHost(results.properties.at(Property3D::WHIM));
    const auto floatResults =
      nvMolKit::calc3DProperties<float>(mols, {Property3D::PBF, Property3D::WHIM}, options, stream.stream());
    const auto floatActual = toHost(floatResults.properties.at(Property3D::WHIM));
    ASSERT_EQ(actual.size(), expected.size());
    ASSERT_EQ(floatActual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(actual[i], expected[i], 1.1e-3) << "WHIM value " << i;
      EXPECT_NEAR(floatActual[i], expected[i], 1.1e-3) << "float WHIM value " << i;
    }
    const auto repeated = nvMolKit::calc3DProperties<double>(mols, {Property3D::WHIM}, options, stream.stream());
    EXPECT_EQ(toHost(repeated.properties.at(Property3D::WHIM)), actual);
  }
}

TEST(Descriptors3DPairwise, MatchesRdkitRdfAndMorse) {
  auto                                   mol  = molWithConformers("CCCO",
                                                                  {
                                 {{-1.3, 0.2, 0.7}, {-0.2, -0.8, 0.1}, {0.9, 0.4, -0.6},  {1.7, 1.1, 0.9}},
                                 {{2.0, -1.0, 0.5},  {2.7, 0.3, -0.4}, {3.9, -0.2, 0.8}, {4.6, 1.0, -0.7}},
  });
  const std::vector<const RDKit::ROMol*> mols = {mol.get()};
  auto results = nvMolKit::calc3DProperties<double>(mols, {Property3D::RDF, Property3D::MORSE}, {}, nullptr);

  const std::array<std::pair<Property3D, int>, 2> cases = {
    std::pair{  Property3D::RDF,   nvMolKit::kNumRdfProperties},
    std::pair{Property3D::MORSE, nvMolKit::kNumMorseProperties}
  };
  for (const auto& [property, width] : cases) {
    const auto values = toHost(results.properties.at(property));
    ASSERT_EQ(values.size(), 2u * width);
    // One pass over atom pairs serves both properties; requesting one alone must not change it.
    auto       alone       = nvMolKit::calc3DProperties<double>(mols, {property}, {}, nullptr);
    const auto aloneValues = toHost(alone.properties.at(property));
    for (int confIdx = 0; confIdx < 2; ++confIdx) {
      std::vector<double> expected;
      if (property == Property3D::RDF) {
        RDKit::Descriptors::RDF(*mol, expected, confIdx);
      } else {
        RDKit::Descriptors::MORSE(*mol, expected, confIdx);
      }
      ASSERT_EQ(expected.size(), static_cast<size_t>(width));
      for (int valueIdx = 0; valueIdx < width; ++valueIdx) {
        const size_t idx = static_cast<size_t>(confIdx) * width + valueIdx;
        EXPECT_NEAR(values[idx], expected[valueIdx], 1.1e-3)
          << nvMolKit::property3DName(property) << " conformer " << confIdx << ", value " << valueIdx;
        EXPECT_EQ(values[idx], aloneValues[idx]);
      }
    }
  }
}

TEST(Descriptors3DPairwise, MatchesRdkitAutocorr3D) {
  auto                                   mol  = molWithConformers("CCCO",
                                                                  {
                                 {{-1.3, 0.2, 0.7}, {-0.2, -0.8, 0.1}, {0.9, 0.4, -0.6},  {1.7, 1.1, 0.9}},
                                 {{2.0, -1.0, 0.5},  {2.7, 0.3, -0.4}, {3.9, -0.2, 0.8}, {4.6, 1.0, -0.7}},
  });
  const std::vector<const RDKit::ROMol*> mols = {mol.get()};
  auto       results = nvMolKit::calc3DProperties<double>(mols, {Property3D::AUTOCORR3D, Property3D::RDF}, {}, nullptr);
  const auto values  = toHost(results.properties.at(Property3D::AUTOCORR3D));
  ASSERT_EQ(values.size(), 2u * nvMolKit::kNumAutocorr3DProperties);
  for (int confIdx = 0; confIdx < 2; ++confIdx) {
    std::vector<double> expected;
    RDKit::Descriptors::AUTOCORR3D(*mol, expected, confIdx);
    ASSERT_EQ(expected.size(), static_cast<size_t>(nvMolKit::kNumAutocorr3DProperties));
    for (int valueIdx = 0; valueIdx < nvMolKit::kNumAutocorr3DProperties; ++valueIdx) {
      EXPECT_NEAR(values[confIdx * nvMolKit::kNumAutocorr3DProperties + valueIdx], expected[valueIdx], 1.1e-3)
        << "conformer " << confIdx << ", value " << valueIdx;
    }
  }
}

TEST(Descriptors3DGetaway, MatchesRdkitGetaway) {
  auto                                   mol  = molWithConformers("CCCO",
                                                                  {
                                 {{-1.3, 0.2, 0.7}, {-0.2, -0.8, 0.1}, {0.9, 0.4, -0.6},  {1.7, 1.1, 0.9}},
                                 {{2.0, -1.0, 0.5},  {2.7, 0.3, -0.4}, {3.9, -0.2, 0.8}, {4.6, 1.0, -0.7}},
  });
  const std::vector<const RDKit::ROMol*> mols = {mol.get()};
  auto       results = nvMolKit::calc3DProperties<double>(mols, {Property3D::GETAWAY}, {}, nullptr);
  const auto values  = toHost(results.properties.at(Property3D::GETAWAY));
  ASSERT_EQ(values.size(), 2u * nvMolKit::kNumGetawayProperties);
  for (int confIdx = 0; confIdx < 2; ++confIdx) {
    std::vector<double> expected;
    RDKit::Descriptors::GETAWAY(*mol, expected, confIdx);
    ASSERT_EQ(expected.size(), static_cast<size_t>(nvMolKit::kNumGetawayProperties));
    for (int valueIdx = 0; valueIdx < nvMolKit::kNumGetawayProperties; ++valueIdx) {
      EXPECT_NEAR(values[confIdx * nvMolKit::kNumGetawayProperties + valueIdx], expected[valueIdx], 1.1e-3)
        << "conformer " << confIdx << ", value " << valueIdx;
    }
  }
}

TEST(Descriptors3DUsr, MatchesRdkitUsrAndUsrcat) {
  auto                                   mol  = molWithConformers("CCCO",
                                                                  {
                                 {{-1.3, 0.2, 0.7}, {-0.2, -0.8, 0.1}, {0.9, 0.4, -0.6},  {1.7, 1.1, 0.9}},
                                 {{2.0, -1.0, 0.5},  {2.7, 0.3, -0.4}, {3.9, -0.2, 0.8}, {4.6, 1.0, -0.7}},
  });
  const std::vector<const RDKit::ROMol*> mols = {mol.get()};
  auto       results = nvMolKit::calc3DProperties<double>(mols, {Property3D::USR, Property3D::USRCAT}, {}, nullptr);
  const auto usr     = toHost(results.properties.at(Property3D::USR));
  const auto usrcat  = toHost(results.properties.at(Property3D::USRCAT));
  ASSERT_EQ(usr.size(), 2u * nvMolKit::kNumUsrProperties);
  ASSERT_EQ(usrcat.size(), 2u * nvMolKit::kNumUsrcatProperties);
  for (int confIdx = 0; confIdx < 2; ++confIdx) {
    std::vector<double> expectedUsr(nvMolKit::kNumUsrProperties);
    RDKit::Descriptors::USR(*mol, expectedUsr, confIdx);
    std::vector<double>                    expectedUsrcat(nvMolKit::kNumUsrcatProperties);
    std::vector<std::vector<unsigned int>> atomIds;
    RDKit::Descriptors::USRCAT(*mol, expectedUsrcat, atomIds, confIdx);
    for (int valueIdx = 0; valueIdx < nvMolKit::kNumUsrcatProperties; ++valueIdx) {
      // Skews (every third value) of near-symmetric distance sets are rounding noise; see the Python tests.
      const double tolerance = valueIdx % 3 == 2 ? 1e-3 : 1e-9;
      EXPECT_NEAR(usrcat[confIdx * nvMolKit::kNumUsrcatProperties + valueIdx], expectedUsrcat[valueIdx], tolerance)
        << "conformer " << confIdx << ", USRCAT value " << valueIdx;
      if (valueIdx < nvMolKit::kNumUsrProperties) {
        EXPECT_NEAR(usr[confIdx * nvMolKit::kNumUsrProperties + valueIdx], expectedUsr[valueIdx], tolerance)
          << "conformer " << confIdx << ", USR value " << valueIdx;
      }
    }
  }
}

TEST(Descriptors3DProjection, PbfMinimumAtomBoundaryAndWhimEmptyShape) {
  auto                                   threeAtoms = molWithConformers("CCC",
                                                                        {
                                        {{0.0, 0.0, 0.0}, {1.0, 0.2, 0.3}, {0.1, 1.0, -0.4}},
  });
  const std::vector<const RDKit::ROMol*> mols       = {threeAtoms.get()};
  auto pbfResults = nvMolKit::calc3DProperties<double>(mols, {Property3D::PBF}, nvMolKit::Property3DOptions{}, nullptr);
  EXPECT_EQ(toHost(pbfResults.properties.at(Property3D::PBF)), (std::vector<double>{0.0}));

  auto non3D = molWithConformers("CCCC",
                                 {
                                   {{0.0, 0.0, 0.0}, {1.0, 0.2, 0.4}, {0.1, 1.3, -0.7}, {1.5, 1.1, 0.8}},
  });
  non3D->getConformer().set3D(false);
  const std::vector<const RDKit::ROMol*> non3DMols = {non3D.get()};
  auto                                   non3DResults =
    nvMolKit::calc3DProperties<double>(non3DMols, {Property3D::PBF}, nvMolKit::Property3DOptions{}, nullptr);
  EXPECT_EQ(toHost(non3DResults.properties.at(Property3D::PBF)), (std::vector<double>{0.0}));

  const std::vector<const RDKit::ROMol*> empty;
  auto                                   whimResults =
    nvMolKit::calc3DProperties<double>(empty, {Property3D::WHIM}, nvMolKit::Property3DOptions{}, nullptr);
  EXPECT_EQ(whimResults.properties.at(Property3D::WHIM).size(), 0u);
}

TEST_F(Descriptors3DTest, WhimScratchChunkingAndOversizedRows) {
  // Square conformers (4 atoms) with unit WHIM weights, evaluated directly at the device level.
  const std::vector<const RDKit::ROMol*> mols     = {square_.get()};
  const auto                             uploaded = nvMolKit::uploadConformerCoordinates(mols, nullptr);
  const auto                             view     = nvMolKit::makeDeviceCoordView(uploaded);
  nvMolKit::AsyncDeviceVector<int32_t>   moleculeAtomStarts(2);
  moleculeAtomStarts.copyFromHost(std::vector<int32_t>{0, 4});
  nvMolKit::AsyncDeviceVector<double> atomPropertyWeights(4 * 6);
  atomPropertyWeights.copyFromHost(std::vector<double>(4 * 6, 1.0));

  nvMolKit::Property3DDeviceInputs inputs;
  inputs.moleculeAtomStarts  = moleculeAtomStarts.data();
  inputs.atomPropertyWeights = atomPropertyWeights.data();
  const auto whimFor         = [&](const int32_t maxMoleculeAtoms) {
    inputs.maxMoleculeAtoms = maxMoleculeAtoms;
    auto results            = nvMolKit::calc3DPropertiesGpu<double>(view, inputs, {Property3D::WHIM}, {}, nullptr);
    return toHost(results.at(Property3D::WHIM));
  };

  const auto reference = whimFor(4);
  ASSERT_EQ(reference.size(), 2u * nvMolKit::kNumWhimProperties);
  for (const double value : reference) {
    EXPECT_FALSE(isNanBits(value));
  }
  // A per-conformer scratch slot larger than the 256 MB budget runs one conformer per launch.
  EXPECT_EQ(whimFor(40'000'000), reference);
  // Rows longer than the scratch slot are rejected rather than overrunning it.
  for (const double value : whimFor(3)) {
    EXPECT_TRUE(isNanBits(value));
  }
}

TEST_F(Descriptors3DTest, AtomCountMismatchProducesNaN) {
  // Coordinates uploaded for [square, line] but interpreted against [line, square] molecules.
  const std::vector<const RDKit::ROMol*> swapped  = {square_.get(), line_.get()};
  const auto                             uploaded = nvMolKit::uploadConformerCoordinates(swapped, nullptr);
  const auto                             view     = nvMolKit::makeDeviceCoordView(uploaded);

  const std::vector<const RDKit::ROMol*> mols = {line_.get(), square_.get()};
  auto                                   results =
    nvMolKit::calc3DProperties<double>(mols, {Property3D::RadiusOfGyration}, unitWeightOptions(), nullptr, &view);
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

    auto results =
      nvMolKit::calc3DProperties<double>(mols, {Property3D::RadiusOfGyration}, unitWeightOptions(), nullptr, &view);
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
                                              static_cast<Property3D>(nvMolKit::kAllProperty3D.size())};
  EXPECT_THROW(nvMolKit::calc3DProperties<double>(mols_, properties, nvMolKit::Property3DOptions{}, nullptr),
               std::invalid_argument);
  EXPECT_THROW(
    nvMolKit::calc3DProperties<float>(mols_, {static_cast<Property3D>(-1)}, nvMolKit::Property3DOptions{}, nullptr),
    std::invalid_argument);
}

TEST_F(Descriptors3DTest, RejectsInvalidSelectionAndBatchMismatch) {
  EXPECT_THROW(nvMolKit::calc3DProperties<double>(mols_, {}, nvMolKit::Property3DOptions{}, nullptr),
               std::invalid_argument);
  EXPECT_THROW(nvMolKit::calc3DProperties<double>(mols_,
                                                  {Property3D::PMI1, Property3D::PMI1},
                                                  nvMolKit::Property3DOptions{},
                                                  nullptr),
               std::invalid_argument);

  const auto                             uploaded  = nvMolKit::uploadConformerCoordinates(mols_, nullptr);
  const auto                             view      = nvMolKit::makeDeviceCoordView(uploaded);
  const std::vector<const RDKit::ROMol*> fewerMols = {line_.get()};
  EXPECT_THROW(
    nvMolKit::calc3DProperties<double>(fewerMols, {Property3D::PMI1}, nvMolKit::Property3DOptions{}, nullptr, &view),
    std::invalid_argument);
}
