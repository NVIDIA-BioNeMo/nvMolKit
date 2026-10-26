// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "src/etkdg_impl.h"
#include "src/etkdg_stage_stereochem_checks.h"
#include "src/forcefields/mmff_kernels.h"
#include "src/forcefields/mmff_kernels_device_dispatch.cuh"
#include "src/utils/device.h"
#include "src/utils/device_vector.h"
#include "src/utils/symmetric_eigenvalues_3x3.cuh"

namespace {

struct TorsionCase {
  double positions[12];
  double coefficients[3];
};

__global__ void torsionEnergyCases(const TorsionCase* cases, int count, double* full, float* single) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const auto& sample = cases[index];
  full[index]        = nvMolKit::MMFF::fp64::torsionEnergy(sample.positions,
                                                    0,
                                                    1,
                                                    2,
                                                    3,
                                                    sample.coefficients[0],
                                                    sample.coefficients[1],
                                                    sample.coefficients[2]);
  single[index]      = nvMolKit::MMFF::fp32::torsionEnergy(sample.positions,
                                                      0,
                                                      1,
                                                      2,
                                                      3,
                                                      static_cast<float>(sample.coefficients[0]),
                                                      static_cast<float>(sample.coefficients[1]),
                                                      static_cast<float>(sample.coefficients[2]));
}

struct SymmetricMatrix {
  double entries[6];
};

__global__ void eigenvalueCases(const SymmetricMatrix* matrices, int count, double* values) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const auto& sample = matrices[index];
  nvMolKit::symmetricEigenvalues3x3(sample.entries[0],
                                    sample.entries[1],
                                    sample.entries[2],
                                    sample.entries[3],
                                    sample.entries[4],
                                    sample.entries[5],
                                    values[3 * index],
                                    values[3 * index + 1],
                                    values[3 * index + 2]);
}

}  // namespace

TEST(TrigIdentities, MmffTorsionMatchesTrigonometricReference) {
  std::vector<TorsionCase> cases;
  std::vector<double>      expected;
  for (const std::array<double, 3> coefficients : {
         std::array{  0.0,   0.0,   0.0},
         std::array{  1.0,   0.0,   0.0},
         std::array{  0.0,   1.0,   0.0},
         std::array{  0.0,   0.0,   1.0},
         std::array{  0.3,  -2.7,   1.2},
         std::array{-10.0, 100.0, -0.01}
  }) {
    for (int degrees = -180; degrees <= 180; ++degrees) {
      const double angle  = degrees * std::numbers::pi / 180.0;
      const double cosine = std::cos(angle);
      const double sine   = std::sin(angle);
      cases.push_back({
        {0, 1, 0, 0, 0, 0, 1, 0, 0, 1, cosine, sine},
        {coefficients[0], coefficients[1], coefficients[2]}
      });
      expected.push_back(0.5 * (coefficients[0] * (1 + cosine) + coefficients[1] * (1 - std::cos(2 * angle)) +
                                coefficients[2] * (1 + std::cos(3 * angle))));
    }
  }
  nvMolKit::AsyncDeviceVector<TorsionCase> inputs;
  nvMolKit::AsyncDeviceVector<double>      full(cases.size());
  nvMolKit::AsyncDeviceVector<float>       single(cases.size());
  inputs.setFromVector(cases);
  torsionEnergyCases<<<(cases.size() + 127) / 128, 128>>>(inputs.data(), cases.size(), full.data(), single.data());
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  std::vector<double> fullValues(cases.size());
  std::vector<float>  singleValues(cases.size());
  full.copyToHost(fullValues);
  single.copyToHost(singleValues);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (size_t index = 0; index < cases.size(); ++index) {
    const auto&  coefficients = cases[index].coefficients;
    const double scale =
      std::max(1.0, std::abs(coefficients[0]) + std::abs(coefficients[1]) + std::abs(coefficients[2]));
    EXPECT_NEAR(fullValues[index], expected[index], 2e-12 * scale) << index;
    EXPECT_NEAR(singleValues[index], expected[index], 2e-6 * scale) << index;
  }
}

TEST(TrigIdentities, CardanoEigenvaluesMatchRotatedKnownSpectra) {
  std::vector<SymmetricMatrix>       matrices;
  std::vector<std::array<double, 3>> expected;
  for (const std::array<double, 3> spectrum : {
         std::array{0.0,         0.0,  0.0},
         std::array{1.0,         1.0,  1.0},
         std::array{3.0,         2.0,  1.0},
         std::array{1.0,         0.0,  0.0},
         std::array{1.0,         1.0,  0.0},
         std::array{1.0,        1e-8,  0.0},
         std::array{1.0, 1.0 - 1e-10, -1.0},
         std::array{0.0,        -1.0, -5.0}
  }) {
    for (const double scale : {1e-6, 1.0, 1e6}) {
      for (int rotation = 0; rotation < 32; ++rotation) {
        const double angle       = rotation * 0.17;
        const double secondAngle = rotation * 0.31;
        const double cosine = std::cos(angle), sine = std::sin(angle);
        const double secondCosine = std::cos(secondAngle), secondSine = std::sin(secondAngle);
        const double basis[3][3] = {
          {cosine * secondCosine,  -sine, cosine * secondSine},
          {  sine * secondCosine, cosine,   sine * secondSine},
          {          -secondSine,      0,        secondCosine}
        };
        double matrix[3][3] = {};
        for (int row = 0; row < 3; ++row) {
          for (int column = 0; column < 3; ++column) {
            for (int axis = 0; axis < 3; ++axis) {
              matrix[row][column] += basis[row][axis] * spectrum[axis] * scale * basis[column][axis];
            }
          }
        }
        matrices.push_back({
          {matrix[0][0], matrix[0][1], matrix[0][2], matrix[1][1], matrix[1][2], matrix[2][2]}
        });
        expected.push_back({spectrum[0] * scale, spectrum[1] * scale, spectrum[2] * scale});
      }
    }
  }
  nvMolKit::AsyncDeviceVector<SymmetricMatrix> inputs;
  nvMolKit::AsyncDeviceVector<double>          outputs(3 * matrices.size());
  inputs.setFromVector(matrices);
  eigenvalueCases<<<(matrices.size() + 127) / 128, 128>>>(inputs.data(), matrices.size(), outputs.data());
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  std::vector<double> values(3 * matrices.size());
  outputs.copyToHost(values);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (size_t index = 0; index < matrices.size(); ++index) {
    const double scale = std::max(std::abs(expected[index][0]), std::abs(expected[index][2]));
    for (int axis = 0; axis < 3; ++axis) {
      EXPECT_NEAR(values[3 * index + axis], expected[index][axis], 2e-7 * scale) << index << ", " << axis;
    }
    EXPECT_GE(values[3 * index], values[3 * index + 1]);
    EXPECT_GE(values[3 * index + 1], values[3 * index + 2]);
    EXPECT_NEAR(values[3 * index] + values[3 * index + 1] + values[3 * index + 2],
                expected[index][0] + expected[index][1] + expected[index][2],
                2e-14 * scale)
      << index;
  }
}

TEST(TrigIdentities, DoubleBondStereoSignsBoundariesAndInactiveSystems) {
  using namespace nvMolKit::detail;
  for (const int dimensions : {3, 4}) {
    nvMolKit::ScopedStream stream;
    ETKDGContext           context;
    std::vector<EmbedArgs> arguments;
    std::vector<uint8_t>   active, expected;
    std::vector<double>    positions;
    for (const double cosine : {-1.0, -0.5, -1e-8, 0.0, 1e-8, 0.5, 1.0}) {
      for (const int sign : {-1, 1}) {
        for (const uint8_t enabled : {0, 1}) {
          const int index = arguments.size();
          arguments.emplace_back();
          arguments.back().stereoDoubleBonds.push_back({
            {0, 1, 2, 3},
            sign
          });
          context.systemHost.atomStarts.push_back(4 * index);
          const double coordinates[4][3] = {
            {0,      1,                              0},
            {0,      0,                              0},
            {1,      0,                              0},
            {1, cosine, std::sqrt(1 - cosine * cosine)}
          };
          for (const auto& coordinate : coordinates) {
            positions.insert(positions.end(), coordinate, coordinate + 3);
            if (dimensions == 4) {
              positions.push_back(123.0);
            }
          }
          active.push_back(enabled);
          expected.push_back(enabled && cosine * sign > 0);
        }
      }
    }
    context.nTotalSystems = arguments.size();
    context.systemHost.atomStarts.push_back(4 * context.nTotalSystems);
    context.systemDevice.positions.setStream(stream.stream());
    context.activeThisStage.setStream(stream.stream());
    context.failedThisStage.setStream(stream.stream());
    context.systemDevice.positions.setFromVector(positions);
    context.activeThisStage.setFromVector(active);
    context.failedThisStage.resize(active.size());
    context.failedThisStage.zero();
    ETKDGDoubleBondStereoCheckStage stage(context, arguments, dimensions, stream.stream());
    stage.execute(context);
    std::vector<uint8_t> failures(expected.size());
    context.failedThisStage.copyToHost(failures);
    ASSERT_EQ(cudaStreamSynchronize(stream.stream()), cudaSuccess);
    EXPECT_EQ(failures, expected) << dimensions;
  }
}

TEST(TrigIdentities, DoubleBondStereoDegenerateEmptyAndMalformedTerms) {
  using namespace nvMolKit::detail;
  ETKDGContext context;
  context.nTotalSystems         = 2;
  context.systemHost.atomStarts = {0, 4, 8};
  context.systemDevice.positions.setFromVector(std::vector<double>(24, 0.0));
  context.activeThisStage.setFromVector({1, 1});
  context.failedThisStage.setFromVector({0, 0});
  std::vector<EmbedArgs> arguments(2);
  arguments[0].stereoDoubleBonds = {
    {{0, 1, 2, 3},  1},
    {{0, 1, 2, 3}, -1}
  };
  ETKDGDoubleBondStereoCheckStage stage(context, arguments, 3);
  stage.execute(context);
  std::vector<uint8_t> failures(2);
  context.failedThisStage.copyToHost(failures);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(failures, (std::vector<uint8_t>{0, 0}));

  arguments[0].stereoDoubleBonds = {
    {{0, 1, 2}, 1}
  };
  EXPECT_THROW(ETKDGDoubleBondStereoCheckStage(context, arguments, 3), std::runtime_error);

  ETKDGContext empty;
  empty.nTotalSystems = 0;
  ETKDGDoubleBondStereoCheckStage emptyStage(empty, {}, 3);
  emptyStage.execute(empty);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}
