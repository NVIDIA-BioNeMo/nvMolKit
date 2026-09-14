// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <random>
#include <type_traits>
#include <vector>

#include "src/minimizer/bfgs_hessian.h"
#include "src/utils/device_vector.h"

namespace nvMolKit {

// CPU implementation for comparison
void updateInverseHessianBFGSCPU(const unsigned int dim,
                                 double*            invHessian,
                                 double*            hessDGrad,
                                 double*            dGrad,
                                 double*            xi,
                                 double*            grad) {
  constexpr double EPS = 3e-8;
  double           fac = 0, fae = 0, sumDGrad = 0, sumXi = 0;
  for (unsigned int i = 0; i < dim; i++) {
    double* ivh     = &(invHessian[i * dim]);
    double& hdgradi = hessDGrad[i];
    double* dgj     = dGrad;
    hdgradi         = 0.0;
    for (unsigned int j = 0; j < dim; ++j, ++ivh, ++dgj) {
      hdgradi += *ivh * *dgj;
    }
    fac += dGrad[i] * xi[i];

    fae += dGrad[i] * hessDGrad[i];
    sumDGrad += dGrad[i] * dGrad[i];
    sumXi += xi[i] * xi[i];
  }

  if (fac > sqrt(EPS * sumDGrad * sumXi)) {
    fac        = 1.0 / fac;
    double fad = 1.0 / fae;
    for (unsigned int i = 0; i < dim; i++) {
      dGrad[i] = fac * xi[i] - fad * hessDGrad[i];
    }
    for (unsigned int i = 0; i < dim; i++) {
      unsigned int itab = i * dim;
      double       pxi = fac * xi[i], hdgi = fad * hessDGrad[i], dgi = fae * dGrad[i];
      double *     pxj = &(xi[i]), *hdgj = &(hessDGrad[i]), *dgj = &(dGrad[i]);
      for (unsigned int j = i; j < dim; ++j, ++pxj, ++hdgj, ++dgj) {
        invHessian[itab + j] += pxi * *pxj - hdgi * *hdgj + dgi * *dgj;
        invHessian[j * dim + i] = invHessian[itab + j];
      }
    }
  }
  // generate the next direction to move:
  for (unsigned int i = 0; i < dim; i++) {
    unsigned int itab = i * dim;
    xi[i]             = 0.0;
    double&       pxi = xi[i];
    const double *ivh = &(invHessian[itab]), *gj = grad;
    for (unsigned int j = 0; j < dim; ++j, ++ivh, ++gj) {
      pxi -= *ivh * *gj;
    }
  }
}

template <typename Scalar> class BFGSHessianTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::random_device rd;
    rng.seed(42);
  }

  void generateRandomSystem(int                  dim,
                            std::vector<double>& invHessian,
                            std::vector<double>& dGrad,
                            std::vector<double>& xi,
                            std::vector<double>& grad,
                            bool                 identityHessian     = true,
                            bool                 correctDGradXiSigns = true) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    invHessian.resize(dim * dim, 0.0);
    if (identityHessian) {
      for (int i = 0; i < dim; ++i) {
        invHessian[i * dim + i] = 1.0;
      }
    } else {
      for (int i = 0; i < dim * dim; ++i) {
        invHessian[i] = dist(rng);
      }
      for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < i; ++j) {
          invHessian[i * dim + j] = invHessian[j * dim + i];
        }
      }
    }

    // Generate random vectors
    dGrad.resize(dim);
    xi.resize(dim);
    grad.resize(dim);
    for (int i = 0; i < dim; ++i) {
      dGrad[i] = dist(rng);
      xi[i]    = dist(rng);
      if (correctDGradXiSigns && dGrad[i] * xi[i] < 0) {
        xi[i] = -xi[i];
      } else if (!correctDGradXiSigns && dGrad[i] * xi[i] > 0) {
        xi[i] = -xi[i];
      }
      grad[i] = dist(rng);
    }
  }

  void copyFromDouble(AsyncDeviceVector<Scalar>& destination, const std::vector<double>& source) {
    const std::vector<Scalar> converted(source.begin(), source.end());
    destination.copyFromHost(converted);
    // AsyncDeviceVector does not own the host input; finish the copy before
    // the converted test buffer leaves scope.
    cudaCheckError(cudaDeviceSynchronize());
  }

  static constexpr double tolerance() { return std::is_same_v<Scalar, float> ? 5e-4 : 1e-5; }

  std::mt19937 rng;
};

using HessianScalarTypes = ::testing::Types<double, float>;
TYPED_TEST_SUITE(BFGSHessianTest, HessianScalarTypes);

TYPED_TEST(BFGSHessianTest, SingleSystem) {
  using Scalar = TypeParam;
  for (const int dataDim : {3, 4}) {
    for (const bool identityHessian : {true, false}) {
      SCOPED_TRACE(::testing::Message() << "dataDim=" << dataDim << ", identityHessian=" << identityHessian);

      constexpr int       numAtoms = 88;  // More than one block of rows.
      const int           dim      = numAtoms * dataDim;
      std::vector<double> cpuInvHessian, cpuDGrad, cpuXi, cpuGrad;
      this->generateRandomSystem(dim, cpuInvHessian, cpuDGrad, cpuXi, cpuGrad, identityHessian);

      AsyncDeviceVector<Scalar> gpuInvHessian(cpuInvHessian.size());
      AsyncDeviceVector<Scalar> gpuDGrad(cpuDGrad.size());
      AsyncDeviceVector<Scalar> gpuXi(cpuXi.size());
      AsyncDeviceVector<Scalar> gpuGrad(cpuGrad.size());
      AsyncDeviceVector<int>    atomStarts(2);  // One extra element for the end
      AsyncDeviceVector<int>    hessianStarts(2);
      AsyncDeviceVector<Scalar> hessDgrad(cpuDGrad.size());
      hessDgrad.zero();

      std::vector<int> atomStartsHost    = {0, numAtoms};
      std::vector<int> hessianStartsHost = {0, dim * dim};

      // Copy data to GPU
      this->copyFromDouble(gpuInvHessian, cpuInvHessian);
      this->copyFromDouble(gpuDGrad, cpuDGrad);
      this->copyFromDouble(gpuXi, cpuXi);
      this->copyFromDouble(gpuGrad, cpuGrad);
      atomStarts.copyFromHost(atomStartsHost);
      hessianStarts.copyFromHost(hessianStartsHost);

      // Run CPU version
      std::vector<double> cpuHessDGrad(cpuDGrad.size());  // Match size with dGrad

      updateInverseHessianBFGSCPU(dim,
                                  cpuInvHessian.data(),
                                  cpuHessDGrad.data(),
                                  cpuDGrad.data(),
                                  cpuXi.data(),
                                  cpuGrad.data());

      PinnedHostVector<int> activeSystemIndicesHost;
      activeSystemIndicesHost.resize(1);
      activeSystemIndicesHost[0] = 0;
      AsyncDeviceVector<int> activeSystemIndices(1);
      activeSystemIndices.zero();

      // Run GPU version
      updateInverseHessianBFGSBatch(1,
                                    nullptr,
                                    hessianStarts.data(),
                                    atomStarts.data(),
                                    gpuInvHessian.data(),
                                    gpuDGrad.data(),
                                    gpuXi.data(),
                                    hessDgrad.data(),
                                    gpuGrad.data(),
                                    dataDim,
                                    /*largeMol=*/false,
                                    activeSystemIndices.data());

      // Copy results back from GPU
      std::vector<Scalar> resInvHessianHost(cpuInvHessian.size());
      std::vector<Scalar> resDGradHost(cpuDGrad.size());
      std::vector<Scalar> resXiHost(cpuXi.size());

      gpuInvHessian.copyToHost(resInvHessianHost);
      gpuDGrad.copyToHost(resDGradHost);
      gpuXi.copyToHost(resXiHost);
      cudaCheckError(cudaDeviceSynchronize());

      EXPECT_THAT(resInvHessianHost, ::testing::Pointwise(::testing::DoubleNear(this->tolerance()), cpuInvHessian));
      EXPECT_THAT(resDGradHost, ::testing::Pointwise(::testing::DoubleNear(this->tolerance()), cpuDGrad));
      EXPECT_THAT(resXiHost, ::testing::Pointwise(::testing::DoubleNear(this->tolerance()), cpuXi));
    }
  }
}

TYPED_TEST(BFGSHessianTest, SingleSystemLarge) {
  using Scalar = TypeParam;
  for (const int dataDim : {3, 4}) {
    for (const bool identityHessian : {true, false}) {
      SCOPED_TRACE(::testing::Message() << "dataDim=" << dataDim << ", identityHessian=" << identityHessian);

      constexpr int       numAtoms = 300;
      const int           dim      = numAtoms * dataDim;
      std::vector<double> cpuInvHessian, cpuDGrad, cpuXi, cpuGrad;
      this->generateRandomSystem(dim, cpuInvHessian, cpuDGrad, cpuXi, cpuGrad, identityHessian);

      AsyncDeviceVector<Scalar> gpuInvHessian(cpuInvHessian.size());
      AsyncDeviceVector<Scalar> gpuDGrad(cpuDGrad.size());
      AsyncDeviceVector<Scalar> gpuXi(cpuXi.size());
      AsyncDeviceVector<Scalar> gpuGrad(cpuGrad.size());
      AsyncDeviceVector<int>    atomStarts(2);  // One extra element for the end
      AsyncDeviceVector<int>    hessianStarts(2);
      AsyncDeviceVector<Scalar> hessDgrad(cpuDGrad.size());
      hessDgrad.zero();

      std::vector<int> atomStartsHost    = {0, numAtoms};
      std::vector<int> hessianStartsHost = {0, dim * dim};

      // Copy data to GPU
      this->copyFromDouble(gpuInvHessian, cpuInvHessian);
      this->copyFromDouble(gpuDGrad, cpuDGrad);
      this->copyFromDouble(gpuXi, cpuXi);
      this->copyFromDouble(gpuGrad, cpuGrad);
      atomStarts.copyFromHost(atomStartsHost);
      hessianStarts.copyFromHost(hessianStartsHost);

      // Run CPU version
      std::vector<double> cpuHessDGrad(cpuDGrad.size());  // Match size with dGrad

      updateInverseHessianBFGSCPU(dim,
                                  cpuInvHessian.data(),
                                  cpuHessDGrad.data(),
                                  cpuDGrad.data(),
                                  cpuXi.data(),
                                  cpuGrad.data());

      PinnedHostVector<int> activeSystemIndicesHost;
      activeSystemIndicesHost.resize(1);
      activeSystemIndicesHost[0] = 0;
      AsyncDeviceVector<int> activeSystemIndices(1);
      activeSystemIndices.zero();

      // Run GPU version
      updateInverseHessianBFGSBatch(1,
                                    nullptr,
                                    hessianStarts.data(),
                                    atomStarts.data(),
                                    gpuInvHessian.data(),
                                    gpuDGrad.data(),
                                    gpuXi.data(),
                                    hessDgrad.data(),
                                    gpuGrad.data(),
                                    dataDim,
                                    /*largeMol=*/true,
                                    activeSystemIndices.data());

      // Copy results back from GPU
      std::vector<Scalar> resInvHessianHost(cpuInvHessian.size());
      std::vector<Scalar> resHessDGradHost(cpuHessDGrad.size());
      std::vector<Scalar> resDGradHost(cpuDGrad.size());
      std::vector<Scalar> resXiHost(cpuXi.size());

      gpuInvHessian.copyToHost(resInvHessianHost);
      hessDgrad.copyToHost(resHessDGradHost);
      gpuDGrad.copyToHost(resDGradHost);
      gpuXi.copyToHost(resXiHost);
      cudaCheckError(cudaDeviceSynchronize());

      EXPECT_THAT(resInvHessianHost, ::testing::Pointwise(::testing::DoubleNear(this->tolerance()), cpuInvHessian));
      EXPECT_THAT(resHessDGradHost, ::testing::Pointwise(::testing::DoubleNear(this->tolerance()), cpuHessDGrad));
      EXPECT_THAT(resDGradHost, ::testing::Pointwise(::testing::DoubleNear(this->tolerance()), cpuDGrad));
      EXPECT_THAT(resXiHost, ::testing::Pointwise(::testing::DoubleNear(this->tolerance()), cpuXi));
    }
  }
}

TYPED_TEST(BFGSHessianTest, MultiSystem) {
  using Scalar = TypeParam;
  for (const int dataDim : {3, 4}) {
    for (const bool identityHessian : {true, false}) {
      SCOPED_TRACE(::testing::Message() << "dataDim=" << dataDim << ", identityHessian=" << identityHessian);

      const std::vector<int> nAtoms = {3, 2, 33, 14};

      std::vector<double> accumulatedInvHessian, accumulatedDGrad, accumulatedXi, accumulatedGrad;
      std::vector<double> wantInvHessian, wantDGrad, wantXi, wantHDGrad;
      std::vector<int>    accumAtomStarts = {0}, accumHessianStarts = {0};

      for (int sysIdx = 0; sysIdx < nAtoms.size(); ++sysIdx) {
        const int           natom   = nAtoms[sysIdx];
        const int           fullDim = natom * dataDim;
        std::vector<double> cpuInvHessian, cpuDGrad, cpuXi, cpuGrad;
        this->generateRandomSystem(natom * dataDim, cpuInvHessian, cpuDGrad, cpuXi, cpuGrad, identityHessian);

        // Append to accumulated vectors
        accumulatedInvHessian.insert(accumulatedInvHessian.end(), cpuInvHessian.begin(), cpuInvHessian.end());
        accumulatedDGrad.insert(accumulatedDGrad.end(), cpuDGrad.begin(), cpuDGrad.end());
        accumulatedXi.insert(accumulatedXi.end(), cpuXi.begin(), cpuXi.end());
        accumulatedGrad.insert(accumulatedGrad.end(), cpuGrad.begin(), cpuGrad.end());

        // Update atom and hessian starts
        accumAtomStarts.push_back(accumAtomStarts.back() + natom);
        accumHessianStarts.push_back(accumHessianStarts.back() + fullDim * fullDim);

        // Compute CPU result
        std::vector<double> cpuHessDGrad(cpuDGrad.size(), 0.0);  // Match size with dGrad

        if (sysIdx != 2) {  // Skip system 2 to test inactive system setup.
          // Only compute if not skipping
          updateInverseHessianBFGSCPU(fullDim,
                                      cpuInvHessian.data(),
                                      cpuHessDGrad.data(),
                                      cpuDGrad.data(),
                                      cpuXi.data(),
                                      cpuGrad.data());
        }

        wantInvHessian.insert(wantInvHessian.end(), cpuInvHessian.begin(), cpuInvHessian.end());
        wantDGrad.insert(wantDGrad.end(), cpuDGrad.begin(), cpuDGrad.end());
        wantXi.insert(wantXi.end(), cpuXi.begin(), cpuXi.end());
        wantHDGrad.insert(wantHDGrad.end(), cpuHessDGrad.begin(), cpuHessDGrad.end());
      }

      // Create GPU data
      AsyncDeviceVector<Scalar> gpuInvHessian(accumulatedInvHessian.size());
      AsyncDeviceVector<Scalar> gpuDGrad(accumulatedDGrad.size());
      AsyncDeviceVector<Scalar> gpuXi(accumulatedXi.size());
      AsyncDeviceVector<Scalar> gpuGrad(accumulatedGrad.size());
      AsyncDeviceVector<Scalar> hessDgrad(accumulatedDGrad.size());
      AsyncDeviceVector<int>    atomStarts(accumAtomStarts.size());  // One extra element for the end
      AsyncDeviceVector<int>    hessianStarts(accumHessianStarts.size());

      hessDgrad.zero();

      // Copy data to GPU
      this->copyFromDouble(gpuInvHessian, accumulatedInvHessian);
      this->copyFromDouble(gpuDGrad, accumulatedDGrad);
      this->copyFromDouble(gpuXi, accumulatedXi);
      this->copyFromDouble(gpuGrad, accumulatedGrad);
      atomStarts.copyFromHost(accumAtomStarts);
      hessianStarts.copyFromHost(accumHessianStarts);

      PinnedHostVector<int> activeSystemIndicesHost;
      activeSystemIndicesHost.resize(3);
      activeSystemIndicesHost[0] = 0;  // System 0
      activeSystemIndicesHost[1] = 1;  // System 1
      activeSystemIndicesHost[2] = 3;  // System 3 (skipping system 2)
      AsyncDeviceVector<int> activeSystemIndices(activeSystemIndicesHost.size());
      activeSystemIndices.copyFromHost(activeSystemIndicesHost.begin(), activeSystemIndicesHost.size());

      // Run GPU version
      updateInverseHessianBFGSBatch(activeSystemIndices.size(),
                                    nullptr,
                                    hessianStarts.data(),
                                    atomStarts.data(),
                                    gpuInvHessian.data(),
                                    gpuDGrad.data(),
                                    gpuXi.data(),
                                    hessDgrad.data(),
                                    gpuGrad.data(),
                                    dataDim,
                                    /*largeMol=*/false,
                                    activeSystemIndices.data());
      // Copy results back from GPU
      std::vector<Scalar> resInvHessianHost(gpuInvHessian.size());
      std::vector<Scalar> resDGradHost(gpuDGrad.size());
      std::vector<Scalar> resXiHost(gpuXi.size());
      std::vector<Scalar> resHessDGradHost(gpuDGrad.size());

      gpuInvHessian.copyToHost(resInvHessianHost);
      gpuDGrad.copyToHost(resDGradHost);
      gpuXi.copyToHost(resXiHost);
      hessDgrad.copyToHost(resHessDGradHost);
      cudaCheckError(cudaDeviceSynchronize());

      EXPECT_THAT(resInvHessianHost, ::testing::Pointwise(::testing::DoubleNear(this->tolerance()), wantInvHessian));
      EXPECT_THAT(resDGradHost, ::testing::Pointwise(::testing::DoubleNear(this->tolerance()), wantDGrad));
      EXPECT_THAT(resXiHost, ::testing::Pointwise(::testing::DoubleNear(this->tolerance()), wantXi));
      EXPECT_THAT(resHessDGradHost, ::testing::Pointwise(::testing::DoubleNear(this->tolerance()), wantHDGrad));
    }
  }
}

TYPED_TEST(BFGSHessianTest, MultiSystemLarge) {
  using Scalar = TypeParam;
  for (const int dataDim : {3, 4}) {
    for (const bool identityHessian : {true, false}) {
      SCOPED_TRACE(::testing::Message() << "dataDim=" << dataDim << ", identityHessian=" << identityHessian);

      const std::vector<int> nAtoms = {3, 2, 300, 14};

      std::vector<double> accumulatedInvHessian, accumulatedDGrad, accumulatedXi, accumulatedGrad;
      std::vector<double> wantInvHessian, wantDGrad, wantXi, wantHDGrad;
      std::vector<int>    accumAtomStarts = {0}, accumHessianStarts = {0};

      for (int sysIdx = 0; sysIdx < nAtoms.size(); ++sysIdx) {
        const int           natom   = nAtoms[sysIdx];
        const int           fullDim = natom * dataDim;
        std::vector<double> cpuInvHessian, cpuDGrad, cpuXi, cpuGrad;
        this->generateRandomSystem(natom * dataDim, cpuInvHessian, cpuDGrad, cpuXi, cpuGrad, identityHessian);

        // Append to accumulated vectors
        accumulatedInvHessian.insert(accumulatedInvHessian.end(), cpuInvHessian.begin(), cpuInvHessian.end());
        accumulatedDGrad.insert(accumulatedDGrad.end(), cpuDGrad.begin(), cpuDGrad.end());
        accumulatedXi.insert(accumulatedXi.end(), cpuXi.begin(), cpuXi.end());
        accumulatedGrad.insert(accumulatedGrad.end(), cpuGrad.begin(), cpuGrad.end());

        // Update atom and hessian starts
        accumAtomStarts.push_back(accumAtomStarts.back() + natom);
        accumHessianStarts.push_back(accumHessianStarts.back() + fullDim * fullDim);

        // Compute CPU result
        std::vector<double> cpuHessDGrad(cpuDGrad.size(), 0.0);  // Match size with dGrad

        if (sysIdx != 2) {  // Skip system 2 to test inactive system setup.
          // Only compute if not skipping
          updateInverseHessianBFGSCPU(fullDim,
                                      cpuInvHessian.data(),
                                      cpuHessDGrad.data(),
                                      cpuDGrad.data(),
                                      cpuXi.data(),
                                      cpuGrad.data());
        }

        wantInvHessian.insert(wantInvHessian.end(), cpuInvHessian.begin(), cpuInvHessian.end());
        wantDGrad.insert(wantDGrad.end(), cpuDGrad.begin(), cpuDGrad.end());
        wantXi.insert(wantXi.end(), cpuXi.begin(), cpuXi.end());
        wantHDGrad.insert(wantHDGrad.end(), cpuHessDGrad.begin(), cpuHessDGrad.end());
      }

      // Create GPU data
      AsyncDeviceVector<Scalar> gpuInvHessian(accumulatedInvHessian.size());
      AsyncDeviceVector<Scalar> gpuDGrad(accumulatedDGrad.size());
      AsyncDeviceVector<Scalar> gpuXi(accumulatedXi.size());
      AsyncDeviceVector<Scalar> gpuGrad(accumulatedGrad.size());
      AsyncDeviceVector<Scalar> hessDgrad(accumulatedDGrad.size());
      AsyncDeviceVector<int>    atomStarts(accumAtomStarts.size());  // One extra element for the end
      AsyncDeviceVector<int>    hessianStarts(accumHessianStarts.size());

      hessDgrad.zero();

      // Copy data to GPU
      this->copyFromDouble(gpuInvHessian, accumulatedInvHessian);
      this->copyFromDouble(gpuDGrad, accumulatedDGrad);
      this->copyFromDouble(gpuXi, accumulatedXi);
      this->copyFromDouble(gpuGrad, accumulatedGrad);
      atomStarts.copyFromHost(accumAtomStarts);
      hessianStarts.copyFromHost(accumHessianStarts);

      PinnedHostVector<int> activeSystemIndicesHost;
      activeSystemIndicesHost.resize(3);
      activeSystemIndicesHost[0] = 0;  // System 0
      activeSystemIndicesHost[1] = 1;  // System 1
      activeSystemIndicesHost[2] = 3;  // System 3 (skipping system 2)
      AsyncDeviceVector<int> activeSystemIndices(activeSystemIndicesHost.size());
      activeSystemIndices.copyFromHost(activeSystemIndicesHost.begin(), activeSystemIndicesHost.size());

      // Run GPU version
      updateInverseHessianBFGSBatch(activeSystemIndices.size(),
                                    nullptr,
                                    hessianStarts.data(),
                                    atomStarts.data(),
                                    gpuInvHessian.data(),
                                    gpuDGrad.data(),
                                    gpuXi.data(),
                                    hessDgrad.data(),
                                    gpuGrad.data(),
                                    dataDim,
                                    /*largeMol=*/false,
                                    activeSystemIndices.data());
      // Copy results back from GPU
      std::vector<Scalar> resInvHessianHost(gpuInvHessian.size());
      std::vector<Scalar> resDGradHost(gpuDGrad.size());
      std::vector<Scalar> resXiHost(gpuXi.size());
      std::vector<Scalar> resHessDGradHost(gpuDGrad.size());

      gpuInvHessian.copyToHost(resInvHessianHost);
      gpuDGrad.copyToHost(resDGradHost);
      gpuXi.copyToHost(resXiHost);
      hessDgrad.copyToHost(resHessDGradHost);
      cudaCheckError(cudaDeviceSynchronize());

      EXPECT_THAT(resInvHessianHost, ::testing::Pointwise(::testing::DoubleNear(this->tolerance()), wantInvHessian));
      EXPECT_THAT(resDGradHost, ::testing::Pointwise(::testing::DoubleNear(this->tolerance()), wantDGrad));
      EXPECT_THAT(resXiHost, ::testing::Pointwise(::testing::DoubleNear(this->tolerance()), wantXi));
      EXPECT_THAT(resHessDGradHost, ::testing::Pointwise(::testing::DoubleNear(this->tolerance()), wantHDGrad));
    }
  }
}

TYPED_TEST(BFGSHessianTest, SkipInvHessianUpdateDueToIncorrectSigns) {
  using Scalar = TypeParam;
  for (const int dataDim : {3, 4}) {
    for (const bool identityHessian : {true, false}) {
      SCOPED_TRACE(::testing::Message() << "dataDim=" << dataDim << ", identityHessian=" << identityHessian);

      const std::vector<int> nAtoms = {3, 2, 33, 14};
      std::vector<double>    accumulatedInvHessian, accumulatedDGrad, accumulatedXi, accumulatedGrad;
      std::vector<double>    wantDGrad, wantXi;
      std::vector<int>       accumAtomStarts = {0}, accumHessianStarts = {0};

      for (const int& natom : nAtoms) {
        const int           fullDim = natom * dataDim;
        std::vector<double> cpuInvHessian, cpuDGrad, cpuXi, cpuGrad;
        // NOTE - here we introduce the incorrect signage.
        this->generateRandomSystem(fullDim,
                                   cpuInvHessian,
                                   cpuDGrad,
                                   cpuXi,
                                   cpuGrad,
                                   identityHessian,
                                   /*correctDGradXiSigns=*/false);

        // Append to accumulated vectors
        accumulatedInvHessian.insert(accumulatedInvHessian.end(), cpuInvHessian.begin(), cpuInvHessian.end());
        accumulatedDGrad.insert(accumulatedDGrad.end(), cpuDGrad.begin(), cpuDGrad.end());
        accumulatedXi.insert(accumulatedXi.end(), cpuXi.begin(), cpuXi.end());
        accumulatedGrad.insert(accumulatedGrad.end(), cpuGrad.begin(), cpuGrad.end());

        // Update atom and hessian starts
        accumAtomStarts.push_back(accumAtomStarts.back() + natom);
        accumHessianStarts.push_back(accumHessianStarts.back() + fullDim * fullDim);

        // Compute CPU result
        std::vector<double> cpuHessDGrad(cpuDGrad.size());  // Match size with dGrad
        updateInverseHessianBFGSCPU(fullDim,
                                    cpuInvHessian.data(),
                                    cpuHessDGrad.data(),
                                    cpuDGrad.data(),
                                    cpuXi.data(),
                                    cpuGrad.data());
        wantDGrad.insert(wantDGrad.end(), cpuDGrad.begin(), cpuDGrad.end());
        wantXi.insert(wantXi.end(), cpuXi.begin(), cpuXi.end());
      }

      // Create GPU data
      AsyncDeviceVector<Scalar> gpuInvHessian(accumulatedInvHessian.size());
      AsyncDeviceVector<Scalar> gpuDGrad(accumulatedDGrad.size());
      AsyncDeviceVector<Scalar> gpuXi(accumulatedXi.size());
      AsyncDeviceVector<Scalar> gpuGrad(accumulatedGrad.size());
      AsyncDeviceVector<int>    atomStarts(accumAtomStarts.size());  // One extra element for the end
      AsyncDeviceVector<int>    hessianStarts(accumHessianStarts.size());
      AsyncDeviceVector<Scalar> hessDgrad(accumulatedDGrad.size());

      hessDgrad.zero();

      // Copy data to GPU
      this->copyFromDouble(gpuInvHessian, accumulatedInvHessian);
      this->copyFromDouble(gpuDGrad, accumulatedDGrad);
      this->copyFromDouble(gpuXi, accumulatedXi);
      this->copyFromDouble(gpuGrad, accumulatedGrad);
      atomStarts.copyFromHost(accumAtomStarts);
      hessianStarts.copyFromHost(accumHessianStarts);

      AsyncDeviceVector<int> activeSystemIndices(nAtoms.size());
      PinnedHostVector<int>  activeSystemIndicesHost(nAtoms.size());
      activeSystemIndicesHost.resize(nAtoms.size());
      std::iota(activeSystemIndicesHost.begin(), activeSystemIndicesHost.end(), 0);

      std::iota(activeSystemIndicesHost.begin(), activeSystemIndicesHost.end(), 0);
      activeSystemIndices.copyFromHost(activeSystemIndicesHost.begin(), activeSystemIndicesHost.size());

      // Run GPU version
      updateInverseHessianBFGSBatch(activeSystemIndices.size(),
                                    nullptr,
                                    hessianStarts.data(),
                                    atomStarts.data(),
                                    gpuInvHessian.data(),
                                    gpuDGrad.data(),
                                    gpuXi.data(),
                                    hessDgrad.data(),
                                    gpuGrad.data(),
                                    dataDim,
                                    /*largeMol=*/false,
                                    activeSystemIndices.data());

      // Copy results back from GPU
      std::vector<Scalar> resInvHessianHost(gpuInvHessian.size());
      std::vector<Scalar> resDGradHost(gpuDGrad.size());
      std::vector<Scalar> resXiHost(gpuXi.size());

      gpuInvHessian.copyToHost(resInvHessianHost);
      gpuDGrad.copyToHost(resDGradHost);
      gpuXi.copyToHost(resXiHost);
      cudaCheckError(cudaDeviceSynchronize());

      // NOTE - inverse should not have been modified so we check against the original inputs. The others are modified.
      EXPECT_THAT(resInvHessianHost,
                  ::testing::Pointwise(::testing::DoubleNear(this->tolerance()), accumulatedInvHessian));
      EXPECT_THAT(resDGradHost, ::testing::Pointwise(::testing::DoubleNear(this->tolerance()), wantDGrad));
      EXPECT_THAT(resXiHost, ::testing::Pointwise(::testing::DoubleNear(this->tolerance()), wantXi));
    }
  }
}

}  // namespace nvMolKit
