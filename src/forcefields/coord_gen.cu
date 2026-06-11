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

#include <curand.h>
#include <curand_kernel.h>
#include <DistGeom/BoundsMatrix.h>
#include <DistGeom/DistGeomUtils.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/CrystalFF/TorsionPreferences.h>
#include <Numerics/SymmMatrix.h>

#include <vector>

#include "rdkit_extensions/bounds_matrix.h"
#include "src/forcefields/coord_gen.h"
#include "src/symmetric_eigensolver.h"
#include "src/utils/device_vector.h"

namespace nvMolKit {

namespace {

std::vector<double> packedNMatrices(const std::vector<RDNumeric::SymmMatrix<double>>& matrices, int& maxDimension) {
  // Compute maximum dimension of input matrices
  int N = 0;
  for (const auto& matrix : matrices) {
    N = std::max<int>(N, matrix.numRows());
  }
  maxDimension = N;

  std::vector<double> result(N * N * matrices.size(), 0.0);
  result.resize(N * N * matrices.size());
  for (size_t b = 0; b < matrices.size(); b++) {
    for (size_t i = 0; i < matrices[b].numRows(); i++) {
      for (size_t j = 0; j < matrices[b].numRows(); j++) {
        result[b * N * N + i * N + j] = matrices[b].getVal(i, j);
      }
    }
  }
  return result;
}

__global__ void projectDistancesToPositions(const int      matrixDim,
                                            const int      numBatches,
                                            double*        positions,
                                            const double*  eigenvalues,
                                            const double*  eigenvectors,
                                            const int*     atomStarts,
                                            curandState*   states,
                                            const uint8_t* active,
                                            const int      seed = 42) {
  // TODO support 4D
  constexpr int DIM = 3;

  const int idx            = blockIdx.x * blockDim.x + threadIdx.x;
  const int batchIdx       = idx / matrixDim;
  const int idxWithinBatch = idx % matrixDim;
  if (batchIdx >= numBatches) {
    return;
  }
  if (active != nullptr && active[batchIdx] == 0) {
    return;
  }

  const int startAtomIdx           = atomStarts[batchIdx];
  const int endAtomIdx             = atomStarts[batchIdx + 1];
  const int numAtomsInBatchElement = endAtomIdx - startAtomIdx;
  if (idxWithinBatch >= numAtomsInBatchElement) {
    return;
  }
  curand_init(seed, idxWithinBatch, 0, &states[idx]);

  const int eigenvalueOffset  = batchIdx * DIM;
  const int eigenvectorOffset = batchIdx * matrixDim * matrixDim;

  for (int j = 0; j < DIM; j++) {
    const int    eigVecIdx = eigenvectorOffset + j * matrixDim + idxWithinBatch;  // eigvec(j, i)
    const double eigval    = eigenvalues[eigenvalueOffset + j];
    if (eigval == 0.0) {
      // from negative eigenvalue, compute randomly. Note that if set to fail elsewhere, we'll be discarding this
      // result.
      positions[DIM * (startAtomIdx + idxWithinBatch) + j] = 1.0 - 2.0 * curand_uniform_double(&states[idx]);
    } else {
      positions[DIM * (startAtomIdx + idxWithinBatch) + j] =
        eigenvalues[eigenvalueOffset + j] * eigenvectors[eigVecIdx];
    }
  }
}

__global__ void squareRootOrZeroKernel(const int N,
                                       const int numEigs,
                                       double*   vals,
                                       uint8_t*  passFail,
                                       bool      randNegEig) {
  const int        idx        = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr double EIGVAL_TOL = 0.001;
  if (idx < N) {
    const double existingVal = vals[idx];
    if (existingVal > 0.0) {
      vals[idx] = sqrt(existingVal);
    } else if (fabs(existingVal) < EIGVAL_TOL) {
      // TODO: support numZeroEig
      // Compute passFail index
      const int batchIdx = idx / numEigs;
      passFail[batchIdx] = 0;
    } else {
      // negative number
      vals[idx] = 0.0;
      if (!randNegEig) {
        const int batchIdx = idx / numEigs;
        passFail[batchIdx] = 0;
      }
    }
  }
}

}  // namespace

namespace detail {

class InitialCoordinateGenerator::Impl {
  // TODO: Support contraction of finished molecules in batch.
 public:
  void computeBoundsMatrices(const std::vector<const RDKit::ROMol*>&                mols,
                             const RDKit::DGeomHelpers::EmbedParameters&            params,
                             std::vector<ForceFields::CrystalFF::CrystalFFDetails>& etkdgDetails) {
    randNegEig_     = params.randNegEig;
    boundsMatrices_ = getBoundsMatrices(mols, params, etkdgDetails);
  }

  void computeInitialCoordinates(double* deviceCoords, const int* deviceAtomStarts, const uint8_t* active) {
    // TODO: Return error cases.
    std::vector<RDNumeric::SymmMatrix<double>> distMatrices;
    const int                                  batchSize = boundsMatrices_.size();
    if (batchSize == 0) {
      throw std::runtime_error("Bounds matrices not computed");
    }
    for (int i = 0; i < batchSize; i++) {
      auto tempMat = RDNumeric::SymmMatrix<double>(boundsMatrices_[i]->numRows(), 0.0);
      ::DistGeom::pickRandomDistMat(*boundsMatrices_[i], tempMat);
      distMatrices.push_back(RDKit::DGeomHelpers::initialCoordsNormDistances(tempMat));
    }
    // TODO - pass as reference parameter, do resize and set to 0 for subsequent runs.
    packedDistanceMatricesHost_ = packedNMatrices(distMatrices, maxDimension_);
    packedDistancesMatricesDevice_.resize(packedDistanceMatricesHost_.size());
    packedDistancesMatricesDevice_.copyFromHost(packedDistanceMatricesHost_);

    // TODO 4D support.
    eigenvaluesDevice_.resize(maxDimension_ * batchSize * 3);
    eigenvaluesDevice_.zero();
    eigenvectorsDevice_.resize(maxDimension_ * maxDimension_ * batchSize);
    passFail_.resize(batchSize);
    solver_.solve(3,
                  maxDimension_,
                  batchSize,
                  packedDistancesMatricesDevice_.data(),
                  eigenvaluesDevice_.data(),
                  eigenvectorsDevice_.data(),
                  active);

    // Copy solver pass fail to passFail_
    cudaMemcpyAsync(passFail_.data(), solver_.converged(), batchSize, cudaMemcpyDeviceToDevice);

    const int numEigvals    = 3 * batchSize;
    const int blockSize     = 128;
    const int numBlocksSqrt = (numEigvals + blockSize - 1) / blockSize;
    squareRootOrZeroKernel<<<numBlocksSqrt, blockSize>>>(numEigvals,
                                                         3,
                                                         eigenvaluesDevice_.data(),
                                                         passFail_.data(),
                                                         randNegEig_);
    // Project eigenvalues
    const int                      numThreads = maxDimension_ * batchSize;
    const int                      numBlocks  = (numThreads + blockSize - 1) / blockSize;
    // TODO reuse, and have path that doesn't use when randnegEig set to off.
    AsyncDeviceVector<curandState> states(numThreads);

    projectDistancesToPositions<<<numBlocks, blockSize>>>(maxDimension_,
                                                          batchSize,
                                                          deviceCoords,
                                                          eigenvaluesDevice_.data(),
                                                          eigenvectorsDevice_.data(),
                                                          deviceAtomStarts,
                                                          states.data(),
                                                          active);
  }

  const uint8_t* getPassFail() const { return passFail_.data(); }

  int numSystemsPrepared() const { return boundsMatrices_.size(); }

 private:
  BatchedEigenSolver                    solver_;
  std::vector<::DistGeom::BoundsMatPtr> boundsMatrices_;
  std::vector<double>                   packedDistanceMatricesHost_;
  AsyncDeviceVector<double>             packedDistancesMatricesDevice_;
  AsyncDeviceVector<double>             eigenvaluesDevice_;
  AsyncDeviceVector<double>             eigenvectorsDevice_;
  AsyncDeviceVector<uint8_t>            passFail_;
  std::vector<double>                   initialDistMat_;
  std::vector<double>                   initialDistMatPacked_;
  int                                   maxDimension_;
  bool                                  randNegEig_ = false;
};

InitialCoordinateGenerator::InitialCoordinateGenerator() : impl_(std::make_unique<Impl>()) {}
InitialCoordinateGenerator::~InitialCoordinateGenerator() = default;

void InitialCoordinateGenerator::computeBoundsMatrices(
  const std::vector<const RDKit::ROMol*>&                mols,
  const RDKit::DGeomHelpers::EmbedParameters&            params,
  std::vector<ForceFields::CrystalFF::CrystalFFDetails>& etkdgDetails) {
  return impl_->computeBoundsMatrices(mols, params, etkdgDetails);
}

void InitialCoordinateGenerator::computeInitialCoordinates(double*        deviceCoords,
                                                           const int*     deviceAtomStarts,
                                                           const uint8_t* active) {
  impl_->computeInitialCoordinates(deviceCoords, deviceAtomStarts, active);
}

const uint8_t* InitialCoordinateGenerator::getPassFail() const {
  return impl_->getPassFail();
}

int InitialCoordinateGenerator::numSystemsPrepared() {
  return impl_->numSystemsPrepared();
}

}  // namespace detail

}  // namespace nvMolKit
