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

#include <DistGeom/BoundsMatrix.h>
#include <DistGeom/DistGeomUtils.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/CrystalFF/TorsionPreferences.h>
#include <Numerics/SymmMatrix.h>
#include <RDGeneral/utils.h>

#include <random>
#include <vector>

#include "rdkit_extensions/bounds_matrix.h"
#include "src/forcefields/coord_gen.h"
#include "src/symmetric_eigensolver.h"
#include "src/utils/device_vector.h"

namespace nvMolKit {

namespace {

__global__ void combineActiveAndValidKernel(const int      numSystems,
                                            const uint8_t* active,
                                            const uint8_t* validDistanceMatrices,
                                            uint8_t*       eigensolverActive) {
  const int systemIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (systemIdx < numSystems) {
    eigensolverActive[systemIdx] = validDistanceMatrices[systemIdx] && (active == nullptr || active[systemIdx]);
  }
}

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
                                            const int      coordinateDim,
                                            double*        positions,
                                            const double*  eigenvalues,
                                            const double*  eigenvectors,
                                            const int*     atomStarts,
                                            const uint8_t* active,
                                            const double*  randomCoordinates,
                                            const int*     coordinateDimensions) {
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
  const int eigenvalueOffset  = batchIdx * coordinateDim;
  const int eigenvectorOffset = batchIdx * matrixDim * matrixDim;

  int numNegativeEigenvalues = 0;
  for (int j = 0; j < coordinateDimensions[batchIdx]; ++j) {
    numNegativeEigenvalues += eigenvalues[eigenvalueOffset + j] < 0.0;
  }
  int negativeEigenvalueIdx = 0;

  for (int j = 0; j < coordinateDim; j++) {
    if (j >= coordinateDimensions[batchIdx]) {
      positions[coordinateDim * (startAtomIdx + idxWithinBatch) + j] = 0.0;
      continue;
    }
    const int    eigVecIdx = eigenvectorOffset + j * matrixDim + idxWithinBatch;  // eigvec(j, i)
    const double eigval    = eigenvalues[eigenvalueOffset + j];
    if (eigval < 0.0) {
      const int randomIdx =
        coordinateDim * startAtomIdx + idxWithinBatch * numNegativeEigenvalues + negativeEigenvalueIdx;
      positions[coordinateDim * (startAtomIdx + idxWithinBatch) + j] = 1.0 - 2.0 * randomCoordinates[randomIdx];
      ++negativeEigenvalueIdx;
    } else {
      positions[coordinateDim * (startAtomIdx + idxWithinBatch) + j] = eigval * eigenvectors[eigVecIdx];
    }
  }
}

__global__ void finalizeEigenvaluesKernel(const int          numSystems,
                                          const int          numEigs,
                                          double*            vals,
                                          uint8_t*           passFail,
                                          const uint8_t*     validDistanceMatrices,
                                          const uint8_t*     eigensolverConverged,
                                          const int*         matrixDimensions,
                                          const int*         coordinateDimensions,
                                          const uint8_t*     active,
                                          const bool         randNegEig,
                                          const unsigned int numZeroFail) {
  const int        systemIdx  = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr double EIGVAL_TOL = 0.001;
  if (systemIdx < numSystems) {
    if (active != nullptr && active[systemIdx] == 0) {
      passFail[systemIdx] = 0;
      return;
    }
    const int matrixDim     = matrixDimensions[systemIdx];
    const int systemNumEigs = min(numEigs, min(matrixDim, coordinateDimensions[systemIdx]));
    bool passed = validDistanceMatrices[systemIdx] != 0 && (eigensolverConverged[systemIdx] != 0 || matrixDim <= 3);
    unsigned int zeroEigs = 0;
    for (int eigIdx = 0; eigIdx < systemNumEigs; ++eigIdx) {
      double&      val         = vals[systemIdx * numEigs + eigIdx];
      const double existingVal = val;
      if (existingVal > EIGVAL_TOL) {
        val = sqrt(existingVal);
      } else if (fabs(existingVal) < EIGVAL_TOL) {
        val = 0.0;
        ++zeroEigs;
      } else if (!randNegEig) {
        passed = false;
      }
    }
    if (matrixDim > 3 && zeroEigs >= numZeroFail) {
      passed = false;
    }
    passFail[systemIdx] = passed;
  }
}

}  // namespace

namespace detail {

class InitialCoordinateGenerator::Impl {
  // TODO: Support contraction of finished molecules in batch.
 public:
  explicit Impl(cudaStream_t stream) : stream_(stream) {
    packedDistancesMatricesDevice_.setStream(stream);
    eigenvaluesDevice_.setStream(stream);
    eigenvectorsDevice_.setStream(stream);
    passFail_.setStream(stream);
    atomCountsDevice_.setStream(stream);
    validDistanceMatricesDevice_.setStream(stream);
    eigenSeedsDevice_.setStream(stream);
    coordinateDimensionsDevice_.setStream(stream);
    randomCoordinatesDevice_.setStream(stream);
    eigensolverActive_.setStream(stream);
  }

  void computeBoundsMatrices(const std::vector<const RDKit::ROMol*>&                mols,
                             const RDKit::DGeomHelpers::EmbedParameters&            params,
                             std::vector<ForceFields::CrystalFF::CrystalFFDetails>& etkdgDetails,
                             const std::vector<int>&                                attemptIds,
                             const std::vector<int>&                                coordinateDimensions) {
    randNegEig_     = params.randNegEig;
    numZeroFail_    = params.numZeroFail;
    boundsMatrices_ = getBoundsMatrices(mols, params, etkdgDetails);
    initializeBatch(params, attemptIds, coordinateDimensions);
  }

  void setBoundsMatrices(const std::vector<::DistGeom::BoundsMatPtr>& boundsMatrices,
                         const RDKit::DGeomHelpers::EmbedParameters&  params,
                         const std::vector<int>&                      attemptIds,
                         const std::vector<int>&                      coordinateDimensions) {
    randNegEig_     = params.randNegEig;
    numZeroFail_    = params.numZeroFail;
    boundsMatrices_ = boundsMatrices;
    initializeBatch(params, attemptIds, coordinateDimensions);
  }

 private:
  void initializeBatch(const RDKit::DGeomHelpers::EmbedParameters& params,
                       const std::vector<int>&                     attemptIds,
                       const std::vector<int>&                     coordinateDimensions) {
    atomCountsHost_.clear();
    atomCountsHost_.reserve(boundsMatrices_.size());
    for (const auto& boundsMatrix : boundsMatrices_) {
      if (boundsMatrix == nullptr) {
        throw std::invalid_argument("Bounds matrices must not contain null entries");
      }
      atomCountsHost_.push_back(static_cast<int>(boundsMatrix->numRows()));
    }
    atomCountsDevice_.setFromVector(atomCountsHost_);

    if (!attemptIds.empty() && attemptIds.size() != boundsMatrices_.size()) {
      throw std::invalid_argument("Attempt IDs must match the molecule batch size");
    }
    attemptIds_ = attemptIds;
    if (!coordinateDimensions.empty() && coordinateDimensions.size() != boundsMatrices_.size()) {
      throw std::invalid_argument("Coordinate dimensions must match the molecule batch size");
    }
    coordinateDimensionsHost_ = coordinateDimensions;
    baseSeed_ = params.randomSeed >= 0 ? static_cast<unsigned int>(params.randomSeed) : std::random_device{}();
  }

 public:
  void computeInitialCoordinates(double*        deviceCoords,
                                 const int*     deviceAtomStarts,
                                 int            coordinateDim,
                                 const uint8_t* active) {
    std::vector<RDNumeric::SymmMatrix<double>> distMatrices;
    const int                                  batchSize = boundsMatrices_.size();
    if (batchSize == 0) {
      throw std::runtime_error("Bounds matrices not computed");
    }
    if (coordinateDim <= 0 || coordinateDim > 4) {
      throw std::invalid_argument("Coordinate dimension must be between 1 and 4");
    }
    validDistanceMatricesHost_.assign(batchSize, 0);
    eigenSeedsHost_.assign(batchSize, 0);
    std::vector<int> coordinateDimensions = coordinateDimensionsHost_;
    if (coordinateDimensions.empty()) {
      coordinateDimensions.assign(batchSize, coordinateDim);
    }
    coordinateDimensionsDevice_.setFromVector(coordinateDimensions);
    randomCoordinatesHost_.clear();
    auto sampleDistanceMatrix = [&](int i, RDKit::double_source_type& rng) {
      auto tempMat = RDNumeric::SymmMatrix<double>(boundsMatrices_[i]->numRows(), 0.0);
      ::DistGeom::pickRandomDistMat(*boundsMatrices_[i], tempMat, rng);
      auto& normalized = distMatrices.emplace_back(tempMat.numRows(), 0.0);
      validDistanceMatricesHost_[i] =
        RDKit::DGeomHelpers::initialCoordsNormDistances(tempMat, normalized, eigenSeedsHost_[i]);
      for (int randomIdx = 0; randomIdx < atomCountsHost_[i] * coordinateDim; ++randomIdx) {
        randomCoordinatesHost_.push_back(rng());
      }
    };

    RDKit::uniform_double distribution(0.0, 1.0);
    if (attemptIds_.empty()) {
      RDKit::rng_type           generator(baseSeed_);
      RDKit::double_source_type rng(generator, distribution);
      for (int i = 0; i < batchSize; ++i) {
        sampleDistanceMatrix(i, rng);
      }
    } else {
      for (int i = 0; i < batchSize; ++i) {
        RDKit::rng_type           generator(baseSeed_ + static_cast<unsigned int>(attemptIds_[i]));
        RDKit::double_source_type rng(generator, distribution);
        sampleDistanceMatrix(i, rng);
      }
    }
    // TODO - pass as reference parameter, do resize and set to 0 for subsequent runs.
    packedDistanceMatricesHost_ = packedNMatrices(distMatrices, maxDimension_);
    if (maxDimension_ > 256) {
      throw std::invalid_argument("Eigenvalue coordinate generation supports at most 256 atoms per molecule");
    }
    packedDistancesMatricesDevice_.resize(packedDistanceMatricesHost_.size());
    packedDistancesMatricesDevice_.copyFromHost(packedDistanceMatricesHost_);

    validDistanceMatricesDevice_.setFromVector(validDistanceMatricesHost_);
    eigenSeedsDevice_.setFromVector(eigenSeedsHost_);
    randomCoordinatesDevice_.setFromVector(randomCoordinatesHost_);
    eigenvaluesDevice_.resize(static_cast<size_t>(coordinateDim) * batchSize);
    eigenvaluesDevice_.zero();
    eigenvectorsDevice_.resize(maxDimension_ * maxDimension_ * batchSize);
    eigenvectorsDevice_.zero();
    passFail_.resize(batchSize);
    eigensolverActive_.resize(batchSize);
    constexpr int maskBlockSize = 128;
    combineActiveAndValidKernel<<<(batchSize + maskBlockSize - 1) / maskBlockSize, maskBlockSize, 0, stream_>>>(
      batchSize,
      active,
      validDistanceMatricesDevice_.data(),
      eigensolverActive_.data());
    cudaCheckError(cudaGetLastError());
    BatchedEigenSolverOptions solverOptions;
    solverOptions.active           = eigensolverActive_.data();
    solverOptions.matrixDimensions = atomCountsDevice_.data();
    solverOptions.randomSeeds      = eigenSeedsDevice_.data();
    solverOptions.eigenDimensions  = coordinateDimensionsDevice_.data();
    solverOptions.stream           = stream_;
    solver_.solve(coordinateDim,
                  maxDimension_,
                  batchSize,
                  packedDistancesMatricesDevice_.data(),
                  eigenvaluesDevice_.data(),
                  eigenvectorsDevice_.data(),
                  solverOptions);

    const int blockSize       = 128;
    const int numBlocksFinish = (batchSize + blockSize - 1) / blockSize;
    finalizeEigenvaluesKernel<<<numBlocksFinish, blockSize, 0, stream_>>>(batchSize,
                                                                          coordinateDim,
                                                                          eigenvaluesDevice_.data(),
                                                                          passFail_.data(),
                                                                          validDistanceMatricesDevice_.data(),
                                                                          solver_.converged(),
                                                                          atomCountsDevice_.data(),
                                                                          coordinateDimensionsDevice_.data(),
                                                                          active,
                                                                          randNegEig_,
                                                                          numZeroFail_);
    cudaCheckError(cudaGetLastError());
    // Project eigenvalues
    const int numThreads = maxDimension_ * batchSize;
    const int numBlocks  = (numThreads + blockSize - 1) / blockSize;
    projectDistancesToPositions<<<numBlocks, blockSize, 0, stream_>>>(maxDimension_,
                                                                      batchSize,
                                                                      coordinateDim,
                                                                      deviceCoords,
                                                                      eigenvaluesDevice_.data(),
                                                                      eigenvectorsDevice_.data(),
                                                                      deviceAtomStarts,
                                                                      active,
                                                                      randomCoordinatesDevice_.data(),
                                                                      coordinateDimensionsDevice_.data());
    cudaCheckError(cudaGetLastError());
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
  std::vector<int>                      atomCountsHost_;
  AsyncDeviceVector<int>                atomCountsDevice_;
  std::vector<uint8_t>                  validDistanceMatricesHost_;
  AsyncDeviceVector<uint8_t>            validDistanceMatricesDevice_;
  std::vector<int>                      eigenSeedsHost_;
  AsyncDeviceVector<int>                eigenSeedsDevice_;
  std::vector<int>                      coordinateDimensionsHost_;
  AsyncDeviceVector<int>                coordinateDimensionsDevice_;
  std::vector<double>                   randomCoordinatesHost_;
  AsyncDeviceVector<double>             randomCoordinatesDevice_;
  AsyncDeviceVector<uint8_t>            eigensolverActive_;
  std::vector<int>                      attemptIds_;
  unsigned int                          baseSeed_     = 0;
  int                                   maxDimension_ = 0;
  bool                                  randNegEig_   = false;
  unsigned int                          numZeroFail_  = 1;
  cudaStream_t                          stream_       = nullptr;
};

InitialCoordinateGenerator::InitialCoordinateGenerator(cudaStream_t stream) : impl_(std::make_unique<Impl>(stream)) {}
InitialCoordinateGenerator::~InitialCoordinateGenerator() = default;

void InitialCoordinateGenerator::computeBoundsMatrices(
  const std::vector<const RDKit::ROMol*>&                mols,
  const RDKit::DGeomHelpers::EmbedParameters&            params,
  std::vector<ForceFields::CrystalFF::CrystalFFDetails>& etkdgDetails,
  const std::vector<int>&                                attemptIds,
  const std::vector<int>&                                coordinateDimensions) {
  return impl_->computeBoundsMatrices(mols, params, etkdgDetails, attemptIds, coordinateDimensions);
}

void InitialCoordinateGenerator::setBoundsMatrices(const std::vector<::DistGeom::BoundsMatPtr>& boundsMatrices,
                                                   const RDKit::DGeomHelpers::EmbedParameters&  params,
                                                   const std::vector<int>&                      attemptIds,
                                                   const std::vector<int>&                      coordinateDimensions) {
  impl_->setBoundsMatrices(boundsMatrices, params, attemptIds, coordinateDimensions);
}

void InitialCoordinateGenerator::computeInitialCoordinates(double*        deviceCoords,
                                                           const int*     deviceAtomStarts,
                                                           int            coordinateDim,
                                                           const uint8_t* active) {
  impl_->computeInitialCoordinates(deviceCoords, deviceAtomStarts, coordinateDim, active);
}

const uint8_t* InitialCoordinateGenerator::getPassFail() const {
  return impl_->getPassFail();
}

int InitialCoordinateGenerator::numSystemsPrepared() {
  return impl_->numSystemsPrepared();
}

}  // namespace detail

}  // namespace nvMolKit
