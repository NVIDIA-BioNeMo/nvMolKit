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

#ifndef NVMOLKIT_ETKDG_STAGE_TETRAHEDRAL_CHECKS_H
#define NVMOLKIT_ETKDG_STAGE_TETRAHEDRAL_CHECKS_H

#include <vector>

#include "src/etkdg_impl.h"

namespace nvMolKit {
namespace detail {

enum class ChiralCheckType {
  Chiral,
  Tetrahedral
};

//! Base class for all checks involving RDKit ChiralSet structures
class ETKDGChiralCheckBase : public ETKDGStage {
 protected:
  //! Sets the stream for all device vectors.
  void setStreams(cudaStream_t stream);

  //! Read in chiral/tetrahedral data from EmbedArgs to device vectors.
  void loadChiralDataset(const ETKDGContext& ctx, const std::vector<EmbedArgs>& eargs, ChiralCheckType checkType);

  AsyncDeviceVector<int>           idx0;
  AsyncDeviceVector<int>           idx1;
  AsyncDeviceVector<int>           idx2;
  AsyncDeviceVector<int>           idx3;
  AsyncDeviceVector<int>           idx4;
  AsyncDeviceVector<std::uint64_t> structureFlags;
  AsyncDeviceVector<int>           sysIdx;

  // Always part of the RDKit data structure, but only used by chiral check.
  AsyncDeviceVector<double> volumeLowerBound;
  AsyncDeviceVector<double> volumeUpperBound;
};

//! Implements RDKit Tetrahedral check stage on GPU.
//! This stage triggers on non-chiral double-ring tetrahedral centers.
class ETKDGTetrahedralCheckStage final : public ETKDGChiralCheckBase {
 public:
  ETKDGTetrahedralCheckStage(const ETKDGContext&           ctx,
                             const std::vector<EmbedArgs>& eargs,
                             int                           dim    = 4,
                             cudaStream_t                  stream = nullptr);

  void execute(ETKDGContext& ctx) override;

  std::string name() const override { return "Tetrahedral Checks"; }

 private:
  //! Data dimensionality
  int                     dim_         = 4;
  static constexpr double volCheckTol_ = 0.3;
  cudaStream_t            stream_      = nullptr;
};

class ETKDGFirstChiralCenterCheckStage final : public ETKDGChiralCheckBase {
 public:
  ETKDGFirstChiralCenterCheckStage(const ETKDGContext&           ctx,
                                   const std::vector<EmbedArgs>& eargs,
                                   int                           dim    = 4,
                                   cudaStream_t                  stream = nullptr);

  void execute(ETKDGContext& ctx) override;

  std::string name() const override { return "First Chirality Check"; }

 private:
  //! Data dimensionality
  int          dim_    = 4;
  cudaStream_t stream_ = nullptr;
};

//! Final check to ensure all chiral centers are in the volume defined by their neighbors.
//! Passes through to the ETKDGFirstChiralCenterCheckStage, which does the actual check, as it is identical.
//! This class just lets us have a separate stage identifier for error reporting.
class ETKDGFinalChiralCenterCheckStage final : public ETKDGStage {
 public:
  ETKDGFinalChiralCenterCheckStage(ETKDGFirstChiralCenterCheckStage& baseStage) : baseStage_(baseStage) {}

  void execute(ETKDGContext& ctx) override { baseStage_.execute(ctx); }

  std::string name() const override { return "Final Chirality Check"; }

 private:
  ETKDGFirstChiralCenterCheckStage& baseStage_;
};

//! Final check to ensure all chiral centers are in the volume defined by their neighbors.
//! This uses the same algorithm as the tetrahedral check, but applies to all chiral centers and does not
//! do the initial chiral volume test, just the center-in-volume test.
class ETKDGChiralCenterVolumeCheckStage final : public ETKDGChiralCheckBase {
 public:
  ETKDGChiralCenterVolumeCheckStage(const ETKDGContext&           ctx,
                                    const std::vector<EmbedArgs>& eargs,
                                    int                           dim    = 4,
                                    cudaStream_t                  stream = nullptr);

  void execute(ETKDGContext& ctx) override;

  std::string name() const override { return "Final Chiral Center in Volume Check"; }

 private:
  //! Data dimensionality
  int                     dim_         = 4;
  static constexpr double volCheckTol_ = 0.1;
  cudaStream_t            stream_      = nullptr;
};

//! Checks chiral atoms against initial distance matrix.
class ETKDGChiralDistMatrixCheckStage final : public ETKDGStage {
 public:
  ETKDGChiralDistMatrixCheckStage(const ETKDGContext&           ctx,
                                  const std::vector<EmbedArgs>& eargs,
                                  int                           dim    = 4,
                                  cudaStream_t                  stream = nullptr);

  void execute(ETKDGContext& ctx) override;

  std::string name() const override { return "Chirality Distance Matrix Check"; }

 private:
  void                   loadDataset(const ETKDGContext& ctx, const std::vector<EmbedArgs>& eargs);
  AsyncDeviceVector<int> idx0;
  AsyncDeviceVector<int> idx1;
  AsyncDeviceVector<int> sysIdx;

  AsyncDeviceVector<double> matLowerBound;
  AsyncDeviceVector<double> matUpperBound;

  int          dim_    = 4;
  cudaStream_t stream_ = nullptr;
};

//! Checks double bond cis/trans stereochemistry as part of final ETKDG checks.
class ETKDGDoubleBondStereoCheckStage final : public ETKDGStage {
 public:
  ETKDGDoubleBondStereoCheckStage(const ETKDGContext&           ctx,
                                  const std::vector<EmbedArgs>& eargs,
                                  int                           dim    = 4,
                                  cudaStream_t                  stream = nullptr);

  void execute(ETKDGContext& ctx) override;

  std::string name() const override { return "Double bond stereo check"; }

 private:
  void                   loadDataset(const ETKDGContext& ctx, const std::vector<EmbedArgs>& eargs);
  AsyncDeviceVector<int> idx0;
  AsyncDeviceVector<int> idx1;
  AsyncDeviceVector<int> idx2;
  AsyncDeviceVector<int> idx3;
  AsyncDeviceVector<int> sysIdx;
  AsyncDeviceVector<int> signs;

  int          dim_    = 4;
  cudaStream_t stream_ = nullptr;
};

//! Checks double bond geometry as part of final ETKDG checks.
//! Effective just checks within a large tolerance that the double bond is not planar with respect to any 3 elements.
class ETKDGDoubleBondGeometryCheckStage final : public ETKDGStage {
 public:
  ETKDGDoubleBondGeometryCheckStage(const ETKDGContext&           ctx,
                                    const std::vector<EmbedArgs>& eargs,
                                    int                           dim    = 4,
                                    cudaStream_t                  stream = nullptr);

  void execute(ETKDGContext& ctx) override;

  std::string name() const override { return "Double bond geometry check"; }

 private:
  void                   loadDataset(const ETKDGContext& ctx, const std::vector<EmbedArgs>& eargs);
  AsyncDeviceVector<int> idx0;
  AsyncDeviceVector<int> idx1;
  AsyncDeviceVector<int> idx2;
  AsyncDeviceVector<int> sysIdx;

  int          dim_    = 4;
  cudaStream_t stream_ = nullptr;
};

}  // namespace detail
}  // namespace nvMolKit

#endif  // NVMOLKIT_ETKDG_STAGE_TETRAHEDRAL_CHECKS_H
