// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "src/tfd/tfd_gpu.h"

#include <cuda_runtime.h>

#include <stdexcept>

#include "src/tfd/tfd_kernels.h"
#include "src/utils/nvtx.h"

namespace nvMolKit {

// ========== TFDGpuResult ==========

std::vector<double> TFDGpuResult::extractMolecule(int molIdx) const {
  ScopedNvtxRange range("GPU: extractMolecule", NvtxColor::kRed);
  if (molIdx < 0 || molIdx >= static_cast<int>(conformerCounts.size())) {
    throw std::out_of_range("Invalid molecule index: " + std::to_string(molIdx));
  }

  int outStart  = tfdOutputStarts[molIdx];
  int outEnd    = tfdOutputStarts[molIdx + 1];
  int numValues = outEnd - outStart;

  if (numValues == 0) {
    return {};
  }

  // Copy from device to host
  std::vector<float> hostFloats(numValues);
  tfdValues.copyToHost(hostFloats.data(), numValues, 0, outStart);
  cudaStreamSynchronize(tfdValues.stream());

  // Convert to double
  std::vector<double> result(numValues);
  for (int i = 0; i < numValues; ++i) {
    result[i] = static_cast<double>(hostFloats[i]);
  }

  return result;
}

std::vector<std::vector<double>> TFDGpuResult::extractAll() const {
  ScopedNvtxRange range("GPU: extractAll (D2H)", NvtxColor::kRed);

  int                              numMolecules = static_cast<int>(conformerCounts.size());
  std::vector<std::vector<double>> results(numMolecules);

  if (tfdValues.size() == 0) {
    return results;
  }

  int totalValues = static_cast<int>(tfdValues.size());

  // Copy all data at once
  std::vector<float> allHostFloats(totalValues);
  tfdValues.copyToHost(allHostFloats.data(), totalValues);
  cudaStreamSynchronize(tfdValues.stream());

  // Convert float→double into per-molecule vectors
  const float* src = allHostFloats.data();
  for (int m = 0; m < numMolecules; ++m) {
    int numValues = tfdOutputStarts[m + 1] - tfdOutputStarts[m];
    results[m].resize(numValues);
    for (int i = 0; i < numValues; ++i) {
      results[m][i] = static_cast<double>(src[i]);
    }
    src += numValues;
  }

  return results;
}

// ========== TFDGpuGenerator ==========

TFDGpuGenerator::TFDGpuGenerator() : stream_() {
  device_.setStream(stream_.stream());
}

TFDGpuResult TFDGpuGenerator::GetTFDMatricesGpuBuffer(const std::vector<const RDKit::ROMol*>& mols,
                                                      const TFDComputeOptions&                options) {
  ScopedNvtxRange outerRange("GPU: GetTFDMatricesGpuBuffer (" + std::to_string(mols.size()) + " mols)",
                             NvtxColor::kBlue);

  TFDGpuResult result;

  if (mols.empty()) {
    return result;
  }

  // Build host system data (CPU preprocessing, parallelized across molecules)
  TFDSystemHost system = buildTFDSystem(mols, options);

  // Build result metadata from MolDescriptors
  result.tfdOutputStarts.resize(system.numMolecules() + 1);
  result.tfdOutputStarts[0] = 0;
  result.conformerCounts.resize(system.numMolecules());
  for (int i = 0; i < system.numMolecules(); ++i) {
    const auto& desc              = system.molDescriptors[i];
    result.conformerCounts[i]     = desc.numConformers;
    int numPairs                  = desc.numConformers * (desc.numConformers - 1) / 2;
    result.tfdOutputStarts[i + 1] = result.tfdOutputStarts[i] + numPairs;
  }

  // Handle edge case: no TFD outputs
  if (system.totalTFDOutputs() == 0) {
    return result;
  }

  cudaStream_t stream = stream_.stream();

  // Transfer to device (handles resize + copy + output buffer allocation)
  {
    ScopedNvtxRange range("GPU: transferToDevice (H2D)", NvtxColor::kYellow);
    transferToDevice(system, device_, stream);
  }

  // Launch dihedral kernel
  {
    ScopedNvtxRange range("GPU: launchDihedralKernel", NvtxColor::kOrange);
    launchDihedralKernel(system.totalDihedralWorkItems(),
                         device_.positions.data(),
                         device_.confPositionStarts.data(),
                         device_.torsionAtoms.data(),
                         device_.molDescriptors.data(),
                         device_.dihedralWorkStarts.data(),
                         system.numMolecules(),
                         device_.dihedralAngles.data(),
                         stream);
  }

  // Launch TFD matrix kernel (one block per molecule)
  {
    ScopedNvtxRange range("GPU: launchTFDMatrixKernel", NvtxColor::kOrange);
    launchTFDMatrixKernel(system.numMolecules(),
                          device_.dihedralAngles.data(),
                          device_.torsionWeights.data(),
                          device_.torsionMaxDevs.data(),
                          device_.quartetStarts.data(),
                          device_.torsionTypes.data(),
                          device_.molDescriptors.data(),
                          device_.tfdOutput.data(),
                          stream);
  }

  // Move output to result (transfer ownership of GPU memory)
  result.tfdValues = std::move(device_.tfdOutput);

  // Reallocate the output buffer for next use
  device_.tfdOutput = AsyncDeviceVector<float>();
  device_.tfdOutput.setStream(stream);

  return result;
}

std::vector<std::vector<double>> TFDGpuGenerator::GetTFDMatrices(const std::vector<const RDKit::ROMol*>& mols,
                                                                 const TFDComputeOptions&                options) {
  ScopedNvtxRange range("GPU: GetTFDMatrices (" + std::to_string(mols.size()) + " mols)", NvtxColor::kBlue);
  TFDGpuResult    gpuResult = GetTFDMatricesGpuBuffer(mols, options);
  return gpuResult.extractAll();
}

std::vector<double> TFDGpuGenerator::GetTFDMatrix(const RDKit::ROMol& mol, const TFDComputeOptions& options) {
  std::vector<const RDKit::ROMol*> mols    = {&mol};
  auto                             results = GetTFDMatrices(mols, options);

  if (results.empty()) {
    return {};
  }
  return results[0];
}

}  // namespace nvMolKit
