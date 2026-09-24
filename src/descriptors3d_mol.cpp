// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "src/descriptors3d_mol.h"

#include <GraphMol/ROMol.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "src/conformer/conformer_coord_upload.h"

namespace nvMolKit {

namespace {

//! Per-molecule atom weights in CSR form, resident on the device.
struct DeviceAtomWeights {
  AsyncDeviceVector<double>  weights;
  AsyncDeviceVector<int32_t> moleculeAtomStarts;
};

DeviceAtomWeights uploadAtomWeights(const std::vector<const RDKit::ROMol*>& mols,
                                    const bool                              useAtomicMasses,
                                    cudaStream_t                            stream) {
  const int            numMols = static_cast<int>(mols.size());
  std::vector<int32_t> atomStarts(numMols + 1, 0);
  int64_t              totalAtoms = 0;
  for (int molIdx = 0; molIdx < numMols; ++molIdx) {
    atomStarts[molIdx] = static_cast<int32_t>(totalAtoms);
    totalAtoms += mols[molIdx]->getNumAtoms();
    if (totalAtoms > std::numeric_limits<int32_t>::max()) {
      throw std::overflow_error("Total molecule atom count exceeds int32 range");
    }
  }
  atomStarts[numMols] = static_cast<int32_t>(totalAtoms);

  std::vector<double> weights(useAtomicMasses ? static_cast<size_t>(totalAtoms) : 0);
  if (useAtomicMasses) {
#pragma omp parallel for schedule(dynamic)
    for (int molIdx = 0; molIdx < numMols; ++molIdx) {
      size_t atomOffset = static_cast<size_t>(atomStarts[molIdx]);
      for (const auto* atom : mols[molIdx]->atoms()) {
        weights[atomOffset++] = atom->getMass();
      }
    }
  }

  DeviceAtomWeights result{AsyncDeviceVector<double>(weights.size(), stream),
                           AsyncDeviceVector<int32_t>(atomStarts.size(), stream)};
  if (!weights.empty()) {
    result.weights.copyFromHost(weights);
  }
  result.moleculeAtomStarts.copyFromHost(atomStarts);
  return result;
}

}  // namespace

template <typename Real>
Property3DBatchResult<Real> calc3DProperties(const std::vector<const RDKit::ROMol*>& mols,
                                             const std::vector<Property3D>&          properties,
                                             const bool                              useAtomicMasses,
                                             cudaStream_t                            stream,
                                             const DeviceCoordView*                  coordinates) {
  for (size_t molIdx = 0; molIdx < mols.size(); ++molIdx) {
    if (mols[molIdx] == nullptr) {
      throw std::invalid_argument("Null molecule at index " + std::to_string(molIdx));
    }
  }
  if (coordinates != nullptr && coordinates->nMols != static_cast<int>(mols.size())) {
    throw std::invalid_argument("Device coordinates describe " + std::to_string(coordinates->nMols) +
                                " molecules, but " + std::to_string(mols.size()) + " molecules were provided");
  }

  // Uploaded buffers outlive the kernel launch below; their stream-ordered frees run after it.
  DeviceCoordResult uploaded;
  DeviceCoordView   view;
  if (coordinates != nullptr) {
    view = *coordinates;
  } else {
    uploaded = uploadConformerCoordinates(mols, stream);
    view     = makeDeviceCoordView(uploaded);
  }
  const DeviceAtomWeights weights = uploadAtomWeights(mols, useAtomicMasses, stream);

  Property3DBatchResult<Real> result;
  const double*               atomWeights = useAtomicMasses ? weights.weights.data() : nullptr;
  result.properties =
    calc3DPropertiesGpu<Real>(view, atomWeights, weights.moleculeAtomStarts.data(), properties, stream);
  result.molIndices  = std::move(uploaded.molIndices);
  result.confIndices = std::move(uploaded.confIndices);
  return result;
}

template Property3DBatchResult<float>  calc3DProperties<float>(const std::vector<const RDKit::ROMol*>&,
                                                              const std::vector<Property3D>&,
                                                              bool,
                                                              cudaStream_t,
                                                              const DeviceCoordView*);
template Property3DBatchResult<double> calc3DProperties<double>(const std::vector<const RDKit::ROMol*>&,
                                                                const std::vector<Property3D>&,
                                                                bool,
                                                                cudaStream_t,
                                                                const DeviceCoordView*);

}  // namespace nvMolKit
