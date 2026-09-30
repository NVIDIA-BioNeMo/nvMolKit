// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "src/descriptors3d_mol.h"

#include <GraphMol/Conformer.h>
#include <GraphMol/Descriptors/MolData3Ddescriptors.h>
#include <GraphMol/ROMol.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "src/conformer/conformer_coord_upload.h"
#include "src/utils/openmp_helpers.h"

namespace nvMolKit {

namespace {

struct DeviceDescriptorInputs {
  AsyncDeviceVector<double>  momentWeights;
  AsyncDeviceVector<double>  whimWeights;
  AsyncDeviceVector<int8_t>  conformerIs3D;
  AsyncDeviceVector<int32_t> moleculeAtomStarts;
};

DeviceDescriptorInputs uploadDescriptorInputs(const std::vector<const RDKit::ROMol*>& mols,
                                              const bool                              includeMomentWeights,
                                              const bool                              includeWhimWeights,
                                              const bool                              includeConformerFlags,
                                              const int                               numThreads,
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

  std::vector<double> weights(includeMomentWeights ? static_cast<size_t>(totalAtoms) : 0);
  std::vector<double> whimWeights(includeWhimWeights ? static_cast<size_t>(totalAtoms) * 6 : 0);
  std::vector<int8_t> conformerIs3D;
  if (includeConformerFlags) {
    for (const RDKit::ROMol* mol : mols) {
      for (auto conformer = mol->beginConformers(); conformer != mol->endConformers(); ++conformer) {
        conformerIs3D.push_back((*conformer)->is3D());
      }
    }
  }
  detail::OpenMPExceptionRegistry exceptionRegistry;
  if (includeMomentWeights) {
#pragma omp parallel for num_threads(numThreads) schedule(dynamic) default(none) \
  shared(numMols, mols, atomStarts, weights, exceptionRegistry)
    for (int molIdx = 0; molIdx < numMols; ++molIdx) {
      try {
        size_t atomOffset = static_cast<size_t>(atomStarts[molIdx]);
        for (const auto* atom : mols[molIdx]->atoms()) {
          weights[atomOffset++] = atom->getMass();
        }
      } catch (...) {
        exceptionRegistry.store(std::current_exception());
      }
    }
    exceptionRegistry.rethrow();
  }
  if (includeWhimWeights) {
#pragma omp parallel for num_threads(numThreads) schedule(dynamic) default(none) \
  shared(numMols, mols, atomStarts, totalAtoms, whimWeights, exceptionRegistry)
    for (int molIdx = 0; molIdx < numMols; ++molIdx) {
      try {
        MolData3Ddescriptors                     descriptorData;
        const std::array<std::vector<double>, 6> moleculeWeights = {
          descriptorData.GetRelativeMW(*mols[molIdx]),
          descriptorData.GetRelativeVdW(*mols[molIdx]),
          descriptorData.GetRelativeENeg(*mols[molIdx]),
          descriptorData.GetRelativePol(*mols[molIdx]),
          descriptorData.GetRelativeIonPol(*mols[molIdx]),
          descriptorData.GetIState(*mols[molIdx]),
        };
        const size_t atomStart = static_cast<size_t>(atomStarts[molIdx]);
        for (size_t channel = 0; channel < moleculeWeights.size(); ++channel) {
          std::copy(moleculeWeights[channel].begin(),
                    moleculeWeights[channel].end(),
                    whimWeights.begin() + static_cast<size_t>(totalAtoms) * channel + atomStart);
        }
      } catch (...) {
        exceptionRegistry.store(std::current_exception());
      }
    }
    exceptionRegistry.rethrow();
  }

  DeviceDescriptorInputs result{AsyncDeviceVector<double>(weights.size(), stream),
                                AsyncDeviceVector<double>(whimWeights.size(), stream),
                                AsyncDeviceVector<int8_t>(conformerIs3D.size(), stream),
                                AsyncDeviceVector<int32_t>(atomStarts.size(), stream)};
  if (!weights.empty()) {
    result.momentWeights.copyFromHost(weights);
  }
  if (!whimWeights.empty()) {
    result.whimWeights.copyFromHost(whimWeights);
  }
  if (!conformerIs3D.empty()) {
    result.conformerIs3D.copyFromHost(conformerIs3D);
  }
  result.moleculeAtomStarts.copyFromHost(atomStarts);
  return result;
}

}  // namespace

template <typename Real>
Property3DBatchResult<Real> calc3DProperties(const std::vector<const RDKit::ROMol*>& mols,
                                             const std::vector<Property3D>&          properties,
                                             const Property3DOptions&                options,
                                             cudaStream_t                            stream,
                                             const DeviceCoordView*                  coordinates,
                                             const int                               preprocessingThreads) {
  const int numThreads = detail::resolveNumThreads(preprocessingThreads);
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
    uploaded = uploadConformerCoordinates(mols, stream, numThreads);
    view     = makeDeviceCoordView(uploaded);
  }
  bool includeMomentWeights  = false;
  bool includeWhimWeights    = false;
  bool includeConformerFlags = false;
  for (const Property3D property : properties) {
    const bool isMoment = property3DFamily(property) == Property3DFamily::Moments;
    includeMomentWeights |= isMoment && options.moments.useAtomicMasses && property != Property3D::SpherocityIndex;
    includeWhimWeights |= property == Property3D::WHIM;
    // Device coordinate rows carry no is3D flag and are treated as three-dimensional.
    includeConformerFlags |= property == Property3D::PBF && coordinates == nullptr;
  }
  const DeviceDescriptorInputs uploadedInputs =
    uploadDescriptorInputs(mols, includeMomentWeights, includeWhimWeights, includeConformerFlags, numThreads, stream);

  Property3DDeviceInputs inputs;
  inputs.moleculeAtomStarts = uploadedInputs.moleculeAtomStarts.data();
  inputs.momentWeights      = uploadedInputs.momentWeights.data();
  inputs.whimWeights        = uploadedInputs.whimWeights.data();
  inputs.conformerIs3D      = uploadedInputs.conformerIs3D.data();
  for (const RDKit::ROMol* mol : mols) {
    inputs.maxMoleculeAtoms = std::max(inputs.maxMoleculeAtoms, static_cast<int32_t>(mol->getNumAtoms()));
  }

  Property3DBatchResult<Real> result;
  result.properties  = calc3DPropertiesGpu<Real>(view, inputs, properties, options, stream);
  result.molIndices  = std::move(uploaded.molIndices);
  result.confIndices = std::move(uploaded.confIndices);
  return result;
}

template Property3DBatchResult<float>  calc3DProperties<float>(const std::vector<const RDKit::ROMol*>&,
                                                              const std::vector<Property3D>&,
                                                              const Property3DOptions&,
                                                              cudaStream_t,
                                                              const DeviceCoordView*,
                                                              int);
template Property3DBatchResult<double> calc3DProperties<double>(const std::vector<const RDKit::ROMol*>&,
                                                                const std::vector<Property3D>&,
                                                                const Property3DOptions&,
                                                                cudaStream_t,
                                                                const DeviceCoordView*,
                                                                int);

}  // namespace nvMolKit
