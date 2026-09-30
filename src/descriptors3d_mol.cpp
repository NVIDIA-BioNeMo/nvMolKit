// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "src/descriptors3d_mol.h"

#include <GraphMol/Conformer.h>
#include <GraphMol/Descriptors/MolData3Ddescriptors.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Substruct/SubstructMatch.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "src/conformer/conformer_coord_upload.h"
#include "src/utils/openmp_helpers.h"

namespace nvMolKit {

namespace {

struct DeviceDescriptorInputs {
  AsyncDeviceVector<double>  momentWeights;
  AsyncDeviceVector<double>  atomPropertyWeights;
  AsyncDeviceVector<double>  iStateDragWeights;
  AsyncDeviceVector<double>  covalentRadiusWeights;
  AsyncDeviceVector<uint8_t> topologicalDistances;
  AsyncDeviceVector<int64_t> topologicalDistanceStarts;
  AsyncDeviceVector<uint8_t> usrcatAtomClasses;
  AsyncDeviceVector<int8_t>  conformerIs3D;
  AsyncDeviceVector<int32_t> moleculeAtomStarts;
};

//! Per-molecule inputs the requested properties read; see Property3DDeviceInputs.
struct DescriptorInputNeeds {
  bool momentWeights         = false;
  bool atomPropertyWeights   = false;
  bool iStateDragWeights     = false;
  bool covalentRadiusWeights = false;
  bool topologicalDistances  = false;
  bool usrcatAtomClasses     = false;
  bool conformerFlags        = false;
};

//! Largest bond-count distance AUTOCORR3D reads.
constexpr int kMaxAutocorrLag = 10;

//! Bond-count distances of every pair j < k, row-major, capped: 0 for pairs farther than kMaxAutocorrLag
//! or disconnected. Matches RDKit's MolOps::getDistanceMat(mol, false) for distances up to the cap.
void writeTopologicalDistances(const RDKit::ROMol& mol, uint8_t* pairDistances) {
  const int        numAtoms = static_cast<int>(mol.getNumAtoms());
  std::vector<int> depth(numAtoms);
  std::vector<int> frontier;
  std::vector<int> next;
  size_t           pairIdx = 0;
  for (int source = 0; source < numAtoms - 1; ++source) {
    std::fill(depth.begin(), depth.end(), -1);
    depth[source] = 0;
    frontier.assign(1, source);
    for (int level = 1; level <= kMaxAutocorrLag && !frontier.empty(); ++level) {
      next.clear();
      for (const int atomIdx : frontier) {
        for (const auto* neighbor : mol.atomNeighbors(mol.getAtomWithIdx(atomIdx))) {
          const int neighborIdx = static_cast<int>(neighbor->getIdx());
          if (depth[neighborIdx] < 0) {
            depth[neighborIdx] = level;
            next.push_back(neighborIdx);
          }
        }
      }
      frontier.swap(next);
    }
    for (int target = source + 1; target < numAtoms; ++target) {
      pairDistances[pairIdx++] = static_cast<uint8_t>(std::max(depth[target], 0));
    }
  }
}

//! RDKit's USRCAT atom classes (hydrophobic, aromatic, acceptor, donor), from
//! Code/GraphMol/Descriptors/USRDescriptor.cpp.
const std::array<std::unique_ptr<RDKit::RWMol>, 4>& usrcatClassPatterns() {
  static const std::array<std::unique_ptr<RDKit::RWMol>, 4> patterns = {
    std::unique_ptr<RDKit::RWMol>(RDKit::SmartsToMol("[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]")),
    std::unique_ptr<RDKit::RWMol>(RDKit::SmartsToMol("[a]")),
    std::unique_ptr<RDKit::RWMol>(RDKit::SmartsToMol(
      "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N&v3;H1,H2]-[!$(*=[O,N,P,S])]),"
      "$([N;v3;H0]),$([n,o,s;+0]),F]")),
    std::unique_ptr<RDKit::RWMol>(RDKit::SmartsToMol("[N!H0v3,N!H0+v4,OH+0,SH+0,nH+0]")),
  };
  return patterns;
}

DeviceDescriptorInputs uploadDescriptorInputs(const std::vector<const RDKit::ROMol*>& mols,
                                              const DescriptorInputNeeds&             needs,
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

  std::vector<int64_t> topologyStarts;
  if (needs.topologicalDistances) {
    topologyStarts.assign(numMols + 1, 0);
    for (int molIdx = 0; molIdx < numMols; ++molIdx) {
      const int64_t numAtoms     = mols[molIdx]->getNumAtoms();
      topologyStarts[molIdx + 1] = topologyStarts[molIdx] + numAtoms * (numAtoms - 1) / 2;
    }
  }
  std::vector<double>  weights(needs.momentWeights ? static_cast<size_t>(totalAtoms) : 0);
  std::vector<double>  atomPropertyWeights(needs.atomPropertyWeights ? static_cast<size_t>(totalAtoms) * 6 : 0);
  std::vector<double>  iStateDragWeights(needs.iStateDragWeights ? static_cast<size_t>(totalAtoms) : 0);
  std::vector<double>  covalentRadiusWeights(needs.covalentRadiusWeights ? static_cast<size_t>(totalAtoms) : 0);
  std::vector<uint8_t> topologicalDistances(needs.topologicalDistances ? static_cast<size_t>(topologyStarts.back()) :
                                                                         0);
  std::vector<uint8_t> usrcatAtomClasses(needs.usrcatAtomClasses ? static_cast<size_t>(totalAtoms) : 0);
  std::vector<int8_t>  conformerIs3D;
  if (needs.conformerFlags) {
    for (const RDKit::ROMol* mol : mols) {
      for (auto conformer = mol->beginConformers(); conformer != mol->endConformers(); ++conformer) {
        conformerIs3D.push_back((*conformer)->is3D());
      }
    }
  }
  detail::OpenMPExceptionRegistry exceptionRegistry;
  if (needs.momentWeights) {
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
  if (needs.atomPropertyWeights || needs.iStateDragWeights || needs.covalentRadiusWeights ||
      needs.topologicalDistances || needs.usrcatAtomClasses) {
#pragma omp parallel for num_threads(numThreads) schedule(dynamic) default(none) shared(numMols,                 \
                                                                                          mols,                  \
                                                                                          atomStarts,            \
                                                                                          totalAtoms,            \
                                                                                          needs,                 \
                                                                                          topologyStarts,        \
                                                                                          atomPropertyWeights,   \
                                                                                          iStateDragWeights,     \
                                                                                          covalentRadiusWeights, \
                                                                                          topologicalDistances,  \
                                                                                          usrcatAtomClasses,     \
                                                                                          exceptionRegistry)
    for (int molIdx = 0; molIdx < numMols; ++molIdx) {
      try {
        const RDKit::ROMol&  mol = *mols[molIdx];
        MolData3Ddescriptors descriptorData;
        const size_t         atomStart = static_cast<size_t>(atomStarts[molIdx]);
        if (needs.atomPropertyWeights) {
          const std::array<std::vector<double>, 6> moleculeWeights = {
            descriptorData.GetRelativeMW(mol),
            descriptorData.GetRelativeVdW(mol),
            descriptorData.GetRelativeENeg(mol),
            descriptorData.GetRelativePol(mol),
            descriptorData.GetRelativeIonPol(mol),
            descriptorData.GetIState(mol),
          };
          for (size_t channel = 0; channel < moleculeWeights.size(); ++channel) {
            std::copy(moleculeWeights[channel].begin(),
                      moleculeWeights[channel].end(),
                      atomPropertyWeights.begin() + static_cast<size_t>(totalAtoms) * channel + atomStart);
          }
        }
        if (needs.iStateDragWeights) {
          const std::vector<double> iStateDrag = descriptorData.GetIStateDrag(mol);
          std::copy(iStateDrag.begin(), iStateDrag.end(), iStateDragWeights.begin() + atomStart);
        }
        if (needs.covalentRadiusWeights) {
          const std::vector<double> radii = descriptorData.GetRelativeRcov(mol);
          std::copy(radii.begin(), radii.end(), covalentRadiusWeights.begin() + atomStart);
        }
        if (needs.topologicalDistances) {
          writeTopologicalDistances(mol, topologicalDistances.data() + topologyStarts[molIdx]);
        }
        if (needs.usrcatAtomClasses) {
          const auto& patterns = usrcatClassPatterns();
          for (size_t classIdx = 0; classIdx < patterns.size(); ++classIdx) {
            // Same call as RDKit's USRCAT, including its default maxMatches.
            std::vector<RDKit::MatchVectType> matches;
            RDKit::SubstructMatch(mol, RDKit::ROMol(*patterns[classIdx], true), matches);
            for (const auto& match : matches) {
              for (const auto& [queryIdx, atomIdx] : match) {
                usrcatAtomClasses[atomStart + atomIdx] |= static_cast<uint8_t>(1u << classIdx);
              }
            }
          }
        }
      } catch (...) {
        exceptionRegistry.store(std::current_exception());
      }
    }
    exceptionRegistry.rethrow();
  }

  DeviceDescriptorInputs result{AsyncDeviceVector<double>(weights.size(), stream),
                                AsyncDeviceVector<double>(atomPropertyWeights.size(), stream),
                                AsyncDeviceVector<double>(iStateDragWeights.size(), stream),
                                AsyncDeviceVector<double>(covalentRadiusWeights.size(), stream),
                                AsyncDeviceVector<uint8_t>(topologicalDistances.size(), stream),
                                AsyncDeviceVector<int64_t>(topologyStarts.size(), stream),
                                AsyncDeviceVector<uint8_t>(usrcatAtomClasses.size(), stream),
                                AsyncDeviceVector<int8_t>(conformerIs3D.size(), stream),
                                AsyncDeviceVector<int32_t>(atomStarts.size(), stream)};
  if (!weights.empty()) {
    result.momentWeights.copyFromHost(weights);
  }
  if (!atomPropertyWeights.empty()) {
    result.atomPropertyWeights.copyFromHost(atomPropertyWeights);
  }
  if (!iStateDragWeights.empty()) {
    result.iStateDragWeights.copyFromHost(iStateDragWeights);
  }
  if (!covalentRadiusWeights.empty()) {
    result.covalentRadiusWeights.copyFromHost(covalentRadiusWeights);
  }
  if (!topologicalDistances.empty()) {
    result.topologicalDistances.copyFromHost(topologicalDistances);
  }
  if (!topologyStarts.empty()) {
    result.topologicalDistanceStarts.copyFromHost(topologyStarts);
  }
  if (!usrcatAtomClasses.empty()) {
    result.usrcatAtomClasses.copyFromHost(usrcatAtomClasses);
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
  DescriptorInputNeeds needs;
  for (const Property3D property : properties) {
    const Property3DFamily family = property3DFamily(property);
    needs.momentWeights |=
      family == Property3DFamily::Moments && options.moments.useAtomicMasses && property != Property3D::SpherocityIndex;
    needs.atomPropertyWeights |= property == Property3D::WHIM || family == Property3DFamily::Pairwise;
    needs.iStateDragWeights |= property == Property3D::RDF;
    needs.covalentRadiusWeights |= property == Property3D::AUTOCORR3D;
    needs.topologicalDistances |= property == Property3D::AUTOCORR3D;
    needs.usrcatAtomClasses |= property == Property3D::USRCAT;
    // Device coordinate rows carry no is3D flag and are treated as three-dimensional.
    needs.conformerFlags |= property == Property3D::PBF && coordinates == nullptr;
  }
  const DeviceDescriptorInputs uploadedInputs = uploadDescriptorInputs(mols, needs, numThreads, stream);

  Property3DDeviceInputs inputs;
  inputs.moleculeAtomStarts        = uploadedInputs.moleculeAtomStarts.data();
  inputs.momentWeights             = uploadedInputs.momentWeights.data();
  inputs.atomPropertyWeights       = uploadedInputs.atomPropertyWeights.data();
  inputs.iStateDragWeights         = uploadedInputs.iStateDragWeights.data();
  inputs.covalentRadiusWeights     = uploadedInputs.covalentRadiusWeights.data();
  inputs.topologicalDistances      = uploadedInputs.topologicalDistances.data();
  inputs.topologicalDistanceStarts = uploadedInputs.topologicalDistanceStarts.data();
  inputs.usrcatAtomClasses         = uploadedInputs.usrcatAtomClasses.data();
  inputs.conformerIs3D             = uploadedInputs.conformerIs3D.data();
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
