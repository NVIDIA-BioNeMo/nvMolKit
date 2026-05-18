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

#include <GraphMol/Conformer.h>

#include <boost/python.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "bfgs_mmff.h"
#include "bfgs_uff.h"
#include "boost_python_utils.h"
#include "device_coord_result.h"
#include "device_result_python.h"
#include "device_vector.h"
#include "ff_utils.h"
#include "forcefield_constraints.h"
#include "hardware_options.h"
#include "mmff_batched_forcefield.h"
#include "mmff_flattened_builder.h"
#include "mmff_properties.h"
#include "mmff_python_utils.h"
#include "uff_batched_forcefield.h"
#include "uff_flattened_builder.h"

namespace bp = boost::python;

namespace {

std::vector<std::vector<double>> splitGradients(const std::vector<double>& flatGrad,
                                                const std::vector<int>&    atomStarts,
                                                int                        dim) {
  std::vector<std::vector<double>> result;
  result.reserve(atomStarts.size() - 1);
  for (size_t i = 0; i + 1 < atomStarts.size(); ++i) {
    const int start = atomStarts[i] * dim;
    const int end   = atomStarts[i + 1] * dim;
    result.emplace_back(flatGrad.begin() + start, flatGrad.begin() + end);
  }
  return result;
}

bp::list reshapeToNested(const std::vector<double>& flat, const std::vector<int>& numConformersPerMol) {
  bp::list outer;
  size_t   idx = 0;
  for (const int nConfs : numConformersPerMol) {
    bp::list inner;
    for (int j = 0; j < nConfs; ++j) {
      inner.append(flat[idx++]);
    }
    outer.append(inner);
  }
  return outer;
}

bp::list reshapeGradientsToNested(const std::vector<std::vector<double>>& perSystem,
                                  const std::vector<int>&                 numConformersPerMol) {
  bp::list outer;
  size_t   idx = 0;
  for (const int nConfs : numConformersPerMol) {
    bp::list inner;
    for (int j = 0; j < nConfs; ++j) {
      inner.append(nvMolKit::vectorToList(perSystem[idx++]));
    }
    outer.append(inner);
  }
  return outer;
}

template <typename T, typename Convert>
bp::list nestedToList(const std::vector<std::vector<T>>& nested, Convert&& convert) {
  bp::list outer;
  for (const auto& innerVec : nested) {
    bp::list inner;
    for (const auto& val : innerVec) {
      inner.append(convert(val));
    }
    outer.append(inner);
  }
  return outer;
}

void throwIfCudaError(cudaError_t err, const std::string& context) {
  if (err != cudaSuccess) {
    throw std::runtime_error(context + ": " + cudaGetErrorString(err));
  }
}

template <typename T> std::vector<T> copyDeviceVector(nvMolKit::AsyncDeviceVector<T>& deviceVec) {
  std::vector<T> hostVec(deviceVec.size());
  deviceVec.copyToHost(hostVec);
  throwIfCudaError(cudaStreamSynchronize(deviceVec.stream()), "copyDeviceVector/sync");
  return hostVec;
}

void uploadConformerPositions(const std::vector<RDKit::ROMol*>&    mols,
                              nvMolKit::AsyncDeviceVector<double>& positionsDevice) {
  std::vector<double> allPositions;
  for (auto* mol : mols) {
    std::vector<double> pos;
    for (auto confIter = mol->beginConformers(); confIter != mol->endConformers(); ++confIter) {
      nvMolKit::confPosToVect(**confIter, pos);
      allPositions.insert(allPositions.end(), pos.begin(), pos.end());
    }
  }
  positionsDevice.copyFromHost(allPositions.data(), allPositions.size());
  throwIfCudaError(cudaStreamSynchronize(positionsDevice.stream()), "uploadConformerPositions/sync");
}

bp::list computeBatchedEnergy(nvMolKit::BatchedForcefield&         forcefield,
                              nvMolKit::AsyncDeviceVector<double>& positionsDevice,
                              nvMolKit::AsyncDeviceVector<double>& energyOutsDevice,
                              const std::vector<int>&              numConformersPerMol) {
  energyOutsDevice.zero();
  throwIfCudaError(forcefield.computeEnergy(energyOutsDevice.data(), positionsDevice.data()), "computeEnergy");
  return reshapeToNested(copyDeviceVector(energyOutsDevice), numConformersPerMol);
}

bp::list computeBatchedGradients(nvMolKit::BatchedForcefield&         forcefield,
                                 nvMolKit::AsyncDeviceVector<double>& positionsDevice,
                                 nvMolKit::AsyncDeviceVector<double>& gradDevice,
                                 const std::vector<int>&              numConformersPerMol) {
  gradDevice.zero();
  throwIfCudaError(forcefield.computeGradients(gradDevice.data(), positionsDevice.data()), "computeGradients");
  auto perSystem = splitGradients(copyDeviceVector(gradDevice), forcefield.atomStartsHost(), 3);
  return reshapeGradientsToNested(perSystem, numConformersPerMol);
}

}  // namespace

template <typename Spec, typename Parser>
static std::vector<std::vector<Spec>> extractConstraintLists(const bp::list&    outerList,
                                                             const int          expectedSize,
                                                             const Parser&      parser,
                                                             const std::string& name) {
  if (bp::len(outerList) != expectedSize) {
    throw std::invalid_argument("Expected " + std::to_string(expectedSize) + " entries for " + name + ", got " +
                                std::to_string(bp::len(outerList)));
  }
  std::vector<std::vector<Spec>> allSpecs(expectedSize);
  for (int molIdx = 0; molIdx < expectedSize; ++molIdx) {
    const bp::list innerList = bp::extract<bp::list>(bp::object(outerList[molIdx]));
    auto&          specs     = allSpecs[molIdx];
    specs.reserve(bp::len(innerList));
    for (int j = 0; j < bp::len(innerList); ++j) {
      specs.push_back(parser(bp::extract<bp::tuple>(bp::object(innerList[j]))));
    }
  }
  return allSpecs;
}

static nvMolKit::ForceFieldConstraints::DistanceConstraintSpec parseDistanceConstraintTuple(const bp::tuple& value) {
  if (bp::len(value) != 6) {
    throw std::invalid_argument("Distance constraint tuples must have 6 elements");
  }
  return {bp::extract<int>(value[0]),
          bp::extract<int>(value[1]),
          bp::extract<bool>(value[2]),
          bp::extract<double>(value[3]),
          bp::extract<double>(value[4]),
          bp::extract<double>(value[5])};
}

static nvMolKit::ForceFieldConstraints::PositionConstraintSpec parsePositionConstraintTuple(const bp::tuple& value) {
  if (bp::len(value) != 3) {
    throw std::invalid_argument("Position constraint tuples must have 3 elements");
  }
  return {bp::extract<int>(value[0]), bp::extract<double>(value[1]), bp::extract<double>(value[2])};
}

static nvMolKit::ForceFieldConstraints::AngleConstraintSpec parseAngleConstraintTuple(const bp::tuple& value) {
  if (bp::len(value) != 7) {
    throw std::invalid_argument("Angle constraint tuples must have 7 elements");
  }
  return {bp::extract<int>(value[0]),
          bp::extract<int>(value[1]),
          bp::extract<int>(value[2]),
          bp::extract<bool>(value[3]),
          bp::extract<double>(value[4]),
          bp::extract<double>(value[5]),
          bp::extract<double>(value[6])};
}

static nvMolKit::ForceFieldConstraints::TorsionConstraintSpec parseTorsionConstraintTuple(const bp::tuple& value) {
  if (bp::len(value) != 8) {
    throw std::invalid_argument("Torsion constraint tuples must have 8 elements");
  }
  return {bp::extract<int>(value[0]),
          bp::extract<int>(value[1]),
          bp::extract<int>(value[2]),
          bp::extract<int>(value[3]),
          bp::extract<bool>(value[4]),
          bp::extract<double>(value[5]),
          bp::extract<double>(value[6]),
          bp::extract<double>(value[7])};
}

namespace FC = nvMolKit::ForceFieldConstraints;

static std::vector<FC::PerMolConstraints> extractAllConstraints(const bp::list& distanceConstraints,
                                                                const bp::list& positionConstraints,
                                                                const bp::list& angleConstraints,
                                                                const bp::list& torsionConstraints,
                                                                int             numMols) {
  const auto distLists    = extractConstraintLists<FC::DistanceConstraintSpec>(distanceConstraints,
                                                                            numMols,
                                                                            parseDistanceConstraintTuple,
                                                                            "distance constraints");
  const auto posLists     = extractConstraintLists<FC::PositionConstraintSpec>(positionConstraints,
                                                                           numMols,
                                                                           parsePositionConstraintTuple,
                                                                           "position constraints");
  const auto angleLists   = extractConstraintLists<FC::AngleConstraintSpec>(angleConstraints,
                                                                          numMols,
                                                                          parseAngleConstraintTuple,
                                                                          "angle constraints");
  const auto torsionLists = extractConstraintLists<FC::TorsionConstraintSpec>(torsionConstraints,
                                                                              numMols,
                                                                              parseTorsionConstraintTuple,
                                                                              "torsion constraints");

  std::vector<FC::PerMolConstraints> result(numMols);
  for (int i = 0; i < numMols; ++i) {
    result[i] = {distLists[i], posLists[i], angleLists[i], torsionLists[i]};
  }
  return result;
}

class NativeMMFFBatchedForcefield {
 public:
  NativeMMFFBatchedForcefield(const bp::list&                       molecules,
                              const bp::list&                       properties,
                              const bp::list&                       distanceConstraints,
                              const bp::list&                       positionConstraints,
                              const bp::list&                       angleConstraints,
                              const bp::list&                       torsionConstraints,
                              const nvMolKit::BatchHardwareOptions& hwOpts)
      : hwOpts_(hwOpts) {
    throwIfCudaError(cudaGetDevice(&gpuId_), "MMFF wrapper/cudaGetDevice");
    mols_             = nvMolKit::extractMolecules(molecules);
    const int numMols = static_cast<int>(mols_.size());
    properties_       = nvMolKit::extractMMFFPropertiesList(properties, numMols);
    constraints_ =
      extractAllConstraints(distanceConstraints, positionConstraints, angleConstraints, torsionConstraints, numMols);

    buildForcefield();
  }

  bp::list computeEnergy() {
    return computeBatchedEnergy(*forcefield_, positionsDevice_, energyOutsDevice_, numConformersPerMol_);
  }

  bp::list computeGradients() {
    return computeBatchedGradients(*forcefield_, positionsDevice_, gradDevice_, numConformersPerMol_);
  }

  bp::tuple minimize(int maxIters, double gradTol) {
    auto result =
      nvMolKit::MMFF::MMFFMinimizeMoleculesConfs(mols_, maxIters, gradTol, properties_, constraints_, hwOpts_);

    uploadConformerPositions(mols_, positionsDevice_);

    return bp::make_tuple(nestedToList(result.energies, [](double v) { return v; }),
                          nestedToList(result.converged, [](int8_t v) { return v != 0; }));
  }

  bp::object minimizeDevice(int maxIters, double gradTol, int targetGpu) {
    if (targetGpu >= 0 && targetGpu != gpuId_) {
      throw std::invalid_argument(
        "MMFFBatchedForcefield.minimize(output=DEVICE) does not support target_gpu != wrapper GPU "
        "(" +
        std::to_string(targetGpu) + " vs " + std::to_string(gpuId_) +
        "). The wrapper's persistent device state lives on a single GPU; consolidating elsewhere "
        "would leave subsequent compute_energy/compute_gradients calls operating on stale "
        "coordinates. Use the standalone MMFFOptimizeMoleculesConfs(output=DEVICE, targetGpu=...) "
        "API for cross-GPU consolidation, or construct the wrapper on the desired GPU.");
    }
    auto result = nvMolKit::MMFF::MMFFMinimizeMoleculesConfs(mols_,
                                                             maxIters,
                                                             gradTol,
                                                             properties_,
                                                             constraints_,
                                                             hwOpts_,
                                                             nvMolKit::BfgsBackend::HYBRID,
                                                             nvMolKit::CoordinateOutput::DEVICE,
                                                             gpuId_);
    if (!result.device.has_value()) {
      throw std::runtime_error("MMFFMinimizeMoleculesConfs(DEVICE) returned no device result");
    }
    auto& dev = *result.device;
    if (dev.positions.size() != positionsDevice_.size()) {
      throw std::runtime_error("MMFFMinimizeMoleculesConfs(DEVICE) positions size does not match wrapper");
    }
    // Refresh the persistent positions buffer in-place so subsequent compute_energy /
    // compute_gradients calls see the optimized coords without a host roundtrip.
    throwIfCudaError(cudaMemcpyAsync(positionsDevice_.data(),
                                     dev.positions.data(),
                                     positionsDevice_.size() * sizeof(double),
                                     cudaMemcpyDeviceToDevice,
                                     positionsDevice_.stream()),
                     "MMFF minimizeDevice/positions refresh");
    throwIfCudaError(cudaStreamSynchronize(positionsDevice_.stream()), "MMFF minimizeDevice/positions refresh sync");
    return nvMolKit::buildOwningDevice3DResult(dev);
  }

  int gpuId() const { return gpuId_; }

 private:
  void buildForcefield() {
    const int                                  numMols = static_cast<int>(mols_.size());
    nvMolKit::MMFF::BatchedMolecularSystemHost systemHost;
    nvMolKit::BatchedForcefieldMetadata        metadata;
    numConformersPerMol_.resize(numMols);

    for (int molIdx = 0; molIdx < numMols; ++molIdx) {
      auto* mol          = mols_[molIdx];
      auto  baseContribs = nvMolKit::MMFF::constructForcefieldContribs(*mol, properties_[molIdx]);

      int confIdx = 0;
      for (auto confIter = mol->beginConformers(); confIter != mol->endConformers(); ++confIter, ++confIdx) {
        auto&               conf = **confIter;
        std::vector<double> positions;
        nvMolKit::confPosToVect(conf, positions);

        auto contribs = baseContribs;
        constraints_[molIdx].applyTo(contribs, positions);

        nvMolKit::MMFF::addMoleculeToBatch(contribs, positions, systemHost, &metadata, molIdx, confIdx);
      }
      numConformersPerMol_[molIdx] = confIdx;
    }

    forcefield_ = std::make_unique<nvMolKit::MMFFBatchedForcefield>(systemHost, metadata);
    positionsDevice_.setFromVector(systemHost.positions);
    gradDevice_.resize(forcefield_->totalPositions());
    energyOutsDevice_.resize(forcefield_->numMolecules());
  }

  std::vector<RDKit::ROMol*>            mols_;
  std::vector<nvMolKit::MMFFProperties> properties_;
  std::vector<FC::PerMolConstraints>    constraints_;
  nvMolKit::BatchHardwareOptions        hwOpts_;

  std::unique_ptr<nvMolKit::MMFFBatchedForcefield> forcefield_;
  nvMolKit::AsyncDeviceVector<double>              positionsDevice_;
  nvMolKit::AsyncDeviceVector<double>              gradDevice_;
  nvMolKit::AsyncDeviceVector<double>              energyOutsDevice_;
  std::vector<int>                                 numConformersPerMol_;
  int                                              gpuId_ = 0;
};

class NativeUFFBatchedForcefield {
 public:
  NativeUFFBatchedForcefield(const bp::list&                       molecules,
                             const bp::list&                       vdwThresholds,
                             const bp::list&                       ignoreInterfragInteractions,
                             const bp::list&                       distanceConstraints,
                             const bp::list&                       positionConstraints,
                             const bp::list&                       angleConstraints,
                             const bp::list&                       torsionConstraints,
                             const nvMolKit::BatchHardwareOptions& hwOpts)
      : hwOpts_(hwOpts) {
    throwIfCudaError(cudaGetDevice(&gpuId_), "UFF wrapper/cudaGetDevice");
    mols_             = nvMolKit::extractMolecules(molecules);
    const int numMols = static_cast<int>(mols_.size());
    vdwThresholds_    = nvMolKit::extractDoubleList(vdwThresholds, numMols, "vdwThreshold");
    ignoreInterfragInteractions_ =
      nvMolKit::extractBoolList(ignoreInterfragInteractions, numMols, "ignoreInterfragInteractions");
    constraints_ =
      extractAllConstraints(distanceConstraints, positionConstraints, angleConstraints, torsionConstraints, numMols);

    buildForcefield();
  }

  bp::list computeEnergy() {
    return computeBatchedEnergy(*forcefield_, positionsDevice_, energyOutsDevice_, numConformersPerMol_);
  }

  bp::list computeGradients() {
    return computeBatchedGradients(*forcefield_, positionsDevice_, gradDevice_, numConformersPerMol_);
  }

  bp::tuple minimize(int maxIters, double gradTol) {
    auto result = nvMolKit::UFF::UFFMinimizeMoleculesConfs(mols_,
                                                           maxIters,
                                                           gradTol,
                                                           vdwThresholds_,
                                                           ignoreInterfragInteractions_,
                                                           constraints_,
                                                           hwOpts_);

    uploadConformerPositions(mols_, positionsDevice_);

    return bp::make_tuple(nestedToList(result.energies, [](double v) { return v; }),
                          nestedToList(result.converged, [](int8_t v) { return v != 0; }));
  }

  bp::object minimizeDevice(int maxIters, double gradTol, int targetGpu) {
    if (targetGpu >= 0 && targetGpu != gpuId_) {
      throw std::invalid_argument(
        "UFFBatchedForcefield.minimize(output=DEVICE) does not support target_gpu != wrapper GPU "
        "(" +
        std::to_string(targetGpu) + " vs " + std::to_string(gpuId_) +
        "). The wrapper's persistent device state lives on a single GPU; consolidating elsewhere "
        "would leave subsequent compute_energy/compute_gradients calls operating on stale "
        "coordinates. Use the standalone UFFOptimizeMoleculesConfs(output=DEVICE, targetGpu=...) "
        "API for cross-GPU consolidation, or construct the wrapper on the desired GPU.");
    }
    auto result = nvMolKit::UFF::UFFMinimizeMoleculesConfs(mols_,
                                                           maxIters,
                                                           gradTol,
                                                           vdwThresholds_,
                                                           ignoreInterfragInteractions_,
                                                           constraints_,
                                                           hwOpts_,
                                                           nvMolKit::CoordinateOutput::DEVICE,
                                                           gpuId_);
    if (!result.device.has_value()) {
      throw std::runtime_error("UFFMinimizeMoleculesConfs(DEVICE) returned no device result");
    }
    auto& dev = *result.device;
    if (dev.positions.size() != positionsDevice_.size()) {
      throw std::runtime_error("UFFMinimizeMoleculesConfs(DEVICE) positions size does not match wrapper");
    }
    throwIfCudaError(cudaMemcpyAsync(positionsDevice_.data(),
                                     dev.positions.data(),
                                     positionsDevice_.size() * sizeof(double),
                                     cudaMemcpyDeviceToDevice,
                                     positionsDevice_.stream()),
                     "UFF minimizeDevice/positions refresh");
    throwIfCudaError(cudaStreamSynchronize(positionsDevice_.stream()), "UFF minimizeDevice/positions refresh sync");
    return nvMolKit::buildOwningDevice3DResult(dev);
  }

  int gpuId() const { return gpuId_; }

 private:
  void buildForcefield() {
    const int                                 numMols = static_cast<int>(mols_.size());
    nvMolKit::UFF::BatchedMolecularSystemHost systemHost;
    nvMolKit::BatchedForcefieldMetadata       metadata;
    numConformersPerMol_.resize(numMols);

    for (int molIdx = 0; molIdx < numMols; ++molIdx) {
      auto* mol     = mols_[molIdx];
      int   confIdx = 0;
      for (auto confIter = mol->beginConformers(); confIter != mol->endConformers(); ++confIter, ++confIdx) {
        auto&               conf = **confIter;
        std::vector<double> positions;
        nvMolKit::confPosToVect(conf, positions);

        auto contribs = nvMolKit::UFF::constructForcefieldContribs(*mol,
                                                                   vdwThresholds_[molIdx],
                                                                   conf.getId(),
                                                                   ignoreInterfragInteractions_[molIdx]);
        constraints_[molIdx].applyTo(contribs, positions);

        nvMolKit::UFF::addMoleculeToBatch(contribs, positions, systemHost, metadata, molIdx, confIdx);
      }
      numConformersPerMol_[molIdx] = confIdx;
    }

    forcefield_ = std::make_unique<nvMolKit::UFFBatchedForcefield>(systemHost, metadata);
    positionsDevice_.setFromVector(systemHost.positions);
    gradDevice_.resize(forcefield_->totalPositions());
    energyOutsDevice_.resize(forcefield_->numMolecules());
  }

  std::vector<RDKit::ROMol*>         mols_;
  std::vector<double>                vdwThresholds_;
  std::vector<bool>                  ignoreInterfragInteractions_;
  std::vector<FC::PerMolConstraints> constraints_;
  nvMolKit::BatchHardwareOptions     hwOpts_;

  std::unique_ptr<nvMolKit::UFFBatchedForcefield> forcefield_;
  nvMolKit::AsyncDeviceVector<double>             positionsDevice_;
  nvMolKit::AsyncDeviceVector<double>             gradDevice_;
  nvMolKit::AsyncDeviceVector<double>             energyOutsDevice_;
  std::vector<int>                                numConformersPerMol_;
  int                                             gpuId_ = 0;
};

BOOST_PYTHON_MODULE(_batchedForcefield) {
  bp::class_<nvMolKit::MMFFProperties>("MMFFProperties")
    .def_readwrite("variant", &nvMolKit::MMFFProperties::variant)
    .def_readwrite("dielectricConstant", &nvMolKit::MMFFProperties::dielectricConstant)
    .def_readwrite("dielectricModel", &nvMolKit::MMFFProperties::dielectricModel)
    .def_readwrite("nonBondedThreshold", &nvMolKit::MMFFProperties::nonBondedThreshold)
    .def_readwrite("ignoreInterfragInteractions", &nvMolKit::MMFFProperties::ignoreInterfragInteractions)
    .def_readwrite("bondTerm", &nvMolKit::MMFFProperties::bondTerm)
    .def_readwrite("angleTerm", &nvMolKit::MMFFProperties::angleTerm)
    .def_readwrite("stretchBendTerm", &nvMolKit::MMFFProperties::stretchBendTerm)
    .def_readwrite("oopTerm", &nvMolKit::MMFFProperties::oopTerm)
    .def_readwrite("torsionTerm", &nvMolKit::MMFFProperties::torsionTerm)
    .def_readwrite("vdwTerm", &nvMolKit::MMFFProperties::vdwTerm)
    .def_readwrite("eleTerm", &nvMolKit::MMFFProperties::eleTerm);

  bp::def("buildMMFFPropertiesFromRDKit",
          &nvMolKit::buildMMFFPropertiesFromRDKit,
          (bp::arg("rdkit_properties"), bp::arg("non_bonded_threshold"), bp::arg("ignore_interfrag_interactions")),
          "Build an nvMolKit MMFFProperties transport from an RDKit MMFFMolProperties Python object.");

  bp::class_<NativeMMFFBatchedForcefield, boost::noncopyable>("NativeMMFFBatchedForcefield",
                                                              bp::init<const bp::list&,
                                                                       const bp::list&,
                                                                       const bp::list&,
                                                                       const bp::list&,
                                                                       const bp::list&,
                                                                       const bp::list&,
                                                                       const nvMolKit::BatchHardwareOptions&>())
    .def("computeEnergy", &NativeMMFFBatchedForcefield::computeEnergy)
    .def("computeGradients", &NativeMMFFBatchedForcefield::computeGradients)
    .def("minimize", &NativeMMFFBatchedForcefield::minimize)
    .def("minimizeDevice", &NativeMMFFBatchedForcefield::minimizeDevice)
    .def("gpuId", &NativeMMFFBatchedForcefield::gpuId);

  bp::class_<NativeUFFBatchedForcefield, boost::noncopyable>("NativeUFFBatchedForcefield",
                                                             bp::init<const bp::list&,
                                                                      const bp::list&,
                                                                      const bp::list&,
                                                                      const bp::list&,
                                                                      const bp::list&,
                                                                      const bp::list&,
                                                                      const bp::list&,
                                                                      const nvMolKit::BatchHardwareOptions&>())
    .def("computeEnergy", &NativeUFFBatchedForcefield::computeEnergy)
    .def("computeGradients", &NativeUFFBatchedForcefield::computeGradients)
    .def("minimize", &NativeUFFBatchedForcefield::minimize)
    .def("minimizeDevice", &NativeUFFBatchedForcefield::minimizeDevice)
    .def("gpuId", &NativeUFFBatchedForcefield::gpuId);
}
