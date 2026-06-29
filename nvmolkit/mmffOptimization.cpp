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

#include <boost/python.hpp>
#include <stdexcept>
#include <string>

#include "nvmolkit/boost_python_utils.h"
#include "nvmolkit/device_result_python.h"
#include "nvmolkit/mmff_python_utils.h"
#include "src/minimizer/bfgs_types.h"
#include "src/minimizer/fire_minimizer.h"
#include "src/minimizer/mmff_minimize.h"

namespace bp = boost::python;

namespace {

enum class MinimizerKind {
  BFGS,
  FIRE
};

nvMolKit::BfgsBackend parseBfgsBackend(const std::string& name) {
  if (name == "BATCHED") {
    return nvMolKit::BfgsBackend::BATCHED;
  }
  if (name == "PER_MOL" || name == "PER_MOLECULE") {
    return nvMolKit::BfgsBackend::PER_MOLECULE;
  }
  if (name == "HYBRID") {
    return nvMolKit::BfgsBackend::HYBRID;
  }
  throw std::invalid_argument("Unknown BFGS backend '" + name + "'. Expected 'BATCHED', 'PER_MOL', or 'HYBRID'.");
}

nvMolKit::FireBackend parseFireBackend(const std::string& name) {
  if (name == "BATCHED") {
    return nvMolKit::FireBackend::BATCHED;
  }
  if (name == "PER_MOL" || name == "PER_MOLECULE") {
    return nvMolKit::FireBackend::PER_MOLECULE;
  }
  if (name == "HYBRID") {
    return nvMolKit::FireBackend::HYBRID;
  }
  throw std::invalid_argument("Unknown FIRE backend '" + name + "'. Expected 'BATCHED', 'PER_MOL', or 'HYBRID'.");
}

MinimizerKind parseMinimizerKind(const std::string& name) {
  if (name == "BFGS" || name == "bfgs") {
    return MinimizerKind::BFGS;
  }
  if (name == "FIRE" || name == "fire") {
    return MinimizerKind::FIRE;
  }
  throw std::invalid_argument("Unknown minimizerKind '" + name + "'. Expected 'BFGS' or 'FIRE'.");
}

}  // namespace

BOOST_PYTHON_MODULE(_mmffOptimization) {
  bp::def(
    "MMFFOptimizeMoleculesConfs",
    +[](const bp::list&                       molecules,
        int                                   maxIters,
        const bp::list&                       propertiesList,
        const nvMolKit::BatchHardwareOptions& hardwareOptions,
        const std::string&                    backend,
        const std::string&                    minimizerKind,
        const nvMolKit::FireOptions&          fireOptions) -> bp::list {
      auto       molsVec    = nvMolKit::extractMolecules(molecules);
      const auto properties = nvMolKit::extractMMFFPropertiesList(propertiesList, static_cast<int>(molsVec.size()));
      const auto kind       = parseMinimizerKind(minimizerKind);
      const auto result     = kind == MinimizerKind::FIRE ?
                                nvMolKit::MMFF::MMFFOptimizeMoleculesConfsFire(molsVec,
                                                                           maxIters,
                                                                           fireOptions,
                                                                           properties,
                                                                           hardwareOptions,
                                                                           parseFireBackend(backend)) :
                                nvMolKit::MMFF::MMFFOptimizeMoleculesConfsBfgs(molsVec,
                                                                           maxIters,
                                                                           properties,
                                                                           hardwareOptions,
                                                                           parseBfgsBackend(backend));
      return nvMolKit::vectorOfVectorsToList(result);
    },
    (bp::arg("molecules"),
     bp::arg("maxIters")        = 200,
     bp::arg("properties")      = bp::list(),
     bp::arg("hardwareOptions") = nvMolKit::BatchHardwareOptions(),
     bp::arg("backend")         = std::string("HYBRID"),
     bp::arg("minimizerKind")   = std::string("BFGS"),
     bp::arg("fireOptions")     = nvMolKit::FireOptions()),
    "Optimize conformers for multiple molecules using MMFF force field.\n"
    "\n"
    "Args:\n"
    "    molecules: List of RDKit molecules to optimize\n"
    "    maxIters: Maximum number of optimization iterations (default: 200)\n"
    "    properties: MMFFProperties-compatible object with forcefield settings\n"
    "    hardwareOptions: BatchHardwareOptions object with hardware settings (default: default options)\n"
    "    backend: Minimizer backend: 'BATCHED', 'PER_MOL', or 'HYBRID' (default: 'HYBRID')\n"
    "    minimizerKind: 'BFGS' or 'FIRE' (default: 'BFGS')\n"
    "    fireOptions: FireOptions used when minimizerKind='FIRE'\n"
    "\n"
    "Returns:\n"
    "    List of lists of energies, where each inner list contains energies for conformers of one molecule");

  bp::def(
    "MMFFOptimizeMoleculesConfsDevice",
    +[](const bp::list&                       molecules,
        int                                   maxIters,
        const bp::list&                       propertiesList,
        const nvMolKit::BatchHardwareOptions& hardwareOptions,
        int                                   targetGpu,
        const std::string&                    backend,
        const std::string&                    minimizerKind,
        const nvMolKit::FireOptions&          fireOptions) -> bp::object {
      auto       molsVec    = nvMolKit::extractMolecules(molecules);
      const auto properties = nvMolKit::extractMMFFPropertiesList(propertiesList, static_cast<int>(molsVec.size()));
      const auto kind       = parseMinimizerKind(minimizerKind);
      auto       result     = kind == MinimizerKind::FIRE ?
                                nvMolKit::MMFF::MMFFMinimizeMoleculesConfsFire(molsVec,
                                                                     maxIters,
                                                                     fireOptions,
                                                                     properties,
                                                                     /*constraints=*/{},
                                                                     hardwareOptions,
                                                                     parseFireBackend(backend),
                                                                     nvMolKit::CoordinateOutput::DEVICE,
                                                                     targetGpu) :
                                nvMolKit::MMFF::MMFFMinimizeMoleculesConfs(molsVec,
                                                                 maxIters,
                                                                 /*gradTol=*/1e-4,
                                                                 properties,
                                                                 /*constraints=*/{},
                                                                 hardwareOptions,
                                                                 parseBfgsBackend(backend),
                                                                 nvMolKit::CoordinateOutput::DEVICE,
                                                                 targetGpu);
      if (!result.device.has_value()) {
        throw std::runtime_error("MMFFMinimizeMoleculesConfs(DEVICE) returned no device result");
      }
      return nvMolKit::buildOwningDevice3DResult(*result.device);
    },
    (bp::arg("molecules"),
     bp::arg("maxIters"),
     bp::arg("properties"),
     bp::arg("hardwareOptions"),
     bp::arg("targetGpu"),
     bp::arg("backend")       = std::string("HYBRID"),
     bp::arg("minimizerKind") = std::string("BFGS"),
     bp::arg("fireOptions")   = nvMolKit::FireOptions()),
    "Optimize conformers for multiple molecules using MMFF force field, returning device-resident "
    "results.\n"
    "\n"
    "Returns:\n"
    "    A Device3DResult carrying optimized coordinates, energies, and convergence flags on GPU.");
}
