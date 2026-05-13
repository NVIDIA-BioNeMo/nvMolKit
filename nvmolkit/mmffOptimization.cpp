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

#include "bfgs_mmff.h"
#include "boost_python_utils.h"
#include "device_result_python.h"
#include "mmff_python_utils.h"

namespace bp = boost::python;

BOOST_PYTHON_MODULE(_mmffOptimization) {
  bp::def(
    "MMFFOptimizeMoleculesConfs",
    +[](const bp::list&                       molecules,
        int                                   maxIters,
        const bp::list&                       propertiesList,
        const nvMolKit::BatchHardwareOptions& hardwareOptions) -> bp::list {
      auto       molsVec    = nvMolKit::extractMolecules(molecules);
      const auto properties = nvMolKit::extractMMFFPropertiesList(propertiesList, static_cast<int>(molsVec.size()));
      const auto result =
        nvMolKit::MMFF::MMFFOptimizeMoleculesConfsBfgs(molsVec, maxIters, properties, hardwareOptions);
      return nvMolKit::vectorOfVectorsToList(result);
    },
    (bp::arg("molecules"),
     bp::arg("maxIters")        = 200,
     bp::arg("properties")      = bp::list(),
     bp::arg("hardwareOptions") = nvMolKit::BatchHardwareOptions()),
    "Optimize conformers for multiple molecules using MMFF force field.\n"
    "\n"
    "Args:\n"
    "    molecules: List of RDKit molecules to optimize\n"
    "    maxIters: Maximum number of optimization iterations (default: 200)\n"
    "    properties: MMFFProperties-compatible object with forcefield settings\n"
    "    hardwareOptions: BatchHardwareOptions object with hardware settings (default: default options)\n"
    "\n"
    "Returns:\n"
    "    List of lists of energies, where each inner list contains energies for conformers of one molecule");

  bp::def(
    "MMFFOptimizeMoleculesConfsDevice",
    +[](const bp::list&                       molecules,
        int                                   maxIters,
        const bp::list&                       propertiesList,
        const nvMolKit::BatchHardwareOptions& hardwareOptions,
        int                                   targetGpu) -> bp::object {
      auto       molsVec    = nvMolKit::extractMolecules(molecules);
      const auto properties = nvMolKit::extractMMFFPropertiesList(propertiesList, static_cast<int>(molsVec.size()));
      auto       result     = nvMolKit::MMFF::MMFFMinimizeMoleculesConfs(molsVec,
                                                               maxIters,
                                                               /*gradTol=*/1e-4,
                                                               properties,
                                                               /*constraints=*/{},
                                                               hardwareOptions,
                                                               nvMolKit::BfgsBackend::HYBRID,
                                                               nvMolKit::CoordinateOutput::DEVICE,
                                                               targetGpu);
      if (!result.device.has_value()) {
        throw std::runtime_error("MMFFMinimizeMoleculesConfs(DEVICE) returned no device result");
      }
      return nvMolKit::buildOwningDevice3DResult(*result.device);
    },
    (bp::arg("molecules"),
     bp::arg("maxIters")        = 200,
     bp::arg("properties")      = bp::list(),
     bp::arg("hardwareOptions") = nvMolKit::BatchHardwareOptions(),
     bp::arg("targetGpu")       = -1),
    "Optimize conformers for multiple molecules using MMFF force field, returning device-resident "
    "results.\n"
    "\n"
    "Returns:\n"
    "    A Device3DResult carrying optimized coordinates, energies, and convergence flags on GPU.");
}
