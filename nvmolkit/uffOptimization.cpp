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

#include <boost/python.hpp>

#include "bfgs_uff.h"
#include "boost_python_utils.h"
#include "device_result_python.h"

namespace bp = boost::python;

BOOST_PYTHON_MODULE(_uffOptimization) {
  bp::def(
    "UFFOptimizeMoleculesConfs",
    +[](const bp::list&                       molecules,
        int                                   maxIters,
        const bp::list&                       vdwThresholds,
        const bp::list&                       ignoreInterfragInteractions,
        const nvMolKit::BatchHardwareOptions& hardwareOptions) -> bp::list {
      auto       molsVec      = nvMolKit::extractMolecules(molecules);
      const int  numMols      = static_cast<int>(molsVec.size());
      const auto thresholdVec = nvMolKit::extractDoubleList(vdwThresholds, numMols, "vdwThreshold");
      const auto ignoreVec =
        nvMolKit::extractBoolList(ignoreInterfragInteractions, numMols, "ignoreInterfragInteractions");
      const auto result =
        nvMolKit::UFF::UFFOptimizeMoleculesConfsBfgs(molsVec, maxIters, thresholdVec, ignoreVec, hardwareOptions);
      return nvMolKit::vectorOfVectorsToList(result);
    },
    (bp::arg("molecules"),
     bp::arg("maxIters") = 1000,
     bp::arg("vdwThresholds"),
     bp::arg("ignoreInterfragInteractions"),
     bp::arg("hardwareOptions") = nvMolKit::BatchHardwareOptions()),
    "Optimize conformers for multiple molecules using UFF force field.\n"
    "\n"
    "Args:\n"
    "    molecules: List of RDKit molecules to optimize\n"
    "    maxIters: Maximum number of optimization iterations (default: 1000)\n"
    "    vdwThresholds: Per-molecule van der Waals thresholds\n"
    "    ignoreInterfragInteractions: Per-molecule interfragment interaction flags\n"
    "    hardwareOptions: BatchHardwareOptions object with hardware settings\n"
    "\n"
    "Returns:\n"
    "    List of lists of energies, where each inner list contains energies for conformers of one molecule");

  bp::def(
    "UFFOptimizeMoleculesConfsDevice",
    +[](const bp::list&                       molecules,
        int                                   maxIters,
        const bp::list&                       vdwThresholds,
        const bp::list&                       ignoreInterfragInteractions,
        const nvMolKit::BatchHardwareOptions& hardwareOptions,
        int                                   targetGpu) -> bp::object {
      auto       molsVec      = nvMolKit::extractMolecules(molecules);
      const int  numMols      = static_cast<int>(molsVec.size());
      const auto thresholdVec = nvMolKit::extractDoubleList(vdwThresholds, numMols, "vdwThreshold");
      const auto ignoreVec =
        nvMolKit::extractBoolList(ignoreInterfragInteractions, numMols, "ignoreInterfragInteractions");
      auto result = nvMolKit::UFF::UFFMinimizeMoleculesConfs(molsVec,
                                                             maxIters,
                                                             /*gradTol=*/1e-4,
                                                             thresholdVec,
                                                             ignoreVec,
                                                             /*constraints=*/{},
                                                             hardwareOptions,
                                                             nvMolKit::CoordinateOutput::DEVICE,
                                                             targetGpu);
      if (!result.device.has_value()) {
        throw std::runtime_error("UFFMinimizeMoleculesConfs(DEVICE) returned no device result");
      }
      return nvMolKit::buildOwningDevice3DResult(*result.device);
    },
    (bp::arg("molecules"),
     bp::arg("maxIters") = 1000,
     bp::arg("vdwThresholds"),
     bp::arg("ignoreInterfragInteractions"),
     bp::arg("hardwareOptions") = nvMolKit::BatchHardwareOptions(),
     bp::arg("targetGpu")       = -1),
    "Optimize conformers for multiple molecules using UFF force field, returning device-resident "
    "results.\n"
    "\n"
    "Returns:\n"
    "    A Device3DResult carrying optimized coordinates, energies, and convergence flags on GPU.");
}
