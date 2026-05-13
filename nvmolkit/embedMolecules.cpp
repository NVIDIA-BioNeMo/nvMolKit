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

#include <GraphMol/DistGeomHelpers/Embedder.h>

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>

#include "boost_python_utils.h"
#include "device_result_python.h"
#include "etkdg.h"

namespace bp = boost::python;

namespace {

bp::list getGpuIdsPy(nvMolKit::BatchHardwareOptions& opts) {
  return nvMolKit::vectorToList(opts.gpuIds);
}

void setGpuIds(nvMolKit::BatchHardwareOptions& opts, const bp::object& iterable) {
  std::vector<int> converted;
  if (PySequence_Check(iterable.ptr())) {
    Py_ssize_t n = PySequence_Size(iterable.ptr());
    converted.reserve(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
      bp::object item(bp::handle<>(bp::borrowed(PySequence_GetItem(iterable.ptr(), i))));
      converted.push_back(bp::extract<int>(item));
    }
  } else {
    bp::stl_input_iterator<int> it(iterable), end;
    for (; it != end; ++it) {
      converted.push_back(*it);
    }
  }
  opts.gpuIds.swap(converted);
}

}  // namespace

BOOST_PYTHON_MODULE(_embedMolecules) {
  bp::class_<nvMolKit::BatchHardwareOptions>("BatchHardwareOptions")
    .def(bp::init<>())
    .def_readwrite("preprocessingThreads", &nvMolKit::BatchHardwareOptions::preprocessingThreads)
    .def_readwrite("batchSize", &nvMolKit::BatchHardwareOptions::batchSize)
    .def_readwrite("batchesPerGpu", &nvMolKit::BatchHardwareOptions::batchesPerGpu)
    .add_property("gpuIds", &getGpuIdsPy, &setGpuIds);

  bp::def(
    "EmbedMolecules",
    +[](const bp::list&                             molecules,
        const RDKit::DGeomHelpers::EmbedParameters& params,
        int                                         confsPerMolecule,
        int                                         maxIterations,
        const nvMolKit::BatchHardwareOptions&       hardwareOptions) {
      auto molsVec = nvMolKit::extractMolecules(molecules);
      nvMolKit::embedMolecules(molsVec,
                               params,
                               confsPerMolecule,
                               maxIterations,
                               /*debugMode=*/false,
                               /*failures=*/nullptr,
                               hardwareOptions);
    },
    (bp::arg("molecules"),
     bp::arg("params"),
     bp::arg("confsPerMolecule") = 1,
     bp::arg("maxIterations")    = -1,
     bp::arg("hardwareOptions")  = nvMolKit::BatchHardwareOptions()),
    "Embed multiple molecules with multiple conformers using ETKDG.\n"
    "\n"
    "Args:\n"
    "    molecules: List of RDKit molecules to embed\n"
    "    params: RDKit EmbedParameters object with embedding settings\n"
    "    confsPerMolecule: Number of conformers to generate per molecule (default: 1)\n"
    "    maxIterations: Maximum iterations, -1 for auto (default: -1)\n"
    "    hardwareOptions: BatchHardwareOptions object with hardware settings (default: default options)\n"
    "\n"
    "Returns:\n"
    "    None (molecules are modified in-place with generated conformers)");

  bp::def(
    "EmbedMoleculesDevice",
    +[](const bp::list&                             molecules,
        const RDKit::DGeomHelpers::EmbedParameters& params,
        int                                         confsPerMolecule,
        int                                         maxIterations,
        const nvMolKit::BatchHardwareOptions&       hardwareOptions,
        int                                         targetGpu) -> bp::object {
      auto molsVec = nvMolKit::extractMolecules(molecules);
      auto result  = nvMolKit::embedMolecules(molsVec,
                                             params,
                                             confsPerMolecule,
                                             maxIterations,
                                             /*debugMode=*/false,
                                             /*failures=*/nullptr,
                                             hardwareOptions,
                                             nvMolKit::BfgsBackend::HYBRID,
                                             nvMolKit::CoordinateOutput::DEVICE,
                                             targetGpu);
      if (!result.has_value()) {
        throw std::runtime_error("embedMolecules(DEVICE) returned no device result");
      }
      return nvMolKit::buildOwningDevice3DResult(*result);
    },
    (bp::arg("molecules"),
     bp::arg("params"),
     bp::arg("confsPerMolecule") = 1,
     bp::arg("maxIterations")    = -1,
     bp::arg("hardwareOptions")  = nvMolKit::BatchHardwareOptions(),
     bp::arg("targetGpu")        = -1),
    "Embed multiple molecules with multiple conformers using ETKDG, returning device-resident "
    "coordinates.\n"
    "\n"
    "Returns:\n"
    "    A Device3DResult holding generated conformer coordinates on GPU. RDKit conformers are "
    "NOT modified.");
}
