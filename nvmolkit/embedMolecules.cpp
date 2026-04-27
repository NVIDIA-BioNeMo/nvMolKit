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

#include "nvmolkit/boost_python_utils.h"
#include "src/etkdg.h"

static boost::python::list getGpuIdsPy(nvMolKit::BatchHardwareOptions& opts) {
  return nvMolKit::vectorToList(opts.gpuIds);
}

static void setGpuIds(nvMolKit::BatchHardwareOptions& opts, const boost::python::object& iterable) {
  std::vector<int> converted;
  using namespace boost::python;
  // Prefer fast sequence path
  if (PySequence_Check(iterable.ptr())) {
    Py_ssize_t n = PySequence_Size(iterable.ptr());
    converted.reserve(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
      object item(handle<>(borrowed(PySequence_GetItem(iterable.ptr(), i))));
      converted.push_back(extract<int>(item));
    }
  } else {
    // Fallback: try generic iterable
    stl_input_iterator<int> it(iterable), end;
    for (; it != end; ++it) {
      converted.push_back(*it);
    }
  }
  opts.gpuIds.swap(converted);
}

BOOST_PYTHON_MODULE(_embedMolecules) {
  // Expose BatchHardwareOptions struct to Python
  boost::python::class_<nvMolKit::BatchHardwareOptions>("BatchHardwareOptions")
    .def(boost::python::init<>())
    .def_readwrite("preprocessingThreads", &nvMolKit::BatchHardwareOptions::preprocessingThreads)
    .def_readwrite("batchSize", &nvMolKit::BatchHardwareOptions::batchSize)
    .def_readwrite("batchesPerGpu", &nvMolKit::BatchHardwareOptions::batchesPerGpu)
    .add_property("gpuIds", &getGpuIdsPy, &setGpuIds);

  boost::python::def(
    "EmbedMolecules",
    +[](const boost::python::list&                  molecules,
        const RDKit::DGeomHelpers::EmbedParameters& params,
        int                                         confsPerMolecule,
        int                                         maxIterations,
        const nvMolKit::BatchHardwareOptions&       hardwareOptions) {
      auto molsVec = nvMolKit::extractMolecules(molecules);

      // Call the C++ function with nullptr for failures
      nvMolKit::embedMolecules(molsVec,
                               params,
                               confsPerMolecule,
                               maxIterations,
                               false,    // debugMode = false
                               nullptr,  // failures = nullptr
                               hardwareOptions);
    },
    (boost::python::arg("molecules"),
     boost::python::arg("params"),
     boost::python::arg("confsPerMolecule") = 1,
     boost::python::arg("maxIterations")    = -1,
     boost::python::arg("hardwareOptions")  = nvMolKit::BatchHardwareOptions()),
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
}
