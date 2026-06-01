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

#include <GraphMol/ROMol.h>

#include <boost/python.hpp>
#include <boost/python/manage_new_object.hpp>

#include "nvmolkit/array_helpers.h"
#include "nvmolkit/boost_python_utils.h"
#include "src/tfd/tfd_gpu.h"
#include "src/utils/nvtx.h"

namespace {

using namespace boost::python;

std::vector<const RDKit::ROMol*> extractConstMolecules(const boost::python::list& mols) {
  auto nonConst = nvMolKit::extractMolecules(mols);
  return {nonConst.begin(), nonConst.end()};
}

nvMolKit::TFDComputeOptions buildOptions(bool               useWeights,
                                         const std::string& maxDev,
                                         int                symmRadius,
                                         bool               ignoreColinearBonds) {
  nvMolKit::TFDMaxDevMode maxDevMode;
  if (maxDev == "equal") {
    maxDevMode = nvMolKit::TFDMaxDevMode::Equal;
  } else if (maxDev == "spec") {
    maxDevMode = nvMolKit::TFDMaxDevMode::Spec;
  } else {
    throw std::invalid_argument("maxDev must be 'equal' or 'spec', got: " + maxDev);
  }

  return {
    .useWeights          = useWeights,
    .maxDevMode          = maxDevMode,
    .symmRadius          = symmRadius,
    .ignoreColinearBonds = ignoreColinearBonds,
  };
}

boost::python::object toOwnedPyArray(nvMolKit::PyArray* array) {
  using Converter = boost::python::manage_new_object::apply<nvMolKit::PyArray*>::type;
  return boost::python::object(boost::python::handle<>(Converter()(array)));
}

nvMolKit::TFDGpuGenerator& getGpuGenerator() {
  static nvMolKit::TFDGpuGenerator generator;
  return generator;
}

}  // namespace

BOOST_PYTHON_MODULE(_TFD) {
  // GPU path: returns GPU-resident buffer + metadata
  def(
    "GetTFDMatricesGpuBuffer",
    +[](const boost::python::list& mols,
        bool                       useWeights,
        const std::string&         maxDev,
        int                        symmRadius,
        bool                       ignoreColinearBonds) -> boost::python::object {
      auto molsVec = extractConstMolecules(mols);
      auto options = buildOptions(useWeights, maxDev, symmRadius, ignoreColinearBonds);

      auto gpuResult = getGpuGenerator().GetTFDMatricesGpuBuffer(molsVec, options);

      nvMolKit::ScopedNvtxRange range("GPU: C++ to Python tuple", nvMolKit::NvtxColor::kYellow);
      boost::python::list       outputStarts = nvMolKit::vectorToList(gpuResult.tfdOutputStarts);

      size_t totalSize = gpuResult.tfdValues.size();
      auto*  pyArray   = nvMolKit::makePyArray(gpuResult.tfdValues, boost::python::make_tuple(totalSize));

      return boost::python::make_tuple(toOwnedPyArray(pyArray), outputStarts);
    },
    (arg("mols"),
     arg("useWeights")          = true,
     arg("maxDev")              = "equal",
     arg("symmRadius")          = 2,
     arg("ignoreColinearBonds") = true));
}
