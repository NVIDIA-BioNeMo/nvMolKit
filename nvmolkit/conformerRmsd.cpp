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

#include <GraphMol/GraphMol.h>

#include <boost/python.hpp>
#include <boost/python/manage_new_object.hpp>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "nvmolkit/array_helpers.h"
#include "src/conformer_rmsd_mol.h"
#include "src/utils/device.h"

namespace {

boost::python::object toOwnedPyArray(nvMolKit::PyArray* array) {
  using Converter = boost::python::manage_new_object::apply<nvMolKit::PyArray*>::type;
  return boost::python::object(boost::python::handle<>(Converter()(array)));
}

}  // namespace

BOOST_PYTHON_MODULE(_conformerRmsd) {
  boost::python::def(
    "GetConformerRMSMatrixBatch",
    +[](boost::python::list& mols, const bool prealigned, std::uintptr_t streamPtr) -> boost::python::object {
      auto streamOpt = nvMolKit::acquireExternalStream(streamPtr);
      if (!streamOpt) {
        throw std::invalid_argument("Invalid CUDA stream");
      }

      const int numMols = boost::python::len(mols);
      if (numMols == 0) {
        return boost::python::list();
      }

      std::vector<const RDKit::ROMol*> molsVec(numMols);
      for (int i = 0; i < numMols; ++i) {
        molsVec[i] = boost::python::extract<const RDKit::ROMol*>(boost::python::object(mols[i]));
        if (!molsVec[i]) {
          throw std::invalid_argument("Invalid molecule at index " + std::to_string(i));
        }
      }

      auto buffers = nvMolKit::conformerRmsdBatchMatrixMol(molsVec, prealigned, *streamOpt);

      boost::python::list results;
      for (int m = 0; m < numMols; ++m) {
        const int nc       = molsVec[m]->getNumConformers();
        const int numPairs = nc >= 2 ? nc * (nc - 1) / 2 : 0;
        results.append(toOwnedPyArray(nvMolKit::makePyArray(buffers[m], boost::python::make_tuple(numPairs))));
      }
      return results;
    },
    (boost::python::arg("mols"), boost::python::arg("prealigned") = false, boost::python::arg("stream") = 0));

  boost::python::def(
    "GetConformerRMSMatrix",
    +[](RDKit::ROMol& mol, const bool prealigned, std::uintptr_t streamPtr) -> boost::python::object {
      auto streamOpt = nvMolKit::acquireExternalStream(streamPtr);
      if (!streamOpt) {
        throw std::invalid_argument("Invalid CUDA stream");
      }

      const int     numConfs = mol.getNumConformers();
      const int64_t numPairs = numConfs >= 2 ? static_cast<int64_t>(numConfs) * (numConfs - 1) / 2 : 0;
      auto          buffer   = nvMolKit::conformerRmsdMatrixMol(mol, prealigned, *streamOpt);
      return toOwnedPyArray(nvMolKit::makePyArray(buffer, boost::python::make_tuple(numPairs)));
    },
    (boost::python::arg("mol"), boost::python::arg("prealigned") = false, boost::python::arg("stream") = 0));
}
