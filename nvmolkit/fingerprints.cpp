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

#include <DataStructs/ExplicitBitVect.h>
#include <GraphMol/ROMol.h>

#include <boost/python.hpp>

#include "nvmolkit/array_helpers.h"
#include "src/morgan_fingerprint.h"
#include "src/utils/device.h"

namespace {

using namespace boost::python;

template <int nBits>
nvMolKit::PyArray* makePyArrayFromFlatBitVects(nvMolKit::AsyncDeviceVector<nvMolKit::FlatBitVect<nBits>>& deviceVect) {
  using dtype                = typename nvMolKit::FlatBitVect<nBits>::StorageType;
  const std::string dTypeStr = nvMolKit::getNumpyType<dtype>();

  // Make a 2D array, rows are the number of fingerprints, columns are the number of block_types
  const int nRows = deviceVect.size();
  const int nCols = nBits / (8 * sizeof(dtype));

  return nvMolKit::makePyArray(deviceVect, dTypeStr, boost::python::make_tuple(nRows, nCols));
}

}  // namespace

BOOST_PYTHON_MODULE(_Fingerprints) {
  class_<nvMolKit::MorganFingerprintGenerator, boost::noncopyable>(
    "MorganFingerprintGenerator",
    init<const std::uint32_t, const std::uint32_t>((boost::python::arg("radius"), boost::python::arg("fpSize"))))
    .def(
      "GetFingerprint",
      +[](nvMolKit::MorganFingerprintGenerator& selfref, const RDKit::ROMol& mol) {
        return selfref.GetFingerprint(mol).release();
      },
      return_value_policy<manage_new_object>())
    .def(
      "GetFingerprintsDevice",
      +[](nvMolKit::MorganFingerprintGenerator& selfref,
          boost::python::list&                  mols,
          int                                   numThreads,
          std::uintptr_t                        streamPtr) {
        std::vector<const RDKit::ROMol*> molsVec;
        molsVec.reserve(len(mols));
        for (int i = 0; i < len(mols); i++) {
          molsVec.push_back(boost::python::extract<const RDKit::ROMol*>(boost::python::object(mols[i])));
          if (molsVec.back() == nullptr) {
            throw std::invalid_argument("Invalid molecule at index " + std::to_string(i));
          }
        }
        nvMolKit::FingerprintComputeOptions computeOptions;
        computeOptions.backend       = nvMolKit::FingerprintComputeBackend::GPU;
        computeOptions.numCpuThreads = numThreads;
        auto streamOpt               = nvMolKit::acquireExternalStream(streamPtr);
        if (!streamOpt) {
          throw std::invalid_argument("Invalid CUDA stream");
        }
        auto        stream  = *streamOpt;
        const auto& options = selfref.GetOptions();
        switch (options.fpSize) {
          case 128: {
            auto array = selfref.GetFingerprintsGpuBuffer<128>(molsVec, stream, computeOptions);
            return makePyArrayFromFlatBitVects<128>(array);
          }
          case 256: {
            auto array = selfref.GetFingerprintsGpuBuffer<256>(molsVec, stream, computeOptions);
            return makePyArrayFromFlatBitVects<256>(array);
          }
          case 512: {
            auto array = selfref.GetFingerprintsGpuBuffer<512>(molsVec, stream, computeOptions);
            return makePyArrayFromFlatBitVects<512>(array);
          }
          case 1024: {
            auto array = selfref.GetFingerprintsGpuBuffer<1024>(molsVec, stream, computeOptions);
            return makePyArrayFromFlatBitVects<1024>(array);
          }
          case 2048: {
            auto array = selfref.GetFingerprintsGpuBuffer<2048>(molsVec, stream, computeOptions);
            return makePyArrayFromFlatBitVects<2048>(array);
          }
          default:
            throw std::invalid_argument("Invalid fpSize: " + std::to_string(options.fpSize) +
                                        ". Supported values are 128, 256, 512, 1024, 2048");
        }
      },
      return_value_policy<manage_new_object>(),
      (boost::python::arg("mols"), boost::python::arg("num_threads") = 0, boost::python::arg("stream") = 0));
}
