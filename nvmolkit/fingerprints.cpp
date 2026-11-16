// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <mutex>

#include "nvmolkit/array_helpers.h"
#include "nvmolkit/boost_python_utils.h"
#include "src/morgan_fingerprint.h"
#include "src/utils/device.h"
#include "src/utils/nvtx.h"

namespace {

using namespace boost::python;

struct PythonMoleculeSequence {
  object                           owner;
  std::vector<const RDKit::ROMol*> molecules;
};

PythonMoleculeSequence convertMolecules(const object& mols) {
  if (!PySequence_Check(mols.ptr())) {
    PyErr_SetString(PyExc_TypeError, "mols must be a sequence of RDKit molecules");
    throw_error_already_set();
  }
  // A tuple copy keeps every molecule alive while the GIL is released, even if the caller's list changes.
  PyObject* sequence = PySequence_Tuple(mols.ptr());
  if (sequence == nullptr) {
    throw_error_already_set();
  }

  PythonMoleculeSequence result{object(handle<>(sequence)), {}};
  const Py_ssize_t       numMols = PySequence_Fast_GET_SIZE(sequence);
  PyObject* const*       items   = PySequence_Fast_ITEMS(sequence);
  result.molecules.reserve(static_cast<std::size_t>(numMols));
  for (Py_ssize_t i = 0; i < numMols; ++i) {
    const RDKit::ROMol* mol = extract<const RDKit::ROMol*>(items[i]);
    if (mol == nullptr) {
      throw std::invalid_argument("Invalid molecule at index " + std::to_string(i));
    }
    result.molecules.push_back(mol);
  }
  return result;
}

template <int nBits>
nvMolKit::PyArray* makePyArrayFromFlatBitVects(nvMolKit::AsyncDeviceVector<nvMolKit::FlatBitVect<nBits>>& deviceVect) {
  using dtype                = typename nvMolKit::FlatBitVect<nBits>::StorageType;
  const std::string dTypeStr = nvMolKit::getNumpyType<dtype>();

  // Make a 2D array, rows are the number of fingerprints, columns are the number of block_types
  const int nRows = deviceVect.size();
  const int nCols = nBits / (8 * sizeof(dtype));

  return nvMolKit::makePyArray(deviceVect, dTypeStr, boost::python::make_tuple(nRows, nCols));
}

template <int nBits>
nvMolKit::PyArray* getFingerprintsDevice(nvMolKit::MorganFingerprintGenerator&      generator,
                                         const std::vector<const RDKit::ROMol*>&    mols,
                                         cudaStream_t                               stream,
                                         const nvMolKit::FingerprintComputeOptions& computeOptions) {
  auto fingerprints = [&]() {
    nvMolKit::ScopedNvtxRange        range("MorganFPBindingNativeCompute", nvMolKit::NvtxColor::kOrange);
    const nvMolKit::ScopedGilRelease release;
    // The generator reuses its GPU buffers across calls, so calls must not overlap once the GIL is released.
    static std::mutex                generatorMutex;
    const std::lock_guard            lock(generatorMutex);
    return generator.GetFingerprintsGpuBuffer<nBits>(mols, stream, computeOptions);
  }();

  nvMolKit::ScopedNvtxRange range("MorganFPBindingOutputWrapping", nvMolKit::NvtxColor::kCyan);
  return makePyArrayFromFlatBitVects<nBits>(fingerprints);
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
          const boost::python::object&          mols,
          int                                   numThreads,
          std::uintptr_t                        streamPtr) {
        auto convertedMols = [&]() {
          nvMolKit::ScopedNvtxRange range("MorganFPBindingInputConversion", nvMolKit::NvtxColor::kYellow);
          return convertMolecules(mols);
        }();

        nvMolKit::FingerprintComputeOptions computeOptions;
        computeOptions.backend       = nvMolKit::FingerprintComputeBackend::GPU;
        computeOptions.numCpuThreads = numThreads;
        auto streamOpt               = [&]() {
          nvMolKit::ScopedNvtxRange range("MorganFPBindingStreamAcquisition", nvMolKit::NvtxColor::kBlue);
          return nvMolKit::acquireExternalStream(streamPtr);
        }();
        if (!streamOpt) {
          throw std::invalid_argument("Invalid CUDA stream");
        }
        auto        stream  = *streamOpt;
        const auto& options = selfref.GetOptions();
        switch (options.fpSize) {
          case 128: {
            return getFingerprintsDevice<128>(selfref, convertedMols.molecules, stream, computeOptions);
          }
          case 256: {
            return getFingerprintsDevice<256>(selfref, convertedMols.molecules, stream, computeOptions);
          }
          case 512: {
            return getFingerprintsDevice<512>(selfref, convertedMols.molecules, stream, computeOptions);
          }
          case 1024: {
            return getFingerprintsDevice<1024>(selfref, convertedMols.molecules, stream, computeOptions);
          }
          case 2048: {
            return getFingerprintsDevice<2048>(selfref, convertedMols.molecules, stream, computeOptions);
          }
          default:
            throw std::invalid_argument("Invalid fpSize: " + std::to_string(options.fpSize) +
                                        ". Supported values are 128, 256, 512, 1024, 2048");
        }
      },
      return_value_policy<manage_new_object>(),
      (boost::python::arg("mols"), boost::python::arg("num_threads") = 0, boost::python::arg("stream") = 0));
}
