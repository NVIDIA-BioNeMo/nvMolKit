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

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <memory>
#include <stdexcept>

#include "nvmolkit/array_helpers.h"
#include "src/similarity.h"
#include "src/utils/device.h"

namespace {

using ::nvMolKit::getSpanFromDictElems;

// Shared CPU-result wrapper for cosine and tanimoto similarity
template <typename ComputeFn>
boost::python::numpy::ndarray crossSimilarityCPUFromRawBuffers(const boost::python::dict& bitsOne,
                                                               const boost::python::dict& bitsTwo,
                                                               ComputeFn                  compute) {
  // Extract boost::python::tuple from dict['shape']
  boost::python::tuple shapeOne = boost::python::extract<boost::python::tuple>(bitsOne["shape"]);
  boost::python::tuple shapeTwo = boost::python::extract<boost::python::tuple>(bitsTwo["shape"]);

  const size_t numMolsOne = boost::python::extract<size_t>(shapeOne[0]);
  const size_t numMolsTwo = boost::python::extract<size_t>(shapeTwo[0]);

  const size_t nInts    = boost::python::extract<size_t>(shapeOne[1]);
  const size_t nIntsTwo = boost::python::extract<size_t>(shapeTwo[1]);
  if (nInts != nIntsTwo) {
    throw std::invalid_argument("Shape of bitsOne and bitsTwo dim 1 must be the same");
  }

  const size_t         nBytes       = sizeof(std::uint32_t);
  const size_t         fpSize       = nInts * 8 * nBytes;
  boost::python::tuple data1        = boost::python::extract<boost::python::tuple>(bitsOne["data"]);
  size_t               data1Pointer = boost::python::extract<std::size_t>(data1[0]);
  boost::python::tuple data2        = boost::python::extract<boost::python::tuple>(bitsTwo["data"]);
  size_t               data2Pointer = boost::python::extract<std::size_t>(data2[0]);

  auto span1 = getSpanFromDictElems<std::uint32_t>(reinterpret_cast<void*>(data1Pointer), shapeOne);
  auto span2 = getSpanFromDictElems<std::uint32_t>(reinterpret_cast<void*>(data2Pointer), shapeTwo);

  auto vec = compute(span1, span2, fpSize);

  // Move vector to heap and tie lifetime to a capsule owner
  auto  heapVec = std::make_unique<std::vector<double>>(std::move(vec));
  void* dataPtr = static_cast<void*>(heapVec->data());

  // Capsule destructor to free heapVec
  auto deleter = [](PyObject* capsule) {
    void* ptr = PyCapsule_GetPointer(capsule, "nvmolkit.double_vector");
    auto* v   = reinterpret_cast<std::vector<double>*>(ptr);
    delete v;
  };
  PyObject* cap = PyCapsule_New(static_cast<void*>(heapVec.get()), "nvmolkit.double_vector", deleter);
  if (cap == nullptr) {
    throw std::runtime_error("Failed to create PyCapsule for CPU similarity result");
  }
  boost::python::object owner{boost::python::handle<>(cap)};
  heapVec.release();

  const Py_intptr_t shape_arr[2]   = {static_cast<Py_intptr_t>(numMolsOne), static_cast<Py_intptr_t>(numMolsTwo)};
  const Py_intptr_t strides_arr[2] = {static_cast<Py_intptr_t>(numMolsTwo * sizeof(double)),
                                      static_cast<Py_intptr_t>(sizeof(double))};

  auto arr = boost::python::numpy::from_data(dataPtr,
                                             boost::python::numpy::dtype::get_builtin<double>(),
                                             boost::python::make_tuple(shape_arr[0], shape_arr[1]),
                                             boost::python::make_tuple(strides_arr[0], strides_arr[1]),
                                             owner);
  return arr;
}

}  // namespace

BOOST_PYTHON_MODULE(_DataStructs) {
  boost::python::numpy::initialize();
  boost::python::def(
    "CrossTanimotoSimilarityRawBuffers",
    +[](const boost::python::dict& bitsOne, const boost::python::dict& bitsTwo, std::uintptr_t streamPtr) {
      // Extract boost::python::tuple from dict['shape']
      boost::python::tuple shapeOne = boost::python::extract<boost::python::tuple>(bitsOne["shape"]);
      boost::python::tuple shapeTwo = boost::python::extract<boost::python::tuple>(bitsTwo["shape"]);

      const size_t nInts    = boost::python::extract<size_t>(shapeOne[1]);
      const size_t nIntsTwo = boost::python::extract<size_t>(shapeTwo[1]);
      if (nInts != nIntsTwo) {
        throw std::invalid_argument("Shape of bitsOne and bitsTwo dim 1 must be the same");
      }
      const size_t numMolsOne = boost::python::extract<size_t>(shapeOne[0]);
      const size_t numMolsTwo = boost::python::extract<size_t>(shapeTwo[0]);

      // Extract the datatype string, and check the number of bytes
      const size_t nBytes = sizeof(std::uint32_t);

      const size_t         fpSize       = nInts * 8 * nBytes;
      boost::python::tuple data1        = boost::python::extract<boost::python::tuple>(bitsOne["data"]);
      size_t               data1Pointer = boost::python::extract<std::size_t>(data1[0]);
      boost::python::tuple data2        = boost::python::extract<boost::python::tuple>(bitsTwo["data"]);
      size_t               data2Pointer = boost::python::extract<std::size_t>(data2[0]);

      auto streamOpt = nvMolKit::acquireExternalStream(streamPtr);
      if (!streamOpt) {
        throw std::invalid_argument("Invalid CUDA stream");
      }
      auto span1 = getSpanFromDictElems<std::uint32_t>(reinterpret_cast<void*>(data1Pointer), shapeOne);
      auto span2 = getSpanFromDictElems<std::uint32_t>(reinterpret_cast<void*>(data2Pointer), shapeTwo);
      auto array = nvMolKit::crossTanimotoSimilarityGpuResult(span1, span2, fpSize, *streamOpt);
      assert(array.size() == numMolsOne * numMolsTwo);
      return makePyArray(array, boost::python::make_tuple(numMolsOne, numMolsTwo));
    },
    boost::python::return_value_policy<boost::python::manage_new_object>(),
    (boost::python::arg("bitsOne"), boost::python::arg("bitsTwo"), boost::python::arg("stream") = 0));

  // --------------------------------
  // Cosine similarity binding functions
  // --------------------------------

  boost::python::def(
    "CrossCosineSimilarityRawBuffers",
    +[](const boost::python::dict& bitsOne, const boost::python::dict& bitsTwo, std::uintptr_t streamPtr) {
      // Extract boost::python::tuple from dict['shape']
      boost::python::tuple shapeOne = boost::python::extract<boost::python::tuple>(bitsOne["shape"]);
      boost::python::tuple shapeTwo = boost::python::extract<boost::python::tuple>(bitsTwo["shape"]);

      const size_t nInts    = boost::python::extract<size_t>(shapeOne[1]);
      const size_t nIntsTwo = boost::python::extract<size_t>(shapeTwo[1]);
      if (nInts != nIntsTwo) {
        throw std::invalid_argument("Shape of bitsOne and bitsTwo dim 1 must be the same");
      }
      const size_t numMolsOne = boost::python::extract<size_t>(shapeOne[0]);
      const size_t numMolsTwo = boost::python::extract<size_t>(shapeTwo[0]);

      // Extract the datatype string, and check the number of bytes
      const size_t nBytes = sizeof(std::uint32_t);

      const size_t         fpSize       = nInts * 8 * nBytes;
      boost::python::tuple data1        = boost::python::extract<boost::python::tuple>(bitsOne["data"]);
      const size_t         data1Pointer = boost::python::extract<std::size_t>(data1[0]);
      boost::python::tuple data2        = boost::python::extract<boost::python::tuple>(bitsTwo["data"]);
      const size_t         data2Pointer = boost::python::extract<std::size_t>(data2[0]);

      auto streamOpt = nvMolKit::acquireExternalStream(streamPtr);
      if (!streamOpt) {
        throw std::invalid_argument("Invalid CUDA stream");
      }
      auto span1 = getSpanFromDictElems<std::uint32_t>(reinterpret_cast<void*>(data1Pointer), shapeOne);
      auto span2 = getSpanFromDictElems<std::uint32_t>(reinterpret_cast<void*>(data2Pointer), shapeTwo);
      auto array = nvMolKit::crossCosineSimilarityGpuResult(span1, span2, fpSize, *streamOpt);
      assert(array.size() == numMolsOne * numMolsTwo);
      return makePyArray(array, boost::python::make_tuple(numMolsOne, numMolsTwo));
    },
    boost::python::return_value_policy<boost::python::manage_new_object>(),
    (boost::python::arg("bitsOne"), boost::python::arg("bitsTwo"), boost::python::arg("stream") = 0));

  // --------------------------------
  // CPU-result similarity binding functions (no options exposed; nullopt by default)
  // --------------------------------

  boost::python::def(
    "CrossTanimotoSimilarityCPURawBuffers",
    +[](const boost::python::dict& bitsOne, const boost::python::dict& bitsTwo) {
      return crossSimilarityCPUFromRawBuffers(bitsOne, bitsTwo, [](const auto& a, const auto& b, int fpSize) {
        return nvMolKit::crossTanimotoSimilarityCPUResult(a, b, fpSize);
      });
    });

  boost::python::def(
    "CrossCosineSimilarityCPURawBuffers",
    +[](const boost::python::dict& bitsOne, const boost::python::dict& bitsTwo) {
      return crossSimilarityCPUFromRawBuffers(bitsOne, bitsTwo, [](const auto& a, const auto& b, int fpSize) {
        return nvMolKit::crossCosineSimilarityCPUResult(a, b, fpSize);
      });
    });
}
