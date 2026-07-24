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
#include <boost/python/numpy.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "nvmolkit/array_helpers.h"
#include "src/similarity.h"
#include "src/utils/device.h"

namespace {

using ::nvMolKit::getSpanFromDictElems;

// Build a CPU similarity result.
template <typename ComputeFn>
boost::python::numpy::ndarray crossSimilarityCPUFromRawBuffers(const boost::python::dict& bitsOne,
                                                               const boost::python::dict& bitsTwo,
                                                               ComputeFn                  compute,
                                                               cudaStream_t               stream) {
  // Read the array shapes.
  boost::python::tuple shapeOne = boost::python::extract<boost::python::tuple>(bitsOne["shape"]);
  boost::python::tuple shapeTwo = boost::python::extract<boost::python::tuple>(bitsTwo["shape"]);

  const size_t numMolsOne = boost::python::extract<size_t>(shapeOne[0]);
  const size_t numMolsTwo = boost::python::extract<size_t>(shapeTwo[0]);

  const size_t nInts    = boost::python::extract<size_t>(shapeOne[1]);
  const size_t nIntsTwo = boost::python::extract<size_t>(shapeTwo[1]);
  if (nInts != nIntsTwo) {
    throw std::invalid_argument("Shape of bitsOne and bitsTwo dim 1 must be the same");
  }

  const int            fpSize       = static_cast<int>(nInts * 32);
  boost::python::tuple data1        = boost::python::extract<boost::python::tuple>(bitsOne["data"]);
  size_t               data1Pointer = boost::python::extract<std::size_t>(data1[0]);
  boost::python::tuple data2        = boost::python::extract<boost::python::tuple>(bitsTwo["data"]);
  size_t               data2Pointer = boost::python::extract<std::size_t>(data2[0]);

  auto span1 = getSpanFromDictElems<std::uint32_t>(reinterpret_cast<void*>(data1Pointer), shapeOne);
  auto span2 = getSpanFromDictElems<std::uint32_t>(reinterpret_cast<void*>(data2Pointer), shapeTwo);

  auto vec = compute(span1, span2, fpSize, stream);

  // Let the NumPy array own the result.
  auto  heapVec = std::make_unique<std::vector<double>>(std::move(vec));
  void* dataPtr = static_cast<void*>(heapVec->data());

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

// Build a GPU similarity result.
template <typename ComputeFn>
nvMolKit::PyArray* crossSimilarityGPUFromRawBuffers(const boost::python::dict& bitsOne,
                                                    const boost::python::dict& bitsTwo,
                                                    ComputeFn                  compute,
                                                    cudaStream_t               stream) {
  boost::python::tuple shapeOne = boost::python::extract<boost::python::tuple>(bitsOne["shape"]);
  boost::python::tuple shapeTwo = boost::python::extract<boost::python::tuple>(bitsTwo["shape"]);

  const size_t nInts    = boost::python::extract<size_t>(shapeOne[1]);
  const size_t nIntsTwo = boost::python::extract<size_t>(shapeTwo[1]);
  if (nInts != nIntsTwo) {
    throw std::invalid_argument("Shape of bitsOne and bitsTwo dim 1 must be the same");
  }

  const size_t numMolsOne = boost::python::extract<size_t>(shapeOne[0]);
  const size_t numMolsTwo = boost::python::extract<size_t>(shapeTwo[0]);
  const int    fpSize     = static_cast<int>(nInts * 32);

  boost::python::tuple dataOne        = boost::python::extract<boost::python::tuple>(bitsOne["data"]);
  boost::python::tuple dataTwo        = boost::python::extract<boost::python::tuple>(bitsTwo["data"]);
  const size_t         dataOnePointer = boost::python::extract<std::size_t>(dataOne[0]);
  const size_t         dataTwoPointer = boost::python::extract<std::size_t>(dataTwo[0]);

  auto spanOne = getSpanFromDictElems<std::uint32_t>(reinterpret_cast<void*>(dataOnePointer), shapeOne);
  auto spanTwo = getSpanFromDictElems<std::uint32_t>(reinterpret_cast<void*>(dataTwoPointer), shapeTwo);
  auto result  = compute(spanOne, spanTwo, fpSize, stream);
  return nvMolKit::makePyArray(result, boost::python::make_tuple(numMolsOne, numMolsTwo));
}

}  // namespace

BOOST_PYTHON_MODULE(_DataStructs) {
  boost::python::numpy::initialize();
  boost::python::def(
    "CrossTanimotoSimilarityRawBuffers",
    +[](const boost::python::dict& bitsOne, const boost::python::dict& bitsTwo, std::uintptr_t streamPtr) {
      auto streamOpt = nvMolKit::acquireExternalStream(streamPtr);
      if (!streamOpt) {
        throw std::invalid_argument("Invalid CUDA stream");
      }
      return crossSimilarityGPUFromRawBuffers(
        bitsOne,
        bitsTwo,
        [](const auto& a, const auto& b, int fpSize, cudaStream_t stream) {
          return nvMolKit::crossTanimotoSimilarityGpuResult(a, b, fpSize, stream);
        },
        *streamOpt);
    },
    boost::python::return_value_policy<boost::python::manage_new_object>(),
    (boost::python::arg("bitsOne"), boost::python::arg("bitsTwo"), boost::python::arg("stream") = 0));

  boost::python::def(
    "CrossCosineSimilarityRawBuffers",
    +[](const boost::python::dict& bitsOne, const boost::python::dict& bitsTwo, std::uintptr_t streamPtr) {
      auto streamOpt = nvMolKit::acquireExternalStream(streamPtr);
      if (!streamOpt) {
        throw std::invalid_argument("Invalid CUDA stream");
      }
      return crossSimilarityGPUFromRawBuffers(
        bitsOne,
        bitsTwo,
        [](const auto& a, const auto& b, int fpSize, cudaStream_t stream) {
          return nvMolKit::crossCosineSimilarityGpuResult(a, b, fpSize, stream);
        },
        *streamOpt);
    },
    boost::python::return_value_policy<boost::python::manage_new_object>(),
    (boost::python::arg("bitsOne"), boost::python::arg("bitsTwo"), boost::python::arg("stream") = 0));

  boost::python::def(
    "CrossTanimotoSimilarityCPURawBuffers",
    +[](const boost::python::dict& bitsOne, const boost::python::dict& bitsTwo, std::uintptr_t streamPtr) {
      auto streamOpt = nvMolKit::acquireExternalStream(streamPtr);
      if (!streamOpt) {
        throw std::invalid_argument("Invalid CUDA stream");
      }
      return crossSimilarityCPUFromRawBuffers(
        bitsOne,
        bitsTwo,
        [](const auto& a, const auto& b, int fpSize, cudaStream_t stream) {
          return nvMolKit::crossTanimotoSimilarityCPUResult(a, b, fpSize, {}, stream);
        },
        *streamOpt);
    },
    (boost::python::arg("bitsOne"), boost::python::arg("bitsTwo"), boost::python::arg("stream") = 0));

  boost::python::def(
    "CrossCosineSimilarityCPURawBuffers",
    +[](const boost::python::dict& bitsOne, const boost::python::dict& bitsTwo, std::uintptr_t streamPtr) {
      auto streamOpt = nvMolKit::acquireExternalStream(streamPtr);
      if (!streamOpt) {
        throw std::invalid_argument("Invalid CUDA stream");
      }
      return crossSimilarityCPUFromRawBuffers(
        bitsOne,
        bitsTwo,
        [](const auto& a, const auto& b, int fpSize, cudaStream_t stream) {
          return nvMolKit::crossCosineSimilarityCPUResult(a, b, fpSize, {}, stream);
        },
        *streamOpt);
    },
    (boost::python::arg("bitsOne"), boost::python::arg("bitsTwo"), boost::python::arg("stream") = 0));
}
