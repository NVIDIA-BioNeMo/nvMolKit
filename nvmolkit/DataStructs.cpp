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

#include <algorithm>
#include <boost/python.hpp>
#include <boost/python/manage_new_object.hpp>
#include <boost/python/numpy.hpp>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nvmolkit/array_helpers.h"
#include "src/fingerprint_search.h"
#include "src/similarity.h"
#include "src/utils/cuda_error_check.h"
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

boost::python::object toOwnedPyArray(nvMolKit::PyArray* array) {
  using Converter = boost::python::manage_new_object::apply<nvMolKit::PyArray*>::type;
  return boost::python::object(boost::python::handle<>(Converter()(array)));
}

cudaStream_t requireStream(const std::uintptr_t streamPtr) {
  auto streamOpt = nvMolKit::acquireExternalStream(streamPtr);
  if (!streamOpt) {
    throw std::invalid_argument("Invalid CUDA stream");
  }
  return *streamOpt;
}

// The cudaCheckError macros call checkReturnCode unqualified.
using nvMolKit::checkReturnCode;

// Python view of FingerprintSearchIndex. Fingerprints arrive as __cuda_array_interface__ dicts of shape (n, words).
class PyFingerprintSearchIndex {
 public:
  PyFingerprintSearchIndex(const boost::python::dict& bits, const std::string& metric, const std::uintptr_t streamPtr)
      : buildStream_(requireStream(streamPtr)),
        index_(spanFromDict(bits), rowsFromDict(bits), wordsFromDict(bits), parseMetric(metric), buildStream_) {}

  PyFingerprintSearchIndex(const PyFingerprintSearchIndex&)            = delete;
  PyFingerprintSearchIndex& operator=(const PyFingerprintSearchIndex&) = delete;

  // The index frees its buffers on the build stream. Order those frees after every search on other streams, which may
  // still be running when Python drops the index.
  ~PyFingerprintSearchIndex() {
    for (const auto& [stream, event] : searchEvents_) {
      cudaCheckErrorNoThrow(cudaStreamWaitEvent(buildStream_, event.event()));
    }
  }

  boost::python::tuple search(const boost::python::dict& queries,
                              const int                  k,
                              const double               threshold,
                              const bool                 prune,
                              const std::uintptr_t       streamPtr) {
    if (wordsFromDict(queries) != index_.numWords()) {
      throw std::invalid_argument("Queries must have the same number of fingerprint words as the index");
    }
    const int    numQueries = rowsFromDict(queries);
    cudaStream_t stream     = requireStream(streamPtr);
    auto         result     = index_.search(spanFromDict(queries), numQueries, k, threshold, prune, stream);
    recordSearch(stream);
    return toTuple(result, numQueries, k);
  }

  boost::python::tuple searchSelf(const int            k,
                                  const double         threshold,
                                  const bool           prune,
                                  const std::uintptr_t streamPtr) {
    cudaStream_t stream = requireStream(streamPtr);
    auto         result = index_.searchSelf(k, threshold, prune, stream);
    recordSearch(stream);
    return toTuple(result, index_.size(), k);
  }

  [[nodiscard]] int size() const { return index_.size(); }
  [[nodiscard]] int numWords() const { return index_.numWords(); }

 private:
  // Re-recording a stream's event captures all of that stream's searches so far.
  void recordSearch(cudaStream_t stream) {
    if (stream == buildStream_) {
      return;
    }
    auto it = std::find_if(searchEvents_.begin(), searchEvents_.end(), [stream](const auto& entry) {
      return entry.first == stream;
    });
    if (it == searchEvents_.end()) {
      searchEvents_.emplace_back(stream, nvMolKit::ScopedCudaEvent());
      it = std::prev(searchEvents_.end());
    }
    cudaCheckError(cudaEventRecord(it->second.event(), stream));
  }

  static nvMolKit::FingerprintSimilarityMetric parseMetric(const std::string& metric) {
    if (metric == "tanimoto") {
      return nvMolKit::FingerprintSimilarityMetric::Tanimoto;
    }
    if (metric == "cosine") {
      return nvMolKit::FingerprintSimilarityMetric::Cosine;
    }
    throw std::invalid_argument("metric must be 'tanimoto' or 'cosine'");
  }

  static int rowsFromDict(const boost::python::dict& bits) {
    const boost::python::tuple shape   = boost::python::extract<boost::python::tuple>(bits["shape"]);
    const std::size_t          numRows = boost::python::extract<std::size_t>(shape[0]);
    if (numRows > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("At most 2^31 - 1 fingerprints are supported");
    }
    return static_cast<int>(numRows);
  }

  static int wordsFromDict(const boost::python::dict& bits) {
    const boost::python::tuple shape = boost::python::extract<boost::python::tuple>(bits["shape"]);
    return static_cast<int>(boost::python::extract<std::size_t>(shape[1]));
  }

  static cuda::std::span<const std::uint32_t> spanFromDict(const boost::python::dict& bits) {
    boost::python::tuple shape   = boost::python::extract<boost::python::tuple>(bits["shape"]);
    boost::python::tuple data    = boost::python::extract<boost::python::tuple>(bits["data"]);
    const std::size_t    pointer = boost::python::extract<std::size_t>(data[0]);
    return getSpanFromDictElems<std::uint32_t>(reinterpret_cast<void*>(pointer), shape);
  }

  static boost::python::tuple toTuple(nvMolKit::FingerprintSearchResult& result, const int numRows, const int k) {
    auto scores  = nvMolKit::makePyArray(result.scores, boost::python::make_tuple(numRows, k));
    // getNumpyType spells int64 as "l8", which torch does not parse.
    auto indices = nvMolKit::makePyArray(result.indices, "i8", boost::python::make_tuple(numRows, k));
    return boost::python::make_tuple(toOwnedPyArray(scores), toOwnedPyArray(indices));
  }

  cudaStream_t                                                    buildStream_;
  std::vector<std::pair<cudaStream_t, nvMolKit::ScopedCudaEvent>> searchEvents_;
  nvMolKit::FingerprintSearchIndex                                index_;
};

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

  boost::python::class_<PyFingerprintSearchIndex, boost::noncopyable>(
    "FingerprintSearchIndex",
    boost::python::init<const boost::python::dict&, const std::string&, std::uintptr_t>(
      (boost::python::arg("bits"), boost::python::arg("metric"), boost::python::arg("stream") = 0)))
    .def("search",
         &PyFingerprintSearchIndex::search,
         (boost::python::arg("queries"),
          boost::python::arg("k"),
          boost::python::arg("threshold"),
          boost::python::arg("prune"),
          boost::python::arg("stream") = 0))
    .def("searchSelf",
         &PyFingerprintSearchIndex::searchSelf,
         (boost::python::arg("k"),
          boost::python::arg("threshold"),
          boost::python::arg("prune"),
          boost::python::arg("stream") = 0))
    .def("size", &PyFingerprintSearchIndex::size)
    .def("numWords", &PyFingerprintSearchIndex::numWords);
}
