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
#include <boost/python/manage_new_object.hpp>
#include <memory>

#include "nvmolkit/array_helpers.h"
#include "src/butina.h"
#include "src/utils/device.h"

namespace {

boost::python::object toOwnedPyArray(nvMolKit::PyArray* array) {
  using Converter = boost::python::manage_new_object::apply<nvMolKit::PyArray*>::type;
  return boost::python::object(boost::python::handle<>(Converter()(array)));
}

}  // namespace

BOOST_PYTHON_MODULE(_clustering) {
  boost::python::def(
    "butina",
    +[](const boost::python::dict& distanceMatrix,
        const double               cutoff,
        const int                  neighborlistMaxSize,
        const bool                 returnCentroids,
        const bool                 reordering,
        std::uintptr_t             streamPtr) -> boost::python::object {
      auto streamOpt = nvMolKit::acquireExternalStream(streamPtr);
      if (!streamOpt) {
        throw std::invalid_argument("Invalid CUDA stream");
      }
      auto                             stream  = *streamOpt;
      // Extract boost::python::tuple from dict['shape']
      boost::python::tuple             shape   = boost::python::extract<boost::python::tuple>(distanceMatrix["shape"]);
      const size_t                     matDim1 = boost::python::extract<size_t>(shape[0]);
      nvMolKit::AsyncDeviceVector<int> clusterIds(matDim1, stream);
      nvMolKit::AsyncDeviceVector<int> centroids;

      boost::python::tuple data        = boost::python::extract<boost::python::tuple>(distanceMatrix["data"]);
      const size_t         dataPointer = boost::python::extract<std::size_t>(data[0]);
      const auto matSpan = nvMolKit::getSpanFromDictElems<double>(reinterpret_cast<void*>(dataPointer), shape);
      if (returnCentroids) {
        centroids.resize(matDim1);
        centroids.setStream(stream);
      }
      const auto centroidSpan = returnCentroids ? toSpan(centroids) : cuda::std::span<int>{};
      const int  numClusters =
        nvMolKit::butinaGpu(matSpan, toSpan(clusterIds), cutoff, neighborlistMaxSize, centroidSpan, reordering, stream);
      if (returnCentroids) {
        auto clusterArray  = nvMolKit::makePyArray(clusterIds, boost::python::make_tuple(matDim1));
        auto centroidArray = nvMolKit::makePyArray(centroids, boost::python::make_tuple(numClusters));
        return boost::python::make_tuple(toOwnedPyArray(clusterArray), toOwnedPyArray(centroidArray));
      }

      return toOwnedPyArray(nvMolKit::makePyArray(clusterIds, boost::python::make_tuple(matDim1)));
    },
    (boost::python::arg("distance_matrix"),
     boost::python::arg("cutoff"),
     boost::python::arg("neighborlist_max_size") = 64,
     boost::python::arg("return_centroids")      = false,
     boost::python::arg("reordering")            = true,
     boost::python::arg("stream")                = 0));
};
