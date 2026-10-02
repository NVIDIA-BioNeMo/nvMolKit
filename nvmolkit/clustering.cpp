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

#include <GraphMol/ROMol.h>

#include <boost/python.hpp>
#include <boost/python/manage_new_object.hpp>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "nvmolkit/array_helpers.h"
#include "nvmolkit/boost_python_utils.h"
#include "src/aap.h"
#include "src/butina.h"
#include "src/diversity_pickers.h"
#include "src/utils/device.h"

namespace {

boost::python::object toOwnedPyArray(nvMolKit::PyArray* array) {
  using Converter = boost::python::manage_new_object::apply<nvMolKit::PyArray*>::type;
  return boost::python::object(boost::python::handle<>(Converter()(array)));
}

boost::python::object wrapButinaResult(nvMolKit::ButinaResult& result, const int numItems, const bool returnCentroids) {
  auto clusterArray = nvMolKit::makePyArray(result.clusterIds, boost::python::make_tuple(numItems));
  if (!returnCentroids) {
    return toOwnedPyArray(clusterArray);
  }

  auto centroidArray = nvMolKit::makePyArray(result.centroids, boost::python::make_tuple(result.numClusters));
  return boost::python::make_tuple(toOwnedPyArray(clusterArray), toOwnedPyArray(centroidArray));
}

boost::python::tuple wrapClusteringResult(const nvMolKit::ClusteringResult& result,
                                          const bool                        deviceOutput,
                                          cudaStream_t                      stream) {
  if (!deviceOutput) {
    return boost::python::make_tuple(nvMolKit::vectorToList(result.clusterIds),
                                     nvMolKit::vectorToList(result.centroids),
                                     nvMolKit::vectorToList(result.clusterSizes));
  }

  nvMolKit::AsyncDeviceVector<int>          clusterIds(result.clusterIds.size(), stream);
  nvMolKit::AsyncDeviceVector<int>          centroids(result.centroids.size(), stream);
  nvMolKit::AsyncDeviceVector<std::int64_t> clusterSizes(result.clusterSizes.size(), stream);
  clusterIds.copyFromHost(result.clusterIds);
  centroids.copyFromHost(result.centroids);
  clusterSizes.copyFromHost(result.clusterSizes);
  // The source vectors are pageable and go out of scope when this binding
  // returns, so complete these small staging copies before releasing them.
  nvMolKit::checkReturnCode<true>(cudaStreamSynchronize(stream), __FILE__, __LINE__);

  return boost::python::make_tuple(
    toOwnedPyArray(nvMolKit::makePyArray(clusterIds)),
    toOwnedPyArray(nvMolKit::makePyArray(centroids)),
    toOwnedPyArray(nvMolKit::makePyArray(clusterSizes, "i8", boost::python::make_tuple(clusterSizes.size()))));
}

boost::python::tuple wrapPickerResult(nvMolKit::PickerResult& result) {
  auto indices = nvMolKit::makePyArray(result.indices);
  return boost::python::make_tuple(toOwnedPyArray(indices), result.lastDistance);
}

cudaStream_t requireStream(const std::uintptr_t streamPtr) {
  auto streamOpt = nvMolKit::acquireExternalStream(streamPtr);
  if (!streamOpt) {
    throw std::invalid_argument("Invalid CUDA stream");
  }
  return *streamOpt;
}

struct MatrixInput {
  std::variant<cuda::std::span<const float>, cuda::std::span<const double>> distances;
  int                                                                       numItems;
};

//! Parses a float32 or float64 square matrix from its CUDA array interface.
MatrixInput parseDistanceMatrix(const boost::python::dict& matrix) {
  boost::python::tuple shape       = boost::python::extract<boost::python::tuple>(matrix["shape"]);
  boost::python::tuple data        = boost::python::extract<boost::python::tuple>(matrix["data"]);
  const std::size_t    dataPointer = boost::python::extract<std::size_t>(data[0]);
  const std::string    typestr     = boost::python::extract<std::string>(matrix["typestr"]);
  const int            numItems    = boost::python::extract<int>(shape[0]);
  auto*                pointer     = reinterpret_cast<void*>(dataPointer);
  if (typestr == "<f4") {
    return {nvMolKit::getSpanFromDictElems<float>(pointer, shape), numItems};
  }
  if (typestr == "<f8") {
    return {nvMolKit::getSpanFromDictElems<double>(pointer, shape), numItems};
  }
  throw std::invalid_argument("distance_matrix must have dtype float32 or float64");
}

void translateAapInvalidMolecules(const nvMolKit::AapInvalidMoleculesError& error) {
  boost::python::dict indices;
  indices["none"]                      = nvMolKit::vectorToList(error.none);
  indices["empty"]                     = nvMolKit::vectorToList(error.empty);
  indices["too_many_atoms"]            = nvMolKit::vectorToList(error.tooManyAtoms);
  indices["unsupported_bond"]          = nvMolKit::vectorToList(error.unsupportedBonds);
  const boost::python::tuple arguments = boost::python::make_tuple(std::string(error.what()), indices);
  PyErr_SetObject(PyExc_ValueError, arguments.ptr());
}

struct FingerprintInput {
  cuda::std::span<const std::uint32_t> fingerprints;
  int                                  numItems;
  int                                  numWords;
};

FingerprintInput parseFingerprints(const boost::python::dict& fingerprints) {
  boost::python::tuple shape       = boost::python::extract<boost::python::tuple>(fingerprints["shape"]);
  boost::python::tuple data        = boost::python::extract<boost::python::tuple>(fingerprints["data"]);
  const std::size_t    dataPointer = boost::python::extract<std::size_t>(data[0]);
  return {nvMolKit::getSpanFromDictElems<std::uint32_t>(reinterpret_cast<void*>(dataPointer), shape),
          boost::python::extract<int>(shape[0]),
          boost::python::extract<int>(shape[1])};
}

//! Converts molecules for AAP, keeping None as nullptr so AAP can report every invalid input together.
std::vector<const RDKit::ROMol*> moleculePointers(const boost::python::list& molecules) {
  const auto                       count = boost::python::len(molecules);
  std::vector<const RDKit::ROMol*> result;
  result.reserve(count);
  for (int index = 0; index < count; ++index) {
    result.push_back(boost::python::extract<RDKit::ROMol*>(boost::python::object(molecules[index]))());
  }
  return result;
}

std::vector<int> extractIndices(const boost::python::object& values) {
  std::vector<int> result;
  const auto       count = boost::python::len(values);
  result.reserve(count);
  for (int index = 0; index < count; ++index) {
    result.push_back(boost::python::extract<int>(values[index]));
  }
  return result;
}

nvMolKit::FingerprintSimilarityMetric parseFingerprintMetric(const std::string& metric) {
  if (metric == "tanimoto") {
    return nvMolKit::FingerprintSimilarityMetric::Tanimoto;
  }
  if (metric == "cosine") {
    return nvMolKit::FingerprintSimilarityMetric::Cosine;
  }
  throw std::invalid_argument("metric must be one of ['tanimoto', 'cosine']");
}

}  // namespace

BOOST_PYTHON_MODULE(_clustering) {
  boost::python::register_exception_translator<nvMolKit::AapInvalidMoleculesError>(translateAapInvalidMolecules);

  boost::python::def(
    "aap_similarity",
    +[](const RDKit::ROMol& left,
        const RDKit::ROMol& right,
        const int           maxPathLength,
        const int           histogramBins,
        const int           sinkhornIterations,
        const float         sinkhornTemperature,
        std::uintptr_t      streamPtr) {
      auto streamOpt = nvMolKit::acquireExternalStream(streamPtr);
      if (!streamOpt) {
        throw std::invalid_argument("Invalid CUDA stream");
      }
      const nvMolKit::AapOptions options{maxPathLength, histogramBins, sinkhornIterations, sinkhornTemperature};
      return nvMolKit::aapSimilarityGpu(left, right, options, *streamOpt);
    },
    (boost::python::arg("left"),
     boost::python::arg("right"),
     boost::python::arg("max_path_length")      = 7,
     boost::python::arg("histogram_bins")       = 2048,
     boost::python::arg("sinkhorn_iterations")  = 8,
     boost::python::arg("sinkhorn_temperature") = 0.104F,
     boost::python::arg("stream")               = 0));

  boost::python::def(
    "aap_leader",
    +[](const boost::python::list&   molecules,
        const double                 cutoff,
        const int                    pickSize,
        const boost::python::object& firstPicks,
        const int                    maxPathLength,
        const int                    histogramBins,
        const int                    sinkhornIterations,
        const float                  sinkhornTemperature,
        const std::uintptr_t         streamPtr) {
      const auto                 stream = requireStream(streamPtr);
      const nvMolKit::AapOptions options{maxPathLength, histogramBins, sinkhornIterations, sinkhornTemperature};
      auto                       result =
        nvMolKit::aapLeader(moleculePointers(molecules), cutoff, options, pickSize, extractIndices(firstPicks), stream);
      return wrapPickerResult(result);
    },
    (boost::python::arg("molecules"),
     boost::python::arg("cutoff"),
     boost::python::arg("pick_size"),
     boost::python::arg("first_picks"),
     boost::python::arg("max_path_length"),
     boost::python::arg("histogram_bins"),
     boost::python::arg("sinkhorn_iterations"),
     boost::python::arg("sinkhorn_temperature"),
     boost::python::arg("stream")));

  boost::python::def(
    "aap_maxmin",
    +[](const boost::python::list&   molecules,
        const int                    pickSize,
        const boost::python::object& firstPicks,
        const int                    seed,
        const double                 threshold,
        const int                    maxPathLength,
        const int                    histogramBins,
        const int                    sinkhornIterations,
        const float                  sinkhornTemperature,
        const std::uintptr_t         streamPtr) {
      const auto                 stream = requireStream(streamPtr);
      const nvMolKit::AapOptions options{maxPathLength, histogramBins, sinkhornIterations, sinkhornTemperature};
      auto                       result = nvMolKit::aapMaxMin(moleculePointers(molecules),
                                        pickSize,
                                        options,
                                        extractIndices(firstPicks),
                                        seed,
                                        threshold,
                                        stream);
      return wrapPickerResult(result);
    },
    (boost::python::arg("molecules"),
     boost::python::arg("pick_size"),
     boost::python::arg("first_picks"),
     boost::python::arg("seed"),
     boost::python::arg("threshold"),
     boost::python::arg("max_path_length"),
     boost::python::arg("histogram_bins"),
     boost::python::arg("sinkhorn_iterations"),
     boost::python::arg("sinkhorn_temperature"),
     boost::python::arg("stream")));

  boost::python::def(
    "aap_dise",
    +[](const boost::python::list& molecules,
        const double               cutoff,
        const bool                 nearestAssignment,
        const int                  maxPathLength,
        const int                  histogramBins,
        const int                  sinkhornIterations,
        const float                sinkhornTemperature,
        const bool                 deviceOutput,
        const std::uintptr_t       streamPtr) {
      const auto                 stream = requireStream(streamPtr);
      const nvMolKit::AapOptions options{maxPathLength, histogramBins, sinkhornIterations, sinkhornTemperature};
      return wrapClusteringResult(
        nvMolKit::aapDise(moleculePointers(molecules), cutoff, options, nearestAssignment, stream),
        deviceOutput,
        stream);
    },
    (boost::python::arg("molecules"),
     boost::python::arg("cutoff"),
     boost::python::arg("nearest_assignment"),
     boost::python::arg("max_path_length"),
     boost::python::arg("histogram_bins"),
     boost::python::arg("sinkhorn_iterations"),
     boost::python::arg("sinkhorn_temperature"),
     boost::python::arg("device_output"),
     boost::python::arg("stream")));

  boost::python::def(
    "leader",
    +[](const boost::python::dict&   distanceMatrix,
        const double                 cutoff,
        const int                    pickSize,
        const boost::python::object& firstPicks,
        const std::uintptr_t         streamPtr) {
      const auto input  = parseDistanceMatrix(distanceMatrix);
      auto       result = std::visit(
        [&](const auto distances) {
          return nvMolKit::leaderFromDistanceMatrix(distances,
                                                    input.numItems,
                                                    cutoff,
                                                    pickSize,
                                                    extractIndices(firstPicks),
                                                    requireStream(streamPtr));
        },
        input.distances);
      return wrapPickerResult(result);
    },
    (boost::python::arg("distance_matrix"),
     boost::python::arg("cutoff"),
     boost::python::arg("pick_size"),
     boost::python::arg("first_picks"),
     boost::python::arg("stream")));

  boost::python::def(
    "fused_leader",
    +[](const boost::python::dict&   fingerprints,
        const double                 cutoff,
        const std::string&           metric,
        const int                    pickSize,
        const boost::python::object& firstPicks,
        const std::uintptr_t         streamPtr) {
      const auto input  = parseFingerprints(fingerprints);
      auto       result = nvMolKit::fusedLeaderGpu(input.fingerprints,
                                             input.numItems,
                                             input.numWords,
                                             cutoff,
                                             parseFingerprintMetric(metric),
                                             pickSize,
                                             extractIndices(firstPicks),
                                             requireStream(streamPtr));
      return wrapPickerResult(result);
    },
    (boost::python::arg("fingerprints"),
     boost::python::arg("cutoff"),
     boost::python::arg("metric"),
     boost::python::arg("pick_size"),
     boost::python::arg("first_picks"),
     boost::python::arg("stream")));

  boost::python::def(
    "maxmin",
    +[](const boost::python::dict&   distanceMatrix,
        const int                    pickSize,
        const boost::python::object& firstPicks,
        const int                    seed,
        const double                 threshold,
        const std::uintptr_t         streamPtr) {
      const auto input  = parseDistanceMatrix(distanceMatrix);
      auto       result = std::visit(
        [&](const auto distances) {
          return nvMolKit::maxMinFromDistanceMatrix(distances,
                                                    input.numItems,
                                                    pickSize,
                                                    extractIndices(firstPicks),
                                                    seed,
                                                    threshold,
                                                    requireStream(streamPtr));
        },
        input.distances);
      return wrapPickerResult(result);
    },
    (boost::python::arg("distance_matrix"),
     boost::python::arg("pick_size"),
     boost::python::arg("first_picks"),
     boost::python::arg("seed"),
     boost::python::arg("threshold"),
     boost::python::arg("stream")));

  boost::python::def(
    "fused_maxmin",
    +[](const boost::python::dict&   fingerprints,
        const int                    pickSize,
        const std::string&           metric,
        const boost::python::object& firstPicks,
        const int                    seed,
        const double                 threshold,
        const std::uintptr_t         streamPtr) {
      const auto input  = parseFingerprints(fingerprints);
      auto       result = nvMolKit::fusedMaxMinGpu(input.fingerprints,
                                             input.numItems,
                                             input.numWords,
                                             pickSize,
                                             parseFingerprintMetric(metric),
                                             extractIndices(firstPicks),
                                             seed,
                                             threshold,
                                             requireStream(streamPtr));
      return wrapPickerResult(result);
    },
    (boost::python::arg("fingerprints"),
     boost::python::arg("pick_size"),
     boost::python::arg("metric"),
     boost::python::arg("first_picks"),
     boost::python::arg("seed"),
     boost::python::arg("threshold"),
     boost::python::arg("stream")));

  boost::python::def(
    "dise",
    +[](const boost::python::dict& distanceMatrix,
        const double               cutoff,
        const bool                 nearestAssignment,
        const bool                 deviceOutput,
        const std::uintptr_t       streamPtr) {
      const auto stream = requireStream(streamPtr);
      const auto input  = parseDistanceMatrix(distanceMatrix);
      auto       result = std::visit(
        [&](const auto distances) {
          return nvMolKit::diseFromDistanceMatrix(distances, input.numItems, cutoff, nearestAssignment, stream);
        },
        input.distances);
      return wrapClusteringResult(result, deviceOutput, stream);
    },
    (boost::python::arg("distance_matrix"),
     boost::python::arg("cutoff"),
     boost::python::arg("nearest_assignment"),
     boost::python::arg("device_output"),
     boost::python::arg("stream")));

  boost::python::def(
    "fused_dise",
    +[](const boost::python::dict& fingerprints,
        const double               cutoff,
        const std::string&         metric,
        const bool                 nearestAssignment,
        const bool                 deviceOutput,
        const std::uintptr_t       streamPtr) {
      const auto stream = requireStream(streamPtr);
      const auto input  = parseFingerprints(fingerprints);
      return wrapClusteringResult(nvMolKit::fusedDiseGpu(input.fingerprints,
                                                         input.numItems,
                                                         input.numWords,
                                                         cutoff,
                                                         parseFingerprintMetric(metric),
                                                         nearestAssignment,
                                                         stream),
                                  deviceOutput,
                                  stream);
    },
    (boost::python::arg("fingerprints"),
     boost::python::arg("cutoff"),
     boost::python::arg("metric"),
     boost::python::arg("nearest_assignment"),
     boost::python::arg("device_output"),
     boost::python::arg("stream")));

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
      const auto           stream  = *streamOpt;
      // Read the matrix shape.
      boost::python::tuple shape   = boost::python::extract<boost::python::tuple>(distanceMatrix["shape"]);
      const int            matDim1 = boost::python::extract<int>(shape[0]);

      boost::python::tuple data        = boost::python::extract<boost::python::tuple>(distanceMatrix["data"]);
      const std::size_t    dataPointer = boost::python::extract<std::size_t>(data[0]);
      const auto matSpan = nvMolKit::getSpanFromDictElems<double>(reinterpret_cast<void*>(dataPointer), shape);
      auto       result  = nvMolKit::butinaFromDistanceMatrix(matSpan,
                                                       matDim1,
                                                       cutoff,
                                                       neighborlistMaxSize,
                                                       returnCentroids,
                                                       reordering,
                                                       stream);
      return wrapButinaResult(result, matDim1, returnCentroids);
    },
    (boost::python::arg("distance_matrix"),
     boost::python::arg("cutoff"),
     boost::python::arg("neighborlist_max_size") = 64,
     boost::python::arg("return_centroids")      = false,
     boost::python::arg("reordering")            = true,
     boost::python::arg("stream")                = 0));

  boost::python::def(
    "fused_butina",
    +[](const boost::python::dict& fingerprints,
        const double               cutoff,
        const bool                 returnCentroids,
        const std::string&         metric,
        std::uintptr_t             streamPtr) -> boost::python::object {
      auto streamOpt = nvMolKit::acquireExternalStream(streamPtr);
      if (!streamOpt) {
        throw std::invalid_argument("Invalid CUDA stream");
      }
      const auto           stream = *streamOpt;
      boost::python::tuple shape  = boost::python::extract<boost::python::tuple>(fingerprints["shape"]);
      if (len(shape) != 2) {
        throw std::invalid_argument("fingerprints must be a 2D matrix");
      }
      const int            n           = boost::python::extract<int>(shape[0]);
      const int            numWords    = boost::python::extract<int>(shape[1]);
      boost::python::tuple data        = boost::python::extract<boost::python::tuple>(fingerprints["data"]);
      const std::size_t    dataPointer = boost::python::extract<std::size_t>(data[0]);
      const auto span = nvMolKit::getSpanFromDictElems<std::uint32_t>(reinterpret_cast<void*>(dataPointer), shape);

      nvMolKit::FingerprintSimilarityMetric parsedMetric;
      if (metric == "tanimoto") {
        parsedMetric = nvMolKit::FingerprintSimilarityMetric::Tanimoto;
      } else if (metric == "cosine") {
        parsedMetric = nvMolKit::FingerprintSimilarityMetric::Cosine;
      } else {
        throw std::invalid_argument("metric must be one of ['tanimoto', 'cosine']");
      }

      auto result = nvMolKit::fusedButinaGpu(span, n, numWords, cutoff, parsedMetric, returnCentroids, stream);
      return wrapButinaResult(result, n, returnCentroids);
    },
    (boost::python::arg("fingerprints"),
     boost::python::arg("cutoff"),
     boost::python::arg("return_centroids") = false,
     boost::python::arg("metric")           = "tanimoto",
     boost::python::arg("stream")           = 0));
}
