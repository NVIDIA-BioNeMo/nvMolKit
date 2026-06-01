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
#include <boost/python/numpy.hpp>
#include <boost/python/stl_iterator.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "nvmolkit/boost_python_utils.h"
#include "src/substruct/substruct_types.h"
#include "src/utils/nvtx.h"

// Forward declarations - avoid including CUDA headers
using cudaStream_t = struct CUstream_st*;

namespace nvMolKit {

void getSubstructMatches(const std::vector<const RDKit::ROMol*>& targets,
                         const std::vector<const RDKit::ROMol*>& queries,
                         SubstructSearchResults&                 results,
                         SubstructAlgorithm                      algorithm,
                         cudaStream_t                            stream,
                         const SubstructSearchConfig&            config);

void countSubstructMatches(const std::vector<const RDKit::ROMol*>& targets,
                           const std::vector<const RDKit::ROMol*>& queries,
                           std::vector<int>&                       counts,
                           SubstructAlgorithm                      algorithm,
                           cudaStream_t                            stream,
                           const SubstructSearchConfig&            config);

void hasSubstructMatch(const std::vector<const RDKit::ROMol*>& targets,
                       const std::vector<const RDKit::ROMol*>& queries,
                       HasSubstructMatchResults&               results,
                       SubstructAlgorithm                      algorithm,
                       cudaStream_t                            stream,
                       const SubstructSearchConfig&            config);

}  // namespace nvMolKit

namespace {

using namespace boost::python;

struct SubstructMatchesCSR {
  std::vector<int32_t> atomIndices;  // concatenated atom indices for all matches
  std::vector<int32_t> matchIndptr;  // offsets into atomIndices, length = numMatches + 1
  std::vector<int32_t> pairIndptr;   // offsets into matchIndptr (match index), length = numPairs + 1
  int                  numTargets = 0;
  int                  numQueries = 0;
};

template <typename T> std::vector<T> listFromIterable(const object& iterable) {
  std::vector<T> converted;
  if (PySequence_Check(iterable.ptr())) {
    Py_ssize_t n = PySequence_Size(iterable.ptr());
    converted.reserve(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
      object item(handle<>(borrowed(PySequence_GetItem(iterable.ptr(), i))));
      converted.push_back(extract<T>(item));
    }
  } else {
    stl_input_iterator<T> it(iterable), end;
    for (; it != end; ++it) {
      converted.push_back(*it);
    }
  }
  return converted;
}

list getGpuIdsPy(nvMolKit::SubstructSearchConfig& config) {
  return nvMolKit::vectorToList(config.gpuIds);
}

void setGpuIdsPy(nvMolKit::SubstructSearchConfig& config, const object& iterable) {
  config.gpuIds = listFromIterable<int>(iterable);
}

}  // namespace

BOOST_PYTHON_MODULE(_substructure) {
  numpy::initialize();

  class_<nvMolKit::SubstructSearchConfig>("SubstructSearchConfig")
    .def(init<>())
    .def_readwrite("batchSize", &nvMolKit::SubstructSearchConfig::batchSize)
    .def_readwrite("workerThreads", &nvMolKit::SubstructSearchConfig::workerThreads)
    .def_readwrite("preprocessingThreads", &nvMolKit::SubstructSearchConfig::preprocessingThreads)
    .def_readwrite("maxMatches", &nvMolKit::SubstructSearchConfig::maxMatches)
    .def_readwrite("uniquify", &nvMolKit::SubstructSearchConfig::uniquify)
    .add_property("gpuIds", &getGpuIdsPy, &setGpuIdsPy);

  def(
    "getSubstructMatches",
    +[](const list& targets, const list& queries, const nvMolKit::SubstructSearchConfig& config) {
      nvMolKit::ScopedNvtxRange        extractRange("Python: extract mol pointers", nvMolKit::NvtxColor::kYellow);
      std::vector<const RDKit::ROMol*> targetsVec;
      std::vector<const RDKit::ROMol*> queriesVec;

      targetsVec.reserve(len(targets));
      for (int i = 0; i < len(targets); ++i) {
        const RDKit::ROMol* mol = extract<const RDKit::ROMol*>(object(targets[i]));
        if (mol == nullptr) {
          throw std::invalid_argument("Invalid target molecule at index " + std::to_string(i));
        }
        targetsVec.push_back(mol);
      }

      queriesVec.reserve(len(queries));
      for (int i = 0; i < len(queries); ++i) {
        const RDKit::ROMol* mol = extract<const RDKit::ROMol*>(object(queries[i]));
        if (mol == nullptr) {
          throw std::invalid_argument("Invalid query molecule at index " + std::to_string(i));
        }
        queriesVec.push_back(mol);
      }
      extractRange.pop();

      nvMolKit::SubstructSearchResults results;
      nvMolKit::getSubstructMatches(targetsVec,
                                    queriesVec,
                                    results,
                                    nvMolKit::SubstructAlgorithm::GSI,
                                    nullptr,
                                    config);

      auto csrPtr        = std::make_unique<SubstructMatchesCSR>();
      csrPtr->numTargets = results.numTargets;
      csrPtr->numQueries = results.numQueries;

      nvMolKit::ScopedNvtxRange csrBuildRange("Python: build CSR buffers", nvMolKit::NvtxColor::kOrange);

      const int     numTargets = csrPtr->numTargets;
      const int     numQueries = csrPtr->numQueries;
      const int64_t numPairs   = static_cast<int64_t>(numTargets) * static_cast<int64_t>(numQueries);

      csrPtr->pairIndptr.resize(static_cast<size_t>(numPairs) + 1, 0);
      csrPtr->matchIndptr.clear();
      csrPtr->matchIndptr.reserve(1024);
      csrPtr->matchIndptr.push_back(0);

      // Populate CSR in deterministic pair order [t,q]
      int32_t matchCount = 0;
      int32_t atomCount  = 0;
      for (int t = 0; t < numTargets; ++t) {
        for (int q = 0; q < numQueries; ++q) {
          const int64_t pairIdx                            = static_cast<int64_t>(t) * numQueries + q;
          csrPtr->pairIndptr[static_cast<size_t>(pairIdx)] = matchCount;

          const auto& matches = results.getMatches(t, q);
          for (const auto& match : matches) {
            csrPtr->atomIndices.insert(csrPtr->atomIndices.end(), match.begin(), match.end());
            atomCount += static_cast<int32_t>(match.size());
            csrPtr->matchIndptr.push_back(atomCount);
            ++matchCount;
          }
        }
      }
      csrPtr->pairIndptr[static_cast<size_t>(numPairs)] = matchCount;
      csrBuildRange.pop();

      nvMolKit::ScopedNvtxRange csrWrapRange("Python: wrap CSR numpy arrays", nvMolKit::NvtxColor::kGreen);
      auto                      deleter = [](PyObject* cap) {
        auto* r = reinterpret_cast<SubstructMatchesCSR*>(PyCapsule_GetPointer(cap, "nvmolkit.substruct_csr"));
        delete r;
      };
      PyObject* cap = PyCapsule_New(static_cast<void*>(csrPtr.get()), "nvmolkit.substruct_csr", deleter);
      if (cap == nullptr) {
        throw std::runtime_error("Failed to create PyCapsule for getSubstructMatches CSR results");
      }
      object owner{handle<>(cap)};
      csrPtr.release();

      auto* csr = reinterpret_cast<SubstructMatchesCSR*>(PyCapsule_GetPointer(cap, "nvmolkit.substruct_csr"));

      const Py_intptr_t atomShape  = static_cast<Py_intptr_t>(csr->atomIndices.size());
      const Py_intptr_t matchShape = static_cast<Py_intptr_t>(csr->matchIndptr.size());
      const Py_intptr_t pairShape  = static_cast<Py_intptr_t>(csr->pairIndptr.size());
      const Py_intptr_t stride32   = static_cast<Py_intptr_t>(sizeof(int32_t));

      numpy::ndarray atomIndicesArr = numpy::from_data(csr->atomIndices.data(),
                                                       numpy::dtype::get_builtin<int32_t>(),
                                                       make_tuple(atomShape),
                                                       make_tuple(stride32),
                                                       owner);
      numpy::ndarray matchIndptrArr = numpy::from_data(csr->matchIndptr.data(),
                                                       numpy::dtype::get_builtin<int32_t>(),
                                                       make_tuple(matchShape),
                                                       make_tuple(stride32),
                                                       owner);
      numpy::ndarray pairIndptrArr  = numpy::from_data(csr->pairIndptr.data(),
                                                      numpy::dtype::get_builtin<int32_t>(),
                                                      make_tuple(pairShape),
                                                      make_tuple(stride32),
                                                      owner);

      return make_tuple(atomIndicesArr, matchIndptrArr, pairIndptrArr, make_tuple(csr->numTargets, csr->numQueries));
    },
    (arg("targets"), arg("queries"), arg("config") = nvMolKit::SubstructSearchConfig()));

  def(
    "countSubstructMatches",
    +[](const list& targets, const list& queries, const nvMolKit::SubstructSearchConfig& config) {
      nvMolKit::ScopedNvtxRange        extractRange("Python: extract mol pointers", nvMolKit::NvtxColor::kYellow);
      std::vector<const RDKit::ROMol*> targetsVec;
      std::vector<const RDKit::ROMol*> queriesVec;

      targetsVec.reserve(len(targets));
      for (int i = 0; i < len(targets); ++i) {
        const RDKit::ROMol* mol = extract<const RDKit::ROMol*>(object(targets[i]));
        if (mol == nullptr) {
          throw std::invalid_argument("Invalid target molecule at index " + std::to_string(i));
        }
        targetsVec.push_back(mol);
      }

      queriesVec.reserve(len(queries));
      for (int i = 0; i < len(queries); ++i) {
        const RDKit::ROMol* mol = extract<const RDKit::ROMol*>(object(queries[i]));
        if (mol == nullptr) {
          throw std::invalid_argument("Invalid query molecule at index " + std::to_string(i));
        }
        queriesVec.push_back(mol);
      }
      extractRange.pop();

      const int numTargets = static_cast<int>(targetsVec.size());
      const int numQueries = static_cast<int>(queriesVec.size());

      auto countsPtr = std::make_unique<std::vector<int>>();
      nvMolKit::countSubstructMatches(targetsVec,
                                      queriesVec,
                                      *countsPtr,
                                      nvMolKit::SubstructAlgorithm::GSI,
                                      nullptr,
                                      config);

      nvMolKit::ScopedNvtxRange wrapRange("Python: wrap numpy array", nvMolKit::NvtxColor::kGreen);
      int*                      dataPtr = countsPtr->data();

      auto deleter = [](PyObject* cap) {
        auto* r = reinterpret_cast<std::vector<int>*>(PyCapsule_GetPointer(cap, "nvmolkit.countsubstruct_results"));
        delete r;
      };
      PyObject* cap = PyCapsule_New(static_cast<void*>(countsPtr.get()), "nvmolkit.countsubstruct_results", deleter);
      if (cap == nullptr) {
        throw std::runtime_error("Failed to create PyCapsule for countSubstructMatches results");
      }
      object owner{handle<>(cap)};
      countsPtr.release();

      const Py_intptr_t shape_arr[2]   = {static_cast<Py_intptr_t>(numTargets), static_cast<Py_intptr_t>(numQueries)};
      const Py_intptr_t strides_arr[2] = {static_cast<Py_intptr_t>(numQueries * sizeof(int)),
                                          static_cast<Py_intptr_t>(sizeof(int))};

      return numpy::from_data(dataPtr,
                              numpy::dtype::get_builtin<int>(),
                              make_tuple(shape_arr[0], shape_arr[1]),
                              make_tuple(strides_arr[0], strides_arr[1]),
                              owner);
    },
    (arg("targets"), arg("queries"), arg("config") = nvMolKit::SubstructSearchConfig()));

  def(
    "hasSubstructMatch",
    +[](const list& targets, const list& queries, const nvMolKit::SubstructSearchConfig& config) {
      nvMolKit::ScopedNvtxRange        extractRange("Python: extract mol pointers", nvMolKit::NvtxColor::kYellow);
      std::vector<const RDKit::ROMol*> targetsVec;
      std::vector<const RDKit::ROMol*> queriesVec;

      targetsVec.reserve(len(targets));
      for (int i = 0; i < len(targets); ++i) {
        const RDKit::ROMol* mol = extract<const RDKit::ROMol*>(object(targets[i]));
        if (mol == nullptr) {
          throw std::invalid_argument("Invalid target molecule at index " + std::to_string(i));
        }
        targetsVec.push_back(mol);
      }

      queriesVec.reserve(len(queries));
      for (int i = 0; i < len(queries); ++i) {
        const RDKit::ROMol* mol = extract<const RDKit::ROMol*>(object(queries[i]));
        if (mol == nullptr) {
          throw std::invalid_argument("Invalid query molecule at index " + std::to_string(i));
        }
        queriesVec.push_back(mol);
      }
      extractRange.pop();

      auto resultsPtr = std::make_unique<nvMolKit::HasSubstructMatchResults>();
      nvMolKit::hasSubstructMatch(targetsVec,
                                  queriesVec,
                                  *resultsPtr,
                                  nvMolKit::SubstructAlgorithm::GSI,
                                  nullptr,
                                  config);

      nvMolKit::ScopedNvtxRange wrapRange("Python: wrap numpy array", nvMolKit::NvtxColor::kGreen);
      const int                 numTargets = resultsPtr->numTargets;
      const int                 numQueries = resultsPtr->numQueries;
      uint8_t*                  dataPtr    = resultsPtr->hasMatch.data();

      auto deleter = [](PyObject* cap) {
        auto* r = reinterpret_cast<nvMolKit::HasSubstructMatchResults*>(
          PyCapsule_GetPointer(cap, "nvmolkit.hassubstruct_results"));
        delete r;
      };
      PyObject* cap = PyCapsule_New(static_cast<void*>(resultsPtr.get()), "nvmolkit.hassubstruct_results", deleter);
      if (cap == nullptr) {
        throw std::runtime_error("Failed to create PyCapsule for hasSubstructMatch results");
      }
      object owner{handle<>(cap)};
      resultsPtr.release();

      const Py_intptr_t shape_arr[2]   = {static_cast<Py_intptr_t>(numTargets), static_cast<Py_intptr_t>(numQueries)};
      const Py_intptr_t strides_arr[2] = {static_cast<Py_intptr_t>(numQueries * sizeof(uint8_t)),
                                          static_cast<Py_intptr_t>(sizeof(uint8_t))};

      return numpy::from_data(dataPtr,
                              numpy::dtype::get_builtin<uint8_t>(),
                              make_tuple(shape_arr[0], shape_arr[1]),
                              make_tuple(strides_arr[0], strides_arr[1]),
                              owner);
    },
    (arg("targets"), arg("queries"), arg("config") = nvMolKit::SubstructSearchConfig()));
}
