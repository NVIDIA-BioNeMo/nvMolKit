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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/mcs/mcs_search.h"
#include "src/utils/nvtx.h"

namespace {

using namespace boost::python;

struct MCSResultBuffers {
  std::vector<unsigned int> numAtoms;
  std::vector<unsigned int> numBonds;
  std::vector<std::uint8_t> canceled;

  std::vector<std::int32_t> atomMapping;
  std::vector<std::int32_t> atomMappingIndptr;
  std::vector<std::int32_t> bondMapping;
  std::vector<std::int32_t> bondMappingIndptr;
};

nvMolKit::MCSAtomCompare parseAtomCompare(const std::string& value) {
  if (value == "any") {
    return nvMolKit::MCSAtomCompare::Any;
  }
  if (value == "elements") {
    return nvMolKit::MCSAtomCompare::Elements;
  }
  if (value == "isotopes") {
    return nvMolKit::MCSAtomCompare::Isotopes;
  }
  if (value == "any_heavy_atom") {
    return nvMolKit::MCSAtomCompare::AnyHeavyAtom;
  }
  throw std::invalid_argument("Unsupported atom_compare value: " + value);
}

nvMolKit::MCSBondCompare parseBondCompare(const std::string& value) {
  if (value == "any") {
    return nvMolKit::MCSBondCompare::Any;
  }
  if (value == "order") {
    return nvMolKit::MCSBondCompare::Order;
  }
  if (value == "order_exact") {
    return nvMolKit::MCSBondCompare::OrderExact;
  }
  throw std::invalid_argument("Unsupported bond_compare value: " + value);
}

std::vector<const RDKit::ROMol*> molsFromPythonList(const list& mols) {
  nvMolKit::ScopedNvtxRange        range("Python MCS: extract mol pointers", nvMolKit::NvtxColor::kYellow);
  std::vector<const RDKit::ROMol*> out;
  out.reserve(len(mols));
  for (int i = 0; i < len(mols); ++i) {
    const RDKit::ROMol* mol = extract<const RDKit::ROMol*>(object(mols[i]));
    if (mol == nullptr) {
      throw std::invalid_argument("Invalid molecule at index " + std::to_string(i));
    }
    out.push_back(mol);
  }
  return out;
}

std::vector<nvMolKit::MCSPair> pairsFromPythonList(const list& pairs) {
  std::vector<nvMolKit::MCSPair> out;
  out.reserve(len(pairs));
  for (int i = 0; i < len(pairs); ++i) {
    object pairObj(pairs[i]);
    if (!PySequence_Check(pairObj.ptr()) || PySequence_Size(pairObj.ptr()) != 2) {
      throw std::invalid_argument("MCS pair at index " + std::to_string(i) + " must be a length-2 sequence");
    }
    object first(handle<>(PySequence_GetItem(pairObj.ptr(), 0)));
    object second(handle<>(PySequence_GetItem(pairObj.ptr(), 1)));
    out.emplace_back(extract<std::size_t>(first), extract<std::size_t>(second));
  }
  return out;
}

template <typename T> T optionValue(const dict& options, const char* key, const T& defaultValue) {
  if (PyMapping_HasKeyString(options.ptr(), key) == 0) {
    return defaultValue;
  }
  return extract<T>(options[key]);
}

template <typename T> std::vector<T> vectorFromIterable(const object& iterable) {
  std::vector<T>        converted;
  stl_input_iterator<T> it(iterable), end;
  for (; it != end; ++it) {
    converted.push_back(*it);
  }
  return converted;
}

std::vector<int> optionIntVector(const dict& options, const char* key) {
  if (PyMapping_HasKeyString(options.ptr(), key) == 0) {
    return {};
  }
  return vectorFromIterable<int>(options[key]);
}

template <typename T> boost::python::numpy::ndarray make1dArray(std::vector<T>& values, const object& owner) {
  const Py_intptr_t shape  = static_cast<Py_intptr_t>(values.size());
  const Py_intptr_t stride = static_cast<Py_intptr_t>(sizeof(T));
  return boost::python::numpy::from_data(values.data(),
                                         boost::python::numpy::dtype::get_builtin<T>(),
                                         make_tuple(shape),
                                         make_tuple(stride),
                                         owner);
}

boost::python::numpy::ndarray makePairArray(std::vector<std::int32_t>& values, const object& owner) {
  const Py_intptr_t rows = static_cast<Py_intptr_t>(values.size() / 2);
  const Py_intptr_t item = static_cast<Py_intptr_t>(sizeof(std::int32_t));
  return boost::python::numpy::from_data(values.data(),
                                         boost::python::numpy::dtype::get_builtin<std::int32_t>(),
                                         make_tuple(rows, 2),
                                         make_tuple(2 * item, item),
                                         owner);
}

}  // namespace

BOOST_PYTHON_MODULE(_mcs) {
  boost::python::numpy::initialize();

  def(
    "_findMCSBatch",
    +[](const list& mols, const list& pairs, const dict& options) {
      auto molVec  = molsFromPythonList(mols);
      auto pairVec = pairsFromPythonList(pairs);

      nvMolKit::MCSParameters params;
      params.atomCompare          = parseAtomCompare(optionValue<std::string>(options, "atom_compare", "elements"));
      params.bondCompare          = parseBondCompare(optionValue<std::string>(options, "bond_compare", "order"));
      params.maximizeBonds        = optionValue<bool>(options, "maximize_bonds", true);
      params.connectedOnly        = optionValue<bool>(options, "connected_only", true);
      params.requireGpu           = optionValue<bool>(options, "require_gpu", false);
      params.timeoutSeconds       = optionValue<unsigned int>(options, "timeout_seconds", 0);
      params.batchSize            = optionValue<int>(options, "batch_size", 0);
      params.workerThreads        = optionValue<int>(options, "worker_threads", -1);
      params.preprocessingThreads = optionValue<int>(options, "preprocessing_threads", -1);
      params.executorsPerRunner   = optionValue<int>(options, "executors_per_runner", -1);
      params.gpuIds               = optionIntVector(options, "gpu_ids");
      params.atomCompareParameters.matchValences     = optionValue<bool>(options, "match_valences", false);
      params.atomCompareParameters.matchFormalCharge = optionValue<bool>(options, "match_formal_charge", false);
      params.atomCompareParameters.ringMatchesRingOnly =
        optionValue<bool>(options, "atom_ring_matches_ring_only", false);
      params.atomCompareParameters.matchIsotope = optionValue<bool>(options, "match_isotope", false);
      params.bondCompareParameters.ringMatchesRingOnly =
        optionValue<bool>(options, "bond_ring_matches_ring_only", false);

      nvMolKit::ScopedNvtxRange mcsRange("Python MCS: findMCSBatch", nvMolKit::NvtxColor::kOrange);
      auto                      results = nvMolKit::findMCSBatch(molVec, pairVec, nullptr, params);
      mcsRange.pop();

      auto buffers = std::make_unique<MCSResultBuffers>();
      buffers->numAtoms.reserve(results.size());
      buffers->numBonds.reserve(results.size());
      buffers->canceled.reserve(results.size());
      buffers->atomMappingIndptr.reserve(results.size() + 1);
      buffers->bondMappingIndptr.reserve(results.size() + 1);
      buffers->atomMappingIndptr.push_back(0);
      buffers->bondMappingIndptr.push_back(0);

      for (const auto& result : results) {
        buffers->numAtoms.push_back(result.numAtoms);
        buffers->numBonds.push_back(result.numBonds);
        buffers->canceled.push_back(result.canceled ? 1 : 0);

        for (const auto& [a, b] : result.atomMapping) {
          buffers->atomMapping.push_back(static_cast<std::int32_t>(a));
          buffers->atomMapping.push_back(static_cast<std::int32_t>(b));
        }
        buffers->atomMappingIndptr.push_back(static_cast<std::int32_t>(buffers->atomMapping.size() / 2));

        for (const auto& [a, b] : result.bondMapping) {
          buffers->bondMapping.push_back(static_cast<std::int32_t>(a));
          buffers->bondMapping.push_back(static_cast<std::int32_t>(b));
        }
        buffers->bondMappingIndptr.push_back(static_cast<std::int32_t>(buffers->bondMapping.size() / 2));
      }

      auto deleter = [](PyObject* cap) {
        auto* ptr = reinterpret_cast<MCSResultBuffers*>(PyCapsule_GetPointer(cap, "nvmolkit.mcs_results"));
        delete ptr;
      };
      PyObject* cap = PyCapsule_New(static_cast<void*>(buffers.get()), "nvmolkit.mcs_results", deleter);
      if (cap == nullptr) {
        throw std::runtime_error("Failed to create PyCapsule for MCS results");
      }
      object owner{handle<>(cap)};
      buffers.release();
      auto* ptr = reinterpret_cast<MCSResultBuffers*>(PyCapsule_GetPointer(cap, "nvmolkit.mcs_results"));

      nvMolKit::ScopedNvtxRange wrapRange("Python MCS: wrap results", nvMolKit::NvtxColor::kGreen);
      return make_tuple(make1dArray(ptr->numAtoms, owner),
                        make1dArray(ptr->numBonds, owner),
                        make1dArray(ptr->canceled, owner),
                        makePairArray(ptr->atomMapping, owner),
                        make1dArray(ptr->atomMappingIndptr, owner),
                        makePairArray(ptr->bondMapping, owner),
                        make1dArray(ptr->bondMappingIndptr, owner));
    },
    (arg("mols"), arg("pairs"), arg("options") = dict()));
}
