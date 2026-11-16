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

#ifndef NVMOLKIT_BOOST_PYTHON_UTILS_H
#define NVMOLKIT_BOOST_PYTHON_UTILS_H

#include <GraphMol/ROMol.h>

#include <boost/python.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace nvMolKit {

//! Releases the GIL for its lifetime so native worker threads can call back into Python (e.g. RDKit's Python log
//! sink). Construct it only after all Python arguments are converted and touch no Python object while it is alive.
//! The destructor re-acquires the GIL, including during exception unwinding, before Boost.Python translates the
//! exception. Other Python threads run meanwhile and may modify the caller's lists, so callers keep their own
//! reference to every molecule the native call reads, e.g. by copying the argument list into a tuple, and take
//! option objects by value rather than referring to the caller's.
class ScopedGilRelease {
 public:
  ScopedGilRelease() : state_(PyEval_SaveThread()) {}
  ScopedGilRelease(const ScopedGilRelease&)            = delete;
  ScopedGilRelease& operator=(const ScopedGilRelease&) = delete;
  ~ScopedGilRelease() { PyEval_RestoreThread(state_); }

 private:
  PyThreadState* state_;
};

template <typename T> boost::python::list vectorToList(const std::vector<T>& vec) {
  boost::python::list list;
  for (const auto& value : vec) {
    list.append(value);
  }
  return list;
}

template <typename T> boost::python::list vectorOfVectorsToList(const std::vector<std::vector<T>>& vecOfVecs) {
  boost::python::list outerList;
  for (const auto& innerVec : vecOfVecs) {
    outerList.append(vectorToList(innerVec));
  }
  return outerList;
}

inline std::vector<RDKit::ROMol*> extractMolecules(const boost::python::object& molecules) {
  const int                  n = boost::python::len(molecules);
  std::vector<RDKit::ROMol*> mols;
  mols.reserve(n);
  for (int i = 0; i < n; ++i) {
    auto* mol = boost::python::extract<RDKit::ROMol*>(boost::python::object(molecules[i]))();
    if (mol == nullptr) {
      throw std::invalid_argument("Invalid molecule at index " + std::to_string(i));
    }
    mols.push_back(mol);
  }
  return mols;
}

inline std::vector<double> extractDoubleList(const boost::python::list& values,
                                             const int                  expectedSize,
                                             const std::string&         name) {
  if (boost::python::len(values) != expectedSize) {
    throw std::invalid_argument("Expected " + std::to_string(expectedSize) + " values for " + name + ", got " +
                                std::to_string(boost::python::len(values)));
  }
  std::vector<double> result;
  result.reserve(expectedSize);
  for (int i = 0; i < expectedSize; ++i) {
    result.push_back(boost::python::extract<double>(values[i]));
  }
  return result;
}

inline std::vector<bool> extractBoolList(const boost::python::list& values,
                                         const int                  expectedSize,
                                         const std::string&         name) {
  if (boost::python::len(values) != expectedSize) {
    throw std::invalid_argument("Expected " + std::to_string(expectedSize) + " values for " + name + ", got " +
                                std::to_string(boost::python::len(values)));
  }
  std::vector<bool> result;
  result.reserve(expectedSize);
  for (int i = 0; i < expectedSize; ++i) {
    // A named extractor avoids GCC's -Wmaybe-uninitialized false positive in Boost.Python.
    const boost::python::extract<bool> value(values[i]);
    result.push_back(value());
  }
  return result;
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_BOOST_PYTHON_UTILS_H
