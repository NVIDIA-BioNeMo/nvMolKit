// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <GraphMol/ROMol.h>

#include <boost/noncopyable.hpp>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <cstddef>
#include <vector>

#include "src/substruct/substruct_library.h"

namespace {

using namespace boost::python;

class ScopedGilRelease {
 public:
  ScopedGilRelease() : state_(PyEval_SaveThread()) {}
  ScopedGilRelease(const ScopedGilRelease&)            = delete;
  ScopedGilRelease& operator=(const ScopedGilRelease&) = delete;
  ~ScopedGilRelease() { PyEval_RestoreThread(state_); }

 private:
  PyThreadState* state_;
};

unsigned int addMolecule(nvMolKit::SubstructLibrary& library, const RDKit::ROMol& molecule) {
  const ScopedGilRelease release;
  return library.addMol(molecule);
}

list addMolecules(nvMolKit::SubstructLibrary& library, const object& molecules) {
  std::vector<object>              owners;
  std::vector<const RDKit::ROMol*> pointers;
  stl_input_iterator<object>       iterator(molecules), end;
  for (; iterator != end; ++iterator) {
    owners.push_back(*iterator);
    pointers.push_back(extract<const RDKit::ROMol*>(owners.back()));
  }

  std::vector<unsigned int> ids;
  {
    const ScopedGilRelease release;
    ids = library.addMols(pointers);
  }

  list result;
  for (const unsigned int id : ids) {
    result.append(id);
  }
  return result;
}

void finalize(nvMolKit::SubstructLibrary& library) {
  const ScopedGilRelease release;
  library.finalize();
}

list getMatches(nvMolKit::SubstructLibrary& library, const RDKit::ROMol& query, int maxResults) {
  std::vector<unsigned int> ids;
  {
    const ScopedGilRelease release;
    ids = library.getMatches(query, maxResults);
  }
  list result;
  for (const unsigned int id : ids) {
    result.append(id);
  }
  return result;
}

std::size_t countMatches(nvMolKit::SubstructLibrary& library, const RDKit::ROMol& query) {
  const ScopedGilRelease release;
  return library.countMatches(query);
}

bool hasMatch(nvMolKit::SubstructLibrary& library, const RDKit::ROMol& query) {
  const ScopedGilRelease release;
  return library.hasMatch(query);
}

// Accessors wait on the library's lock, so they release the GIL like the operations do.
template <std::size_t (nvMolKit::SubstructLibrary::*Accessor)() const>
std::size_t withoutGil(const nvMolKit::SubstructLibrary& library) {
  const ScopedGilRelease release;
  return (library.*Accessor)();
}

}  // namespace

BOOST_PYTHON_MODULE(_substructLibrary) {
  import("rdkit.Chem.rdchem");
  import("nvmolkit._substructure");

  class_<nvMolKit::SubstructLibrary, boost::noncopyable>(
    "SubstructLibrary",
    init<nvMolKit::SubstructSearchConfig>((arg("config") = nvMolKit::SubstructSearchConfig())))
    .def("addMol", &addMolecule)
    .def("addMols", &addMolecules)
    .def("finalize", &finalize)
    .def("getMatches", &getMatches, (arg("query"), arg("maxResults") = -1))
    .def("countMatches", &countMatches)
    .def("hasMatch", &hasMatch)
    .def("__len__", &withoutGil<&nvMolKit::SubstructLibrary::size>)
    .add_property("pendingSize", &withoutGil<&nvMolKit::SubstructLibrary::pendingSize>)
    .add_property("maxConcurrentQueries", &withoutGil<&nvMolKit::SubstructLibrary::maxConcurrentQueries>);
}
