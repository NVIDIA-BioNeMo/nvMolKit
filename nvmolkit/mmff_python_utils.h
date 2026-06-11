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

#ifndef NVMOLKIT_MMFF_PYTHON_UTILS_H
#define NVMOLKIT_MMFF_PYTHON_UTILS_H

#include <GraphMol/ForceFieldHelpers/MMFF/AtomTyper.h>

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <stdexcept>
#include <vector>

#include "src/forcefields/mmff_properties.h"

namespace ForceFields {

/// \brief Layout-compatible shim for RDKit's Python wrapper class.
///
/// RDKit registers its Python \c MMFFMolProperties binding as
/// \c ForceFields::PyMMFFMolProperties (declared in the un-installed Wrap-layer
/// header \c Code/ForceField/Wrap/PyForceField.h) which holds the real C++
/// \c RDKit::MMFF::MMFFMolProperties via a public \c boost::shared_ptr member.
/// This declaration matches the layout of RDKit's class so Boost.Python's type
/// registry maps the Python object to the same \c type_info, letting us read
/// the underlying \c MMFFMolProperties through its public shared_ptr member.
/// The class has no virtual methods in RDKit, so only the single member needs
/// to match for RTTI-based lookup to resolve.
///
/// \warning This is brittle and relies on Linux symbol resolution to make the
/// \c type_info here compare equal to RDKit's. The long-term fix is for RDKit
/// to expose the scalar-setting getters on its Python binding so this shim
/// can go away.
/// TODO: This will go away after https://github.com/rdkit/rdkit/issues/9253 is implemented
///       but we'll need to keep it as backup as long as we support older versions of RDKit.
class PyMMFFMolProperties {
 public:
  boost::shared_ptr<RDKit::MMFF::MMFFMolProperties> mmffMolProperties;
};

static_assert(sizeof(PyMMFFMolProperties) == sizeof(boost::shared_ptr<RDKit::MMFF::MMFFMolProperties>),
              "nvMolKit's PyMMFFMolProperties shim must hold exactly one boost::shared_ptr; "
              "adding fields or virtual methods here breaks the layout contract with RDKit.");

}  // namespace ForceFields

namespace nvMolKit {

/// \brief Populate an nvMolKit MMFF transport from an RDKit MMFFMolProperties Python object.
///
/// RDKit's Python binding for \c MMFFMolProperties only exposes setters for the
/// scalar settings (variant, dielectric, per-term flags); the corresponding getters
/// are unbound. This helper peeks at the underlying C++
/// \c RDKit::MMFF::MMFFMolProperties through RDKit's Python wrapper shim and reads
/// the settings with the C++ getters directly.
inline MMFFProperties buildMMFFPropertiesFromRDKit(const boost::python::object& pyProps,
                                                   double                       nonBondedThreshold,
                                                   bool                         ignoreInterfragInteractions) {
  boost::python::extract<ForceFields::PyMMFFMolProperties*> extractor(pyProps);
  if (!extractor.check()) {
    throw std::invalid_argument("buildMMFFPropertiesFromRDKit: expected an RDKit MMFFMolProperties object");
  }
  ForceFields::PyMMFFMolProperties* pyWrapper = extractor();
  if (pyWrapper == nullptr || pyWrapper->mmffMolProperties.get() == nullptr) {
    throw std::invalid_argument("buildMMFFPropertiesFromRDKit: null MMFFMolProperties pointer");
  }
  RDKit::MMFF::MMFFMolProperties* rdProps = pyWrapper->mmffMolProperties.get();

  MMFFProperties props;
  props.variant                     = rdProps->getMMFFVariant();
  props.dielectricConstant          = rdProps->getMMFFDielectricConstant();
  props.dielectricModel             = static_cast<int>(rdProps->getMMFFDielectricModel());
  props.nonBondedThreshold          = nonBondedThreshold;
  props.ignoreInterfragInteractions = ignoreInterfragInteractions;
  props.bondTerm                    = rdProps->getMMFFBondTerm();
  props.angleTerm                   = rdProps->getMMFFAngleTerm();
  props.stretchBendTerm             = rdProps->getMMFFStretchBendTerm();
  props.oopTerm                     = rdProps->getMMFFOopTerm();
  props.torsionTerm                 = rdProps->getMMFFTorsionTerm();
  props.vdwTerm                     = rdProps->getMMFFVdWTerm();
  props.eleTerm                     = rdProps->getMMFFEleTerm();
  return props;
}

inline MMFFProperties extractMMFFProperties(const boost::python::object& obj,
                                            double                       nonBondedThreshold          = 100.0,
                                            bool                         ignoreInterfragInteractions = true) {
  MMFFProperties props;
  if (obj.is_none()) {
    props.nonBondedThreshold          = nonBondedThreshold;
    props.ignoreInterfragInteractions = ignoreInterfragInteractions;
    return props;
  }
  props = boost::python::extract<MMFFProperties>(obj);
  return props;
}

inline std::vector<MMFFProperties> extractMMFFPropertiesList(const boost::python::list& properties, int numMols) {
  const int                   n = boost::python::len(properties);
  std::vector<MMFFProperties> props;
  props.reserve(numMols);
  for (int i = 0; i < numMols; ++i) {
    if (i < n) {
      props.push_back(extractMMFFProperties(boost::python::object(properties[i])));
    } else {
      props.emplace_back();
    }
  }
  return props;
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_MMFF_PYTHON_UTILS_H
