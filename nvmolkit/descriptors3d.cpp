// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <GraphMol/ROMol.h>

#include <boost/python.hpp>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "nvmolkit/array_helpers.h"
#include "src/descriptors3d_mol.h"
#include "src/precision/precision_mode.h"
#include "src/utils/device.h"

namespace {

namespace bp = boost::python;

template <typename T> const T* devicePointer(const bp::dict& interface) {
  const bp::tuple      data    = bp::extract<bp::tuple>(interface["data"]);
  const std::uintptr_t address = bp::extract<std::uintptr_t>(data[0]);
  return reinterpret_cast<const T*>(address);
}

int firstDimension(const bp::dict& interface) {
  const bp::tuple shape = bp::extract<bp::tuple>(interface["shape"]);
  if (bp::len(shape) == 0) {
    throw std::invalid_argument("CUDA array must have at least one dimension");
  }
  return bp::extract<int>(shape[0]);
}

//! Unpack a (positions, atomStarts, molIndices, nMols) tuple of CUDA array interfaces.
nvMolKit::DeviceCoordView deviceCoordViewFromPython(const bp::tuple& coordinates) {
  const bp::dict positions  = bp::extract<bp::dict>(coordinates[0]);
  const bp::dict atomStarts = bp::extract<bp::dict>(coordinates[1]);
  const bp::dict molIndices = bp::extract<bp::dict>(coordinates[2]);

  nvMolKit::DeviceCoordView view;
  view.positions     = devicePointer<double>(positions);
  view.atomStarts    = devicePointer<int32_t>(atomStarts);
  view.molIndices    = devicePointer<int32_t>(molIndices);
  view.numConformers = firstDimension(molIndices);
  view.nMols         = bp::extract<int>(coordinates[3]);
  view.numAtoms      = firstDimension(positions);
  if (firstDimension(atomStarts) != view.numConformers + 1) {
    throw std::invalid_argument("atom_starts must contain one more entry than mol_indices");
  }
  return view;
}

bp::object toOwnedPyArray(nvMolKit::PyArray* array) {
  using Converter = bp::manage_new_object::apply<nvMolKit::PyArray*>::type;
  return bp::object(bp::handle<>(Converter()(array)));
}

//! Returns (properties dict in request order, molIndices or None, confIndices or None).
template <typename Real>
bp::object calc3DPropertiesAs(const std::vector<const RDKit::ROMol*>&  mols,
                              const std::vector<nvMolKit::Property3D>& properties,
                              const bool                               useAtomicMasses,
                              cudaStream_t                             stream,
                              const nvMolKit::DeviceCoordView*         view) {
  auto result = nvMolKit::calc3DProperties<Real>(mols, properties, useAtomicMasses, stream, view);

  bp::dict output;
  for (const nvMolKit::Property3D property : properties) {
    auto& values = result.properties.at(property);
    output[std::string(nvMolKit::property3DName(property))] =
      toOwnedPyArray(nvMolKit::makePyArray(values, bp::make_tuple(values.size())));
  }
  // Row labels exist only when coordinates came from the molecules; otherwise the caller already
  // holds the labels of its own coordinate batch.
  if (view != nullptr) {
    return bp::make_tuple(output, bp::object(), bp::object());
  }
  return bp::make_tuple(output,
                        toOwnedPyArray(nvMolKit::makePyArray(result.molIndices)),
                        toOwnedPyArray(nvMolKit::makePyArray(result.confIndices)));
}

bp::object calc3DProperties(const bp::list&                mols,
                            const bp::list&                propertyNames,
                            const bool                     useAtomicMasses,
                            const bp::object&              coordinates,
                            const nvMolKit::PrecisionMode& precision,
                            const std::uintptr_t           streamPtr) {
  const auto stream = nvMolKit::acquireExternalStream(streamPtr);
  if (!stream) {
    throw std::invalid_argument("Invalid CUDA stream");
  }

  std::vector<nvMolKit::Property3D> properties;
  for (int i = 0; i < bp::len(propertyNames); ++i) {
    properties.push_back(nvMolKit::property3DFromName(bp::extract<std::string>(propertyNames[i])()));
  }

  std::vector<const RDKit::ROMol*> molPtrs(bp::len(mols));
  for (size_t i = 0; i < molPtrs.size(); ++i) {
    molPtrs[i] = bp::extract<const RDKit::ROMol*>(bp::object(mols[i]));
  }

  std::optional<nvMolKit::DeviceCoordView> view;
  if (!coordinates.is_none()) {
    view = deviceCoordViewFromPython(bp::extract<bp::tuple>(coordinates));
  }
  const nvMolKit::DeviceCoordView* viewPtr = view ? &*view : nullptr;
  if (nvMolKit::usesSinglePrecision(precision)) {
    return calc3DPropertiesAs<float>(molPtrs, properties, useAtomicMasses, *stream, viewPtr);
  }
  return calc3DPropertiesAs<double>(molPtrs, properties, useAtomicMasses, *stream, viewPtr);
}

}  // namespace

BOOST_PYTHON_MODULE(_descriptors3d) {
  bp::def("Calc3DProperties",
          &calc3DProperties,
          (bp::arg("mols"),
           bp::arg("properties"),
           bp::arg("useAtomicMasses"),
           bp::arg("coordinates"),
           bp::arg("precision"),
           bp::arg("stream")));
}
