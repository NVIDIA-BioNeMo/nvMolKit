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

#ifndef NVMOLKIT_DEVICE_RESULT_PYTHON_H
#define NVMOLKIT_DEVICE_RESULT_PYTHON_H

#include <boost/python.hpp>
#include <cstdint>

#include "array_helpers.h"
#include "device_coord_result.h"
#include "device_vector.h"

namespace nvMolKit {

inline boost::python::object wrapAsync(PyArray* arr, const int gpuId, const boost::python::object& asyncCls) {
  return asyncCls(boost::python::object(boost::python::ptr(arr)), gpuId);
}

/**
 * @brief Build a Python @c Device3DResult that takes ownership of all device buffers.
 *
 * Each @ref AsyncDeviceVector argument is consumed (via @c release() inside @ref makePyArray),
 * so callers must pass freshly-built buffers, not references into longer-lived state.
 */
inline boost::python::object buildOwningDevice3DResult(AsyncDeviceVector<double>&  values,
                                                       AsyncDeviceVector<int32_t>& atomStarts,
                                                       AsyncDeviceVector<int32_t>& molIndices,
                                                       AsyncDeviceVector<int32_t>& confIndices,
                                                       const int                   gpuId,
                                                       const int                   nMols,
                                                       AsyncDeviceVector<double>*  energies  = nullptr,
                                                       AsyncDeviceVector<int8_t>*  converged = nullptr) {
  boost::python::object types_module = boost::python::import("nvmolkit.types");
  boost::python::object d3d_cls      = types_module.attr("Device3DResult");
  boost::python::object async_cls    = types_module.attr("AsyncGpuResult");
  const int             natoms       = static_cast<int>(values.size() / 3);
  PyArray*              valuesPy     = makePyArray(values, "f8", boost::python::make_tuple(natoms, 3));
  PyArray*              atomStartsPy = makePyArray(atomStarts);
  PyArray*              molIdxPy     = makePyArray(molIndices);
  PyArray*              confIdxPy    = makePyArray(confIndices);
  boost::python::object energiesObj  = boost::python::object();
  boost::python::object convergedObj = boost::python::object();
  if (energies != nullptr) {
    energiesObj = wrapAsync(makePyArray(*energies), gpuId, async_cls);
  }
  if (converged != nullptr) {
    convergedObj = wrapAsync(makePyArray(*converged), gpuId, async_cls);
  }
  return d3d_cls(wrapAsync(valuesPy, gpuId, async_cls),
                 wrapAsync(atomStartsPy, gpuId, async_cls),
                 wrapAsync(molIdxPy, gpuId, async_cls),
                 wrapAsync(confIdxPy, gpuId, async_cls),
                 gpuId,
                 nMols,
                 energiesObj,
                 convergedObj);
}

/**
 * @brief Convenience overload that pulls all fields from a C++ @ref DeviceCoordResult.
 *
 * The @c DeviceCoordResult's buffers are moved into the Python result via @c release(). When
 * @c energies / @c converged are size 0 (e.g. ETKDG output) they are forwarded as @c nullptr so
 * the Python result's corresponding fields are @c None rather than empty tensors.
 */
inline boost::python::object buildOwningDevice3DResult(DeviceCoordResult& dev) {
  AsyncDeviceVector<double>* energiesPtr  = dev.energies.size() > 0 ? &dev.energies : nullptr;
  AsyncDeviceVector<int8_t>* convergedPtr = dev.converged.size() > 0 ? &dev.converged : nullptr;
  return buildOwningDevice3DResult(dev.positions,
                                   dev.atomStarts,
                                   dev.molIndices,
                                   dev.confIndices,
                                   dev.gpuId,
                                   dev.nMols,
                                   energiesPtr,
                                   convergedPtr);
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_DEVICE_RESULT_PYTHON_H
