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

#ifndef NVMOLKIT_ARRAY_HELPERS
#define NVMOLKIT_ARRAY_HELPERS

#include <boost/python.hpp>
#include <optional>

#include "src/utils/device_vector.h"

namespace nvMolKit {

template <typename blockT> cuda::std::span<const blockT> getSpanFromDictElems(void* data, boost::python::tuple& shape) {
  size_t size = boost::python::extract<size_t>(shape[0]);
  // multiply by any other dimensions
  for (int i = 1; i < len(shape); i++) {
    size *= boost::python::extract<size_t>(shape[i]);
  }

  return cuda::std::span<blockT>(reinterpret_cast<blockT*>(data), size);
}

struct PyArray {
  PyArray() = default;
  ~PyArray() {
    if (devicePtr != nullptr && owned) {
      cudaFreeAsync(devicePtr, stream);
    }
    devicePtr = nullptr;
  }

  boost::python::dict __cuda_array_interface__;
  boost::python::dict __array_interface__;
  void*               devicePtr = nullptr;
  cudaStream_t        stream    = nullptr;
  //! When true, the destructor frees devicePtr. When false, the underlying allocation is owned
  //! elsewhere (e.g. by a long-lived AsyncDeviceVector) and this PyArray is a non-owning view.
  bool                owned     = true;
};

template <typename T> std::string getNumpyType() {
  if constexpr (std::is_same_v<T, float>) {
    return "f4";
  } else if (std::is_same_v<T, double>) {
    return "f8";
  } else if (std::is_same_v<T, std::int32_t> || std::is_same_v<T, int>) {
    return "i4";
  } else if (std::is_same_v<T, std::uint32_t>) {
    return "u4";
  } else if (std::is_same_v<T, std::int64_t>) {
    return "l8";
  } else if (std::is_same_v<T, std::uint64_t>) {
    return "L8";
  } else if (std::is_same_v<T, std::int16_t>) {
    return "h2";
  } else if (std::is_same_v<T, std::uint16_t>) {
    return "H2";
  } else if (std::is_same_v<T, std::int8_t>) {
    return "b1";
  } else if (std::is_same_v<T, std::uint8_t>) {
    return "B1";
  } else {
    throw std::runtime_error("Unsupported type for numpy array:" + std::string(typeid(T).name()));
  }
}

template <typename T>
PyArray* makePyArray(AsyncDeviceVector<T>& deviceVector, const std::string& dTypeStr, boost::python::tuple shape) {
  auto thisPyArray                      = new PyArray();
  thisPyArray->__cuda_array_interface__ = boost::python::dict();
  auto& dict                            = thisPyArray->__cuda_array_interface__;

  thisPyArray->stream    = deviceVector.stream();
  T* releasedPtr         = deviceVector.release();
  thisPyArray->devicePtr = releasedPtr;

  dict["shape"]   = shape;
  dict["typestr"] = boost::python::str("|" + dTypeStr);
  dict["data"]    = boost::python::make_tuple(reinterpret_cast<std::size_t>(releasedPtr), /*readOnly=*/false);
  dict["version"] = 2;

  return thisPyArray;
}

template <typename T, typename = std::enable_if_t<std::is_integral_v<T> || std::is_floating_point_v<T>>>
PyArray* makePyArray(AsyncDeviceVector<T>& deviceVector, std::optional<boost::python::tuple> shape = std::nullopt) {
  return makePyArray(deviceVector, getNumpyType<T>(), shape.value_or(boost::python::make_tuple(deviceVector.size())));
}

/**
 * @brief Construct a PyArray that borrows an existing AsyncDeviceVector without taking ownership.
 *
 * The returned object exposes @c __cuda_array_interface__ pointing at the device vector's data, but
 * the PyArray destructor will not free it. The caller must keep @p deviceVector alive at least as
 * long as any Python consumer of the returned array. Use this for persistent device buffers held by
 * long-lived wrapper classes (e.g. MMFFBatchedForcefield) where each compute call hands out a view.
 */
template <typename T>
PyArray* makePyArrayBorrowed(AsyncDeviceVector<T>& deviceVector,
                             const std::string&    dTypeStr,
                             boost::python::tuple  shape) {
  auto thisPyArray                      = new PyArray();
  thisPyArray->__cuda_array_interface__ = boost::python::dict();
  auto& dict                            = thisPyArray->__cuda_array_interface__;

  thisPyArray->stream    = deviceVector.stream();
  thisPyArray->devicePtr = static_cast<void*>(deviceVector.data());
  thisPyArray->owned     = false;

  dict["shape"]   = shape;
  dict["typestr"] = boost::python::str("|" + dTypeStr);
  dict["data"]    = boost::python::make_tuple(reinterpret_cast<std::size_t>(deviceVector.data()), /*readOnly=*/false);
  dict["version"] = 2;

  return thisPyArray;
}

template <typename T, typename = std::enable_if_t<std::is_integral_v<T> || std::is_floating_point_v<T>>>
PyArray* makePyArrayBorrowed(AsyncDeviceVector<T>&               deviceVector,
                             std::optional<boost::python::tuple> shape = std::nullopt) {
  return makePyArrayBorrowed(deviceVector,
                             getNumpyType<T>(),
                             shape.value_or(boost::python::make_tuple(deviceVector.size())));
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_ARRAY_HELPERS
