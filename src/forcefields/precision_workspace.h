// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef NVMOLKIT_FORCEFIELDS_PRECISION_WORKSPACE_H
#define NVMOLKIT_FORCEFIELDS_PRECISION_WORKSPACE_H

#include <cuda_runtime.h>

#include "src/utils/device_vector.h"

namespace nvMolKit {

//! Device scratch used only when a force-field call crosses precision boundaries.
//! The storage type is the force field's native precision, not the caller's.
template <typename storageT> struct ForcefieldConversionWorkspace {
  AsyncDeviceVector<storageT> positions;
  AsyncDeviceVector<storageT> gradients;
  AsyncDeviceVector<storageT> energies;

  void setStream(cudaStream_t stream) {
    positions.setStream(stream);
    gradients.setStream(stream);
    energies.setStream(stream);
  }
};

using SingleForcefieldConversionWorkspace = ForcefieldConversionWorkspace<float>;

}  // namespace nvMolKit

#endif  // NVMOLKIT_FORCEFIELDS_PRECISION_WORKSPACE_H
