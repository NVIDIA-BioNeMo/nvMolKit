// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef NVMOLKIT_DISTGEOM_KERNELS_DEVICE_DISPATCH_CUH
#define NVMOLKIT_DISTGEOM_KERNELS_DEVICE_DISPATCH_CUH

#include <cooperative_groups.h>

#include "src/forcefields/dist_geom_kernels.h"
#include "src/forcefields/kernel_utils.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace nvMolKit::FFKernelUtils;

namespace nvMolKit {
namespace DistGeom {

namespace fp64 {
using EnergyForceContribsDevicePtr   = DistGeom::EnergyForceContribsDevicePtr;
using Energy3DForceContribsDevicePtr = DistGeom::Energy3DForceContribsDevicePtr;
constexpr double RAD2DEG             = 180.0 / M_PI;
#define DG_REAL       double
#define DG_PARAM_REAL double
#define DG_RSQRTF     rsqrtf
#define DG_RSQRT      rsqrt
#define DG_SQRTF      sqrtf
#define DG_SQRT       sqrt
#define DG_ACOS       acos
#define DG_FMAX       fmax
#define DG_IS_ZERO    isDoubleZero
#include "src/forcefields/dist_geom_kernels_device.cuh"
#undef DG_REAL
#undef DG_PARAM_REAL
#undef DG_RSQRTF
#undef DG_RSQRT
#undef DG_SQRTF
#undef DG_SQRT
#undef DG_ACOS
#undef DG_FMAX
#undef DG_IS_ZERO
}  // namespace fp64

namespace fp32 {
using EnergyForceContribsDevicePtr   = DistGeom::EnergyForceContribsDevicePtrSingle;
using Energy3DForceContribsDevicePtr = DistGeom::Energy3DForceContribsDevicePtrSingle;
constexpr float RAD2DEG              = 180.0f / static_cast<float>(M_PI);
#define DG_REAL       float
#define DG_PARAM_REAL float
#define DG_RSQRTF     rsqrtf
#define DG_RSQRT      rsqrtf
#define DG_SQRTF      sqrtf
#define DG_SQRT       sqrtf
#define DG_ACOS       acosf
#define DG_FMAX       fmaxf
#define DG_IS_ZERO    isFloatZero
#include "src/forcefields/dist_geom_kernels_device.cuh"
#undef DG_REAL
#undef DG_PARAM_REAL
#undef DG_RSQRTF
#undef DG_RSQRT
#undef DG_SQRTF
#undef DG_SQRT
#undef DG_ACOS
#undef DG_FMAX
#undef DG_IS_ZERO
}  // namespace fp32

using namespace fp64;

}  // namespace DistGeom
}  // namespace nvMolKit

#endif  // NVMOLKIT_DISTGEOM_KERNELS_DEVICE_DISPATCH_CUH
