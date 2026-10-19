// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef NVMOLKIT_UFF_KERNELS_DEVICE_DISPATCH_CUH
#define NVMOLKIT_UFF_KERNELS_DEVICE_DISPATCH_CUH
#include <cmath>

#include "src/forcefields/mmff_kernels_device_dispatch.cuh"

namespace nvMolKit::UFF::fp64 {
using namespace nvMolKit::MMFF::fp64;
using real = double;
#define NVMOLKIT_FMIN    fmin
#define NVMOLKIT_FMAX    fmax
#define NVMOLKIT_RSQRT   rsqrt
#define NVMOLKIT_SQRT    sqrt
#define NVMOLKIT_ACOS    acos
#define NVMOLKIT_ASIN    asin
#define NVMOLKIT_ATAN2   atan2
#define NVMOLKIT_EXP     exp
#define NVMOLKIT_IS_ZERO isDoubleZero
#include "src/forcefields/uff_kernels_device.cuh"
#undef NVMOLKIT_FMIN
#undef NVMOLKIT_FMAX
#undef NVMOLKIT_RSQRT
#undef NVMOLKIT_SQRT
#undef NVMOLKIT_ACOS
#undef NVMOLKIT_ASIN
#undef NVMOLKIT_ATAN2
#undef NVMOLKIT_EXP
#undef NVMOLKIT_IS_ZERO

}  // namespace nvMolKit::UFF::fp64

namespace nvMolKit::UFF::fp32 {
using namespace nvMolKit::MMFF::fp32;
using real = float;
#define NVMOLKIT_FMIN    fminf
#define NVMOLKIT_FMAX    fmaxf
#define NVMOLKIT_RSQRT   rsqrtf
#define NVMOLKIT_SQRT    sqrtf
#define NVMOLKIT_ACOS    acosf
#define NVMOLKIT_ASIN    asinf
#define NVMOLKIT_ATAN2   atan2f
#define NVMOLKIT_EXP     expf
#define NVMOLKIT_IS_ZERO isFloatZero
#include "src/forcefields/uff_kernels_device.cuh"
#undef NVMOLKIT_FMIN
#undef NVMOLKIT_FMAX
#undef NVMOLKIT_RSQRT
#undef NVMOLKIT_SQRT
#undef NVMOLKIT_ACOS
#undef NVMOLKIT_ASIN
#undef NVMOLKIT_ATAN2
#undef NVMOLKIT_EXP
#undef NVMOLKIT_IS_ZERO

}  // namespace nvMolKit::UFF::fp32

namespace nvMolKit::UFF {
using namespace fp64;
}
#endif  // NVMOLKIT_UFF_KERNELS_DEVICE_DISPATCH_CUH
