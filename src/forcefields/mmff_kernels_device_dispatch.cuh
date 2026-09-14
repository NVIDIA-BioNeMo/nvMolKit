// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef NVMOLKIT_MMFF_KERNELS_DEVICE_DISPATCH_CUH
#define NVMOLKIT_MMFF_KERNELS_DEVICE_DISPATCH_CUH
#include "src/forcefields/kernel_utils.cuh"

// Instantiate the shared device implementation for each arithmetic type.
namespace nvMolKit::MMFF::fp64 {
using real = double;
#define NVMOLKIT_FMOD    fmod
#define NVMOLKIT_FMIN    fmin
#define NVMOLKIT_FMAX    fmax
#define NVMOLKIT_RSQRT   rsqrt
#define NVMOLKIT_SQRT    sqrt
#define NVMOLKIT_ACOS    acos
#define NVMOLKIT_ASIN    asin
#define NVMOLKIT_COS     cos
#define NVMOLKIT_ATAN2   atan2
#define NVMOLKIT_EXP     exp
#define NVMOLKIT_IS_ZERO isDoubleZero
#include "src/forcefields/mmff_kernels_device.cuh"
#undef NVMOLKIT_FMOD
#undef NVMOLKIT_FMIN
#undef NVMOLKIT_FMAX
#undef NVMOLKIT_RSQRT
#undef NVMOLKIT_SQRT
#undef NVMOLKIT_ACOS
#undef NVMOLKIT_ASIN
#undef NVMOLKIT_COS
#undef NVMOLKIT_ATAN2
#undef NVMOLKIT_EXP
#undef NVMOLKIT_IS_ZERO

}  // namespace nvMolKit::MMFF::fp64

namespace nvMolKit::MMFF {
// Unqualified device helpers use the full-precision instantiation.
using namespace fp64;
}  // namespace nvMolKit::MMFF
#endif  // NVMOLKIT_MMFF_KERNELS_DEVICE_DISPATCH_CUH
