// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_DESCRIPTORS3D_DCLV_CUH
#define NVMOLKIT_DESCRIPTORS3D_DCLV_CUH

#include <cmath>
#include <cstdint>

#include "src/descriptors3d.h"
#include "src/descriptors3d_kernel.cuh"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"

namespace nvMolKit::descriptors3d_detail {

//! The DCLV properties, in RDKit's DoubleCubicLatticeVolume getter declaration order.
enum DclvColumn : int {
  kDclvSurfaceArea      = 0,
  kDclvPolarSurfaceArea = 1,
  kDclvVolume           = 2,
  kDclvVdwVolume        = 3,
  kDclvPolarVolume      = 4,
  kDclvCompactness      = 5,
  kDclvPackingDensity   = 6,
  kNumDclvColumns       = 7,
};

//! Column of a Dclv-family property.
constexpr int dclvColumn(const Property3D property) {
  switch (property) {
    case Property3D::DCLVPolarSurfaceArea:
      return kDclvPolarSurfaceArea;
    case Property3D::DCLVVolume:
      return kDclvVolume;
    case Property3D::DCLVVDWVolume:
      return kDclvVdwVolume;
    case Property3D::DCLVPolarVolume:
      return kDclvPolarVolume;
    case Property3D::DCLVCompactness:
      return kDclvCompactness;
    case Property3D::DCLVPackingDensity:
      return kDclvPackingDensity;
    default:
      return kDclvSurfaceArea;
  }
}

//! Output buffer of each requested DCLV property, indexed by DclvColumn; null when not requested.
template <typename Real> struct DclvOutputs {
  Real* values[kNumDclvColumns] = {};

  bool any() const {
    for (const Real* output : values) {
      if (output != nullptr) {
        return true;
      }
    }
    return false;
  }
};

//! Polar classes DCLVPolarSurfaceArea and DCLVPolarVolume count under @p options, following RDKit's includeAsPolar.
inline uint8_t dclvPolarMask(const DclvOptions& options) {
  uint8_t mask = kDclvPolarNitrogenOxygen;
  if (options.includeSandP) {
    mask |= kDclvPolarSulfurPhosphorus;
  }
  if (options.includeHs) {
    mask |= kDclvPolarHydrogenOnNO;
    if (options.includeSandP) {
      mask |= kDclvPolarHydrogenOnSulfurPhos;
    }
  }
  return mask;
}

//! Surface dots tested per atom sphere.
constexpr int    kDclvNumDots          = 320;
//! RDKit's area of one dot and of the whole dot set on the unit sphere (Code/GraphMol/Descriptors/DCLV_dots.h).
constexpr double kDclvDotArea          = 0.03853087;
constexpr double kDclvStandardArea     = 12.3298;
//! Neighbors cached per warp in shared memory; atoms with more scan the whole conformer for every dot instead.
constexpr int    kDclvNeighborCapacity = 128;
//! Widens the neighbor cutoff (Angstrom). Any atom that buries a dot lies strictly inside the cutoff, so a
//! wider list only adds atoms that bury nothing; the margin keeps rounding from dropping a borderline atom.
constexpr double kDclvNeighborMargin   = 0.01;

/**
 * RDKit's dot set, 320 unit vectors "based on a 320 faced polyhedron", copied from
 * Code/GraphMol/Descriptors/DCLV_dots.h. RDKit stores them as doubles initialized from these float literals, so
 * converting the floats reproduces its values exactly.
 */
__device__ const float kDclvDots[kDclvNumDots][3] = {
  {-0.577350f, -0.577350f, -0.577350f},
  {-0.448278f, -0.547076f, -0.706934f},
  {-0.706934f, -0.448278f, -0.547076f},
  {-0.547076f, -0.706934f, -0.448278f},
  {-0.286211f, -0.762327f, -0.580467f},
  {-0.142204f, -0.816797f, -0.559125f},
  {-0.302648f, -0.639218f, -0.706968f},
  {-0.400803f, -0.798035f, -0.449997f},
  {-0.580467f, -0.286211f, -0.762327f},
  {-0.449997f, -0.400803f, -0.798035f},
  {-0.559125f, -0.142204f, -0.816797f},
  {-0.706968f, -0.302648f, -0.639218f},
  {-0.762327f, -0.580467f, -0.286211f},
  {-0.639218f, -0.706968f, -0.302648f},
  {-0.798035f, -0.449997f, -0.400803f},
  {-0.816797f, -0.559125f, -0.142204f},
  { 0.000000f, -0.356822f, -0.934172f},
  {-0.159858f, -0.436909f, -0.885187f},
  { 0.000000f, -0.178253f, -0.983985f},
  { 0.159858f, -0.436909f, -0.885187f},
  { 0.000000f, -0.653004f, -0.757355f},
  { 0.000000f, -0.762480f, -0.647012f},
  {-0.158817f, -0.584279f, -0.795861f},
  { 0.158817f, -0.584279f, -0.795861f},
  {-0.294256f, -0.176888f, -0.939215f},
  {-0.306166f, -0.345864f, -0.886928f},
  {-0.416921f, -0.087887f, -0.904684f},
  {-0.147349f, -0.088893f, -0.985082f},
  { 0.294256f, -0.176888f, -0.939215f},
  { 0.306166f, -0.345864f, -0.886928f},
  { 0.147349f, -0.088893f, -0.985082f},
  { 0.416921f, -0.087887f, -0.904684f},
  {-0.356822f, -0.934172f,  0.000000f},
  {-0.436909f, -0.885187f, -0.159858f},
  {-0.436909f, -0.885187f,  0.159858f},
  {-0.178253f, -0.983985f,  0.000000f},
  {-0.176888f, -0.939215f, -0.294256f},
  {-0.087887f, -0.904684f, -0.416921f},
  {-0.345864f, -0.886928f, -0.306166f},
  {-0.088893f, -0.985082f, -0.147349f},
  {-0.653004f, -0.757355f,  0.000000f},
  {-0.584279f, -0.795861f, -0.158817f},
  {-0.762480f, -0.647012f,  0.000000f},
  {-0.584279f, -0.795861f,  0.158817f},
  {-0.176888f, -0.939215f,  0.294256f},
  {-0.088893f, -0.985082f,  0.147349f},
  {-0.345864f, -0.886928f,  0.306166f},
  {-0.087887f, -0.904684f,  0.416921f},
  { 0.356822f, -0.934172f,  0.000000f},
  { 0.178253f, -0.983985f,  0.000000f},
  { 0.436909f, -0.885187f,  0.159858f},
  { 0.436909f, -0.885187f, -0.159858f},
  { 0.176888f, -0.939215f, -0.294256f},
  { 0.087887f, -0.904684f, -0.416921f},
  { 0.088893f, -0.985082f, -0.147349f},
  { 0.345864f, -0.886928f, -0.306166f},
  { 0.176888f, -0.939215f,  0.294256f},
  { 0.088893f, -0.985082f,  0.147349f},
  { 0.087887f, -0.904684f,  0.416921f},
  { 0.345864f, -0.886928f,  0.306166f},
  { 0.653004f, -0.757355f,  0.000000f},
  { 0.584279f, -0.795861f, -0.158817f},
  { 0.584279f, -0.795861f,  0.158817f},
  { 0.762480f, -0.647012f,  0.000000f},
  { 0.577350f, -0.577350f, -0.577350f},
  { 0.448278f, -0.547076f, -0.706934f},
  { 0.706934f, -0.448278f, -0.547076f},
  { 0.547076f, -0.706934f, -0.448278f},
  { 0.286211f, -0.762327f, -0.580467f},
  { 0.142204f, -0.816797f, -0.559125f},
  { 0.302648f, -0.639218f, -0.706968f},
  { 0.400803f, -0.798035f, -0.449997f},
  { 0.580467f, -0.286211f, -0.762327f},
  { 0.449997f, -0.400803f, -0.798035f},
  { 0.559125f, -0.142204f, -0.816797f},
  { 0.706968f, -0.302648f, -0.639218f},
  { 0.762327f, -0.580467f, -0.286211f},
  { 0.639218f, -0.706968f, -0.302648f},
  { 0.798035f, -0.449997f, -0.400803f},
  { 0.816797f, -0.559125f, -0.142204f},
  {-0.934172f,  0.000000f, -0.356822f},
  {-0.885187f, -0.159858f, -0.436909f},
  {-0.983985f,  0.000000f, -0.178253f},
  {-0.885187f,  0.159858f, -0.436909f},
  {-0.757355f,  0.000000f, -0.653004f},
  {-0.647012f,  0.000000f, -0.762480f},
  {-0.795861f, -0.158817f, -0.584279f},
  {-0.795861f,  0.158817f, -0.584279f},
  {-0.939215f, -0.294256f, -0.176888f},
  {-0.886928f, -0.306166f, -0.345864f},
  {-0.904684f, -0.416921f, -0.087887f},
  {-0.985082f, -0.147349f, -0.088893f},
  {-0.939215f,  0.294256f, -0.176888f},
  {-0.886928f,  0.306166f, -0.345864f},
  {-0.985082f,  0.147349f, -0.088893f},
  {-0.904684f,  0.416921f, -0.087887f},
  { 0.000000f,  0.356822f, -0.934172f},
  { 0.000000f,  0.178253f, -0.983985f},
  { 0.159858f,  0.436909f, -0.885187f},
  {-0.159858f,  0.436909f, -0.885187f},
  {-0.294256f,  0.176888f, -0.939215f},
  {-0.416921f,  0.087887f, -0.904684f},
  {-0.147349f,  0.088893f, -0.985082f},
  {-0.306166f,  0.345864f, -0.886928f},
  { 0.294256f,  0.176888f, -0.939215f},
  { 0.147349f,  0.088893f, -0.985082f},
  { 0.416921f,  0.087887f, -0.904684f},
  { 0.306166f,  0.345864f, -0.886928f},
  { 0.000000f,  0.653004f, -0.757355f},
  {-0.158817f,  0.584279f, -0.795861f},
  { 0.158817f,  0.584279f, -0.795861f},
  { 0.000000f,  0.762480f, -0.647012f},
  {-0.577350f,  0.577350f, -0.577350f},
  {-0.706934f,  0.448278f, -0.547076f},
  {-0.547076f,  0.706934f, -0.448278f},
  {-0.448278f,  0.547076f, -0.706934f},
  {-0.580467f,  0.286211f, -0.762327f},
  {-0.559125f,  0.142204f, -0.816797f},
  {-0.706968f,  0.302648f, -0.639218f},
  {-0.449997f,  0.400803f, -0.798035f},
  {-0.762327f,  0.580467f, -0.286211f},
  {-0.798035f,  0.449997f, -0.400803f},
  {-0.816797f,  0.559125f, -0.142204f},
  {-0.639218f,  0.706968f, -0.302648f},
  {-0.286211f,  0.762327f, -0.580467f},
  {-0.302648f,  0.639218f, -0.706968f},
  {-0.400803f,  0.798035f, -0.449997f},
  {-0.142204f,  0.816797f, -0.559125f},
  {-0.577350f, -0.577350f,  0.577350f},
  {-0.547076f, -0.706934f,  0.448278f},
  {-0.448278f, -0.547076f,  0.706934f},
  {-0.706934f, -0.448278f,  0.547076f},
  {-0.762327f, -0.580467f,  0.286211f},
  {-0.816797f, -0.559125f,  0.142204f},
  {-0.639218f, -0.706968f,  0.302648f},
  {-0.798035f, -0.449997f,  0.400803f},
  {-0.286211f, -0.762327f,  0.580467f},
  {-0.400803f, -0.798035f,  0.449997f},
  {-0.142204f, -0.816797f,  0.559125f},
  {-0.302648f, -0.639218f,  0.706968f},
  {-0.580467f, -0.286211f,  0.762327f},
  {-0.706968f, -0.302648f,  0.639218f},
  {-0.449997f, -0.400803f,  0.798035f},
  {-0.559125f, -0.142204f,  0.816797f},
  {-0.934172f,  0.000000f,  0.356822f},
  {-0.983985f,  0.000000f,  0.178253f},
  {-0.885187f,  0.159858f,  0.436909f},
  {-0.885187f, -0.159858f,  0.436909f},
  {-0.939215f, -0.294256f,  0.176888f},
  {-0.904684f, -0.416921f,  0.087887f},
  {-0.985082f, -0.147349f,  0.088893f},
  {-0.886928f, -0.306166f,  0.345864f},
  {-0.939215f,  0.294256f,  0.176888f},
  {-0.985082f,  0.147349f,  0.088893f},
  {-0.904684f,  0.416921f,  0.087887f},
  {-0.886928f,  0.306166f,  0.345864f},
  {-0.757355f,  0.000000f,  0.653004f},
  {-0.795861f, -0.158817f,  0.584279f},
  {-0.795861f,  0.158817f,  0.584279f},
  {-0.647012f,  0.000000f,  0.762480f},
  { 0.934172f,  0.000000f, -0.356822f},
  { 0.885187f, -0.159858f, -0.436909f},
  { 0.983985f,  0.000000f, -0.178253f},
  { 0.885187f,  0.159858f, -0.436909f},
  { 0.757355f,  0.000000f, -0.653004f},
  { 0.647012f,  0.000000f, -0.762480f},
  { 0.795861f, -0.158817f, -0.584279f},
  { 0.795861f,  0.158817f, -0.584279f},
  { 0.939215f, -0.294256f, -0.176888f},
  { 0.886928f, -0.306166f, -0.345864f},
  { 0.904684f, -0.416921f, -0.087887f},
  { 0.985082f, -0.147349f, -0.088893f},
  { 0.939215f,  0.294256f, -0.176888f},
  { 0.886928f,  0.306166f, -0.345864f},
  { 0.985082f,  0.147349f, -0.088893f},
  { 0.904684f,  0.416921f, -0.087887f},
  { 0.577350f,  0.577350f, -0.577350f},
  { 0.448278f,  0.547076f, -0.706934f},
  { 0.547076f,  0.706934f, -0.448278f},
  { 0.706934f,  0.448278f, -0.547076f},
  { 0.580467f,  0.286211f, -0.762327f},
  { 0.559125f,  0.142204f, -0.816797f},
  { 0.449997f,  0.400803f, -0.798035f},
  { 0.706968f,  0.302648f, -0.639218f},
  { 0.286211f,  0.762327f, -0.580467f},
  { 0.302648f,  0.639218f, -0.706968f},
  { 0.142204f,  0.816797f, -0.559125f},
  { 0.400803f,  0.798035f, -0.449997f},
  { 0.762327f,  0.580467f, -0.286211f},
  { 0.798035f,  0.449997f, -0.400803f},
  { 0.639218f,  0.706968f, -0.302648f},
  { 0.816797f,  0.559125f, -0.142204f},
  { 0.577350f, -0.577350f,  0.577350f},
  { 0.547076f, -0.706934f,  0.448278f},
  { 0.706934f, -0.448278f,  0.547076f},
  { 0.448278f, -0.547076f,  0.706934f},
  { 0.286211f, -0.762327f,  0.580467f},
  { 0.142204f, -0.816797f,  0.559125f},
  { 0.400803f, -0.798035f,  0.449997f},
  { 0.302648f, -0.639218f,  0.706968f},
  { 0.762327f, -0.580467f,  0.286211f},
  { 0.639218f, -0.706968f,  0.302648f},
  { 0.816797f, -0.559125f,  0.142204f},
  { 0.798035f, -0.449997f,  0.400803f},
  { 0.580467f, -0.286211f,  0.762327f},
  { 0.449997f, -0.400803f,  0.798035f},
  { 0.706968f, -0.302648f,  0.639218f},
  { 0.559125f, -0.142204f,  0.816797f},
  { 0.000000f, -0.356822f,  0.934172f},
  {-0.159858f, -0.436909f,  0.885187f},
  { 0.000000f, -0.178253f,  0.983985f},
  { 0.159858f, -0.436909f,  0.885187f},
  { 0.000000f, -0.653004f,  0.757355f},
  { 0.000000f, -0.762480f,  0.647012f},
  {-0.158817f, -0.584279f,  0.795861f},
  { 0.158817f, -0.584279f,  0.795861f},
  {-0.294256f, -0.176888f,  0.939215f},
  {-0.306166f, -0.345864f,  0.886928f},
  {-0.416921f, -0.087887f,  0.904684f},
  {-0.147349f, -0.088893f,  0.985082f},
  { 0.294256f, -0.176888f,  0.939215f},
  { 0.306166f, -0.345864f,  0.886928f},
  { 0.147349f, -0.088893f,  0.985082f},
  { 0.416921f, -0.087887f,  0.904684f},
  { 0.934172f,  0.000000f,  0.356822f},
  { 0.885187f, -0.159858f,  0.436909f},
  { 0.885187f,  0.159858f,  0.436909f},
  { 0.983985f,  0.000000f,  0.178253f},
  { 0.939215f, -0.294256f,  0.176888f},
  { 0.904684f, -0.416921f,  0.087887f},
  { 0.886928f, -0.306166f,  0.345864f},
  { 0.985082f, -0.147349f,  0.088893f},
  { 0.757355f,  0.000000f,  0.653004f},
  { 0.795861f, -0.158817f,  0.584279f},
  { 0.647012f,  0.000000f,  0.762480f},
  { 0.795861f,  0.158817f,  0.584279f},
  { 0.939215f,  0.294256f,  0.176888f},
  { 0.985082f,  0.147349f,  0.088893f},
  { 0.886928f,  0.306166f,  0.345864f},
  { 0.904684f,  0.416921f,  0.087887f},
  {-0.577350f,  0.577350f,  0.577350f},
  {-0.706934f,  0.448278f,  0.547076f},
  {-0.448278f,  0.547076f,  0.706934f},
  {-0.547076f,  0.706934f,  0.448278f},
  {-0.762327f,  0.580467f,  0.286211f},
  {-0.816797f,  0.559125f,  0.142204f},
  {-0.798035f,  0.449997f,  0.400803f},
  {-0.639218f,  0.706968f,  0.302648f},
  {-0.580467f,  0.286211f,  0.762327f},
  {-0.706968f,  0.302648f,  0.639218f},
  {-0.559125f,  0.142204f,  0.816797f},
  {-0.449997f,  0.400803f,  0.798035f},
  {-0.286211f,  0.762327f,  0.580467f},
  {-0.400803f,  0.798035f,  0.449997f},
  {-0.302648f,  0.639218f,  0.706968f},
  {-0.142204f,  0.816797f,  0.559125f},
  {-0.356822f,  0.934172f,  0.000000f},
  {-0.436909f,  0.885187f, -0.159858f},
  {-0.178253f,  0.983985f,  0.000000f},
  {-0.436909f,  0.885187f,  0.159858f},
  {-0.653004f,  0.757355f,  0.000000f},
  {-0.762480f,  0.647012f,  0.000000f},
  {-0.584279f,  0.795861f, -0.158817f},
  {-0.584279f,  0.795861f,  0.158817f},
  {-0.176888f,  0.939215f, -0.294256f},
  {-0.345864f,  0.886928f, -0.306166f},
  {-0.087887f,  0.904684f, -0.416921f},
  {-0.088893f,  0.985082f, -0.147349f},
  {-0.176888f,  0.939215f,  0.294256f},
  {-0.345864f,  0.886928f,  0.306166f},
  {-0.088893f,  0.985082f,  0.147349f},
  {-0.087887f,  0.904684f,  0.416921f},
  { 0.356822f,  0.934172f,  0.000000f},
  { 0.178253f,  0.983985f,  0.000000f},
  { 0.436909f,  0.885187f,  0.159858f},
  { 0.436909f,  0.885187f, -0.159858f},
  { 0.176888f,  0.939215f, -0.294256f},
  { 0.087887f,  0.904684f, -0.416921f},
  { 0.088893f,  0.985082f, -0.147349f},
  { 0.345864f,  0.886928f, -0.306166f},
  { 0.176888f,  0.939215f,  0.294256f},
  { 0.088893f,  0.985082f,  0.147349f},
  { 0.087887f,  0.904684f,  0.416921f},
  { 0.345864f,  0.886928f,  0.306166f},
  { 0.653004f,  0.757355f,  0.000000f},
  { 0.584279f,  0.795861f, -0.158817f},
  { 0.584279f,  0.795861f,  0.158817f},
  { 0.762480f,  0.647012f,  0.000000f},
  { 0.000000f,  0.356822f,  0.934172f},
  {-0.159858f,  0.436909f,  0.885187f},
  { 0.159858f,  0.436909f,  0.885187f},
  { 0.000000f,  0.178253f,  0.983985f},
  {-0.294256f,  0.176888f,  0.939215f},
  {-0.416921f,  0.087887f,  0.904684f},
  {-0.306166f,  0.345864f,  0.886928f},
  {-0.147349f,  0.088893f,  0.985082f},
  { 0.000000f,  0.653004f,  0.757355f},
  {-0.158817f,  0.584279f,  0.795861f},
  { 0.000000f,  0.762480f,  0.647012f},
  { 0.158817f,  0.584279f,  0.795861f},
  { 0.294256f,  0.176888f,  0.939215f},
  { 0.147349f,  0.088893f,  0.985082f},
  { 0.306166f,  0.345864f,  0.886928f},
  { 0.416921f,  0.087887f,  0.904684f},
  { 0.577350f,  0.577350f,  0.577350f},
  { 0.448278f,  0.547076f,  0.706934f},
  { 0.706934f,  0.448278f,  0.547076f},
  { 0.547076f,  0.706934f,  0.448278f},
  { 0.286211f,  0.762327f,  0.580467f},
  { 0.142204f,  0.816797f,  0.559125f},
  { 0.302648f,  0.639218f,  0.706968f},
  { 0.400803f,  0.798035f,  0.449997f},
  { 0.580467f,  0.286211f,  0.762327f},
  { 0.449997f,  0.400803f,  0.798035f},
  { 0.559125f,  0.142204f,  0.816797f},
  { 0.706968f,  0.302648f,  0.639218f},
  { 0.762327f,  0.580467f,  0.286211f},
  { 0.639218f,  0.706968f,  0.302648f},
  { 0.798035f,  0.449997f,  0.400803f},
  { 0.816797f,  0.559125f,  0.142204f}
};

//! A cached neighbor: its position relative to the current atom and its radius.
template <typename Real> struct DclvNeighbor {
  Real x;
  Real y;
  Real z;
  Real radius;
};

//! Whether a sphere of @p reach around (@p x, @p y, @p z) contains the point (@p vx, @p vy, @p vz); strict, as in
//! RDKit's testPoint.
template <typename Real>
__device__ __forceinline__ bool
dclvCovers(const Real x, const Real y, const Real z, const Real reach, const Real vx, const Real vy, const Real vz) {
  const Real dx = x - vx;
  const Real dy = y - vy;
  const Real dz = z - vz;
  return dx * dx + dy * dy + dz * dz < reach * reach;
}

//! Exposed dots of one atom sphere, for the probe-expanded sphere (surface area and volume) and the bare van der
//! Waals sphere (VDW volume): dot counts and sums of the exposed unit vectors.
template <typename Real> struct DclvAtomDots {
  Real probeCount = 0;
  Real probeX     = 0;
  Real probeY     = 0;
  Real probeZ     = 0;
  Real vdwCount   = 0;
  Real vdwX       = 0;
  Real vdwY       = 0;
  Real vdwZ       = 0;
};

/**
 * @brief Warp-collective. Tests atom @p atomIdx's dots against the other atoms and returns the warp-wide exposed
 *        dot counts and unit-vector sums to every lane.
 *
 * With @p kProbe, a dot on the probe-expanded sphere (radius r + probe) is exposed unless it lies inside another
 * atom's probe-expanded sphere; with @p kVdw, a dot on the bare sphere (radius r) is exposed unless it lies inside
 * another atom's bare sphere. Both use the neighbors of RDKit's findNeighbours, atoms closer than
 * r_i + 2 probe + r_j, which are the only atoms that can bury either kind of dot. Disabled kinds stay zero.
 */
template <typename Real, bool kProbe, bool kVdw>
__device__ __forceinline__ DclvAtomDots<Real> dclvAtomDots(const Real*         centered,
                                                           const double*       radii,
                                                           const int           numAtoms,
                                                           const int           atomIdx,
                                                           const Real          probe,
                                                           const int           lane,
                                                           DclvNeighbor<Real>* neighbors) {
  const Real px     = centered[atomIdx * 3 + 0];
  const Real py     = centered[atomIdx * 3 + 1];
  const Real pz     = centered[atomIdx * 3 + 2];
  const Real radius = static_cast<Real>(radii[atomIdx]);
  const Real range  = radius + Real(2) * probe + static_cast<Real>(kDclvNeighborMargin);

  // Every lane has finished reading the previous atom's neighbors before they are overwritten.
  __syncwarp();
  int numNeighbors = 0;
  for (int base = 0; base < numAtoms; base += kWarpSize) {
    const int          other = base + lane;
    DclvNeighbor<Real> candidate{};
    bool               isNeighbor = false;
    if (other < numAtoms && other != atomIdx) {
      candidate.radius = static_cast<Real>(radii[other]);
      candidate.x      = centered[other * 3 + 0] - px;
      candidate.y      = centered[other * 3 + 1] - py;
      candidate.z      = centered[other * 3 + 2] - pz;
      isNeighbor =
        dclvCovers(candidate.x, candidate.y, candidate.z, range + candidate.radius, Real(0), Real(0), Real(0));
    }
    const unsigned ballot = __ballot_sync(0xffffffffu, isNeighbor);
    const int      slot   = numNeighbors + __popc(ballot & ((1u << lane) - 1u));
    if (isNeighbor && slot < kDclvNeighborCapacity) {
      neighbors[slot] = candidate;
    }
    numNeighbors += __popc(ballot);
  }
  __syncwarp();
  const bool cached = numNeighbors <= kDclvNeighborCapacity;

  DclvAtomDots<Real> dots;
  const Real         probeScale = radius + probe;
  for (int dotIdx = lane; dotIdx < kDclvNumDots; dotIdx += kWarpSize) {
    const Real ux        = static_cast<Real>(kDclvDots[dotIdx][0]);
    const Real uy        = static_cast<Real>(kDclvDots[dotIdx][1]);
    const Real uz        = static_cast<Real>(kDclvDots[dotIdx][2]);
    // The dot on each sphere, relative to the atom, stays in registers across the neighbor loop.
    const Real probeDotX = ux * probeScale;
    const Real probeDotY = uy * probeScale;
    const Real probeDotZ = uz * probeScale;
    const Real vdwDotX   = ux * radius;
    const Real vdwDotY   = uy * radius;
    const Real vdwDotZ   = uz * radius;
    bool       probeOpen = kProbe;
    bool       vdwOpen   = kVdw;
    auto       test      = [&](const DclvNeighbor<Real>& other) {
      if constexpr (kProbe) {
        probeOpen =
          probeOpen && !dclvCovers(other.x, other.y, other.z, other.radius + probe, probeDotX, probeDotY, probeDotZ);
      }
      if constexpr (kVdw) {
        vdwOpen = vdwOpen && !dclvCovers(other.x, other.y, other.z, other.radius, vdwDotX, vdwDotY, vdwDotZ);
      }
    };
    if (cached) {
      for (int neighborIdx = 0; neighborIdx < numNeighbors && (probeOpen || vdwOpen); ++neighborIdx) {
        test(neighbors[neighborIdx]);
      }
    } else {
      for (int other = 0; other < numAtoms && (probeOpen || vdwOpen); ++other) {
        if (other != atomIdx) {
          test(DclvNeighbor<Real>{centered[other * 3 + 0] - px,
                                  centered[other * 3 + 1] - py,
                                  centered[other * 3 + 2] - pz,
                                  static_cast<Real>(radii[other])});
        }
      }
    }
    if (probeOpen) {
      dots.probeCount += Real(1);
      dots.probeX += ux;
      dots.probeY += uy;
      dots.probeZ += uz;
    }
    if (vdwOpen) {
      dots.vdwCount += Real(1);
      dots.vdwX += ux;
      dots.vdwY += uy;
      dots.vdwZ += uz;
    }
  }
  if constexpr (kProbe) {
    dots.probeCount = groupAllReduceSum<kWarpSize>(dots.probeCount);
    dots.probeX     = groupAllReduceSum<kWarpSize>(dots.probeX);
    dots.probeY     = groupAllReduceSum<kWarpSize>(dots.probeY);
    dots.probeZ     = groupAllReduceSum<kWarpSize>(dots.probeZ);
  }
  if constexpr (kVdw) {
    dots.vdwCount = groupAllReduceSum<kWarpSize>(dots.vdwCount);
    dots.vdwX     = groupAllReduceSum<kWarpSize>(dots.vdwX);
    dots.vdwY     = groupAllReduceSum<kWarpSize>(dots.vdwY);
    dots.vdwZ     = groupAllReduceSum<kWarpSize>(dots.vdwZ);
  }
  return dots;
}

/**
 * @brief RDKit's DoubleCubicLatticeVolume for every conformer; one warp per conformer.
 *
 * Each warp centers its conformer on the mean atom position (RDKit's centreOfGravity) in float64 (see
 * centeredPosition()), then walks the atoms: it gathers each atom's neighbors in shared memory and each lane tests
 * every kWarpSize-th dot. Per atom, RDKit's getAtomSurfaceArea gives
 * area = exposed * dotArea * 4 pi f^2 / standardArea and its getAtomVolume (Gauss-Ostrogradskii) gives
 * volume = f^2 ((x - centre) . sum of exposed dots + f exposed) 4 pi / (3 * 320), with f = r + probe for the
 * surface and volume (@p kProbe) and f = r for the VDW volume (@p kVdw). Only requested outputs are written. Rows
 * with a non-finite coordinate or a zero-radius (dummy) atom produce NaN.
 */
template <typename Real, bool kProbe, bool kVdw>
__global__ void dclv3DKernel(const DeviceCoordView        coordinates,
                             const Property3DDeviceInputs inputs,
                             const Real                   probe,
                             const uint8_t                polarMask,
                             Real* __restrict__ centered,
                             const DclvOutputs<Real> outputs) {
  __shared__ DclvNeighbor<Real> blockNeighbors[kWarpsPerBlock][kDclvNeighborCapacity];
  const int                     lane         = static_cast<int>(threadIdx.x) % kWarpSize;
  const int                     warpInBlock  = static_cast<int>(threadIdx.x) / kWarpSize;
  const int                     conformerIdx = blockIdx.x * kWarpsPerBlock + warpInBlock;
  if (conformerIdx >= coordinates.numConformers) {
    return;  // Uniform across the warp.
  }
  auto writeRow = [&](const Real(&values)[kNumDclvColumns]) {
    if (lane == 0) {
      for (int column = 0; column < kNumDclvColumns; ++column) {
        if (outputs.values[column] != nullptr) {
          outputs.values[column][conformerIdx] = values[column];
        }
      }
    }
  };
  const ConformerAtoms atoms   = loadConformer(coordinates, nullptr, inputs.moleculeAtomStarts, conformerIdx);
  bool                 valid   = atoms.valid;
  const double*        radii   = nullptr;
  const uint8_t*       polar   = nullptr;
  double               centreX = 0;
  double               centreY = 0;
  double               centreZ = 0;
  if (valid) {
    const int moleculeStart = inputs.moleculeAtomStarts[coordinates.molIndices[conformerIdx]];
    radii                   = inputs.vdwRadii + moleculeStart;
    polar                   = inputs.dclvPolarClasses + moleculeStart;
    double sumX             = 0;
    double sumY             = 0;
    double sumZ             = 0;
    bool   usable           = true;
    for (int atomIdx = lane; atomIdx < atoms.numAtoms; atomIdx += kWarpSize) {
      const double x = atoms.positions[atomIdx * 3 + 0];
      const double y = atoms.positions[atomIdx * 3 + 1];
      const double z = atoms.positions[atomIdx * 3 + 2];
      usable         = usable && isfinite(x) && isfinite(y) && isfinite(z) && radii[atomIdx] != 0.0;
      sumX += x;
      sumY += y;
      sumZ += z;
    }
    valid                     = __all_sync(0xffffffffu, usable);
    const double inverseAtoms = 1.0 / static_cast<double>(atoms.numAtoms);
    centreX                   = groupAllReduceSum<kWarpSize>(sumX) * inverseAtoms;
    centreY                   = groupAllReduceSum<kWarpSize>(sumY) * inverseAtoms;
    centreZ                   = groupAllReduceSum<kWarpSize>(sumZ) * inverseAtoms;
  }
  if (!valid) {
    Real nanRow[kNumDclvColumns];
    for (Real& value : nanRow) {
      value = static_cast<Real>(nan(""));
    }
    writeRow(nanRow);
    return;
  }

  Real* const conformerCentered = centered + (atoms.positions - coordinates.positions);
  for (int atomIdx = lane; atomIdx < atoms.numAtoms; atomIdx += kWarpSize) {
    centeredPosition(atoms.positions,
                     atomIdx,
                     centreX,
                     centreY,
                     centreZ,
                     conformerCentered[atomIdx * 3 + 0],
                     conformerCentered[atomIdx * 3 + 1],
                     conformerCentered[atomIdx * 3 + 2]);
  }
  __syncwarp();

  constexpr Real kAreaScale   = static_cast<Real>(kDclvDotArea * 4.0 * M_PI / kDclvStandardArea);
  constexpr Real kVolumeScale = static_cast<Real>(4.0 / 3.0 * M_PI / kDclvNumDots);
  Real           surfaceArea  = 0;
  Real           polarArea    = 0;
  Real           volume       = 0;
  Real           polarVolume  = 0;
  Real           vdwVolume    = 0;
  for (int atomIdx = 0; atomIdx < atoms.numAtoms; ++atomIdx) {
    const DclvAtomDots<Real> dots   = dclvAtomDots<Real, kProbe, kVdw>(conformerCentered,
                                                                     radii,
                                                                     atoms.numAtoms,
                                                                     atomIdx,
                                                                     probe,
                                                                     lane,
                                                                     blockNeighbors[warpInBlock]);
    const Real               x      = conformerCentered[atomIdx * 3 + 0];
    const Real               y      = conformerCentered[atomIdx * 3 + 1];
    const Real               z      = conformerCentered[atomIdx * 3 + 2];
    const Real               radius = static_cast<Real>(radii[atomIdx]);
    if constexpr (kProbe) {
      const Real probeScale = radius + probe;
      const Real atomArea   = dots.probeCount * kAreaScale * probeScale * probeScale;
      const Real atomVolume = probeScale * probeScale *
                              (x * dots.probeX + y * dots.probeY + z * dots.probeZ + probeScale * dots.probeCount) *
                              kVolumeScale;
      surfaceArea += atomArea;
      volume += atomVolume;
      if ((polar[atomIdx] & polarMask) != 0) {
        polarArea += atomArea;
        polarVolume += atomVolume;
      }
    }
    if constexpr (kVdw) {
      vdwVolume +=
        radius * radius * (x * dots.vdwX + y * dots.vdwY + z * dots.vdwZ + radius * dots.vdwCount) * kVolumeScale;
    }
  }
  const Real values[kNumDclvColumns] = {
    surfaceArea,
    polarArea,
    volume,
    vdwVolume,
    polarVolume,
    surfaceArea / cbrt(Real(36 * M_PI) * volume * volume),
    vdwVolume / volume,
  };
  writeRow(values);
}

template <typename Real, bool kProbe, bool kVdw>
void launchDclvKernel(const DeviceCoordView&        coordinates,
                      const Property3DDeviceInputs& inputs,
                      const DclvOptions&            options,
                      Real*                         centered,
                      const DclvOutputs<Real>&      outputs,
                      const cudaStream_t            stream) {
  const int numBlocks = (coordinates.numConformers + kWarpsPerBlock - 1) / kWarpsPerBlock;
  dclv3DKernel<Real, kProbe, kVdw><<<numBlocks, kBlockSize, 0, stream>>>(coordinates,
                                                                         inputs,
                                                                         static_cast<Real>(options.probeRadius),
                                                                         dclvPolarMask(options),
                                                                         centered,
                                                                         outputs);
  cudaCheckError(cudaGetLastError());
}

//! Launches the DCLV kernel when any DCLV property is requested. The probe-expanded dots serve every output but
//! DCLVVDWVolume; the bare-sphere dots serve DCLVVDWVolume and DCLVPackingDensity. Each is tested only when needed.
template <typename Real>
void launchDclvProperties(const DeviceCoordView&        coordinates,
                          const Property3DDeviceInputs& inputs,
                          const DclvOptions&            options,
                          const DclvOutputs<Real>&      outputs,
                          const cudaStream_t            stream) {
  if (coordinates.numConformers == 0 || !outputs.any()) {
    return;
  }
  const bool vdw   = outputs.values[kDclvVdwVolume] != nullptr || outputs.values[kDclvPackingDensity] != nullptr;
  bool       probe = false;
  for (int column = 0; column < kNumDclvColumns; ++column) {
    probe |= column != kDclvVdwVolume && outputs.values[column] != nullptr;
  }
  AsyncDeviceVector<Real> centered(static_cast<size_t>(coordinates.numAtoms) * 3, stream);
  if (probe && vdw) {
    launchDclvKernel<Real, true, true>(coordinates, inputs, options, centered.data(), outputs, stream);
  } else if (probe) {
    launchDclvKernel<Real, true, false>(coordinates, inputs, options, centered.data(), outputs, stream);
  } else {
    launchDclvKernel<Real, false, true>(coordinates, inputs, options, centered.data(), outputs, stream);
  }
}

}  // namespace nvMolKit::descriptors3d_detail

#endif  // NVMOLKIT_DESCRIPTORS3D_DCLV_CUH
