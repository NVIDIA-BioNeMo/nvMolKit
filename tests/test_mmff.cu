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

#include <ForceField/AngleConstraints.h>
#include <ForceField/DistanceConstraints.h>
#include <ForceField/ForceField.h>
#include <ForceField/MMFF/PositionConstraint.h>
#include <ForceField/MMFF/TorsionConstraint.h>
#include <gmock/gmock.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/FileParsers/MolSupplier.h>
#include <GraphMol/ForceFieldHelpers/MMFF/AtomTyper.h>
#include <GraphMol/ForceFieldHelpers/MMFF/Builder.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <functional>
#include <random>
#include <stdexcept>

#include "rdkit_extensions/mmff_flattened_builder.h"
#include "rdkit_extensions/mmff_optimize.h"
#include "src/forcefields/ff_utils.h"
#include "src/forcefields/forcefield_constraints.h"
#include "src/forcefields/kernel_utils.cuh"
#include "src/forcefields/mmff.h"
#include "src/forcefields/mmff_batched_forcefield.h"
#include "src/forcefields/mmff_kernels.h"
#include "src/minimizer/mmff_minimize.h"
#include "src/utils/device.h"
#include "tests/test_utils.h"
using namespace nvMolKit::MMFF;

constexpr double GRAD_TOL       = 1.0e-4;
// Tighter energy tolerance for function level tests
constexpr double FUNCTION_E_TOL = 5.0e-5;
// General energy tolerance for minimized systems
constexpr double MINIMIZE_E_TOL = 1.0e-3;
constexpr double EDGE_CASE_TOL  = 1.0e-1;

enum class FFTerm {
  BondStretch,
  AngleBend,
  StretchBend,
  OopBend,
  Torsion,
  VdW,
  Elec,
  DistanceConstraint,
  PositionConstraint,
  AngleConstraint,
  TorsionConstraint,
};
constexpr std::array<FFTerm, 7>   allTerms = {FFTerm::BondStretch,
                                              FFTerm::AngleBend,
                                              FFTerm::StretchBend,
                                              FFTerm::OopBend,
                                              FFTerm::Torsion,
                                              FFTerm::VdW,
                                              FFTerm::Elec};
static std::array<std::string, 7> contribNames =
  {"BondStretch", "AngleBend", "StretchBend", "OopBend", "Torsion", "VdW", "Elec"};

double              getCombinedEnergyViaForcefield(const BatchedMolecularSystemHost& systemHost);
std::vector<double> getCombinedGradientViaForcefield(const BatchedMolecularSystemHost& systemHost);

double getEnergyTerm(BatchedMolecularDeviceBuffers& deviceFF, const FFTerm& term) {
  switch (term) {
    case FFTerm::BondStretch:
      if (deviceFF.contribs.bondTerms.idx1.size() == 0) {
        return 0.0;
      }
      CHECK_CUDA_RETURN(launchBondStretchEnergyKernel(deviceFF.contribs.bondTerms.idx1.size(),
                                                      deviceFF.contribs.bondTerms.idx1.data(),
                                                      deviceFF.contribs.bondTerms.idx2.data(),
                                                      deviceFF.contribs.bondTerms.r0.data(),
                                                      deviceFF.contribs.bondTerms.kb.data(),
                                                      deviceFF.positions.data(),
                                                      deviceFF.energyBuffer.data(),
                                                      deviceFF.indices.energyBufferStarts.data(),
                                                      deviceFF.indices.atomIdxToBatchIdx.data(),
                                                      deviceFF.indices.bondTermStarts.data()));
      break;
    case FFTerm::AngleBend:
      if (deviceFF.contribs.angleTerms.idx1.size() == 0) {
        return 0.0;
      }
      CHECK_CUDA_RETURN(launchAngleBendEnergyKernel(deviceFF.contribs.angleTerms.idx1.size(),
                                                    deviceFF.contribs.angleTerms.idx1.data(),
                                                    deviceFF.contribs.angleTerms.idx2.data(),
                                                    deviceFF.contribs.angleTerms.idx3.data(),
                                                    deviceFF.contribs.angleTerms.theta0.data(),
                                                    deviceFF.contribs.angleTerms.ka.data(),
                                                    deviceFF.contribs.angleTerms.isLinear.data(),
                                                    deviceFF.positions.data(),
                                                    deviceFF.energyBuffer.data(),
                                                    deviceFF.indices.energyBufferStarts.data(),
                                                    deviceFF.indices.atomIdxToBatchIdx.data(),
                                                    deviceFF.indices.angleTermStarts.data()));
      break;
    case FFTerm::StretchBend:
      if (deviceFF.contribs.bendTerms.idx1.size() == 0) {
        return 0.0;
      }
      CHECK_CUDA_RETURN(launchBendStretchEnergyKernel(deviceFF.contribs.bendTerms.idx1.size(),
                                                      deviceFF.contribs.bendTerms.idx1.data(),
                                                      deviceFF.contribs.bendTerms.idx2.data(),
                                                      deviceFF.contribs.bendTerms.idx3.data(),
                                                      deviceFF.contribs.bendTerms.theta0.data(),
                                                      deviceFF.contribs.bendTerms.restLen1.data(),
                                                      deviceFF.contribs.bendTerms.restLen2.data(),
                                                      deviceFF.contribs.bendTerms.forceConst1.data(),
                                                      deviceFF.contribs.bendTerms.forceConst2.data(),
                                                      deviceFF.positions.data(),
                                                      deviceFF.energyBuffer.data(),
                                                      deviceFF.indices.energyBufferStarts.data(),
                                                      deviceFF.indices.atomIdxToBatchIdx.data(),
                                                      deviceFF.indices.bendTermStarts.data()));
      break;
    case FFTerm::OopBend:
      if (deviceFF.contribs.oopTerms.idx1.size() == 0) {
        return 0.0;
      }
      CHECK_CUDA_RETURN(launchOopBendEnergyKernel(deviceFF.contribs.oopTerms.idx1.size(),
                                                  deviceFF.contribs.oopTerms.idx1.data(),
                                                  deviceFF.contribs.oopTerms.idx2.data(),
                                                  deviceFF.contribs.oopTerms.idx3.data(),
                                                  deviceFF.contribs.oopTerms.idx4.data(),
                                                  deviceFF.contribs.oopTerms.koop.data(),
                                                  deviceFF.positions.data(),
                                                  deviceFF.energyBuffer.data(),
                                                  deviceFF.indices.energyBufferStarts.data(),
                                                  deviceFF.indices.atomIdxToBatchIdx.data(),
                                                  deviceFF.indices.oopTermStarts.data()));
      break;
    case FFTerm::Torsion:
      if (deviceFF.contribs.torsionTerms.idx1.size() == 0) {
        return 0.0;
      }
      CHECK_CUDA_RETURN(launchTorsionEnergyKernel(deviceFF.contribs.torsionTerms.idx1.size(),
                                                  deviceFF.contribs.torsionTerms.idx1.data(),
                                                  deviceFF.contribs.torsionTerms.idx2.data(),
                                                  deviceFF.contribs.torsionTerms.idx3.data(),
                                                  deviceFF.contribs.torsionTerms.idx4.data(),
                                                  deviceFF.contribs.torsionTerms.V1.data(),
                                                  deviceFF.contribs.torsionTerms.V2.data(),
                                                  deviceFF.contribs.torsionTerms.V3.data(),
                                                  deviceFF.positions.data(),
                                                  deviceFF.energyBuffer.data(),
                                                  deviceFF.indices.energyBufferStarts.data(),
                                                  deviceFF.indices.atomIdxToBatchIdx.data(),
                                                  deviceFF.indices.torsionTermStarts.data()));
      break;
    case FFTerm::VdW:
      if (deviceFF.contribs.vdwTerms.idx1.size() == 0) {
        return 0.0;
      }
      CHECK_CUDA_RETURN(launchVdwEnergyKernel(deviceFF.contribs.vdwTerms.idx1.size(),
                                              deviceFF.contribs.vdwTerms.idx1.data(),
                                              deviceFF.contribs.vdwTerms.idx2.data(),
                                              deviceFF.contribs.vdwTerms.R_ij_star.data(),
                                              deviceFF.contribs.vdwTerms.wellDepth.data(),
                                              deviceFF.positions.data(),
                                              deviceFF.energyBuffer.data(),
                                              deviceFF.indices.energyBufferStarts.data(),
                                              deviceFF.indices.atomIdxToBatchIdx.data(),
                                              deviceFF.indices.vdwTermStarts.data()));
      break;
    case FFTerm::Elec:
      if (deviceFF.contribs.eleTerms.idx1.size() == 0) {
        return 0.0;
      }
      CHECK_CUDA_RETURN(launchEleEnergyKernel(deviceFF.contribs.eleTerms.idx1.size(),
                                              deviceFF.contribs.eleTerms.idx1.data(),
                                              deviceFF.contribs.eleTerms.idx2.data(),
                                              deviceFF.contribs.eleTerms.chargeTerm.data(),
                                              deviceFF.contribs.eleTerms.dielModel.data(),
                                              deviceFF.contribs.eleTerms.is1_4.data(),
                                              deviceFF.positions.data(),
                                              deviceFF.energyBuffer.data(),
                                              deviceFF.indices.energyBufferStarts.data(),
                                              deviceFF.indices.atomIdxToBatchIdx.data(),
                                              deviceFF.indices.eleTermStarts.data()));
      break;
    case FFTerm::DistanceConstraint:
      if (deviceFF.contribs.distanceConstraintTerms.idx1.size() == 0) {
        return 0.0;
      }
      CHECK_CUDA_RETURN(
        launchDistanceConstraintEnergyKernel(deviceFF.contribs.distanceConstraintTerms.idx1.size(),
                                             deviceFF.contribs.distanceConstraintTerms.idx1.data(),
                                             deviceFF.contribs.distanceConstraintTerms.idx2.data(),
                                             deviceFF.contribs.distanceConstraintTerms.minLen.data(),
                                             deviceFF.contribs.distanceConstraintTerms.maxLen.data(),
                                             deviceFF.contribs.distanceConstraintTerms.forceConstant.data(),
                                             deviceFF.positions.data(),
                                             deviceFF.energyBuffer.data(),
                                             deviceFF.indices.energyBufferStarts.data(),
                                             deviceFF.indices.atomIdxToBatchIdx.data(),
                                             deviceFF.indices.distanceConstraintTermStarts.data()));
      break;
    case FFTerm::PositionConstraint:
      if (deviceFF.contribs.positionConstraintTerms.idx.size() == 0) {
        return 0.0;
      }
      CHECK_CUDA_RETURN(
        launchPositionConstraintEnergyKernel(deviceFF.contribs.positionConstraintTerms.idx.size(),
                                             deviceFF.contribs.positionConstraintTerms.idx.data(),
                                             deviceFF.contribs.positionConstraintTerms.refX.data(),
                                             deviceFF.contribs.positionConstraintTerms.refY.data(),
                                             deviceFF.contribs.positionConstraintTerms.refZ.data(),
                                             deviceFF.contribs.positionConstraintTerms.maxDispl.data(),
                                             deviceFF.contribs.positionConstraintTerms.forceConstant.data(),
                                             deviceFF.positions.data(),
                                             deviceFF.energyBuffer.data(),
                                             deviceFF.indices.energyBufferStarts.data(),
                                             deviceFF.indices.atomIdxToBatchIdx.data(),
                                             deviceFF.indices.positionConstraintTermStarts.data()));
      break;
    case FFTerm::AngleConstraint:
      if (deviceFF.contribs.angleConstraintTerms.idx1.size() == 0) {
        return 0.0;
      }
      CHECK_CUDA_RETURN(launchAngleConstraintEnergyKernel(deviceFF.contribs.angleConstraintTerms.idx1.size(),
                                                          deviceFF.contribs.angleConstraintTerms.idx1.data(),
                                                          deviceFF.contribs.angleConstraintTerms.idx2.data(),
                                                          deviceFF.contribs.angleConstraintTerms.idx3.data(),
                                                          deviceFF.contribs.angleConstraintTerms.minAngleDeg.data(),
                                                          deviceFF.contribs.angleConstraintTerms.maxAngleDeg.data(),
                                                          deviceFF.contribs.angleConstraintTerms.forceConstant.data(),
                                                          deviceFF.positions.data(),
                                                          deviceFF.energyBuffer.data(),
                                                          deviceFF.indices.energyBufferStarts.data(),
                                                          deviceFF.indices.atomIdxToBatchIdx.data(),
                                                          deviceFF.indices.angleConstraintTermStarts.data()));
      break;
    case FFTerm::TorsionConstraint:
      if (deviceFF.contribs.torsionConstraintTerms.idx1.size() == 0) {
        return 0.0;
      }
      CHECK_CUDA_RETURN(
        launchTorsionConstraintEnergyKernel(deviceFF.contribs.torsionConstraintTerms.idx1.size(),
                                            deviceFF.contribs.torsionConstraintTerms.idx1.data(),
                                            deviceFF.contribs.torsionConstraintTerms.idx2.data(),
                                            deviceFF.contribs.torsionConstraintTerms.idx3.data(),
                                            deviceFF.contribs.torsionConstraintTerms.idx4.data(),
                                            deviceFF.contribs.torsionConstraintTerms.minDihedralDeg.data(),
                                            deviceFF.contribs.torsionConstraintTerms.maxDihedralDeg.data(),
                                            deviceFF.contribs.torsionConstraintTerms.forceConstant.data(),
                                            deviceFF.positions.data(),
                                            deviceFF.energyBuffer.data(),
                                            deviceFF.indices.energyBufferStarts.data(),
                                            deviceFF.indices.atomIdxToBatchIdx.data(),
                                            deviceFF.indices.torsionConstraintTermStarts.data()));
      break;
  }
  CHECK_CUDA_RETURN(launchReduceEnergiesKernel(deviceFF.indices.energyBufferBlockIdxToBatchIdx.size(),
                                               deviceFF.energyBuffer.data(),
                                               deviceFF.indices.energyBufferBlockIdxToBatchIdx.data(),
                                               deviceFF.energyOuts.data()));
  double energy;
  CHECK_CUDA_RETURN(cudaMemcpy(&energy, deviceFF.energyOuts.data() + 0, sizeof(double), cudaMemcpyDeviceToHost));
  return energy;
}

std::vector<double> getGradientTerm(BatchedMolecularDeviceBuffers& deviceFF, const FFTerm& term) {
  switch (term) {
    case FFTerm::BondStretch:
      if (deviceFF.contribs.bondTerms.idx1.size() == 0) {
        return std::vector<double>(deviceFF.positions.size(), 0.0);
      }
      CHECK_CUDA_RETURN(launchBondStretchGradientKernel(deviceFF.contribs.bondTerms.idx1.size(),
                                                        deviceFF.contribs.bondTerms.idx1.data(),
                                                        deviceFF.contribs.bondTerms.idx2.data(),
                                                        deviceFF.contribs.bondTerms.r0.data(),
                                                        deviceFF.contribs.bondTerms.kb.data(),
                                                        deviceFF.positions.data(),
                                                        deviceFF.grad.data()));
      break;
    case FFTerm::AngleBend:
      if (deviceFF.contribs.angleTerms.idx1.size() == 0) {
        return std::vector<double>(deviceFF.positions.size(), 0.0);
      }
      CHECK_CUDA_RETURN(launchAngleBendGradientKernel(deviceFF.contribs.angleTerms.idx1.size(),
                                                      deviceFF.contribs.angleTerms.idx1.data(),
                                                      deviceFF.contribs.angleTerms.idx2.data(),
                                                      deviceFF.contribs.angleTerms.idx3.data(),
                                                      deviceFF.contribs.angleTerms.theta0.data(),
                                                      deviceFF.contribs.angleTerms.ka.data(),
                                                      deviceFF.contribs.angleTerms.isLinear.data(),
                                                      deviceFF.positions.data(),
                                                      deviceFF.grad.data()));
      break;
    case FFTerm::StretchBend:
      if (deviceFF.contribs.bendTerms.idx1.size() == 0) {
        return std::vector<double>(deviceFF.positions.size(), 0.0);
      }
      CHECK_CUDA_RETURN(launchBendStretchGradientKernel(deviceFF.contribs.bendTerms.idx1.size(),
                                                        deviceFF.contribs.bendTerms.idx1.data(),
                                                        deviceFF.contribs.bendTerms.idx2.data(),
                                                        deviceFF.contribs.bendTerms.idx3.data(),
                                                        deviceFF.contribs.bendTerms.theta0.data(),
                                                        deviceFF.contribs.bendTerms.restLen1.data(),
                                                        deviceFF.contribs.bendTerms.restLen2.data(),
                                                        deviceFF.contribs.bendTerms.forceConst1.data(),
                                                        deviceFF.contribs.bendTerms.forceConst2.data(),
                                                        deviceFF.positions.data(),
                                                        deviceFF.grad.data()));
      break;
    case FFTerm::OopBend:
      if (deviceFF.contribs.oopTerms.idx1.size() == 0) {
        return std::vector<double>(deviceFF.positions.size(), 0.0);
      }
      CHECK_CUDA_RETURN(launchOopBendGradientKernel(deviceFF.contribs.oopTerms.idx1.size(),
                                                    deviceFF.contribs.oopTerms.idx1.data(),
                                                    deviceFF.contribs.oopTerms.idx2.data(),
                                                    deviceFF.contribs.oopTerms.idx3.data(),
                                                    deviceFF.contribs.oopTerms.idx4.data(),
                                                    deviceFF.contribs.oopTerms.koop.data(),
                                                    deviceFF.positions.data(),
                                                    deviceFF.grad.data()));
      break;
    case FFTerm::Torsion:
      if (deviceFF.contribs.torsionTerms.idx1.size() == 0) {
        return std::vector<double>(deviceFF.positions.size(), 0.0);
      }
      CHECK_CUDA_RETURN(launchTorsionGradientKernel(deviceFF.contribs.torsionTerms.idx1.size(),
                                                    deviceFF.contribs.torsionTerms.idx1.data(),
                                                    deviceFF.contribs.torsionTerms.idx2.data(),
                                                    deviceFF.contribs.torsionTerms.idx3.data(),
                                                    deviceFF.contribs.torsionTerms.idx4.data(),
                                                    deviceFF.contribs.torsionTerms.V1.data(),
                                                    deviceFF.contribs.torsionTerms.V2.data(),
                                                    deviceFF.contribs.torsionTerms.V3.data(),
                                                    deviceFF.positions.data(),
                                                    deviceFF.grad.data()));
      break;

    case FFTerm::VdW:
      if (deviceFF.contribs.vdwTerms.idx1.size() == 0) {
        return std::vector<double>(deviceFF.positions.size(), 0.0);
      }
      CHECK_CUDA_RETURN(launchVdwGradientKernel(deviceFF.contribs.vdwTerms.idx1.size(),
                                                deviceFF.contribs.vdwTerms.idx1.data(),
                                                deviceFF.contribs.vdwTerms.idx2.data(),
                                                deviceFF.contribs.vdwTerms.R_ij_star.data(),
                                                deviceFF.contribs.vdwTerms.wellDepth.data(),
                                                deviceFF.positions.data(),
                                                deviceFF.grad.data()));
      break;
    case FFTerm::Elec:
      if (deviceFF.contribs.eleTerms.idx1.size() == 0) {
        return std::vector<double>(deviceFF.positions.size(), 0.0);
      }
      CHECK_CUDA_RETURN(launchEleGradientKernel(deviceFF.contribs.eleTerms.idx1.size(),
                                                deviceFF.contribs.eleTerms.idx1.data(),
                                                deviceFF.contribs.eleTerms.idx2.data(),
                                                deviceFF.contribs.eleTerms.chargeTerm.data(),
                                                deviceFF.contribs.eleTerms.dielModel.data(),
                                                deviceFF.contribs.eleTerms.is1_4.data(),
                                                deviceFF.positions.data(),
                                                deviceFF.grad.data()));
      break;
    case FFTerm::DistanceConstraint:
      if (deviceFF.contribs.distanceConstraintTerms.idx1.size() == 0) {
        return std::vector<double>(deviceFF.positions.size(), 0.0);
      }
      CHECK_CUDA_RETURN(
        launchDistanceConstraintGradientKernel(deviceFF.contribs.distanceConstraintTerms.idx1.size(),
                                               deviceFF.contribs.distanceConstraintTerms.idx1.data(),
                                               deviceFF.contribs.distanceConstraintTerms.idx2.data(),
                                               deviceFF.contribs.distanceConstraintTerms.minLen.data(),
                                               deviceFF.contribs.distanceConstraintTerms.maxLen.data(),
                                               deviceFF.contribs.distanceConstraintTerms.forceConstant.data(),
                                               deviceFF.positions.data(),
                                               deviceFF.grad.data()));
      break;
    case FFTerm::PositionConstraint:
      if (deviceFF.contribs.positionConstraintTerms.idx.size() == 0) {
        return std::vector<double>(deviceFF.positions.size(), 0.0);
      }
      CHECK_CUDA_RETURN(
        launchPositionConstraintGradientKernel(deviceFF.contribs.positionConstraintTerms.idx.size(),
                                               deviceFF.contribs.positionConstraintTerms.idx.data(),
                                               deviceFF.contribs.positionConstraintTerms.refX.data(),
                                               deviceFF.contribs.positionConstraintTerms.refY.data(),
                                               deviceFF.contribs.positionConstraintTerms.refZ.data(),
                                               deviceFF.contribs.positionConstraintTerms.maxDispl.data(),
                                               deviceFF.contribs.positionConstraintTerms.forceConstant.data(),
                                               deviceFF.positions.data(),
                                               deviceFF.grad.data()));
      break;
    case FFTerm::AngleConstraint:
      if (deviceFF.contribs.angleConstraintTerms.idx1.size() == 0) {
        return std::vector<double>(deviceFF.positions.size(), 0.0);
      }
      CHECK_CUDA_RETURN(launchAngleConstraintGradientKernel(deviceFF.contribs.angleConstraintTerms.idx1.size(),
                                                            deviceFF.contribs.angleConstraintTerms.idx1.data(),
                                                            deviceFF.contribs.angleConstraintTerms.idx2.data(),
                                                            deviceFF.contribs.angleConstraintTerms.idx3.data(),
                                                            deviceFF.contribs.angleConstraintTerms.minAngleDeg.data(),
                                                            deviceFF.contribs.angleConstraintTerms.maxAngleDeg.data(),
                                                            deviceFF.contribs.angleConstraintTerms.forceConstant.data(),
                                                            deviceFF.positions.data(),
                                                            deviceFF.grad.data()));
      break;
    case FFTerm::TorsionConstraint:
      if (deviceFF.contribs.torsionConstraintTerms.idx1.size() == 0) {
        return std::vector<double>(deviceFF.positions.size(), 0.0);
      }
      CHECK_CUDA_RETURN(
        launchTorsionConstraintGradientKernel(deviceFF.contribs.torsionConstraintTerms.idx1.size(),
                                              deviceFF.contribs.torsionConstraintTerms.idx1.data(),
                                              deviceFF.contribs.torsionConstraintTerms.idx2.data(),
                                              deviceFF.contribs.torsionConstraintTerms.idx3.data(),
                                              deviceFF.contribs.torsionConstraintTerms.idx4.data(),
                                              deviceFF.contribs.torsionConstraintTerms.minDihedralDeg.data(),
                                              deviceFF.contribs.torsionConstraintTerms.maxDihedralDeg.data(),
                                              deviceFF.contribs.torsionConstraintTerms.forceConstant.data(),
                                              deviceFF.positions.data(),
                                              deviceFF.grad.data()));
      break;
  }
  std::vector<double> grad(deviceFF.positions.size(), 0.0);
  deviceFF.grad.copyToHost(grad);
  cudaDeviceSynchronize();
  return grad;
}

std::unique_ptr<ForceFields::ForceField> referenceSetupCommon(RDKit::ROMol*        mol,
                                                              const FFTerm&        term,
                                                              std::vector<double>& positions) {
  auto mmffMolProperties = std::make_unique<RDKit::MMFF::MMFFMolProperties>(*mol);
  mmffMolProperties->setMMFFVerbosity(0);
  EXPECT_NE(mmffMolProperties, nullptr);
  EXPECT_TRUE(mmffMolProperties->isValid());

  auto referenceForceField = std::make_unique<ForceFields::ForceField>();
  // add the atomic positions:
  nvMolKit::setFFPosFromConf(*mol, referenceForceField.get());
  referenceForceField->initialize();
  nvMolKit::confPosToVect(*mol, positions);

  boost::shared_array<std::uint8_t> neighborMat;
  switch (term) {
    case FFTerm::BondStretch:
      RDKit::MMFF::Tools::addBonds(*mol, mmffMolProperties.get(), referenceForceField.get());
      break;
    case FFTerm::AngleBend:
      RDKit::MMFF::Tools::addAngles(*mol, mmffMolProperties.get(), referenceForceField.get());
      break;
    case FFTerm::StretchBend:
      RDKit::MMFF::Tools::addStretchBend(*mol, mmffMolProperties.get(), referenceForceField.get());
      break;
    case FFTerm::OopBend:
      RDKit::MMFF::Tools::addOop(*mol, mmffMolProperties.get(), referenceForceField.get());
      break;
    case FFTerm::Torsion:
      RDKit::MMFF::Tools::addTorsions(*mol, mmffMolProperties.get(), referenceForceField.get());
      break;
    case FFTerm::VdW:
      neighborMat = RDKit::MMFF::Tools::buildNeighborMatrix(*mol);
      RDKit::MMFF::Tools::addVdW(*mol, -1, mmffMolProperties.get(), referenceForceField.get(), neighborMat);
      break;
    case FFTerm::Elec:
      neighborMat = RDKit::MMFF::Tools::buildNeighborMatrix(*mol);
      RDKit::MMFF::Tools::addEle(*mol, -1, mmffMolProperties.get(), referenceForceField.get(), neighborMat);
      break;
    case FFTerm::DistanceConstraint:
    case FFTerm::PositionConstraint:
    case FFTerm::AngleConstraint:
    case FFTerm::TorsionConstraint:
      break;
  }
  return referenceForceField;
}

double getReferenceEnergyTerm(RDKit::ROMol* mol, const FFTerm& term) {
  std::vector<double> positions;
  auto                FF = referenceSetupCommon(mol, term, positions);
  return FF->calcEnergy(positions.data());
}

std::vector<double> getReferenceGradientTerm(RDKit::ROMol* mol, const FFTerm& term) {
  std::vector<double> positions;
  auto                FF = referenceSetupCommon(mol, term, positions);
  std::vector<double> gradients(3 * mol->getNumAtoms(), 0.0);
  FF->calcGrad(positions.data(), gradients.data());
  return gradients;
}

void addConstraintTermToReferenceForcefield(ForceFields::ForceField&       referenceForceField,
                                            const EnergyForceContribsHost& contribs,
                                            const FFTerm                   term) {
  switch (term) {
    case FFTerm::DistanceConstraint: {
      auto* distanceContribs = new ForceFields::DistanceConstraintContribs(&referenceForceField);
      for (size_t i = 0; i < contribs.distanceConstraintTerms.idx1.size(); ++i) {
        distanceContribs->addContrib(contribs.distanceConstraintTerms.idx1[i],
                                     contribs.distanceConstraintTerms.idx2[i],
                                     contribs.distanceConstraintTerms.minLen[i],
                                     contribs.distanceConstraintTerms.maxLen[i],
                                     contribs.distanceConstraintTerms.forceConstant[i]);
      }
      referenceForceField.contribs().push_back(ForceFields::ContribPtr(distanceContribs));
      break;
    }
    case FFTerm::PositionConstraint: {
      for (size_t i = 0; i < contribs.positionConstraintTerms.idx.size(); ++i) {
        auto* positionContrib =
          new ForceFields::MMFF::PositionConstraintContrib(&referenceForceField,
                                                           contribs.positionConstraintTerms.idx[i],
                                                           contribs.positionConstraintTerms.maxDispl[i],
                                                           contribs.positionConstraintTerms.forceConstant[i]);
        referenceForceField.contribs().push_back(ForceFields::ContribPtr(positionContrib));
      }
      break;
    }
    case FFTerm::AngleConstraint: {
      auto* angleContribs = new ForceFields::AngleConstraintContribs(&referenceForceField);
      for (size_t i = 0; i < contribs.angleConstraintTerms.idx1.size(); ++i) {
        angleContribs->addContrib(contribs.angleConstraintTerms.idx1[i],
                                  contribs.angleConstraintTerms.idx2[i],
                                  contribs.angleConstraintTerms.idx3[i],
                                  contribs.angleConstraintTerms.minAngleDeg[i],
                                  contribs.angleConstraintTerms.maxAngleDeg[i],
                                  contribs.angleConstraintTerms.forceConstant[i]);
      }
      referenceForceField.contribs().push_back(ForceFields::ContribPtr(angleContribs));
      break;
    }
    case FFTerm::TorsionConstraint: {
      for (size_t i = 0; i < contribs.torsionConstraintTerms.idx1.size(); ++i) {
        auto* torsionContrib =
          new ForceFields::MMFF::TorsionConstraintContrib(&referenceForceField,
                                                          contribs.torsionConstraintTerms.idx1[i],
                                                          contribs.torsionConstraintTerms.idx2[i],
                                                          contribs.torsionConstraintTerms.idx3[i],
                                                          contribs.torsionConstraintTerms.idx4[i],
                                                          contribs.torsionConstraintTerms.minDihedralDeg[i],
                                                          contribs.torsionConstraintTerms.maxDihedralDeg[i],
                                                          contribs.torsionConstraintTerms.forceConstant[i]);
        referenceForceField.contribs().push_back(ForceFields::ContribPtr(torsionContrib));
      }
      break;
    }
    default:
      throw std::invalid_argument("Unsupported term for constraint reference forcefield");
  }
}

double getReferenceConstraintEnergyTerm(RDKit::ROMol*                  mol,
                                        const EnergyForceContribsHost& contribs,
                                        const FFTerm                   term,
                                        const std::vector<double>&     positions) {
  auto referenceForceField = std::make_unique<ForceFields::ForceField>();
  nvMolKit::setFFPosFromConf(*mol, referenceForceField.get());
  referenceForceField->initialize();
  addConstraintTermToReferenceForcefield(*referenceForceField, contribs, term);

  std::vector<double> evalPositions = positions;
  return referenceForceField->calcEnergy(evalPositions.data());
}

std::vector<double> getReferenceConstraintGradientTerm(RDKit::ROMol*                  mol,
                                                       const EnergyForceContribsHost& contribs,
                                                       const FFTerm                   term,
                                                       const std::vector<double>&     positions) {
  auto referenceForceField = std::make_unique<ForceFields::ForceField>();
  nvMolKit::setFFPosFromConf(*mol, referenceForceField.get());
  referenceForceField->initialize();
  addConstraintTermToReferenceForcefield(*referenceForceField, contribs, term);

  std::vector<double> evalPositions = positions;
  std::vector<double> gradients(positions.size(), 0.0);
  referenceForceField->calcGrad(evalPositions.data(), gradients.data());
  return gradients;
}

EnergyForceContribsHost filterContribsForTerm(const EnergyForceContribsHost& contribs, const FFTerm& term) {
  EnergyForceContribsHost filtered;
  switch (term) {
    case FFTerm::BondStretch:
      filtered.bondTerms = contribs.bondTerms;
      break;
    case FFTerm::AngleBend:
      filtered.angleTerms = contribs.angleTerms;
      break;
    case FFTerm::StretchBend:
      filtered.bendTerms = contribs.bendTerms;
      break;
    case FFTerm::OopBend:
      filtered.oopTerms = contribs.oopTerms;
      break;
    case FFTerm::Torsion:
      filtered.torsionTerms = contribs.torsionTerms;
      break;
    case FFTerm::VdW:
      filtered.vdwTerms = contribs.vdwTerms;
      break;
    case FFTerm::Elec:
      filtered.eleTerms = contribs.eleTerms;
      break;
    case FFTerm::DistanceConstraint:
      filtered.distanceConstraintTerms = contribs.distanceConstraintTerms;
      break;
    case FFTerm::PositionConstraint:
      filtered.positionConstraintTerms = contribs.positionConstraintTerms;
      break;
    case FFTerm::AngleConstraint:
      filtered.angleConstraintTerms = contribs.angleConstraintTerms;
      break;
    case FFTerm::TorsionConstraint:
      filtered.torsionConstraintTerms = contribs.torsionConstraintTerms;
      break;
  }
  return filtered;
}

double getEnergyViaForcefield(const EnergyForceContribsHost& contribs, const std::vector<double>& positions) {
  BatchedMolecularSystemHost systemHost;
  addMoleculeToBatch(contribs, positions, systemHost);
  return getCombinedEnergyViaForcefield(systemHost);
}

std::vector<double> getGradientViaForcefield(const EnergyForceContribsHost& contribs,
                                             const std::vector<double>&     positions) {
  BatchedMolecularSystemHost systemHost;
  addMoleculeToBatch(contribs, positions, systemHost);
  return getCombinedGradientViaForcefield(systemHost);
}

double getEnergyTermViaForcefield(const BatchedMolecularSystemHost& systemHost, const FFTerm& term) {
  BatchedMolecularSystemHost filteredSystem;
  addMoleculeToBatch(filterContribsForTerm(systemHost.contribs, term), systemHost.positions, filteredSystem);
  nvMolKit::MMFFBatchedForcefield forcefield(filteredSystem);

  nvMolKit::AsyncDeviceVector<double> positionsDevice;
  nvMolKit::AsyncDeviceVector<double> energyOutsDevice;
  positionsDevice.setFromVector(systemHost.positions);
  energyOutsDevice.resize(1);
  energyOutsDevice.zero();
  CHECK_CUDA_RETURN(forcefield.computeEnergy(energyOutsDevice.data(), positionsDevice.data()));
  double energy = 0.0;
  CHECK_CUDA_RETURN(cudaMemcpy(&energy, energyOutsDevice.data(), sizeof(double), cudaMemcpyDeviceToHost));
  return energy;
}

std::vector<double> getGradientTermViaForcefield(const BatchedMolecularSystemHost& systemHost, const FFTerm& term) {
  BatchedMolecularSystemHost filteredSystem;
  addMoleculeToBatch(filterContribsForTerm(systemHost.contribs, term), systemHost.positions, filteredSystem);
  nvMolKit::MMFFBatchedForcefield forcefield(filteredSystem);

  nvMolKit::AsyncDeviceVector<double> positionsDevice;
  nvMolKit::AsyncDeviceVector<double> gradDevice;
  positionsDevice.setFromVector(systemHost.positions);
  gradDevice.resize(systemHost.positions.size());
  gradDevice.zero();
  CHECK_CUDA_RETURN(forcefield.computeGradients(gradDevice.data(), positionsDevice.data()));
  std::vector<double> grad(systemHost.positions.size(), 0.0);
  gradDevice.copyToHost(grad);
  cudaDeviceSynchronize();
  return grad;
}

double getCombinedEnergyViaForcefield(const BatchedMolecularSystemHost& systemHost) {
  nvMolKit::MMFFBatchedForcefield     forcefield(systemHost);
  nvMolKit::AsyncDeviceVector<double> positionsDevice;
  nvMolKit::AsyncDeviceVector<double> energyOutsDevice;
  positionsDevice.setFromVector(systemHost.positions);
  energyOutsDevice.resize(1);
  energyOutsDevice.zero();
  CHECK_CUDA_RETURN(forcefield.computeEnergy(energyOutsDevice.data(), positionsDevice.data()));
  double energy = 0.0;
  CHECK_CUDA_RETURN(cudaMemcpy(&energy, energyOutsDevice.data(), sizeof(double), cudaMemcpyDeviceToHost));
  return energy;
}

std::vector<double> getCombinedGradientViaForcefield(const BatchedMolecularSystemHost& systemHost) {
  nvMolKit::MMFFBatchedForcefield     forcefield(systemHost);
  nvMolKit::AsyncDeviceVector<double> positionsDevice;
  nvMolKit::AsyncDeviceVector<double> gradDevice;
  positionsDevice.setFromVector(systemHost.positions);
  gradDevice.resize(systemHost.positions.size());
  gradDevice.zero();
  CHECK_CUDA_RETURN(forcefield.computeGradients(gradDevice.data(), positionsDevice.data()));
  std::vector<double> grad(systemHost.positions.size(), 0.0);
  gradDevice.copyToHost(grad);
  cudaDeviceSynchronize();
  return grad;
}

TEST(BatchedForcefieldMetadata, TracksSystemsPerMoleculeAndConformer) {
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CC"));
  ASSERT_NE(mol, nullptr);
  RDKit::MolOps::sanitizeMol(*mol);
  mol->addConformer(new RDKit::Conformer(mol->getNumAtoms()));

  auto                ffParams = constructForcefieldContribs(*mol);
  std::vector<double> positions;
  nvMolKit::confPosToVect(*mol, positions);

  BatchedMolecularSystemHost          systemHost;
  nvMolKit::BatchedForcefieldMetadata metadata;
  nvMolKit::MMFF::addMoleculeToBatch(ffParams, positions, systemHost, &metadata, 0, 0);
  nvMolKit::MMFF::addMoleculeToBatch(ffParams, positions, systemHost, &metadata, 0, 1);
  nvMolKit::MMFF::addMoleculeToBatch(ffParams, positions, systemHost, &metadata, 1, 0);

  nvMolKit::MMFFBatchedForcefield forcefield(systemHost, metadata);
  EXPECT_EQ(forcefield.numMolecules(), 3);
  EXPECT_EQ(forcefield.numLogicalMolecules(), 2);
  EXPECT_THAT(forcefield.systemToMoleculeIdx(), ::testing::ElementsAre(0, 0, 1));
  EXPECT_THAT(forcefield.systemToConformerIdx(), ::testing::ElementsAre(0, 1, 0));
  EXPECT_THAT(forcefield.systemsForMolecule(0), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(forcefield.systemsForMolecule(1), ::testing::ElementsAre(2));
}

TEST(BatchedForcefieldCustomization, AppliesForcefieldModifierBeforeFlattening) {
  const std::string mol2FilePath = getTestDataFolderPath() + "/rdkit_smallmol_1.mol2";
  ASSERT_TRUE(std::filesystem::exists(mol2FilePath));
  auto mol = std::unique_ptr<RDKit::RWMol>(RDKit::MolFileToMol(mol2FilePath, false));
  ASSERT_NE(mol, nullptr);
  RDKit::MolOps::sanitizeMol(*mol);
  std::vector<double> positions;
  nvMolKit::confPosToVect(*mol, positions);

  auto filteredContribs = filterContribsForTerm(constructForcefieldContribs(*mol), FFTerm::BondStretch);
  BatchedMolecularSystemHost systemHostBaseline;
  nvMolKit::MMFF::addMoleculeToBatch(filteredContribs, positions, systemHostBaseline);
  const double baselineEnergy = getCombinedEnergyViaForcefield(systemHostBaseline);

  BatchedMolecularSystemHost          systemHostCustomized;
  nvMolKit::BatchedForcefieldMetadata metadata;
  nvMolKit::MMFF::ForcefieldModifier  customization =
    [](const nvMolKit::BatchedSystemInfo&, const std::vector<double>&, EnergyForceContribsHost& contribs) {
      for (auto& kb : contribs.bondTerms.kb) {
        kb *= 2.0;
      }
    };
  nvMolKit::MMFF::addMoleculeToBatch(filteredContribs, positions, systemHostCustomized, &metadata, 0, 0, customization);
  const double customizedEnergy = getCombinedEnergyViaForcefield(systemHostCustomized);

  EXPECT_NEAR(customizedEnergy, 2.0 * baselineEnergy, FUNCTION_E_TOL);
}

class MMffGpuTestFixture : public ::testing::Test {
 public:
  MMffGpuTestFixture() { testDataFolderPath_ = getTestDataFolderPath(); }

  void SetUp() override {
    const std::string mol2FilePath = testDataFolderPath_ + "/rdkit_smallmol_1.mol2";
    ASSERT_TRUE(std::filesystem::exists(mol2FilePath)) << "Could not find " << mol2FilePath;
    mol_ = std::unique_ptr<RDKit::RWMol>(RDKit::MolFileToMol(mol2FilePath, false));
    ASSERT_NE(mol_, nullptr);
    RDKit::MolOps::sanitizeMol(*mol_);

    // add the atomic positions:
    std::vector<double> positions;
    nvMolKit::confPosToVect(*mol_, positions);

    auto ffParams = constructForcefieldContribs(*mol_);
    addMoleculeToBatch(ffParams, positions, systemHost);
    sendContribsAndIndicesToDevice(systemHost, systemDevice);
    systemDevice.positions.setFromVector(systemHost.positions);
    allocateIntermediateBuffers(systemHost, systemDevice);
    systemDevice.grad.resize(systemHost.positions.size());
    systemDevice.grad.zero();
  }

 protected:
  std::string                   testDataFolderPath_;
  std::unique_ptr<RDKit::RWMol> mol_;

  BatchedMolecularSystemHost    systemHost;
  BatchedMolecularDeviceBuffers systemDevice;
};

TEST_F(MMffGpuTestFixture, BondStretchEnergySingleMolecule) {
  double wantEnergy = getReferenceEnergyTerm(mol_.get(), FFTerm::BondStretch);
  double gotEnergy  = getEnergyTermViaForcefield(systemHost, FFTerm::BondStretch);
  ASSERT_NE(wantEnergy, 0.0);
  EXPECT_NEAR(gotEnergy, wantEnergy, FUNCTION_E_TOL);
}

TEST_F(MMffGpuTestFixture, BondStretchGradientSingleMolecule) {
  std::vector<double> wantGradients = getReferenceGradientTerm(mol_.get(), FFTerm::BondStretch);
  std::vector<double> gotGrad       = getGradientTermViaForcefield(systemHost, FFTerm::BondStretch);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(GRAD_TOL), wantGradients));
}

TEST_F(MMffGpuTestFixture, AngleBendEnergySingleMolecule) {
  double wantEnergy = getReferenceEnergyTerm(mol_.get(), FFTerm::AngleBend);
  double gotEnergy  = getEnergyTermViaForcefield(systemHost, FFTerm::AngleBend);
  ASSERT_NE(wantEnergy, 0.0);
  EXPECT_NEAR(gotEnergy, wantEnergy, FUNCTION_E_TOL);
}

TEST_F(MMffGpuTestFixture, AngleBendGradientSingleMolecule) {
  std::vector<double> wantGradients = getReferenceGradientTerm(mol_.get(), FFTerm::AngleBend);
  std::vector<double> gotGrad       = getGradientTermViaForcefield(systemHost, FFTerm::AngleBend);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(GRAD_TOL), wantGradients));
}

TEST_F(MMffGpuTestFixture, BendStretchEnergySingleMolecule) {
  double wantEnergy = getReferenceEnergyTerm(mol_.get(), FFTerm::StretchBend);
  double gotEnergy  = getEnergyTermViaForcefield(systemHost, FFTerm::StretchBend);
  ASSERT_NE(wantEnergy, 0.0);
  EXPECT_NEAR(gotEnergy, wantEnergy, FUNCTION_E_TOL);
}

TEST_F(MMffGpuTestFixture, StretchBendGradientSingleMolecule) {
  std::vector<double> wantGradients = getReferenceGradientTerm(mol_.get(), FFTerm::StretchBend);
  std::vector<double> gotGrad       = getGradientTermViaForcefield(systemHost, FFTerm::StretchBend);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(GRAD_TOL), wantGradients));
}

TEST_F(MMffGpuTestFixture, OutofPlaneEnergySingleMolecule) {
  double wantEnergy = getReferenceEnergyTerm(mol_.get(), FFTerm::OopBend);
  double gotEnergy  = getEnergyTermViaForcefield(systemHost, FFTerm::OopBend);
  ASSERT_NE(wantEnergy, 0.0);
  EXPECT_NEAR(gotEnergy, wantEnergy, FUNCTION_E_TOL);
}

TEST_F(MMffGpuTestFixture, OutOfPlaneGradientSingleMolecule) {
  std::vector<double> wantGradients = getReferenceGradientTerm(mol_.get(), FFTerm::OopBend);
  std::vector<double> gotGrad       = getGradientTermViaForcefield(systemHost, FFTerm::OopBend);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(GRAD_TOL), wantGradients));
}

TEST_F(MMffGpuTestFixture, TorsionEnergySingleMolecule) {
  double wantEnergy = getReferenceEnergyTerm(mol_.get(), FFTerm::Torsion);
  double gotEnergy  = getEnergyTermViaForcefield(systemHost, FFTerm::Torsion);
  ASSERT_NE(wantEnergy, 0.0);
  EXPECT_NEAR(gotEnergy, wantEnergy, FUNCTION_E_TOL);
}

TEST_F(MMffGpuTestFixture, TorsionGradientSingleMolecule) {
  std::vector<double> wantGradients = getReferenceGradientTerm(mol_.get(), FFTerm::Torsion);
  std::vector<double> gotGrad       = getGradientTermViaForcefield(systemHost, FFTerm::Torsion);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(GRAD_TOL), wantGradients));
}

TEST_F(MMffGpuTestFixture, VdwEnergySingleMolecule) {
  double wantEnergy = getReferenceEnergyTerm(mol_.get(), FFTerm::VdW);
  double gotEnergy  = getEnergyTermViaForcefield(systemHost, FFTerm::VdW);
  ASSERT_NE(wantEnergy, 0.0);
  EXPECT_NEAR(gotEnergy, wantEnergy, FUNCTION_E_TOL);
}

TEST_F(MMffGpuTestFixture, VdwGradientSingleMolecule) {
  std::vector<double> wantGradients = getReferenceGradientTerm(mol_.get(), FFTerm::VdW);
  std::vector<double> gotGrad       = getGradientTermViaForcefield(systemHost, FFTerm::VdW);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(GRAD_TOL), wantGradients));
}

TEST_F(MMffGpuTestFixture, EleEnergySingleMolecule) {
  double wantEnergy = getReferenceEnergyTerm(mol_.get(), FFTerm::Elec);
  double gotEnergy  = getEnergyTermViaForcefield(systemHost, FFTerm::Elec);
  ASSERT_NE(wantEnergy, 0.0);
  EXPECT_NEAR(gotEnergy, wantEnergy, FUNCTION_E_TOL);
}

TEST_F(MMffGpuTestFixture, EleGradientSingleMolecule) {
  std::vector<double> wantGradients = getReferenceGradientTerm(mol_.get(), FFTerm::Elec);
  std::vector<double> gotGrad       = getGradientTermViaForcefield(systemHost, FFTerm::Elec);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(GRAD_TOL), wantGradients));
}

TEST_F(MMffGpuTestFixture, DistanceConstraintEnergySingleMolecule) {
  EnergyForceContribsHost                                       contribs;
  const nvMolKit::ForceFieldConstraints::DistanceConstraintSpec spec{0, 2, true, 0.3, 0.6, 15.0};
  nvMolKit::ForceFieldConstraints::appendDistanceConstraint(contribs, systemHost.positions, spec);

  const double wantEnergy =
    getReferenceConstraintEnergyTerm(mol_.get(), contribs, FFTerm::DistanceConstraint, systemHost.positions);
  EXPECT_NEAR(getEnergyViaForcefield(contribs, systemHost.positions), wantEnergy, FUNCTION_E_TOL);
}

TEST_F(MMffGpuTestFixture, DistanceConstraintGradientSingleMolecule) {
  EnergyForceContribsHost                                       contribs;
  const nvMolKit::ForceFieldConstraints::DistanceConstraintSpec spec{0, 2, true, 0.3, 0.6, 15.0};
  nvMolKit::ForceFieldConstraints::appendDistanceConstraint(contribs, systemHost.positions, spec);

  const auto wantGradients =
    getReferenceConstraintGradientTerm(mol_.get(), contribs, FFTerm::DistanceConstraint, systemHost.positions);
  const auto gotGradients = getGradientViaForcefield(contribs, systemHost.positions);
  EXPECT_THAT(gotGradients, ::testing::Pointwise(::testing::FloatNear(1.0e-4), wantGradients));
}

TEST_F(MMffGpuTestFixture, PositionConstraintEnergySingleMolecule) {
  EnergyForceContribsHost                                       contribs;
  const nvMolKit::ForceFieldConstraints::PositionConstraintSpec spec{0, 0.1, 50.0};
  nvMolKit::ForceFieldConstraints::appendPositionConstraint(contribs, systemHost.positions, spec);

  std::vector<double> evalPositions = systemHost.positions;
  evalPositions[0] += 0.25;
  const double wantEnergy =
    getReferenceConstraintEnergyTerm(mol_.get(), contribs, FFTerm::PositionConstraint, evalPositions);
  EXPECT_NEAR(getEnergyViaForcefield(contribs, evalPositions), wantEnergy, FUNCTION_E_TOL);
}

TEST_F(MMffGpuTestFixture, PositionConstraintGradientSingleMolecule) {
  EnergyForceContribsHost                                       contribs;
  const nvMolKit::ForceFieldConstraints::PositionConstraintSpec spec{0, 0.1, 50.0};
  nvMolKit::ForceFieldConstraints::appendPositionConstraint(contribs, systemHost.positions, spec);

  std::vector<double> evalPositions = systemHost.positions;
  evalPositions[0] += 0.25;
  const auto wantGradients =
    getReferenceConstraintGradientTerm(mol_.get(), contribs, FFTerm::PositionConstraint, evalPositions);
  const auto gotGradients = getGradientViaForcefield(contribs, evalPositions);
  EXPECT_THAT(gotGradients, ::testing::Pointwise(::testing::FloatNear(1.0e-4), wantGradients));
}

TEST_F(MMffGpuTestFixture, AngleConstraintEnergySingleMolecule) {
  EnergyForceContribsHost                                    contribs;
  const nvMolKit::ForceFieldConstraints::AngleConstraintSpec spec{0, 1, 2, true, 5.0, 10.0, 20.0};
  nvMolKit::ForceFieldConstraints::appendAngleConstraint(contribs, systemHost.positions, spec);

  const double wantEnergy =
    getReferenceConstraintEnergyTerm(mol_.get(), contribs, FFTerm::AngleConstraint, systemHost.positions);
  EXPECT_NEAR(getEnergyViaForcefield(contribs, systemHost.positions), wantEnergy, FUNCTION_E_TOL);
}

TEST_F(MMffGpuTestFixture, AngleConstraintGradientSingleMolecule) {
  EnergyForceContribsHost                                    contribs;
  const nvMolKit::ForceFieldConstraints::AngleConstraintSpec spec{0, 1, 2, true, 5.0, 10.0, 20.0};
  nvMolKit::ForceFieldConstraints::appendAngleConstraint(contribs, systemHost.positions, spec);

  const auto wantGradients =
    getReferenceConstraintGradientTerm(mol_.get(), contribs, FFTerm::AngleConstraint, systemHost.positions);
  const auto gotGradients = getGradientViaForcefield(contribs, systemHost.positions);
  EXPECT_THAT(gotGradients, ::testing::Pointwise(::testing::FloatNear(1.0e-3), wantGradients));
}

TEST_F(MMffGpuTestFixture, TorsionConstraintEnergySingleMolecule) {
  EnergyForceContribsHost                                      contribs;
  const nvMolKit::ForceFieldConstraints::TorsionConstraintSpec spec{0, 1, 2, 3, true, 15.0, 30.0, 12.0};
  nvMolKit::ForceFieldConstraints::appendTorsionConstraint(contribs, systemHost.positions, spec);

  const double wantEnergy =
    getReferenceConstraintEnergyTerm(mol_.get(), contribs, FFTerm::TorsionConstraint, systemHost.positions);
  EXPECT_NEAR(getEnergyViaForcefield(contribs, systemHost.positions), wantEnergy, FUNCTION_E_TOL);
}

TEST_F(MMffGpuTestFixture, TorsionConstraintGradientSingleMolecule) {
  EnergyForceContribsHost                                      contribs;
  const nvMolKit::ForceFieldConstraints::TorsionConstraintSpec spec{0, 1, 2, 3, true, 15.0, 30.0, 12.0};
  nvMolKit::ForceFieldConstraints::appendTorsionConstraint(contribs, systemHost.positions, spec);

  const auto wantGradients =
    getReferenceConstraintGradientTerm(mol_.get(), contribs, FFTerm::TorsionConstraint, systemHost.positions);
  const auto gotGradients = getGradientViaForcefield(contribs, systemHost.positions);
  EXPECT_THAT(gotGradients, ::testing::Pointwise(::testing::FloatNear(1.0e-3), wantGradients));
}

TEST_F(MMffGpuTestFixture, CombinedEnergies) {
  auto                                     mmffProperties = std::make_unique<RDKit::MMFF::MMFFMolProperties>(*mol_);
  std::unique_ptr<ForceFields::ForceField> ff(RDKit::MMFF::constructForceField(*mol_, mmffProperties.get()));

  double wantEnergy = ff->calcEnergy(systemHost.positions.data());
  double gotEnergy  = getCombinedEnergyViaForcefield(systemHost);
  EXPECT_NEAR(gotEnergy, wantEnergy, FUNCTION_E_TOL);
}

TEST_F(MMffGpuTestFixture, CombinedGradients) {
  auto                                     mmffProperties = std::make_unique<RDKit::MMFF::MMFFMolProperties>(*mol_);
  std::unique_ptr<ForceFields::ForceField> ff(RDKit::MMFF::constructForceField(*mol_, mmffProperties.get()));

  std::vector<double> wantGradients(3 * mol_->getNumAtoms(), 0.0);
  ff->calcGrad(systemHost.positions.data(), wantGradients.data());

  std::vector<double> gotGrad = getCombinedGradientViaForcefield(systemHost);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(1e-4), wantGradients));
}

TEST_F(MMffGpuTestFixture, CombinedEnergiesPerMolKernels) {
  auto                                     mmffProperties = std::make_unique<RDKit::MMFF::MMFFMolProperties>(*mol_);
  std::unique_ptr<ForceFields::ForceField> ff(RDKit::MMFF::constructForceField(*mol_, mmffProperties.get()));

  double wantEnergy = ff->calcEnergy(systemHost.positions.data());
  CHECK_CUDA_RETURN(computeEnergyBlockPerMol(systemDevice));
  double gotEnergy;
  CHECK_CUDA_RETURN(cudaMemcpy(&gotEnergy, systemDevice.energyOuts.data() + 0, sizeof(double), cudaMemcpyDeviceToHost));
  EXPECT_NEAR(gotEnergy, wantEnergy, FUNCTION_E_TOL);
}

TEST_F(MMffGpuTestFixture, CombinedGradientsPerMolKernels) {
  auto                                     mmffProperties = std::make_unique<RDKit::MMFF::MMFFMolProperties>(*mol_);
  std::unique_ptr<ForceFields::ForceField> ff(RDKit::MMFF::constructForceField(*mol_, mmffProperties.get()));

  std::vector<double> wantGradients(3 * mol_->getNumAtoms(), 0.0);
  ff->calcGrad(systemHost.positions.data(), wantGradients.data());

  CHECK_CUDA_RETURN(computeGradBlockPerMol(systemDevice));
  std::vector<double> gotGrad(systemHost.positions.size(), 0.0);
  systemDevice.grad.copyToHost(gotGrad);
  cudaDeviceSynchronize();
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(GRAD_TOL), wantGradients));
}

class MMffGpuEdgeCasesBase : public ::testing::Test {
 public:
  void SetUp() override {
    ASSERT_NE(mol_, nullptr);
    RDKit::MolOps::sanitizeMol(*mol_);

    // Get energy from RDKit
    mmffMolProperties_ = std::make_unique<RDKit::MMFF::MMFFMolProperties>(*mol_);
    mmffMolProperties_->setMMFFVerbosity(0);
    mol_->addConformer(new RDKit::Conformer(mol_->getNumAtoms()));
    ASSERT_NE(mmffMolProperties_, nullptr);
    ASSERT_TRUE(mmffMolProperties_->isValid());

    referenceForceField_ = std::make_unique<ForceFields::ForceField>();
    // add the atomic positions:
    positions.resize(3 * mol_->getNumAtoms());

    auto ffParams = constructForcefieldContribs(*mol_);
    addMoleculeToBatch(ffParams, positions, systemHost);
    systemHost.contribs = std::move(ffParams);
    sendContribsAndIndicesToDevice(systemHost, systemDevice);
    allocateIntermediateBuffers(systemHost, systemDevice);
    systemDevice.grad.resize(systemHost.positions.size());
    systemDevice.grad.zero();
  }

 protected:
  std::unique_ptr<ForceFields::ForceField>        referenceForceField_;
  std::unique_ptr<RDKit::RWMol>                   mol_;
  std::unique_ptr<RDKit::MMFF::MMFFMolProperties> mmffMolProperties_;
  std::vector<double>                             positions;
  EnergyForceContribsHost                         contribs;

  BatchedMolecularSystemHost    systemHost;
  BatchedMolecularDeviceBuffers systemDevice;
};

class MMffGpuEdgeCases2Atoms : public MMffGpuEdgeCasesBase {
 public:
  MMffGpuEdgeCases2Atoms() { mol_ = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CC")); }

  void setPositions(const double x1,
                    const double y1,
                    const double z1,
                    const double x2,
                    const double y2,
                    const double z2) {
    positions[0]         = x1;
    positions[1]         = y1;
    positions[2]         = z1;
    positions[3]         = x2;
    positions[4]         = y2;
    positions[5]         = z2;
    systemHost.positions = positions;
    systemDevice.positions.setFromVector(positions);
    for (unsigned int i = 0; i < mol_->getNumAtoms(); ++i) {
      mol_->getConformer().setAtomPos(i, RDGeom::Point3D(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]));
      referenceForceField_->positions().push_back(&(mol_->getConformer().getAtomPos(i)));
    }
    referenceForceField_->initialize();
    systemDevice.grad.resize(systemHost.positions.size());
    systemDevice.grad.zero();
  }
};

class MMffGpuEdgeCases3Atoms : public MMffGpuEdgeCasesBase {
 public:
  MMffGpuEdgeCases3Atoms() { mol_ = std::unique_ptr<RDKit::RWMol>(RDKit::SmilesToMol("CCC")); }

  void setPositions(const double x1,
                    const double y1,
                    const double z1,
                    const double x2,
                    const double y2,
                    const double z2,
                    const double x3,
                    const double y3,
                    const double z3) {
    positions[0]         = x1;
    positions[1]         = y1;
    positions[2]         = z1;
    positions[3]         = x2;
    positions[4]         = y2;
    positions[5]         = z2;
    positions[6]         = x3;
    positions[7]         = y3;
    positions[8]         = z3;
    systemHost.positions = positions;
    systemDevice.positions.setFromVector(positions);
    for (unsigned int i = 0; i < mol_->getNumAtoms(); ++i) {
      mol_->getConformer().setAtomPos(i, RDGeom::Point3D(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]));
      referenceForceField_->positions().push_back(&(mol_->getConformer().getAtomPos(i)));
    }
    referenceForceField_->initialize();
    systemDevice.grad.resize(systemHost.positions.size());
    systemDevice.grad.zero();
  }
};

TEST_F(MMffGpuEdgeCases2Atoms, ZeroBondLength) {
  setPositions(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  RDKit::MMFF::Tools::addBonds(*mol_, mmffMolProperties_.get(), referenceForceField_.get());
  ASSERT_EQ(referenceForceField_->contribs().size(), 1);

  double wantEnergy = referenceForceField_->calcEnergy(positions.data());
  double gotEnergy  = getEnergyTermViaForcefield(systemHost, FFTerm::BondStretch);
  EXPECT_NEAR(gotEnergy, wantEnergy, EDGE_CASE_TOL);

  std::vector<double> wantGradients(3 * mol_->getNumAtoms(), 0.0);
  referenceForceField_->calcGrad(positions.data(), wantGradients.data());
  std::vector<double> gotGrad = getGradientTermViaForcefield(systemHost, FFTerm::BondStretch);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(GRAD_TOL), wantGradients));
}

TEST_F(MMffGpuEdgeCases2Atoms, ZeroEnergyBond) {
  const double referenceDistance = systemHost.contribs.bondTerms.r0[0];
  setPositions(0.0, 0.0, 0.0, 0.0, 0.0, referenceDistance);

  double wantEnergy = 0.0;
  double gotEnergy  = getEnergyTermViaForcefield(systemHost, FFTerm::BondStretch);
  EXPECT_NEAR(gotEnergy, wantEnergy, FUNCTION_E_TOL);

  std::vector<double> wantGradients(3 * mol_->getNumAtoms(), 0.0);
  std::vector<double> gotGrad = getGradientTermViaForcefield(systemHost, FFTerm::BondStretch);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(GRAD_TOL), wantGradients));
}

TEST_F(MMffGpuEdgeCases3Atoms, ZeroThetaAngle) {
  setPositions(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0);
  RDKit::MMFF::Tools::addAngles(*mol_, mmffMolProperties_.get(), referenceForceField_.get());
  ASSERT_EQ(referenceForceField_->contribs().size(), 1);
  double wantEnergy = referenceForceField_->calcEnergy(positions.data());
  double gotEnergy  = getEnergyTermViaForcefield(systemHost, FFTerm::AngleBend);
  EXPECT_NEAR(gotEnergy, wantEnergy, FUNCTION_E_TOL);

  std::vector<double> wantGradients(3 * mol_->getNumAtoms(), 0.0);
  referenceForceField_->calcGrad(positions.data(), wantGradients.data());
  std::vector<double> gotGrad = getGradientTermViaForcefield(systemHost, FFTerm::AngleBend);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(GRAD_TOL), wantGradients));
}

TEST_F(MMffGpuEdgeCases3Atoms, OneEightyThetaAngle) {
  setPositions(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0);
  RDKit::MMFF::Tools::addAngles(*mol_, mmffMolProperties_.get(), referenceForceField_.get());
  ASSERT_EQ(referenceForceField_->contribs().size(), 1);
  double wantEnergy = referenceForceField_->calcEnergy(positions.data());
  double gotEnergy  = getEnergyTermViaForcefield(systemHost, FFTerm::AngleBend);
  EXPECT_NEAR(gotEnergy, wantEnergy, FUNCTION_E_TOL);

  std::vector<double> wantGradients(3 * mol_->getNumAtoms(), 0.0);
  referenceForceField_->calcGrad(positions.data(), wantGradients.data());
  std::vector<double> gotGrad = getGradientTermViaForcefield(systemHost, FFTerm::AngleBend);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(GRAD_TOL), wantGradients));
}

TEST_F(MMffGpuEdgeCases3Atoms, ZeroThetaAngleStretchBend) {
  setPositions(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0);
  RDKit::MMFF::Tools::addStretchBend(*mol_, mmffMolProperties_.get(), referenceForceField_.get());
  ASSERT_EQ(referenceForceField_->contribs().size(), 1);
  double wantEnergy = referenceForceField_->calcEnergy(positions.data());
  double gotEnergy  = getEnergyTermViaForcefield(systemHost, FFTerm::StretchBend);
  EXPECT_NEAR(gotEnergy, wantEnergy, EDGE_CASE_TOL);

  std::vector<double> wantGradients(3 * mol_->getNumAtoms(), 0.0);
  referenceForceField_->calcGrad(positions.data(), wantGradients.data());
  std::vector<double> gotGrad = getGradientTermViaForcefield(systemHost, FFTerm::StretchBend);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(EDGE_CASE_TOL), wantGradients));
}

TEST_F(MMffGpuEdgeCases3Atoms, OneEightyThetaAngleStretchBend) {
  setPositions(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0);
  RDKit::MMFF::Tools::addStretchBend(*mol_, mmffMolProperties_.get(), referenceForceField_.get());
  ASSERT_EQ(referenceForceField_->contribs().size(), 1);
  double wantEnergy = referenceForceField_->calcEnergy(positions.data());
  double gotEnergy  = getEnergyTermViaForcefield(systemHost, FFTerm::StretchBend);
  EXPECT_NEAR(gotEnergy, wantEnergy, EDGE_CASE_TOL);

  std::vector<double> wantGradients(3 * mol_->getNumAtoms(), 0.0);
  referenceForceField_->calcGrad(positions.data(), wantGradients.data());
  std::vector<double> gotGrad = getGradientTermViaForcefield(systemHost, FFTerm::StretchBend);
  EXPECT_THAT(gotGrad, ::testing::Pointwise(::testing::FloatNear(GRAD_TOL), wantGradients));
}

class MMFFValidationSuiteFixture : public ::testing::Test {
 public:
  MMFFValidationSuiteFixture() { testDataFolderPath_ = getTestDataFolderPath(); }

  void runTestInBatch(const std::string& fileName);
  void runTestInSerial(const std::string& fileName);

  std::string         testDataFolderPath_;
  std::vector<double> positions;
};

struct ValidationFailures {
  std::string         name;
  double              delta = -1.0;
  std::string         exception;
  std::vector<double> deltaComponents;
};

void checkFailures3DArray(const std::vector<ValidationFailures>& gradFailures, const int numMols) {
  if (!gradFailures.empty()) {
    std::cerr << "Gradient Failed on " << gradFailures.size() << " out of " << numMols << " molecules\n";
    std::cerr << std::setw(20) << std::left << "Molecule"
              << "Max Delta grad";
    for (const auto& name : contribNames) {
      std::cerr << std::setw(10) << name << " ";
    }
    std::cerr << "\n";
    for (const auto& failure : gradFailures) {
      if (failure.exception.size()) {
        std::cerr << failure.name << " " << failure.exception << "\n";
      } else {
        std::cerr << std::setw(20) << std::left << failure.name << " " << std::fixed << std::setprecision(4)
                  << std::setw(10) << failure.delta << " ";
        for (const auto& delta : failure.deltaComponents) {
          std::cerr << std::fixed << std::setprecision(4) << std::setw(10) << delta << " ";
        }
        std::cerr << "\n";
      }
    }
    FAIL();
  }
}

void checkFailures(const std::vector<ValidationFailures>  failures,
                   const std::vector<ValidationFailures>& gradFailures,
                   const int                              numMols) {
  if (!failures.empty()) {
    std::cerr << "Energy Failed on " << failures.size() << " out of " << numMols << " molecules\n";
    std::cerr << std::setw(20) << std::left << "Molecule"
              << "DeltaEnergy ";
    for (const auto& name : contribNames) {
      std::cerr << std::setw(10) << name << " ";
    }
    std::cerr << "\n";
    for (const auto& failure : failures) {
      if (failure.exception.size()) {
        std::cerr << failure.name << " " << failure.exception << "\n";
      } else {
        std::cerr << std::setw(20) << std::left << failure.name << " " << std::fixed << std::setprecision(4)
                  << std::setw(10) << failure.delta << " ";
        for (const auto& delta : failure.deltaComponents) {
          std::cerr << std::fixed << std::setprecision(4) << std::setw(10) << delta << " ";
        }
        std::cerr << "\n";
      }
    }
    FAIL();
  }
  checkFailures3DArray(gradFailures, numMols);
}

void getEnergyTermBreakdown(BatchedMolecularDeviceBuffers& systemDevice,
                            RDKit::ROMol*                  mol,
                            ValidationFailures&            failureLog) {
  for (size_t i = 0; i < allTerms.size(); i++) {
    auto term = allTerms[i];
    systemDevice.energyBuffer.zero();
    systemDevice.energyOuts.zero();

    double gotEnergyTerm  = getEnergyTerm(systemDevice, term);
    double wantEnergyTerm = getReferenceEnergyTerm(mol, term);
    failureLog.deltaComponents.push_back(std::abs(gotEnergyTerm - wantEnergyTerm));
  }
}

void getGradTermBreakdown(BatchedMolecularDeviceBuffers& systemDevice,
                          RDKit::ROMol*                  mol,
                          ValidationFailures&            failureLog) {
  for (size_t i = 0; i < allTerms.size(); i++) {
    auto term = allTerms[i];
    systemDevice.grad.zero();

    std::vector<double> gotGradTerm  = getGradientTerm(systemDevice, term);
    std::vector<double> wantGradTerm = getReferenceGradientTerm(mol, term);
    // Find max absolute delta:
    double              maxDelta     = 0.0;
    for (size_t j = 0; j < wantGradTerm.size(); j++) {
      maxDelta = std::max(maxDelta, std::abs(wantGradTerm[j] - gotGradTerm[j]));
    }
    failureLog.deltaComponents.push_back(maxDelta);
  }
}

void MMFFValidationSuiteFixture::runTestInBatch(const std::string& fileName) {
  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  getMols(fileName, mols);
  const int                        numMols = mols.size();
  std::vector<ValidationFailures>  failures;
  std::vector<ValidationFailures>  gradFailures;
  std::vector<std::vector<double>> wantGrads;
  std::vector<double>              wantEnergies;

  BatchedMolecularSystemHost    systemHost;
  BatchedMolecularDeviceBuffers systemDevice;

  for (const auto& mol : mols) {
    auto mmffMolProperties = std::make_unique<RDKit::MMFF::MMFFMolProperties>(*mol);
    mmffMolProperties->setMMFFVerbosity(0);
    ASSERT_NE(mmffMolProperties, nullptr);
    ASSERT_TRUE(mmffMolProperties->isValid());

    // add the atomic positions:
    nvMolKit::confPosToVect(*mol, positions);

    // Get reference energy and forces, store
    std::unique_ptr<ForceFields::ForceField> ff(RDKit::MMFF::constructForceField(*mol, mmffMolProperties.get()));
    wantEnergies.push_back(ff->calcEnergy(positions.data()));
    std::vector<double> wantGrad(3 * mol->getNumAtoms(), 0.0);
    ff->calcGrad(positions.data(), wantGrad.data());
    wantGrads.push_back(wantGrad);

    auto ffParams = constructForcefieldContribs(*mol);
    addMoleculeToBatch(ffParams, positions, systemHost);
  }

  sendContribsAndIndicesToDevice(systemHost, systemDevice);
  systemDevice.positions.setFromVector(systemHost.positions);
  allocateIntermediateBuffers(systemHost, systemDevice);
  systemDevice.grad.resize(systemHost.positions.size());
  systemDevice.grad.zero();

  computeEnergy(systemDevice);
  computeGradients(systemDevice);
  std::vector<double> gotEnergies(systemDevice.energyOuts.size(), 0.0);
  systemDevice.energyOuts.copyToHost(gotEnergies);
  std::vector<double> gotGradFlat(systemHost.positions.size(), 0.0);
  systemDevice.grad.copyToHost(gotGradFlat);
  std::vector<std::vector<double>> gotGradFormatted;
  for (int i = 0; i < numMols; i++) {
    const int atomStart = systemHost.indices.atomStarts[i];
    const int atomEnd   = systemHost.indices.atomStarts[i + 1];

    gotGradFormatted.push_back(
      std::vector<double>(gotGradFlat.begin() + 3 * atomStart, gotGradFlat.begin() + 3 * atomEnd));
  }

  // Check energies first.
  for (int i = 0; i < numMols; i++) {
    const double gotEnergy  = gotEnergies[i];
    const double wantEnergy = wantEnergies[i];

    if (std::abs(gotEnergy - wantEnergy) > 1e-4) {
      auto& failure = failures.emplace_back();
      failure.name  = mols[i]->getProp<std::string>("_Name");
      failure.delta = gotEnergy - wantEnergy;
      getEnergyTermBreakdown(systemDevice, mols[i].get(), failure);
    }
  }

  // Check gradients
  for (int i = 0; i < numMols; i++) {
    const std::vector<double>& gotGrad  = gotGradFormatted[i];
    const std::vector<double>& wantGrad = wantGrads[i];

    bool foundFailure = false;
    for (size_t j = 0; j < wantGrad.size(); j++) {
      if (std::abs(wantGrad[j] - gotGrad[j]) > GRAD_TOL) {
        auto& failure = gradFailures.emplace_back();
        failure.name  = mols[i]->getProp<std::string>("_Name");
        failure.delta = std::abs(wantGrad[j] - gotGrad[j]);
        foundFailure  = true;
        break;
      }
    }
    if (foundFailure) {
      // Calc each grad component for the breakdown, find the max delta
      getGradTermBreakdown(systemDevice, mols[i].get(), gradFailures.back());
    }
  }
  checkFailures(failures, gradFailures, numMols);
}

void MMFFValidationSuiteFixture::runTestInSerial(const std::string& fileName) {
  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  getMols(fileName, mols);
  const int                       numMols = mols.size();
  std::vector<ValidationFailures> failures;
  std::vector<ValidationFailures> gradFailures;

  for (const auto& mol : mols) {
    auto mmffMolProperties = std::make_unique<RDKit::MMFF::MMFFMolProperties>(*mol);
    mmffMolProperties->setMMFFVerbosity(0);
    ASSERT_NE(mmffMolProperties, nullptr);
    ASSERT_TRUE(mmffMolProperties->isValid());

    std::unique_ptr<ForceFields::ForceField> ff(RDKit::MMFF::constructForceField(*mol, mmffMolProperties.get()));

    // add the atomic positions:
    nvMolKit::confPosToVect(*mol, positions);

    auto                          ffParams = constructForcefieldContribs(*mol);
    BatchedMolecularSystemHost    systemHost;
    BatchedMolecularDeviceBuffers systemDevice;
    addMoleculeToBatch(ffParams, positions, systemHost);
    sendContribsAndIndicesToDevice(systemHost, systemDevice);
    systemDevice.positions.setFromVector(systemHost.positions);
    allocateIntermediateBuffers(systemHost, systemDevice);
    systemDevice.grad.resize(systemHost.positions.size());
    systemDevice.grad.zero();

    double      gotEnergy = 0.0;
    std::string exceptionStr;
    try {
      CHECK_CUDA_RETURN(computeEnergy(systemDevice));
      CHECK_CUDA_RETURN(
        cudaMemcpy(&gotEnergy, systemDevice.energyOuts.data() + 0, sizeof(double), cudaMemcpyDeviceToHost));
    } catch (const std::runtime_error& e) {
      exceptionStr = e.what();
    };

    double wantEnergy = ff->calcEnergy(positions.data());
    if (exceptionStr.size() || std::abs(gotEnergy - wantEnergy) > 1e-4) {
      auto& failure = failures.emplace_back();
      failure.name  = mol->getProp<std::string>("_Name");
      if (exceptionStr.size()) {
        failure.exception = exceptionStr;
      }
      failure.delta = gotEnergy - wantEnergy;
      getEnergyTermBreakdown(systemDevice, mol.get(), failure);
    }
    std::vector<double> wantGrad(3 * mol->getNumAtoms(), 0.0);
    ff->calcGrad(positions.data(), wantGrad.data());
    systemDevice.grad.zero();
    CHECK_CUDA_RETURN(computeGradients(systemDevice));
    std::vector<double> gotGrad(systemHost.positions.size(), 0.0);
    systemDevice.grad.copyToHost(gotGrad);
    cudaDeviceSynchronize();

    bool foundFailure = false;
    for (size_t i = 0; i < wantGrad.size(); i++) {
      if (std::abs(wantGrad[i] - gotGrad[i]) > GRAD_TOL) {
        auto& failure = gradFailures.emplace_back();
        failure.name  = mol->getProp<std::string>("_Name");
        failure.delta = std::abs(wantGrad[i] - gotGrad[i]);
        foundFailure  = true;
        break;
      }
    }
    if (foundFailure) {
      // Calc each grad component for the breakdown, find the max delta
      getGradTermBreakdown(systemDevice, mol.get(), gradFailures.back());
    }
  }
  checkFailures(failures, gradFailures, numMols);
}

TEST_F(MMFFValidationSuiteFixture, MMFF94_dative_serial) {
  runTestInSerial(testDataFolderPath_ + "/MMFF94_dative.sdf");
}

TEST_F(MMFFValidationSuiteFixture, MMFF94_hypervalent_serial) {
  runTestInSerial(testDataFolderPath_ + "/MMFF94_hypervalent.sdf");
}

TEST_F(MMFFValidationSuiteFixture, MMFF94_dative_batched) {
  runTestInBatch(testDataFolderPath_ + "/MMFF94_dative.sdf");
}

TEST_F(MMFFValidationSuiteFixture, MMFF94_hypervalent_batched) {
  runTestInBatch(testDataFolderPath_ + "/MMFF94_hypervalent.sdf");
}

void perturbConformer(RDKit::Conformer& conf, const float delta = 0.1, const int seed = 0) {
  std::mt19937                          gen(seed);  // Mersenne Twister engine
  std::uniform_real_distribution<float> dist(-delta, delta);
  for (unsigned int i = 0; i < conf.getNumAtoms(); ++i) {
    RDGeom::Point3D pos = conf.getAtomPos(i);
    pos.x += delta * dist(gen);
    pos.y += delta * dist(gen);
    pos.z += delta * dist(gen);
    conf.setAtomPos(i, pos);
  }
}

// Tweak the x dimension of the molecule by delta, alternating between + and -
void perturbMolecule(RDKit::ROMol& mol, const float delta = 0.1, const int seed = 0) {
  perturbConformer(mol.getConformer(), delta, seed);
}

void printEnergies(const std::vector<std::unique_ptr<RDKit::ROMol>>& mols) {
  for (const auto& mol : mols) {
    auto                                     mmffProperties = std::make_unique<RDKit::MMFF::MMFFMolProperties>(*mol);
    std::unique_ptr<ForceFields::ForceField> ff(RDKit::MMFF::constructForceField(*mol, mmffProperties.get()));
    const double                             gotEnergy = ff->calcEnergy();
    std::cout << "Energy: " << gotEnergy << "\n";
  }
}

void printEnergies(const std::vector<RDKit::ROMol*>& mols) {
  for (const auto& mol : mols) {
    auto                                     mmffProperties = std::make_unique<RDKit::MMFF::MMFFMolProperties>(*mol);
    std::unique_ptr<ForceFields::ForceField> ff(RDKit::MMFF::constructForceField(*mol, mmffProperties.get()));
    const double                             gotEnergy = ff->calcEnergy();
    std::cout << "Energy: " << gotEnergy << "\n";
  }
}

TEST_F(MMFFValidationSuiteFixture, MinimizeBFGSMultipleConfsSameMolecule) {
  constexpr int    numConfs   = 50;
  constexpr double wantEnergy = 26.8743;

  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  getMols(getTestDataFolderPath() + "/MMFF94_dative.sdf", mols, 1);
  auto& mol = *mols[0];

  std::vector<uint32_t> confIds = {0};
  // We start with one conformer, so index from there.
  for (int i = 1; i < numConfs; i++) {
    auto conf = new RDKit::Conformer(mol.getConformer());
    perturbConformer(*conf, 0.5, i + 5);
    confIds.push_back(mols[0]->addConformer(conf, true));
  }
  std::vector<RDKit::ROMol*>               molPtrs     = {&mol};
  const std::vector<double>                gotEnergies = MMFFOptimizeMoleculesConfsBfgs(molPtrs)[0];
  auto                                     molProps    = std::make_unique<RDKit::MMFF::MMFFMolProperties>(mol);
  std::unique_ptr<ForceFields::ForceField> outMolFF(RDKit::MMFF::constructForceField(mol, molProps.get()));
  ASSERT_EQ(confIds.size(), numConfs);
  int i = 0;
  for (auto confIter = mol.beginConformers(); confIter != mol.endConformers(); ++confIter) {
    std::vector<double> pos;
    nvMolKit::confPosToVect(**confIter, pos);
    const double outEnergy = outMolFF->calcEnergy(pos.data());
    ASSERT_NEAR(gotEnergies[i], outEnergy, 1e-4);  // Inconsistency between output positions and reported energy.
    EXPECT_NEAR(wantEnergy, outEnergy, 1e-4) << "Energy mismatch for conformer " << i;
    i++;
  }
}

TEST_F(MMFFValidationSuiteFixture, MinimizeBFGSMultipleConfsMultipleMolecules) {
  constexpr int numMols        = 4;   // Use first 4 molecules from the dataset
  constexpr int numConfsPerMol = 10;  // Add multiple conformers per molecule

  // Expected minimum energies for the first 4 molecules from MMFF94_dative.sdf
  const std::vector<double> wantEnergies = {26.8743, 66.1801, -18.7326, -207.436};

  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  getMols(getTestDataFolderPath() + "/MMFF94_dative.sdf", mols, numMols);

  // Convert to vector of pointers and add multiple conformers to each molecule
  std::vector<RDKit::ROMol*> molPtrs;
  for (auto& mol : mols) {
    // Add additional conformers by perturbing the original
    for (int i = 1; i < numConfsPerMol; i++) {
      auto conf = new RDKit::Conformer(mol->getConformer());
      perturbConformer(*conf, 0.5, i + mol->getNumAtoms());  // Use mol size as seed variation
      mol->addConformer(conf, true);
    }
    molPtrs.push_back(mol.get());
  }

  // Test our new API that optimizes multiple molecules with multiple conformers
  std::vector<std::vector<double>> gotEnergies = MMFFOptimizeMoleculesConfsBfgs(molPtrs);

  // Verify results by comparing with RDKit energies for each optimized conformer
  ASSERT_EQ(gotEnergies.size(), numMols);

  for (size_t molIdx = 0; molIdx < mols.size(); ++molIdx) {
    auto&                                    mol      = *mols[molIdx];
    auto                                     molProps = std::make_unique<RDKit::MMFF::MMFFMolProperties>(mol);
    std::unique_ptr<ForceFields::ForceField> outMolFF(RDKit::MMFF::constructForceField(mol, molProps.get()));

    const auto& energiesForMol = gotEnergies[molIdx];
    ASSERT_EQ(energiesForMol.size(), mol.getNumConformers());

    int confIdx = 0;
    for (auto confIter = mol.beginConformers(); confIter != mol.endConformers(); ++confIter) {
      std::vector<double> pos;
      nvMolKit::confPosToVect(**confIter, pos);
      const double outEnergy = outMolFF->calcEnergy(pos.data());

      // Compare our reported energy with RDKit's calculation on the optimized positions
      ASSERT_NEAR(energiesForMol[confIdx], outEnergy, MINIMIZE_E_TOL)
        << "Energy mismatch for molecule " << molIdx << ", conformer " << confIdx;

      // Verify that each conformer reaches the expected minimum energy
      EXPECT_NEAR(wantEnergies[molIdx], outEnergy, MINIMIZE_E_TOL)
        << "Optimized energy mismatch for molecule " << molIdx << ", conformer " << confIdx
        << " (expected: " << wantEnergies[molIdx] << ", got: " << outEnergy << ")";

      confIdx++;
    }
  }
}

// Size 50, should trigger the large molecule paths in cuda kernels.
TEST_F(MMFFValidationSuiteFixture, MinimizeBFGSLargeMol) {
  constexpr double                           wantEnergy = 33.0842;
  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  getMols(getTestDataFolderPath() + "/50_atom_mol.sdf", mols, 1);
  auto& mol = *mols[0];
  ASSERT_EQ(mol.getNumAtoms(), 50);
  perturbConformer(mol.getConformer(), 0.5, 0);
  ASSERT_EQ(mol.getNumConformers(), 1);

  auto                                     molProps = std::make_unique<RDKit::MMFF::MMFFMolProperties>(mol);
  std::unique_ptr<ForceFields::ForceField> outMolFF(RDKit::MMFF::constructForceField(mol, molProps.get()));
  std::vector<RDKit::ROMol*>               molPtrs     = {&mol};
  const std::vector<double>                gotEnergies = MMFFOptimizeMoleculesConfsBfgs(molPtrs)[0];
  int                                      i           = 0;
  for (auto confIter = mol.beginConformers(); confIter != mol.endConformers(); ++confIter) {
    std::vector<double> pos;
    nvMolKit::confPosToVect(**confIter, pos);
    const double outEnergy = outMolFF->calcEnergy(pos.data());
    ASSERT_NEAR(gotEnergies[i],
                outEnergy,
                MINIMIZE_E_TOL);  // Inconsistency between output positions and reported energy.
    EXPECT_NEAR(wantEnergy, outEnergy, MINIMIZE_E_TOL) << "Energy mismatch for conformer " << i;
    i++;
  }
}

class MMffGpuWrapperTestFixture : public ::testing::Test {
 public:
  MMffGpuWrapperTestFixture() { testDataFolderPath_ = getTestDataFolderPath(); }

  void SetUp() override {
    const std::string mol2FilePath = testDataFolderPath_ + "/rdkit_smallmol_1.mol2";
    ASSERT_TRUE(std::filesystem::exists(mol2FilePath)) << "Could not find " << mol2FilePath;
    molReference_ = std::unique_ptr<RDKit::RWMol>(RDKit::MolFileToMol(mol2FilePath, false));
    ASSERT_NE(molReference_, nullptr);
    RDKit::MolOps::sanitizeMol(*molReference_);
    molTest_ = std::make_unique<RDKit::RWMol>(*molReference_);
  }

 protected:
  std::string                   testDataFolderPath_;
  std::unique_ptr<RDKit::RWMol> molReference_;
  std::unique_ptr<RDKit::RWMol> molTest_;
};

TEST_F(MMffGpuWrapperTestFixture, MMffConstructorEnergy) {
  auto ffReference = std::unique_ptr<ForceFields::ForceField>(RDKit::MMFF::constructForceField(*molReference_));
  auto ffTest      = std::unique_ptr<ForceFields::ForceField>(nvMolKit::MMFF::constructForceField(*molTest_));

  double initEnRef  = ffReference->calcEnergy();
  double initEnTest = ffTest->calcEnergy();

  EXPECT_NEAR(initEnTest, initEnRef, FUNCTION_E_TOL);

  RDKit::ForceFieldsHelper::OptimizeMolecule(*ffReference, 1000);
  RDKit::ForceFieldsHelper::OptimizeMolecule(*ffTest, 1000);

  double finalEnRef  = ffReference->calcEnergy();
  double finalEnTest = ffTest->calcEnergy();

  EXPECT_NEAR(finalEnTest, finalEnRef, FUNCTION_E_TOL);
}

TEST_F(MMffGpuWrapperTestFixture, MMffConstructorEnergyWithConstraints) {
  auto ffReference = std::unique_ptr<ForceFields::ForceField>(RDKit::MMFF::constructForceField(*molReference_));
  auto ffTest      = std::unique_ptr<ForceFields::ForceField>(nvMolKit::MMFF::constructForceField(*molTest_));

  const auto addConstraints = [](ForceFields::ForceField& forcefield) {
    auto* distanceContribs = new ForceFields::DistanceConstraintContribs(&forcefield);
    distanceContribs->addContrib(0, 2, true, 0.3, 0.6, 15.0);
    forcefield.contribs().push_back(ForceFields::ContribPtr(distanceContribs));

    auto* positionContrib = new ForceFields::MMFF::PositionConstraintContrib(&forcefield, 0, 0.1, 50.0);
    forcefield.contribs().push_back(ForceFields::ContribPtr(positionContrib));

    auto* angleContribs = new ForceFields::AngleConstraintContribs(&forcefield);
    angleContribs->addContrib(0, 1, 2, true, 5.0, 10.0, 20.0);
    forcefield.contribs().push_back(ForceFields::ContribPtr(angleContribs));

    auto* torsionContrib =
      new ForceFields::MMFF::TorsionConstraintContrib(&forcefield, 0, 1, 2, 3, true, 15.0, 30.0, 12.0);
    forcefield.contribs().push_back(ForceFields::ContribPtr(torsionContrib));
  };
  addConstraints(*ffReference);
  addConstraints(*ffTest);

  double initEnRef  = ffReference->calcEnergy();
  double initEnTest = ffTest->calcEnergy();
  EXPECT_NEAR(initEnTest, initEnRef, FUNCTION_E_TOL);

  RDKit::ForceFieldsHelper::OptimizeMolecule(*ffReference, 1000);
  RDKit::ForceFieldsHelper::OptimizeMolecule(*ffTest, 1000);

  double finalEnRef  = ffReference->calcEnergy();
  double finalEnTest = ffTest->calcEnergy();
  EXPECT_NEAR(finalEnTest, finalEnRef, FUNCTION_E_TOL);
}

TEST_F(MMffGpuWrapperTestFixture, MMffOptimizerEnergy) {
  RDKit::MMFF::MMFFOptimizeMolecule(*molReference_, 1000);
  nvMolKit::MMFF::MMFFOptimizeMolecule(*molTest_, 1000);

  auto ffReference = std::unique_ptr<ForceFields::ForceField>(RDKit::MMFF::constructForceField(*molReference_));
  auto ffTest      = std::unique_ptr<ForceFields::ForceField>(nvMolKit::MMFF::constructForceField(*molTest_));

  double finalEnRef  = ffReference->calcEnergy();
  double finalEnTest = ffTest->calcEnergy();

  EXPECT_NEAR(finalEnTest, finalEnRef, FUNCTION_E_TOL);
}

TEST_F(MMffGpuWrapperTestFixture, MMffOptimizConfEnergy) {
  std::vector<std::pair<int, double>> resRef(molReference_->getNumConformers(), {-1, -1});
  RDKit::MMFF::MMFFOptimizeMoleculeConfs(*molReference_, resRef);

  std::vector<std::pair<int, double>> resTest(molTest_->getNumConformers(), {-1, -1});
  nvMolKit::MMFF::MMFFOptimizeMoleculeConfs(*molTest_, resTest);

  auto ffReference = std::unique_ptr<ForceFields::ForceField>(RDKit::MMFF::constructForceField(*molReference_));
  auto ffTest      = std::unique_ptr<ForceFields::ForceField>(nvMolKit::MMFF::constructForceField(*molTest_));

  double finalEnRef  = ffReference->calcEnergy();
  double finalEnTest = ffTest->calcEnergy();

  EXPECT_NEAR(finalEnTest, finalEnRef, FUNCTION_E_TOL);
}

TEST_F(MMffGpuWrapperTestFixture, MMffConstructorGrad) {
  auto ffReference = std::unique_ptr<ForceFields::ForceField>(RDKit::MMFF::constructForceField(*molReference_));
  auto ffTest      = std::unique_ptr<ForceFields::ForceField>(nvMolKit::MMFF::constructForceField(*molTest_));

  std::vector<double> gradRef(3 * molReference_->getNumAtoms(), 0.0);
  std::vector<double> gradTest(3 * molTest_->getNumAtoms(), 0.0);

  ffReference->calcGrad(gradRef.data());
  ffTest->calcGrad(gradTest.data());

  EXPECT_THAT(gradTest, ::testing::Pointwise(::testing::FloatNear(GRAD_TOL), gradRef));
}

class MMffGpuWrapperNonDefaultTestFixture : public ::testing::Test {
 public:
  MMffGpuWrapperNonDefaultTestFixture() { testDataFolderPath_ = getTestDataFolderPath(); }

  void SetUp() override {
    const std::string mol2FilePath = testDataFolderPath_ + "/rdkit_smallmol_1.mol2";
    ASSERT_TRUE(std::filesystem::exists(mol2FilePath)) << "Could not find " << mol2FilePath;
    molReference_ = std::unique_ptr<RDKit::RWMol>(RDKit::MolFileToMol(mol2FilePath, false));
    ASSERT_NE(molReference_, nullptr);
    RDKit::MolOps::sanitizeMol(*molReference_);
    auto options = RDKit::DGeomHelpers::ETKDGv3;
    RDKit::DGeomHelpers::EmbedMultipleConfs(*molReference_, 2, options);
    molTest_         = std::make_unique<RDKit::RWMol>(*molReference_);
    nonBondedThresh_ = 50.0;
    confId_          = 1;
  }

 protected:
  std::string                   testDataFolderPath_;
  std::unique_ptr<RDKit::RWMol> molReference_;
  std::unique_ptr<RDKit::RWMol> molTest_;
  double                        nonBondedThresh_;
  int                           confId_;
};

TEST_F(MMffGpuWrapperNonDefaultTestFixture, MMffConstructorEnergy) {
  auto ffReference = std::unique_ptr<ForceFields::ForceField>(
    RDKit::MMFF::constructForceField(*molReference_, nonBondedThresh_, confId_));
  auto ffTest =
    std::unique_ptr<ForceFields::ForceField>(nvMolKit::MMFF::constructForceField(*molTest_, nonBondedThresh_, confId_));
  double initEnRef  = ffReference->calcEnergy();
  double initEnTest = ffTest->calcEnergy();

  EXPECT_NEAR(initEnTest, initEnRef, FUNCTION_E_TOL);

  RDKit::ForceFieldsHelper::OptimizeMolecule(*ffReference, 1000);
  RDKit::ForceFieldsHelper::OptimizeMolecule(*ffTest, 1000);

  double finalEnRef  = ffReference->calcEnergy();
  double finalEnTest = ffTest->calcEnergy();

  EXPECT_NEAR(finalEnTest, finalEnRef, FUNCTION_E_TOL);
}

TEST_F(MMffGpuWrapperNonDefaultTestFixture, MMffOptimizerEnergy) {
  RDKit::MMFF::MMFFOptimizeMolecule(*molReference_, 1000, "MMFF94", nonBondedThresh_, confId_);
  nvMolKit::MMFF::MMFFOptimizeMolecule(*molTest_, 1000, nonBondedThresh_, confId_);

  auto ffReference = std::unique_ptr<ForceFields::ForceField>(
    RDKit::MMFF::constructForceField(*molReference_, nonBondedThresh_, confId_));
  auto ffTest =
    std::unique_ptr<ForceFields::ForceField>(nvMolKit::MMFF::constructForceField(*molTest_, nonBondedThresh_, confId_));

  double finalEnRef  = ffReference->calcEnergy();
  double finalEnTest = ffTest->calcEnergy();

  EXPECT_NEAR(finalEnTest, finalEnRef, FUNCTION_E_TOL);
}

TEST_F(MMffGpuWrapperNonDefaultTestFixture, MMffOptimizConfEnergy) {
  std::vector<std::pair<int, double>> resRef(molReference_->getNumConformers(), {-1, -1});
  RDKit::MMFF::MMFFOptimizeMoleculeConfs(*molReference_, resRef, 1, 1000, "MMFF94", nonBondedThresh_);

  std::vector<std::pair<int, double>> resTest(molTest_->getNumConformers(), {-1, -1});
  nvMolKit::MMFF::MMFFOptimizeMoleculeConfs(*molTest_, resTest, 1, 1000, nonBondedThresh_);

  auto ffReference = std::unique_ptr<ForceFields::ForceField>(
    RDKit::MMFF::constructForceField(*molReference_, nonBondedThresh_, confId_));
  auto ffTest =
    std::unique_ptr<ForceFields::ForceField>(nvMolKit::MMFF::constructForceField(*molTest_, nonBondedThresh_, confId_));

  double finalEnRef  = ffReference->calcEnergy();
  double finalEnTest = ffTest->calcEnergy();

  EXPECT_NEAR(finalEnTest, finalEnRef, FUNCTION_E_TOL);
}

TEST_F(MMffGpuWrapperNonDefaultTestFixture, MMffConstructorGrad) {
  auto ffReference = std::unique_ptr<ForceFields::ForceField>(
    RDKit::MMFF::constructForceField(*molReference_, nonBondedThresh_, confId_));
  auto ffTest =
    std::unique_ptr<ForceFields::ForceField>(nvMolKit::MMFF::constructForceField(*molTest_, nonBondedThresh_, confId_));

  std::vector<double> gradRef(3 * molReference_->getNumAtoms(), 0.0);
  std::vector<double> gradTest(3 * molTest_->getNumAtoms(), 0.0);

  ffReference->calcGrad(gradRef.data());
  ffTest->calcGrad(gradTest.data());

  EXPECT_THAT(gradTest, ::testing::Pointwise(::testing::FloatNear(GRAD_TOL), gradRef));
}

TEST(MMFFMultiGPU, SpecificGpuIds) {
  // Use GPU 0 explicitly
  nvMolKit::BatchHardwareOptions options;
  options.preprocessingThreads = 2;
  options.batchSize            = 2;
  options.batchesPerGpu        = 1;
  options.gpuIds.push_back(0);

  // Use first 4 molecules from dataset, add multiple conformers per molecule
  constexpr int numMols        = 4;
  constexpr int numConfsPerMol = 5;

  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  getMols(getTestDataFolderPath() + "/MMFF94_dative.sdf", mols, numMols);

  std::vector<RDKit::ROMol*> molPtrs;
  for (auto& mol : mols) {
    for (int i = 1; i < numConfsPerMol; i++) {
      auto conf = new RDKit::Conformer(mol->getConformer());
      perturbConformer(*conf, 0.5f, i + mol->getNumAtoms());
      mol->addConformer(conf, true);
    }
    molPtrs.push_back(mol.get());
  }

  // Create hard copies of the perturbed molecules and minimize via RDKit for reference
  std::vector<std::unique_ptr<RDKit::RWMol>> molCopies;
  std::vector<RDKit::ROMol*>                 molCopyPtrs;
  for (const auto& mol : mols) {
    molCopies.push_back(std::make_unique<RDKit::RWMol>(*mol));
    molCopyPtrs.push_back(molCopies.back().get());
    std::vector<std::pair<int, double>> res(molCopies.back()->getNumConformers(), {-1, -1});
    RDKit::MMFF::MMFFOptimizeMoleculeConfs(*molCopies.back(), res);
  }

  // Run optimizer on specific GPU
  std::vector<std::vector<double>> gotEnergies =
    nvMolKit::MMFF::MMFFOptimizeMoleculesConfsBfgs(molPtrs, 200, nvMolKit::MMFFProperties{}, options);

  // Verify energies against RDKit-minimized reference energies
  ASSERT_EQ(gotEnergies.size(), mols.size());
  for (size_t molIdx = 0; molIdx < mols.size(); ++molIdx) {
    auto&                                    molRef   = *molCopies[molIdx];
    auto                                     molProps = std::make_unique<RDKit::MMFF::MMFFMolProperties>(molRef);
    std::unique_ptr<ForceFields::ForceField> refFF(RDKit::MMFF::constructForceField(molRef, molProps.get()));

    const auto& energiesForMol = gotEnergies[molIdx];
    ASSERT_EQ(energiesForMol.size(), molRef.getNumConformers());

    int confIdx = 0;
    for (auto confIter = molRef.beginConformers(); confIter != molRef.endConformers(); ++confIter) {
      std::vector<double> posRef;
      nvMolKit::confPosToVect(**confIter, posRef);
      const double refEnergy = refFF->calcEnergy(posRef.data());
      ASSERT_NEAR(energiesForMol[confIdx], refEnergy, MINIMIZE_E_TOL)
        << "Energy mismatch vs RDKit reference for molecule " << molIdx << ", conformer " << confIdx;
      confIdx++;
    }
  }
}

TEST(MMFFMultiGPU, NonZeroGPUID) {
  // Requires multiple GPUs
  const int numDevices = nvMolKit::countCudaDevices();
  if (numDevices < 2) {
    GTEST_SKIP() << "Test requires multiple GPUs, only " << numDevices << " available";
  }

  nvMolKit::BatchHardwareOptions options;
  options.preprocessingThreads = 2;
  options.batchSize            = 2;
  options.batchesPerGpu        = 1;
  options.gpuIds.push_back(1);  // Use GPU 1

  // Use first 2 molecules, add multiple conformers per molecule
  constexpr int numMols        = 2;
  constexpr int numConfsPerMol = 4;

  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  getMols(getTestDataFolderPath() + "/MMFF94_dative.sdf", mols, numMols);

  std::vector<RDKit::ROMol*> molPtrs;
  for (auto& mol : mols) {
    for (int i = 1; i < numConfsPerMol; i++) {
      auto conf = new RDKit::Conformer(mol->getConformer());
      perturbConformer(*conf, 0.5f, i + 23);
      mol->addConformer(conf, true);
    }
    molPtrs.push_back(mol.get());
  }

  // Create copies and minimize via RDKit for reference
  std::vector<std::unique_ptr<RDKit::RWMol>> molCopies;
  for (const auto& mol : mols) {
    molCopies.push_back(std::make_unique<RDKit::RWMol>(*mol));
    std::vector<std::pair<int, double>> res(molCopies.back()->getNumConformers(), {-1, -1});
    RDKit::MMFF::MMFFOptimizeMoleculeConfs(*molCopies.back(), res);
  }

  std::vector<std::vector<double>> gotEnergies =
    nvMolKit::MMFF::MMFFOptimizeMoleculesConfsBfgs(molPtrs, 200, nvMolKit::MMFFProperties{}, options);

  ASSERT_EQ(gotEnergies.size(), mols.size());
  for (size_t molIdx = 0; molIdx < mols.size(); ++molIdx) {
    auto&                                    molRef   = *molCopies[molIdx];
    auto                                     molProps = std::make_unique<RDKit::MMFF::MMFFMolProperties>(molRef);
    std::unique_ptr<ForceFields::ForceField> refFF(RDKit::MMFF::constructForceField(molRef, molProps.get()));

    const auto& energiesForMol = gotEnergies[molIdx];
    ASSERT_EQ(energiesForMol.size(), molRef.getNumConformers());

    int confIdx = 0;
    for (auto confIter = molRef.beginConformers(); confIter != molRef.endConformers(); ++confIter) {
      std::vector<double> posRef;
      nvMolKit::confPosToVect(**confIter, posRef);
      const double refEnergy = refFF->calcEnergy(posRef.data());
      ASSERT_NEAR(energiesForMol[confIdx], refEnergy, 1e-4)
        << "Energy mismatch vs RDKit reference for molecule " << molIdx << ", conformer " << confIdx;
      confIdx++;
    }
  }
}

TEST(MMFFMultiGPU, MultiGPUSpecificIds) {
  // Requires multiple GPUs
  const int numDevices = nvMolKit::countCudaDevices();
  if (numDevices < 2) {
    GTEST_SKIP() << "Test requires multiple GPUs, only " << numDevices << " available";
  }

  nvMolKit::BatchHardwareOptions options;
  options.preprocessingThreads = 4;
  options.batchSize            = 2;  // Force multiple small batches to exercise both devices
  options.batchesPerGpu        = 2;
  options.gpuIds.push_back(0);
  options.gpuIds.push_back(1);

  // Use first 3 molecules, add multiple conformers per molecule
  constexpr int numMols        = 3;
  constexpr int numConfsPerMol = 6;

  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  getMols(getTestDataFolderPath() + "/MMFF94_dative.sdf", mols, numMols);

  std::vector<RDKit::ROMol*> molPtrs;
  for (auto& mol : mols) {
    for (int i = 1; i < numConfsPerMol; i++) {
      auto conf = new RDKit::Conformer(mol->getConformer());
      perturbConformer(*conf, 0.5f, i + 101);
      mol->addConformer(conf, true);
    }
    molPtrs.push_back(mol.get());
  }

  // Create RDKit-minimized references
  std::vector<std::unique_ptr<RDKit::RWMol>> molCopies;
  for (const auto& mol : mols) {
    molCopies.push_back(std::make_unique<RDKit::RWMol>(*mol));
    std::vector<std::pair<int, double>> res(molCopies.back()->getNumConformers(), {-1, -1});
    RDKit::MMFF::MMFFOptimizeMoleculeConfs(*molCopies.back(), res);
  }

  std::vector<std::vector<double>> gotEnergies =
    nvMolKit::MMFF::MMFFOptimizeMoleculesConfsBfgs(molPtrs, 200, nvMolKit::MMFFProperties{}, options);

  ASSERT_EQ(gotEnergies.size(), mols.size());
  for (size_t molIdx = 0; molIdx < mols.size(); ++molIdx) {
    auto&                                    molRef   = *molCopies[molIdx];
    auto                                     molProps = std::make_unique<RDKit::MMFF::MMFFMolProperties>(molRef);
    std::unique_ptr<ForceFields::ForceField> refFF(RDKit::MMFF::constructForceField(molRef, molProps.get()));

    const auto& energiesForMol = gotEnergies[molIdx];
    ASSERT_EQ(energiesForMol.size(), molRef.getNumConformers());

    int confIdx = 0;
    for (auto confIter = molRef.beginConformers(); confIter != molRef.endConformers(); ++confIter) {
      std::vector<double> posRef;
      nvMolKit::confPosToVect(**confIter, posRef);
      const double refEnergy = refFF->calcEnergy(posRef.data());
      ASSERT_NEAR(energiesForMol[confIdx], refEnergy, 1e-4)
        << "Energy mismatch vs RDKit reference for molecule " << molIdx << ", conformer " << confIdx;
      confIdx++;
    }
  }
}

std::unique_ptr<RDKit::RWMol> createHydroCarbon(const int numCarbons, const double bondLength) {
  if (numCarbons < 1) {
    throw std::invalid_argument("numCarbons must be at least 1");
  }

  auto mol  = std::make_unique<RDKit::RWMol>();
  auto conf = new RDKit::Conformer();
  conf->set3D(true);
  mol->addConformer(conf);

  auto addAtomWithPosition = [&](const int atomicNumber, const RDGeom::Point3D& position) {
    const int         idx     = mol->addAtom(new RDKit::Atom(atomicNumber), true, true);
    RDKit::Conformer& confRef = mol->getConformer();
    if (confRef.getNumAtoms() <= static_cast<unsigned int>(idx)) {
      confRef.resize(idx + 1);
    }
    confRef.setAtomPos(idx, position);

    return idx;
  };

  std::vector<int> carbonIndices;
  carbonIndices.reserve(static_cast<size_t>(numCarbons));
  std::random_device                     dev;
  std::mt19937                           rng(dev());
  std::uniform_real_distribution<double> dist(-0.1, 0.1);
  for (int i = 0; i < numCarbons; ++i) {
    const RDGeom::Point3D carbonPos(0.0 + dist(rng), 0.0 + dist(rng), i * bondLength + dist(rng));
    const int             carbonIdx = addAtomWithPosition(6, carbonPos);

    if (!carbonIndices.empty()) {
      mol->addBond(carbonIndices.back(), carbonIdx, RDKit::Bond::BondType::DOUBLE);
    }
    carbonIndices.push_back(carbonIdx);
  }
  RDKit::MolOps::sanitizeMol(*mol);
  RDKit::MolOps::addHs((*mol), /*explicitOnly=*/false, /*addCoords=*/true);
  return mol;
}

TEST(MMFFAllowsLargeMol, LargeMoleculeInterleavedOptimizes) {
  constexpr double bondLength = 1.0;

  const auto small1 = createHydroCarbon(5, bondLength);
  const auto small2 = createHydroCarbon(3, bondLength);
  const auto big    = createHydroCarbon(300, bondLength);
  ASSERT_NE(small2, nullptr);
  ASSERT_NE(big, nullptr);

  std::vector<std::unique_ptr<RDKit::RWMol>> rdkitRefs;
  rdkitRefs.push_back(std::make_unique<RDKit::RWMol>(*small1));
  rdkitRefs.push_back(std::make_unique<RDKit::RWMol>(*big));
  rdkitRefs.push_back(std::make_unique<RDKit::RWMol>(*small2));
  std::vector<double> startEnergies;
  for (const auto& molCopy : rdkitRefs) {
    auto   molProps = std::make_unique<RDKit::MMFF::MMFFMolProperties>(*molCopy);
    auto   ff = std::unique_ptr<ForceFields::ForceField>(RDKit::MMFF::constructForceField(*molCopy, molProps.get()));
    double energy = ff->calcEnergy();
    startEnergies.push_back(energy);
  }

  std::vector<RDKit::ROMol*>     molPtrs = {small1.get(), big.get(), small2.get()};
  nvMolKit::BatchHardwareOptions options;
  const auto                     energies =
    nvMolKit::MMFF::MMFFOptimizeMoleculesConfsBfgs(molPtrs, 10, nvMolKit::MMFFProperties{}, options);

  for (size_t molIdx = 0; molIdx < rdkitRefs.size(); ++molIdx) {
    auto& molRef   = *rdkitRefs[molIdx];
    auto  molProps = std::make_unique<RDKit::MMFF::MMFFMolProperties>(molRef);

    const auto& perMol  = energies[molIdx];
    int         confIdx = 0;
    for (auto confIter = molRef.beginConformers(); confIter != molRef.endConformers(); ++confIter) {
      std::vector<double> posRef;
      nvMolKit::confPosToVect(**confIter, posRef);
      const double refEnergy = startEnergies[molIdx];
      ASSERT_LT(perMol[confIdx], refEnergy)
        << "Energy not decreased for molecule " << molIdx << ", conformer " << confIdx;
      confIdx++;
    }
  }
}
