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

#include "src/forcefields/forcefield_constraints.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nvMolKit::ForceFieldConstraints {

namespace {
constexpr double kRadiansToDegrees = 180.0 / M_PI;

void crossProduct(const double ax,
                  const double ay,
                  const double az,
                  const double bx,
                  const double by,
                  const double bz,
                  double&      outX,
                  double&      outY,
                  double&      outZ) {
  outX = ay * bz - az * by;
  outY = az * bx - ax * bz;
  outZ = ax * by - ay * bx;
}

double dotProduct(const double ax,
                  const double ay,
                  const double az,
                  const double bx,
                  const double by,
                  const double bz) {
  return ax * bx + ay * by + az * bz;
}
}  // namespace

void validateAtomIndex(const int idx, const int numAtoms, const std::string& what) {
  if (idx < 0 || idx >= numAtoms) {
    throw std::out_of_range(what + " index " + std::to_string(idx) + " is out of range for molecule with " +
                            std::to_string(numAtoms) + " atoms");
  }
}

double distanceFromPositions(const std::vector<double>& positions, const int idx1, const int idx2) {
  const double dx = positions[3 * idx1 + 0] - positions[3 * idx2 + 0];
  const double dy = positions[3 * idx1 + 1] - positions[3 * idx2 + 1];
  const double dz = positions[3 * idx1 + 2] - positions[3 * idx2 + 2];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double computeAngleDeg(const std::vector<double>& positions, const int idx1, const int idx2, const int idx3) {
  const double r1x       = positions[3 * idx1 + 0] - positions[3 * idx2 + 0];
  const double r1y       = positions[3 * idx1 + 1] - positions[3 * idx2 + 1];
  const double r1z       = positions[3 * idx1 + 2] - positions[3 * idx2 + 2];
  const double r2x       = positions[3 * idx3 + 0] - positions[3 * idx2 + 0];
  const double r2y       = positions[3 * idx3 + 1] - positions[3 * idx2 + 1];
  const double r2z       = positions[3 * idx3 + 2] - positions[3 * idx2 + 2];
  const double lengthSq1 = std::max(1.0e-5, r1x * r1x + r1y * r1y + r1z * r1z);
  const double lengthSq2 = std::max(1.0e-5, r2x * r2x + r2y * r2y + r2z * r2z);
  const double cosTheta =
    std::max(-1.0, std::min(1.0, (r1x * r2x + r1y * r2y + r1z * r2z) / std::sqrt(lengthSq1 * lengthSq2)));
  return kRadiansToDegrees * std::acos(cosTheta);
}

double normalizeAngleDeg(double angleDeg) {
  angleDeg = std::fmod(angleDeg, 360.0);
  if (angleDeg < -180.0) {
    angleDeg += 360.0;
  } else if (angleDeg > 180.0) {
    angleDeg -= 360.0;
  }
  return angleDeg;
}

double computeDihedralDeg(const std::vector<double>& positions,
                          const int                  idx1,
                          const int                  idx2,
                          const int                  idx3,
                          const int                  idx4) {
  const double r0x = positions[3 * idx1 + 0] - positions[3 * idx2 + 0];
  const double r0y = positions[3 * idx1 + 1] - positions[3 * idx2 + 1];
  const double r0z = positions[3 * idx1 + 2] - positions[3 * idx2 + 2];
  const double r1x = positions[3 * idx3 + 0] - positions[3 * idx2 + 0];
  const double r1y = positions[3 * idx3 + 1] - positions[3 * idx2 + 1];
  const double r1z = positions[3 * idx3 + 2] - positions[3 * idx2 + 2];
  const double r2x = -r1x;
  const double r2y = -r1y;
  const double r2z = -r1z;
  const double r3x = positions[3 * idx4 + 0] - positions[3 * idx3 + 0];
  const double r3y = positions[3 * idx4 + 1] - positions[3 * idx3 + 1];
  const double r3z = positions[3 * idx4 + 2] - positions[3 * idx3 + 2];

  double t0x, t0y, t0z;
  crossProduct(r0x, r0y, r0z, r1x, r1y, r1z, t0x, t0y, t0z);
  const double d0 = std::max(std::sqrt(t0x * t0x + t0y * t0y + t0z * t0z), 1.0e-5);
  t0x /= d0;
  t0y /= d0;
  t0z /= d0;

  double t1x, t1y, t1z;
  crossProduct(r2x, r2y, r2z, r3x, r3y, r3z, t1x, t1y, t1z);
  const double d1 = std::max(std::sqrt(t1x * t1x + t1y * t1y + t1z * t1z), 1.0e-5);
  t1x /= d1;
  t1y /= d1;
  t1z /= d1;

  const double cosPhi = std::clamp(dotProduct(t0x, t0y, t0z, t1x, t1y, t1z), -1.0, 1.0);
  double       mX, mY, mZ;
  crossProduct(t0x, t0y, t0z, r1x, r1y, r1z, mX, mY, mZ);
  const double mLength  = std::max(std::sqrt(mX * mX + mY * mY + mZ * mZ), 1.0e-5);
  const double dihedral = -std::atan2(dotProduct(mX, mY, mZ, t1x, t1y, t1z) / mLength, cosPhi);
  return kRadiansToDegrees * dihedral;
}

template <typename Contribs>
void appendDistanceConstraintImpl(Contribs&                     contribs,
                                  const std::vector<double>&    positions,
                                  const DistanceConstraintSpec& spec) {
  const int numAtoms = static_cast<int>(positions.size() / 3);
  validateAtomIndex(spec.idx1, numAtoms, "Distance constraint atom");
  validateAtomIndex(spec.idx2, numAtoms, "Distance constraint atom");
  double minLen = spec.minLen;
  double maxLen = spec.maxLen;
  if (maxLen < minLen) {
    throw std::invalid_argument("Distance constraint maxLen must be >= minLen");
  }
  if (spec.relative) {
    const double distance = distanceFromPositions(positions, spec.idx1, spec.idx2);
    minLen                = std::max(minLen + distance, 0.0);
    maxLen                = std::max(maxLen + distance, 0.0);
  }
  contribs.distanceConstraintTerms.idx1.push_back(spec.idx1);
  contribs.distanceConstraintTerms.idx2.push_back(spec.idx2);
  contribs.distanceConstraintTerms.minLen.push_back(minLen);
  contribs.distanceConstraintTerms.maxLen.push_back(maxLen);
  contribs.distanceConstraintTerms.forceConstant.push_back(spec.forceConstant);
}

template <typename Contribs>
void appendPositionConstraintImpl(Contribs&                     contribs,
                                  const std::vector<double>&    positions,
                                  const PositionConstraintSpec& spec) {
  const int numAtoms = static_cast<int>(positions.size() / 3);
  validateAtomIndex(spec.idx, numAtoms, "Position constraint atom");
  contribs.positionConstraintTerms.idx.push_back(spec.idx);
  contribs.positionConstraintTerms.refX.push_back(positions[3 * spec.idx + 0]);
  contribs.positionConstraintTerms.refY.push_back(positions[3 * spec.idx + 1]);
  contribs.positionConstraintTerms.refZ.push_back(positions[3 * spec.idx + 2]);
  contribs.positionConstraintTerms.maxDispl.push_back(spec.maxDispl);
  contribs.positionConstraintTerms.forceConstant.push_back(spec.forceConstant);
}

template <typename Contribs>
void appendAngleConstraintImpl(Contribs&                  contribs,
                               const std::vector<double>& positions,
                               const AngleConstraintSpec& spec) {
  const int numAtoms = static_cast<int>(positions.size() / 3);
  validateAtomIndex(spec.idx1, numAtoms, "Angle constraint atom");
  validateAtomIndex(spec.idx2, numAtoms, "Angle constraint atom");
  validateAtomIndex(spec.idx3, numAtoms, "Angle constraint atom");
  if (spec.maxAngleDeg < spec.minAngleDeg) {
    throw std::invalid_argument("Angle constraint maxAngleDeg must be >= minAngleDeg");
  }
  double minAngleDeg = spec.minAngleDeg;
  double maxAngleDeg = spec.maxAngleDeg;
  if (spec.relative) {
    const double angle = computeAngleDeg(positions, spec.idx1, spec.idx2, spec.idx3);
    minAngleDeg += angle;
    maxAngleDeg += angle;
  }
  if (minAngleDeg < 0.0 || minAngleDeg > 180.0 || maxAngleDeg < 0.0 || maxAngleDeg > 180.0) {
    throw std::invalid_argument("Angle constraint bounds must be within [0, 180]");
  }
  contribs.angleConstraintTerms.idx1.push_back(spec.idx1);
  contribs.angleConstraintTerms.idx2.push_back(spec.idx2);
  contribs.angleConstraintTerms.idx3.push_back(spec.idx3);
  contribs.angleConstraintTerms.minAngleDeg.push_back(minAngleDeg);
  contribs.angleConstraintTerms.maxAngleDeg.push_back(maxAngleDeg);
  contribs.angleConstraintTerms.forceConstant.push_back(spec.forceConstant);
}

template <typename Contribs>
void appendTorsionConstraintImpl(Contribs&                    contribs,
                                 const std::vector<double>&   positions,
                                 const TorsionConstraintSpec& spec) {
  const int numAtoms = static_cast<int>(positions.size() / 3);
  validateAtomIndex(spec.idx1, numAtoms, "Torsion constraint atom");
  validateAtomIndex(spec.idx2, numAtoms, "Torsion constraint atom");
  validateAtomIndex(spec.idx3, numAtoms, "Torsion constraint atom");
  validateAtomIndex(spec.idx4, numAtoms, "Torsion constraint atom");
  if (spec.maxDihedralDeg < spec.minDihedralDeg) {
    throw std::invalid_argument("Torsion constraint maxDihedralDeg must be >= minDihedralDeg");
  }
  double minDihedralDeg = spec.minDihedralDeg;
  double maxDihedralDeg = spec.maxDihedralDeg;
  if (spec.relative) {
    const double dihedral = computeDihedralDeg(positions, spec.idx1, spec.idx2, spec.idx3, spec.idx4);
    minDihedralDeg += dihedral;
    maxDihedralDeg += dihedral;
  }
  minDihedralDeg = normalizeAngleDeg(minDihedralDeg);
  maxDihedralDeg = normalizeAngleDeg(maxDihedralDeg);
  contribs.torsionConstraintTerms.idx1.push_back(spec.idx1);
  contribs.torsionConstraintTerms.idx2.push_back(spec.idx2);
  contribs.torsionConstraintTerms.idx3.push_back(spec.idx3);
  contribs.torsionConstraintTerms.idx4.push_back(spec.idx4);
  contribs.torsionConstraintTerms.minDihedralDeg.push_back(minDihedralDeg);
  contribs.torsionConstraintTerms.maxDihedralDeg.push_back(maxDihedralDeg);
  contribs.torsionConstraintTerms.forceConstant.push_back(spec.forceConstant);
}

void appendDistanceConstraint(nvMolKit::MMFF::EnergyForceContribsHost& contribs,
                              const std::vector<double>&               positions,
                              const DistanceConstraintSpec&            spec) {
  appendDistanceConstraintImpl(contribs, positions, spec);
}
void appendPositionConstraint(nvMolKit::MMFF::EnergyForceContribsHost& contribs,
                              const std::vector<double>&               positions,
                              const PositionConstraintSpec&            spec) {
  appendPositionConstraintImpl(contribs, positions, spec);
}
void appendAngleConstraint(nvMolKit::MMFF::EnergyForceContribsHost& contribs,
                           const std::vector<double>&               positions,
                           const AngleConstraintSpec&               spec) {
  appendAngleConstraintImpl(contribs, positions, spec);
}
void appendTorsionConstraint(nvMolKit::MMFF::EnergyForceContribsHost& contribs,
                             const std::vector<double>&               positions,
                             const TorsionConstraintSpec&             spec) {
  appendTorsionConstraintImpl(contribs, positions, spec);
}

void appendDistanceConstraint(nvMolKit::UFF::EnergyForceContribsHost& contribs,
                              const std::vector<double>&              positions,
                              const DistanceConstraintSpec&           spec) {
  appendDistanceConstraintImpl(contribs, positions, spec);
}
void appendPositionConstraint(nvMolKit::UFF::EnergyForceContribsHost& contribs,
                              const std::vector<double>&              positions,
                              const PositionConstraintSpec&           spec) {
  appendPositionConstraintImpl(contribs, positions, spec);
}
void appendAngleConstraint(nvMolKit::UFF::EnergyForceContribsHost& contribs,
                           const std::vector<double>&              positions,
                           const AngleConstraintSpec&              spec) {
  appendAngleConstraintImpl(contribs, positions, spec);
}
void appendTorsionConstraint(nvMolKit::UFF::EnergyForceContribsHost& contribs,
                             const std::vector<double>&              positions,
                             const TorsionConstraintSpec&            spec) {
  appendTorsionConstraintImpl(contribs, positions, spec);
}

}  // namespace nvMolKit::ForceFieldConstraints
