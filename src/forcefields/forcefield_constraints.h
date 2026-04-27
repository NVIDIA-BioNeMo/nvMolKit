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

#ifndef NVMOLKIT_FORCEFIELD_CONSTRAINTS_H
#define NVMOLKIT_FORCEFIELD_CONSTRAINTS_H

#include <string>
#include <vector>

// TODO: Constraint term types, constraint kernels, and launchReduceEnergiesKernel currently live in
// mmff.h / mmff_kernels.h but are forcefield-generic. They should be extracted into shared headers
// so that UFF (and future forcefields) don't depend on MMFF.
#include "src/forcefields/mmff.h"
#include "src/forcefields/uff.h"

namespace nvMolKit::ForceFieldConstraints {

struct DistanceConstraintSpec {
  int    idx1          = -1;
  int    idx2          = -1;
  bool   relative      = false;
  double minLen        = 0.0;
  double maxLen        = 0.0;
  double forceConstant = 0.0;
};

struct PositionConstraintSpec {
  int    idx           = -1;
  double maxDispl      = 0.0;
  double forceConstant = 0.0;
};

struct AngleConstraintSpec {
  int    idx1          = -1;
  int    idx2          = -1;
  int    idx3          = -1;
  bool   relative      = false;
  double minAngleDeg   = 0.0;
  double maxAngleDeg   = 0.0;
  double forceConstant = 0.0;
};

struct TorsionConstraintSpec {
  int    idx1           = -1;
  int    idx2           = -1;
  int    idx3           = -1;
  int    idx4           = -1;
  bool   relative       = false;
  double minDihedralDeg = 0.0;
  double maxDihedralDeg = 0.0;
  double forceConstant  = 0.0;
};

void   validateAtomIndex(int idx, int numAtoms, const std::string& what);
double distanceFromPositions(const std::vector<double>& positions, int idx1, int idx2);
double computeAngleDeg(const std::vector<double>& positions, int idx1, int idx2, int idx3);
double computeDihedralDeg(const std::vector<double>& positions, int idx1, int idx2, int idx3, int idx4);
double normalizeAngleDeg(double angleDeg);

void appendDistanceConstraint(nvMolKit::MMFF::EnergyForceContribsHost& contribs,
                              const std::vector<double>&               positions,
                              const DistanceConstraintSpec&            spec);
void appendPositionConstraint(nvMolKit::MMFF::EnergyForceContribsHost& contribs,
                              const std::vector<double>&               positions,
                              const PositionConstraintSpec&            spec);
void appendAngleConstraint(nvMolKit::MMFF::EnergyForceContribsHost& contribs,
                           const std::vector<double>&               positions,
                           const AngleConstraintSpec&               spec);
void appendTorsionConstraint(nvMolKit::MMFF::EnergyForceContribsHost& contribs,
                             const std::vector<double>&               positions,
                             const TorsionConstraintSpec&             spec);

void appendDistanceConstraint(nvMolKit::UFF::EnergyForceContribsHost& contribs,
                              const std::vector<double>&              positions,
                              const DistanceConstraintSpec&           spec);
void appendPositionConstraint(nvMolKit::UFF::EnergyForceContribsHost& contribs,
                              const std::vector<double>&              positions,
                              const PositionConstraintSpec&           spec);
void appendAngleConstraint(nvMolKit::UFF::EnergyForceContribsHost& contribs,
                           const std::vector<double>&              positions,
                           const AngleConstraintSpec&              spec);
void appendTorsionConstraint(nvMolKit::UFF::EnergyForceContribsHost& contribs,
                             const std::vector<double>&              positions,
                             const TorsionConstraintSpec&            spec);

struct PerMolConstraints {
  std::vector<DistanceConstraintSpec> distance;
  std::vector<PositionConstraintSpec> position;
  std::vector<AngleConstraintSpec>    angle;
  std::vector<TorsionConstraintSpec>  torsion;

  bool empty() const { return distance.empty() && position.empty() && angle.empty() && torsion.empty(); }

  template <typename Contribs> void applyTo(Contribs& contribs, const std::vector<double>& positions) const {
    for (const auto& s : distance) {
      appendDistanceConstraint(contribs, positions, s);
    }
    for (const auto& s : position) {
      appendPositionConstraint(contribs, positions, s);
    }
    for (const auto& s : angle) {
      appendAngleConstraint(contribs, positions, s);
    }
    for (const auto& s : torsion) {
      appendTorsionConstraint(contribs, positions, s);
    }
  }
};

}  // namespace nvMolKit::ForceFieldConstraints

#endif  // NVMOLKIT_FORCEFIELD_CONSTRAINTS_H
