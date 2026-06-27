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

#ifndef NVMOLKIT_FIRE_OPTIONS_H
#define NVMOLKIT_FIRE_OPTIONS_H

namespace nvMolKit {

//! \brief Algorithm parameters for the FIRE minimizer.
//!
//! Implements the ASE FIRE 2.0 update rules
//! (https://gitlab.com/ase/ase/-/blob/master/ase/optimize/fire2.py).
//! The default parameter values below are tuned for nvMolKit GPU workloads.
//! Working units inside the kernel are kcal/mol for energy, Å for position,
//! amu for mass and ps for time.
struct FireOptions {
  //! Defaults below are the optimum from the @c benchmarks/fire_optuna.py "gpu" study
  //! (stored in @c benchmarks/fire_optuna_gpu_v4.db) at @c maxIters=200 on the
  //! perturbed-MMFF dataset. They differ from the literal ASE FIRE2 reference values.
  double dtInit      = 0.0035256954965291066;   //!< Initial time step in picoseconds.
  double dtMinFactor = 0.00014570290330215527;  //!< Lower bound for dt as a fraction of dtInit.
  double dtMaxFactor = 5.3536466978846375;      //!< Upper bound for dt as a fraction of dtInit.

  //! \brief Maximum 2-norm of the per-step displacement vector dr = dt*v, in Å.
  //! Skipped when @ref abcCorrection is true (matches ASE FIRE2 behavior).
  double dMax = 0.6925293686798697;

  double timeStepIncrement =
    1.2751646491363886;  //!< Multiplicative dt increase factor when power has been positive for nMinForIncrease steps.
  double timeStepDecrement = 0.6158984212819867;  //!< Multiplicative dt decrease factor when power becomes negative.

  int nMinForIncrease = 3;  //!< Number of consecutive positive-power steps required before dt is allowed to grow.

  double alphaInit      = 0.2890058136581572;  //!< Initial value of the mixing coefficient alpha.
  double alphaDecrement = 0.9574425933142592;  //!< Multiplicative alpha decay applied while power stays positive.

  //! \brief When true, divide the per-coordinate force kick by the per-atom mass.
  //! Note: ASE FIRE2 implicitly uses mass = 1 in its native unit system. Enabling
  //! @ref useMass here weights the integrator by per-atom masses (a deliberate
  //! deviation from ASE).
  bool useMass = false;

  double gradTol = 1e-4;  //!< Convergence threshold on sqrt(sum(grad^2)) per system.

  //! \brief Take a half step backward when the power becomes negative
  //! (r -= 0.5 * dt * v with the post-decrement dt). Always true for ASE FIRE2;
  //! disable to recover the FIRE 1.0 reset behavior.
  bool takeHalfStepBack = true;

  //! \brief Apply the Accelerated Bias-Correction multiplier
  //! 1 / (1 - (1 - alpha)^(N+1)) to the mixer (ABC-FIRE).
  bool abcCorrection = false;

  //! \brief Detect "stuck" systems via energy plateau and declare them converged.
  //!
  //! FIRE has no analog of BFGS's MOVETOL/FUNCTOL exits, so a system that oscillates
  //! around a local minimum (or plateaus) without ever reaching @ref gradTol burns
  //! the full iteration budget. When enabled, the minimizer evaluates the energy at
  //! each convergence-poll boundary, tracks per-system min/max energy across a
  //! sliding window of @ref stuckStreakLength polls, and declares the system
  //! converged (status 0) once the windowed extrema satisfy
  //! @code
  //! (max - min) / max(|E_now|, 1) < stuckEnergyRelTol
  //! @endcode
  //! for @ref stuckStreakLength consecutive polls. Streak resets whenever a poll
  //! sees a relative energy change above the tolerance.
  bool   stuckDetectionEnabled = false;
  double stuckEnergyRelTol     = 1e-3;  //!< Relative |windowed extrema| / max(|E|, 1) tolerance.
  int    stuckStreakLength     = 3;     //!< Consecutive plateau polls required to declare stuck.
  int    stuckEvalEveryNPolls  = 1;     //!< Sample energy every Nth convergence poll (1 = every poll).
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_FIRE_OPTIONS_H
