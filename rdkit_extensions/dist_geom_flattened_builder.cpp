#include "rdkit_extensions/dist_geom_flattened_builder.h"

#include <DistGeom/BoundsMatrix.h>
#include <DistGeom/ChiralSet.h>
#include <GraphMol/ForceFieldHelpers/CrystalFF/TorsionPreferences.h>
#include <RDGeneral/Invariant.h>

#include <boost/dynamic_bitset.hpp>
#include <map>
#include <sstream>

#include "versions.h"

// No clang-tidy for 1:1 RDKit ports.
// NOLINTBEGIN

namespace nvMolKit {
namespace DistGeom {
constexpr double KNOWN_DIST_FORCE_CONSTANT      = 100.0;  // Force constant for known distances
constexpr double KNOWN_DIST_TOL                 = 0.01;   // Tolerance for known distances
constexpr double TRIPLE_BOND_MIN_ANGLE          = 179.0;  // Minimum angle for triple bonds (degrees)
constexpr double TRIPLE_BOND_MAX_ANGLE          = 180.0;  // Maximum angle for triple bonds (degrees)
constexpr double MAX_ANGLE_DEGREES              = 180.0;  // Maximum valid angle in degrees
constexpr double IMPROPER_TORSION_FORCE_SCALING = 10.0;   // Force scaling factor for improper torsion terms
constexpr int    IMPROPER_ATOM_IS_C_BOUND_TO_O  = 5;      // Index for isCBoundToO flag in improperAtom array

// Atomic numbers for group 5 elements
constexpr int ATOMIC_NUMBER_PHOSPHORUS = 15;  // Phosphorus
constexpr int ATOMIC_NUMBER_ARSENIC    = 33;  // Arsenic
constexpr int ATOMIC_NUMBER_ANTIMONY   = 51;  // Antimony
constexpr int ATOMIC_NUMBER_BISMUTH    = 83;  // Bismuth

// Inversion angle constants for group 5 elements (degrees)
constexpr double PHOSPHORUS_INVERSION_ANGLE = 84.4339;  // Phosphorus inversion angle
constexpr double ARSENIC_INVERSION_ANGLE    = 86.9735;  // Arsenic inversion angle
constexpr double ANTIMONY_INVERSION_ANGLE   = 87.7047;  // Antimony inversion angle
constexpr double BISMUTH_INVERSION_ANGLE    = 90.0;     // Bismuth inversion angle

// Mathematical constants for inversion calculations
constexpr double INVERSION_COSINE_FACTOR = 4.0;   // Factor for C1 calculation
constexpr double INVERSION_DOUBLE_ANGLE  = 2.0;   // Factor for double angle calculation
constexpr double INVERSION_FORCE_FACTOR  = 22.0;  // Force constant factor
constexpr double INVERSION_DIVISOR       = 3.0;   // Divisor for final force constant

// Atomic numbers for sp2 elements
constexpr int ATOMIC_NUMBER_CARBON   = 6;  // Carbon
constexpr int ATOMIC_NUMBER_NITROGEN = 7;  // Nitrogen
constexpr int ATOMIC_NUMBER_OXYGEN   = 8;  // Oxygen

// Force constants for sp2 elements
constexpr double SP2_C_BOUND_TO_O_FORCE_CONSTANT = 50.0;  // Force constant for sp2 carbon bound to oxygen
constexpr double SP2_DEFAULT_FORCE_CONSTANT      = 6.0;   // Default force constant for sp2 elements

// Mathematical constants
constexpr double DEGREES_TO_RADIANS_FACTOR = 180.0;  // Factor to convert degrees to radians
constexpr int    TORSION_TERMS_PER_ANGLE   = 6;      // Number of terms per torsion angle

void addDistViolationContribs(nvMolKit::DistGeom::EnergyForceContribsHost& contribs,
                              unsigned int                                 numAtoms,
                              const ::DistGeom::BoundsMatrix&              mmat,
                              std::map<std::pair<int, int>, double>*       extraWeights,
                              double                                       basinSizeTol) {
  for (unsigned int i = 1; i < numAtoms; i++) {
    for (unsigned int j = 0; j < i; j++) {
      double       weight     = 1.0;
      const double lowerBound = mmat.getLowerBound(i, j);
      const double upperBound = mmat.getUpperBound(i, j);
      bool         includeIt  = false;

      if (extraWeights != nullptr) {
        auto mapIt = extraWeights->find(std::make_pair(i, j));
        if (mapIt != extraWeights->end()) {
          weight    = mapIt->second;
          includeIt = true;
        }
      }
      if (upperBound - lowerBound <= basinSizeTol) {
        includeIt = true;
      }
      if (includeIt) {
        contribs.distTerms.idx1.push_back(static_cast<int>(i));
        contribs.distTerms.idx2.push_back(static_cast<int>(j));
        contribs.distTerms.lb2.push_back(lowerBound * lowerBound);
        contribs.distTerms.ub2.push_back(upperBound * upperBound);
        contribs.distTerms.weight.push_back(weight);
      }
    }
  }
}

void addChiralViolationContribs(nvMolKit::DistGeom::EnergyForceContribsHost& contribs,
                                unsigned int                                 numAtoms,
                                const ::DistGeom::VECT_CHIRALSET&            csets,
                                double                                       weightChiral) {
  constexpr double CHIRAL_WEIGHT_THRESHOLD = 1.e-8;
  if (weightChiral <= CHIRAL_WEIGHT_THRESHOLD) {
    return;
  }
  for (const auto& cset : csets) {
    URANGE_CHECK(cset->d_idx1, numAtoms);
    URANGE_CHECK(cset->d_idx2, numAtoms);
    URANGE_CHECK(cset->d_idx3, numAtoms);
    URANGE_CHECK(cset->d_idx4, numAtoms);
    contribs.chiralTerms.idx1.push_back(static_cast<int>(cset->d_idx1));
    contribs.chiralTerms.idx2.push_back(static_cast<int>(cset->d_idx2));
    contribs.chiralTerms.idx3.push_back(static_cast<int>(cset->d_idx3));
    contribs.chiralTerms.idx4.push_back(static_cast<int>(cset->d_idx4));
    contribs.chiralTerms.volLower.push_back(cset->d_volumeLowerBound);
    contribs.chiralTerms.volUpper.push_back(cset->d_volumeUpperBound);
  }
}

void addFourthDimContribs(nvMolKit::DistGeom::EnergyForceContribsHost& contribs,
                          const int                                    dim,
                          unsigned int                                 numAtoms,
                          double                                       weightFourthDim) {
  constexpr double FOURTHDIM_WEIGHT_THRESHOLD = 1.e-8;
  if ((dim != 4) || (weightFourthDim <= FOURTHDIM_WEIGHT_THRESHOLD)) {
    return;
  }
  for (unsigned int i = 0; i < numAtoms; i++) {
    contribs.fourthTerms.idx.push_back(static_cast<int>(i));
  }
}

void addExperimentalTorsionTerms(nvMolKit::DistGeom::Energy3DForceContribsHost&    contribs,
                                 const ::ForceFields::CrystalFF::CrystalFFDetails& etkdgDetails,
                                 unsigned int                                      numAtoms,
                                 boost::dynamic_bitset<>&                          atomPairs) {
  // Process each experimental torsion term
  for (unsigned int torsionIdx = 0; torsionIdx < etkdgDetails.expTorsionAtoms.size(); ++torsionIdx) {
    const int atomIdx1 = etkdgDetails.expTorsionAtoms[torsionIdx][0];
    const int atomIdx2 = etkdgDetails.expTorsionAtoms[torsionIdx][1];
    const int atomIdx3 = etkdgDetails.expTorsionAtoms[torsionIdx][2];
    const int atomIdx4 = etkdgDetails.expTorsionAtoms[torsionIdx][3];

    // Validate indices using RDKit's precondition and range checks
    PRECONDITION((atomIdx1 != atomIdx2) && (atomIdx1 != atomIdx3) && (atomIdx1 != atomIdx4) && (atomIdx2 != atomIdx3) &&
                   (atomIdx2 != atomIdx4) && (atomIdx3 != atomIdx4),
                 "degenerate points");
    URANGE_CHECK(static_cast<unsigned int>(atomIdx1), numAtoms);
    URANGE_CHECK(static_cast<unsigned int>(atomIdx2), numAtoms);
    URANGE_CHECK(static_cast<unsigned int>(atomIdx3), numAtoms);
    URANGE_CHECK(static_cast<unsigned int>(atomIdx4), numAtoms);

    // Update atomPairs (similar to RDKit's atomPairs logic)
    if (atomIdx1 < atomIdx4) {
      atomPairs[atomIdx1 * numAtoms + atomIdx4] = true;
    } else {
      atomPairs[atomIdx4 * numAtoms + atomIdx1] = true;
    }

    // Add the torsion contribution
    contribs.experimentalTorsionTerms.idx1.push_back(atomIdx1);
    contribs.experimentalTorsionTerms.idx2.push_back(atomIdx2);
    contribs.experimentalTorsionTerms.idx3.push_back(atomIdx3);
    contribs.experimentalTorsionTerms.idx4.push_back(atomIdx4);

    // Add force constants and signs (6 values per torsion)
    const auto& signs          = etkdgDetails.expTorsionAngles[torsionIdx].first;
    const auto& forceConstants = etkdgDetails.expTorsionAngles[torsionIdx].second;

    // Ensure we have 6 values for each torsion
    for (int term = 0; term < TORSION_TERMS_PER_ANGLE; ++term) {
      if (term < static_cast<int>(forceConstants.size())) {
        contribs.experimentalTorsionTerms.forceConstants.push_back(forceConstants[term]);
      } else {
        contribs.experimentalTorsionTerms.forceConstants.push_back(0.0);
      }

      if (term < static_cast<int>(signs.size())) {
        contribs.experimentalTorsionTerms.signs.push_back(static_cast<int>(signs[term]));
      } else {
        contribs.experimentalTorsionTerms.signs.push_back(0);
      }
    }
  }
}

std::tuple<double, double, double, double> calcInversionCoefficientsAndForceConstant(int  at2AtomicNum,
                                                                                     bool isCBoundToO) {
  double res             = 0.0;
  double inversionCoeff0 = 0.0;
  double inversionCoeff1 = 0.0;
  double inversionCoeff2 = 0.0;
  // if the central atom is sp2 carbon, nitrogen or oxygen
  if ((at2AtomicNum == ATOMIC_NUMBER_CARBON) || (at2AtomicNum == ATOMIC_NUMBER_NITROGEN) ||
      (at2AtomicNum == ATOMIC_NUMBER_OXYGEN)) {
    inversionCoeff0 = 1.0;
    inversionCoeff1 = -1.0;
    inversionCoeff2 = 0.0;
    res             = (isCBoundToO ? SP2_C_BOUND_TO_O_FORCE_CONSTANT : SP2_DEFAULT_FORCE_CONSTANT);
  } else {
    // group 5 elements are not clearly explained in the UFF paper
    // the following code was inspired by MCCCS Towhee's ffuff.F
    double angleInRadians = M_PI / DEGREES_TO_RADIANS_FACTOR;
    switch (at2AtomicNum) {
      // if the central atom is phosphorous
      case ATOMIC_NUMBER_PHOSPHORUS:
        angleInRadians *= PHOSPHORUS_INVERSION_ANGLE;
        break;

      // if the central atom is arsenic
      case ATOMIC_NUMBER_ARSENIC:
        angleInRadians *= ARSENIC_INVERSION_ANGLE;
        break;

      // if the central atom is antimonium
      case ATOMIC_NUMBER_ANTIMONY:
        angleInRadians *= ANTIMONY_INVERSION_ANGLE;
        break;

      // if the central atom is bismuth
      case ATOMIC_NUMBER_BISMUTH:
        angleInRadians *= BISMUTH_INVERSION_ANGLE;
        break;

      default:
        // For any other atomic number, use default behavior (angleInRadians remains M_PI / 180.0)
        // This handles cases where the atomic number is not a group 5 element
        break;
    }
    inversionCoeff2 = 1.0;
    inversionCoeff1 = -INVERSION_COSINE_FACTOR * cos(angleInRadians);
    inversionCoeff0 =
      -(inversionCoeff1 * cos(angleInRadians) + inversionCoeff2 * cos(INVERSION_DOUBLE_ANGLE * angleInRadians));
    res = INVERSION_FORCE_FACTOR / (inversionCoeff0 + inversionCoeff1 + inversionCoeff2);
  }
  res /= INVERSION_DIVISOR;

  return std::make_tuple(res, inversionCoeff0, inversionCoeff1, inversionCoeff2);
}

void addImproperTorsionTerms(nvMolKit::DistGeom::Energy3DForceContribsHost&    contribs,
                             const ::ForceFields::CrystalFF::CrystalFFDetails& etkdgDetails,
                             unsigned int                                      numAtoms,
                             double                                            forceScalingFactor,
                             boost::dynamic_bitset<>&                          isImproperConstrained) {
  // Process each improper torsion term
  contribs.improperTorsionTerms.numImpropers.push_back(etkdgDetails.improperAtoms.size());
  for (const auto& improperAtom : etkdgDetails.improperAtoms) {
    // Create 3 different permutations for each improper torsion
    std::vector<int> atomOrder(4);
    for (unsigned int i = 0; i < 3; ++i) {
      atomOrder[1] = 1;  // Central atom is always at position 1
      switch (i) {
        case 0:
          atomOrder[0] = 0;
          atomOrder[2] = 2;
          atomOrder[3] = 3;
          break;

        case 1:
          atomOrder[0] = 0;
          atomOrder[2] = 3;
          atomOrder[3] = 2;
          break;

        case 2:
          atomOrder[0] = 2;
          atomOrder[2] = 3;
          atomOrder[3] = 0;
          break;

        default:
          // This should never happen since i is constrained to 0, 1, 2 in the for loop
          // But adding default case to satisfy QA requirements
          break;
      }

      // Extract atom indices
      const int  idx1         = improperAtom[atomOrder[0]];
      const int  idx2         = improperAtom[atomOrder[1]];  // Central atom
      const int  idx3         = improperAtom[atomOrder[2]];
      const int  idx4         = improperAtom[atomOrder[3]];
      const int  at2AtomicNum = improperAtom[4];
      const bool isCBoundToO  = static_cast<bool>(improperAtom[IMPROPER_ATOM_IS_C_BOUND_TO_O]);

      // Validate indices using RDKit's range checks
      URANGE_CHECK(static_cast<unsigned int>(idx1), numAtoms);
      URANGE_CHECK(static_cast<unsigned int>(idx2), numAtoms);
      URANGE_CHECK(static_cast<unsigned int>(idx3), numAtoms);
      URANGE_CHECK(static_cast<unsigned int>(idx4), numAtoms);

      // Calculate inversion coefficients and force constant
      // Note: This would need to be implemented or imported from RDKit
      // For now, we'll use placeholder values that should be replaced with actual calculations
      auto invCoeffForceCon = calcInversionCoefficientsAndForceConstant(at2AtomicNum, isCBoundToO);

      const double inversionCoeff0 = std::get<1>(invCoeffForceCon);
      const double inversionCoeff1 = std::get<2>(invCoeffForceCon);
      const double inversionCoeff2 = std::get<3>(invCoeffForceCon);
      const double forceConstant   = std::get<0>(invCoeffForceCon) * forceScalingFactor;

      // Add the improper torsion contribution
      contribs.improperTorsionTerms.idx1.push_back(idx1);
      contribs.improperTorsionTerms.idx2.push_back(idx2);
      contribs.improperTorsionTerms.idx3.push_back(idx3);
      contribs.improperTorsionTerms.idx4.push_back(idx4);
      contribs.improperTorsionTerms.at2AtomicNum.push_back(at2AtomicNum);
      contribs.improperTorsionTerms.isCBoundToO.push_back(isCBoundToO);
      contribs.improperTorsionTerms.C0.push_back(inversionCoeff0);
      contribs.improperTorsionTerms.C1.push_back(inversionCoeff1);
      contribs.improperTorsionTerms.C2.push_back(inversionCoeff2);
      contribs.improperTorsionTerms.forceConstant.push_back(forceConstant);

      // Mark the central atom as constrained
      isImproperConstrained[idx2] = true;
    }
  }
}

void addDistanceConstraintContrib(nvMolKit::DistGeom::DistanceConstraintContribTerms& contribs,
                                  unsigned int                                        idx1,
                                  unsigned int                                        idx2,
                                  double                                              minLen,
                                  double                                              maxLen,
                                  double                                              forceConstant) {
  PRECONDITION(maxLen >= minLen, "bad bounds");
  contribs.idx1.push_back(static_cast<int>(idx1));
  contribs.idx2.push_back(static_cast<int>(idx2));
  contribs.minLen.push_back(minLen);
  contribs.maxLen.push_back(maxLen);
  contribs.forceConstant.push_back(forceConstant);
}

void add12Terms(nvMolKit::DistGeom::Energy3DForceContribsHost&    contribs,
                const ::ForceFields::CrystalFF::CrystalFFDetails& etkdgDetails,
                boost::dynamic_bitset<>&                          atomPairs,
                const std::vector<double>&                        positions,
                int                                               dim,
                double                                            forceConstant,
                unsigned int                                      numAtoms) {
  for (const auto& bond : etkdgDetails.bonds) {
    const unsigned int atomIdx1 = bond.first;
    const unsigned int atomIdx2 = bond.second;

    // Update atomPairs bitset
    if (atomIdx1 < atomIdx2) {
      atomPairs[atomIdx1 * numAtoms + atomIdx2] = true;
    } else {
      atomPairs[atomIdx2 * numAtoms + atomIdx1] = true;
    }

    // Calculate current distance between atoms using flattened positions
    const double deltaX   = positions[atomIdx1 * dim + 0] - positions[atomIdx2 * dim + 0];
    const double deltaY   = positions[atomIdx1 * dim + 1] - positions[atomIdx2 * dim + 1];
    const double deltaZ   = positions[atomIdx1 * dim + 2] - positions[atomIdx2 * dim + 2];
    const double distance = sqrt(deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ);

    // Add distance constraint contribution
    addDistanceConstraintContrib(contribs.dist12Terms,
                                 atomIdx1,
                                 atomIdx2,
                                 distance - KNOWN_DIST_TOL,
                                 distance + KNOWN_DIST_TOL,
                                 forceConstant);
  }
}

void addAngleConstraintContrib(nvMolKit::DistGeom::AngleConstraintContribTerms& contribs,
                               unsigned int                                     idx1,
                               unsigned int                                     idx2,
                               unsigned int                                     idx3,
                               double                                           minAngleDeg,
                               double                                           maxAngleDeg) {
  // Range checks for angles (0-180 degrees)
  PRECONDITION(minAngleDeg >= 0.0 && minAngleDeg <= MAX_ANGLE_DEGREES, "minAngleDeg out of range");
  PRECONDITION(maxAngleDeg >= 0.0 && maxAngleDeg <= MAX_ANGLE_DEGREES, "maxAngleDeg out of range");
  PRECONDITION(maxAngleDeg >= minAngleDeg, "minAngleDeg must be <= maxAngleDeg");

  contribs.idx1.push_back(static_cast<int>(idx1));
  contribs.idx2.push_back(static_cast<int>(idx2));
  contribs.idx3.push_back(static_cast<int>(idx3));
  contribs.minAngle.push_back(minAngleDeg);
  contribs.maxAngle.push_back(maxAngleDeg);
}

void add13Terms(nvMolKit::DistGeom::Energy3DForceContribsHost&    contribs,
                const ::ForceFields::CrystalFF::CrystalFFDetails& etkdgDetails,
                boost::dynamic_bitset<>&                          atomPairs,
                const std::vector<double>&                        positions,
                int                                               dim,
                double                                            forceConstant,
                const boost::dynamic_bitset<>&                    isImproperConstrained,
                bool                                              useBasicKnowledge,
                const ::DistGeom::BoundsMatrix&                   mmat,
                unsigned int                                      numAtoms) {
  for (const auto& angle : etkdgDetails.angles) {
    const unsigned int atomIdx1 = angle[0];
    const unsigned int atomIdx2 = angle[1];  // Central atom
    const unsigned int atomIdx3 = angle[2];

    // Update atomPairs bitset
    if (atomIdx1 < atomIdx3) {
      atomPairs[atomIdx1 * numAtoms + atomIdx3] = true;
    } else {
      atomPairs[atomIdx3 * numAtoms + atomIdx1] = true;
    }

    // Check for triple bonds
    if (useBasicKnowledge && angle[3] != 0) {
      // Add angle constraint for triple bonds (179-180 degrees)
      addAngleConstraintContrib(contribs.angle13Terms,
                                atomIdx1,
                                atomIdx2,
                                atomIdx3,
                                TRIPLE_BOND_MIN_ANGLE,
                                TRIPLE_BOND_MAX_ANGLE);
    } else if (isImproperConstrained[atomIdx2]) {
      // Use bounds matrix for improper constrained central atoms
      addDistanceConstraintContrib(contribs.dist13Terms,
                                   atomIdx1,
                                   atomIdx3,
                                   mmat.getLowerBound(atomIdx1, atomIdx3),
                                   mmat.getUpperBound(atomIdx1, atomIdx3),
                                   forceConstant);
      contribs.dist13Terms.isImproperConstrained.push_back(true);
    } else {
      // Calculate current distance between atoms using flattened positions
      const double deltaX   = positions[atomIdx1 * dim + 0] - positions[atomIdx3 * dim + 0];
      const double deltaY   = positions[atomIdx1 * dim + 1] - positions[atomIdx3 * dim + 1];
      const double deltaZ   = positions[atomIdx1 * dim + 2] - positions[atomIdx3 * dim + 2];
      const double distance = sqrt(deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ);

      // Add distance constraint contribution
      addDistanceConstraintContrib(contribs.dist13Terms,
                                   atomIdx1,
                                   atomIdx3,
                                   distance - KNOWN_DIST_TOL,
                                   distance + KNOWN_DIST_TOL,
                                   forceConstant);
      contribs.dist13Terms.isImproperConstrained.push_back(false);
    }
  }
}

void addLongRangeDistanceConstraints(nvMolKit::DistGeom::Energy3DForceContribsHost&    contribs,
                                     const ::ForceFields::CrystalFF::CrystalFFDetails& etkdgDetails,
                                     const boost::dynamic_bitset<>&                    atomPairs,
                                     [[maybe_unused]] const std::vector<double>&       positions,
                                     [[maybe_unused]] int                              dim,
                                     [[maybe_unused]] double                           knownDistanceForceConstant,
                                     const ::DistGeom::BoundsMatrix&                   mmat,
                                     unsigned int                                      numAtoms) {
  for (unsigned int i = 1; i < numAtoms; ++i) {
    for (unsigned int j = 0; j < i; ++j) {
      if (!atomPairs[j * numAtoms + i]) {
        double l = mmat.getLowerBound(i, j);
        double u = mmat.getUpperBound(i, j);
#if RDKIT_NEW_FLAG_API
        double fdist = etkdgDetails.boundsMatForceScaling * 10.0;
        if (!etkdgDetails.constrainedAtoms.empty() && etkdgDetails.constrainedAtoms[i] &&
            etkdgDetails.constrainedAtoms[j]) {
          // We're constrained, so use very tight bounds
          double dx          = positions[i * dim + 0] - positions[j * dim + 0];
          double dy          = positions[i * dim + 1] - positions[j * dim + 1];
          double dz          = positions[i * dim + 2] - positions[j * dim + 2];
          double currentDist = sqrt(dx * dx + dy * dy + dz * dz);

          l     = currentDist - KNOWN_DIST_TOL;
          u     = currentDist + KNOWN_DIST_TOL;
          fdist = knownDistanceForceConstant;
        }
#else
        const double fdist = etkdgDetails.boundsMatForceScaling * 10.0;
#endif
        addDistanceConstraintContrib(contribs.longRangeDistTerms, i, j, l, u, fdist);
      }
    }
  }
}

nvMolKit::DistGeom::EnergyForceContribsHost constructForceFieldContribs(
  const int                              dim,
  const ::DistGeom::BoundsMatrix&        mmat,
  const ::DistGeom::VECT_CHIRALSET&      csets,
  double                                 weightChiral,
  double                                 weightFourthDim,
  std::map<std::pair<int, int>, double>* extraWeights,
  double                                 basinSizeTol) {
  nvMolKit::DistGeom::EnergyForceContribsHost contribs;
  const unsigned int                          numAtoms = mmat.numRows();

  // Add contributions
  addDistViolationContribs(contribs, numAtoms, mmat, extraWeights, basinSizeTol);
  addChiralViolationContribs(contribs, numAtoms, csets, weightChiral);
  addFourthDimContribs(contribs, dim, numAtoms, weightFourthDim);

  return contribs;
}

nvMolKit::DistGeom::Energy3DForceContribsHost construct3DForceFieldContribs(
  const ::DistGeom::BoundsMatrix&                   mmat,
  const ::ForceFields::CrystalFF::CrystalFFDetails& etkdgDetails,
  const std::vector<double>&                        positions,
  int                                               dim,
  bool                                              useBasicKnowledge) {
  nvMolKit::DistGeom::Energy3DForceContribsHost contribs;
  const unsigned int                            numAtoms = mmat.numRows();

  // Initialize atomPairs bitset for tracking atom pairs
  boost::dynamic_bitset<> atomPairs(numAtoms * numAtoms);

  // 1. addExperimentalTorsionTerms
  addExperimentalTorsionTerms(contribs, etkdgDetails, numAtoms, atomPairs);

  // 2. addImproperTorsionTerms (only if useBasicKnowledge is true)
  boost::dynamic_bitset<> isImproperConstrained(numAtoms);
  if (useBasicKnowledge) {
    addImproperTorsionTerms(contribs, etkdgDetails, numAtoms, IMPROPER_TORSION_FORCE_SCALING, isImproperConstrained);
  }

  // 3. add12Terms
  add12Terms(contribs, etkdgDetails, atomPairs, positions, dim, KNOWN_DIST_FORCE_CONSTANT, numAtoms);

  // 4. add13Terms (pass useBasicKnowledge parameter)
  add13Terms(contribs,
             etkdgDetails,
             atomPairs,
             positions,
             dim,
             KNOWN_DIST_FORCE_CONSTANT,
             isImproperConstrained,
             useBasicKnowledge,
             mmat,
             numAtoms);

  // 5. addLongRangeDistanceConstraints
  addLongRangeDistanceConstraints(contribs,
                                  etkdgDetails,
                                  atomPairs,
                                  positions,
                                  dim,
                                  KNOWN_DIST_FORCE_CONSTANT,
                                  mmat,
                                  numAtoms);

  return contribs;
}

// NOLINTEND
}  // namespace DistGeom
}  // namespace nvMolKit