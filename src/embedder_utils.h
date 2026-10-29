// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * This code is primarily adapted from RDKit's embedder.h to facilitate
 * generating a reference ETKDG force field for unit testing.
 *
 * Note: The code has not been fully optimized and may contain redundant
 * functions or components that are unnecssary for generating the reference
 * force field.
 *
 * TODO:
 * - Extract and refine the core components to implement clean, minimal
 * functions for generating a reference force field for unit testing.
 */
#ifndef NVMOLKIT_EMBEDDER_UTILS_H
#define NVMOLKIT_EMBEDDER_UTILS_H

#include <DistGeom/BoundsMatrix.h>
#include <DistGeom/ChiralSet.h>
#include <ForceField/ForceField.h>
#include <Geometry/point.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/CrystalFF/TorsionPreferences.h>
#include <GraphMol/ROMol.h>
#include <RDGeneral/export.h>

#include <boost/dynamic_bitset.hpp>
#include <boost/shared_ptr.hpp>
#include <map>
#include <memory>
#include <utility>

using namespace RDKit;

namespace nvMolKit {

namespace detail {

/**
 * @brief Enum representing different minimize stages
 */
enum class MinimizeStage {
  FirstMinimize,
  FourthDimMinimize
};

/**
 * @brief Struct to hold conformer-related data
 */
struct ConformerData {
  std::vector<std::unique_ptr<Conformer>> confs;
  boost::dynamic_bitset<>                 confsOk;
};

//! Embed arguments for the ETKDG algorithm.
struct EmbedArgs {
  int                                                               dim  = 3;
  ::DistGeom::BoundsMatPtr                                          mmat = nullptr;
  ::DistGeom::VECT_CHIRALSET                                        chiralCenters;
  ::DistGeom::VECT_CHIRALSET                                        tetrahedralCarbons;
  ForceFields::CrystalFF::CrystalFFDetails                          etkdgDetails{};
  std::vector<std::tuple<unsigned int, unsigned int, unsigned int>> doubleBondEnds;
  std::vector<std::pair<std::vector<unsigned int>, int>>            stereoDoubleBonds;
  std::vector<double>                                               posVec;
  MinimizeStage                                                     stage = MinimizeStage::FirstMinimize;
  //! Owns the topological distance matrix etkdgDetails.distMat points into; clones share it.
  std::shared_ptr<double[]>                                         topologicalDistances;
};

//! Copy prepared arguments for one conformer attempt, including owned RDKit state.
EmbedArgs cloneEmbedArgs(const EmbedArgs& source);

}  // namespace detail

namespace DGeomHelpers {

/**
 * @brief Enum representing dimensionality control for force field generation
 */
enum class Dimensionality {
  DIM_3D,  // Force 3D coordinates
  DIM_4D   // Force 4D coordinates
};

/**
 * @brief Throws ValueErrorException for embedding parameters that nvMolKit does not implement yet
 *
 * Unsupported: ETversion other than 1 or 2, and on
 * RDKit >= 2026.09, embedForceField other than UFF, onlyInitialEmbedding and internal-coordinate initial embedding.
 */
void validateSupportedEmbedParameters(const RDKit::DGeomHelpers::EmbedParameters& params);

/**
 * @brief Returns the parameters nvMolKit embeds with.
 *
 * Random starting coordinates raise the basin threshold to 1e8 as RDKit does. On RDKit >= 2026.09, useRandomCoords
 * selects initialEmbeddingMode=RANDOM_COORDINATE_EMBEDDING as in RDKit, and that mode also selects useRandomCoords:
 * RDKit deprecates the flag in favor of the mode but still applies its random-coordinate settings only for the flag.
 */
RDKit::DGeomHelpers::EmbedParameters normalizeEmbedParameters(const RDKit::DGeomHelpers::EmbedParameters& params);

/**
 * @brief Prepares embedder arguments for the ETKDG algorithm
 *
 * @param mol The molecule to prepare arguments for
 * @param params The embedding parameters
 * @param eargs The embedder arguments to be populated
 * @param setupBoundsMatrix Whether to set up topology bounds and triangle smoothing (default: true)
 * @return true if successful, false if bounds matrix setup failed
 */
bool prepareEmbedderArgs(ROMol&                                      mol,
                         const RDKit::DGeomHelpers::EmbedParameters& params,
                         detail::EmbedArgs&                          eargs,
                         bool                                        setupBoundsMatrix = true);

/**
 * @brief Generates a force field using RDKit's functionality
 *
 * @param mol The molecule to generate force field for
 * @param params The embedding parameters
 * @param eargs The embedder arguments
 * @param positions The positions vector to be populated
 * @param dimensionality The dimensionality control
 * @return std::unique_ptr<ForceFields::ForceField> The generated force field
 */
std::unique_ptr<ForceFields::ForceField> generateRDKitFF(ROMol&                                       mol,
                                                         RDKit::DGeomHelpers::EmbedParameters&        params,
                                                         detail::EmbedArgs&                           eargs,
                                                         std::vector<std::unique_ptr<RDGeom::Point>>& positions,
                                                         Dimensionality dimensionality = Dimensionality::DIM_4D);

/**
 * @brief Sets up a force field and updates position vector
 *
 * @param mol The molecule to set up force field for
 * @param params The ETKDG params to use. May have a few fields modified based on coord gen settings.
 * @param field The force field to be populated
 * @param eargs The embedder arguments
 * @param positions The positions vector to be populated
 * @param confId The conformer ID to use (-1 means use default conformer)
 * @param dimensionality The dimensionality control
 */
void setupRDKitFFWithPos(RDKit::ROMol*                                mol,
                         RDKit::DGeomHelpers::EmbedParameters&        params,
                         std::unique_ptr<ForceFields::ForceField>&    field,
                         detail::EmbedArgs&                           eargs,
                         std::vector<std::unique_ptr<RDGeom::Point>>& positions,
                         int                                          confId = -1,  // -1 means use default conformer
                         Dimensionality                               dimensionality = Dimensionality::DIM_4D);

namespace EmbeddingOps {
//! \brief Identifies chiral centers and tetrahedral carbons in a molecule per RDKit spec.
void findChiralSets(const ROMol&                          mol,
                    ::DistGeom::VECT_CHIRALSET&           chiralCenters,
                    ::DistGeom::VECT_CHIRALSET&           tetrahedralCenters,
                    const std::map<int, RDGeom::Point3D>* coordMap);

//! \brief Finds double bonds in a molecule and populates the provided vectors with their endpoints and stereo
//! information.
void findDoubleBonds(const ROMol&                                                       mol,
                     std::vector<std::tuple<unsigned int, unsigned int, unsigned int>>& doubleBondEnds,
                     std::vector<std::pair<std::vector<unsigned int>, int>>&            stereoDoubleBonds,
                     const std::map<int, RDGeom::Point3D>*                              coordMap);

//! \brief Fills the ETKDG experimental torsion, improper and force-constant details as nvMolKit's embedding does:
//! the refinement's force constants first, which RDKit's torsion setup scales by, then the torsions. On RDKit 2026.09
//! it must run after the topology bounds are set.
void initETKDG(const ROMol*                                mol,
               const RDKit::DGeomHelpers::EmbedParameters& params,
               ForceFields::CrystalFF::CrystalFFDetails&   etkdgDetails);

//! \brief Sets up topological bounds in the bounds matrix (before triangle smoothing)
void setupTopologyBounds(const ROMol*                                mol,
                         const ::DistGeom::BoundsMatPtr&             mmat,
                         const RDKit::DGeomHelpers::EmbedParameters& params,
                         ForceFields::CrystalFF::CrystalFFDetails&   etkdgDetails);

//! \brief Sets up relaxed bounds matrix when first triangle smoothing fails
void setupRelaxedBounds(const ROMol*                                mol,
                        const ::DistGeom::BoundsMatPtr&             mmat,
                        const RDKit::DGeomHelpers::EmbedParameters& params);

//! \brief Sets up bounds matrix when ignoring triangle smoothing failures
void setupIgnoredSmoothingBounds(const ROMol*                                mol,
                                 const ::DistGeom::BoundsMatPtr&             mmat,
                                 const RDKit::DGeomHelpers::EmbedParameters& params);

std::unique_ptr<ForceFields::ForceField> constructForceField(std::vector<std::unique_ptr<RDGeom::Point>>& positions,
                                                             const detail::EmbedArgs&                     eargs,
                                                             RDKit::DGeomHelpers::EmbedParameters&        embedParams);
}  // namespace EmbeddingOps
}  // namespace DGeomHelpers
}  // namespace nvMolKit

#endif  // NVMOLKIT_EMBEDDER_UTILS_H
