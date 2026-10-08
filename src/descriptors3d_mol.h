// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_DESCRIPTORS3D_MOL_H
#define NVMOLKIT_DESCRIPTORS3D_MOL_H

#include <cuda_runtime.h>

#include <vector>

#include "src/conformer/device_coord_result.h"
#include "src/descriptors3d.h"

namespace RDKit {
class ROMol;
}  // namespace RDKit

namespace nvMolKit {

//! Properties for a molecule batch together with the labels of each output row.
template <typename Real> struct Property3DBatchResult {
  Property3DResults<Real>    properties;
  //! Conformer row -> input molecule index and per-molecule conformer position, and each conformer's range of
  //! per-atom rows (`numConformers + 1` offsets). Populated only when coordinates were taken from the
  //! molecules; otherwise rows follow the caller's DeviceCoordView and these are empty.
  AsyncDeviceVector<int32_t> molIndices;
  AsyncDeviceVector<int32_t> confIndices;
  AsyncDeviceVector<int32_t> atomStarts;
};

/**
 * @brief Calculate the requested 3D properties for every conformer of a molecule batch.
 *
 * Molecules always supply atom identity (and therefore atom weights). Coordinates come from the
 * molecules' RDKit conformers unless @p coordinates is given, in which case they are read in place
 * from that device batch and the molecules' own conformers are ignored.
 *
 * @tparam Real           float (PrecisionMode::SINGLE) or double (PrecisionMode::FULL) arithmetic and output.
 * @param mols            Non-null molecules. Output rows follow input-molecule order, then
 *                        conformer order (or the row order of @p coordinates).
 * @param properties      Non-empty, duplicate-free property selection.
 * @param options         Per-family options; see Property3DOptions.
 * @param stream          CUDA stream for all transfers and computation.
 * @param coordinates     Optional device coordinates; `coordinates->nMols` must equal @c mols.size().
 *                        Device coordinate rows are treated as three-dimensional for PBF; molecule
 *                        conformers keep their RDKit is3D flag.
 * @param preprocessingThreads CPU threads for coordinate and atom-weight extraction; -1 uses every
 *                        available OpenMP thread.
 * @throws std::invalid_argument on null molecules, an invalid property selection, options or thread
 *                               count, or a molecule count mismatch with @p coordinates.
 */
template <typename Real>
Property3DBatchResult<Real> calc3DProperties(const std::vector<const RDKit::ROMol*>& mols,
                                             const std::vector<Property3D>&          properties,
                                             const Property3DOptions&                options,
                                             cudaStream_t                            stream,
                                             const DeviceCoordView*                  coordinates          = nullptr,
                                             int                                     preprocessingThreads = -1);

}  // namespace nvMolKit

#endif  // NVMOLKIT_DESCRIPTORS3D_MOL_H
