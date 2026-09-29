// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_DIVERSITY_PICKER_ALGORITHMS_CUH
#define NVMOLKIT_DIVERSITY_PICKER_ALGORITHMS_CUH

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "src/butina_common.cuh"
#include "src/diversity_pickers.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"
#include "src/utils/host_vector.h"

// Leader is written against a distance provider. A provider implements
//
//   int size() const;
//   int leaderWindow() const;
//   template <typename Op> void forEachDistance(const int* sources, int numSources, const Op& op, cudaStream_t);
//
// forEachDistance visits every candidate c for which op.skip(c) is false:
//
//   auto state = op.start(c);
//   for k in [0, numSources): op.visit(state, k, sources[k] == c, distance(sources[k] -> c));
//   op.finish(c, state);
//
// Sources are visited in order and read from device memory; sources outside [0, size()) are not visited. Distances
// are single precision. leaderWindow() is how many consecutive Leader candidates to resolve per pass, from 1 through
// kMaxLeaderWindow: cheap metrics amortize synchronization over a wide window, while expensive ones avoid evaluating
// candidates that the window resolution then rejects.

namespace nvMolKit::detail {

constexpr int kPickerBlockSize = 256;
constexpr int kMaxLeaderWindow = 32;

inline int pickerGridSize(const int numItems) {
  return (numItems + kPickerBlockSize - 1) / kPickerBlockSize;
}

//! Records which window members are within the cutoff of each active candidate.
struct LeaderHitsOp {
  const std::uint8_t* active;
  std::uint32_t*      hits;
  float               cutoff;

  using State = std::uint32_t;
  __device__ bool  skip(const int candidate) const { return active[candidate] == 0; }
  __device__ State start(const int /*candidate*/) const { return 0U; }
  __device__ void  visit(State& state, const int ordinal, const bool /*self*/, const float distance) const {
    if (distance <= cutoff) {
      state |= 1U << ordinal;
    }
  }
  __device__ void finish(const int candidate, const State state) const { hits[candidate] = state; }
};

//! Device state for windowed Leader selection.
struct LeaderState {
  std::uint8_t*  active;
  std::uint32_t* hits;
  int*           picks;
  int*           count;
  int*           window;
  std::uint32_t* accepted;
  int*           scanStart;
};

// Implemented in diversity_pickers.cu.
void launchGatherLeaderWindow(const LeaderState& state, int numItems, int windowSize, cudaStream_t stream);
//! Forced windows accept every member. Otherwise the resolution also decides whether the selection loop continues.
void launchResolveLeaderWindow(const LeaderState&         state,
                               int                        windowSize,
                               bool                       forced,
                               int                        limit,
                               cudaGraphConditionalHandle loop,
                               cudaStream_t               stream);
void launchApplyLeaderWindow(const LeaderState& state, int numItems, cudaStream_t stream);
void validateFirstPicks(const std::vector<int>& firstPicks, int numItems);

//! Returns the first @p count picks as an exactly sized result.
inline PickerResult makePickerResult(const AsyncDeviceVector<int>& picks, const int count, cudaStream_t stream) {
  PickerResult result{AsyncDeviceVector<int>(count, stream)};
  if (count > 0) {
    cudaCheckError(
      cudaMemcpyAsync(result.indices.data(), picks.data(), count * sizeof(int), cudaMemcpyDeviceToDevice, stream));
  }
  return result;
}

/**
 * Sequential sphere exclusion, resolved a window at a time.
 *
 * Each window holds the next consecutive active candidates. One provider pass records, for every active candidate,
 * which window members are within the cutoff. A member is accepted as a leader when no earlier accepted member
 * excludes it, and every other active candidate is then excluded by its first accepted hit. This reproduces the
 * one-leader-at-a-time result. Forced picks use the same passes with every member accepted.
 */
template <typename Provider>
PickerResult leaderPick(Provider&               provider,
                        const float             cutoff,
                        const int               pickSize,
                        const std::vector<int>& firstPicks,
                        cudaStream_t            stream) {
  const int numItems = provider.size();
  validateFirstPicks(firstPicks, numItems);
  if (pickSize < 0 || pickSize > numItems) {
    throw std::invalid_argument("pick_size must be between 0 and the input size");
  }
  if (numItems == 0) {
    return PickerResult{AsyncDeviceVector<int>(0, stream)};
  }

  const int                        limit      = pickSize == 0 ? numItems : pickSize;
  const int                        windowSize = std::clamp(provider.leaderWindow(), 1, kMaxLeaderWindow);
  AsyncDeviceVector<int>           picks(std::max<std::size_t>(limit, firstPicks.size()), stream);
  AsyncDeviceVector<std::uint8_t>  active(numItems, stream);
  AsyncDeviceVector<std::uint32_t> hits(numItems, stream);
  AsyncDeviceVector<int>           window(kMaxLeaderWindow, stream);
  AsyncDevicePtr<int>              count(0, stream);
  AsyncDevicePtr<int>              scanStart(0, stream);
  AsyncDevicePtr<std::uint32_t>    accepted(0U, stream);
  cudaCheckError(cudaMemsetAsync(active.data(), 1, numItems, stream));

  const LeaderState
    state{active.data(), hits.data(), picks.data(), count.data(), window.data(), accepted.data(), scanStart.data()};
  const LeaderHitsOp hitsOp{active.data(), hits.data(), cutoff};

  for (std::size_t first = 0; first < firstPicks.size(); first += kMaxLeaderWindow) {
    const auto chunk = std::min<std::size_t>(kMaxLeaderWindow, firstPicks.size() - first);
    window.copyFromHost(firstPicks, chunk, first);
    provider.forEachDistance(window.data(), static_cast<int>(chunk), hitsOp, stream);
    launchResolveLeaderWindow(state, static_cast<int>(chunk), true, limit, {}, stream);
    launchApplyLeaderWindow(state, numItems, stream);
  }

  // Windows are resolved on the device until the pick limit is reached or no active candidate remains.
  if (static_cast<int>(firstPicks.size()) < limit) {
    const ConditionalLoopGraph loop([&](cudaStream_t captureStream, cudaGraphConditionalHandle handle) {
      launchGatherLeaderWindow(state, numItems, windowSize, captureStream);
      provider.forEachDistance(window.data(), windowSize, hitsOp, captureStream);
      launchResolveLeaderWindow(state, windowSize, false, limit, handle, captureStream);
      launchApplyLeaderWindow(state, numItems, captureStream);
    });
    loop.launch(stream);
  }
  int countHost = 0;
  count.get(countHost);
  cudaCheckError(cudaStreamSynchronize(stream));
  return makePickerResult(picks, countHost, stream);
}

}  // namespace nvMolKit::detail

#endif  // NVMOLKIT_DIVERSITY_PICKER_ALGORITHMS_CUH
