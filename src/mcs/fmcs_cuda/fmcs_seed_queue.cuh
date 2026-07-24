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

#ifndef FMCS_CUDA_FMCS_SEED_QUEUE_CUH
#define FMCS_CUDA_FMCS_SEED_QUEUE_CUH

#include <cooperative_groups.h>

#include <cstdint>

namespace mcs {
namespace fmcs {

/// Scope tag selecting the backing store + synchronization for
/// @ref SeedQueue.  ThreadBlockScope is the only implemented variant.
struct ThreadBlockScope {};

/// Cooperative-group-scoped seed queue.  LIFO semantics match fMCS's
/// SEED_GROW_DEEP: grow the biggest child first, only backtrack on match
/// failure.  @c push returns false when full; callers are expected to
/// flag @ref MCSResult::overflowed at that point.
///
/// Capacity is a runtime value so the host can size the backing slab
/// against a memory budget rather than a compile-time tier.  The backing
/// store lives in global memory (one per-block slab handed in via
/// @ref init); only the queue header (top index and pointer) lives in
/// the block's shared memory, so capacity is bound by GPU RAM, not
/// shared memory.
///
/// The default constructor leaves the queue empty with a null backing
/// store so the class can be declared as a @c __shared__ variable; lane 0
/// must call @ref init before any push/pop.
template <class Element, class Scope> class SeedQueue {
 public:
  __device__ __forceinline__ SeedQueue() = default;

  __device__ __forceinline__ void init(Element* storage, int capacity) {
    storage_  = storage;
    capacity_ = capacity;
    top_      = 0;
  }

  /// Within-thread: single lane atomically reserves one slot and writes
  /// @p element into it.  Returns false if the queue is full, in which
  /// case no slot was consumed.
  ///
  /// Concurrency: safe under multiple concurrent pushers (each gets a
  /// distinct slot via CAS).  The kernel's iteration structure also
  /// guarantees pushes and pops never overlap (block.sync rendezvous
  /// between iterations), so the slot-write after the CAS is not
  /// racing with a popper read of the same slot.
  __device__ __forceinline__ bool pushWithinThread(const Element& element) {
    const int slot = adjustTopAtomic(+1);
    if (slot < 0)
      return false;
    storage_[slot] = element;
    return true;
  }

  /// Within-thread: single lane atomically claims the LIFO top and
  /// copies it out into @p outElement.  Returns false if the queue is
  /// empty.
  ///
  /// Concurrency: safe under multiple concurrent poppers (CAS gives
  /// each successful caller a distinct slot).  Relies on the kernel's
  /// temporal separation of pops (start of iteration) from pushes
  /// (later in the same iteration) -- the slot read after a successful
  /// CAS is therefore not racing with a pusher write to that slot.
  __device__ __forceinline__ bool popWithinThread(Element& outElement) {
    const int slot = adjustTopAtomic(-1);
    if (slot < 0)
      return false;
    outElement = storage_[slot - 1];
    return true;
  }

  /// Despite the @c Cooperative suffix, the actual reservation work is lane-0
  /// serial: lane 0 runs the atomicCAS-loop on @c top_, and the only
  /// group-coordinated step is broadcasting the result via
  /// @c group.shfl.  Returns -1 to all lanes when the reservation
  /// would overflow @ref capacity, in which case no slots were
  /// consumed.  Successful callers write directly into @ref slot
  /// starting at the returned offset; THOSE writes are the parallel
  /// part (one per lane).
  template <class GroupT> __device__ __forceinline__ int batchReserveCooperative(const GroupT& group, int count) {
    int start = -1;
    if (group.thread_rank() == 0)
      start = adjustTopAtomic(count);
    return group.shfl(start, 0);
  }

  /// Cooperative reservation for a single LIFO pop.  Returns the old top to
  /// every lane, or -1 if the queue was empty.  Callers copy from
  /// @c slot(oldTop - 1) after a successful reservation.
  template <class GroupT> __device__ __forceinline__ int popReserveCooperative(const GroupT& group) {
    int oldTop = -1;
    if (group.thread_rank() == 0)
      oldTop = adjustTopAtomic(-1);
    return group.shfl(oldTop, 0);
  }

  __device__ __forceinline__ Element&       slot(int i) { return storage_[i]; }
  __device__ __forceinline__ const Element& slot(int i) const { return storage_[i]; }

  __device__ __forceinline__ int  size() const { return top_; }
  __device__ __forceinline__ bool empty() const { return top_ == 0; }
  __device__ __forceinline__ int  capacity() const { return capacity_; }

  __device__ __forceinline__ void setSizeWithinThread(int size) { top_ = size; }

 private:
  /// Within-thread CAS-loop on @c top_ that atomically applies a signed
  /// @p delta to the queue cursor.  Returns the @c top_ value before
  /// the update on success, or -1 if the resulting top would fall
  /// outside @c [0, capacity_] (no slots consumed).  All three of
  /// @ref pushWithinThread, @ref popWithinThread, and
  /// @ref batchReserveCooperative reduce to a single call here.
  __device__ __forceinline__ int adjustTopAtomic(int delta) {
    int oldTop = top_;
    while (true) {
      const int newTop = oldTop + delta;
      if (newTop < 0 || newTop > capacity_)
        return -1;
      const int prev = atomicCAS(&top_, oldTop, newTop);
      if (prev == oldTop)
        return oldTop;
      oldTop = prev;
    }
  }

  Element* storage_  = nullptr;
  int      capacity_ = 0;
  int      top_      = 0;
};

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_SEED_QUEUE_CUH
