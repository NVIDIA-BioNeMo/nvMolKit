# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent semantic model of nvMolKit's concurrent BitBIRCH schedule.

This CPU semantic model is intentionally not a scalable storage implementation.
It uses unpacked input and uint64 sums to make membership auditing straightforward.
Logical batches and structural subrounds define the algorithm. No bblean
implementation is used here.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

POPCOUNT = np.asarray([value.bit_count() for value in range(256)], dtype=np.uint8)
ORDERED_WARMUP_SIZE = 16384


def isim(sums: np.ndarray, count: int) -> float:
    values = sums.astype(np.float64)
    common = np.sum(values * (values - 1) * 0.5)
    mismatch = np.sum(values * (count - values))
    return float(common / (common + mismatch)) if common + mismatch else 1.0


def centroid(sums: np.ndarray, count: int) -> np.ndarray:
    return np.packbits(sums >= (count + 1) // 2)


def similarities(queries: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    intersection = POPCOUNT[queries[:, None, :] & candidates[None, :, :]].sum(axis=2)
    union = POPCOUNT[queries[:, None, :] | candidates[None, :, :]].sum(axis=2)
    return np.divide(intersection, union, out=np.ones_like(intersection, dtype=float), where=union != 0)


@dataclass(eq=False)
class Entry:
    uid: int
    count: int
    sums: np.ndarray
    packed: np.ndarray
    members: list[int] = field(default_factory=list)
    child: "Node | None" = None


@dataclass(eq=False)
class Node:
    uid: int
    leaf: bool
    entries: list[Entry] = field(default_factory=list)
    parent: "Node | None" = None
    cached_centroids: np.ndarray | None = None

    def matrix(self) -> np.ndarray:
        if self.cached_centroids is None:
            self.cached_centroids = np.stack([entry.packed for entry in self.entries])
        return self.cached_centroids


class BatchedBitBirch:
    def __init__(
        self,
        threshold=0.25,
        branching_factor=254,
        batch_size=256,
    ):
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be finite and in [0, 1]")
        if branching_factor < 3 or batch_size < 1:
            raise ValueError("branching factor >= 3 and positive batch size required")
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.batch_size = batch_size
        self.root = Node(0, True)
        self.next_node = 1
        self.stats = Counter()
        self.bits = np.empty((0, 0), dtype=np.uint8)
        self.packed = np.empty((0, 0), dtype=np.uint8)

    def _node(self, leaf: bool, entries: list[Entry], parent=None) -> Node:
        result = Node(self.next_node, leaf, entries, parent)
        self.next_node += 1
        for entry in entries:
            if entry.child is not None:
                entry.child.parent = result
        return result

    def _summary(self, node: Node) -> Entry:
        sums = np.sum([entry.sums for entry in node.entries], axis=0, dtype=np.uint64)
        count = sum(entry.count for entry in node.entries)
        return Entry(-node.uid - 1, count, sums, centroid(sums, count), child=node)

    def _refresh(self, node: Node) -> None:
        if not node.leaf:
            for index, entry in enumerate(node.entries):
                self._refresh(entry.child)
                node.entries[index] = self._summary(entry.child)
        node.cached_centroids = None

    def _route(self, molecule_ids: np.ndarray):
        pending = [(self.root, molecule_ids)]
        groups = defaultdict(list)
        leaves = {}
        entries = {}
        while pending:
            node, ids = pending.pop()
            leaves[node.uid] = node
            if not node.entries:
                groups[(node.uid, None)].extend(ids.tolist())
                continue
            choices = np.argmax(similarities(self.packed[ids], node.matrix()), axis=1)
            for choice in np.unique(choices):
                selected = ids[choices == choice]
                entry = node.entries[int(choice)]
                if node.leaf:
                    groups[(node.uid, entry.uid)].extend(selected.tolist())
                    entries[entry.uid] = entry
                else:
                    pending.append((entry.child, selected))
        self.stats["route_attempts"] += len(molecule_ids)
        self.stats["max_cluster_queue"] = max(self.stats["max_cluster_queue"], *map(len, groups.values()))
        return groups, leaves, entries

    def _commit(self, entry: Entry, ids: list[int], sums: np.ndarray) -> None:
        entry.sums = sums
        entry.count += len(ids)
        entry.packed = centroid(sums, entry.count)
        entry.members.extend(ids)

    def _try_group(self, entry: Entry, ids: list[int]) -> list[int]:
        residual = []
        eligible = []
        for molecule in ids:
            if isim(entry.sums + self.bits[molecule], entry.count + 1) >= self.threshold:
                eligible.append(molecule)
            else:
                residual.append(molecule)
        if not eligible:
            return residual
        combined = entry.sums + self.bits[eligible].sum(axis=0, dtype=np.uint64)
        if isim(combined, entry.count + len(eligible)) >= self.threshold:
            self._commit(entry, eligible, combined)
            return residual
        for molecule in eligible:
            combined = entry.sums + self.bits[molecule]
            if isim(combined, entry.count + 1) >= self.threshold:
                self._commit(entry, [molecule], combined)
            else:
                residual.append(molecule)
        return residual

    def _insert_residual(self, node: Node, molecule: int) -> None:
        if node.entries:
            selected = int(np.argmax(similarities(self.packed[molecule : molecule + 1], node.matrix())[0]))
            entry = node.entries[selected]
            combined = entry.sums + self.bits[molecule]
            self.stats["residual_checks"] += 1
            if isim(combined, entry.count + 1) >= self.threshold:
                self._commit(entry, [molecule], combined)
                node.cached_centroids = None
                self.stats["residual_committed"] += 1
                return
        sums = self.bits[molecule].astype(np.uint64)
        node.entries.append(Entry(molecule, 1, sums, self.packed[molecule].copy(), [molecule]))
        node.cached_centroids = None
        self.stats["created_clusters"] += 1

    def _split(self, node: Node) -> None:
        count = len(node.entries)
        assert count == self.branching_factor + 1
        centroids = node.matrix()
        distances = similarities(centroids, centroids)
        distances[np.tril_indices(count)] = np.inf
        lhs, rhs = np.unravel_index(np.argmin(distances), distances.shape)
        seeds = similarities(centroids, centroids[[lhs, rhs]])
        left = [node.entries[lhs]]
        right = [node.entries[rhs]]
        capacity = (count + 1) // 2
        for index, entry in enumerate(node.entries):
            if index in (lhs, rhs):
                continue
            take_left = seeds[index, 0] > seeds[index, 1] or (
                seeds[index, 0] == seeds[index, 1] and len(left) <= len(right)
            )
            if len(left) >= capacity:
                take_left = False
            elif len(right) >= capacity:
                take_left = True
            if take_left:
                left.append(entry)
            else:
                right.append(entry)
        node.entries = left
        node.cached_centroids = None
        for entry in left:
            if entry.child is not None:
                entry.child.parent = node
        sibling = self._node(node.leaf, right, node.parent)
        if node.parent is None:
            self.root = self._node(False, [self._summary(node), self._summary(sibling)])
        else:
            parent = node.parent
            for index, entry in enumerate(parent.entries):
                if entry.child is node:
                    parent.entries[index] = self._summary(node)
                    break
            parent.entries.append(self._summary(sibling))
            parent.cached_centroids = None
            if len(parent.entries) > self.branching_factor:
                self._split(parent)
        self.stats["splits"] += 1

    def _epoch(self, ids: np.ndarray) -> None:
        ordered_leaf = ids[0] < ORDERED_WARMUP_SIZE
        pending = ids
        while len(pending):
            self.stats["subrounds"] += 1
            groups, nodes, entries = self._route(pending)
            residual = defaultdict(list)
            # All routing has finished. Leaf owners touch disjoint data.
            keys = sorted(groups)
            for node_id, entry_id in keys:
                ids = groups[(node_id, entry_id)]
                if entry_id is None or ordered_leaf:
                    residual[node_id].extend(ids)
                else:
                    residual[node_id].extend(self._try_group(entries[entry_id], ids))
                nodes[node_id].cached_centroids = None
            overflow = []
            parked = []
            for node_id in sorted(residual):
                node = nodes[node_id]
                ids = sorted(residual[node_id])
                self.stats["max_residual_leaf_queue"] = max(self.stats["max_residual_leaf_queue"], len(ids))
                for offset, molecule in enumerate(ids):
                    self._insert_residual(node, molecule)
                    if len(node.entries) > self.branching_factor:
                        overflow.append(node)
                        parked.extend(ids[offset + 1 :])
                        break
            # All payload updates finish before topology changes. Refresh first so
            # cascading splits see current summaries of every sibling.
            self._refresh(self.root)
            for node in sorted(overflow, key=node_uid):
                self._split(node)
                # A previous split may have created/moved an ancestor of the next.
                self._refresh(self.root)
            pending = np.asarray(sorted(parked), dtype=np.int64)
            self.stats["parked_for_split"] += len(pending)

    def fit(self, packed: np.ndarray):
        data = np.asarray(packed)
        if data.ndim != 2 or data.dtype != np.uint8 or data.shape[1] == 0:
            raise ValueError("expected packed uint8 matrix with nonzero width")
        if self.root.entries:
            raise ValueError("reference fit requires a fresh tree")
        self.packed = data
        self.bits = np.unpackbits(data, axis=1)
        begin = 0
        while begin < len(data):
            boundary = min(ORDERED_WARMUP_SIZE, len(data)) if begin < ORDERED_WARMUP_SIZE else len(data)
            end = min(begin + self.batch_size, boundary)
            self.stats["epochs"] += 1
            self._epoch(np.arange(begin, end, dtype=np.int64))
            begin = end
        return self

    def clusters(self) -> list[Entry]:
        pending = [self.root]
        result = []
        while pending:
            node = pending.pop()
            if node.leaf:
                result.extend(node.entries)
            else:
                pending.extend(entry.child for entry in node.entries)
        return sorted(result, key=first_member)

    def labels(self) -> np.ndarray:
        labels = np.full(len(self.packed), -1, dtype=np.int64)
        for label, entry in enumerate(self.clusters()):
            labels[entry.members] = label
        return labels

    def audit(self) -> dict:
        """Recompute from actual members, independently of incremental updates."""
        members = []
        scores = []
        pending = [self.root]
        while pending:
            node = pending.pop()
            assert len(node.entries) <= self.branching_factor
            if node is not self.root:
                assert len(node.entries) >= 2
            for entry in node.entries:
                if node.leaf:
                    assert entry.child is None
                    assert entry.count == len(entry.members)
                    expected = self.bits[entry.members].sum(axis=0, dtype=np.uint64)
                    members.extend(entry.members)
                    value = isim(expected, entry.count)
                    assert value + 1e-12 >= self.threshold
                    scores.append(value)
                else:
                    assert entry.child.parent is node
                    assert not entry.members
                    assert entry.count == sum(child.count for child in entry.child.entries)
                    expected = np.sum([child.sums for child in entry.child.entries], axis=0, dtype=np.uint64)
                    pending.append(entry.child)
                np.testing.assert_array_equal(entry.sums, expected)
                np.testing.assert_array_equal(entry.packed, centroid(expected, entry.count))
        assert sorted(members) == list(range(len(self.packed)))
        return {"audited_molecules": len(members), "minimum_isim": min(scores, default=1.0)}


def node_uid(node: Node) -> int:
    return node.uid


def first_member(entry: Entry) -> int:
    return min(entry.members)
