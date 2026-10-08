# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Molecules kept on the GPU for repeated substructure queries."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, TypeVar

from rdkit.Chem import Mol

from nvmolkit._substructLibrary import SubstructLibrary as _NativeSubstructLibrary
from nvmolkit.substructure import SubstructSearchConfig

__all__ = ["SubstructLibrary"]

_T = TypeVar("_T")


class SubstructLibrary:
    """A collection of molecules kept on the GPU for repeated substructure searches.

    This is the GPU counterpart of RDKit's ``rdSubstructLibrary.SubstructLibrary``:
    target molecules are prepared and uploaded once, then any number of queries
    run against them. Use it when the same targets are searched many times; for a
    one-off batch of targets and queries, :func:`nvmolkit.substructure.hasSubstructMatch`
    avoids the upload step.

    Molecules added with :meth:`addMol` or :meth:`addMols` receive stable indices
    in insertion order and become searchable after :meth:`finalize`. Queries keep
    seeing the last finalized collection while newly added molecules are pending,
    so the library can grow between query rounds.

    Which calls wait:

    * :meth:`addMol`, :meth:`addMols`, and :meth:`finalize` return when their work
      is done. They also wait for any queries currently running to finish.
    * :meth:`getMatches`, :meth:`countMatches`, and :meth:`hasMatch` return a
      :class:`concurrent.futures.Future` right away and search in the background,
      so you can keep working or start more queries. Up to
      :attr:`maxConcurrentQueries` queries run at once; the rest wait their turn.
      Call ``result()`` on the future to get the answer.
    * :meth:`getMatchesSync`, :meth:`countMatchesSync`, and :meth:`hasMatchSync`
      start a query the same way and return its answer once it is ready.

    For the best throughput, start all of your queries before calling ``result()``
    on the first one, rather than waiting for each answer before starting the next.

    A query's errors, including querying before the first :meth:`finalize`, are
    raised by the future's ``result()`` or by the ``*Sync`` call. A query searches
    the molecules that were finalized when it begins running, which may include
    molecules finalized after it was started. Do not wait for a query's result
    inside another future's done-callback; that can hang.

    The library can be used from several threads at once.

    Matching follows RDKit's ``SubstructLibrary`` with ``useChirality=False``:
    recursive SMARTS are supported, and stereochemistry in targets and in SMILES
    queries is ignored. SMARTS queries that specify atom chirality (``@`` or
    ``@@``) are not supported; their results raise ``RuntimeError``. Targets outside the
    GPU representation limits (more than 128 atoms, high-degree atoms, isotopes
    above 255, or dative and other unsupported bond types) are matched with
    RDKit on the CPU, so results cover every added molecule.

    Example::

        from rdkit import Chem
        from nvmolkit.substruct_library import SubstructLibrary

        library = SubstructLibrary()
        library.addMols(Chem.MolFromSmiles(smiles) for smiles in ["CCO", "c1ccccc1O", "CCN"])
        library.finalize()
        library.getMatchesSync(Chem.MolFromSmarts("[OX2H]"))  # [0, 1]

        # Start several queries, then collect their results.
        futures = [library.hasMatch(Chem.MolFromSmarts(smarts)) for smarts in ["N", "S"]]
        [future.result() for future in futures]  # [True, False]

    Args:
        config: Search configuration. ``algorithm`` selects the ``"dfs"`` (default)
            or ``"gsi"`` backend, and ``gpuIds`` spreads the molecules across
            several GPUs. ``batchSize``, ``workerThreads``, and
            ``preprocessingThreads`` tune throughput; ``maxMatches`` and
            ``uniquify`` do not apply because queries return molecule indices.
    """

    def __init__(self, config: SubstructSearchConfig | None = None) -> None:
        """Create an empty library."""
        if config is None:
            config = SubstructSearchConfig()
        self._native = _NativeSubstructLibrary(config._as_native())
        self._executorLock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nvmolkit-substruct")

    def __len__(self) -> int:
        """Return the number of searchable molecules."""
        return len(self._native)

    @property
    def pendingSize(self) -> int:
        """Number of molecules added since the last :meth:`finalize`."""
        return self._native.pendingSize

    @property
    def maxConcurrentQueries(self) -> int:
        """How many queries can run at once, based on the GPU memory left after :meth:`finalize`; 0 before it."""
        return int(self._native.maxConcurrentQueries)

    def addMol(self, molecule: Mol) -> int:
        """Copy ``molecule`` into the library and return its index.

        The molecule becomes searchable after the next :meth:`finalize`.
        """
        return self._native.addMol(molecule)

    def addMols(self, molecules: Iterable[Mol]) -> list[int]:
        """Copy ``molecules`` into the library and return their indices."""
        return self._native.addMols(molecules)

    def finalize(self) -> None:
        """Upload molecules added since the last call and make them searchable.

        Must be called at least once before querying; until then, query results
        raise ``RuntimeError``. Queries that have not begun running yet will search
        the newly finalized molecules. If it raises, the previously finalized
        molecules stay searchable and the new ones stay pending.
        """
        try:
            self._native.finalize()
        finally:
            # A failed finalize keeps the previous molecules searchable, so always size the
            # executor to what the native library now allows. Queued queries keep running.
            with self._executorLock:
                previous = self._executor
                self._executor = ThreadPoolExecutor(
                    max_workers=max(1, self.maxConcurrentQueries),
                    thread_name_prefix="nvmolkit-substruct",
                )
            previous.shutdown(wait=False)

    def _submit(self, function: Callable[..., _T], *args: Any) -> Future[_T]:
        with self._executorLock:
            return self._executor.submit(function, *args)

    def getMatches(self, query: Mol, maxResults: int = -1) -> Future[list[int]]:
        """Start a query and return a future for the indices of matching molecules, without waiting.

        Args:
            query: Query molecule, typically from ``Chem.MolFromSmarts`` or ``Chem.MolFromSmiles``.
            maxResults: Return at most this many indices, keeping the lowest. -1 (default) returns all.

        Returns:
            Future resolving to matching indices in ascending order. Its ``result()``
            raises any error from the search.
        """
        return self._submit(self._native.getMatches, query, int(maxResults))

    def countMatches(self, query: Mol) -> Future[int]:
        """Start a query and return a future for the number of matching molecules, without waiting."""
        return self._submit(self._native.countMatches, query)

    def hasMatch(self, query: Mol) -> Future[bool]:
        """Start a query and return a future for whether any molecule matches, without waiting."""
        return self._submit(self._native.hasMatch, query)

    def getMatchesSync(self, query: Mol, maxResults: int = -1) -> list[int]:
        """Like :meth:`getMatches`, but wait for and return the matching indices."""
        return self.getMatches(query, maxResults).result()

    def countMatchesSync(self, query: Mol) -> int:
        """Like :meth:`countMatches`, but wait for and return the number of matches."""
        return self.countMatches(query).result()

    def hasMatchSync(self, query: Mol) -> bool:
        """Like :meth:`hasMatch`, but wait for and return whether any molecule matches."""
        return self.hasMatch(query).result()
