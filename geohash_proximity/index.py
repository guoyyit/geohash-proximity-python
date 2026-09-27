"""Thread-safe in-memory geohash proximity index.

Concurrency model
-----------------
Every public operation takes a single :class:`threading.RLock`.  Query
methods gather an immutable snapshot of candidate :class:`Point` objects
while holding the lock, then release it before doing the (CPU-bound)
haversine math.  Because :class:`~geohash_proximity.models.Point` is an
immutable value object and points are replaced rather than mutated, every
query observes exactly one consistent point set -- never a half-applied
insert/update/delete and never a duplicate or vanished id.

Duplicate-id policy
-------------------
* ``insert`` rejects an existing id with ``DuplicatePointError``.
* ``update``/``delete`` reject unknown ids with ``PointNotFoundError``.
* ``upsert`` deterministically replaces the existing point.
* ``bulk_load`` supports ``on_duplicate`` of ``"replace"`` (default, last
  write wins), ``"skip"`` (first write wins) or ``"raise"`` (fail the whole
  atomic batch).  Within-batch ordering follows Python's iteration order;
  the outcome never depends on the history of the index.
"""
from __future__ import annotations

import math
import threading
from collections.abc import Iterable, Mapping
from typing import Any

from . import geo
from .config import IndexConfig
from .errors import (
    DuplicatePointError,
    InvalidParameterError,
    LimitExceededError,
    PointNotFoundError,
)
from .models import Point, ScoredPoint

# Distance tolerance for the inclusive "<= radius" boundary test.
_DISTANCE_EPSILON_M = 1e-9


class ProximityIndex:
    """Maintains points in a multi-precision geohash grid."""

    def __init__(self, config: IndexConfig | None = None) -> None:
        self._config = config or IndexConfig()
        self._lock = threading.RLock()
        self._points: dict[str, Point] = {}
        # One grid per geohash precision 1..P: cell -> set(point ids).
        self._grids: list[dict[str, set[str]]] = [
            {} for _ in range(self._config.geohash_precision)
        ]
        # Refcount of every geohash prefix under which points currently live
        # (points themselves included at their full precision).  Used to
        # prune empty subtrees during best-first nearest traversal.
        self._prefix_refcount: dict[str, int] = {}

    @property
    def config(self) -> IndexConfig:
        return self._config

    def __len__(self) -> int:
        with self._lock:
            return len(self._points)

    def __contains__(self, point_id: str) -> bool:
        with self._lock:
            return point_id in self._points

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_id(point_id: Any) -> str:
        if not isinstance(point_id, str) or not point_id:
            raise InvalidParameterError("point id must be a non-empty string")
        return point_id

    def _make_point(
        self,
        point_id: str,
        latitude: float,
        longitude: float,
        attributes: Mapping[str, Any] | None,
    ) -> Point:
        self._check_id(point_id)
        geo.validate_coordinates(latitude, longitude)
        if attributes is None:
            attrs: dict[str, Any] = {}
        elif isinstance(attributes, Mapping):
            attrs = dict(attributes)
        else:
            raise InvalidParameterError("attributes must be a mapping or None")
        return Point(point_id, float(latitude), float(longitude), attrs)

    def _index_point(self, point: Point) -> None:
        for precision in range(1, self._config.geohash_precision + 1):
            cell = geo.encode_geohash(
                point.latitude, point.longitude, precision
            )
            self._grids[precision - 1].setdefault(cell, set()).add(point.id)
            self._prefix_refcount[cell] = self._prefix_refcount.get(cell, 0) + 1

    def _unindex_point(self, point: Point) -> None:
        for precision in range(1, self._config.geohash_precision + 1):
            cell = geo.encode_geohash(
                point.latitude, point.longitude, precision
            )
            bucket = self._grids[precision - 1].get(cell)
            if bucket is not None:
                bucket.discard(point.id)
                if not bucket:
                    del self._grids[precision - 1][cell]
            count = self._prefix_refcount.get(cell, 0) - 1
            if count <= 0:
                self._prefix_refcount.pop(cell, None)
            else:
                self._prefix_refcount[cell] = count

    def _snapshot_candidates(self, cells: frozenset[str]) -> list[Point]:
        """Gather an immutable, de-duplicated snapshot of points in cells."""
        candidates: dict[str, Point] = {}
        with self._lock:
            for cell in cells:
                precision = len(cell)
                ids = self._grids[precision - 1].get(cell)
                if not ids:
                    continue
                for point_id in ids:
                    if point_id not in candidates:
                        candidates[point_id] = self._points[point_id]
        return list(candidates.values())

    def _count_candidates(self, cells: frozenset[str]) -> int:
        """Cheap distinct-id count in the bbox cells (union under lock)."""
        seen: set[str] = set()
        with self._lock:
            for cell in cells:
                ids = self._grids[len(cell) - 1].get(cell)
                if ids:
                    seen.update(ids)
        return len(seen)

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def insert(
        self,
        point_id: str,
        latitude: float,
        longitude: float,
        attributes: Mapping[str, Any] | None = None,
    ) -> Point:
        """Insert a new point; raise ``DuplicatePointError`` if id exists."""
        point = self._make_point(point_id, latitude, longitude, attributes)
        with self._lock:
            if point.id in self._points:
                raise DuplicatePointError(f"point id already exists: {point.id!r}")
            if len(self._points) >= self._config.max_points:
                raise LimitExceededError(
                    f"index is full (max_points={self._config.max_points})"
                )
            self._points[point.id] = point
            self._index_point(point)
        return point

    def upsert(
        self,
        point_id: str,
        latitude: float,
        longitude: float,
        attributes: Mapping[str, Any] | None = None,
    ) -> Point:
        """Insert the point or deterministically replace the existing one."""
        point = self._make_point(point_id, latitude, longitude, attributes)
        with self._lock:
            old = self._points.get(point.id)
            if old is None:
                if len(self._points) >= self._config.max_points:
                    raise LimitExceededError(
                        f"index is full (max_points={self._config.max_points})"
                    )
            else:
                self._unindex_point(old)
            self._points[point.id] = point
            self._index_point(point)
        return point

    def update(
        self,
        point_id: str,
        latitude: float | None = None,
        longitude: float | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> Point:
        """Update fields of an existing point.

        ``None`` leaves a field unchanged; providing ``attributes`` replaces
        the whole attribute mapping.  Unknown ids raise
        ``PointNotFoundError``.
        """
        self._check_id(point_id)
        with self._lock:
            old = self._points.get(point_id)
            if old is None:
                raise PointNotFoundError(f"no such point id: {point_id!r}")
            new_lat = old.latitude if latitude is None else latitude
            new_lon = old.longitude if longitude is None else longitude
            new_attrs = old.attributes if attributes is None else attributes
            point = self._make_point(point_id, new_lat, new_lon, new_attrs)
            self._unindex_point(old)
            self._points[point_id] = point
            self._index_point(point)
        return point

    def delete(self, point_id: str) -> None:
        """Delete a point; raise ``PointNotFoundError`` for unknown ids."""
        self._check_id(point_id)
        with self._lock:
            point = self._points.pop(point_id, None)
            if point is None:
                raise PointNotFoundError(f"no such point id: {point_id!r}")
            self._unindex_point(point)

    def bulk_load(
        self,
        points: Iterable[Point],
        *,
        on_duplicate: str = "replace",
    ) -> int:
        """Atomically load many points; return the number of points written.

        ``on_duplicate`` is one of:

        * ``"replace"`` (default): later writes win, both within the batch
          and against points already stored;
        * ``"skip"``: earlier writes win (existing index points first, then
          the first occurrence in the batch);
        * ``"raise"``: any id that is duplicated within the batch or already
          stored raises ``DuplicatePointError`` and nothing is written.
        """
        if on_duplicate not in ("replace", "skip", "raise"):
            raise InvalidParameterError(
                "on_duplicate must be one of 'replace', 'skip', 'raise'"
            )
        # Validate/normalize the whole batch before touching the index.
        normalized: list[Point] = []
        for raw in points:
            if not isinstance(raw, Point):
                raise InvalidParameterError(
                    "bulk_load expects Point instances"
                )
            if not isinstance(raw.id, str) or not raw.id:
                raise InvalidParameterError(
                    "point id must be a non-empty string"
                )
            geo.validate_coordinates(raw.latitude, raw.longitude)
            if not isinstance(raw.attributes, dict):
                raise InvalidParameterError("point attributes must be a dict")
            normalized.append(
                Point(raw.id, raw.latitude, raw.longitude, dict(raw.attributes))
            )

        # Resolve within-batch duplicates deterministically.
        batch_order: list[str] = []
        batch_points: dict[str, Point] = {}
        for point in normalized:
            if point.id not in batch_points:
                batch_order.append(point.id)
                batch_points[point.id] = point
            elif on_duplicate == "raise":
                raise DuplicatePointError(
                    f"duplicate id within batch: {point.id!r}"
                )
            elif on_duplicate == "replace":
                batch_points[point.id] = point  # last write wins
            # "skip": first write wins -> keep existing entry

        # Capacity and conflict resolution all happen in one critical
        # section so the batch is atomic: either every applicable write
        # lands or none does.
        written = 0
        with self._lock:
            if on_duplicate == "raise":
                existing = {pid for pid in batch_order if pid in self._points}
                if existing:
                    raise DuplicatePointError(
                        f"batch ids already stored: {sorted(existing)!r}"
                    )
            replaced = sum(1 for pid in batch_order if pid in self._points)
            added = len(batch_order) - replaced
            if on_duplicate != "skip":
                projected_size = len(self._points) - replaced + added
            else:
                projected_size = len(self._points) + added
            if projected_size > self._config.max_points:
                raise LimitExceededError(
                    "bulk_load would exceed max_points="
                    f"{self._config.max_points}; nothing was written"
                )
            for pid in batch_order:
                incoming = batch_points[pid]
                old = self._points.get(pid)
                if old is not None:
                    if on_duplicate == "skip":
                        continue
                    self._unindex_point(old)
                self._points[pid] = incoming
                self._index_point(incoming)
                written += 1
        return written

    def get(self, point_id: str) -> Point | None:
        """Return the stored point or ``None``; attributes are copied."""
        with self._lock:
            point = self._points.get(point_id)
            if point is None:
                return None
            return Point(
                point.id, point.latitude, point.longitude, dict(point.attributes)
            )

    def contains(self, point_id: str) -> bool:
        self._check_id(point_id)
        with self._lock:
            return point_id in self._points

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def _validate_radius(self, radius_m: Any) -> float:
        if isinstance(radius_m, bool) or not isinstance(radius_m, (int, float)):
            raise InvalidParameterError("radius_m must be a non-negative number")
        radius = float(radius_m)
        if not math.isfinite(radius) or radius < 0:
            raise InvalidParameterError("radius_m must be a finite number >= 0")
        if radius > self._config.max_radius_m:
            raise LimitExceededError(
                f"radius_m {radius:g} exceeds max_radius_m="
                f"{self._config.max_radius_m:g}"
            )
        return radius

    def _rank(
        self, candidates: list[Point], latitude: float, longitude: float
    ) -> list[ScoredPoint]:
        scored = [
            ScoredPoint(
                Point(p.id, p.latitude, p.longitude, dict(p.attributes)),
                geo.haversine_distance_m(
                    latitude, longitude, p.latitude, p.longitude
                ),
            )
            for p in candidates
        ]
        # Deterministic order regardless of insertion order: distance, then
        # Unicode code-point ordering of the id.
        scored.sort(key=lambda sp: (sp.distance, sp.point.id))
        return scored

    def radius_query(
        self, latitude: float, longitude: float, radius_m: float
    ) -> list[ScoredPoint]:
        """Return all points with distance ``<= radius_m`` from the centre.

        Results are sorted by ``(distance, id)``.  The radius cap is
        :attr:`IndexConfig.max_radius_m`.
        """
        geo.validate_coordinates(latitude, longitude)
        radius = self._validate_radius(radius_m)
        cells = geo.cells_in_circle(
            latitude,
            longitude,
            radius,
            self._config.geohash_precision,
            self._config.max_scanned_cells,
        )
        candidates = self._snapshot_candidates(cells)
        scored = self._rank(candidates, latitude, longitude)
        return [
            sp
            for sp in scored
            if sp.distance <= radius + _DISTANCE_EPSILON_M
        ]

    def nearest(
        self, latitude: float, longitude: float, limit: int
    ) -> list[ScoredPoint]:
        """Return the *limit* nearest points sorted by ``(distance, id)``.

        ``limit`` must be an integer in ``[1, config.max_limit]``.  If fewer
        than *limit* points exist, every stored point is returned.  The
        result is never ``None`` and is empty only when the index is empty.

        Implementation: best-first traversal of the geohash prefix tree.
        Cells are popped from a min-heap keyed by their minimum possible
        distance to the query; once the next cell cannot improve on the
        current *limit*-th answer, traversal terminates.  Points are
        de-duplicated and snapshotted under a single lock acquisition.
        """
        geo.validate_coordinates(latitude, longitude)
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise InvalidParameterError("limit must be an integer")
        if limit < 1:
            raise InvalidParameterError("limit must be >= 1")
        if limit > self._config.max_limit:
            raise LimitExceededError(
                f"limit {limit} exceeds max_limit={self._config.max_limit}"
            )

        import bisect
        import heapq

        max_precision = self._config.geohash_precision
        heap: list[tuple[float, int, str]] = []
        # Sorted list of current best ``(distance, id)`` (ascending).
        top: list[tuple[float, str]] = []
        cells_visited = 0

        def _admit(d: float, pid: str) -> None:
            entry = (d, pid)
            if len(top) < limit:
                bisect.insort(top, entry)
            elif entry < top[-1]:
                top.pop()
                bisect.insort(top, entry)

        # The whole traversal and final id collection run inside ONE
        # critical section.  Splitting it across two lock acquisitions would
        # allow a bulk write between them to vacate already-selected roots,
        # producing an empty/stale answer.
        with self._lock:
            occupied_roots = self._grids[0]
            for root in geo.root_cells():
                if root not in occupied_roots:
                    continue
                d = geo.minimum_distance_to_cell_m(latitude, longitude, root)
                heapq.heappush(heap, (d, 0, root))

            while heap:
                mind, depth, cell = heapq.heappop(heap)
                cells_visited += 1
                if cells_visited > self._config.max_scanned_cells:
                    raise LimitExceededError(
                        "nearest query scanned more than "
                        f"{self._config.max_scanned_cells} geohash cells; "
                        "raise max_scanned_cells or reduce the population"
                    )
                if len(top) >= limit and mind > top[-1][0] + _DISTANCE_EPSILON_M:
                    break
                leaf_grid = self._grids[depth]
                ids = leaf_grid.get(cell)
                # Points live at every precision; collect each point only
                # once, at the deepest level.  Intermediate cells merely
                # guide the traversal downwards.
                if ids and depth == max_precision - 1:
                    for point_id in tuple(ids):
                        point = self._points[point_id]
                        d = geo.haversine_distance_m(
                            latitude,
                            longitude,
                            point.latitude,
                            point.longitude,
                        )
                        _admit(d, point_id)
                if depth + 1 < max_precision:
                    for child in geo.child_cells(cell):
                        # Skip subtrees containing zero points.
                        if child not in self._prefix_refcount:
                            continue
                        d_child = geo.minimum_distance_to_cell_m(
                            latitude, longitude, child
                        )
                        # Only prune by bound once we already have a full
                        # answer set; otherwise distant points are still
                        # candidates for the remaining slots.
                        if len(top) >= limit and d_child > top[-1][0] + _DISTANCE_EPSILON_M:
                            continue
                        heapq.heappush(heap, (d_child, depth + 1, child))

            chosen_ids = [pid for _, pid in top]
            snapshot = {
                pid: self._points[pid]
                for pid in chosen_ids
                if pid in self._points
            }

        result: list[ScoredPoint] = []
        for pid in chosen_ids:
            p = snapshot.get(pid)
            if p is None:
                continue
            d = geo.haversine_distance_m(
                latitude, longitude, p.latitude, p.longitude
            )
            result.append(
                ScoredPoint(
                    Point(p.id, p.latitude, p.longitude, dict(p.attributes)), d
                )
            )
        result.sort(key=lambda sp: (sp.distance, sp.point.id))
        return result
