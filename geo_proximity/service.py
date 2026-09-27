"""Thread-safe in-memory proximity service.

Concurrency model
-----------------
All mutations hold an exclusive writer lock; all queries hold a shared
reader lock (writer-preferring, :class:`~geo_proximity.rwlock.RWLock`).
A query therefore observes a single immutable snapshot of the point set:
it never sees a half-applied update and a point that moves between cells
is never returned twice.

Write semantics
---------------
* ``insert`` honours an explicit :class:`~geo_proximity.models.OnDuplicate`
  policy (default ``error``); ``upsert`` is ``insert(replace)``.
* ``bulk_load`` is atomic: the whole batch is validated before any point
  is applied, so a limit/validation failure leaves the store untouched.
* Longitudes outside ``[-180, 180]`` are accepted and wrapped modulo 360
  (e.g. ``181`` becomes ``-179``); latitudes out of range are rejected.

Read semantics
--------------
* Radius queries return **every** point with great-circle distance
  ``<= radius`` (boundary inclusive).
* Nearest-neighbour queries return up to ``limit`` points sorted by
  ``(distance asc, id asc)``; the id tie-break makes ordering total and
  insertion-order independent.  Fewer results are returned when the store
  is exhausted (never an error); an illegal ``limit``/``radius`` is.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from .errors import (
    DuplicateIdError,
    InvalidCoordinatesError,
    InvalidParameterError,
    LimitExceededError,
    PointNotFoundError,
)
from .geo import (
    EARTH_RADIUS_M,
    HALF_CIRCUMFERENCE_M,
    haversine_m,
    validate_latlng,
)
from .gridindex import GridIndex
from .models import OnDuplicate, Point, ScoredPoint, ServiceConfig
from .rwlock import RWLock

_INITIAL_SEARCH_M = 1.0


def make_point(
    point_id: object,
    latitude: object,
    longitude: object,
    attributes: object = None,
) -> Point:
    """Validate raw fields and build a normalized :class:`Point`.

    Longitudes are wrapped into ``[-180, 180)``; invalid ids/coordinates
    raise the package's typed errors.
    """
    if not isinstance(point_id, str) or not point_id:
        raise InvalidParameterError("point id must be a non-empty string")
    if attributes is None:
        attrs: dict = {}
    elif isinstance(attributes, dict):
        attrs = dict(attributes)
    else:
        raise InvalidParameterError("attributes must be an object/dict")
    try:
        lat, lon = validate_latlng(latitude, longitude)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise InvalidCoordinatesError(str(exc)) from None
    return Point(id=point_id, latitude=lat, longitude=lon, attributes=attrs)


def coerce_point(value: Point | dict | tuple) -> Point:
    """Turn a Point / mapping / (id, lat, lon[, attrs]) into a Point."""
    if isinstance(value, Point):
        return make_point(value.id, value.latitude, value.longitude,
                          value.attributes)
    if isinstance(value, dict):
        try:
            return make_point(
                value["id"], value["latitude"], value["longitude"],
                value.get("attributes"),
            )
        except KeyError as exc:
            raise InvalidParameterError(f"missing field: {exc}") from None
    if isinstance(value, (tuple, list)):
        if len(value) == 3:
            pid, lat, lon = value  # type: ignore[misc]
            return make_point(pid, lat, lon)
        if len(value) == 4:
            pid, lat, lon, attrs = value  # type: ignore[misc]
            return make_point(pid, lat, lon, attrs)
    raise InvalidParameterError(
        "point must be a Point, mapping, or (id, lat, lon[, attributes])"
    )


class ProximityService:
    """Maintains points and answers radius / nearest-neighbour queries."""

    def __init__(self, config: ServiceConfig | None = None) -> None:
        self._config = config if config is not None else ServiceConfig()
        self._index = GridIndex(self._config.index_levels)
        self._lock = RWLock()

    # ---- introspection ------------------------------------------------
    @property
    def config(self) -> ServiceConfig:
        return self._config

    @property
    def index(self) -> GridIndex:
        return self._index

    def __len__(self) -> int:
        with self._lock.read_lock():
            return len(self._index)

    def __contains__(self, point_id: object) -> bool:
        with self._lock.read_lock():
            return point_id in self._index

    def snapshot_ids(self) -> Sequence[str]:
        """Return a sorted tuple of current point ids (tests/debug)."""
        with self._lock.read_lock():
            return tuple(sorted(p.id for p in self._index.all_points()))

    # ---- writes -------------------------------------------------------
    def _insert_validated(
        self, point: Point, on_duplicate: OnDuplicate
    ) -> bool:
        """Caller must hold the write lock."""
        existing = self._index.get_point(point.id)
        if existing is not None:
            if on_duplicate is OnDuplicate.ERROR:
                raise DuplicateIdError(
                    f"point id already exists: {point.id!r}"
                )
            if on_duplicate is OnDuplicate.SKIP:
                return False
            self._index.move(point.id, point)  # REPLACE
            return True
        if self._index.__len__() >= self._config.max_points:
            raise LimitExceededError(
                f"store is full (max_points={self._config.max_points})"
            )
        self._index.add(point)
        return True

    def insert(
        self,
        point: Point | dict | tuple,
        *,
        on_duplicate: OnDuplicate = OnDuplicate.ERROR,
    ) -> bool:
        """Insert one point; returns True if stored, False if skipped.

        Raises :class:`DuplicateIdError` in the default ``error`` mode.
        """
        p = coerce_point(point)
        with self._lock.write_lock():
            return self._insert_validated(p, on_duplicate)

    def upsert(self, point: Point | dict | tuple) -> None:
        """Insert or replace (no error on duplicate id)."""
        p = coerce_point(point)
        with self._lock.write_lock():
            self._insert_validated(p, OnDuplicate.REPLACE)

    def update(
        self,
        point_id: str,
        *,
        latitude: float | None = None,
        longitude: float | None = None,
        attributes: dict | None = None,
        merge_attributes: bool = False,
    ) -> Point:
        """Update coordinates and/or attributes of an existing point."""
        with self._lock.write_lock():
            old = self._index.get_point(point_id)
            if old is None:
                raise PointNotFoundError(
                    f"no such point id: {point_id!r}"
                )
            new_lat = old.latitude if latitude is None else latitude
            new_lon = old.longitude if longitude is None else longitude
            if attributes is None:
                new_attrs = dict(old.attributes)
            elif merge_attributes:
                new_attrs = {**old.attributes, **attributes}
            else:
                new_attrs = dict(attributes)
            new_point = make_point(point_id, new_lat, new_lon, new_attrs)
            self._index.move(point_id, new_point)
            return new_point

    def delete(self, point_id: str) -> bool:
        """Delete a point; returns whether it existed."""
        with self._lock.write_lock():
            if point_id not in self._index:
                return False
            self._index.remove(point_id)
            return True

    def bulk_load(
        self,
        points: Iterable[Point | dict | tuple],
        *,
        on_duplicate: OnDuplicate = OnDuplicate.ERROR,
    ) -> int:
        """Atomically load many points; returns number actually stored.

        The iterable is fully validated first (coordinates, batch size);
        only then is the store mutated under one write lock, so an invalid
        batch leaves the store untouched.  Duplicate ids *inside* the batch
        and ids already present are both resolved by ``on_duplicate``.
        """
        validated: list[Point] = []
        for raw in points:
            validated.append(coerce_point(raw))
            if len(validated) > self._config.max_batch_size:
                raise LimitExceededError(
                    f"batch exceeds max_batch_size="
                    f"{self._config.max_batch_size}"
                )

        with self._lock.write_lock():
            # Collapse repeated ids within the batch (last write wins) but
            # remember which ids were repeated for policy enforcement.
            by_id: dict[str, Point] = {}
            repeated_in_batch: set[str] = set()
            for p in validated:
                if p.id in by_id:
                    repeated_in_batch.add(p.id)
                by_id[p.id] = p

            def policy_for(pid: str) -> str:
                exists = self._index.get_point(pid) is not None
                if not exists:
                    # Repeated ids within the batch collapse to one add
                    # (the last write); that is never a store-level
                    # duplicate.
                    return "add"
                if on_duplicate is OnDuplicate.ERROR:
                    return "error"
                if on_duplicate is OnDuplicate.SKIP:
                    return "skip"
                return "replace"

            actions = {pid: policy_for(pid) for pid in by_id}
            # ERROR mode still rejects ids repeated *within* the batch even
            # though they are not yet present in the store.
            if on_duplicate is OnDuplicate.ERROR and repeated_in_batch:
                bad = next(iter(sorted(repeated_in_batch)))
                raise DuplicateIdError(
                    f"duplicate id within batch: {bad!r}"
                )
            if "error" in actions.values():
                bad = next(pid for pid, a in actions.items() if a == "error")
                raise DuplicateIdError(
                    f"point id already exists: {bad!r}"
                )
            new_count = sum(1 for a in actions.values() if a == "add")
            # Replaced/skipped existing ids do not change the store size.
            if len(self._index) + new_count > self._config.max_points:
                raise LimitExceededError(
                    f"bulk load would exceed max_points="
                    f"{self._config.max_points} ({new_count} new points, "
                    f"{len(self._index)} already stored)"
                )

            stored = 0
            for pid in sorted(by_id):  # deterministic application order
                action = actions[pid]
                point = by_id[pid]
                if action == "skip":
                    continue
                if action == "replace":
                    self._index.move(pid, point)
                else:
                    self._index.add(point)
                stored += 1
            return stored

    def get(self, point_id: str) -> Point | None:
        with self._lock.read_lock():
            return self._index.get_point(point_id)

    # ---- query parameter validation -----------------------------------
    def _validate_center(self, latitude: object, longitude: object):
        try:
            return validate_latlng(latitude, longitude)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise InvalidCoordinatesError(str(exc)) from None

    def _validate_radius(self, radius_m: object) -> float:
        if isinstance(radius_m, bool) or not isinstance(
            radius_m, (int, float)
        ):
            raise InvalidParameterError("radius_m must be a number")
        radius_m = float(radius_m)
        if not math.isfinite(radius_m) or radius_m < 0.0:
            raise InvalidParameterError(
                "radius_m must be a finite, non-negative number"
            )
        if radius_m > self._config.max_query_radius_m:
            raise LimitExceededError(
                f"radius_m {radius_m} exceeds configured "
                f"max_query_radius_m {self._config.max_query_radius_m}"
            )
        return radius_m

    def _validate_limit(self, limit: object) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise InvalidParameterError("limit must be an integer")
        if limit < 1:
            raise InvalidParameterError("limit must be >= 1")
        if limit > self._config.max_neighbors:
            raise LimitExceededError(
                f"limit {limit} exceeds configured max_neighbors "
                f"{self._config.max_neighbors}"
            )
        return limit

    # ---- reads --------------------------------------------------------
    @staticmethod
    def _score(point: Point, lat: float, lon: float) -> ScoredPoint:
        return ScoredPoint(
            point=point,
            distance_m=haversine_m(lat, lon, point.latitude, point.longitude),
        )

    def radius_query(
        self,
        latitude: float,
        longitude: float,
        radius_m: float,
        *,
        limit: int | None = None,
    ) -> list[ScoredPoint]:
        """All points with distance <= radius_m (inclusive).

        Results are sorted by ``(distance_m, point.id)``.  ``limit`` is an
        optional hard cap (1..``max_neighbors``); the store itself is the
        natural cap, so prefer setting it only when required.
        """
        lat, lon = self._validate_center(latitude, longitude)
        radius = self._validate_radius(radius_m)
        cap = limit if limit is not None else None
        if cap is not None:
            self._validate_limit(cap)
        with self._lock.read_lock():
            results: list[ScoredPoint] = []
            for point in self._index.candidates(lat, lon, radius):
                scored = self._score(point, lat, lon)
                if scored.distance_m <= radius + 1e-9:
                    results.append(scored)
            results.sort(key=lambda s: (s.distance_m, s.point.id))
            if cap is not None and len(results) > cap:
                raise LimitExceededError(
                    f"radius query matched {len(results)} points but limit="
                    f"{cap}; raise the limit or narrow the radius"
                )
            return results

    def nearest(
        self,
        latitude: float,
        longitude: float,
        limit: int,
        *,
        max_distance_m: float | None = None,
    ) -> list[ScoredPoint]:
        """Up to ``limit`` nearest points, ascending distance.

        Ties break by point id ascending.  Fewer than ``limit`` results
        are returned when the store is exhausted.  ``max_distance_m``
        optionally bounds the search (same configuration cap as radius).
        """
        lat, lon = self._validate_center(latitude, longitude)
        k = self._validate_limit(limit)
        cap_m = HALF_CIRCUMFERENCE_M
        if max_distance_m is not None:
            cap_m = self._validate_radius(max_distance_m)

        with self._lock.read_lock():
            total = len(self._index)
            if total == 0:
                return []
            wanted = min(k, total)
            found: dict[str, ScoredPoint] = {}
            search_m = min(
                cap_m,
                max(_INITIAL_SEARCH_M, self._finest_edge_m(lat)),
            )
            while True:
                for point in self._index.candidates(lat, lon, search_m):
                    if point.id in found:
                        continue
                    scored = self._score(point, lat, lon)
                    if scored.distance_m <= cap_m + 1e-9:
                        found[point.id] = scored
                # Termination: either the whole search domain has been
                # scanned, or at least ``wanted`` known points lie strictly
                # inside the searched cap.  Any point the index has not yet
                # offered is outside the cap and therefore farther than
                # those points, so the answer can no longer change.
                if search_m >= cap_m:
                    break
                inside = sum(
                    1 for s in found.values()
                    if s.distance_m < search_m - 1e-9
                )
                if inside >= wanted:
                    break
                search_m = min(cap_m, search_m * 2.0)
            ranked = sorted(
                found.values(), key=lambda s: (s.distance_m, s.point.id)
            )
            return ranked[:wanted]

    def _finest_edge_m(self, latitude: float) -> float:
        """Finest-level cell edge (metres) at the query latitude.

        Used as the initial nearest-neighbour search radius: it guarantees
        the first expansion already covers the containing cell plus its
        ring when points are dense and very close.
        """
        edge = (math.pi / 4.0) * EARTH_RADIUS_M / (
            2 ** (self._config.index_levels - 1)
        )
        # Longitude edges shrink away from the equator; use the smaller
        # (longitude) edge so the initial cap is not under-sized.
        return max(1.0, edge * max(0.0, math.cos(math.radians(latitude))))
