"""Adaptive hierarchical lat/lng grid index.

Grid layout
-----------
At level ``l`` (``1 <= l <= L``):

* latitude is split into ``2**l`` bands of ``180 / 2**l`` degrees;
* longitude is split into ``2**(l+1)`` bands of ``360 / 2**(l+1)``
  degrees, so cells are approximately square at the equator (equatorial
  edge at level 1 is 1/8 of a great circle, ~5 004 km).

Every point is stored once in a cell **at every level** (a point in the
finest cell is implicitly in exactly one coarser cell per level, so there
is no duplication of *results*; only the membership sets are stored per
level).  A query selects the single level whose cell edge is closest to
the query radius and reads only cells at that level.  The number of
inspected cells is therefore roughly ``O((r / edge)^2)`` -- a small
constant -- instead of proportional to the store size.

Correctness near geographic edges
----------------------------------
* Longitude bins wrap modulo the band count; intervals are computed in
  *unwrapped* coordinates and wrapped afterwards, so a cap crossing the
  antimeridian reads cells on both sides with no duplicates.
* Latitude bins clamp to the polar bands; a cap containing a pole scans
  full longitude rings for the covered polar parallels
  (:func:`geo_proximity.geo.ring_threshold_latitudes`).
* All cells are held in sets, so results never depend on insertion order.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

from .geo import (
    EARTH_RADIUS_M,
    HALF_CIRCUMFERENCE_M,
    max_longitude_extent,
    ring_threshold_latitudes,
    spherical_cap_bbox,
)

if TYPE_CHECKING:
    from .models import Point

CellBucket = dict[tuple[int, int], set[str]]
"""Cell key ``(lat_bin, lon_bin)`` at one level -> point id set."""

_LEVEL1_EDGE_M = (math.pi / 4.0) * EARTH_RADIUS_M
"""Equatorial cell edge at level 1 (~5 004 km)."""


class GridIndex:
    """In-memory multi-level hierarchical grid over points."""

    __slots__ = ("_levels", "_lat_bins", "_lon_bins", "_cells", "_points", "_size")

    def __init__(self, levels: int) -> None:
        if isinstance(levels, bool) or not isinstance(levels, int):
            raise TypeError("levels must be an integer")
        if not 1 <= levels <= 20:
            raise ValueError("levels must be in [1, 20]")
        self._levels = levels
        self._lat_bins = 1 << levels
        self._lon_bins = 1 << (levels + 1)
        # One cell dictionary per level (index 0 unused; level l at l-1).
        self._cells: list[CellBucket] = [{} for _ in range(levels)]
        self._points: dict[str, Point] = {}
        self._size = 0

    # ---- introspection ------------------------------------------------
    @property
    def levels(self) -> int:
        return self._levels

    def __len__(self) -> int:
        return self._size

    def __contains__(self, point_id: object) -> bool:
        return point_id in self._points

    def get_point(self, point_id: str) -> Point | None:
        return self._points.get(point_id)

    def all_points(self) -> Iterable[Point]:
        return self._points.values()

    def _bin(self, latitude: float, longitude: float, level: int) -> tuple[int, int]:
        """Cell key at ``level`` for a validated coordinate."""
        lat_bins = 1 << level
        lon_bins = 1 << (level + 1)
        lat_bin = int(math.floor((latitude + 90.0) * lat_bins / 180.0))
        lon_bin = int(math.floor((longitude + 180.0) * lon_bins / 360.0))
        if lat_bin >= lat_bins:
            lat_bin = lat_bins - 1
        if lat_bin < 0:
            lat_bin = 0
        if lon_bin >= lon_bins:
            lon_bin = lon_bins - 1
        if lon_bin < 0:
            lon_bin = 0
        return lat_bin, lon_bin

    def cell_of(self, latitude: float, longitude: float) -> tuple[int, int]:
        """Finest-level cell containing a (validated) coordinate."""
        return self._bin(latitude, longitude, self._levels)

    # ---- mutation -----------------------------------------------------
    def add(self, point: Point) -> None:
        """Insert a point whose id must not already be present."""
        if point.id in self._points:
            raise KeyError(f"point id already indexed: {point.id!r}")
        for level in range(1, self._levels + 1):
            key = self._bin(point.latitude, point.longitude, level)
            self._cells[level - 1].setdefault(key, set()).add(point.id)
        self._points[point.id] = point
        self._size += 1

    def remove(self, point_id: str) -> Point:
        """Remove and return the point with ``point_id``."""
        point = self._points.pop(point_id)
        for level in range(1, self._levels + 1):
            key = self._bin(point.latitude, point.longitude, level)
            bucket = self._cells[level - 1][key]
            bucket.remove(point_id)
            if not bucket:
                del self._cells[level - 1][key]
        self._size -= 1
        return point

    def move(self, point_id: str, new_point: Point) -> None:
        """Replace an existing point, touching only levels whose cell moved."""
        old = self._points[point_id]
        for level in range(1, self._levels + 1):
            old_key = self._bin(old.latitude, old.longitude, level)
            new_key = self._bin(new_point.latitude, new_point.longitude, level)
            if old_key != new_key:
                cells = self._cells[level - 1]
                bucket = cells[old_key]
                bucket.remove(point_id)
                if not bucket:
                    del cells[old_key]
                cells.setdefault(new_key, set()).add(point_id)
        self._points[point_id] = new_point

    # ---- query planning ----------------------------------------------
    def choose_level(self, radius_m: float) -> int:
        """Level whose equatorial cell edge is closest to ``radius_m``.

        The chosen edge is the smallest one still >= the radius (so a
        radius-0 / exact query uses the finest level and inspects a few
        neighbouring cells), clamped to ``[1, L]``.
        """
        if radius_m <= 0.0:
            return self._levels
        ratio = _LEVEL1_EDGE_M / radius_m
        if ratio <= 1.0:
            return 1
        level = 1 + int(math.ceil(math.log2(ratio)))
        return min(self._levels, max(1, level))

    def cells_for_cap(
        self,
        latitude: float,
        longitude: float,
        radius_m: float,
    ) -> list[tuple[int, int]]:
        """Cell keys at the chosen level that may overlap the cap."""
        level = self.choose_level(radius_m)
        bands_lat = 1 << level
        bands_lon = 1 << (level + 1)
        h_lat = 180.0 / bands_lat
        h_lon = 360.0 / bands_lon

        min_lat, max_lat, _, _ = spherical_cap_bbox(
            latitude, longitude, radius_m
        )
        north_ring, south_ring = ring_threshold_latitudes(
            latitude, radius_m
        )

        def lat_band(value_deg: float) -> int:
            idx = int(math.floor((value_deg + 90.0) / h_lat - 1e-12))
            return min(bands_lat - 1, max(0, idx))

        def lon_band_unwrapped(value_deg: float) -> int:
            # Valid for arbitrary unwrapped longitudes; wrap modulo
            # bands_lon where the key is produced.
            return int(math.floor((value_deg + 180.0) / h_lon + 1e-12))

        lo = lat_band(min_lat)
        hi = lat_band(max_lat)

        seen: set[tuple[int, int]] = set()

        if radius_m == 0.0:
            lat_b = int(math.floor((latitude + 90.0) / h_lat - 1e-12))
            lat_b = min(bands_lat - 1, max(0, lat_b))
            lon_b = int(math.floor((longitude + 180.0) / h_lon - 1e-12))
            lon_b = min(bands_lon - 1, max(0, lon_b))
            if abs(abs(latitude) - 90.0) < 1e-12:
                # Exactly on a pole longitude is meaningless: every point
                # at that pole is distance 0 regardless of longitude, so
                # scan the whole polar band.
                for lb in range(bands_lon):
                    seen.add((lat_b, lb))
            else:
                # Containing cell plus its 8 neighbours so a coincident
                # point on a shared edge is never lost; the exact distance
                # test discards non-coincident neighbours.
                for dla in (-1, 0, 1):
                    la_n = lat_b + dla
                    if 0 <= la_n < bands_lat:
                        for dlo in (-1, 0, 1):
                            seen.add((la_n, (lon_b + dlo) % bands_lon))
            return list(seen)

        def emit(lat_band_idx: int, start_lon: float, end_lon: float) -> None:
            start = lon_band_unwrapped(start_lon)
            end = lon_band_unwrapped(end_lon)
            for b in range(start, end + 1):
                seen.add((lat_band_idx, b % bands_lon))

        for la in range(lo, hi + 1):
            band_lo = -90.0 + la * h_lat
            band_hi = min(90.0, band_lo + h_lat)
            finite_lo, finite_hi = band_lo, band_hi
            if south_ring is not None and band_lo <= south_ring + 1e-12:
                emit(la, longitude - 180.0, longitude + 180.0)
                finite_lo = max(finite_lo, south_ring)
            if north_ring is not None and band_hi >= north_ring - 1e-12:
                emit(la, longitude - 180.0, longitude + 180.0)
                finite_hi = min(finite_hi, north_ring)
            if finite_hi >= finite_lo - 1e-12:
                half = max_longitude_extent(
                    latitude,
                    radius_m,
                    max(band_lo, finite_lo),
                    min(band_hi, finite_hi),
                )
                if half is not None and half > 0.0:
                    emit(la, longitude - half, longitude + half)
        return list(seen)

    def candidates(
        self,
        latitude: float,
        longitude: float,
        radius_m: float,
    ) -> Iterator[Point]:
        """Yield each point in cap-overlapping cells at most once."""
        level = self.choose_level(radius_m)
        cells = self._cells[level - 1]
        for key in self.cells_for_cap(latitude, longitude, radius_m):
            bucket = cells.get(key)
            if bucket is not None:
                for pid in bucket:
                    yield self._points[pid]

    # ---- diagnostics --------------------------------------------------
    def cell_counts(self, level: int | None = None) -> list[int]:
        """Sorted cell population sizes at one level (finest by default)."""
        lv = self._levels if level is None else level
        return sorted(len(v) for v in self._cells[lv - 1].values())

    def global_radius(self) -> float:
        """Distance guaranteed to cover every point on Earth."""
        return HALF_CIRCUMFERENCE_M
