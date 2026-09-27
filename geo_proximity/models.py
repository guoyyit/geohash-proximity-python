"""Core data models for the proximity service.

Distance convention
---------------------
All distances are **great-circle (haversine) distances in metres** on a
sphere whose radius is :data:`~geo_proximity.geo.EARTH_RADIUS_M` (the mean
Earth radius, 6 371 008.8 m).  See :mod:`geo_proximity.geo`.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

from .errors import InvalidParameterError
from .geo import HALF_CIRCUMFERENCE_M

PointId = str
"""Unique point identifier. ``id`` must be a non-empty string."""

Attributes = dict[str, Any]
"""Free-form JSON-serialisable attributes attached to a point."""


class OnDuplicate(enum.Enum):
    """Behaviour when writing a point whose id already exists.

    * ``error``    -> raise :class:`~geo_proximity.errors.DuplicateIdError`.
    * ``replace``  -> overwrite coordinates/attributes (upsert).
    * ``skip``     -> keep the existing point and ignore the new one.
    """

    ERROR = "error"
    REPLACE = "replace"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class Point:
    """A geographic point: immutable value object."""

    id: PointId
    latitude: float
    longitude: float
    attributes: Attributes = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScoredPoint:
    """A point paired with its great-circle distance (metres) to a query."""

    point: Point
    distance_m: float


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    """Service limits.

    Every limit is explicit; exceeding one raises
    :class:`~geo_proximity.errors.LimitExceededError` rather than causing
    silent truncation.

    Attributes:
        max_points: Maximum number of points the store may hold.
        max_query_radius_m: Maximum radius accepted by radius queries.
        max_neighbors: Maximum ``limit`` accepted by nearest-neighbour
            queries.
        index_levels: Depth of the adaptive grid index in
            ``[1, 26]``; level ``l`` has ``2**l`` latitude bands.
            Default 20 (~19 m cells at the equator).
        max_batch_size: Maximum number of points in one bulk-load call.
    """

    max_points: int = 1_000_000
    max_query_radius_m: float = HALF_CIRCUMFERENCE_M
    max_neighbors: int = 10_000
    index_levels: int = 16
    max_batch_size: int = 100_000

    def __post_init__(self) -> None:
        def check_int(name: str, value: int, minimum: int) -> None:
            if isinstance(value, bool) or not isinstance(value, int):
                raise InvalidParameterError(f"{name} must be an integer")
            if value < minimum:
                raise InvalidParameterError(
                    f"{name} must be >= {minimum}, got {value}"
                )

        check_int("max_points", self.max_points, 1)
        check_int("max_neighbors", self.max_neighbors, 1)
        check_int("index_levels", self.index_levels, 1)
        check_int("max_batch_size", self.max_batch_size, 1)
        if not 1 <= self.index_levels <= 20:
            raise InvalidParameterError("index_levels must be in [1, 20]")
        radius = self.max_query_radius_m
        if isinstance(radius, bool) or not isinstance(radius, (int, float)):
            raise InvalidParameterError("max_query_radius_m must be a number")
        if not float(radius) > 0.0:
            raise InvalidParameterError("max_query_radius_m must be > 0")
        if self.max_query_radius_m > HALF_CIRCUMFERENCE_M + 1e-6:
            # Antipodal distance on the model sphere; anything larger
            # covers the whole globe.
            raise InvalidParameterError(
                "max_query_radius_m must be <= the antipodal distance "
                f"({HALF_CIRCUMFERENCE_M:.1f} m)"
            )
