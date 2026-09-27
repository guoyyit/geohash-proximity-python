"""Core data types for the geohash proximity service."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Point:
    """A geospatial point with a unique id and arbitrary attributes."""

    id: str
    latitude: float
    longitude: float
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScoredPoint:
    """A point paired with its distance (metres) from the query centre."""

    point: Point
    distance: float
