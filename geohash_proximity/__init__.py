"""Local latitude/longitude proximity query service."""
from __future__ import annotations

from .config import IndexConfig
from .errors import (
    DuplicatePointError,
    InvalidCoordinateError,
    InvalidParameterError,
    LimitExceededError,
    PointNotFoundError,
    ProximityError,
)
from .index import ProximityIndex
from .models import Point, ScoredPoint

__all__ = [
    "DuplicatePointError",
    "IndexConfig",
    "InvalidCoordinateError",
    "InvalidParameterError",
    "LimitExceededError",
    "Point",
    "PointNotFoundError",
    "ProximityError",
    "ProximityIndex",
    "ScoredPoint",
]
