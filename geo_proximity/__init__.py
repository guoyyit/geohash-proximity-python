"""Local in-memory lat/lng proximity query service.

Public API:

* :class:`~geo_proximity.models.Point`,
  :class:`~geo_proximity.models.ScoredPoint`,
  :class:`~geo_proximity.models.ServiceConfig`,
  :class:`~geo_proximity.models.OnDuplicate`
* :class:`~geo_proximity.service.ProximityService`
* error types in :mod:`geo_proximity.errors`
"""

from __future__ import annotations

from .errors import (
    DuplicateIdError,
    InvalidCoordinatesError,
    InvalidParameterError,
    LimitExceededError,
    PointNotFoundError,
    ProximityError,
)
from .models import OnDuplicate, Point, ScoredPoint, ServiceConfig
from .service import ProximityService

__all__ = [
    "OnDuplicate",
    "Point",
    "ProximityError",
    "ProximityService",
    "ScoredPoint",
    "ServiceConfig",
    "DuplicateIdError",
    "InvalidCoordinatesError",
    "InvalidParameterError",
    "LimitExceededError",
    "PointNotFoundError",
]

__version__ = "0.1.0"
