"""Exceptions raised by the geohash proximity service."""
from __future__ import annotations


class ProximityError(Exception):
    """Base class for all service errors."""


class InvalidCoordinateError(ProximityError, ValueError):
    """Raised when a latitude/longitude value is outside its valid range."""


class InvalidParameterError(ProximityError, ValueError):
    """Raised when a query/write parameter is invalid (radius, limit, ...)."""


class DuplicatePointError(ProximityError):
    """Raised when inserting a point whose id already exists."""


class PointNotFoundError(ProximityError, KeyError):
    """Raised when updating/deleting an unknown point id."""


class LimitExceededError(ProximityError):
    """Raised when a configured size/cost cap would be exceeded."""
