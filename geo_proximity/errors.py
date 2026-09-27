"""Exceptions raised by the geohash proximity service.

All exceptions derive from :class:`ProximityError` so callers can catch
service-level errors with a single ``except`` clause.
"""

from __future__ import annotations


class ProximityError(Exception):
    """Base class for all errors raised by this package."""


class InvalidCoordinatesError(ProximityError, ValueError):
    """Raised when a latitude/longitude value is missing or out of range."""


class InvalidParameterError(ProximityError, ValueError):
    """Raised when a query or write parameter has an illegal value."""


class DuplicateIdError(ProximityError):
    """Raised (in ``error`` mode) when an inserted id already exists."""


class PointNotFoundError(ProximityError, KeyError):
    """Raised when an operation references an unknown point id."""


class LimitExceededError(ProximityError):
    """Raised when a configured size or query limit would be exceeded.

    The service never silently truncates: either the operation is refused
    or the limit is raised explicitly by the caller.
    """
