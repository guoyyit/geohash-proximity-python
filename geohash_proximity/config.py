"""Configuration for the proximity index."""
from __future__ import annotations

from dataclasses import dataclass

from .errors import InvalidParameterError


@dataclass(frozen=True, slots=True)
class IndexConfig:
    """Tunable caps for the proximity index.

    Attributes:
        max_points: maximum number of stored points (the index memory cap).
            Inserting beyond this raises ``LimitExceededError``; the index
            never silently evicts data.
        max_radius_m: maximum allowed radius-query radius in metres.
        max_limit: maximum allowed ``nearest`` ``limit`` argument.
        max_scanned_cells: maximum number of geohash grid cells a single
            query may expand to.  When finer grids exceed the cap the index
            automatically uses a coarser grid; if even precision 1 exceeds
            the cap, ``LimitExceededError`` is raised rather than scanning
            the whole dataset.
        geohash_precision: finest geohash length used by the index (1..12).
            Points are indexed at every precision 1..P.
    """

    max_points: int = 1_000_000
    max_radius_m: float = 5_000_000.0
    max_limit: int = 10_000
    max_scanned_cells: int = 100_000
    geohash_precision: int = 9

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``InvalidParameterError`` if any value is out of range."""
        if not isinstance(self.max_points, int) or isinstance(self.max_points, bool):
            raise InvalidParameterError("max_points must be an integer")
        if self.max_points < 1:
            raise InvalidParameterError("max_points must be >= 1")
        if not isinstance(self.max_radius_m, (int, float)) or isinstance(
            self.max_radius_m, bool
        ):
            raise InvalidParameterError("max_radius_m must be a number")
        if not self.max_radius_m > 0:
            raise InvalidParameterError("max_radius_m must be > 0")
        if not isinstance(self.max_limit, int) or isinstance(self.max_limit, bool):
            raise InvalidParameterError("max_limit must be an integer")
        if self.max_limit < 1:
            raise InvalidParameterError("max_limit must be >= 1")
        if not isinstance(self.max_scanned_cells, int) or isinstance(
            self.max_scanned_cells, bool
        ):
            raise InvalidParameterError("max_scanned_cells must be an integer")
        if self.max_scanned_cells < 1:
            raise InvalidParameterError("max_scanned_cells must be >= 1")
        if not isinstance(self.geohash_precision, int) or isinstance(
            self.geohash_precision, bool
        ):
            raise InvalidParameterError("geohash_precision must be an integer")
        if not 1 <= self.geohash_precision <= 12:
            raise InvalidParameterError("geohash_precision must be in [1, 12]")
