"""Geographic helpers: validation, great-circle distance, geohash grid.

Distance convention
-------------------
Distances are great-circle distances on a sphere with mean radius
:data:`EARTH_RADIUS_METRES` (6 371 000 m), computed with the haversine
formula.  Results are in metres and are frozen by the test-suite, so the
Earth radius must not change without updating the tests.

Geohash convention
------------------
Standard Niemeyer base32 geohashing using the alphabet
``0123456789bcdefghjkmnpqrstuvwxyz``.  Cells are laid out on a regular
lat/lon grid; no special-cell shapes are assumed, so the antimeridian and
the poles are handled by plain integer index arithmetic (longitude wraps,
latitude is clamped to the polar cells).
"""
from __future__ import annotations

import math
from collections.abc import Iterable

from .errors import InvalidCoordinateError, InvalidParameterError, LimitExceededError

EARTH_RADIUS_METRES = 6_371_000.0
MAX_LATITUDE = 90.0
MAX_LONGITUDE = 180.0
MAX_GEOHASH_PRECISION = 12
_GEOHASH_ALPHABET = "0123456789bcdefghjkmnpqrstuvwxyz"
_CHAR_TO_VALUE = {ch: i for i, ch in enumerate(_GEOHASH_ALPHABET)}


def validate_coordinates(latitude: float, longitude: float) -> None:
    """Raise ``InvalidCoordinateError`` if *latitude* / *longitude* are invalid.

    Values must be real, finite numbers with ``-90 <= lat <= 90`` and
    ``-180 <= lon <= 180``.  Booleans (which are technically ``int``) are
    rejected to catch common call-site mistakes.
    """
    if isinstance(latitude, bool) or not isinstance(latitude, (int, float)):
        raise InvalidCoordinateError(
            f"latitude must be a real number, got {type(latitude).__name__}"
        )
    if isinstance(longitude, bool) or not isinstance(longitude, (int, float)):
        raise InvalidCoordinateError(
            f"longitude must be a real number, got {type(longitude).__name__}"
        )
    lat = float(latitude)
    lon = float(longitude)
    if not math.isfinite(lat) or not math.isfinite(lon):
        raise InvalidCoordinateError("latitude/longitude must be finite")
    if lat < -MAX_LATITUDE or lat > MAX_LATITUDE:
        raise InvalidCoordinateError(
            f"latitude {lat} out of range [-{MAX_LATITUDE}, {MAX_LATITUDE}]"
        )
    if lon < -MAX_LONGITUDE or lon > MAX_LONGITUDE:
        raise InvalidCoordinateError(
            f"longitude {lon} out of range [-{MAX_LONGITUDE}, {MAX_LONGITUDE}]"
        )


def haversine_distance_m(
    latitude1: float, longitude1: float, latitude2: float, longitude2: float
) -> float:
    """Great-circle distance in metres.

    Uses the haversine formula on a sphere of radius
    :data:`EARTH_RADIUS_METRES`.  The distance is symmetric, non-negative
    and order independent.
    """
    phi1 = math.radians(latitude1)
    phi2 = math.radians(latitude2)
    dphi = math.radians(latitude2 - latitude1)
    dlmb = math.radians(longitude2 - longitude1)
    sin_dphi = math.sin(dphi / 2.0)
    sin_dlmb = math.sin(dlmb / 2.0)
    a = sin_dphi * sin_dphi + math.cos(phi1) * math.cos(phi2) * sin_dlmb * sin_dlmb
    # Clamp to [0, 1] to avoid NaN from floating-point overshoot.
    c = 2.0 * math.asin(min(1.0, math.sqrt(a)))
    return EARTH_RADIUS_METRES * c


def _grid_dimensions(precision: int) -> tuple[int, int, int]:
    """Return ``(total_bits, longitude_columns, latitude_rows)``."""
    if not isinstance(precision, int) or isinstance(precision, bool):
        raise InvalidParameterError("geohash precision must be an integer")
    if precision < 1 or precision > MAX_GEOHASH_PRECISION:
        raise InvalidParameterError(
            f"geohash precision must be in [1, {MAX_GEOHASH_PRECISION}]"
        )
    total_bits = 5 * precision
    lon_bits = (total_bits + 1) // 2
    lat_bits = total_bits // 2
    return total_bits, 1 << lon_bits, 1 << lat_bits


def encode_geohash(latitude: float, longitude: float, precision: int) -> str:
    """Encode lat/lon into a base32 geohash of *precision* characters."""
    validate_coordinates(latitude, longitude)
    total_bits, lon_cols, lat_rows = _grid_dimensions(precision)
    # Minus epsilon-free: the 180/90 boundaries clamp into the last cell.
    lon_col = math.floor((longitude + MAX_LONGITUDE) / 360.0 * lon_cols)
    if lon_col == lon_cols:
        lon_col = lon_cols - 1
    lat_row = math.floor((latitude + MAX_LATITUDE) / 180.0 * lat_rows)
    if lat_row == lat_rows:
        lat_row = lat_rows - 1
    lon_bits = (total_bits + 1) // 2
    lat_bits = total_bits // 2
    chars: list[str] = []
    accumulated = 0
    for i in range(total_bits):
        if i % 2 == 0:
            bit = (lon_col >> (lon_bits - 1 - i // 2)) & 1
        else:
            bit = (lat_row >> (lat_bits - 1 - (i - 1) // 2)) & 1
        accumulated = (accumulated << 1) | bit
        if (i + 1) % 5 == 0:
            chars.append(_GEOHASH_ALPHABET[accumulated])
            accumulated = 0
    return "".join(chars)


def decode_geohash(cell: str) -> tuple[float, float, float, float, float, float]:
    """Return ``(lat_south, lat_north, lon_west, lon_east, clat, clon)``."""
    if not isinstance(cell, str) or not cell:
        raise InvalidParameterError("geohash cell must be a non-empty string")
    precision = len(cell)
    try:
        values = [_CHAR_TO_VALUE[ch] for ch in cell]
    except KeyError as exc:
        raise InvalidParameterError(f"invalid geohash character: {exc.args[0]!r}")
    _, lon_cols, lat_rows = _grid_dimensions(precision)
    lon_col = 0
    lat_row = 0
    for i, value in enumerate(values):
        for j in range(5):
            bit = (value >> (4 - j)) & 1
            bit_index = i * 5 + j
            if bit_index % 2 == 0:
                lon_col = (lon_col << 1) | bit
            else:
                lat_row = (lat_row << 1) | bit
    lat_height = 180.0 / lat_rows
    lon_width = 360.0 / lon_cols
    lat_south = -MAX_LATITUDE + lat_row * lat_height
    lat_north = lat_south + lat_height
    lon_west = -MAX_LONGITUDE + lon_col * lon_width
    lon_east = lon_west + lon_width
    return (
        lat_south,
        lat_north,
        lon_west,
        lon_east,
        lat_south + lat_height / 2.0,
        lon_west + lon_width / 2.0,
    )


def circle_bbox(
    latitude: float, longitude: float, radius_m: float
) -> tuple[float, float, float, float, bool]:
    """Bounding lat/lon box of a spherical disc.

    Returns ``(min_lat, max_lat, min_lon, max_lon, lon_full)`` where
    *lon_full* is True when the disc touches a pole and therefore spans
    every longitude.  Longitudes are kept in ``[-180, 180]``; the box is a
    *conservative* superset of the disc (exact filtering happens with the
    haversine formula later).
    """
    r_ang = radius_m / EARTH_RADIUS_METRES
    min_lat = max(-MAX_LATITUDE, latitude - math.degrees(r_ang))
    max_lat = min(MAX_LATITUDE, latitude + math.degrees(r_ang))
    if r_ang >= math.pi / 2.0 - math.radians(abs(latitude)):
        # Disc reaches (or crosses) a pole: it covers every longitude.
        # Threshold is exact for |lat| > 0 and harmless for lat == 0.
        return min_lat, max_lat, -MAX_LONGITUDE, MAX_LONGITUDE, True
    # Non-polar disc: cos(lat) > sin(r) so the asin argument is < 1.
    sin_r = math.sin(r_ang)
    cos_lat = math.cos(math.radians(latitude))
    dlon = math.degrees(math.asin(min(1.0, sin_r / cos_lat)))
    # Longitudes are intentionally left unclamped: callers enumerate grid
    # columns with modular arithmetic, which handles antimeridian wrapping.
    return min_lat, max_lat, longitude - dlon, longitude + dlon, False


def bbox_cell_count(
    min_lat: float, max_lat: float, min_lon: float, max_lon: float,
    lon_full: bool, precision: int,
) -> int:
    """Number of geohash cells at *precision* intersecting the bbox."""
    _, lon_cols, lat_rows = _grid_dimensions(precision)
    lat_height = 180.0 / lat_rows
    lon_width = 360.0 / lon_cols
    row_lo = min(lat_rows - 1, max(0, math.floor((min_lat + 90.0) / lat_height)))
    row_hi = min(lat_rows - 1, max(0, math.floor((max_lat + 90.0) / lat_height)))
    n_rows = row_hi - row_lo + 1
    if lon_full:
        n_cols = lon_cols
    else:
        # Raw columns may be negative or >= lon_cols when the bbox crosses
        # the antimeridian; measure with modular arithmetic and cap at the
        # full row.
        col_lo = math.floor((min_lon + 180.0) / lon_width) % lon_cols
        col_hi_raw = math.floor((max_lon + 180.0) / lon_width) % lon_cols
        n_cols = (col_hi_raw - col_lo) % lon_cols + 1
        n_cols = min(lon_cols, n_cols)
    return n_rows * n_cols


def choose_precision(
    min_lat: float, max_lat: float, min_lon: float, max_lon: float,
    lon_full: bool, max_precision: int, max_cells: int,
) -> int:
    """Pick the finest geohash precision whose expansion stays under cap."""
    for precision in range(max_precision, 0, -1):
        if (
            bbox_cell_count(
                min_lat, max_lat, min_lon, max_lon, lon_full, precision
            )
            <= max_cells
        ):
            return precision
    raise LimitExceededError(
        f"query would scan more than {max_cells} geohash cells even at "
        "precision 1; reduce the radius or raise max_scanned_cells"
    )


def _cell_from_indices(lat_row: int, lon_col: int, precision: int) -> str:
    total_bits, _lon_cols, _lat_rows = _grid_dimensions(precision)
    lon_bits = (total_bits + 1) // 2
    lat_bits = total_bits // 2
    chars: list[str] = []
    accumulated = 0
    for i in range(total_bits):
        if i % 2 == 0:
            bit = (lon_col >> (lon_bits - 1 - i // 2)) & 1
        else:
            bit = (lat_row >> (lat_bits - 1 - (i - 1) // 2)) & 1
        accumulated = (accumulated << 1) | bit
        if (i + 1) % 5 == 0:
            chars.append(_GEOHASH_ALPHABET[accumulated])
            accumulated = 0
    return "".join(chars)


def enumerate_cells(
    min_lat: float, max_lat: float, min_lon: float, max_lon: float,
    lon_full: bool, precision: int,
) -> Iterable[str]:
    """Yield every geohash cell at *precision* intersecting the bbox.

    Each cell is yielded exactly once regardless of antimeridian wrapping
    or pole coverage.
    """
    _, lon_cols, lat_rows = _grid_dimensions(precision)
    lat_height = 180.0 / lat_rows
    lon_width = 360.0 / lon_cols
    row_lo = min(lat_rows - 1, max(0, math.floor((min_lat + 90.0) / lat_height)))
    row_hi = min(lat_rows - 1, max(0, math.floor((max_lat + 90.0) / lat_height)))
    if lon_full:
        cols = range(lon_cols)
    else:
        col_lo = math.floor((min_lon + 180.0) / lon_width) % lon_cols
        col_hi_raw = math.floor((max_lon + 180.0) / lon_width) % lon_cols
        n_cols = (col_hi_raw - col_lo) % lon_cols + 1
        if n_cols >= lon_cols:
            cols = range(lon_cols)
        else:
            cols = ((col_lo + k) % lon_cols for k in range(n_cols))
    for lon_col in cols:
        for lat_row in range(row_lo, row_hi + 1):
            yield _cell_from_indices(lat_row, lon_col, precision)


def cells_in_circle(
    latitude: float,
    longitude: float,
    radius_m: float,
    max_precision: int,
    max_cells: int,
) -> frozenset[str]:
    """Set of geohash cells whose union covers the query disc.

    The set is a (small) conservative superset; callers must filter
    candidate points with :func:`haversine_distance_m`.  Raises
    ``LimitExceededError`` if even precision 1 would exceed *max_cells*.
    """
    min_lat, max_lat, min_lon, max_lon, lon_full = circle_bbox(
        latitude, longitude, radius_m
    )
    precision = choose_precision(
        min_lat, max_lat, min_lon, max_lon, lon_full, max_precision, max_cells
    )
    return frozenset(
        enumerate_cells(
            min_lat, max_lat, min_lon, max_lon, lon_full, precision
        )
    )


def child_cells(cell: str) -> tuple[str, ...]:
    """The 32 direct geohash sub-cells of *cell* (one character longer)."""
    return tuple(cell + ch for ch in _GEOHASH_ALPHABET)


def minimum_distance_to_cell_m(
    latitude: float, longitude: float, cell: str
) -> float:
    """Minimum great-circle distance from a point to a cell's lat/lon box.

    The nearest point of the box is obtained by independently clamping
    latitude to the cell's latitude band and longitude to the nearest arc
    of the cell's longitude band (antimeridian aware).  The haversine is
    monotonic in both angular separations, so this is a true lower bound on
    the distance to any point in the cell.
    """
    south, north, west, east, _, _ = decode_geohash(cell)
    nearest_lat = min(max(latitude, south), north)
    lon_width = east - west
    # Signed shortest-arc offset east of the cell's west edge, (-180, 180].
    offset = (longitude - west + 180.0) % 360.0 - 180.0
    if offset < 0.0:
        nearest_lon = west
    elif offset > lon_width:
        nearest_lon = east
    else:
        nearest_lon = longitude
    return haversine_distance_m(
        latitude, longitude, nearest_lat, nearest_lon
    )


def root_cells() -> tuple[str, ...]:
    """All precision-1 geohash cells (32), covering the globe once."""
    return tuple(_GEOHASH_ALPHABET)
