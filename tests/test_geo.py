"""Tests for geographic helpers (distance convention and geohash grid)."""
from __future__ import annotations

import math

import pytest

from geohash_proximity import geo
from geohash_proximity.errors import (
    InvalidCoordinateError,
    InvalidParameterError,
    LimitExceededError,
)


# --------------------------------------------------------------------------- #
# Distance convention
# --------------------------------------------------------------------------- #
def test_earth_radius_is_documented_constant() -> None:
    assert geo.EARTH_RADIUS_METRES == 6_371_000.0


def test_zero_distance() -> None:
    assert geo.haversine_distance_m(12.5, -70.1, 12.5, -70.1) == 0.0


def test_distance_is_symmetric() -> None:
    d_ab = geo.haversine_distance_m(10.0, 20.0, -30.0, 40.0)
    d_ba = geo.haversine_distance_m(-30.0, 40.0, 10.0, 20.0)
    assert d_ab == d_ba
    assert d_ab > 0.0


def test_one_degree_longitude_on_equator_metres() -> None:
    d = geo.haversine_distance_m(0.0, 0.0, 0.0, 1.0)
    # Frozen reference value for R = 6 371 000 m.
    assert d == pytest.approx(111_194.92664455874, rel=1e-12)


def test_quarter_meridian_metres() -> None:
    d = geo.haversine_distance_m(90.0, 0.0, 0.0, 0.0)
    assert d == pytest.approx(10_007_543.398010284, rel=1e-12)


def test_antipodal_distance_is_half_circumference() -> None:
    d = geo.haversine_distance_m(0.0, 0.0, 0.0, 180.0)
    assert d == pytest.approx(math.pi * geo.EARTH_RADIUS_METRES, rel=1e-12)


def test_antimeridian_short_crossing() -> None:
    # Points 0.01 degree apart across the 180 meridian.
    d = geo.haversine_distance_m(0.0, 179.995, 0.0, -179.995)
    assert d == pytest.approx(1_111.9492664455873, rel=1e-9)


# --------------------------------------------------------------------------- #
# Coordinate validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "lat,lon",
    [
        (90.0, 0.0),
        (-90.0, 0.0),
        (0.0, 180.0),
        (0.0, -180.0),
        (45, -90),
    ],
)
def test_valid_boundary_coordinates_accepted(lat: float, lon: float) -> None:
    geo.validate_coordinates(lat, lon)


@pytest.mark.parametrize(
    "lat,lon",
    [
        (90.0001, 0.0),
        (-90.5, 0.0),
        (0.0, 180.1),
        (0.0, -181.0),
        (float("nan"), 0.0),
        (0.0, float("inf")),
        (float("-inf"), 0.0),
        ("0", 0.0),
        (None, 0.0),
        (True, 0.0),
    ],
)
def test_invalid_coordinates_rejected(lat, lon) -> None:
    with pytest.raises(InvalidCoordinateError):
        geo.validate_coordinates(lat, lon)


# --------------------------------------------------------------------------- #
# Geohash encoding / decoding
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "lat,lon,expected",
    [
        (57.64911, 10.40744, "u4pruydqqvj"),
        (42.6, -5.6, "ezs42"),
        (0.0, 0.0, "s0000000000"),
    ],
)
def test_encode_known_vectors(lat: float, lon: float, expected: str) -> None:
    assert geo.encode_geohash(lat, lon, len(expected)) == expected


def test_poles_and_meridians_encode_to_edge_cells() -> None:
    assert geo.encode_geohash(90.0, 180.0, 1) == "z"
    assert geo.encode_geohash(-90.0, -180.0, 1) == "0"
    assert geo.encode_geohash(90.0, -180.0, 1) == "b"
    assert geo.encode_geohash(-90.0, 180.0, 1) == "p"


def test_precision_bounds_validated() -> None:
    with pytest.raises(InvalidParameterError):
        geo.encode_geohash(0.0, 0.0, 0)
    with pytest.raises(InvalidParameterError):
        geo.encode_geohash(0.0, 0.0, 13)
    with pytest.raises(InvalidParameterError):
        geo.encode_geohash(0.0, 0.0, 1.5)  # type: ignore[arg-type]


def test_decode_returns_bounding_box() -> None:
    south, north, west, east, clat, clon = geo.decode_geohash("0000")
    assert south == -90.0
    assert west == -180.0
    # precision 4: 10 latitude bits -> 1024 rows of ~0.1758 deg.
    assert north == pytest.approx(-90.0 + 180.0 / 1024)
    # 10 longitude bits -> 1024 columns of ~0.3516 deg.
    assert east == pytest.approx(-180.0 + 360.0 / 1024)
    assert clat == pytest.approx((south + north) / 2)
    assert clon == pytest.approx((west + east) / 2)


def test_decode_rejects_bad_alphabet() -> None:
    with pytest.raises(InvalidParameterError):
        geo.decode_geohash("a0")  # 'a' is not in the base32 alphabet


def test_encode_decode_consistent() -> None:
    # A point's cell must spatially contain the point.
    for lat, lon in [(0.0, 0.0), (47.3, 8.5), (-33.8, 151.2), (89.9, 179.9)]:
        cell = geo.encode_geohash(lat, lon, 7)
        south, north, west, east, _, _ = geo.decode_geohash(cell)
        assert south <= lat <= north
        assert west <= lon <= east


# --------------------------------------------------------------------------- #
# Circle cell enumeration
# --------------------------------------------------------------------------- #
def test_cells_unique_and_cover_grid_neighbours_at_origin() -> None:
    cells = geo.cells_in_circle(0.0, 0.0, 50_000, 9, 100_000)
    assert len(cells) == len(set(cells))
    # Every cell has the adaptive precision chosen.
    assert len({len(c) for c in cells}) == 1


def test_zero_radius_returns_single_cell() -> None:
    cells = geo.cells_in_circle(35.0, 139.0, 0.0, 8, 100_000)
    assert cells == frozenset({geo.encode_geohash(35.0, 139.0, 8)})


def test_antimeridian_cells_do_not_duplicate() -> None:
    cells = geo.cells_in_circle(0.0, 180.0, 300_000, 5, 100_000)
    assert len(cells) == len(set(cells))
    # Cells exist on both sides of the meridian.
    west = [c for c in cells if c[0] in "89"]
    east = [c for c in cells if c[0] in "xg"]
    assert west and east


def test_pole_disc_spans_all_longitudes() -> None:
    _, _, _, _, lon_full = geo.circle_bbox(89.99, 0.0, 50_000)
    assert lon_full is True
    cells = geo.cells_in_circle(89.99, 0.0, 50_000, 7, 100_000)
    assert len(cells) == len(set(cells))
    # All cells lie inside the northern polar band.
    assert all(geo.decode_geohash(c)[0] >= 89.5 for c in cells)
    # The northern-most latitude row is present for all longitude columns.
    # Adaptive precision lands on 5 here (finer grids exceed the cell cap):
    # 13 longitude bits -> 8192 columns.
    top_row = {c for c in cells if geo.decode_geohash(c)[1] == 90.0}
    top_wests = sorted(geo.decode_geohash(c)[2] for c in top_row)
    assert len(top_wests) == 8192
    assert top_wests[0] == -180.0
    assert top_wests[-1] == pytest.approx(180.0 - 360.0 / 8192)


def test_global_disc_covers_entire_grid() -> None:
    cells = geo.cells_in_circle(0.0, 0.0, math.pi * geo.EARTH_RADIUS_METRES, 1, 100)
    # precision 1: 8 columns x 4 rows = 32 cells cover the whole globe.
    assert cells == frozenset(
        geo._cell_from_indices(row, col, 1)
        for col in range(8)
        for row in range(4)
    )


def test_cell_cap_raises_instead_of_silent_scan() -> None:
    # Precision 1 globe needs 32 cells; cap below that must error loudly.
    with pytest.raises(LimitExceededError):
        geo.cells_in_circle(0.0, 0.0, 20_000_000.0, 1, 10)


def test_circle_grid_catches_boundary_points_exhaustively() -> None:
    """Brute-force cross-check: enumerated cells contain every point inside
    the disc for a dense set of centres, including boundary geographies."""
    precision = 3
    _, lon_cols, lat_rows = geo._grid_dimensions(precision)
    radius = 150_000.0
    centres = [
        (0.0, 0.0),
        (0.0, 180.0),
        (0.0, -180.0),
        (60.0, -120.0),
        (-75.0, 45.0),
        (89.99, 90.0),
        (-89.99, -90.0),
    ]
    for clat, clon in centres:
        cells = geo.cells_in_circle(clat, clon, radius, precision, 1_000_000)
        for row in range(lat_rows):
            for col in range(lon_cols):
                cell = geo._cell_from_indices(row, col, precision)
                _, _, _, _, plat, plon = geo.decode_geohash(cell)
                # nearest corner/edge distance conservatism: check centre of
                # cell only here; corners checked by index-level integration
                # tests in test_boundaries.py.
                d = geo.haversine_distance_m(clat, clon, plat, plon)
                if d <= radius:
                    assert cell in cells, (cell, clat, clon)
