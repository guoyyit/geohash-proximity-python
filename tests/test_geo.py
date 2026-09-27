"""Pinned geometric conventions and edge handling.

These tests fix the documented distance convention: great-circle
(haversine) distance in metres on a sphere of radius
``EARTH_RADIUS_M = 6 371 008.8 m``.  If these values change, the
documentation must change too.
"""

from __future__ import annotations

import math

import pytest

from geo_proximity.geo import (
    EARTH_RADIUS_M,
    HALF_CIRCUMFERENCE_M,
    haversine_m,
    max_longitude_extent,
    normalize_longitude,
    ring_threshold_latitudes,
    spherical_cap_bbox,
    validate_latlng,
)


def test_earth_constants_are_pinned() -> None:
    assert EARTH_RADIUS_M == 6_371_008.8
    assert HALF_CIRCUMFERENCE_M == pytest.approx(
        math.pi * EARTH_RADIUS_M, rel=1e-15
    )
    # The antipodal distance used by global queries: half the mean
    # circumference pi * R (~20 015 114 m with R = 6 371 008.8 m).
    assert HALF_CIRCUMFERENCE_M == pytest.approx(20_015_114.4, abs=1.0)


def test_haversine_known_values() -> None:
    assert haversine_m(0.0, 0.0, 0.0, 0.0) == 0.0
    # One degree of longitude on the equator.
    assert haversine_m(0.0, 0.0, 0.0, 1.0) == pytest.approx(
        EARTH_RADIUS_M * math.radians(1.0), abs=1e-6
    )
    # Quarter/half great circle.
    assert haversine_m(0.0, 0.0, 0.0, 90.0) == pytest.approx(
        HALF_CIRCUMFERENCE_M / 2.0, abs=1e-5
    )
    assert haversine_m(0.0, 0.0, 0.0, 180.0) == pytest.approx(
        HALF_CIRCUMFERENCE_M, abs=1e-5
    )
    # Symmetry.
    assert haversine_m(12.3, -45.6, -7.8, 100.1) == pytest.approx(
        haversine_m(-7.8, 100.1, 12.3, -45.6), abs=1e-6
    )


def test_haversine_seam_and_poles_are_continuous() -> None:
    # The ±180° antimeridian is a single seam: 2 degrees apart across it.
    two_deg = 2.0 * EARTH_RADIUS_M * math.radians(1.0)
    assert haversine_m(0.0, 179.0, 0.0, -179.0) == pytest.approx(
        two_deg, abs=1e-5
    )
    # Every longitude converges at the poles (distance to pole).
    for lon in (-180.0, -90.0, 0.0, 90.0, 179.9):
        assert haversine_m(89.0, lon, 90.0, 0.0) == pytest.approx(
            EARTH_RADIUS_M * math.radians(1.0), abs=1e-5
        )


def test_normalize_longitude_wraps() -> None:
    assert normalize_longitude(180.0) == -180.0
    assert normalize_longitude(-180.0) == -180.0
    assert normalize_longitude(360.0) == 0.0
    assert normalize_longitude(181.0) == pytest.approx(-179.0)
    assert normalize_longitude(-270.0) == 90.0


@pytest.mark.parametrize(
    "lat,lon",
    [(90.0001, 0.0), (-90.0001, 0.0), (float("nan"), 0.0),
     (0.0, float("inf")), ("1", 0.0), (0.0, None)],
)
def test_invalid_coordinates_rejected(lat, lon) -> None:
    with pytest.raises(ValueError):
        validate_latlng(lat, lon)


def test_cap_bbox_basic_and_poles() -> None:
    d = 10_000.0
    deg = d / EARTH_RADIUS_M * 180.0 / math.pi
    min_lat, max_lat, min_lon, max_lon = spherical_cap_bbox(0.0, 0.0, d)
    assert min_lat == pytest.approx(-deg)
    assert max_lat == pytest.approx(deg)
    assert min_lon == pytest.approx(-deg)
    assert max_lon == pytest.approx(deg)

    # Antimeridian crossing: box is returned unwrapped around -180/180.
    *_, lo_lon, hi_lon = spherical_cap_bbox(0.0, 180.0, d)
    assert lo_lon < -180.0 and hi_lon < -179.0

    # Pole-centered cap covers the full longitude ring.
    min_lat, max_lat, _, _ = spherical_cap_bbox(90.0, 0.0, 100_000.0)
    assert max_lat == 90.0
    assert min_lat == pytest.approx(
        90.0 - 100_000.0 / EARTH_RADIUS_M * 180.0 / math.pi
    )

    # Whole globe.
    assert spherical_cap_bbox(10.0, 20.0, HALF_CIRCUMFERENCE_M) == (
        -90.0, 90.0, pytest.approx(20.0 - 360.0), pytest.approx(20.0 + 360.0)
    )


def test_ring_thresholds() -> None:
    # Cap centered exactly on the north pole: parallels above 90 - c are
    # full rings.
    n, s = ring_threshold_latitudes(90.0, 100_000.0)
    assert s is None
    assert n == pytest.approx(
        90.0 - 100_000.0 / EARTH_RADIUS_M * 180.0 / math.pi
    )
    # A cap that does not reach a pole has no ring threshold there.
    n, s = ring_threshold_latitudes(0.0, 100_000.0)
    assert n is None and s is None
    # Antipodal cap covers every point; at centre lat 0 the ring region
    # starts exactly at the equator on both sides (parallels at/beyond are
    # wholly covered).
    n, s = ring_threshold_latitudes(0.0, HALF_CIRCUMFERENCE_M)
    assert n == pytest.approx(0.0, abs=1e-9)
    assert s == pytest.approx(0.0, abs=1e-9)


def test_max_extent_matches_numeric_scan() -> None:
    import random
    random.seed(42)
    from geo_proximity.geo import longitude_extent_at_lat
    for _ in range(200):
        cap_lat = random.uniform(-89.9, 89.9)
        radius = random.uniform(1_000.0, 10_000_000.0)
        lo = random.uniform(-90.0, 89.0)
        hi = min(90.0, lo + random.uniform(0.1, 2.0))
        n_thr, s_thr = ring_threshold_latitudes(cap_lat, radius)
        exact = max_longitude_extent(cap_lat, radius, lo, hi)

        def in_finite_region(lat: float) -> bool:
            if n_thr is not None and lat >= n_thr - 1e-9:
                return False
            if s_thr is not None and lat <= s_thr + 1e-9:
                return False
            return True

        sampled = (lo + (hi - lo) * i / 400 for i in range(401))
        scanned = [
            longitude_extent_at_lat(cap_lat, 0.0, radius, lat)
            for lat in sampled if in_finite_region(lat)
        ]
        numeric = max(
            (h for h in scanned if h is not None and not math.isinf(h)),
            default=None,
        )
        if exact is None:
            assert numeric is None or numeric < 1e-9
        else:
            assert numeric is not None
            assert exact >= numeric - 1e-7
