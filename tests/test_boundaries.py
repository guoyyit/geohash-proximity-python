"""Tests for antimeridian, pole and global-coverage behaviour.

These tests place dense grids of points (including on both sides of the
180th meridian and near the poles) and cross-check query results against a
brute-force linear scan so that no point is missed or double-counted.
"""
from __future__ import annotations

import math

import pytest

from geohash_proximity import Point, ProximityIndex, geo


def _brute_force_radius(points, lat, lon, radius):
    expected = []
    for p in points:
        d = geo.haversine_distance_m(lat, lon, p.latitude, p.longitude)
        if d <= radius + 1e-9:
            expected.append((d, p.id))
    expected.sort()
    return expected


def _grid(lat_step=5.0, lon_step=5.0):
    points = []
    lat = -85.0
    while lat <= 85.0 + 1e-9:
        lon = -180.0
        while lon <= 180.0 + 1e-9:
            points.append(
                Point(
                    f"{lat:.4f},{lon:.4f}",
                    lat,
                    ((lon + 180.0) % 360.0) - 180.0,
                )
            )
            lon += lon_step
        lat += lat_step
    return points


@pytest.fixture(scope="module")
def populated() -> tuple[ProximityIndex, list[Point]]:
    points = _grid()
    idx = ProximityIndex()
    idx.bulk_load(points)
    return idx, points


CENTRES = [
    (0.0, 180.0),
    (0.0, -180.0),
    (0.0, 179.99),
    (0.0, -179.99),
    (70.0, 180.0),
    (-70.0, -180.0),
    (85.0, 0.0),
    (-85.0, 0.0),
    (85.0, 180.0),
    (-85.0, 180.0),
    (0.0, 0.0),
]
RADII = [50_000.0, 500_000.0, 3_000_000.0]


@pytest.mark.parametrize("radius", RADII)
@pytest.mark.parametrize("clat,clon", CENTRES)
def test_radius_matches_brute_force_on_global_grid(
    populated, clat, clon, radius
) -> None:
    idx, points = populated
    got = idx.radius_query(clat, clon, radius)
    ids = [sp.point.id for sp in got]
    # No duplicates.
    assert len(ids) == len(set(ids))
    expected = _brute_force_radius(points, clat, clon, radius)
    assert [(round(sp.distance, 9), sp.point.id) for sp in got] == [
        (round(d, 9), pid) for d, pid in expected
    ]


@pytest.mark.parametrize("clat,clon", CENTRES)
def test_nearest_matches_brute_force_on_global_grid(populated, clat, clon) -> None:
    idx, points = populated
    for limit in (1, 5, 20):
        got = idx.nearest(clat, clon, limit)
        expected = sorted(
            (
                geo.haversine_distance_m(clat, clon, p.latitude, p.longitude),
                p.id,
            )
            for p in points
        )[:limit]
        assert [(round(sp.distance, 9), sp.point.id) for sp in got] == [
            (round(d, 9), pid) for d, pid in expected
        ]


def test_points_straddling_meridian_all_returned() -> None:
    idx = ProximityIndex()
    idx.insert("east", 0.0, 179.98)
    idx.insert("west", 0.0, -179.98)
    idx.insert("far", 0.0, 0.0)
    got = idx.radius_query(0.0, 180.0, 5000)
    assert {sp.point.id for sp in got} == {"east", "west"}
    got_neg = idx.radius_query(0.0, -180.0, 5000)
    assert {sp.point.id for sp in got_neg} == {"east", "west"}


def test_points_straddling_pole_all_returned() -> None:
    idx = ProximityIndex()
    # Points surrounding the north pole across every longitude.
    for lon in range(-180, 180, 30):
        idx.insert(f"n{lon}", 89.9, float(lon))
    idx.insert("equator", 0.0, 0.0)
    got = idx.radius_query(89.95, 45.0, 20_000)
    # All 12 ring points are within ~14 km of each other along the parallel,
    # and the 20 km disc from a mid-ring centre reaches around the whole ring.
    assert len(got) == 12
    assert len({sp.point.id for sp in got}) == 12
    assert "equator" not in {sp.point.id for sp in got}


def test_results_independent_of_insertion_order() -> None:
    import random

    points = _grid(lat_step=10.0, lon_step=10.0)
    rng = random.Random(1234)
    rng.shuffle(points)
    idx = ProximityIndex()
    idx.bulk_load(points)

    ordered = _grid(lat_step=10.0, lon_step=10.0)
    reference = ProximityIndex()
    reference.bulk_load(ordered)

    for clat, clon in [(0.0, 180.0), (40.0, -75.0), (-80.0, 90.0)]:
        a = idx.radius_query(clat, clon, 750_000)
        b = reference.radius_query(clat, clon, 750_000)
        assert [(sp.point.id, round(sp.distance, 9)) for sp in a] == [
            (sp.point.id, round(sp.distance, 9)) for sp in b
        ]
        na = idx.nearest(clat, clon, 15)
        nb = reference.nearest(clat, clon, 15)
        assert [sp.point.id for sp in na] == [sp.point.id for sp in nb]


def test_move_point_across_meridian_is_visible_once() -> None:
    idx = ProximityIndex()
    idx.insert("m", 0.0, 0.0)
    idx.update("m", longitude=179.99)
    got = idx.radius_query(0.0, 180.0, 5000)
    assert [sp.point.id for sp in got] == ["m"]
    assert idx.radius_query(0.0, 0.0, 5000) == []
    # Move it to the other side.
    idx.update("m", longitude=-179.99)
    got2 = idx.radius_query(0.0, -180.0, 5000)
    assert [sp.point.id for sp in got2] == ["m"]
    # Stale grid buckets do not linger.
    assert idx.radius_query(0.0, 180.0, 100) == []


def test_global_query_at_antipodal_radius_returns_everything_once() -> None:
    points = _grid(lat_step=15.0, lon_step=15.0)
    idx = ProximityIndex(
        __import__("geohash_proximity").IndexConfig(
            max_radius_m=math.pi * geo.EARTH_RADIUS_METRES
        )
    )
    idx.bulk_load(points)
    radius = math.pi * geo.EARTH_RADIUS_METRES
    got = idx.radius_query(33.0, 111.0, radius)
    assert len(got) == len(points)
    assert len({sp.point.id for sp in got}) == len(points)


def test_deleted_point_reappears_nowhere_near_meridian() -> None:
    idx = ProximityIndex()
    idx.insert("d", 0.0, 179.999)
    idx.delete("d")
    for lon in (180.0, -180.0):
        assert idx.radius_query(0.0, lon, 50_000) == []
