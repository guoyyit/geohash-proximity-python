"""ProximityService: writes, queries, duplicate/limit rules and edges."""

from __future__ import annotations

import math

import pytest

from geo_proximity import OnDuplicate, ProximityService, ServiceConfig
from geo_proximity.errors import (
    DuplicateIdError,
    InvalidCoordinatesError,
    InvalidParameterError,
    LimitExceededError,
    PointNotFoundError,
)
from geo_proximity.geo import EARTH_RADIUS_M, HALF_CIRCUMFERENCE_M, haversine_m


# ---------------------------------------------------------------- writes
def test_insert_get_upsert_delete() -> None:
    svc = ProximityService()
    assert svc.insert(("a", 10.0, 20.0, {"k": 1})) is True
    assert len(svc) == 1 and "a" in svc
    assert svc.get("a").attributes == {"k": 1}

    svc.upsert(("a", 11.0, 21.0))
    p = svc.get("a")
    assert (p.latitude, p.longitude) == (11.0, 21.0)

    svc.update("a", latitude=12.0)
    assert svc.get("a").latitude == 12.0 and svc.get("a").longitude == 21.0
    svc.update("a", attributes={"x": 1})
    assert svc.get("a").attributes == {"x": 1}
    svc.update("a", attributes={"y": 2}, merge_attributes=True)
    assert svc.get("a").attributes == {"x": 1, "y": 2}

    assert svc.delete("a") is True
    assert svc.delete("a") is False
    assert svc.get("a") is None
    with pytest.raises(PointNotFoundError):
        svc.update("a", latitude=1.0)


def test_duplicate_policies_are_deterministic() -> None:
    svc = ProximityService()
    svc.insert(("a", 1.0, 1.0))
    with pytest.raises(DuplicateIdError):
        svc.insert(("a", 2.0, 2.0))
    # Original untouched after the error.
    assert (svc.get("a").latitude, svc.get("a").longitude) == (1.0, 1.0)

    assert svc.insert(("a", 2.0, 2.0), on_duplicate=OnDuplicate.SKIP) is False
    assert svc.get("a").latitude == 1.0

    assert svc.insert(("a", 2.0, 2.0), on_duplicate=OnDuplicate.REPLACE) is True
    assert svc.get("a").latitude == 2.0


def test_longitude_wrapping_and_invalid_coordinates() -> None:
    svc = ProximityService()
    svc.insert(("w", 0.0, 540.0))  # 540 -> 180 -> -180
    assert svc.get("w").longitude == -180.0
    svc.insert(("w2", 0.0, -359.0))
    assert svc.get("w2").longitude == 1.0

    for lat, lon in [(90.0001, 0.0), (-90.0001, 0.0), (0.0, float("nan")),
                     (float("inf"), 0.0), ("x", 0.0), (0.0, [1])]:
        with pytest.raises(InvalidCoordinatesError):
            svc.insert(("bad", lat, lon))  # type: ignore[arg-type]
    with pytest.raises(InvalidParameterError):
        svc.insert(("", 0.0, 0.0))
    with pytest.raises(InvalidParameterError):
        svc.insert((123, 0.0, 0.0))  # type: ignore[arg-type]


def test_invalid_query_parameters() -> None:
    svc = ProximityService()
    svc.insert(("a", 0.0, 0.0))
    with pytest.raises(InvalidCoordinatesError):
        svc.radius_query(91.0, 0.0, 10.0)
    for r in (-1.0, float("inf"), "10", True):
        with pytest.raises(InvalidParameterError):
            svc.radius_query(0.0, 0.0, r)  # type: ignore[arg-type]
    for k in (0, -1, "3", True):
        with pytest.raises(InvalidParameterError):
            svc.nearest(0.0, 0.0, k)  # type: ignore[arg-type]


def test_configured_limits_raise_not_truncate() -> None:
    cfg = ServiceConfig(max_points=3, max_neighbors=2, max_batch_size=5,
                        max_query_radius_m=1000.0)
    svc = ProximityService(cfg)
    for i in range(3):
        svc.insert((f"p{i}", 0.0, 0.0))
    with pytest.raises(LimitExceededError):
        svc.insert(("p3", 0.0, 0.0))
    with pytest.raises(LimitExceededError):
        svc.nearest(0.0, 0.0, 3)
    with pytest.raises(LimitExceededError):
        svc.radius_query(0.0, 0.0, 1001.0)
    with pytest.raises(LimitExceededError):
        svc.bulk_load([(f"q{i}", 0.0, 0.0) for i in range(6)])
    # Store unchanged after the failed bulk load.
    assert len(svc) == 3


def test_bulk_load_policies_and_atomicity() -> None:
    svc = ProximityService(ServiceConfig(max_points=100, max_batch_size=100))
    svc.insert(("existing", 0.0, 0.0))

    # Duplicate within the batch and against the store both raise by default.
    with pytest.raises(DuplicateIdError):
        svc.bulk_load([("x", 1.0, 1.0), ("x", 2.0, 2.0)])
    assert svc.get("x") is None  # atomic: nothing applied
    with pytest.raises(DuplicateIdError):
        svc.bulk_load([("existing", 1.0, 1.0)])
    assert svc.get("existing").latitude == 0.0

    assert svc.bulk_load(
        [("a", 1.0, 1.0), ("a", 2.0, 2.0),
         ("existing", 5.0, 5.0), ("b", 3.0, 3.0)],
        on_duplicate=OnDuplicate.REPLACE,
    ) == 3
    assert svc.get("a").latitude == 2.0
    assert svc.get("existing").latitude == 5.0

    stored = svc.bulk_load(
        [("a", 9.0, 9.0), ("c", 4.0, 4.0)],
        on_duplicate=OnDuplicate.SKIP,
    )
    assert stored == 1
    assert svc.get("a").latitude == 2.0  # skipped
    assert svc.get("c").latitude == 4.0


def test_bulk_load_invalid_is_atomic() -> None:
    svc = ProximityService()
    with pytest.raises(InvalidCoordinatesError):
        svc.bulk_load([("ok", 0.0, 0.0), ("bad", 99.0, 0.0)])
    assert len(svc) == 0


# ---------------------------------------------------------------- queries
def test_radius_query_inclusive_boundary_and_order() -> None:
    svc = ProximityService(ServiceConfig(index_levels=12))
    # Ring of points at exactly ~1 km from origin; ids chosen so tie order
    # is not insertion order.
    radius_m = 1000.0
    deg = radius_m / EARTH_RADIUS_M * 180.0 / math.pi
    svc.bulk_load([
        ("z", 0.0, deg, {}),
        ("a", 0.0, -deg, {}),
        ("far", 5.0 * deg, 0.0, {}),
    ], on_duplicate=OnDuplicate.REPLACE)
    res = svc.radius_query(0.0, 0.0, radius_m)
    ids = [s.point.id for s in res]
    assert "far" not in ids
    assert set(ids) == {"a", "z"}  # boundary inclusive
    # Exact mirror pair: same distance by construction; tie broken by id.
    assert ids == sorted(ids)
    for s in res:
        assert s.distance_m == pytest.approx(radius_m, abs=1e-6)


def test_radius_query_distance_pinned() -> None:
    svc = ProximityService()
    svc.insert(("p", 0.0, 1.0))
    res = svc.radius_query(0.0, 0.0, EARTH_RADIUS_M * math.radians(1.0))
    assert len(res) == 1
    assert res[0].distance_m == pytest.approx(
        EARTH_RADIUS_M * math.radians(1.0), abs=1e-6
    )


def test_nearest_sorting_limit_and_exhaustion() -> None:
    svc = ProximityService(ServiceConfig(index_levels=12))
    svc.bulk_load([(f"p{i:02d}",
                    0.0, 0.01 * (i + 1), {}) for i in range(20)],
                  on_duplicate=OnDuplicate.REPLACE)
    res = svc.nearest(0.0, 0.0, 5)
    assert len(res) == 5
    dists = [s.distance_m for s in res]
    assert dists == sorted(dists)
    assert res[0].point.id == "p00"
    # Asking for more than the store returns all points, no error.
    all_res = svc.nearest(0.0, 0.0, 100)
    assert len(all_res) == 20
    # Empty store -> empty list, not an error.
    assert ProximityService().nearest(0.0, 0.0, 5) == []


def test_nearest_tie_breaks_by_id() -> None:
    svc = ProximityService(ServiceConfig(index_levels=10))
    deg = 0.05
    # Two equidistant points east/west; ids intentionally reversed.
    svc.insert(("z", 0.0, deg))
    svc.insert(("a", 0.0, -deg))
    res = svc.nearest(0.0, 0.0, 2)
    assert [s.point.id for s in res] == ["a", "z"]
    assert res[0].distance_m == pytest.approx(res[1].distance_m)


def test_nearest_max_distance() -> None:
    svc = ProximityService()
    svc.bulk_load([("near", 0.0, 0.01), ("far", 50.0, 50.0)])
    res = svc.nearest(0.0, 0.0, 10, max_distance_m=50_000.0)
    assert [s.point.id for s in res] == ["near"]
    with pytest.raises(LimitExceededError):
        svc.nearest(0.0, 0.0, 10, max_distance_m=HALF_CIRCUMFERENCE_M + 1)


@pytest.mark.parametrize("center", [
    (0.0, 180.0), (0.0, -180.0), (90.0, 0.0), (-90.0, 0.0),
    (90.0, 180.0), (-90.0, -180.0), (89.9999, 179.9999),
])
def test_queries_across_antimeridian_and_poles(center) -> None:
    svc = ProximityService(ServiceConfig(index_levels=14))
    # Points at longitude ±(180 - delta) and near the pole at every lon.
    delta = 0.05
    svc.bulk_load([
        ("east", center[0] if abs(center[0]) < 89 else center[0],
         180.0 - delta),
        ("west", center[0] if abs(center[0]) < 89 else center[0],
         -180.0 + delta),
        ("polar_n", 89.999, 45.0),
        ("polar_s", -89.999, -135.0),
        ("other", 0.0, 0.0),
    ], on_duplicate=OnDuplicate.REPLACE)
    r = 50_000.0
    res = svc.radius_query(center[0], center[1], r)
    ids = {s.point.id for s in res}
    if abs(center[0]) < 89:
        # Seam centre: both sides of the seam, neither the far point.
        assert {"east", "west"} <= ids
        assert "other" not in ids
    else:
        # Pole centre: longitude irrelevant; polar point on the same pole
        # is found regardless of its longitude.
        expected_id = "polar_n" if center[0] > 0 else "polar_s"
        assert expected_id in ids
    # nearest across the seam
    near = svc.nearest(center[0], center[1], 2,
                       max_distance_m=200_000.0)
    assert len(near) >= 1
    # No duplicates and ascending distance.
    ids_near = [s.point.id for s in near]
    assert len(ids_near) == len(set(ids_near))
    assert [s.distance_m for s in near] == sorted(s.distance_m for s in near)


def test_radius_limit_cap_raises_when_exceeded() -> None:
    cfg = ServiceConfig(max_neighbors=2, index_levels=12)
    svc = ProximityService(cfg)
    svc.bulk_load([(f"p{i}", 0.0, 0.001 * i, {}) for i in range(5)])
    with pytest.raises(LimitExceededError):
        svc.radius_query(0.0, 0.0, 100_000.0, limit=2)
    res = svc.radius_query(0.0, 0.0, 100_000.0)  # no limit: all returned
    assert len(res) == 5


# ---------------------------------------------------------------- fuzz
def test_matches_brute_force() -> None:
    import random
    random.seed(2026)
    svc = ProximityService(ServiceConfig(index_levels=12))
    data = []
    for i in range(1500):
        roll = random.random()
        if roll < 0.25:
            lat = random.choice([90.0, -90.0]) + random.uniform(-1, 1)
            lat = max(-90.0, min(90.0, lat))
            lon = random.uniform(-180, 180)
        elif roll < 0.5:
            lat = random.uniform(-90, 90)
            lon = (180.0 + random.uniform(-4, 4) + 180.0) % 360.0 - 180.0
        else:
            lat = math.degrees(math.asin(random.uniform(-1, 1)))
            lon = random.uniform(-180, 180)
        pid = f"p{i:04d}"
        data.append((pid, lat, lon))
        svc.upsert((pid, lat, lon))

    for _ in range(600):
        clat = random.uniform(-90, 90)
        clon = random.choice([random.uniform(-180, 180), 180.0, -180.0])
        r = random.choice([100.0, 5_000.0, 50_000.0, 500_000.0,
                           5_000_000.0, HALF_CIRCUMFERENCE_M])
        expected = sorted(
            (haversine_m(clat, clon, lat, lon), pid)
            for pid, lat, lon in data
            if haversine_m(clat, clon, lat, lon) <= r + 1e-9
        )
        got = svc.radius_query(clat, clon, r)
        assert len(got) == len(expected)
        for s, (d, pid) in zip(got, expected):
            assert s.point.id == pid
            assert s.distance_m == pytest.approx(d, abs=1e-6)

        k = random.choice([1, 3, 10, 100])
        exp_near = sorted(
            (haversine_m(clat, clon, lat, lon), pid)
            for pid, lat, lon in data
        )[:k]
        got_near = svc.nearest(clat, clon, k)
        assert len(got_near) == len(exp_near)
        for s, (d, pid) in zip(got_near, exp_near):
            assert s.point.id == pid
            assert s.distance_m == pytest.approx(d, abs=1e-6)
