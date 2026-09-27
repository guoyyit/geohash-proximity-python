"""Grid index: containment, order independence and geographic edges."""

from __future__ import annotations

import math

import pytest

from geo_proximity.geo import HALF_CIRCUMFERENCE_M, haversine_m
from geo_proximity.gridindex import GridIndex
from geo_proximity.models import Point


def _p(pid: str, lat: float, lon: float) -> Point:
    return Point(pid, lat, lon, {})


def test_add_remove_move_and_contains() -> None:
    idx = GridIndex(6)
    p = _p("a", 39.9, 116.4)
    idx.add(p)
    assert len(idx) == 1 and "a" in idx
    assert idx.get_point("a") is p
    # Move within and across cells; every level stays consistent.
    idx.move("a", _p("a", -33.0, 151.0))
    assert idx.get_point("a").latitude == -33.0
    assert len(idx) == 1
    removed = idx.remove("a")
    assert removed.id == "a"
    assert len(idx) == 0 and "a" not in idx
    with pytest.raises(KeyError):
        idx.remove("a")
    with pytest.raises(KeyError):
        idx.add(p)
        idx.add(p)


def test_insertion_order_independence() -> None:
    pts = [_p(f"p{i:03d}",
              math.degrees(math.asin(((i * 37) % 200) / 100.0 - 1.0)),
              (i * 137.5) % 360.0 - 180.0) for i in range(300)]
    a = GridIndex(8)
    b = GridIndex(8)
    for p in pts:
        a.add(p)
    for p in reversed(pts):
        b.add(p)
    for lat, lon, r in [(0, 0, 100_000.0), (60, 120, 2_000_000.0),
                        (-70, -100, 500_000.0)]:
        ka = sorted(x.id for x in a.candidates(lat, lon, r))
        kb = sorted(x.id for x in b.candidates(lat, lon, r))
        assert ka == kb


@pytest.mark.parametrize("levels", [1, 2, 7, 16])
def test_candidates_superset_and_unique(levels: int) -> None:
    import random
    random.seed(int(f"{levels}123"))
    idx = GridIndex(levels)
    pts = []
    for i in range(800):
        roll = i % 4
        if roll == 0:  # seam cluster
            lat = random.uniform(-90, 90)
            lon = (180.0 + random.uniform(-3, 3) + 180.0) % 360.0 - 180.0
        elif roll == 1:  # polar cluster
            lat = max(-90, min(90, random.choice([90.0, -90.0])
                               + random.uniform(-3, 0)
                               * random.choice([-1, 1])))
            lon = random.uniform(-180, 180)
        else:
            lat = math.degrees(math.asin(random.uniform(-1, 1)))
            lon = random.uniform(-180, 180)
        p = _p(f"p{i}", lat, lon)
        idx.add(p)
        pts.append(p)

    queries = [
        (0.0, 0.0, 0.0), (0.0, 180.0, 10_000.0), (0.0, -180.0, 10_000.0),
        (89.99, 0.0, 50_000.0), (-89.99, 0.0, 50_000.0),
        (90.0, 0.0, 0.0), (-90.0, 0.0, 100_000.0),
        (45.0, 179.9, 200_000.0), (-45.0, -179.9, 200_000.0),
        (10.0, 10.0, HALF_CIRCUMFERENCE_M),
    ]
    for _ in range(200):
        queries.append((
            random.uniform(-90, 90),
            random.choice([random.uniform(-180, 180), 180.0, -180.0]),
            random.choice([1.0, 10_000.0, 1_000_000.0, 10_000_000.0]),
        ))

    for lat, lon, r in queries:
        exact = {p.id for p in pts
                 if haversine_m(lat, lon, p.latitude, p.longitude) <= r + 1e-9}
        got = [p.id for p in idx.candidates(lat, lon, r)]
        # No duplicate candidates ever.
        assert len(got) == len(set(got)), (lat, lon, r)
        # Completeness: every true hit is a candidate.
        assert exact <= set(got), (
            f"missing {len(exact - set(got))} points at "
            f"({lat},{lon}) r={r} levels={levels}"
        )


def test_global_radius_covers_all_points() -> None:
    import random
    random.seed(5)
    idx = GridIndex(10)
    pts = [_p(f"p{i}", random.uniform(-90, 90), random.uniform(-180, 180))
           for i in range(500)]
    for p in pts:
        idx.add(p)
    got = {p.id for p in idx.candidates(0.0, 0.0, HALF_CIRCUMFERENCE_M)}
    assert got == {p.id for p in pts}


def test_zero_radius_includes_coincident_point() -> None:
    idx = GridIndex(12)
    p = _p("x", 12.5, -70.25)
    idx.add(p)
    # A coincident point must always be a candidate; exact distance
    # filtering is the service layer's responsibility.
    assert "x" in [q.id for q in idx.candidates(12.5, -70.25, 0.0)]


def test_level_bounds() -> None:
    GridIndex(1)
    GridIndex(20)
    with pytest.raises(ValueError):
        GridIndex(0)
    with pytest.raises(ValueError):
        GridIndex(21)
    with pytest.raises(TypeError):
        GridIndex(True)  # type: ignore[arg-type]
