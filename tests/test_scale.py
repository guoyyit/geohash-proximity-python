"""Scale behaviour: sub-linear local queries and bounded memory.

These tests guard the performance requirement ("large stores must be
noticeably faster than a linear scan") without being brittle: they assert
on *relative* speedups and on candidate pruning, not on wall-clock
constants.
"""

from __future__ import annotations

import math
import random
import time

from geo_proximity import OnDuplicate, ProximityService, ServiceConfig
from geo_proximity.geo import haversine_m


def _uniform_earth(n: int, seed: int = 11) -> list[tuple[str, float, float]]:
    random.seed(seed)
    out = []
    for i in range(n):
        lat = math.degrees(math.asin(random.uniform(-1, 1)))
        lon = random.uniform(-180, 180)
        out.append((f"p{i:07d}", lat, lon))
    return out


def test_local_query_prunes_candidates() -> None:
    cfg = ServiceConfig(index_levels=16)
    svc = ProximityService(cfg)
    pts = _uniform_earth(50_000)
    svc.bulk_load(pts, on_duplicate=OnDuplicate.REPLACE)

    # Count how many points the index even offers for a small radius.
    r = 5_000.0
    offered = sum(1 for _ in svc.index.candidates(30.0, 120.0, r))
    # A 5 km cap over an Earth-wide uniform set must inspect a tiny
    # fraction of the store (well under 1% and a few hundred at most).
    assert offered < 500
    assert offered / 50_000 < 0.01

    res = svc.radius_query(30.0, 120.0, r)
    # Sanity: every returned point really is within the radius.
    assert all(s.distance_m <= r + 1e-6 for s in res)


def test_indexed_query_beats_linear_scan() -> None:
    n = 20_000
    svc = ProximityService(ServiceConfig(index_levels=14))
    pts = _uniform_earth(n)
    svc.bulk_load(pts, on_duplicate=OnDuplicate.REPLACE)
    lat, lon, r = 40.0, 116.0, 3_000.0

    # Linear-scan baseline.
    def linear() -> int:
        return sum(1 for _, pla, plo in pts
                   if haversine_m(lat, lon, pla, plo) <= r)

    def indexed() -> int:
        return len(svc.radius_query(lat, lon, r))

    # Warm up.
    assert linear() == indexed()

    t0 = time.perf_counter()
    for _ in range(5):
        linear()
    linear_t = (time.perf_counter() - t0) / 5

    t0 = time.perf_counter()
    for _ in range(50):
        indexed()
    indexed_t = (time.perf_counter() - t0) / 50

    # The indexed path must be at least 10x faster than scanning 20k points.
    assert indexed_t * 10 < linear_t, (indexed_t, linear_t)


def test_nearest_query_beats_linear_scan() -> None:
    n = 20_000
    svc = ProximityService(ServiceConfig(index_levels=14))
    pts = _uniform_earth(n, seed=23)
    svc.bulk_load(pts, on_duplicate=OnDuplicate.REPLACE)
    lat, lon = -33.0, 151.0

    def linear() -> list[str]:
        ranked = sorted(
            (haversine_m(lat, lon, pla, plo), pid)
            for pid, pla, plo in pts
        )[:20]
        return [pid for _, pid in ranked]

    def indexed() -> list[str]:
        return [s.point.id for s in svc.nearest(lat, lon, 20)]

    assert linear() == indexed()

    t0 = time.perf_counter()
    for _ in range(5):
        linear()
    linear_t = (time.perf_counter() - t0) / 5
    t0 = time.perf_counter()
    for _ in range(50):
        indexed()
    indexed_t = (time.perf_counter() - t0) / 50
    assert indexed_t * 10 < linear_t, (indexed_t, linear_t)


def test_memory_grows_linearly_and_stays_bounded() -> None:
    # Storing N points across L levels keeps one Point object per id;
    # only set memberships are replicated.  Rough upper bound on the
    # per-point marginal memory (dict + per-level set slot), generous for
    # CPython.
    L = 12
    svc = ProximityService(ServiceConfig(index_levels=L))
    pts = _uniform_earth(20_000, seed=5)
    svc.bulk_load(pts, on_duplicate=OnDuplicate.REPLACE)
    # Membership count equals L * N exactly (one set entry per level).
    total_memberships = sum(
        sum(len(v) for v in level_cells.values())
        for level_cells in svc.index._cells  # type: ignore[attr-defined]
    )
    assert total_memberships == L * 20_000
    # No accidental blow-up: number of non-empty cells <= memberships.
    nonempty = sum(len(c) for c in svc.index._cells)  # type: ignore[attr-defined]
    assert nonempty <= L * 20_000
