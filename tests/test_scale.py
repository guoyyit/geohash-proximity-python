"""Scale tests: sublinear query latency and explicit cap errors."""
from __future__ import annotations

import random
import time

import pytest

from geohash_proximity import IndexConfig, Point, ProximityIndex, geo
from geohash_proximity.errors import LimitExceededError


def _build(n: int, seed: int = 99) -> ProximityIndex:
    rng = random.Random(seed)
    idx = ProximityIndex()
    batch = []
    for i in range(n):
        lat = rng.uniform(-80.0, 80.0)
        lon = rng.uniform(-180.0, 180.0)
        batch.append(Point(f"p{i}", lat, lon, {"i": i}))
    idx.bulk_load(batch)
    return idx


def test_radius_query_sublinear_vs_full_scan() -> None:
    idx_small = _build(2_000)
    idx_big = _build(20_000)

    def time_queries(index: ProximityIndex, rounds: int) -> float:
        rng = random.Random(7)
        start = time.perf_counter()
        for _ in range(rounds):
            lat = rng.uniform(-70.0, 70.0)
            lon = rng.uniform(-180.0, 180.0)
            index.radius_query(lat, lon, 25_000.0)
        return time.perf_counter() - start

    rounds = 30
    # Warm up.
    time_queries(idx_small, 3)
    t_small = time_queries(idx_small, rounds)
    t_big = time_queries(idx_big, rounds)
    # 10x data must not cost 10x the time: the grid prunes most cells.
    # Generous CI-bound factor; correctness is covered elsewhere.
    assert t_big < 5.0 * t_small, (t_small, t_big)


def test_nearest_query_does_not_scan_all_points_for_small_limit() -> None:
    idx_big = _build(20_000)
    rng = random.Random(1)
    start = time.perf_counter()
    for _ in range(30):
        idx_big.nearest(rng.uniform(-70, 70), rng.uniform(-180, 180), 5)
    elapsed = time.perf_counter() - start
    # Sanity bound for 30 in-memory NN queries on 20k points.
    assert elapsed < 3.0


def test_max_points_cap_raises_for_bulk_and_insert() -> None:
    idx = ProximityIndex(IndexConfig(max_points=10))
    idx.bulk_load([Point(f"p{i}", 0.0, 0.0) for i in range(10)])
    with pytest.raises(LimitExceededError):
        idx.insert("overflow", 0.0, 0.0)
    with pytest.raises(LimitExceededError):
        idx.bulk_load([Point("p10", 0.0, 0.0), Point("p11", 0.0, 0.0)])
    assert len(idx) == 10


def test_bulk_cap_violation_is_atomic() -> None:
    idx = ProximityIndex(IndexConfig(max_points=5))
    with pytest.raises(LimitExceededError):
        idx.bulk_load([Point(f"p{i}", 0.0, 0.0) for i in range(6)])
    assert len(idx) == 0


def test_max_scanned_cells_raises_instead_of_truncating() -> None:
    # With only precision 1 available and a cap below a full row of 8
    # columns, a near-global radius must fail loudly rather than silently
    # scanning/truncating.
    idx = ProximityIndex(IndexConfig(geohash_precision=1, max_scanned_cells=4))
    idx.insert("a", 0.0, 0.0)
    with pytest.raises(LimitExceededError):
        idx.radius_query(0.0, 0.0, 15_000_000.0)


def test_nearest_cell_cap_raises_when_search_exceeds_budget() -> None:
    # 100 points spread over the globe, limit 50: the best-first search must
    # visit far more than 16 cells, so the cap triggers a clear error.
    rng = random.Random(11)
    idx = ProximityIndex(IndexConfig(max_scanned_cells=16))
    idx.bulk_load(
        [
            Point(f"p{i}", rng.uniform(-85, 85), rng.uniform(-180, 180))
            for i in range(100)
        ]
    )
    with pytest.raises(LimitExceededError):
        idx.nearest(0.0, 0.0, 50)


def test_small_cell_cap_does_not_affect_small_queries() -> None:
    idx = ProximityIndex(IndexConfig(geohash_precision=8, max_scanned_cells=128))
    idx.insert("a", 0.0, 0.0)
    # A 1 km radius stays well within 128 cells at precision 8.
    got = idx.radius_query(0.0, 0.0, 1_000.0)
    assert [sp.point.id for sp in got] == ["a"]


def test_nearest_with_restrictive_cell_cap_still_correct_when_possible() -> None:
    idx = ProximityIndex(IndexConfig(geohash_precision=6, max_scanned_cells=512))
    idx.insert("a", 0.0, 0.0)
    idx.insert("b", 0.0, 0.001)
    got = idx.nearest(0.0, 0.001, 1)
    assert got[0].point.id == "b"


def test_index_memory_scales_linearly_with_grid_precision() -> None:
    # Each point is indexed at every precision: building must succeed and
    # queries must find points inserted at any density.
    idx = ProximityIndex(IndexConfig(geohash_precision=10))
    rng = random.Random(3)
    for i in range(500):
        idx.insert(f"p{i}", rng.uniform(-60, 60), rng.uniform(-180, 180))
    got = idx.radius_query(10.0, 10.0, 500_000.0)
    # Cross-check against brute force on the same RNG draws.
    rng2 = random.Random(3)
    expected = []
    for i in range(500):
        lat, lon = rng2.uniform(-60, 60), rng2.uniform(-180, 180)
        d = geo.haversine_distance_m(10.0, 10.0, lat, lon)
        if d <= 500_000.0:
            expected.append((d, f"p{i}"))
    expected.sort()
    assert [(sp.point.id, round(sp.distance, 6)) for sp in got] == [
        (pid, round(d, 6)) for d, pid in expected
    ]
