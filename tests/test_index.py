"""Tests for point CRUD, radius and nearest-neighbour queries."""
from __future__ import annotations

import pytest

from geohash_proximity import (
    IndexConfig,
    Point,
    ProximityIndex,
    ScoredPoint,
    geo,
)
from geohash_proximity.errors import (
    DuplicatePointError,
    InvalidCoordinateError,
    InvalidParameterError,
    LimitExceededError,
    PointNotFoundError,
)


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #
def test_insert_and_get_copies_attributes() -> None:
    idx = ProximityIndex()
    attrs = {"k": "v"}
    p = idx.insert("p1", 10.0, 20.0, attrs)
    attrs["k"] = "mutated"
    assert p.attributes == {"k": "v"}
    fetched = idx.get("p1")
    assert fetched is not None
    assert fetched.attributes == {"k": "v"}
    assert idx.get("missing") is None
    assert idx.contains("p1")
    assert "p1" in idx
    assert len(idx) == 1


def test_insert_duplicate_id_rejected() -> None:
    idx = ProximityIndex()
    idx.insert("p1", 0.0, 0.0)
    with pytest.raises(DuplicatePointError):
        idx.insert("p1", 1.0, 1.0)
    # Original point is untouched.
    assert idx.get("p1") == Point("p1", 0.0, 0.0, {})


def test_upsert_replaces_deterministically() -> None:
    idx = ProximityIndex()
    idx.upsert("p1", 0.0, 0.0, {"v": 1})
    idx.upsert("p1", 10.0, 10.0, {"v": 2})
    assert len(idx) == 1
    assert idx.get("p1") == Point("p1", 10.0, 10.0, {"v": 2})


def test_update_changes_coordinates_and_index() -> None:
    idx = ProximityIndex()
    idx.insert("p1", 0.0, 0.0)
    idx.update("p1", latitude=10.0, longitude=10.0)
    assert idx.get("p1") == Point("p1", 10.0, 10.0, {})
    assert idx.radius_query(0.0, 0.0, 1000) == []
    near = idx.radius_query(10.0, 10.0, 1000)
    assert [sp.point.id for sp in near] == ["p1"]


def test_update_partial_fields() -> None:
    idx = ProximityIndex()
    idx.insert("p1", 0.0, 0.0, {"a": 1})
    idx.update("p1", latitude=5.0)
    p = idx.get("p1")
    assert p is not None
    assert p.latitude == 5.0
    assert p.longitude == 0.0
    assert p.attributes == {"a": 1}
    idx.update("p1", attributes={"b": 2})
    assert idx.get("p1").attributes == {"b": 2}


def test_update_and_delete_unknown_id() -> None:
    idx = ProximityIndex()
    with pytest.raises(PointNotFoundError):
        idx.update("nope", latitude=1.0)
    with pytest.raises(PointNotFoundError):
        idx.delete("nope")


def test_delete_removes_from_queries() -> None:
    idx = ProximityIndex()
    idx.insert("p1", 0.0, 0.0)
    idx.delete("p1")
    assert len(idx) == 0
    assert idx.radius_query(0.0, 0.0, 1000) == []
    assert not idx.contains("p1")


@pytest.mark.parametrize(
    "lat,lon",
    [(90.1, 0.0), (-91.0, 0.0), (0.0, 180.1), (0.0, -200), ("x", 0.0), (1, None)],
)
def test_insert_rejects_invalid_coordinates(lat, lon) -> None:
    idx = ProximityIndex()
    with pytest.raises(InvalidCoordinateError):
        idx.insert("p", lat, lon)


def test_insert_rejects_invalid_id_and_attributes() -> None:
    idx = ProximityIndex()
    with pytest.raises(InvalidParameterError):
        idx.insert("", 0.0, 0.0)
    with pytest.raises(InvalidParameterError):
        idx.insert(123, 0.0, 0.0)  # type: ignore[arg-type]
    with pytest.raises(InvalidParameterError):
        idx.insert("p", 0.0, 0.0, [("a", 1)])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Bulk load
# --------------------------------------------------------------------------- #
def test_bulk_load_replace_default() -> None:
    idx = ProximityIndex()
    idx.insert("a", 0.0, 0.0, {"v": "old"})
    n = idx.bulk_load(
        [
            Point("a", 1.0, 1.0, {"v": "new"}),
            Point("b", 2.0, 2.0, {}),
            Point("b", 3.0, 3.0, {}),
        ]
    )
    assert n == 2
    assert len(idx) == 2
    assert idx.get("a").attributes == {"v": "new"}
    assert idx.get("b") == Point("b", 3.0, 3.0, {})


def test_bulk_load_skip_keeps_first() -> None:
    idx = ProximityIndex()
    idx.insert("a", 9.0, 9.0)
    n = idx.bulk_load(
        [Point("a", 1.0, 1.0), Point("b", 2.0, 2.0), Point("b", 3.0, 3.0)],
        on_duplicate="skip",
    )
    assert n == 1
    assert idx.get("a") == Point("a", 9.0, 9.0, {})
    assert idx.get("b") == Point("b", 2.0, 2.0, {})


def test_bulk_load_raise_is_atomic() -> None:
    idx = ProximityIndex()
    idx.insert("a", 0.0, 0.0)
    with pytest.raises(DuplicatePointError):
        idx.bulk_load(
            [Point("b", 1.0, 1.0), Point("a", 2.0, 2.0)],
            on_duplicate="raise",
        )
    # Nothing from the batch landed.
    assert len(idx) == 1
    assert idx.get("b") is None

    with pytest.raises(DuplicatePointError):
        idx.bulk_load(
            [Point("c", 1.0, 1.0), Point("c", 2.0, 2.0)],
            on_duplicate="raise",
        )
    assert idx.get("c") is None


def test_bulk_load_invalid_mode_and_payload() -> None:
    idx = ProximityIndex()
    with pytest.raises(InvalidParameterError):
        idx.bulk_load([], on_duplicate="bogus")
    with pytest.raises(InvalidParameterError):
        idx.bulk_load([("x", 1.0, 1.0)])  # type: ignore[list-item]
    with pytest.raises(InvalidCoordinateError):
        idx.bulk_load([Point("x", 999.0, 0.0, {})])


# --------------------------------------------------------------------------- #
# Radius query
# --------------------------------------------------------------------------- #
def test_radius_query_matches_brute_force() -> None:
    rng = pytest.importorskip("random")
    rng.seed(42)
    idx = ProximityIndex()
    points = []
    for i in range(500):
        lat = rng.uniform(-60, 60)
        lon = rng.uniform(-170, 170)
        idx.insert(f"p{i}", lat, lon, {"i": i})
        points.append((f"p{i}", lat, lon))
    clat, clon, radius = 12.3, -45.6, 400_000.0
    expected = []
    for pid, lat, lon in points:
        d = geo.haversine_distance_m(clat, clon, lat, lon)
        if d <= radius:
            expected.append((d, pid))
    expected.sort()
    got = idx.radius_query(clat, clon, radius)
    assert [(round(sp.distance, 9), sp.point.id) for sp in got] == [
        (round(d, 9), pid) for d, pid in expected
    ]


def test_radius_query_inclusive_boundary() -> None:
    idx = ProximityIndex()
    # A point exactly one degree away on the equator.
    idx.insert("edge", 0.0, 1.0)
    radius = geo.haversine_distance_m(0.0, 0.0, 0.0, 1.0)
    got = idx.radius_query(0.0, 0.0, radius)
    assert [sp.point.id for sp in got] == ["edge"]
    # Anything infinitesimally smaller (beyond epsilon) excludes it.
    got2 = idx.radius_query(0.0, 0.0, radius - 1e-6)
    assert got2 == []


def test_radius_query_returns_scored_points_sorted() -> None:
    idx = ProximityIndex()
    idx.insert("far", 0.0, 1.0)
    idx.insert("near", 0.0, 0.1)
    got = idx.radius_query(0.0, 0.0, 500_000)
    assert [sp.point.id for sp in got] == ["near", "far"]
    assert all(isinstance(sp, ScoredPoint) for sp in got)
    assert got[0].distance < got[1].distance
    # Attributes are returned and detached.
    assert got[0].point.attributes == {}


def test_radius_invalid_arguments() -> None:
    idx = ProximityIndex()
    with pytest.raises(InvalidCoordinateError):
        idx.radius_query(200.0, 0.0, 100)
    with pytest.raises(InvalidParameterError):
        idx.radius_query(0.0, 0.0, -1)
    with pytest.raises(InvalidParameterError):
        idx.radius_query(0.0, 0.0, float("inf"))
    with pytest.raises(InvalidParameterError):
        idx.radius_query(0.0, 0.0, "100")  # type: ignore[arg-type]


def test_radius_cap_rejected_not_silently_cropped() -> None:
    idx = ProximityIndex(IndexConfig(max_radius_m=1_000.0))
    with pytest.raises(LimitExceededError):
        idx.radius_query(0.0, 0.0, 5_000)


# --------------------------------------------------------------------------- #
# Nearest neighbour
# --------------------------------------------------------------------------- #
def test_nearest_orders_by_distance_ascending() -> None:
    idx = ProximityIndex()
    idx.insert("a", 0.0, 1.0)
    idx.insert("b", 0.0, 0.1)
    idx.insert("c", 0.0, 0.5)
    got = idx.nearest(0.0, 0.0, 2)
    assert [sp.point.id for sp in got] == ["b", "c"]


def test_nearest_tie_broken_by_id_stably() -> None:
    idx = ProximityIndex()
    # Symmetric points at identical distance from (0,0).
    idx.insert("zeta", 0.0, 1.0)
    idx.insert("alpha", 1.0, 0.0)
    idx.insert("mid", 0.0, -1.0)
    got = idx.nearest(0.0, 0.0, 3)
    assert [sp.point.id for sp in got] == ["alpha", "mid", "zeta"]
    assert got[0].distance == pytest.approx(got[1].distance)
    # Same result regardless of insertion order.
    idx2 = ProximityIndex()
    for pid in ["mid", "zeta", "alpha"]:
        lat, lon = {"alpha": (1, 0), "mid": (0, -1), "zeta": (0, 1)}[pid]
        idx2.insert(pid, lat, lon)
    assert [sp.point.id for sp in idx2.nearest(0.0, 0.0, 3)] == [
        "alpha",
        "mid",
        "zeta",
    ]


def test_nearest_limit_exceeding_available_returns_all() -> None:
    idx = ProximityIndex()
    assert idx.nearest(0.0, 0.0, 5) == []
    idx.insert("only", 45.0, 45.0)
    got = idx.nearest(0.0, 0.0, 5)
    assert len(got) == 1
    assert got[0].point.id == "only"


def test_nearest_matches_brute_force_sparse_area() -> None:
    rng = pytest.importorskip("random")
    rng.seed(7)
    idx = ProximityIndex()
    points = []
    for i in range(300):
        lat = rng.uniform(-85, 85)
        lon = rng.uniform(-180, 180)
        idx.insert(f"p{i:04d}", lat, lon)
        points.append((lat, lon, f"p{i:04d}"))
    clat, clon = -50.0, 170.0
    expected = sorted(
        (geo.haversine_distance_m(clat, clon, lat, lon), pid)
        for lat, lon, pid in points
    )[:10]
    got = idx.nearest(clat, clon, 10)
    assert [(round(sp.distance, 9), sp.point.id) for sp in got] == [
        (round(d, 9), pid) for d, pid in expected
    ]


def test_nearest_invalid_limit() -> None:
    idx = ProximityIndex()
    idx.insert("a", 0.0, 0.0)
    for bad in (0, -1, 1.5, "2", True):
        with pytest.raises(InvalidParameterError):
            idx.nearest(0.0, 0.0, bad)


def test_nearest_limit_cap_enforced() -> None:
    idx = ProximityIndex(IndexConfig(max_limit=3))
    with pytest.raises(LimitExceededError):
        idx.nearest(0.0, 0.0, 4)


def test_nearest_global_search_when_points_antipodal() -> None:
    idx = ProximityIndex()
    idx.insert("a", 0.0, 0.0)
    idx.insert("b", 0.0, 179.9)
    got = idx.nearest(0.0, -180.0, 2)
    assert [sp.point.id for sp in got] == ["b", "a"]


# --------------------------------------------------------------------------- #
# Caps
# --------------------------------------------------------------------------- #
def test_max_points_cap_enforced_for_inserts_and_bulk() -> None:
    idx = ProximityIndex(IndexConfig(max_points=2))
    idx.insert("a", 0.0, 0.0)
    idx.insert("b", 0.0, 0.0)
    with pytest.raises(LimitExceededError):
        idx.insert("c", 0.0, 0.0)
    # Upserting an existing id stays within the cap.
    idx.upsert("a", 1.0, 1.0)
    with pytest.raises(LimitExceededError):
        idx.bulk_load([Point("c", 0.0, 0.0, {}), Point("d", 1.0, 1.0, {})])
    assert len(idx) == 2


def test_invalid_config_rejected() -> None:
    with pytest.raises(InvalidParameterError):
        IndexConfig(max_points=0)
    with pytest.raises(InvalidParameterError):
        IndexConfig(max_radius_m=-1)
    with pytest.raises(InvalidParameterError):
        IndexConfig(max_scanned_cells=0)
    with pytest.raises(InvalidParameterError):
        IndexConfig(geohash_precision=13)
    with pytest.raises(InvalidParameterError):
        IndexConfig(max_limit=0)
