"""Tests for the FastAPI HTTP layer."""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from geohash_proximity import IndexConfig
from geohash_proximity.api import create_app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app(IndexConfig(max_points=5, max_limit=3)))


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "points": 0}


def test_crud_flow(client: TestClient) -> None:
    payload = {"latitude": 10.0, "longitude": 20.0, "attributes": {"k": "v"}}
    r = client.put("/points/a", json=payload)
    assert r.status_code == 201
    assert r.json()["id"] == "a"

    # Duplicate insert conflicts.
    r = client.put("/points/a", json=payload)
    assert r.status_code == 409

    # Get.
    r = client.get("/points/a")
    assert r.status_code == 200
    assert r.json()["attributes"] == {"k": "v"}
    assert client.get("/points/missing").status_code == 404

    # Update.
    r = client.patch("/points/a", json={"latitude": 11.0})
    assert r.status_code == 200
    assert r.json()["latitude"] == 11.0
    assert r.json()["longitude"] == 20.0

    # Upsert replaces.
    r = client.post("/points/a", json={"latitude": 1.0, "longitude": 2.0})
    assert r.status_code == 200
    assert r.json()["latitude"] == 1.0

    # Delete.
    assert client.delete("/points/a").status_code == 204
    assert client.delete("/points/a").status_code == 404


def test_invalid_coordinates_map_to_422(client: TestClient) -> None:
    r = client.put("/points/bad", json={"latitude": 200, "longitude": 0})
    assert r.status_code == 422
    assert "error" in r.json() or "detail" in r.json()


def test_bulk_endpoint(client: TestClient) -> None:
    r = client.post(
        "/bulk",
        json={
            "points": [
                {"id": "a", "latitude": 0.0, "longitude": 0.0},
                {"id": "b", "latitude": 0.0, "longitude": 0.01},
            ],
            "on_duplicate": "raise",
        },
    )
    assert r.status_code == 200
    assert r.json()["written"] == 2

    # Duplicate within batch with raise -> 409 and atomic.
    r = client.post(
        "/bulk",
        json={
            "points": [
                {"id": "c", "latitude": 1.0, "longitude": 1.0},
                {"id": "c", "latitude": 2.0, "longitude": 2.0},
            ],
            "on_duplicate": "raise",
        },
    )
    assert r.status_code == 409
    assert client.get("/health").json()["points"] == 2


def test_radius_endpoint(client: TestClient) -> None:
    client.post("/bulk", json={"points": [
        {"id": "near", "latitude": 0.0, "longitude": 0.01},
        {"id": "far", "latitude": 0.0, "longitude": 1.0},
    ]})
    r = client.get("/query/radius", params={"latitude": 0, "longitude": 0, "radius_m": 5000})
    assert r.status_code == 200
    ids = [p["id"] for p in r.json()["results"]]
    assert ids == ["near"]
    assert r.json()["results"][0]["distance"] > 0

    r = client.get("/query/radius", params={"latitude": 0, "longitude": 0, "radius_m": -1})
    assert r.status_code == 422


def test_nearest_endpoint_and_caps(client: TestClient) -> None:
    client.post("/bulk", json={"points": [
        {"id": "a", "latitude": 0.0, "longitude": 0.0},
        {"id": "b", "latitude": 0.0, "longitude": 1.0},
    ]})
    r = client.get("/query/nearest", params={"latitude": 0, "longitude": 0, "limit": 1})
    assert [p["id"] for p in r.json()["results"]] == ["a"]

    r = client.get("/query/nearest", params={"latitude": 0, "longitude": 0, "limit": 0})
    assert r.status_code == 422
    r = client.get("/query/nearest", params={"latitude": 0, "longitude": 0, "limit": 99})
    assert r.status_code == 413


def test_point_cap_maps_to_413(client: TestClient) -> None:
    for i in range(5):
        client.post("/bulk", json={"points": [{"id": f"p{i}", "latitude": 0, "longitude": 0}]})
    r = client.put("/points/overflow", json={"latitude": 0, "longitude": 0})
    assert r.status_code == 413
