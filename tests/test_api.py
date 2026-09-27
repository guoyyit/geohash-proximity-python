"""HTTP layer: endpoints, status codes and error mapping."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from geo_proximity import ProximityService, ServiceConfig  # noqa: E402
from geo_proximity.api import create_app  # noqa: E402


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


def test_health_and_config(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["points"] == 0
    cfg = client.get("/config").json()
    assert cfg["distance"] == "haversine_m"
    assert cfg["radius_bound"] == "inclusive (<=)"
    assert cfg["tie_breaker"] == "point_id_ascending"
    assert cfg["earth_radius_m"] == 6_371_008.8


def test_full_point_lifecycle(client: TestClient) -> None:
    r = client.post("/points", json={"id": "a", "latitude": 39.9,
                                     "longitude": 116.4, "attributes": {}})
    assert r.status_code == 201 and r.json()["stored"] is True

    # duplicate -> 409
    assert client.post("/points", json={"id": "a", "latitude": 1,
                                        "longitude": 1}).status_code == 409
    # skip -> 201 stored false
    r = client.post("/points?on_duplicate=skip",
                    json={"id": "a", "latitude": 1, "longitude": 1})
    assert r.status_code == 201 and r.json()["stored"] is False
    assert client.get("/points/a").json()["point"]["latitude"] == 39.9

    # put (upsert) with mismatched id -> 400
    assert client.put("/points/a", json={"id": "z", "latitude": 1,
                                         "longitude": 1}).status_code == 400
    assert client.put("/points/a", json={"id": "a", "latitude": 40.0,
                                         "longitude": 116.0}).status_code == 200
    # patch replace vs merge
    client.patch("/points/a", json={"attributes": {"x": 1}})
    client.patch("/points/a", json={"attributes": {"y": 2},
                                    "merge_attributes": True})
    assert client.get("/points/a").json()["point"]["attributes"] == \
        {"x": 1, "y": 2}

    assert client.delete("/points/a").json()["deleted"] is True
    assert client.get("/points/a").status_code == 404
    assert client.delete("/points/a").json()["deleted"] is False


def test_bulk_endpoint(client: TestClient) -> None:
    r = client.post("/points/bulk", json={"points": [
        {"id": "a", "latitude": 0.0, "longitude": 0.0},
        {"id": "b", "latitude": 0.0, "longitude": 1.0},
    ], "on_duplicate": "replace"})
    assert r.status_code == 201 and r.json()["stored"] == 2
    # invalid coordinates in batch -> 400
    r = client.post("/points/bulk", json={"points": [
        {"id": "c", "latitude": 91.0, "longitude": 0.0},
    ]})
    assert r.status_code == 400


def test_radius_and_nearest_queries(client: TestClient) -> None:
    client.post("/points/bulk", json={"points": [
        {"id": "seam_e", "latitude": 0.0, "longitude": 179.99},
        {"id": "seam_w", "latitude": 0.0, "longitude": -179.99},
        {"id": "c", "latitude": 0.0, "longitude": 0.0},
    ]})
    r = client.get("/query/radius",
                   params={"latitude": 0, "longitude": 180,
                           "radius_m": 200_000})
    assert r.status_code == 200
    ids = [x["point"]["id"] for x in r.json()["results"]]
    assert ids == ["seam_e", "seam_w"]  # tie -> id ascending
    assert "c" not in ids

    r = client.get("/query/nearest",
                   params={"latitude": 0, "longitude": 0, "limit": 1})
    assert r.json()["results"][0]["point"]["id"] == "c"
    assert r.json()["count"] == 1

    # exhausted limit is fine
    r = client.get("/query/nearest",
                   params={"latitude": 0, "longitude": 0, "limit": 99})
    assert r.json()["count"] == 3


def test_longitude_out_of_range_is_wrapped(client: TestClient) -> None:
    r = client.post("/points", json={"id": "w", "latitude": 0,
                                     "longitude": 181.0})
    assert r.status_code == 201
    assert client.get("/points/w").json()["point"]["longitude"] == -179.0


def test_http_parameter_and_limit_errors() -> None:
    app = create_app(ProximityService(ServiceConfig(
        max_points=1, max_neighbors=1, max_batch_size=10)))
    c = TestClient(app)
    c.post("/points", json={"id": "x", "latitude": 0, "longitude": 0})

    # framework-level validation: out-of-range query params
    assert c.get("/query/radius",
                 params={"latitude": 91, "longitude": 0,
                         "radius_m": 1}).status_code == 422
    assert c.get("/query/nearest",
                 params={"latitude": 0, "longitude": 0,
                         "limit": 0}).status_code == 422
    assert c.get("/query/radius",
                 params={"latitude": 0, "longitude": 0,
                         "radius_m": -1}).status_code == 422
    # malformed body
    assert c.post("/points", json={"id": "y", "latitude": "north",
                                   "longitude": 0}).status_code == 422
    # configured limits -> 413
    assert c.post("/points", json={"id": "y", "latitude": 0,
                                   "longitude": 0}).status_code == 413
    assert c.get("/query/nearest",
                 params={"latitude": 0, "longitude": 0,
                         "limit": 2}).status_code == 413
