"""Optional FastAPI HTTP layer wrapping a single :class:`ProximityIndex`.

The app owns one global index (configurable via :func:`create_app`).  All
mutations and queries go through the same thread-safe index, so concurrent
HTTP requests observe the same consistency guarantees as direct library
use.  Error mapping:

* ``InvalidCoordinateError`` / ``InvalidParameterError`` -> HTTP 422
* ``DuplicatePointError``                                -> HTTP 409
* ``PointNotFoundError``                                 -> HTTP 404
* ``LimitExceededError``                                 -> HTTP 413
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import IndexConfig
from .errors import (
    DuplicatePointError,
    InvalidCoordinateError,
    InvalidParameterError,
    LimitExceededError,
    PointNotFoundError,
)
from .index import ProximityIndex
from .models import Point, ScoredPoint


class PointBody(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    attributes: dict[str, Any] = Field(default_factory=dict)


class BulkPointBody(BaseModel):
    id: str = Field(..., min_length=1)
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    attributes: dict[str, Any] = Field(default_factory=dict)


class UpdateBody(BaseModel):
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    attributes: dict[str, Any] | None = None


class BulkBody(BaseModel):
    points: list[BulkPointBody] = Field(default_factory=list)
    on_duplicate: str = "replace"


def _scored_dict(sp: ScoredPoint) -> dict[str, Any]:
    return {
        "id": sp.point.id,
        "latitude": sp.point.latitude,
        "longitude": sp.point.longitude,
        "attributes": sp.point.attributes,
        "distance": sp.distance,
    }


def _point_dict(point: Point) -> dict[str, Any]:
    return {
        "id": point.id,
        "latitude": point.latitude,
        "longitude": point.longitude,
        "attributes": point.attributes,
    }


def create_app(config: IndexConfig | None = None) -> FastAPI:
    """Build a FastAPI application bound to a fresh proximity index."""
    index = ProximityIndex(config)
    app = FastAPI(title="geohash-proximity", version="0.1.0")

    @app.exception_handler(InvalidCoordinateError)
    @app.exception_handler(InvalidParameterError)
    async def _invalid(_request, exc):
        return JSONResponse(status_code=422, content={"error": str(exc)})

    @app.exception_handler(DuplicatePointError)
    async def _conflict(_request, exc):
        return JSONResponse(status_code=409, content={"error": str(exc)})

    @app.exception_handler(PointNotFoundError)
    async def _not_found(_request, exc):
        return JSONResponse(status_code=404, content={"error": str(exc)})

    @app.exception_handler(LimitExceededError)
    async def _payload_too_large(_request, exc):
        return JSONResponse(status_code=413, content={"error": str(exc)})

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "points": len(index)}

    @app.put("/points/{point_id}", status_code=201)
    def insert_point(point_id: str, body: PointBody) -> dict[str, Any]:
        point = index.insert(
            point_id, body.latitude, body.longitude, body.attributes
        )
        return _point_dict(point)

    @app.post("/points/{point_id}")
    def upsert_point(point_id: str, body: PointBody) -> dict[str, Any]:
        point = index.upsert(
            point_id, body.latitude, body.longitude, body.attributes
        )
        return _point_dict(point)

    @app.patch("/points/{point_id}")
    def update_point(point_id: str, body: UpdateBody) -> dict[str, Any]:
        point = index.update(
            point_id, body.latitude, body.longitude, body.attributes
        )
        return _point_dict(point)

    @app.delete("/points/{point_id}", status_code=204)
    def delete_point(point_id: str) -> None:
        index.delete(point_id)

    @app.get("/points/{point_id}")
    def get_point(point_id: str) -> dict[str, Any]:
        point = index.get(point_id)
        if point is None:
            raise PointNotFoundError(f"no such point id: {point_id!r}")
        return _point_dict(point)

    @app.post("/bulk")
    def bulk_load(body: BulkBody) -> dict[str, Any]:
        points = [
            Point(
                p.id,
                p.latitude,
                p.longitude,
                dict(p.attributes),
            )
            for p in body.points
        ]
        written = index.bulk_load(points, on_duplicate=body.on_duplicate)
        return {"written": written, "total": len(index)}

    @app.get("/query/radius")
    def radius_query(
        latitude: float,
        longitude: float,
        radius_m: float,
    ) -> dict[str, Any]:
        results = index.radius_query(latitude, longitude, radius_m)
        return {"results": [_scored_dict(sp) for sp in results]}

    @app.get("/query/nearest")
    def nearest_query(
        latitude: float,
        longitude: float,
        limit: int = 10,
    ) -> dict[str, Any]:
        results = index.nearest(latitude, longitude, limit)
        return {"results": [_scored_dict(sp) for sp in results]}

    return app


app = create_app()
