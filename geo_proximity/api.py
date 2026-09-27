"""FastAPI HTTP layer around :class:`ProximityService`.

Endpoints
---------
* ``GET    /health``
* ``POST   /points``                 insert (``x-on-duplicate`` optional)
* ``PUT    /points/{point_id}``      upsert
* ``PATCH  /points/{point_id}``      update coordinates/attributes
* ``DELETE /points/{point_id}``
* ``GET    /points/{point_id}``
* ``POST   /points/bulk``            bulk load
* ``GET    /query/radius``
* ``GET    /query/nearest``

The same typed service errors are mapped to deterministic HTTP status
codes (400 invalid coordinates/parameters, 404 unknown id, 409 duplicate
id, 413 configured limit exceeded, 422 malformed body).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .errors import (
    DuplicateIdError,
    InvalidCoordinatesError,
    InvalidParameterError,
    LimitExceededError,
    PointNotFoundError,
)
from .models import OnDuplicate, ScoredPoint
from .service import ProximityService


class PointIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    latitude: float
    longitude: float
    attributes: dict[str, Any] = Field(default_factory=dict)


class PointUpdate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    latitude: float | None = None
    longitude: float | None = None
    attributes: dict[str, Any] | None = None
    merge_attributes: bool = False


class BulkIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    points: list[PointIn]
    on_duplicate: OnDuplicate = OnDuplicate.ERROR


def _point_json(point) -> dict[str, Any]:
    return {
        "id": point.id,
        "latitude": point.latitude,
        "longitude": point.longitude,
        "attributes": point.attributes,
    }


def _scored_json(scored: ScoredPoint) -> dict[str, Any]:
    return {
        "point": _point_json(scored.point),
        "distance_m": scored.distance_m,
    }


def create_app(service: ProximityService | None = None) -> FastAPI:
    app = FastAPI(
        title="geo-proximity",
        version="0.1.0",
        description=(
            "Local in-memory great-circle proximity service. Distances "
            "are haversine metres on a sphere of radius 6,371,008.8 m; "
            "radius bounds are inclusive; nearest-neighbour ties break by "
            "point id ascending."
        ),
    )
    svc = service if service is not None else ProximityService()
    app.state.service = svc

    # ---- error mapping -------------------------------------------------
    @app.exception_handler(InvalidCoordinatesError)
    async def _bad_coords(_: Request, exc: InvalidCoordinatesError):
        return JSONResponse(status_code=400,
                            content={"error": "invalid_coordinates",
                                     "detail": str(exc)})

    @app.exception_handler(InvalidParameterError)
    async def _bad_param(_: Request, exc: InvalidParameterError):
        return JSONResponse(status_code=400,
                            content={"error": "invalid_parameter",
                                     "detail": str(exc)})

    @app.exception_handler(DuplicateIdError)
    async def _dup(_: Request, exc: DuplicateIdError):
        return JSONResponse(status_code=409,
                            content={"error": "duplicate_id",
                                     "detail": str(exc)})

    @app.exception_handler(PointNotFoundError)
    async def _not_found(_: Request, exc: PointNotFoundError):
        return JSONResponse(status_code=404,
                            content={"error": "point_not_found",
                                     "detail": str(exc)})

    @app.exception_handler(LimitExceededError)
    async def _limit(_: Request, exc: LimitExceededError):
        return JSONResponse(status_code=413,
                            content={"error": "limit_exceeded",
                                     "detail": str(exc)})

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422,
                            content={"error": "validation_error",
                                     "detail": exc.errors()})

    # ---- health / config ----------------------------------------------
    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "points": len(svc)}

    @app.get("/config")
    async def config() -> dict[str, Any]:
        c = svc.config
        return {
            "max_points": c.max_points,
            "max_query_radius_m": c.max_query_radius_m,
            "max_neighbors": c.max_neighbors,
            "index_levels": c.index_levels,
            "max_batch_size": c.max_batch_size,
            "distance": "haversine_m",
            "earth_radius_m": 6_371_008.8,
            "radius_bound": "inclusive (<=)",
            "tie_breaker": "point_id_ascending",
        }

    # ---- point CRUD ----------------------------------------------------
    @app.post("/points", status_code=status.HTTP_201_CREATED)
    async def create_point(
        body: PointIn,
        on_duplicate: OnDuplicate = Query(
            default=OnDuplicate.ERROR,
            description="error | replace | skip",
        ),
    ) -> dict[str, Any]:
        stored = svc.insert(
            (body.id, body.latitude, body.longitude, body.attributes),
            on_duplicate=on_duplicate,
        )
        point = svc.get(body.id)
        return {"stored": stored, "point": _point_json(point)}

    @app.put("/points/{point_id}")
    async def upsert_point(point_id: str, body: PointIn) -> dict[str, Any]:
        if body.id != point_id:
            raise InvalidParameterError(
                "path point_id and body id must match"
            )
        svc.upsert((body.id, body.latitude, body.longitude, body.attributes))
        return {"point": _point_json(svc.get(point_id))}

    @app.patch("/points/{point_id}")
    async def patch_point(point_id: str, body: PointUpdate) -> dict[str, Any]:
        point = svc.update(
            point_id,
            latitude=body.latitude,
            longitude=body.longitude,
            attributes=body.attributes,
            merge_attributes=body.merge_attributes,
        )
        return {"point": _point_json(point)}

    @app.delete("/points/{point_id}")
    async def delete_point(point_id: str) -> dict[str, Any]:
        deleted = svc.delete(point_id)
        return {"deleted": deleted}

    @app.get("/points/{point_id}")
    async def get_point(point_id: str) -> dict[str, Any]:
        point = svc.get(point_id)
        if point is None:
            raise PointNotFoundError(f"no such point id: {point_id!r}")
        return {"point": _point_json(point)}

    # ---- bulk ----------------------------------------------------------
    @app.post("/points/bulk", status_code=status.HTTP_201_CREATED)
    async def bulk(body: BulkIn) -> dict[str, Any]:
        stored = svc.bulk_load(
            [(p.id, p.latitude, p.longitude, p.attributes) for p in body.points],
            on_duplicate=body.on_duplicate,
        )
        return {"stored": stored, "points": len(svc)}

    # ---- queries -------------------------------------------------------
    @app.get("/query/radius")
    async def query_radius(
        latitude: float = Query(..., ge=-90, le=90),
        longitude: float = Query(...),
        radius_m: float = Query(..., ge=0),
        limit: int | None = Query(default=None, ge=1),
    ) -> dict[str, Any]:
        results = svc.radius_query(
            latitude, longitude, radius_m, limit=limit
        )
        return {
            "query": {"latitude": latitude, "longitude": longitude,
                      "radius_m": radius_m},
            "count": len(results),
            "results": [_scored_json(s) for s in results],
        }

    @app.get("/query/nearest")
    async def query_nearest(
        latitude: float = Query(..., ge=-90, le=90),
        longitude: float = Query(...),
        limit: int = Query(..., ge=1),
        max_distance_m: float | None = Query(default=None, ge=0),
    ) -> dict[str, Any]:
        results = svc.nearest(
            latitude, longitude, limit, max_distance_m=max_distance_m
        )
        return {
            "query": {"latitude": latitude, "longitude": longitude,
                      "limit": limit, "max_distance_m": max_distance_m},
            "count": len(results),
            "results": [_scored_json(s) for s in results],
        }

    return app
