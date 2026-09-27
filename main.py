"""Entry point: ``uv run uvicorn main:app``."""
from geohash_proximity.api import app

__all__ = ["app"]
