"""HTTP entry point: ``uvicorn main:app``."""

from geo_proximity.api import create_app

app = create_app()
