"""FastAPI app factory.

Usage:
    from pycaret_server.app import create_app
    app = create_app()
    # uvicorn pycaret_server.app:create_app --factory --reload

Or via the CLI:
    pycaret-server serve --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pycaret_server import __version__
from pycaret_server.api import (
    auth,
    describe,
    experiments,
    projects,
    runs,
    setup,
    workspaces,
)
from pycaret_server.config import get_settings
from pycaret_server.db import Base, engine
from pycaret_server.runs.orchestrator import reset_orchestrator


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Lifespan hook: create tables on first run if missing, then yield.

    For v1 (single-process, SQLite default), Base.metadata.create_all gives
    us a working schema from a blank database in ~10ms. Production deploys
    should run ``alembic upgrade head`` instead.
    """
    settings = get_settings()
    if settings.database_url.startswith("sqlite"):
        Base.metadata.create_all(engine)
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield
    finally:
        # Tear down the run-orchestrator singleton so worker threads stop.
        reset_orchestrator()


def create_app() -> FastAPI:
    """Build and return a configured FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "PyCaret 4.0 application-platform backend. "
            "See https://github.com/pycaret/pycaret/blob/v4/docs/revamp/PLATFORM_PLAN.md"
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=_lifespan,
    )

    # CORS for the React UI dev server. In prod, set PYCARET_CORS_ORIGINS.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount all /api/v1/* routers.
    for router in (
        setup.router,
        auth.router,
        describe.router,
        workspaces.router,
        projects.router,
        experiments.router,
        runs.router,
    ):
        app.include_router(router, prefix="/api/v1")

    # Health + version endpoints at root.
    @app.get("/", tags=["meta"])
    def root() -> dict:
        return {
            "app": settings.app_name,
            "version": __version__,
            "docs": "/docs",
            "openapi": "/openapi.json",
        }

    @app.get("/healthz", tags=["meta"])
    def healthz() -> dict:
        return {"ok": True}

    return app


# Uvicorn entry point for `uvicorn pycaret_server.app:app`.
# Prefer `--factory` + `create_app` for test-friendliness.
app = create_app()
