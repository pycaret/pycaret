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
    api_keys,
    audit,
    auth,
    data_sources,
    deployments,
    describe,
    drift,
    experiments,
    llm,
    members,
    plots,
    projects,
    runs,
    setup,
    workspaces,
)
from pycaret_server.audit import AuditLogMiddleware
from pycaret_server.config import get_settings
from pycaret_server.db import Base, engine
from pycaret_server.db.bootstrap import ensure_schema
from pycaret_server.llm.router import reset_router as reset_llm_router
from pycaret_server.runs.orchestrator import reset_orchestrator
from pycaret_server.serving import reset_registry


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Lifespan hook: make sure the schema exists, then yield.

    Strategy:
    - If the DB already has the alembic_version table, no-op — the operator
      is managing migrations manually.
    - Else, if this is SQLite (dev / CI), apply the Alembic baseline. That
      keeps local dev one-command (`pycaret-server serve`) while still routing
      every schema change through a proper migration.
    - Else (Postgres / MySQL in production), fail loudly: explicit
      `alembic upgrade head` is required so schema drift is never silent.
    """
    settings = get_settings()
    ensure_schema(engine, dev_auto_migrate=settings.database_url.startswith("sqlite"))
    # Side-channel for tests + factory reuse: keep Base.metadata as a known-good
    # target in case an alternate runner wants to bypass Alembic entirely.
    _ = Base

    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield
    finally:
        # Tear down the run-orchestrator singleton so worker threads stop,
        # drop the in-memory deployment registry so cached pipelines don't
        # carry across processes (matters in reload mode), and reset the LLM
        # router so providers get rebuilt from fresh settings next boot.
        reset_orchestrator()
        reset_registry()
        reset_llm_router()


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

    # Audit-log middleware records POST/PATCH/PUT/DELETE on /api/v1/*.
    # Added here so every route is covered. See pycaret_server.audit.
    app.add_middleware(AuditLogMiddleware)

    # Mount all /api/v1/* routers.
    for router in (
        setup.router,
        auth.router,
        api_keys.router,
        describe.router,
        workspaces.router,
        members.router,
        projects.router,
        experiments.router,
        runs.router,
        data_sources.router,
        deployments.router,
        drift.router,
        llm.router,
        audit.router,
        plots.router,
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
