"""Alembic environment.

Pulls the database URL from `pycaret_server.config.get_settings()` and the
metadata object from `pycaret_server.db.Base`, so new tables auto-appear in
``alembic revision --autogenerate``.

Override the URL for one-off ops (e.g. running migrations against a staging
DB without touching env vars) by setting ``ALEMBIC_URL`` in the environment.
"""

from __future__ import annotations

import os

from alembic import context
from sqlalchemy import engine_from_config, pool

from pycaret_server.config import get_settings
from pycaret_server.db import Base

# Alembic Config object, giving access to values in .ini.
config = context.config

# Runtime-resolved URL: env override > settings.
url = os.environ.get("ALEMBIC_URL") or get_settings().database_url
config.set_main_option("sqlalchemy.url", url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations without a live DB connection (emit SQL scripts)."""
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against the live DB."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            # SQLite needs batch mode for any ALTER TABLE we might autogen.
            render_as_batch=url.startswith("sqlite"),
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
