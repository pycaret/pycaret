"""Declarative base + reusable mixins.

Every table gets UUID v4 `id` + `created_at` / `updated_at`. Most get
`created_by` as an FK to `users.id` — that's added per-model rather than in a
mixin because the FK is awkward to express before `User` is defined.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _now() -> datetime:
    """UTC timestamp with timezone — SQLAlchemy's func.now() is backend-specific."""
    return datetime.now(UTC)


def _uuid_str() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    """Declarative base for all PyCaret server models."""


class UUIDMixin:
    """Adds a primary key ``id`` column of type UUID-string (36 chars)."""

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)


class TimestampMixin:
    """Adds ``created_at`` and ``updated_at`` columns with timezone."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_now,
        onupdate=_now,
        nullable=False,
    )
