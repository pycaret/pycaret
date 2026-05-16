"""session 26: trial notes (free-form annotation)

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-05-09 10:00:00.000000+00:00

Adds a free-form ``notes`` column on ``trials`` so users can annotate
candidates straight from the trial-detail page. Optional, nullable —
existing rows just stay empty until the user types something.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "d4e5f6a7b8c9"
down_revision: str | None = "c3d4e5f6a7b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("trials", schema=None) as batch_op:
        batch_op.add_column(sa.Column("notes", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("trials", schema=None) as batch_op:
        batch_op.drop_column("notes")
