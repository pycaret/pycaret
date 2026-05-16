"""session 25: trial artifacts (stored_path + sha256 + size_bytes + params)

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-05-08 20:00:00.000000+00:00

Promotes ``Trial`` from a metrics-only row to a full candidate-pipeline
record. Each compare/search trial now keeps its fitted pipeline pickle
on disk + extracted estimator params, so the UI's model-detail page
can render params, offer a download, and promote any trial directly.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "c3d4e5f6a7b8"
down_revision: str | None = "b2c3d4e5f6a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("trials", schema=None) as batch_op:
        batch_op.add_column(sa.Column("stored_path", sa.String(length=1024), nullable=True))
        batch_op.add_column(sa.Column("sha256", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("size_bytes", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("params", sa.JSON(), nullable=True))
        batch_op.create_index(batch_op.f("ix_trials_sha256"), ["sha256"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("trials", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_trials_sha256"))
        batch_op.drop_column("params")
        batch_op.drop_column("size_bytes")
        batch_op.drop_column("sha256")
        batch_op.drop_column("stored_path")
