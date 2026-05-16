"""session 27: trial kind + parent_trial_ids (follow-on action lineage)

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-05-11 20:00:00.000000+00:00

Trials gain two columns:

- ``kind`` — what the trial *is* (``compare``, ``tuned``, ``ensembled``,
  ``blended``, ``stacked``, or ``manual``). Defaults to ``compare`` so every
  pre-existing row reads as a regular compare candidate.
- ``parent_trial_ids`` — JSON array of source trial ids for follow-on
  actions. NULL for ``compare`` rows; ``[X]`` for tune/ensemble; ``[X,Y,Z]``
  for blend/stack.

This lets tune/ensemble/blend/stack live as new trials in the same Run,
labelled by kind and back-linked to their sources — no extra Run layer,
no extra navigation. Lineage queryable in SQL via JSON ops.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "e5f6a7b8c9d0"
down_revision: str | None = "d4e5f6a7b8c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("trials", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "kind",
                sa.String(length=16),
                nullable=False,
                server_default="compare",
            )
        )
        batch_op.add_column(sa.Column("parent_trial_ids", sa.JSON(), nullable=True))
        batch_op.create_index(batch_op.f("ix_trials_kind"), ["kind"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("trials", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_trials_kind"))
        batch_op.drop_column("parent_trial_ids")
        batch_op.drop_column("kind")
