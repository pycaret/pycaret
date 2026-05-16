"""Session 32 — revert Trial/Run hierarchy + per-Trial lifecycle

Revision ID: d4e5f6a8b9c0
Revises: c3d4e5f6a8b9
Create Date: 2026-05-16 12:00:00.000000+00:00

Phase 0 (revision ``f0a1b2c3d4e5``) flipped the model so that ``Trial``
owned ``Runs``. After hands-on use, we're flipping it back to the
PyCaret 3.x convention that matches both user mental model and the
``create_model``-per-trial parallel-dispatch architecture:

    Run ──▶ contains 1..N Trials (one per algorithm candidate)
    Trial = one fitted pipeline with its own status / metrics / artifact

The key win from this revert is that each Trial gets its OWN status
lifecycle (``queued → running → succeeded|failed|cancelled``) — that's
what lets the orchestrator decompose a ``compare`` Run into N
``create_model`` Jobs, dispatch them to workers in parallel, and have
the UI render rows lighting up as workers complete.

What we keep from Phase 0 (still good):
  - ``Trial.kind`` (compare | tuned | ensembled | blended | stacked | manual)
  - ``Trial.parent_trial_ids`` (lineage for tune / blend / stack)
  - ``Trial.created_by_action_id`` (groups trials born from one click)
  - ``Trial.experiment_id`` (denorm for experiment-scoped listing)
  - ``Trial.name``, ``Trial.notes``, ``Trial.fitted_pipeline_id``

What we flip back:
  - ``trials`` regains ``run_id``, ``rank``, ``is_best``, ``metrics``,
    ``stored_path``, ``sha256``, ``size_bytes``, ``params``.
  - ``trials`` gains NEW ``status`` / ``started_at`` / ``finished_at``
    / ``duration_ms`` / ``error`` — these didn't exist before; they
    only make sense once each Trial is its own scheduled unit.
  - ``runs`` loses ``trial_id``, ``sequence``, ``metrics``,
    ``stored_path``, ``sha256``, ``size_bytes``, ``params``,
    ``triggered_by``, ``triggered_by_id``. Run is back to "the
    user-visible event"; per-Trial detail lives on the Trial row.

The dev DB has no production data — this migration is destructive for
any in-flight Phase 0 rows. Re-bootstrap via
``pycaret-server migrate --reset-dev``.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "d4e5f6a8b9c0"
down_revision: str | None = "c3d4e5f6a8b9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ─── trials: restore Phase 0 fields + add per-Trial status. ──────
    with op.batch_alter_table("trials", schema=None) as batch_op:
        batch_op.add_column(sa.Column("run_id", sa.String(length=36), nullable=True))
        batch_op.create_foreign_key(
            "fk_trials_run_id",
            "runs",
            ["run_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch_op.create_index("ix_trials_run_id", ["run_id"], unique=False)

        batch_op.add_column(sa.Column("rank", sa.Integer(), nullable=True))
        batch_op.create_index("ix_trials_rank", ["rank"], unique=False)
        batch_op.add_column(
            sa.Column(
                "is_best",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("0"),
            )
        )
        batch_op.add_column(sa.Column("metrics", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("stored_path", sa.String(length=1024), nullable=True))
        batch_op.add_column(sa.Column("sha256", sa.String(length=64), nullable=True))
        batch_op.create_index("ix_trials_sha256", ["sha256"], unique=False)
        batch_op.add_column(sa.Column("size_bytes", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("params", sa.JSON(), nullable=True))

        # Per-Trial lifecycle — new in Phase 0-v2.
        batch_op.add_column(
            sa.Column(
                "status",
                sa.String(length=16),
                nullable=False,
                server_default="succeeded",
                index=True,
            )
        )
        batch_op.add_column(sa.Column("started_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("duration_ms", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("error", sa.Text(), nullable=True))
        batch_op.create_index("ix_trials_status", ["status"], unique=False)

    # ─── runs: drop the Phase 0 fields that moved back to Trial. ─────
    with op.batch_alter_table("runs", schema=None) as batch_op:
        batch_op.drop_index("ix_runs_trial_id")
        batch_op.drop_constraint("fk_runs_trial_id", type_="foreignkey")
        batch_op.drop_column("trial_id")
        batch_op.drop_column("sequence")
        batch_op.drop_column("metrics")
        batch_op.drop_column("stored_path")
        batch_op.drop_index("ix_runs_sha256")
        batch_op.drop_column("sha256")
        batch_op.drop_column("size_bytes")
        batch_op.drop_column("params")
        batch_op.drop_column("triggered_by")
        batch_op.drop_column("triggered_by_id")


def downgrade() -> None:
    # Reverses to the Phase 0-v1 shape. Data loss for any new Trial
    # status / timing fields that weren't part of the old schema.
    with op.batch_alter_table("runs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("trial_id", sa.String(length=36), nullable=True))
        batch_op.create_foreign_key(
            "fk_runs_trial_id",
            "trials",
            ["trial_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch_op.create_index("ix_runs_trial_id", ["trial_id"], unique=False)
        batch_op.add_column(sa.Column("sequence", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("metrics", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("stored_path", sa.String(length=1024), nullable=True))
        batch_op.add_column(sa.Column("sha256", sa.String(length=64), nullable=True))
        batch_op.create_index("ix_runs_sha256", ["sha256"], unique=False)
        batch_op.add_column(sa.Column("size_bytes", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("params", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("triggered_by", sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column("triggered_by_id", sa.String(length=36), nullable=True))

    with op.batch_alter_table("trials", schema=None) as batch_op:
        batch_op.drop_index("ix_trials_status")
        batch_op.drop_column("error")
        batch_op.drop_column("duration_ms")
        batch_op.drop_column("finished_at")
        batch_op.drop_column("started_at")
        batch_op.drop_column("status")
        batch_op.drop_column("params")
        batch_op.drop_column("size_bytes")
        batch_op.drop_index("ix_trials_sha256")
        batch_op.drop_column("sha256")
        batch_op.drop_column("stored_path")
        batch_op.drop_column("metrics")
        batch_op.drop_column("is_best")
        batch_op.drop_index("ix_trials_rank")
        batch_op.drop_column("rank")
        batch_op.drop_index("ix_trials_run_id")
        batch_op.drop_constraint("fk_trials_run_id", type_="foreignkey")
        batch_op.drop_column("run_id")
