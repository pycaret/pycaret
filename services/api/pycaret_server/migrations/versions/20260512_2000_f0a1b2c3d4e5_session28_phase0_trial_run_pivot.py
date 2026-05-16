"""Phase 0: pivot Trial/Run model — Trial owns Runs, not the other way around

Revision ID: f0a1b2c3d4e5
Revises: e5f6a7b8c9d0
Create Date: 2026-05-12 20:00:00.000000+00:00

The platform's mental model flips here:

- Pre-Phase 0: ``Run`` owned the compare invocation; ``Trial`` was a row
  inside a Run's leaderboard.
- Phase 0: ``Trial`` is a *logical pipeline candidate* (LR, Tuned XGBoost,
  Stacked Ensemble) that lives inside an Experiment. Each Trial owns 1..N
  ``Run`` rows — one per execution (initial training, retrain on fresh
  data, drift-triggered retrain, ...).

Schema deltas
-------------

trials
  - DROP ``run_id`` FK, ``rank``, ``metrics``, ``is_best``,
    ``stored_path``, ``sha256``, ``size_bytes``, ``params``
  - ADD ``experiment_id`` FK → experiments (nullable; enforced at app
    layer once the dev DB is reset)
  - ADD ``name`` (user-facing trial label)
  - ADD ``created_by_action_id`` (groups trials spawned by one user click
    such as a compare batch)

runs
  - ADD ``trial_id`` FK → trials (nullable; enforced going forward)
  - ADD ``sequence`` (1-based per-trial run number)
  - ADD ``metrics`` (aggregated per-run metrics JSON)
  - ADD ``stored_path``, ``sha256``, ``size_bytes``, ``params``
    (artifact + estimator-params live on the Run from now on)
  - ADD ``triggered_by`` / ``triggered_by_id`` (user / schedule / drift /
    api source attribution)

deployments
  - ADD ``trial_id`` FK → trials
  - ADD ``run_id`` FK → runs
  (both nullable in DB, required by the API on create — Phase 7 collapses
  pipeline_id into a registered-model reference, the trial+run pair stays
  for exact-execution reproducibility)

Migration is destructive for ``trials`` row content (the dropped columns).
The dev DB has no production data; ``pycaret-server migrate --reset-dev``
provides a clean wipe path. A separate forward migration will be written
when the first production install needs it.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "f0a1b2c3d4e5"
down_revision: str | None = "e5f6a7b8c9d0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ─── trials ────────────────────────────────────────────────────
    with op.batch_alter_table("trials", schema=None) as batch_op:
        # New: link to Experiment (was via Run → experiment_id).
        batch_op.add_column(
            sa.Column(
                "experiment_id",
                sa.String(length=36),
                nullable=True,
            )
        )
        batch_op.create_foreign_key(
            "fk_trials_experiment_id",
            "experiments",
            ["experiment_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch_op.create_index(
            "ix_trials_experiment_id", ["experiment_id"], unique=False
        )
        batch_op.add_column(sa.Column("name", sa.String(length=128), nullable=True))
        batch_op.add_column(
            sa.Column(
                "created_by_action_id",
                sa.String(length=36),
                nullable=True,
            )
        )
        batch_op.create_index(
            "ix_trials_created_by_action_id",
            ["created_by_action_id"],
            unique=False,
        )
        # Drop the columns that move to Run (or no longer make sense).
        # Indexes are dropped explicitly so Postgres rejects-this-cleanly
        # if a future hot-fix re-adds the columns (SQLite's batch mode
        # would handle it transparently, but Postgres won't).
        batch_op.drop_index("ix_trials_sha256")
        batch_op.drop_index("ix_trials_rank")
        batch_op.drop_index("ix_trials_run_id")
        batch_op.drop_column("rank")
        batch_op.drop_column("metrics")
        batch_op.drop_column("is_best")
        batch_op.drop_column("stored_path")
        batch_op.drop_column("sha256")
        batch_op.drop_column("size_bytes")
        batch_op.drop_column("params")
        # The legacy FK back to Run.
        batch_op.drop_column("run_id")

    # ─── runs ──────────────────────────────────────────────────────
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
        batch_op.add_column(
            sa.Column("stored_path", sa.String(length=1024), nullable=True)
        )
        batch_op.add_column(sa.Column("sha256", sa.String(length=64), nullable=True))
        batch_op.create_index("ix_runs_sha256", ["sha256"], unique=False)
        batch_op.add_column(sa.Column("size_bytes", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("params", sa.JSON(), nullable=True))
        batch_op.add_column(
            sa.Column("triggered_by", sa.String(length=32), nullable=True)
        )
        batch_op.add_column(
            sa.Column("triggered_by_id", sa.String(length=36), nullable=True)
        )

    # ─── deployments ───────────────────────────────────────────────
    with op.batch_alter_table("deployments", schema=None) as batch_op:
        batch_op.add_column(sa.Column("trial_id", sa.String(length=36), nullable=True))
        batch_op.create_foreign_key(
            "fk_deployments_trial_id",
            "trials",
            ["trial_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index(
            "ix_deployments_trial_id", ["trial_id"], unique=False
        )
        batch_op.add_column(sa.Column("run_id", sa.String(length=36), nullable=True))
        batch_op.create_foreign_key(
            "fk_deployments_run_id",
            "runs",
            ["run_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index(
            "ix_deployments_run_id", ["run_id"], unique=False
        )


def downgrade() -> None:
    # Best-effort reverse — pre-Phase 0 trials had ``run_id``, ``rank``,
    # ``metrics``, ``is_best``, ``stored_path``, ``sha256``, ``size_bytes``,
    # ``params``. The data in those columns is lost on the way down.
    with op.batch_alter_table("deployments", schema=None) as batch_op:
        batch_op.drop_index("ix_deployments_run_id")
        batch_op.drop_constraint("fk_deployments_run_id", type_="foreignkey")
        batch_op.drop_column("run_id")
        batch_op.drop_index("ix_deployments_trial_id")
        batch_op.drop_constraint("fk_deployments_trial_id", type_="foreignkey")
        batch_op.drop_column("trial_id")

    with op.batch_alter_table("runs", schema=None) as batch_op:
        batch_op.drop_column("triggered_by_id")
        batch_op.drop_column("triggered_by")
        batch_op.drop_column("params")
        batch_op.drop_column("size_bytes")
        batch_op.drop_index("ix_runs_sha256")
        batch_op.drop_column("sha256")
        batch_op.drop_column("stored_path")
        batch_op.drop_column("metrics")
        batch_op.drop_column("sequence")
        batch_op.drop_index("ix_runs_trial_id")
        batch_op.drop_constraint("fk_runs_trial_id", type_="foreignkey")
        batch_op.drop_column("trial_id")

    with op.batch_alter_table("trials", schema=None) as batch_op:
        # Re-add columns dropped in upgrade with permissive defaults so the
        # rollback works on a Phase 0 schema; the data is gone, but the
        # shape is back.
        batch_op.add_column(sa.Column("run_id", sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column("rank", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("metrics", sa.JSON(), nullable=True))
        batch_op.add_column(
            sa.Column(
                "is_best",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("0"),
            )
        )
        batch_op.add_column(
            sa.Column("stored_path", sa.String(length=1024), nullable=True)
        )
        batch_op.add_column(sa.Column("sha256", sa.String(length=64), nullable=True))
        batch_op.create_index("ix_trials_sha256", ["sha256"], unique=False)
        batch_op.add_column(sa.Column("size_bytes", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("params", sa.JSON(), nullable=True))
        batch_op.create_index("ix_trials_rank", ["rank"], unique=False)
        batch_op.create_index("ix_trials_run_id", ["run_id"], unique=False)
        batch_op.drop_index("ix_trials_created_by_action_id")
        batch_op.drop_column("created_by_action_id")
        batch_op.drop_column("name")
        batch_op.drop_index("ix_trials_experiment_id")
        batch_op.drop_constraint("fk_trials_experiment_id", type_="foreignkey")
        batch_op.drop_column("experiment_id")
