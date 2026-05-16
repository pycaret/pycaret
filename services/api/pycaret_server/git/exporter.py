"""Pure serialisers — ORM row → wire-friendly YAML/JSON dict.

These functions don't talk to Git or the filesystem. They're called by
both the live "publish to Git" service AND the standalone CLI export
(``pycaret-server export <project>``, a future cut), so keeping them
side-effect-free matters.

The on-disk layout each export produces (Phase 0-v2: Run ▶ Trials):

    <repo>/<path_prefix>/
        experiments/
            <experiment_name>/
                experiment.yaml
                runs/
                    <run_id>/
                        run.yaml
                        leaderboard.json
                        trials/
                            <trial_name>/
                                trial.yaml
                                metrics.json
                                params.json
                                artifact.pointer.yaml   # uri only

The manifests carry URIs, never raw bytes — the artifact lives on
object storage (Phase 2) and the YAML points at it.
"""

from __future__ import annotations

from typing import Any

from pycaret_server.db import Experiment, Run, Trial


def serialise_experiment(exp: Experiment) -> dict[str, Any]:
    """Return the wire shape for ``experiment.yaml``."""
    return {
        "id": exp.id,
        "name": exp.name,
        "task": exp.task,
        "target": exp.target,
        "setup_params": dict(exp.setup_params or {}),
        "data_source_id": exp.data_source_id,
        "description": exp.description,
        "created_at": exp.created_at.isoformat() if exp.created_at else None,
        "created_by": exp.created_by,
    }


def serialise_run(run: Run) -> dict[str, Any]:
    """Return the wire shape for ``run.yaml``.

    The Run is now the dispatch *event* — leaderboard + status flow
    out of its Trials, but the metadata stays on the Run itself.
    """
    return {
        "id": run.id,
        "experiment_id": run.experiment_id,
        "status": run.status,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "duration_ms": run.duration_ms,
        "error": run.error,
        "snapshot": dict(run.snapshot or {}),
        "metrics_summary": dict(run.metrics_summary or {}),
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "created_by": run.created_by,
    }


def serialise_trial(trial: Trial) -> dict[str, dict[str, Any]]:
    """Return the file-bundle for one Trial's directory.

    The caller writes each key as its own file:

    - ``metadata`` → ``trial.yaml``
    - ``metrics`` → ``metrics.json``
    - ``params`` → ``params.json``
    - ``artifact`` → ``artifact.pointer.yaml``

    Splitting them keeps Git diffs informative — a hyperparameter tweak
    shows as a one-line change in ``params.json`` rather than the whole
    Trial blob churning.
    """
    metadata = {
        "id": trial.id,
        "run_id": trial.run_id,
        "experiment_id": trial.experiment_id,
        "name": trial.name,
        "model_id": trial.model_id,
        "kind": trial.kind,
        "status": trial.status,
        "rank": trial.rank,
        "is_best": trial.is_best,
        "parent_trial_ids": list(trial.parent_trial_ids or []),
        "created_by_action_id": trial.created_by_action_id,
        "started_at": trial.started_at.isoformat() if trial.started_at else None,
        "finished_at": trial.finished_at.isoformat() if trial.finished_at else None,
        "duration_ms": trial.duration_ms,
        "error": trial.error,
        "notes": trial.notes,
        "created_at": trial.created_at.isoformat() if trial.created_at else None,
    }
    metrics = dict(trial.metrics or {})
    params = dict(trial.params or {})
    artifact = {
        "artifact_uri": trial.stored_path,
        "sha256": trial.sha256,
        "size_bytes": trial.size_bytes,
    }
    return {
        "metadata": metadata,
        "metrics": metrics,
        "params": params,
        "artifact": artifact,
    }
