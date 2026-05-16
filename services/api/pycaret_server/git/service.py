"""High-level Git publish service.

Takes a GitRepository row + a project's data and writes the on-disk
layout described in :mod:`exporter`, then commits and pushes via the
``git`` binary over HTTPS (with the PAT injected into the URL).

We shell out instead of using GitPython / pygit2 because:

1. The set of operations is tiny (clone shallow, write files, commit,
   push) and ``git`` is universally available where this code runs.
2. GitPython carries enough quirks (handling of subprocesses on
   Windows, auth) that the wins don't outweigh the install cost.

For production deploys with restrictive networks, the PAT URL form
(``https://x-access-token:<pat>@host/owner/repo.git``) plays nicely
with most corporate proxies and tunnels.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.crypto import decrypt
from pycaret_server.db import Experiment, GitRepository, Run, Secret, Trial
from pycaret_server.git.exporter import (
    serialise_experiment,
    serialise_run,
    serialise_trial,
)

_log = logging.getLogger(__name__)


def _safe_dir_name(name: str) -> str:
    """Slugify a name for use as a directory component."""
    out = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return out or "unnamed"


def _auth_url(repo: GitRepository, pat: str | None) -> str:
    """Inject a PAT into the clone URL for HTTPS push."""
    if not pat:
        return repo.url
    if repo.url.startswith("https://"):
        return repo.url.replace(
            "https://", f"https://x-access-token:{pat}@", 1
        )
    return repo.url


def _git(args: list[str], cwd: str) -> tuple[int, str, str]:
    """Run a git command; return (rc, stdout, stderr)."""
    res = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},  # never prompt
    )
    return res.returncode, res.stdout, res.stderr


def publish_project(
    db: Session,
    repo: GitRepository,
    project_id: str,
    *,
    commit_message: str | None = None,
) -> dict[str, Any]:
    """Materialise + push the project's experiments/trials/runs.

    Returns ``{ok: bool, sha?: str, error?: str}``. Stamps the repo row
    with the result (``last_push_status``, ``last_push_sha``, …) on the
    way out.
    """
    pat = _resolve_pat(db, repo)
    if pat is None and repo.url.startswith("https://"):
        return _record_result(
            db, repo, ok=False, error="no PAT secret configured for this repo"
        )

    experiments = db.scalars(
        select(Experiment).where(Experiment.project_id == project_id)
    ).all()

    with tempfile.TemporaryDirectory(prefix="pyc-git-") as tmp:
        clone_dir = Path(tmp) / "repo"
        url = _auth_url(repo, pat)
        rc, _so, se = _git(
            ["clone", "--depth", "1", "--branch", repo.default_branch, url, str(clone_dir)],
            cwd=tmp,
        )
        if rc != 0:
            # Fresh repos may not have the default branch yet — init locally.
            clone_dir.mkdir(parents=True, exist_ok=True)
            _git(["init", "-b", repo.default_branch], cwd=str(clone_dir))
            _git(["remote", "add", "origin", url], cwd=str(clone_dir))
            _log.info("git clone failed (%s); initialised empty repo", se.strip())

        prefix = (repo.path_prefix or "").strip("/")
        base = clone_dir / prefix if prefix else clone_dir
        _write_layout(base, db, experiments)

        # Stage + commit + push. ``git commit`` returns non-zero on
        # "nothing to commit" which we treat as success.
        _git(["add", "-A"], cwd=str(clone_dir))
        msg = commit_message or f"pycaret: publish project {project_id} @ {datetime.now(UTC).isoformat()}"
        # Set a stable identity so commits land cleanly even when
        # ``user.email`` isn't configured globally on the runner.
        _git(["config", "user.email", "bot@pycaret.local"], cwd=str(clone_dir))
        _git(["config", "user.name", "pycaret-bot"], cwd=str(clone_dir))
        rc, so, se = _git(["commit", "-m", msg], cwd=str(clone_dir))
        if rc != 0 and "nothing to commit" not in (so + se).lower():
            return _record_result(db, repo, ok=False, error=f"commit failed: {se}")

        rc, so, se = _git(
            ["push", "origin", repo.default_branch],
            cwd=str(clone_dir),
        )
        if rc != 0:
            return _record_result(db, repo, ok=False, error=f"push failed: {se}")

        rc, sha, _se = _git(["rev-parse", "HEAD"], cwd=str(clone_dir))
        return _record_result(
            db, repo, ok=True, sha=sha.strip() if rc == 0 else None
        )


def _resolve_pat(db: Session, repo: GitRepository) -> str | None:
    if not repo.secret_id:
        return None
    s = db.get(Secret, repo.secret_id)
    if s is None:
        return None
    try:
        return decrypt(s.value_encrypted)
    except Exception:  # noqa: BLE001
        _log.exception("could not decrypt PAT for repo %s", repo.id)
        return None


def _write_layout(base: Path, db: Session, experiments: list[Experiment]) -> None:
    """Write the Phase 0-v2 directory tree under ``base``.

    Layout::

        experiments/<exp_name>/
            experiment.yaml
            runs/<run_id>/
                run.yaml
                leaderboard.json
                trials/<trial_name>/
                    trial.yaml
                    metrics.json
                    params.json
                    artifact.pointer.yaml
    """
    base.mkdir(parents=True, exist_ok=True)
    (base / "experiments").mkdir(parents=True, exist_ok=True)
    for exp in experiments:
        exp_dir = base / "experiments" / _safe_dir_name(exp.name)
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "experiment.yaml").write_text(
            yaml.safe_dump(serialise_experiment(exp), sort_keys=False)
        )
        runs_dir = exp_dir / "runs"
        runs_dir.mkdir(exist_ok=True)
        runs = db.scalars(
            select(Run).where(Run.experiment_id == exp.id)
        ).all()
        for run in runs:
            r_dir = runs_dir / run.id
            r_dir.mkdir(parents=True, exist_ok=True)
            (r_dir / "run.yaml").write_text(
                yaml.safe_dump(serialise_run(run), sort_keys=False)
            )
            (r_dir / "leaderboard.json").write_text(
                json.dumps(run.leaderboard or [], indent=2, default=str)
            )
            trials_dir = r_dir / "trials"
            trials_dir.mkdir(exist_ok=True)
            trials = db.scalars(
                select(Trial).where(Trial.run_id == run.id)
            ).all()
            for trial in trials:
                t_dir = trials_dir / _safe_dir_name(
                    trial.name or trial.model_id or trial.id
                )
                t_dir.mkdir(parents=True, exist_ok=True)
                payload = serialise_trial(trial)
                (t_dir / "trial.yaml").write_text(
                    yaml.safe_dump(payload["metadata"], sort_keys=False)
                )
                (t_dir / "metrics.json").write_text(
                    json.dumps(payload["metrics"], indent=2, default=str)
                )
                (t_dir / "params.json").write_text(
                    json.dumps(payload["params"], indent=2, default=str)
                )
                (t_dir / "artifact.pointer.yaml").write_text(
                    yaml.safe_dump(payload["artifact"], sort_keys=False)
                )


def _record_result(
    db: Session,
    repo: GitRepository,
    *,
    ok: bool,
    sha: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    repo.last_push_at = datetime.now(UTC)
    repo.last_push_status = "ok" if ok else "error"
    repo.last_push_sha = sha
    repo.last_push_error = error
    db.commit()
    return {"ok": ok, "sha": sha, "error": error}
