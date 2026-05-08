"""Automated drift detection on top of ``prediction_logs``.

Algorithm (kept deliberately simple for v1):

1. Pull the most recent ``BASELINE_ROWS`` successful predictions for the
   deployment — these are the "current window".
2. Pull the next-oldest ``BASELINE_ROWS`` successful predictions — these
   are the "baseline window". (First-pass before there's a real baseline
   artifact.)
3. For each numeric feature present in both windows, compute a Population
   Stability Index (PSI). Cap per-feature PSI at 1.0 to keep scores in
   [0, 1]. Aggregate the feature-level scores into a single
   ``drift_score`` = mean.
4. Bucket via the existing ``bucket_status(score)`` helper.

Persists one ``DriftReport`` row per call. Idempotent: caller (the
scheduler) decides cadence; this function just snapshots.

Limitations called out in code:
- No prediction-distribution drift yet (would require ground-truth labels
  arriving asynchronously).
- Categorical features handled via simple frequency-table chi-square,
  capped the same way.
- Baseline-vs-window split is naive; a richer flow would persist a
  reference distribution at deployment time and compare against it.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.api.drift import bucket_status
from pycaret_server.db import Deployment, DriftReport, PredictionLog

_log = logging.getLogger(__name__)

BASELINE_ROWS = 200
WINDOW_ROWS = 200
MIN_ROWS_FOR_CHECK = 50


def run_drift_check(
    db: Session,
    deployment_id: str,
    user_id: str | None,
) -> DriftReport | None:
    """Snapshot drift for ``deployment_id`` and persist a ``DriftReport``.

    Returns the new row, or ``None`` if there isn't enough data to compare.
    """
    dep = db.get(Deployment, deployment_id)
    if dep is None:
        raise ValueError(f"deployment {deployment_id!r} not found")

    rows = db.scalars(
        select(PredictionLog)
        .where(
            PredictionLog.deployment_id == deployment_id,
            PredictionLog.status == "ok",
            PredictionLog.request_sample.isnot(None),
        )
        .order_by(PredictionLog.created_at.desc())
        .limit(BASELINE_ROWS + WINDOW_ROWS)
    ).all()

    if len(rows) < MIN_ROWS_FOR_CHECK * 2:
        _log.info(
            "drift_check skipped for deployment %s: only %d logs (need %d)",
            deployment_id,
            len(rows),
            MIN_ROWS_FOR_CHECK * 2,
        )
        return None

    half = len(rows) // 2
    current_logs = rows[:half]
    baseline_logs = rows[half:]

    current_features = _flatten_request_samples(current_logs)
    baseline_features = _flatten_request_samples(baseline_logs)
    if not current_features or not baseline_features:
        return None

    feature_names = sorted(set(current_features.keys()) & set(baseline_features.keys()))
    feature_drift: dict[str, dict[str, Any]] = {}
    scores: list[float] = []

    for name in feature_names:
        current = current_features[name]
        baseline = baseline_features[name]
        if _looks_numeric(current) and _looks_numeric(baseline):
            score = _psi(_to_floats(baseline), _to_floats(current))
            kind = "psi"
        else:
            score = _chi2_categorical(baseline, current)
            kind = "chi2"
        score = max(0.0, min(1.0, score))
        feature_drift[name] = {"score": round(score, 4), "kind": kind}
        scores.append(score)

    drift_score = round(sum(scores) / max(1, len(scores)), 4)

    window_start = baseline_logs[-1].created_at
    window_end = current_logs[0].created_at if current_logs else datetime.now(UTC)

    row = DriftReport(
        deployment_id=deployment_id,
        baseline_artifact_id=None,
        window_start=window_start,
        window_end=window_end,
        drift_score=drift_score,
        drift_status=bucket_status(drift_score),
        feature_drift_json=feature_drift,
        prediction_drift_json=None,
        sample_size=len(current_logs),
        created_by=user_id or _system_user_id(db),
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    # Fire drift.alert webhook on moderate / severe scores. Best-effort.
    if row.drift_status in ("moderate", "severe"):
        try:
            from pycaret_server.webhooks import fire_event_async

            fire_event_async(
                "drift.alert",
                {
                    "workspace_id": dep.workspace_id,
                    "deployment_id": deployment_id,
                    "drift_score": row.drift_score,
                    "drift_status": row.drift_status,
                    "report_id": row.id,
                },
            )
        except Exception:  # noqa: BLE001
            _log.exception("drift.alert webhook fan-out failed for %s", deployment_id)

    return row


# ----------------------------------------------------------------- helpers


def _flatten_request_samples(logs: list[PredictionLog]) -> dict[str, list[Any]]:
    """Flatten ``request_sample`` lists across logs into per-feature columns."""
    cols: dict[str, list[Any]] = {}
    for log in logs:
        sample = log.request_sample or []
        for record in sample:
            if not isinstance(record, dict):
                continue
            for k, v in record.items():
                cols.setdefault(k, []).append(v)
    return cols


def _looks_numeric(values: list[Any]) -> bool:
    """At least 80% of entries should be int/float for a 'numeric' classification."""
    if not values:
        return False
    n_num = sum(1 for v in values if isinstance(v, (int, float)) and not isinstance(v, bool))
    return (n_num / len(values)) >= 0.8


def _to_floats(values: list[Any]) -> list[float]:
    out: list[float] = []
    for v in values:
        try:
            out.append(float(v))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return out


def _psi(baseline: list[float], current: list[float], bins: int = 10) -> float:
    """Population Stability Index across ``bins`` quantile buckets.

    Robust to identical baseline + current and to empty bins.
    """
    if not baseline or not current:
        return 0.0
    quantiles = [i / bins for i in range(bins + 1)]
    sorted_baseline = sorted(baseline)
    edges = [_quantile(sorted_baseline, q) for q in quantiles]
    edges = sorted(set(edges))
    if len(edges) < 2:
        return 0.0
    base_counts = _bucketize(baseline, edges)
    cur_counts = _bucketize(current, edges)
    base_total = sum(base_counts) or 1
    cur_total = sum(cur_counts) or 1
    psi = 0.0
    for b, c in zip(base_counts, cur_counts, strict=False):
        b_pct = max(b / base_total, 1e-6)
        c_pct = max(c / cur_total, 1e-6)
        psi += (c_pct - b_pct) * math.log(c_pct / b_pct)
    return abs(psi)


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = max(0, min(len(sorted_values) - 1, int(q * (len(sorted_values) - 1))))
    return sorted_values[idx]


def _bucketize(values: list[float], edges: list[float]) -> list[int]:
    counts = [0] * (len(edges) - 1)
    for v in values:
        for i in range(len(edges) - 1):
            if v <= edges[i + 1]:
                counts[i] += 1
                break
        else:
            counts[-1] += 1
    return counts


def _chi2_categorical(baseline: list[Any], current: list[Any]) -> float:
    """Normalised chi-square for categorical drift, mapped to [0, 1].

    Uses Cramer's V capped at 1.0. Empty / single-category inputs return 0.
    """
    base = Counter(str(v) for v in baseline)
    cur = Counter(str(v) for v in current)
    cats = sorted(set(base) | set(cur))
    if len(cats) < 2:
        return 0.0
    n = sum(base.values()) + sum(cur.values()) or 1
    chi2 = 0.0
    for c in cats:
        observed_b = base.get(c, 0)
        observed_c = cur.get(c, 0)
        row_total_b = sum(base.values())
        row_total_c = sum(cur.values())
        col_total = observed_b + observed_c
        if col_total == 0:
            continue
        exp_b = row_total_b * col_total / n
        exp_c = row_total_c * col_total / n
        if exp_b > 0:
            chi2 += (observed_b - exp_b) ** 2 / exp_b
        if exp_c > 0:
            chi2 += (observed_c - exp_c) ** 2 / exp_c
    cramers_v = math.sqrt(chi2 / max(n, 1))
    return min(1.0, cramers_v)


def _system_user_id(db: Session) -> str:
    """Pick any superuser as the 'system' actor for unattended drift writes."""
    from pycaret_server.db import User

    u = db.scalars(
        select(User).where(User.is_superuser.is_(True), User.is_active.is_(True)).limit(1)
    ).first()
    if u is None:
        raise RuntimeError("no superuser found to attribute scheduled drift report to")
    return u.id
