"""Phase 10: monitoring + alerting routes.

Endpoints:

- ``GET    /workspaces/{ws}/alert-rules``           list
- ``POST   /workspaces/{ws}/alert-rules``           create
- ``PATCH  /alert-rules/{id}``                      toggle / edit
- ``DELETE /alert-rules/{id}``
- ``GET    /deployments/{id}/metrics``              time-series read-out
- ``POST   /deployments/{id}/metrics``              ingest (workers + recorders)
- ``POST   /alert-rules/evaluate``                  cron-style evaluator
   (called from the worker on the ``alert_evaluate`` Job kind)

Destinations supported v1: ``slack`` (webhook URL), ``email``
(``{"to": [...]}``), ``webhook`` (generic JSON POST). Real Slack /
SMTP delivery is on the worker side; the API is just CRUD + ingest.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from pycaret_server.api.workspaces import _require_access
from pycaret_server.auth import CurrentUser
from pycaret_server.db import AlertRule, Deployment, MetricPoint, get_db

router = APIRouter(tags=["monitoring"])


def _serialise_rule(r: AlertRule) -> dict:
    return {
        "id": r.id,
        "workspace_id": r.workspace_id,
        "deployment_id": r.deployment_id,
        "name": r.name,
        "metric": r.metric,
        "comparator": r.comparator,
        "threshold": r.threshold,
        "window_seconds": r.window_seconds,
        "destination_kind": r.destination_kind,
        "destination_config": dict(r.destination_config or {}),
        "enabled": r.enabled,
        "last_fired_at": r.last_fired_at.isoformat() if r.last_fired_at else None,
        "last_status": r.last_status,
        "last_error": r.last_error,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "created_by": r.created_by,
    }


_VALID_COMPARATORS = {"gt", "gte", "lt", "lte", "eq"}
_VALID_DESTINATIONS = {"slack", "email", "webhook"}


@router.get("/workspaces/{workspace_id}/alert-rules")
def list_rules(
    workspace_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> list[dict]:
    _require_access(user, db, workspace_id)
    rows = db.scalars(
        select(AlertRule)
        .where(AlertRule.workspace_id == workspace_id)
        .order_by(AlertRule.created_at.desc())
    ).all()
    return [_serialise_rule(r) for r in rows]


@router.post(
    "/workspaces/{workspace_id}/alert-rules",
    status_code=status.HTTP_201_CREATED,
)
def create_rule(
    workspace_id: str,
    payload: dict[str, Any],
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Body: ``{name, metric, comparator, threshold, window_seconds?,
    deployment_id?, destination_kind, destination_config}``."""
    _require_access(user, db, workspace_id)
    for key in ("name", "metric", "comparator", "threshold", "destination_kind"):
        if key not in payload:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"{key} is required"
            )
    if payload["comparator"] not in _VALID_COMPARATORS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"comparator must be one of {sorted(_VALID_COMPARATORS)}",
        )
    if payload["destination_kind"] not in _VALID_DESTINATIONS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"destination_kind must be one of {sorted(_VALID_DESTINATIONS)}",
        )
    r = AlertRule(
        workspace_id=workspace_id,
        deployment_id=payload.get("deployment_id"),
        name=str(payload["name"]),
        metric=str(payload["metric"]),
        comparator=str(payload["comparator"]),
        threshold=float(payload["threshold"]),
        window_seconds=int(payload.get("window_seconds") or 300),
        destination_kind=str(payload["destination_kind"]),
        destination_config=dict(payload.get("destination_config") or {}),
        enabled=bool(payload.get("enabled", True)),
        created_by=user.id,
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    return _serialise_rule(r)


@router.patch("/alert-rules/{rule_id}")
def patch_rule(
    rule_id: str,
    payload: dict[str, Any],
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    r = db.get(AlertRule, rule_id)
    if r is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "rule not found")
    _require_access(user, db, r.workspace_id)
    if "enabled" in payload:
        r.enabled = bool(payload["enabled"])
    if "threshold" in payload:
        r.threshold = float(payload["threshold"])
    if "comparator" in payload and payload["comparator"] in _VALID_COMPARATORS:
        r.comparator = str(payload["comparator"])
    if "window_seconds" in payload:
        r.window_seconds = int(payload["window_seconds"])
    if "destination_config" in payload:
        r.destination_config = dict(payload["destination_config"] or {})
    db.commit()
    db.refresh(r)
    return _serialise_rule(r)


@router.delete("/alert-rules/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_rule(
    rule_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    r = db.get(AlertRule, rule_id)
    if r is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "rule not found")
    _require_access(user, db, r.workspace_id)
    db.delete(r)
    db.commit()
    return None


# ─────────────────────────────────────────────── metric ingest + read


@router.get("/deployments/{deployment_id}/metrics")
def read_metrics(
    deployment_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    metric: str | None = None,
    since_seconds: int = 3600,
    limit: int = 500,
) -> dict:
    """Time-series readout. Returns each (metric, ts_bucket, value) row.

    Defaults to the last hour. Sorted ascending by bucket so chart
    libraries can plot directly.
    """
    d = db.get(Deployment, deployment_id)
    if d is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "deployment not found")
    _require_access(user, db, d.workspace_id)
    cutoff = datetime.now(UTC) - timedelta(seconds=max(60, int(since_seconds)))
    q = (
        select(MetricPoint)
        .where(
            MetricPoint.deployment_id == deployment_id,
            MetricPoint.ts_bucket >= cutoff,
        )
        .order_by(MetricPoint.ts_bucket.asc())
        .limit(max(1, min(int(limit), 5000)))
    )
    if metric:
        q = q.where(MetricPoint.metric == metric)
    rows = db.scalars(q).all()
    return {
        "deployment_id": deployment_id,
        "since": cutoff.isoformat(),
        "points": [
            {
                "metric": p.metric,
                "ts": p.ts_bucket.isoformat(),
                "value": p.value,
                "count": p.count,
                "extra": p.extra,
            }
            for p in rows
        ],
    }


@router.post(
    "/deployments/{deployment_id}/metrics",
    status_code=status.HTTP_201_CREATED,
)
def ingest_metric(
    deployment_id: str,
    payload: dict[str, Any],
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Append a metric point. Body: ``{metric, value, ts?, count?, extra?}``.

    For inbound auth tokens (workers / external recorders), this is the
    write surface — auth + workspace scope are enforced via the calling
    user just like every other endpoint.
    """
    d = db.get(Deployment, deployment_id)
    if d is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "deployment not found")
    _require_access(user, db, d.workspace_id)
    metric = payload.get("metric")
    value = payload.get("value")
    if not metric or value is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "metric + value required"
        )
    ts = payload.get("ts")
    ts_bucket: datetime
    if ts:
        try:
            ts_bucket = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"invalid ts {ts!r}"
            ) from None
    else:
        ts_bucket = datetime.now(UTC).replace(microsecond=0, second=0)
    p = MetricPoint(
        id=str(uuid.uuid4()),
        workspace_id=d.workspace_id,
        deployment_id=deployment_id,
        metric=str(metric),
        ts_bucket=ts_bucket,
        value=float(value),
        count=int(payload.get("count") or 1),
        extra=payload.get("extra"),
    )
    db.add(p)
    db.commit()
    return {"id": p.id, "ts": p.ts_bucket.isoformat()}


# ─────────────────────────────────────────────── evaluator (worker callable)


def evaluate_rules_for_workspace(
    db: Session, workspace_id: str
) -> list[dict[str, Any]]:
    """Evaluate every enabled rule in a workspace; return ones that fired.

    Walked by the worker's ``alert_evaluate`` handler. For each rule:

    1. Aggregate the metric over the rule's window.
    2. Compare against threshold.
    3. If breached and not already fired in this window, append a
       fired-rule entry; the caller delivers to the destination.
    """
    rules = db.scalars(
        select(AlertRule).where(
            AlertRule.workspace_id == workspace_id,
            AlertRule.enabled.is_(True),
        )
    ).all()
    fired: list[dict[str, Any]] = []
    now = datetime.now(UTC)
    for r in rules:
        window_start = now - timedelta(seconds=r.window_seconds)
        q = select(MetricPoint).where(
            MetricPoint.metric == r.metric,
            MetricPoint.ts_bucket >= window_start,
        )
        if r.deployment_id:
            q = q.where(MetricPoint.deployment_id == r.deployment_id)
        points = list(db.scalars(q).all())
        if not points:
            continue
        agg = sum(p.value * p.count for p in points) / max(
            sum(p.count for p in points), 1
        )
        if _comparator_trips(agg, r.comparator, r.threshold):
            # Dedup: if last_fired_at is within the same window, skip.
            if r.last_fired_at and r.last_fired_at >= window_start:
                continue
            r.last_fired_at = now
            r.last_status = "fired"
            fired.append(
                {
                    "rule_id": r.id,
                    "name": r.name,
                    "metric": r.metric,
                    "agg_value": agg,
                    "threshold": r.threshold,
                    "comparator": r.comparator,
                    "destination_kind": r.destination_kind,
                    "destination_config": dict(r.destination_config or {}),
                }
            )
    if fired:
        db.commit()
    return fired


def _comparator_trips(value: float, op: str, threshold: float) -> bool:
    if op == "gt":
        return value > threshold
    if op == "gte":
        return value >= threshold
    if op == "lt":
        return value < threshold
    if op == "lte":
        return value <= threshold
    if op == "eq":
        return value == threshold
    return False
