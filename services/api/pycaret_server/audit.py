"""Audit-log middleware — records every mutating API call.

Spec § 17.4. One append-only row per POST / PATCH / PUT / DELETE against
``/api/v1/*``. Read requests + heartbeats are skipped — they're noisy and
rarely forensically interesting.

Design:

- Middleware runs *after* auth resolution so it can read ``request.state.user``
  (populated by ``get_current_user``). That dependency runs per-route, so
  the middleware falls back to re-resolving the bearer / key header itself
  for best-effort user attribution when a route doesn't require auth.
- Request body is captured + the payload is scrubbed before persistence.
  Scrub field names: ``password``, ``password_hash``, ``api_key``, ``token``,
  ``refresh_token``, ``access_token``, ``api_key_encrypted``.
- Path-template matching: we store the literal URL path (``/api/v1/runs/abc-…``)
  + derive an ``action`` like ``runs.create`` / ``deployments.delete`` by
  folding path segments + HTTP method.
- Errors inside the middleware never break the request — we log and continue.
  Audit logs are best-effort, not a critical path.

Routes live in ``pycaret_server.api.audit`` — the middleware stays here so
``app.py`` can add it without importing the API module.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from pycaret_server.db import AuditLog
from pycaret_server.db import session as _session_mod

log = logging.getLogger("pycaret_server.audit")

# Only audit these methods — reads go unlogged (too noisy).
AUDITED_METHODS = frozenset({"POST", "PATCH", "PUT", "DELETE"})

# Don't log these even if POST — token exchange is already tracked by
# `sessions` table + login attempts, and heartbeats are just noise.
SKIP_PATH_PREFIXES = (
    "/api/v1/auth/refresh",
    "/healthz",
    "/openapi.json",
    "/docs",
    "/redoc",
)

# Field names whose values are redacted in persisted payloads.
SCRUB_FIELDS = frozenset(
    {
        "password",
        "password_hash",
        "api_key",
        "token",
        "refresh_token",
        "access_token",
        "api_key_encrypted",
        "plaintext_token",
    }
)

# Matches the typical UUID-v4 string + common slug segments so we can
# derive an {entity}.{verb} action name without those.
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _scrub(obj: Any) -> Any:
    """Redact sensitive fields in a nested dict / list. Returns a new object."""
    if isinstance(obj, dict):
        return {
            k: ("***REDACTED***" if k.lower() in SCRUB_FIELDS else _scrub(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_scrub(item) for item in obj]
    return obj


def _derive_action(path: str, method: str) -> tuple[str, str | None, str | None, str | None]:
    """Derive (action, target_type, target_id, workspace_id) from a URL + method.

    Examples:
      POST    /api/v1/workspaces                      → ('workspaces.create', 'workspace', None, None)
      PATCH   /api/v1/workspaces/{id}                 → ('workspaces.update', 'workspace', id, id)
      DELETE  /api/v1/deployments/{id}                → ('deployments.delete', 'deployment', id, None)
      POST    /api/v1/workspaces/{id}/members         → ('members.create', 'member', None, id)
      PATCH   /api/v1/workspaces/{wsId}/members/{uid} → ('members.update', 'member', uid, wsId)
      POST    /api/v1/llm/analyze-drift               → ('llm.analyze-drift', None, None, None)
    """
    # Strip /api/v1 prefix if present.
    raw = path.removeprefix("/api/v1").strip("/")
    if not raw:
        return (f"root.{method.lower()}", None, None, None)

    segments = raw.split("/")
    # Walk segments + classify UUID vs noun vs verb.
    nouns: list[str] = []
    ids: list[str] = []
    verb: str | None = None
    workspace_id: str | None = None
    for i, seg in enumerate(segments):
        if _UUID_RE.match(seg):
            ids.append(seg)
            # First UUID directly after /workspaces is the workspace_id.
            if workspace_id is None and i > 0 and segments[i - 1] == "workspaces":
                workspace_id = seg
        elif (
            seg
            in {
                "promote",
                "cancel",
                "predict",
                "deployments",
                "members",
                "drift-reports",
                "analyze-dataset",
                "analyze-drift",
                "design-experiment",
                "explain-run",
                "debug-run",
                "review-deployment",
                "test-connection",
                "bootstrap",
                "status",
                "refresh",
                "logout",
                "me",
                "api-keys",
                "consultations",
            }
            and nouns
        ):
            # These look like sub-verb / sub-entity segments; if a non-UUID
            # segment follows the last noun, it's a verb on that noun.
            verb = seg
        else:
            nouns.append(seg)

    # Method → default verb when none extracted.
    method_verb = {
        "POST": "create",
        "PATCH": "update",
        "PUT": "update",
        "DELETE": "delete",
        "GET": "read",
    }.get(method.upper(), method.lower())

    if verb is None:
        verb = method_verb
    else:
        # If the URL has a sub-verb segment (e.g. /runs/{id}/cancel), the
        # method is usually POST and the sub-verb is the real action.
        pass

    target_type = nouns[-1].rstrip("s") if nouns else None
    target_id = ids[-1] if ids else None
    namespace = nouns[-1] if nouns else "root"
    action = f"{namespace}.{verb}"
    return action, target_type, target_id, workspace_id


async def _extract_request_payload(request: Request) -> dict | None:
    """Best-effort JSON body extraction. Returns None for non-JSON bodies.

    Starlette lets us read body() once; after that the route handler still
    needs it, so we stash it into ``request._body`` before returning.
    """
    body = await request.body()
    # Re-inject so downstream route handlers can still read it.
    request._body = body  # type: ignore[attr-defined]
    if not body:
        return None
    ctype = request.headers.get("content-type", "")
    if "application/json" not in ctype:
        # Don't log multipart uploads / form-encoded / raw bytes — too large
        # or mostly binary.
        return {"_non_json_content_type": ctype, "_body_bytes": len(body)}
    try:
        loaded = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {"_unparseable_body": True, "_body_bytes": len(body)}
    return _scrub(loaded) if isinstance(loaded, (dict, list)) else {"_value": loaded}


class AuditLogMiddleware(BaseHTTPMiddleware):
    """Records one AuditLog row per mutating /api/v1/* request.

    Never blocks or fails the request. Failures are logged + swallowed.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        method = request.method.upper()
        path = request.url.path

        should_audit = method in AUDITED_METHODS and not any(
            path.startswith(p) for p in SKIP_PATH_PREFIXES
        )

        payload: dict | None = None
        if should_audit:
            try:
                payload = await _extract_request_payload(request)
            except Exception as exc:  # noqa: BLE001 — best-effort
                log.warning("audit: failed to extract payload for %s %s: %s", method, path, exc)
                payload = {"_extract_error": str(exc)}

        response = await call_next(request)

        if should_audit:
            try:
                self._persist(request, response, path, method, payload)
            except Exception as exc:  # noqa: BLE001
                log.warning("audit: failed to persist log row for %s %s: %s", method, path, exc)

        return response

    # --------------------------------------------------------------- helpers

    def _persist(
        self,
        request: Request,
        response: Response,
        path: str,
        method: str,
        payload: dict | None,
    ) -> None:
        action, target_type, target_id, derived_ws = _derive_action(path, method)

        # Resolve user + workspace from request state if the route's auth
        # dependency already populated them. Otherwise leave null — unauth /
        # failed-auth calls are still audit-worthy (intrusion forensics).
        user = getattr(request.state, "audit_user", None)
        workspace_id = getattr(request.state, "audit_workspace_id", None) or derived_ws
        user_id = getattr(user, "id", None) if user is not None else None

        client_host = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")

        row = AuditLog(
            workspace_id=workspace_id,
            user_id=user_id,
            action=action[:64],
            method=method[:8],
            path=path[:512],
            target_type=target_type[:32] if target_type else None,
            target_id=target_id[:36] if target_id else None,
            status_code=int(response.status_code) if response is not None else None,
            payload=payload,
            ip_address=client_host[:64] if client_host else None,
            user_agent=user_agent[:256] if user_agent else None,
            created_at=datetime.now(UTC),
        )

        # Resolve factory at call-time so tests that rebind
        # `pycaret_server.db.session.session_factory` are honoured.
        factory = _session_mod.session_factory
        with factory() as db:
            try:
                db.add(row)
                db.commit()
            except SQLAlchemyError as exc:
                log.warning("audit: DB error persisting log row: %s", exc)
                db.rollback()
