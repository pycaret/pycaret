"""Alert destination delivery."""

from __future__ import annotations

import json
import logging
import smtplib
from email.message import EmailMessage
from typing import Any

import urllib.request
import urllib.error

from pycaret_server.config import get_settings

_log = logging.getLogger(__name__)


def deliver_fired_rule(fired: dict[str, Any]) -> tuple[bool, str | None]:
    """Send one fired-rule notification to its destination.

    ``fired`` matches the dict shape returned by
    ``evaluate_rules_for_workspace`` (see ``api/monitoring.py``).
    Returns ``(ok, error_message)`` so the caller can stamp the
    AlertRule row with the result.
    """
    kind = str(fired.get("destination_kind", "")).lower()
    config = dict(fired.get("destination_config") or {})
    text = _format_message(fired)
    try:
        if kind == "slack":
            return _post_slack(config, text, fired)
        if kind == "webhook":
            return _post_webhook(config, text, fired)
        if kind == "email":
            return _send_email(config, text, fired)
    except Exception as exc:  # noqa: BLE001
        _log.exception("alert delivery failed")
        return False, f"{type(exc).__name__}: {exc}"
    return False, f"unknown destination_kind {kind!r}"


def _format_message(fired: dict[str, Any]) -> str:
    name = fired.get("name") or "unnamed"
    metric = fired.get("metric")
    val = fired.get("agg_value")
    comp = fired.get("comparator")
    thr = fired.get("threshold")
    return (
        f":rotating_light: PyCaret alert *{name}* fired — "
        f"`{metric}` {val:.4g} {comp} {thr}"
        if isinstance(val, (int, float))
        else f":rotating_light: PyCaret alert *{name}* fired — {metric} {comp} {thr}"
    )


def _post_slack(
    config: dict[str, Any], text: str, fired: dict[str, Any]
) -> tuple[bool, str | None]:
    url = config.get("webhook_url")
    if not url:
        return False, "slack config missing webhook_url"
    payload = {"text": text, "attachments": [{"text": json.dumps(fired, default=str)}]}
    return _http_post_json(str(url), payload)


def _post_webhook(
    config: dict[str, Any], text: str, fired: dict[str, Any]
) -> tuple[bool, str | None]:
    url = config.get("url")
    if not url:
        return False, "webhook config missing url"
    body = {"text": text, **fired}
    return _http_post_json(str(url), body, extra_headers=config.get("headers") or {})


def _send_email(
    config: dict[str, Any], text: str, fired: dict[str, Any]
) -> tuple[bool, str | None]:
    settings = get_settings()
    if not settings.smtp_host:
        return False, "SMTP not configured (set PYCARET_SMTP_HOST)"
    to_list = config.get("to")
    if not isinstance(to_list, list) or not to_list:
        return False, "email config needs to: [...]"
    msg = EmailMessage()
    msg["Subject"] = f"PyCaret alert: {fired.get('name', 'fired')}"
    msg["From"] = settings.smtp_from or settings.smtp_username or "pycaret@localhost"
    msg["To"] = ", ".join(map(str, to_list))
    msg.set_content(text + "\n\n" + json.dumps(fired, indent=2, default=str))
    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as s:
            if settings.smtp_use_tls:
                s.starttls()
            if settings.smtp_username and settings.smtp_password:
                s.login(settings.smtp_username, settings.smtp_password)
            s.send_message(msg)
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _http_post_json(
    url: str,
    body: dict[str, Any],
    extra_headers: dict[str, str] | None = None,
) -> tuple[bool, str | None]:
    data = json.dumps(body, default=str).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **(extra_headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            if 200 <= resp.status < 300:
                return True, None
            return False, f"http {resp.status}"
    except urllib.error.HTTPError as exc:
        return False, f"http {exc.code}: {exc.reason}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
