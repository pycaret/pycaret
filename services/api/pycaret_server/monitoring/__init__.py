"""Phase 10 alert-delivery package.

Pure-function delivery for each ``destination_kind``:

- ``slack``  — POST a JSON body with ``text`` to a webhook URL.
- ``webhook`` — generic JSON POST.
- ``email``  — SMTP via ``smtplib``, configured per-server via
  ``PYCARET_SMTP_*`` env vars.

Best-effort by design: a destination outage never escalates into an
alerting failure. Errors are stamped back onto the AlertRule row via
``last_status='error'`` / ``last_error=...`` so an admin can see what
went wrong in the UI.
"""

from pycaret_server.monitoring.delivery import deliver_fired_rule

__all__ = ["deliver_fired_rule"]
