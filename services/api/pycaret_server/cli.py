"""`pycaret-server` CLI entry point.

Usage:
    pycaret-server init [--data-dir ./data]
    pycaret-server serve [--host 0.0.0.0] [--port 8020] [--reload]
    pycaret-server migrate [--url sqlite:///...] [--revision head] [--reset-dev]
    pycaret-server worker [--queues default,gpu] [--worker-id pyc-1]
    pycaret-server doctor
    pycaret-server version

Default port is 8020 so PyCaret can run alongside resumly (8000) and
halani (8005) without stomping on either.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

from pycaret_server import __version__


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    reload_kwargs: dict[str, object] = {}
    if args.reload:
        # Watch only source dirs; don't restart on .venv installs, DB writes,
        # artifact uploads, or node_modules churn (a .venv reload mid-session
        # silently rotates the ephemeral PYCARET_SECRETS_KEY and breaks every
        # already-encrypted secret in the DB).
        reload_kwargs["reload_excludes"] = [
            "*/.venv/*",
            "*/node_modules/*",
            "*/artifacts/*",
            "*/__pycache__/*",
            "*.db",
            "*.db-journal",
            "*.log",
        ]
    uvicorn.run(
        "pycaret_server.app:create_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=True,
        **reload_kwargs,
    )
    return 0


def _migrate(args: argparse.Namespace) -> int:
    from pycaret_server.config import get_settings
    from pycaret_server.db.bootstrap import _run_alembic

    url = args.url or get_settings().database_url

    if args.reset_dev:
        # Phase 0 ships a destructive trials/runs pivot. The dev DB has no
        # production data, so the cleanest path forward on a developer machine
        # is to wipe the SQLite file and re-apply head from scratch. Refuse on
        # non-SQLite URLs to avoid catastrophic accidents.
        if not url.startswith("sqlite"):
            print(
                f"--reset-dev refuses to touch a non-SQLite URL: {url!r}. "
                "Drop and recreate the database manually instead.",
                file=sys.stderr,
            )
            return 2
        from pathlib import Path
        from sqlalchemy.engine import make_url

        sa_url = make_url(url)
        db_path = sa_url.database
        if db_path and db_path not in (":memory:", ""):
            p = Path(db_path)
            if p.exists():
                p.unlink()
                print(f"reset-dev: removed {p}")
            else:
                print(f"reset-dev: no existing file at {p} (continuing)")

    _run_alembic("upgrade", args.revision, url=url)
    print(f"migrated {url} to revision {args.revision}")
    return 0


def _worker(args: argparse.Namespace) -> int:
    """Run the Phase 1 background worker."""
    from pycaret_server.worker import serve as worker_serve

    queues = [q.strip() for q in (args.queues or "").split(",") if q.strip()]
    return worker_serve(
        queues=queues or None,
        worker_id=args.worker_id,
        redis_url=args.redis_url,
    )


def _doctor(_args: argparse.Namespace) -> int:
    """Quick health check: DB reachable, Redis reachable, storage writable.

    Useful as the first thing a new self-hoster runs after `init`.
    """
    from pycaret_server.config import get_settings
    from pycaret_server.runs import queue_redis as _q

    settings = get_settings()
    ok = True

    # DB.
    try:
        from sqlalchemy import create_engine, text

        eng = create_engine(settings.database_url)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"database  OK   url={settings.database_url}")
    except Exception as exc:  # noqa: BLE001
        print(f"database  FAIL url={settings.database_url} err={exc}")
        ok = False

    # Redis (only when using the redis backend).
    if settings.runs_backend == "redis":
        if _q.is_healthy(settings.redis_url):
            print(f"redis     OK   url={settings.redis_url}")
        else:
            print(f"redis     FAIL url={settings.redis_url}")
            ok = False
    else:
        print(f"redis     SKIP runs_backend={settings.runs_backend}")

    # Artifact storage.
    try:
        settings.artifact_dir.mkdir(parents=True, exist_ok=True)
        probe = settings.artifact_dir / ".doctor-probe"
        probe.write_text("ok")
        probe.unlink()
        print(f"storage   OK   dir={settings.artifact_dir}")
    except Exception as exc:  # noqa: BLE001
        print(f"storage   FAIL dir={settings.artifact_dir} err={exc}")
        ok = False

    return 0 if ok else 1


def _init(args: argparse.Namespace) -> int:
    """Phase 13: zero-touch bootstrap for a new self-host.

    Writes a ``.env`` file under ``args.data_dir`` with sane defaults
    (random JWT secret + Fernet encryption key, SQLite DB path under
    the data dir, artifact dir likewise), and runs migrations so the
    first ``pycaret-server serve`` boots into a working state.

    Idempotent — re-running on an initialised data dir prints the
    existing config and skips overwriting the .env.
    """
    import secrets as _secrets
    from pathlib import Path

    from cryptography.fernet import Fernet

    data_dir = Path(args.data_dir or "./data").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    env_path = data_dir / ".env"
    db_path = data_dir / "pycaret.db"
    artifact_dir = data_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if env_path.exists() and not args.force:
        print(f"init: {env_path} already exists; pass --force to overwrite")
        print(f"init: data dir is {data_dir}")
        return 0

    jwt = _secrets.token_urlsafe(48)
    enc_key = Fernet.generate_key().decode()
    db_url = f"sqlite:///{db_path}"
    env_body = (
        f"# Generated by `pycaret-server init` on {datetime.now(UTC).isoformat()}\n"
        f"PYCARET_DATABASE_URL={db_url}\n"
        f"PYCARET_ARTIFACT_DIR={artifact_dir}\n"
        f"PYCARET_JWT_SECRET={jwt}\n"
        f"PYCARET_SECRETS_KEY={enc_key}\n"
        f"PYCARET_ENVIRONMENT=dev\n"
        f"PYCARET_RUNS_BACKEND=inprocess\n"
        f"PYCARET_STORAGE_BACKEND=local\n"
    )
    env_path.write_text(env_body)
    print(f"init: wrote config to {env_path}")
    print(f"init: data dir is {data_dir}")
    print("init: applying migrations…")
    from pycaret_server.db.bootstrap import _run_alembic

    _run_alembic("upgrade", "head", url=db_url)
    print("init: ready. Run `pycaret-server serve` to start the API.")
    return 0


def _version(_args: argparse.Namespace) -> int:
    print(__version__)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pycaret-server")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Start the FastAPI server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8020)
    serve.add_argument("--reload", action="store_true", help="auto-reload on code change")
    serve.set_defaults(fn=_serve)

    migrate = sub.add_parser("migrate", help="Run Alembic migrations")
    migrate.add_argument(
        "--url", default=None, help="override database URL (default: PYCARET_DATABASE_URL)"
    )
    migrate.add_argument("--revision", default="head", help="target revision (default: head)")
    migrate.add_argument(
        "--reset-dev",
        action="store_true",
        dest="reset_dev",
        help=(
            "Delete the SQLite dev DB file before migrating. Refuses to run "
            "against non-SQLite URLs. Use this when a destructive migration "
            "lands and you'd rather start clean than write a data-migration "
            "by hand."
        ),
    )
    migrate.set_defaults(fn=_migrate)

    worker = sub.add_parser("worker", help="Run the background worker (Phase 1)")
    worker.add_argument(
        "--queues",
        default=None,
        help=(
            "comma-separated queue names to listen on (default: from "
            "PYCARET_WORKER_QUEUES env var, falls back to 'default')"
        ),
    )
    worker.add_argument(
        "--worker-id",
        dest="worker_id",
        default=None,
        help="stable id for this worker process (default: random uuid)",
    )
    worker.add_argument(
        "--redis-url",
        dest="redis_url",
        default=None,
        help="override Redis URL (default: PYCARET_REDIS_URL)",
    )
    worker.set_defaults(fn=_worker)

    doc = sub.add_parser(
        "doctor",
        help="Quick health check: DB / Redis / storage reachable",
    )
    doc.set_defaults(fn=_doctor)

    init = sub.add_parser(
        "init",
        help="One-shot bootstrap for a new self-host (writes .env + applies migrations)",
    )
    init.add_argument(
        "--data-dir",
        dest="data_dir",
        default="./data",
        help="Directory for SQLite DB + artifacts + .env (default: ./data)",
    )
    init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing .env in the data dir",
    )
    init.set_defaults(fn=_init)

    ver = sub.add_parser("version", help="Print the server version and exit")
    ver.set_defaults(fn=_version)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
