"""`pycaret-server` CLI entry point.

Usage:
    pycaret-server serve [--host 0.0.0.0] [--port 8000] [--reload]
    pycaret-server version
"""

from __future__ import annotations

import argparse
import sys

from pycaret_server import __version__


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "pycaret_server.app:create_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=True,
    )
    return 0


def _version(_args: argparse.Namespace) -> int:
    print(__version__)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pycaret-server")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Start the FastAPI server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true", help="auto-reload on code change")
    serve.set_defaults(fn=_serve)

    ver = sub.add_parser("version", help="Print the server version and exit")
    ver.set_defaults(fn=_version)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
