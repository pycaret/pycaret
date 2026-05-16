"""Phase 5 Git integration package.

Exports YAML for Experiment / Trial / Run and pushes to a repo via
``git`` over HTTPS with a PAT.

Layout:

- ``exporter`` — pure function ``serialise_*`` that turns an ORM row
  into the wire YAML shape. No I/O, no Git, no DB writes.
- ``service`` — high-level "publish project to repo" entrypoint that
  reads from the DB, materialises files in a temp clone, commits,
  pushes. Used by both the API endpoint and the worker handler.
"""

from pycaret_server.git.exporter import (
    serialise_experiment,
    serialise_run,
    serialise_trial,
)

__all__ = ["serialise_experiment", "serialise_run", "serialise_trial"]
