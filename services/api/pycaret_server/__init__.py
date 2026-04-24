"""PyCaret 4.0 application-platform backend.

A FastAPI server that fronts the ``pycaret`` engine with a typed REST + WebSocket
API, a SQL-backed workspace/project/experiment/run/pipeline/deployment model, and
in-house pipeline serving. See ``docs/revamp/PLATFORM_PLAN.md`` for the design.
"""

__version__ = "0.1.0a0"

__all__ = ["__version__"]
