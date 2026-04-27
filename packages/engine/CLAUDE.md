# CLAUDE.md — `packages/engine/` (the `pycaret` library)

Path-scoped pointer for Claude Code working in this directory. The
**authoritative** Python conventions live in the repo-root
[`AGENTS.md`](../../AGENTS.md). This file just calls out a few
engine-only quick references and the test loop.

## Quick test loop

```bash
uv run pytest packages/engine/tests/ -q                # fast loop
uv run pytest packages/engine/tests/ -q -k <pattern>   # focused
uv run ruff check packages/engine/                     # lint
uv run ruff format packages/engine/                    # format
```

## Engine-specific reminders

- **Public API is OOP-only.** Use `Experiment(...).fit(df)`. The 3.x
  module-level functional API was removed and is on the kill list
  (`docs/revamp/KILL_LIST.md`) — do not restore.
- **Every public verb returns a typed result dataclass** (see
  `pycaret/core/results.py`): `CompareResult`, `TuneResult`,
  `PredictResult`. New verbs follow the same pattern. Never return
  bare DataFrames or dicts from the public API.
- **Pipelines are real `sklearn.pipeline.Pipeline` instances.** Set
  `set_output("pandas")` on them.
- **Events, not prints.** Use `self.logger.log(EventKind.X, ...)`,
  never `print()` inside the engine.
- **No upper-bound version pins** on numpy / pandas / sklearn / scipy
  / joblib in `pyproject.toml`.

## Test-first for bug fixes

Always write the failing test first, see it fail, then fix. Never
commit a test you haven't seen go red.

## More

For deeper conventions, see:

- [`AGENTS.md`](../../AGENTS.md) — the cross-vendor agent briefing
- [`docs/revamp/ARCHITECTURE_ENGINE.md`](../../docs/revamp/ARCHITECTURE_ENGINE.md)
- [`docs/for_agents/ENGINE_WALKTHROUGH.md`](../../docs/for_agents/ENGINE_WALKTHROUGH.md)
- [`docs/for_developers/DRAINING_THE_GODCLASS.md`](../../docs/for_developers/DRAINING_THE_GODCLASS.md)
