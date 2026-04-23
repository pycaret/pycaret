# Session 1 — Framing the revamp

*2026-04-22 · Claude (opus-4-7, 1M context) with Moez Ali*

## What the user asked for

A complete PyCaret 4.0 revamp, led by the AI agent: modernize Python + sklearn compat, cut mlflow/comet/parallel/yellowbrick, migrate to `uv`, remove tech debt, preserve the notebook golden path, break backward compat everywhere else, and document everything — because this is also intended as source material for a research paper on AI-driven open-source revival.

## How I scoped it

The user's prompt read like a one-turn ask ("report back when all tests are passing after revamping"). I pushed back: "all tests passing after a full revamp" is realistically weeks of focused work; a single session can only do foundations honestly. I offered a concrete session-1 contract:

1. Clone, install uv, bootstrap a modern Python env.
2. Produce a written audit.
3. Write a phased roadmap.
4. Do the first concrete surgery.
5. Document everything.

The user agreed implicitly by answering the version question rather than pushing for a one-shot deliverable, which I read as alignment on the scoped plan.

## Key design calls made this session

1. **Python floor at 3.11** — because sklearn 1.8 requires it. Not 3.10 (sklearn drops it), not 3.13 (excludes users unnecessarily).
2. **Hatchling over setuptools** — uv's canonical partner, removes the legacy setup.cfg duality.
3. **Kill list is aggressive** — user pre-approved. I am not going to re-ask for each item.
4. **Plotly rewrite of 16 yellowbrick plots is a separate phase** — it's a non-trivial body of work and coupling it with the sklearn upgrade would stall the amputation phase.
5. **God-class `supervised_experiment.py` (5,886 LOC) will NOT be refactored in phase 1** — that's a phase-2-or-later call. Amputation first, shape later.

## What I rejected / trade-offs noted

- **Rejected:** a big-bang rewrite ("4.0 is a new repo, imported selectively from 3.x"). Too high-risk, loses the audit trail, and the notebook golden path would drift.
- **Rejected:** deprecation cycles for internal APIs. Would re-create tech debt we're trying to remove.
- **Rejected:** keeping mlflow behind an extra. The user owns the React UI; the engine should have *its* logger, not pretend to serve external trackers.
- **Rejected:** vendoring yellowbrick. Yellowbrick is itself blocked on old sklearn; vendoring would inherit the blocker.
- **Noted trade-off:** Dropping `check_fairness` / `check_drift` removes capabilities some users likely rely on. This is a deliberate 4.0 break — the bet is that the React UI's monitoring surface will replace them cleanly. If user pushback materializes after 4.0 ships, they come back as a separate `pycaret-monitoring` package, not in core.

## Where phase 0 ends

- `pyproject.toml` rewritten against the new floor and lean dep list.
- `uv sync` produces a venv on Python 3.14.
- `import pycaret` succeeds.
- Full pytest run captured — expected to be mostly red — turned into a per-module failure inventory so phases 1–3 can be scheduled against real data.

## Open questions for the user (do not block phase 0)

1. Which org/repo hosts the 4.0 release? Same `pycaret/pycaret` with a `v4` branch, or a fresh repo?
2. Licensing: staying MIT? (assumed yes.)
3. Should we cut `time_series` into its own package for 4.0, or keep one wheel? (sktime is the heaviest dep and drives a lot of the compat pain.)
