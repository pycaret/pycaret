# PyCaret 4.0 Revamp — Documentation Hub

> "The first major open-source ML library revived end-to-end by an AI engineering agent."

This directory is the authoritative narrative of the PyCaret 4.0 revamp. Everything non-trivial — every audit, every cut, every architectural decision, every intermediate thought — lives here. It is designed to be citeable for a research paper on AI-driven open-source revival.

## Meta

- **Project owner:** Moez Ali (creator of PyCaret)
- **Engineering agent:** Claude (Anthropic), driving the revamp end-to-end
- **Revamp start:** 2026-04-22
- **Target release:** PyCaret 4.0 engine + open-source application platform
- **Baseline:** PyCaret 3.4.0, ~62K LOC, 300+ open issues, unmaintained ~3 years, does not run on current Python/sklearn

## Documents

| File | Purpose |
|------|---------|
| [`README.md`](README.md) | This file — index and meta |
| [`AUDIT.md`](AUDIT.md) | Baseline inventory: LOC, deps, kill-list evidence, test landscape |
| [`ROADMAP.md`](ROADMAP.md) | **Phased plan — engine (Part 1) + application platform (Part 2).** Checkboxes reflect current state |
| [`DECISIONS.md`](DECISIONS.md) | ADR-style decision log, newest first |
| [`KILL_LIST.md`](KILL_LIST.md) | Explicit register of deps and subsystems being removed, with replacements |
| [`STATUS.md`](STATUS.md) | Current session status, headline metrics, next-step plan |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | The 4.0 **engine** architecture — 8 design principles, package layout, `Experiment` interface contract |
| [`PLATFORM_PLAN.md`](PLATFORM_PLAN.md) | **The Part-2 plan: CLI + FastAPI backend + SQL database + React UI + Docker deploy.** Gated on Part-1 engine release |
| [`release_notes_pycaret4.md`](release_notes_pycaret4.md) | **Engineering change log — every non-trivial edit, append-only.** Source of truth for generating the user-facing `RELEASE_NOTES.md` at ship time |
| [`thinking/`](thinking/) | Intermediate reasoning, trade-off analyses, rejected approaches |
| [`github_issues/`](github_issues/) | Snapshot + triage of the 388 open issues (224 bulk-actionable) |

## Reading order for a new reader

1. `AUDIT.md` — what the 3.x codebase looked like at the start
2. `KILL_LIST.md` — what is going away and why
3. `ARCHITECTURE.md` — the new 4.0 engine design
4. `ROADMAP.md` — phased plan with progress checkmarks
5. `PLATFORM_PLAN.md` — the Part-2 vision (CLI + API + DB + React)
6. `STATUS.md` — where we are right now
7. `DECISIONS.md` — the record of choices
8. `thinking/` — the *why* behind decisions, as it unfolded
9. `release_notes_pycaret4.md` — the complete change log

## Two parts, one programme

**Part 1 — Engine.** PyCaret 4.0 the Python library. OOP-only, sklearn-composable, lean, agent-friendly. Sessions 1-6 and ongoing.

**Part 2 — Application Platform.** CLI + FastAPI backend + SQL database + React UI + Docker deploy. A credible open-source alternative to DataRobot / H2O.ai. Gated on Part 1 shipping `4.0.0alpha0`.

See `ROADMAP.md` for the current phase breakdown.
