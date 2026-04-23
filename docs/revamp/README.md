# PyCaret 4.0 Revamp — Documentation Hub

> "The first major open-source ML library revived end-to-end by an AI engineering agent."

This directory is the authoritative narrative of the PyCaret 4.0 revamp. Everything non-trivial — every audit, every cut, every architectural decision, every intermediate thought — lives here. It is designed to be citeable for a research paper on AI-driven open-source revival.

## Meta

- **Project owner:** Moez Ali (creator of PyCaret)
- **Engineering agent:** Claude (Anthropic), driving the revamp end-to-end
- **Revamp start:** 2026-04-22
- **Target release:** PyCaret 4.0
- **Baseline:** PyCaret 3.4.0, ~62K LOC, 300+ open issues, unmaintained ~3 years, does not run on current Python/sklearn

## Documents

| File | Purpose |
|------|---------|
| [`README.md`](README.md) | This file — index and meta |
| [`AUDIT.md`](AUDIT.md) | Baseline inventory: LOC, deps, kill-list evidence, test landscape |
| [`ROADMAP.md`](ROADMAP.md) | Phased plan: what lands in which phase, exit criteria per phase |
| [`DECISIONS.md`](DECISIONS.md) | ADR-style decision log, newest first |
| [`KILL_LIST.md`](KILL_LIST.md) | Explicit register of deps and subsystems being removed, with replacements |
| [`STATUS.md`](STATUS.md) | Current session status, headline metrics, next-step plan |
| [`release_notes_pycaret4.md`](release_notes_pycaret4.md) | **Engineering change log — every non-trivial edit, append-only.** Source of truth for generating the user-facing `RELEASE_NOTES.md` at 4.0 ship time. |
| [`thinking/`](thinking/) | Intermediate reasoning, trade-off analyses, rejected approaches |

## Reading order for a new reader

1. `AUDIT.md` — what the codebase looks like today
2. `KILL_LIST.md` — what is going away and why
3. `ROADMAP.md` — how we get from here to 4.0
4. `DECISIONS.md` — the record of choices
5. `thinking/` — the *why* behind decisions, as it unfolded
