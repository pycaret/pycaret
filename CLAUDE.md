# CLAUDE.md — Claude Code guide for this repo

This file is the entry point Claude Code reads at session start. It
imports the cross-vendor [`AGENTS.md`](AGENTS.md) (which OpenAI Codex,
Cursor, and other agents also read) and adds Claude-Code-specific
guidance on top.

@AGENTS.md

---

## At a glance

PyCaret 4.0 is an open-source AutoML library + self-hosted control
plane. Monorepo:

| Path | Ships as |
|---|---|
| `packages/engine/` | `pip install pycaret` (the library) |
| `services/api/` | `pip install pycaret-server` (FastAPI control plane) |
| `apps/web/` | React + Vite frontend for the control plane |
| `apps/site/` | The public docs/marketing site at pycaret.org |

Default branch: `main`. The 3.x line is frozen on PyPI; this repo
only ships 4.0+.

---

## How contributors use this repo with Claude Code

This project is **Claude-Code-first**. Instead of a CI bot that fixes
issues, the workflow is:

1. A user opens an issue.
2. The maintainer (@moezali1) reviews it and either closes it or
   adds the **`Approved`** label.
3. **Anyone** can clone the repo, fire up Claude Code in their own
   checkout (using their own subscription / API key), pick an
   Approved issue, and have the agent fix it locally.
4. The agent opens a PR against `main`.
5. The maintainer reviews and merges.

No CI bot fixes issues automatically. No `ANTHROPIC_API_KEY` lives in
this repo's secrets. Compute is community-funded by whoever runs the
agent.

**One-line contributor flow:**

```bash
gh repo clone pycaret/pycaret && cd pycaret
claude
> /work-on-approved-issue
```

The slash command lists the Approved issues, helps you pick one, and
runs the `issue-fixer` sub-agent end-to-end.

---

## Files Claude Code reads (in order)

1. This file (`/CLAUDE.md`)
2. `/AGENTS.md` (imported above) — the cross-vendor brief
3. Subdirectory `CLAUDE.md` files when you cd into them:
   - `packages/engine/CLAUDE.md` — Python conventions for the library
   - `apps/web/CLAUDE.md` — React + Vite + FastAPI proxy conventions
   - `apps/site/CLAUDE.md` — Next.js static-export site conventions
4. `.claude/settings.json` — permission allowlist + denylist
5. `.claude/commands/*.md` — reusable slash commands
6. `.claude/agents/*.md` — task-focused sub-agents

---

## Common commands

```bash
# Install everything (engine + control plane + site)
uv sync --all-packages --all-extras
cd apps/site && npm install && cd -
cd apps/web && npm install && cd -

# Engine: lint + test loop
uv run ruff check . && uv run ruff format --check .
uv run pytest packages/engine/tests/ -q

# Site: dev preview at http://localhost:3001
cd apps/site && npm run dev

# Control plane: two terminals
uv run --package pycaret-server pycaret-server serve --reload   # :8000
cd apps/web && npm run dev                                       # :3000
```

---

## Things the agent must NOT do without explicit user approval

- **Force-push** to `main` (or to anyone else's branch). Branch
  protection on `main` should also stop this — but don't try.
- Skip git hooks (`--no-verify`, `--no-gpg-sign`).
- Modify `LICENSE`, `LICENSE.engine.txt`, or any other top-level legal
  file.
- Modify `version = …` in any `pyproject.toml` (release process is
  manual + maintainer-signed).
- Publish to PyPI (`twine upload`, `npm publish`, etc.).
- Modify anything under `docs/revamp/` — those are archival session
  notes and design records.
- Push directly to `main`. **Always** open a PR; the maintainer
  reviews and merges.
- Add new top-level dependencies without updating
  `docs/revamp/DECISIONS.md` (we audit the dep surface deliberately).

---

## Things the agent IS allowed and encouraged to do

- Read everything.
- Modify everything else (tests, docs, code, configs, workflows).
- Open PRs against `main` from `fix-<issue>-<slug>` style branches.
- Comment on issues with status / clarifying questions.
- Run lint, tests, and any non-publishing build command.
- Create new files where the project structure suggests they belong.
- Refactor mercilessly inside a single PR's scope; keep PR diffs
  focused on one logical change.

---

## When the agent is stuck

If the issue is ambiguous, the fix is risky, or the work would touch
something on the **must-not** list above:

1. Comment on the issue with the question or scope concern.
2. Stop. Don't open a half-baked PR.

The maintainer will either clarify in the issue or change scope.
