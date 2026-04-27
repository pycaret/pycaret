---
description: Pre-flight checklist before cutting a release
---

You're walking the maintainer through the release checklist for
PyCaret 4.0. Run each step, report ✅ or ❌, and only proceed when
everything is green.

**You are NOT permitted to actually publish** — `pyproject.toml`
version bumps, `git tag`, `twine upload`, and `npm publish` are all
on the do-not-touch list. This command produces a status report so
the maintainer can decide to publish manually.

## 1. No open Approved issues for this milestone

```bash
gh issue list --repo pycaret/pycaret --label Approved --state open
```

If anything's open and assigned to the milestone, list them.

## 2. Engine version in `packages/engine/pyproject.toml`

Read the file. Report the current version. Do **not** modify it.

## 3. CHANGELOG.md has an entry for this version

Read `CHANGELOG.md`. The top section should be `## [<version>] — <date>`
with the version matching #2.

## 4. Release-notes session block exists

Open `docs/revamp/release_notes_pycaret4.md`. The most recent session
block should describe what's shipping in this release.

## 5. All tests passing

```bash
uv run pytest packages/engine/tests/ -q
```

## 6. Lint clean

```bash
uv run ruff check . && uv run ruff format --check .
```

## 7. Site builds

```bash
cd apps/site && npm run build
```

Verify the build emits without errors (95+ pages expected).

## 8. Control plane backend imports clean

```bash
uv run --package pycaret-server python -c "from pycaret_server.app import create_app; create_app()"
```

## Output format

```
PyCaret 4.0 release-prep — <version>

  1. ✅ No open Approved issues
  2. ✅ Version: 4.0.0a2
  3. ❌ CHANGELOG missing entry for 4.0.0a2
  4. ✅ Release notes session block present
  5. ✅ Tests: 142 passed, 0 failed
  6. ✅ Lint clean
  7. ✅ Site built (95 pages)
  8. ✅ Backend imports clean

  Status: NOT READY — fix #3 before publishing.
```

If everything's green, end with:

```
  Status: READY. Maintainer to run release.yml workflow + sign tag.
```
