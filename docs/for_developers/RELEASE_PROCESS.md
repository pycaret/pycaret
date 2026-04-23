# Release process

PyCaret 4.0 ships from `main` with semver. Releases are cut by a maintainer; contributors don't need to know the full details, but this is the reference.

## Versioning

- `0.dev0` — in-flight pre-release (current state).
- `0.alpha0`, `0.alpha1`, … — feature-complete but API-breakable alphas.
- `0.beta0`, `0.beta1`, … — API-frozen, bug-fix-only.
- `0.rc0`, `0.rc1`, … — release candidates, installable from PyPI as `pycaret==4.0.0rc0`.
- `4.0.0` — final.

## Where the version lives

- `pycaret/__init__.py` → `__version__ = "4.0.0.dev0"`.
- `pyproject.toml` → `[project].version`.

Both must be updated together. (Future: we'll drive this with `hatch version`.)

## Cutting a release

1. **All tests green** across the CI matrix.
2. **All tutorial notebooks regenerated** (`uv run python scripts/build_notebooks.py --run`).
3. **Bump the version** in both files.
4. **Generate the user-facing release notes** from `docs/revamp/release_notes_pycaret4.md`:
   - Summarise the session blocks into bullets grouped by category.
   - Write to `CHANGELOG.md` under a new heading.
   - Tag `BREAKING` changes prominently at the top.
5. **Commit** with message `Release 4.0.0rcN` (or whichever).
6. **Tag** with `git tag v4.0.0rcN -m "..."` and push the tag.
7. **Build + publish**:
   ```bash
   uv build                                      # wheel + sdist -> dist/
   uv publish --repository testpypi dist/*       # dry-run on TestPyPI first
   uv publish dist/*                             # prod PyPI
   ```
8. **GitHub release** — copy the CHANGELOG entry into the release body; attach the built wheel.
9. **Post-release**: bump `__version__` back to the next dev marker (`4.0.1.dev0` or `4.1.0.dev0`).

## CHANGELOG.md vs `docs/revamp/release_notes_pycaret4.md`

- `release_notes_pycaret4.md` is the **engineering log** — append-only, session-dated, full detail.
- `CHANGELOG.md` is the **user-facing** summary derived from it at each release.

Do not hand-edit `CHANGELOG.md` entries for past releases.
