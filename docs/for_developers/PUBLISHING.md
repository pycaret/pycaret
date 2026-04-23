# Publishing to PyPI

PyCaret 4.0 uses **PyPI Trusted Publishing** via the `.github/workflows/release.yml` workflow. No API tokens are stored in GitHub secrets; PyPI authenticates the GitHub Actions runner via OIDC.

## One-time setup (already done on the first release)

Performed at the PyPI project level, **not** in the repo:

1. Go to https://pypi.org/manage/project/pycaret/settings/publishing/ (PyPI) and https://test.pypi.org/manage/project/pycaret/settings/publishing/ (TestPyPI).
2. Under "Trusted publishers", click "Add a new publisher" for each target:
   - **Owner:** `pycaret`
   - **Repo:** `pycaret`
   - **Workflow name:** `release.yml`
   - **Environment name:** `pypi` (for PyPI) or `testpypi` (for TestPyPI)
3. Create matching GitHub environments:
   - Repo → Settings → Environments → New environment → `pypi` (with deployment branch = `v4` for now, `main` later).
   - Same for `testpypi`.

Once this is done, no secrets are required; the workflow authenticates via OIDC on every run.

## Pre-flight checklist (every release)

Before tagging:

- [ ] Version bumped in both `pyproject.toml` `[project].version` and `pycaret/__init__.py` `__version__`.
- [ ] `CHANGELOG.md` has an entry for the new version with `## [X.Y.Zaa] — YYYY-MM-DD` header.
- [ ] `uv run ruff check pycaret tests` passes.
- [ ] `uv run pytest tests/` passes (32/32 on Python 3.13).
- [ ] `uv build` produces a valid wheel + sdist in `dist/`.
- [ ] `uv run --with twine twine check dist/*` returns PASSED for both.
- [ ] A fresh `uv venv` + `uv pip install dist/*.whl` + import smoke works.

## Release to PyPI

### Option A — Tag-triggered (recommended)

```bash
# Make sure you're on the right branch (v4 during the revamp, main after)
git checkout v4

# Tag and push
git tag v4.0.0a0 -m "4.0.0a0: first test release of the 4.0 engine"
git push origin v4.0.0a0
```

The tag push triggers `.github/workflows/release.yml`:

1. **build** job — `uv build`, then `twine check` validates the artifacts.
2. **smoke** matrix — installs the built wheel in a fresh venv on ubuntu-latest + windows-latest × Python 3.11/3.12/3.13, runs import + surface-area smoke.
3. **publish-pypi** job — uploads `dist/*` to PyPI via `pypa/gh-action-pypi-publish@release/v1`. Uses trusted publishing (OIDC); no token needed.

### Option B — Manual via workflow_dispatch

Useful for publishing to TestPyPI first:

1. Go to https://github.com/pycaret/pycaret/actions/workflows/release.yml
2. Click "Run workflow".
3. Select branch `v4`, `target=testpypi`.
4. The `publish-testpypi` job runs instead of `publish-pypi`; same `dist/*` is uploaded to https://test.pypi.org/project/pycaret/.

Test the TestPyPI install:

```bash
pip install --pre --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pycaret==4.0.0a0
```

After confirming, re-run with `target=pypi` or create the tag for the real release.

## Local build (without publishing)

```bash
uv build                                          # -> dist/*.whl + dist/*.tar.gz
uv run --with twine twine check dist/*            # -> PASSED for both
```

To test the built wheel locally in a fresh env:

```bash
uv venv --python 3.13 /tmp/pycaret-testenv
uv pip install --python /tmp/pycaret-testenv/bin/python dist/*.whl
/tmp/pycaret-testenv/bin/python -c "import pycaret; print(pycaret.__version__)"
```

## Post-release

- [ ] Verify `pip install --pre pycaret==X.Y.Zaa` works from a throwaway machine.
- [ ] Create a GitHub Release from the tag; paste the relevant `CHANGELOG.md` entry into the body.
- [ ] Bump the version in `pyproject.toml` + `__init__.py` back to the next dev marker (`X.Y.Zaa.dev0` or similar).
- [ ] Commit the bump: `git commit -am "chore: bump to X.Y.Zaa+1.dev0"` and push.

## Emergency unpublish

PyPI does not allow deleting a version once uploaded — you can only yank it (still downloadable by pinned installers, hidden from the `pip install` default). To yank:

```bash
# Via pypi.org UI: project → Manage → Releases → [version] → Yank
# OR via twine (uses token auth, not OIDC):
twine yank pycaret==X.Y.Zaa -r pypi
```

Fix the bug, bump the version (e.g. `X.Y.Zaa+1`), re-release. Never reuse a version number.

## Dependencies of the release tooling

Nothing above pulls new PyCaret deps. The release pipeline uses:

- `uv` — build + venv.
- `twine` — only for `twine check`; actual upload is done by `pypa/gh-action-pypi-publish@release/v1` in CI.
- `hatchling` — build backend declared in `pyproject.toml`.
- PyPI's trusted-publishing OIDC integration — no tokens.
