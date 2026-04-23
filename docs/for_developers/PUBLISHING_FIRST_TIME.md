# First-time PyPI publishing setup — PyCaret 4.0

Do this **once**. After it's in place, every `git tag v*` + `git push --tags` triggers a hands-free release via GitHub Actions.

## Why trusted publishing (OIDC) and not an API token

- No long-lived secrets in GitHub.
- No risk of a compromised token being used to push malicious versions.
- Audit trail on PyPI shows each publish tied to a specific GitHub workflow run.

## Step 1 — Configure PyPI

**For the production PyPI:**

1. Log into https://pypi.org as the `pycaret` maintainer.
2. Go to https://pypi.org/manage/project/pycaret/settings/publishing/
3. Under "Add a new pending publisher" (if there's no existing publisher) or "Add a new publisher", fill in:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `pycaret` |
   | Owner | `pycaret` |
   | Repository name | `pycaret` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

4. Click "Add". PyPI will now trust the `pypi` environment of the `pycaret/pycaret` repo's `release.yml` workflow.

**For TestPyPI (optional, recommended for dress rehearsals):**

Same flow at https://test.pypi.org/manage/project/pycaret/settings/publishing/ but with **Environment name: `testpypi`**.

If the TestPyPI project doesn't exist yet, use the "Add a new pending publisher" section instead — PyPI pre-registers the project for the first publish.

## Step 2 — Configure matching GitHub environments

1. Go to https://github.com/pycaret/pycaret/settings/environments
2. Click "New environment", name it `pypi`. Save.
3. Inside the `pypi` environment, under "Deployment branches and tags":
   - Change to "Selected branches and tags".
   - Add rule: pattern `v4` (for now; add `main` later when v4 is merged).
   - This prevents the environment from being used from branches other than `v4`.
4. Repeat for `testpypi`.

No secrets or variables needed in the environments. OIDC handles auth at runtime.

## Step 3 — Verify

Trigger a test run via workflow_dispatch:

1. Go to https://github.com/pycaret/pycaret/actions/workflows/release.yml
2. Click "Run workflow".
3. Branch: `v4`. Target: `testpypi`.
4. Click "Run workflow".

The `build` and `smoke` jobs should pass first. When `publish-testpypi` starts, it will print the trusted-publisher identity assertion and upload `dist/*.whl` + `dist/*.tar.gz` to https://test.pypi.org/project/pycaret/.

If trusted publishing isn't yet configured, the publish step fails with:
```
OIDC: could not retrieve a token from the GitHub Actions OIDC provider
```
or
```
invalid-publisher: The publisher claimed in the given token matches no configured trusted publisher
```

That's the signal to go back and complete Step 1.

## Step 4 — Real release

Once Step 3 succeeds:

```bash
git checkout v4
git tag v4.0.0a0 -m "PyCaret 4.0.0a0"   # only if the tag doesn't already exist
git push origin v4.0.0a0
```

The `release.yml` workflow runs automatically on the tag push and publishes to PyPI.

Verify:

```bash
pip install --pre pycaret==4.0.0a0
python -c "import pycaret; print(pycaret.__version__)"
```

## Re-running publish for an existing tag

If the tag is pushed but the publish step failed (e.g. trusted publishing wasn't set up yet), after fixing Step 1:

1. Go to https://github.com/pycaret/pycaret/actions/workflows/release.yml
2. "Run workflow" → branch `v4` → target `pypi`.
3. It rebuilds from HEAD of `v4` (since `workflow_dispatch` isn't tag-scoped; if HEAD drifted past the tag, bump the patch version and re-tag instead).

For tag-scoped retry, the cleanest path is to delete and recreate the tag pointing to the same SHA:

```bash
git push --delete origin v4.0.0a0
git tag -d v4.0.0a0
git tag v4.0.0a0 -m "PyCaret 4.0.0a0"
git push origin v4.0.0a0
```

Caveat: once a version is uploaded to PyPI, that version number is burned — don't recreate the tag after a successful upload. For a successful upload followed by a bug, bump the version (`4.0.0a1` or `4.0.0b0`) and re-release.

## Emergency: tag exists but I don't want to publish

If you pushed a tag prematurely, delete it from remote immediately:

```bash
git push --delete origin v4.0.0a0
git tag -d v4.0.0a0
```

If the release workflow already uploaded to PyPI, you need to yank the version (see `PUBLISHING.md` → "Emergency unpublish").
