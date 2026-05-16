# Test plan — full-platform smoke

Everything between "boot the stack" and "exercise each Phase surface
end-to-end". Each step calls out what success looks like; if anything
doesn't match, stop and capture the error so we can debug it instead
of chasing cascading failures.

## Pre-flight (one-time)

```powershell
# 0a. From the repo root, blow away any stale dev DB and re-apply migrations.
cd C:\Users\moezs\pycaret\pycaret
.venv\Scripts\python -m pycaret_server.cli migrate --reset-dev --url sqlite:///./pycaret-dev.db
#  → expect: "reset-dev: removed …" then "migrated … to revision head"

# 0b. Health check.
.venv\Scripts\python -m pycaret_server.cli doctor
#  → expect: database OK, redis SKIP, storage OK
```

If `doctor` complains about anything other than `redis SKIP`, fix that
before continuing.

## Boot

Two terminals:

```powershell
# Terminal 1 — API.
cd C:\Users\moezs\pycaret\pycaret
$env:PYCARET_DATABASE_URL = "sqlite:///./pycaret-dev.db"
$env:PYCARET_JWT_SECRET   = "dev-only-do-not-ship-but-long-enough-for-hmac"
.venv\Scripts\python -m pycaret_server.cli serve --reload

# Terminal 2 — UI.
cd C:\Users\moezs\pycaret\pycaret\apps\web
npm run dev
```

Open <http://localhost:3020> in a browser. Expect either the **Setup**
wizard (fresh DB) or the **Login** screen (DB already bootstrapped).

## 1 · Bootstrap + sign-in

| # | Action | Expected |
|---|---|---|
| 1.1 | Sign up the first user from the Setup wizard | Redirected to the workspace list; sidebar populated |
| 1.2 | Click into the seeded **Default** workspace | URL becomes `/workspaces/<id>`; sidebar shows the *Workspace* group |
| 1.3 | Open the workspace switcher in the top-right | Lists Default + a "Pick a workspace" CTA |
| 1.4 | `⌘K` / `Ctrl+K` | Command palette opens with sections for Workspaces / Workspace items / Account |

## 2 · Sidebar new-page reachability

Click each new sidebar entry in order. Each should land cleanly with an
empty-state if there's no data yet.

| Sidebar item | Expected page |
|---|---|
| **Model registry** | `/workspaces/:wsId/models` — empty state with "New model" CTA |
| **Monitoring** | `/workspaces/:wsId/monitoring` — empty rules list + dropdown asking you to pick a deployment |
| **Lineage** | `/workspaces/:wsId/lineage` — empty "no edges yet" |
| **Approvals** | `/workspaces/:wsId/approvals` — filter chips work; empty list under each |
| **Connections** | `/workspaces/:wsId/connections` — empty state |
| **Secrets** | `/workspaces/:wsId/secrets` — empty state |
| **Git** | `/workspaces/:wsId/git` — empty state |
| **Queues & workers** (superuser only) | `/admin/queues` — shows queues table; auto-refreshes every 5s |

If a sidebar item 404s, the route wasn't wired — note it and move on.

## 3 · Phase 4 · Catalog (secrets → connections → datasets)

| # | Action | Expected |
|---|---|---|
| 3.1 | Sidebar **Secrets** → "New secret" → name `pg-pw`, kind `db_password`, value `topsecret` → Save | New row appears with `••••cret` |
| 3.2 | Sidebar **Connections** → "New connection" → name `pg-local`, kind PostgreSQL, host `localhost`, database `pycaret`, user `pycaret`, pick the secret → Create | New row appears |
| 3.3 | Click **Test** on the connection | Status pill flips to `error` (no real Postgres running) and the error text is shown — that's fine, this confirms the driver path |
| 3.4 | Inside a project, upload a CSV via the data-sources card (existing flow) | New CSV row appears |
| 3.5 | Click **Versions** on a CSV row | Lands on `/workspaces/:wsId/datasets/:id` — versions list empty |
| 3.6 | Click **Refresh now** | New version row appears with schema columns + sample rows |

## 4 · Phase 0 · Trial / Run pivot

| # | Action | Expected |
|---|---|---|
| 4.1 | Inside a project, create a Classification experiment with `target=target` against the iris sklearn sample | Experiment detail page loads |
| 4.2 | Click **New run** → plan `compare` → submit | Run appears in queued/running, then succeeded within ~30s |
| 4.3 | Scroll down — the new **Trials** card lives below the runs table | Shows multiple trials grouped under one "compare batch · ..." header |
| 4.4 | Click any trial → opens the legacy trial detail page (compat shim) | Pipeline diagram + metrics + Predict tab all render |
| 4.5 | Hit the **Promote** button on the best trial → name `iris-best` | 201 created; back-link visible |
| 4.6 | Open the **Model registry** sidebar entry | The promoted Trial now appears? **No** — promotion still writes a Pipeline row, not a RegisteredModel. The registry page is empty. **This is expected** — see §5 to test the registry surface explicitly via the API. |

## 5 · Phase 7 · Model registry v2

Curl-based since the promote-to-registry button isn't on the trial detail
page yet — the API surface is fully wired.

```powershell
$token = "<paste your access_token from network tab>"
$ws    = "<workspace-id from URL>"
$run   = "<succeeded run id>"
$trial = "<id of best trial>"

# Create the named registered model.
curl -X POST "http://localhost:8020/api/v1/workspaces/$ws/registered-models" `
  -H "Authorization: Bearer $token" -H "Content-Type: application/json" `
  -d '{"name":"iris-prod","description":"production iris classifier"}'

# Promote (trial_id, run_id) → version 1.
$model = "<id returned above>"
curl -X POST "http://localhost:8020/api/v1/registered-models/$model/versions" `
  -H "Authorization: Bearer $token" -H "Content-Type: application/json" `
  -d "{`"trial_id`":`"$trial`",`"run_id`":`"$run`",`"status`":`"staging`"}"
```

Now reload the **Model registry** sidebar page → `iris-prod` should
appear with one version in *staging*. Click into it → version timeline
renders. Hit **Promote to prod** → status flips to *production*; archive
button replaces promote.

## 6 · Phase 10 · Monitoring

| # | Action | Expected |
|---|---|---|
| 6.1 | **Monitoring** → "New rule" → name `p95 > 500`, metric `p95_latency_ms`, comparator `gt`, threshold `500`, destination `slack`, config `{"webhook_url":"https://hooks.slack.com/services/INVALID"}` → Save | New rule row appears, enabled |
| 6.2 | Toggle the rule off/on | Pill swaps; persists across reloads |
| 6.3 | In the deployment dropdown, pick any deployment | If you have one from §4.6's promote, time-series panel renders. If not, "No data points" message |
| 6.4 | Hit the `/api/v1/deployments/<id>/metrics` endpoint via curl to seed a value | Re-render the page → chart fills in |

## 7 · Phase 11 · Statistical computing

| # | Action | Expected |
|---|---|---|
| 7.1 | Project header → **Analyses** | Cards by category (compare two groups, ANOVA, regression, survival, forecasting) |
| 7.2 | Click *Two-sample t-test* | Wizard opens |
| 7.3 | Name `iris-petal-by-target`, pick the iris CSV, grouping `target`, measure `petal length (cm)` (or whatever your CSV calls it) | Form populates |
| 7.4 | **Preview** | Result card shows test statistic + p-value + Cohen's d + bar chart |
| 7.5 | **Save** | Lands on the result pane; saved analysis appears in the list |
| 7.6 | Back to list → click into the saved analysis → **Run again** | New row in the history table |

Try one more: OLS regression with `response=petal length (cm)`,
`predictors=sepal length (cm), sepal width (cm)`. Result card shows
coefficient table + residuals-vs-fitted scatter.

## 8 · Phase 12 · Approvals

| # | Action | Expected |
|---|---|---|
| 8.1 | Open **Approvals** filter to `pending` | Empty |
| 8.2 | Curl-open an approval: `POST /api/v1/workspaces/<ws>/approvals` body `{"target_kind":"registered_model_version","target_id":"<version-id>","action":"promote_to_production","required_approvals":1}` | New row appears in the pending list |
| 8.3 | Click the row → detail pane shows action / required signatures / request payload | — |
| 8.4 | Add a comment "looks good" → **Approve** | Status flips to `approved`; **Execute action** appears |
| 8.5 | **Execute action** | Status flips to `executed`; the referenced RegisteredModelVersion is now `production` (check on the registry page) |

## 9 · Phase 14 · Queues & workers

| # | Action | Expected |
|---|---|---|
| 9.1 | Account → **Queues & workers** (superuser only) | Table shows `default` queue with throughput stats from §4's runs |
| 9.2 | While a run is in progress in another tab, refresh | `running` count increments; auto-refresh every 5s |
| 9.3 | Workers section | Empty unless `RUNS_BACKEND=redis` and a `pycaret-server worker` is running |

## 10 · Phase 4 · Lineage

| # | Action | Expected |
|---|---|---|
| 10.1 | Sidebar **Lineage** | After §5's promote + §8.5's execute, the SVG renders nodes + edges for `run → registered_model_version`, `registered_model_version → deployment` (if you created one in §11) |
| 10.2 | Click any node | Subgraph re-roots at that node with depth=2 |
| 10.3 | "← clear focus" | Returns to the full workspace view |

## 11 · Phase 5 · Git integration

Requires a real GitHub/GitLab repo + a PAT secret.

| # | Action | Expected |
|---|---|---|
| 11.1 | Secrets page → new secret, kind `git_pat`, value = your PAT | Row appears |
| 11.2 | Git page → "Link repository" → provider GitHub, clone URL `https://github.com/<you>/<repo>.git`, pick the PAT secret, pick a project → Link | New repo row appears |
| 11.3 | Hit **Publish** | Spinner; success banner with commit SHA OR an error message if your PAT lacks write access |
| 11.4 | Check the repo on GitHub | New commit; layout is `experiments/<name>/trials/<name>/runs/<id>/metadata.yaml` etc. |

## 12 · Phase 8 · Notebooks

With the default `PYCARET_NOTEBOOK_BACKEND=local`:

| # | Action | Expected |
|---|---|---|
| 12.1 | Project header → **Notebooks** | Empty state |
| 12.2 | "New notebook" → name `exploration.ipynb`, kernel python3 → Create | New row |
| 12.3 | **Open** | Dialog shows a "Notebook backend unavailable" placeholder explaining how to switch to `docker` |

To smoke-test docker mode: set `PYCARET_NOTEBOOK_BACKEND=docker`,
restart the API, ensure Docker Desktop is running, repeat 12.3 → iframe
loads a JupyterLab tab at `localhost:<port>`. (Optional — skip if no
Docker.)

## 13 · Phase 13 · Polish

| # | Action | Expected |
|---|---|---|
| 13.1 | `pycaret-server init --data-dir ./fresh-test --force` | Writes `.env`, applies migrations, prints "ready" |
| 13.2 | `pycaret-server doctor` against the new dir | All checks pass |
| 13.3 | `INSTALL.md` + `OPERATIONS.md` exist at the repo root | Read for accuracy against what you observed |

## 14 · Cross-cutting smoke

These touch multiple phases at once. If any of these break, it's
usually a glue bug rather than a per-phase failure.

| # | Action | Expected |
|---|---|---|
| 14.1 | `⌘K` from anywhere → type "registry" | Picks Model registry. Try "lineage", "approvals", "secrets", "git", "queues" — each lands on the right page |
| 14.2 | Refresh any deep-linked page (e.g. `/workspaces/:wsId/models/:modelId`) | Page re-loads cleanly without auth bounce |
| 14.3 | Sign out → sign back in | Lands on the workspace list |
| 14.4 | Workspace switcher → switch to a different workspace | Sidebar context updates (all workspace-scoped links repoint) |
| 14.5 | Browser back/forward through 5+ pages | History stays intact; no double-renders |

---

## What success looks like overall

- Every sidebar entry navigates without 404.
- Every new page has either real data or a clear empty state — no
  blank white screens, no "undefined" / "[object Object]" leakage.
- Promoting → registry → approval → execute is a coherent flow even
  if some buttons aren't yet wired into the trial detail page.
- `pytest` (158/158) and `npm run typecheck` + `npm test` (59/59)
  both clean.
- `pycaret-server doctor` reports green.

## What to expect to NOT work (known follow-ups)

- **Notebook backend in docker mode** — works in theory; not smoke-
  tested against a real Docker daemon on Windows.
- **Phase 6 engine-side event widening** — the Redis pub/sub bus is
  in place, but the engine still only emits the legacy events. Tune
  charts look the same as before.
- **Phase 9 schedule v2 UI** — the worker handlers exist (`retrain`,
  `drift_check`, `batch_predict`, `dataset_refresh`), but the
  schedules page doesn't yet let you pick those kinds. Wire by curl
  against `/api/v1/workspaces/<ws>/schedules`.
- **Helm chart** — files exist; not actually deployed to a real K8s
  cluster yet.
- **Airgapped install bundle** — described in INSTALL.md as a future
  cut.

If any of these unexpectedly break in subtle ways, capture the failing
HTTP call (URL + body) so we have a clean repro for the next session.
