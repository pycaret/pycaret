# Phases 8 / 11 / 13 — implementation notes (session 31)

The last three phases of the 0–14 roadmap. Notebooks, statistical
computing, enterprise polish — all shipped as foundation + API +
client + (where applicable) docs. UI pages per surface are the next
session's lift.

---

## Phase 8 — Notebook runtime

**Tables.** `notebooks`, `notebook_sessions`.

**Backend.**

- `pycaret_server.notebooks` package: `NotebookManager` protocol +
  two backends.
  - `LocalManager` (default) — fake-spawn for dev boxes without
    Docker. The session row shows ``running`` and the frontend
    renders a "backend unavailable" placeholder.
  - `DockerManager` — shells out to ``docker run`` to spawn one
    JupyterLab container per session with `--memory` / `--cpus`
    caps, allocates an ephemeral port, returns a token-bearing
    iframe URL.
- Selection via `PYCARET_NOTEBOOK_BACKEND=local|docker`. K8s pod-spec
  backend slots in as a third driver in a future cut without touching
  call sites.
- `api/notebooks.py`:
  - CRUD on `/projects/{id}/notebooks`, `/notebooks/{id}`.
  - Session lifecycle: `POST /notebooks/{id}/sessions` (start —
    only response that surfaces the JupyterLab token),
    `GET /notebooks/{id}/sessions`, `POST /sessions/{id}/heartbeat`,
    `DELETE /sessions/{id}`.
  - `reap_idle_sessions(db)` — stops sessions whose
    `last_active_at` is older than their timeout; called from a
    future cron Job.

**Frontend.** `notebooksApi` (`forProject`, `create`, `get`, `patch`,
`delete`, `startSession`, `listSessions`, `heartbeat`, `stopSession`)
+ `Notebook` / `NotebookSessionRow` / `NotebookSessionStart` types.

**What's deferred.** Notebook content sync — the `.ipynb` bytes
themselves. v1 the manager spawns a generic JupyterLab; the next cut
mounts the workspace's object store as `/data` and saves the
notebook back via the ObjectStore on container shutdown.

---

## Phase 11 — Statistical computing v1

**Tables.** `analyses`. Result rows reuse the existing `runs` table
(Phase 0 Run is already kind-agnostic — its `metrics` JSON carries the
result envelope, `params` carries the analysis input).

**Backend.**

- `pycaret_server.analyses` package:
  - `AnalysisResult` envelope (test_statistic, p_value, effect_size,
    CI, table, interpretation, Plotly figure, free-form extras).
  - 13 procedures wired end-to-end:
    - **Two-group means**: `ttest`, `welch_ttest`, `paired_ttest`,
      `mannwhitney`.
    - **Many-group**: `anova_oneway`, `kruskal`.
    - **Categorical**: `chi2` (with Cramér's V).
    - **Regression**: `ols` with full diagnostic suite (VIF,
      Durbin-Watson, residuals-vs-fitted figure).
    - **Survival**: `kaplan_meier` (single + grouped), `logrank`,
      `cox_ph`.
    - **Forecasting**: `arima`, `prophet` (optional extra).
  - Heavy deps (`statsmodels`, `lifelines`, `prophet`) import lazily
    so the base install stays slim.
- `api/analyses.py`:
  - `GET /analysis-kinds` — drives the New-Analysis wizard.
  - CRUD on `/projects/{id}/analyses`, `/analyses/{id}`.
  - `POST /analyses/{id}/run` — execute + persist as a Run row keyed
    by synthetic `experiment_id=analysis:<id>`.
  - `POST /analyses/run-once` — transient preview without persistence.
  - `GET /analyses/{id}/results` — past Run history.

**Frontend.** `analysesApi` (`kinds`, `forProject`, `create`, `get`,
`patch`, `delete`, `run`, `runOnce`, `results`) + `Analysis` /
`AnalysisResult` / `AnalysisKind` / `AnalysisRunResponse` /
`AnalysisRunRecord` types.

**What's deferred.** The per-kind UI page (a guided form per
procedure with column pickers + interpretation card). Backend is in
place; one component per kind tomorrow.

---

## Phase 13 — Enterprise polish

**Helm chart** — `infra/helm/pycaret/`.

- `Chart.yaml` + `values.yaml` (Postgres, Redis, MinIO, api,
  worker, web, ingress + observability knobs).
- Templates: `_helpers.tpl` (name / labels / connection URLs),
  `api-deployment.yaml`, `worker-deployment.yaml`,
  `web-deployment.yaml`, `dependencies.yaml` (opt-out
  Postgres/Redis/MinIO StatefulSets), `ingress.yaml`.
- Bring-your-own backends: set `*.enabled=false` on a sub-chart and
  supply `external*.url` instead. Doc'd in `INSTALL.md`.
- Phase 14 GPU pool: install a second release with
  `worker.queues=gpu` + `worker.resources.limits."nvidia\.com/gpu"=1`.

**CLI bootstrap.**

- `pycaret-server init [--data-dir ./data] [--force]` — writes a
  `.env` with random JWT + Fernet keys, applies migrations, prints
  the next step. Idempotent.

**Docs.**

- `INSTALL.md` — three install paths (single-process / Compose / Helm)
  with bring-your-own-backend recipes and the GPU pool example.
- `OPERATIONS.md` — backup (DB + object store; ordering matters),
  upgrade runbook (migrations first, drain workers, roll API),
  observability (`/healthz`, `doctor`, admin endpoints), scaling
  decision matrix per resource class, queue separation guide,
  security (JWT / Fernet rotation, JupyterLab tokens, approval
  workflows), and a troubleshooting matrix.

**What's deferred.** Airgapped install bundle (`docker save` every
image + `pip download` wheelhouse + bundled install script). Easy
to build once we publish images; the workflow is described in
`INSTALL.md` as a future cut.

---

## Migration head & bootstrap detector

One additional migration: `c3d4e5f6a8b9` adds `notebooks`,
`notebook_sessions`, `analyses`. Chains
`f0a1b2c3d4e5 → a1b2c3d4e5f7 → b2c3d4e5f6a8 → c3d4e5f6a8b9`.

Bootstrap detector recognises both `has_phase_8_11` (head) and the
previous Phase 4/5/7/10/12 fingerprint, so existing dev installs
upgrade in place.

---

## What this completes

After this session, **every phase 0 through 14 has shipped** — schema +
API + worker handlers + driver layer + frontend client surfaces for
the data path, plus a Helm chart + INSTALL/OPERATIONS docs to actually
deploy it.

The remaining work is **UI maturity** — per-phase pages on the
frontend (notebooks, analyses, registry detail, monitoring dashboard,
approvals inbox, lineage graph viewer, queue admin). Each is one
component-tree against the existing client methods. No more schema
changes needed for the 0-14 surface.
