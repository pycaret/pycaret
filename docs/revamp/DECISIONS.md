# PyCaret 4.0 — Decision Log

ADR-style. Newest at top. Each entry: **date | decision | alternatives considered | why**.

---

## 2026-04-27 (post-shipping · decision 2) · Engine moves from MIT to FSL-1.1-MIT

- **Decision:** Relicense the `pycaret` engine (`packages/engine/`) and the public `apps/site/` from **MIT** to **FSL-1.1-MIT** (Functional Source License with MIT Future grant) starting with **PyCaret 4.0**. The 3.x line on PyPI (`pycaret <= 3.4.0`) stays MIT — the relicense applies prospectively to 4.0+ only. Each released 4.0+ version auto-converts to plain MIT on the second anniversary of its release date. The platform packages (`services/api/`, `apps/web/`) stay dual-licensed under FSL-1.1-MIT OR BUSL-1.1.
- **Alternatives:**
  - **Stay on MIT** (rejected — gives the maintainer no commercial recourse if a competitor builds a hosted PyCaret-as-a-Service).
  - **Apache 2.0 + Commons Clause** (rejected — Commons Clause is widely seen as deceptive "open-source-but-not-actually" branding; FSL is more honest about the restriction).
  - **AGPLv3** (rejected — many enterprise legal teams blanket-block AGPL imports; for a `pip install` library, this would tank corporate adoption hard).
  - **Elastic License v2** (rejected — its "no offering as a service" trigger doesn't fire usefully for a library people import; FSL's "Competing Use" definition is the right shape for this product).
  - **PolyForm Small Business / Noncommercial** (rejected — too restrictive for a library; would block legitimate internal corporate use that we want to encourage).
- **Why:**
  - PyCaret is a Python *library* people `pip install`, not an application people deploy. The license has to be permissive enough that enterprise legal review doesn't blanket-block it. FSL passes most filters because it explicitly permits internal commercial use; what it forbids is *competing* commercial use. That's a narrow restriction in practice.
  - The two-year auto-MIT clause is the trust mechanism: every release becomes truly open-source eventually, so the ecosystem doesn't permanently fork. A user who's nervous about the FSL terms can pin to a 2+ year-old version, which is plain MIT.
  - The maintainer is the long-term steward and has stated commercial intent. FSL-1.1-MIT is the most-mainstream, least-adoption-hostile way to retain commercial leverage on the engine.
  - Aligns the engine's posture with the platform. Previously engine was MIT (any use) while platform was MIT-or-BSL — the asymmetry was confusing. Now both prevent direct competing-product hosting; both auto-convert to permissive eventually.
  - Owner answer: "MIT is too free, lets make it more restrictive so if there is commercial angle in this, i as a creator and maintainer benefit from it. ... FSL on engine."

---

## 2026-04-27 (post-shipping · decision 1) · Repo is Claude-Code-first; no CI bot auto-fixes issues

- **Decision:** Add `CLAUDE.md` at the repo root + per-major-directory, plus `.claude/{settings.json, commands/, agents/}`, to make the repo first-class for Claude Code. Contributors clone the repo, run Claude Code in their own checkout (using their own subscription / API key), pick a maintainer-`Approved` issue, and have the agent open a PR. **There is no `ANTHROPIC_API_KEY` in repo secrets and no GitHub Action that auto-fixes issues.** Compute is community-funded by whoever runs the agent.
- **Alternatives:**
  - **GitHub Action with maintainer-paid Anthropic API key, triggered on `Approved` label** (rejected — cost concentrates on the maintainer; opens API-abuse vectors via crafted issue payloads; the maintainer explicitly opted out: "i dont wanna put my own api key for any runners").
  - **Hybrid: action for trusted contributors + local for everyone else** (rejected — the trust gate adds operational complexity for marginal benefit; user explicitly chose pure-local).
- **Why:**
  - Distributes the compute cost across whoever wants to actually fix issues, instead of concentrating on the maintainer.
  - Fixes happen with the contributor's own context window, identity, and judgment — they own the PR they open.
  - The maintainer retains the merge gate. The bottleneck is intentional and is *human review*, not *agent execution*.
  - Owner answer: "everybody woudl do locally and use their own claude. when i m working i m using my own claude code windows. i dont wanna put my own api key for any runners. ... agent should be allowed to touch anything. eventually the right to approve merge to main branch is with me."

---

## 2026-04-24 (session 13 · restructure decision 4) · Product name is "PyCaret"; UI brand is "PyCaret Control Plane"

- **Decision:** The product — library + platform together — is branded **PyCaret**. The web UI's top-level title and marketing surface is **PyCaret Control Plane** (to communicate "this is not just the library, it's the managed platform"). Package names on registries stay simple: `pycaret` on PyPI (engine), `pycaret-server` on PyPI (backend), `@pycaret/ui` on npm (web app). OpenAPI `info.title` = "PyCaret Control Plane" so `/docs` looks like the right product.
- **Alternatives:** Rename the product entirely (`pycaret-platform`, `pycaret-cp`, etc.) — rejected, bad for brand continuity; silo the platform as a sub-brand — rejected, fragments community attention.
- **Why:** Keeps 10 years of PyCaret brand equity while distinguishing the new platform story. Package-name stability matters for PyPI import paths that are scattered across a decade of notebooks; product-name polish matters for the "open-source Databricks/DataRobot" positioning. Owner answer: "PyCaret Control Plane is fine for UI. The name itself is PyCaret only."

## 2026-04-24 (session 13 · restructure decision 3) · LLM router, not a single provider

- **Decision:** Build an `LLMRouter` abstraction in `services/api/pycaret_server/llm/` from day one. Implement **Anthropic (Claude)** and **OpenAI** as the two first-class backends. Every LLM consultation call flows through the router: provider selection driven by `LLMProviderSetting` rows; request shape normalised to chat-completion + tool-use; responses coerced into the `LLMConsultation` schema. Additional providers (Google / Azure / Ollama / custom OpenAI-compatible) are added by implementing one `LLMProvider` subclass.
- **Alternatives:** Ship Claude-only first and "add OpenAI in session 14" (rejected — architectural entrenchment tends to stick; cheaper to do the abstraction up front); pass a raw OpenAI client through (rejected — leaks the provider's quirks into every call site).
- **Why:** The platform is meant to be the open-source choice in an agentic-ML world; not supporting the two dominant LLM providers out of the box is a credibility problem. The abstraction cost is small (one dispatcher class, one `LLMProvider` Protocol, two implementations) and it keeps the LLM layer swappable as provider APIs evolve. Owner answer: "we should have a router that works with claude as well as openai api."

## 2026-04-24 (session 13 · restructure decision 2) · Electron desktop app is V2, not MVP

- **Decision:** Ship the desktop distribution in V2, after: (a) `services/api` is stable, (b) `apps/web` covers all 8 screens, (c) the engine has 4.0.0 on PyPI (not an alpha). The `apps/desktop/` directory is scaffolded now with a README stub explaining the scope, so the structure is locked in but no code is written yet.
- **Alternatives:** Ship Electron in the MVP (rejected — adds signed-installer tooling, per-OS CI, bundled-Python packaging, auto-update infra, all before the core web flow is polished); skip Electron entirely (rejected — a desktop installer is the cleanest "try it in 5 minutes" onboarding for analysts without Docker).
- **Why:** Local dev already works via `uv run pycaret-server serve` + `npm run dev`. Electron is a polish item, not a gating one; doing it in V2 lets the core product stabilise before we spend a session per OS on installers. Owner answer: "Lets roadmap Electron and we will do once our backend, frontend everything is stable, pycaret engine is mature and 4.0 released on pip."

## 2026-04-24 (session 13 · restructure decision 1) · Monorepo restructure — apps / services / packages / infra

- **Decision:** Adopt the canonical monorepo layout from `CONTROL_PLANE_SPEC.md § 19`:
  ```
  apps/            web, desktop (stub)
  services/        api, worker (stub), deployment-runtime (stub)
  packages/        engine, sdk-python (stub), shared-schemas (stub)
  infra/           docker, helm (stub), terraform/{aws,gcp,azure} (stubs)
  docs/            unchanged
  ```
  Concretely: `pycaret/` → `packages/engine/pycaret/`, `pycaret-server/` → `services/api/`, `pycaret-ui/` → `apps/web/`, `docker/` → `infra/docker/`. Root `pyproject.toml` becomes a pure workspace manifest (no package metadata); engine pyproject moves alongside the engine source at `packages/engine/pyproject.toml`.
- **Alternatives:** Keep flat (`pycaret-server`, `pycaret-ui` at root) and add missing pieces as top-level dirs as they arrive (rejected — drifts progressively from spec, every new piece forces a rename discussion); delay restructure until V2 (rejected — restructuring later = more churn, every session adds call sites to update).
- **Why:** "Wash away all the old sins" — the flat layout was an incremental artefact; the canonical structure is what the spec describes and what operators expect when they clone the repo. Python package names are unchanged (`import pycaret` / `import pycaret_server` still work exactly the same), so users + PyPI are unaffected. Notebook users are unaffected. Only the source-tree paths change, and that pain is paid once now instead of distributed across every future session. Owner answer: "Restructure now. … setup a solid foundation for structure now and wash away all the old sins."

---

## 2026-04-23 (session 6 · platform-plan decision 6) · Metrics storage = summary AND per-fold (both, comprehensive)

- **Decision:** Every Run stores two metric representations: (a) `runs.metrics_summary` — leaderboard-shape aggregates (one row per model, `mean_*` / `std_*` columns), (b) `fold_metrics` — per-fold × per-model × per-metric rows (`roughly n_models × n_folds × n_metrics` per Run).
- **Alternatives:** Summary only (smaller DB, but kills variance/stability analysis); per-fold only (forces every leaderboard query to aggregate at read time).
- **Why:** The summary drives the leaderboard screen; the per-fold table unlocks variance-across-folds plots, time-to-train analysis, stability checks, and "is this model truly better than the runner-up within CV noise?" comparisons. Storage cost is trivial relative to the fitted-pipeline pickles. Owner answer: "Both. very comprehensive so value can be realized."

## 2026-04-23 (session 6 · platform-plan decision 5) · Dual-licensed platform packages

- **Decision:** Engine `pycaret` stays MIT. Platform packages (`pycaret-server`, `pycaret-cli`, `pycaret-ui`) are dual-licensed: MIT for self-hosted and internal-enterprise use; Business Source License (BSL 1.1) for offering the platform as a multi-tenant hosted service to third parties. BSL auto-converts to MIT/Apache-2.0 after 3 years. A CLA is added to `CONTRIBUTING.md` for platform-package PRs.
- **Alternatives:** Everything MIT (no commercial-use gate for a future hosted SaaS); AGPL (too aggressive for enterprise adoption); everything BSL (hurts self-hosters' comfort).
- **Why:** Mirrors the posture of Sentry / Cal.com / Supabase / Plausible — credible OSS core + preserved commercial freedom for a future hosted layer. Self-hosters and internal-enterprise deployments are completely unaffected. Owner answer: "Yes."

## 2026-04-23 (session 6 · platform-plan decision 4) · In-house serving system, not MLServer / BentoML

- **Decision:** Platform owns its inference serving. Each deployed Pipeline becomes a `deployments` row with an `endpoint_slug`; `DeploymentRegistry` loads the pickle into the FastAPI process memory and a single catch-all route `POST /api/v1/deployments/{slug}/predict` serves inference. Per-deployment auth modes: `workspace` (JWT), `api-key` (`X-PyCaret-Key` header), `public` (rate-limited, opt-in). Per-deployment metrics: inference count, p50/p95 latency, error rate.
- **Alternatives:** MLServer (V2 protocol, production-grade but heavier dep surface); BentoML (own DSL, lock-in); Seldon-Core (K8s-only); no serving (force users to roll their own).
- **Why:** Owning the serving surface means end-to-end UX consistency and no third-party dep to reason about. V1 is deliberately simple (single-process, in-memory pipeline load); isolation / autoscaling come later. Target users — teams under ~20 — don't need Seldon complexity. Owner answer: "lets build a in house serving system rather than using MLServer or BentoML. no need."

## 2026-04-23 (session 6 · platform-plan decision 3) · Pipelines are workspace-scoped and shareable across projects

- **Decision:** `Pipeline` is a workspace-level object (not project-scoped). Projects reference pipelines via a many-to-many `pipeline_project_links` table. Workspace gets a top-level "Pipelines" screen; Project experiment view has a "Use an existing Pipeline" selector; Deployment is a workspace-level action.
- **Alternatives:** Project-scoped only (simpler, but blocks reuse across related projects); per-user scoping (unnecessary for v1).
- **Why:** Model-registry pattern. A team often has one pipeline ("churn model v2") that multiple projects consume. Owner answer: "Yes" to "do we expose Pipelines as a first-class shareable object across projects?"

## 2026-04-23 (session 6 · platform-plan decision 2) · v1 data-source connectors = CSV upload, S3, Postgres

- **Decision:** v1 ships three connectors — CSV upload (via UI), S3 (read-only: list + sample + load CSV/Parquet), Postgres (read-only: list tables + load). A `DataSourceConnector` ABC is in place from v1 so Snowflake / Google Sheets / MySQL can land later without core changes.
- **Alternatives:** CSV only v1 (too limited given imminent AWS deployment); ship all of Snowflake / BigQuery / GSheets v1 (scope creep).
- **Why:** AWS is the immediate deployment target, so S3 is non-negotiable. Postgres is the most common internal-DB source for analyst workflows. CSV upload covers everything else for quick testing. Owner answer: "for connectors for now lets build few. once everything work locally i will immediately test it by deploying this on AWS."

## 2026-04-23 (session 6 · platform-plan decision 1) · Run notebooks are first-class artifacts

- **Decision:** Every Run persists an artifact bundle: `run.ipynb` (executable, generated from config + event stream), `fitted_pipeline.pkl`, `leaderboard.json`, `events.jsonl`, `preview.html` (pre-rendered). Artifacts are immutable per Run, downloadable, shareable via signed URL, previewable in-app without download. Storage: local disk v1; S3-backed when deployed.
- **Alternatives:** Event-stream replay only (doesn't give users a tangible notebook they can send to a colleague); static HTML only (loses reproducibility).
- **Why:** Users reach for notebooks for sharing, debugging, and reproducibility. A modern SaaS stores these as first-class objects — not just logs. Owner answer: "do what is expected from a modern SaaS."

---

## 2026-04-23 (session 3) · Functional API deleted wholesale — PyCaret 4.0 is OOP-only

- **Decision:** All 5 `pycaret/*/functional.py` files (~11,300 LOC across 145 module-level functions) were deleted. The canonical 4.0 API is `pycaret.tasks.{Task}Experiment(...)`. `set_current_experiment` / `get_current_experiment` / the `ContextVar` state machinery (`pycaret/core/state.py`) were also deleted.
- **Alternatives:** Keep functional as a thin shim layer; preserve 3.x call shape via a wrapper that constructs an `Experiment` internally; provide a deprecation cycle.
- **Why:** The user said it plainly: "nobody will migrate 3→4. 4 in my mind is totally new thing. I really would like to get rid of 90% tech debt now." The functional layer was pure duplication — every function forwarded kwargs to an OOP method while re-declaring the docstring. Keeping it meant carrying 11,300 LOC of pass-through code and an entire class of drift-bugs between `functional.py` and `oop.py` signatures. The OOP cost is exactly one extra line per script; in exchange users get a typed object they can inspect, clone, pickle, pass around, and — critically — use from a React UI / LLM agent without caring about "which experiment is implicitly current."
- **Knock-on deletions:** 41 functional-API-coupled test files (replaced by 4 OOP-native files with 100% pass rate), 6 collapsed task-module `__init__.py`s, the `_CURRENT_EXPERIMENT` reset fixture in `conftest.py`.

## 2026-04-23 (session 3) · `TSForecastingExperiment` → `TimeSeriesExperiment`

- **Decision:** The 3.x class `TSForecastingExperiment` is renamed to `TimeSeriesExperiment` in the 4.0 public API. The old name is *not* aliased — importing it raises `ImportError`.
- **Alternatives:** Keep the 3.x name; add an alias for compat.
- **Why:** 4.0 is a clean break (no migration expected). `TimeSeriesExperiment` matches the task's module name (`pycaret.time_series`) and the pattern of the other 4 task classes. The legacy class `TSForecastingExperiment` still exists inside `pycaret/time_series/forecasting/oop.py` as the delegation target.

## 2026-04-23 (session 3) · Split `Experiment` into `SupervisedExperiment` + `UnsupervisedExperiment`

- **Decision:** Supervised-only verbs (`compare_models`, `tune_model`, `ensemble_model`, `blend_models`, `stack_models`, `calibrate_model`, `finalize_model`, `interpret_model`, `automl`, `get_leaderboard`) moved from the `Experiment` base to a `SupervisedExperiment` subclass. `UnsupervisedExperiment` is the sibling that adds `assign_model`. Task subclasses inherit from the appropriate intermediate.
- **Alternatives:** Keep all verbs on `Experiment` and have unsupervised tasks override with `NotImplementedError`; use a class-level `_supports` registry.
- **Why:** Static typing clarity. A user inspecting `ClusteringExperiment` via IDE autocomplete should see only verbs that work. `NotImplementedError`-at-call-time is a worse UX; discovery in the type system is strictly better.

## 2026-04-22 (session 2) · Engine architecture = `Experiment(BaseEstimator)` + delegation to legacy god-class during migration

- **Decision:** The PyCaret 4.0 engine core is a single `Experiment` class that subclasses `sklearn.base.BaseEstimator`. Task subclasses in `pycaret.tasks.*` pre-configure it for classification/regression/clustering/anomaly/time-series. The legacy `_SupervisedExperiment` god-class is retained as `self._legacy` and every verb delegates to it during the multi-session migration; each verb is progressively rewritten on top of `sklearn.pipeline.Pipeline` + `sklearn.model_selection` and the delegation call is replaced in place.
- **Alternatives:** (a) rewrite every verb at once (too big to land in one session without breaking the golden path); (b) keep functional-only API (rejected — user said revamp); (c) use composition rather than subclassing for `Experiment` (rejected — `BaseEstimator` subclassing gives us clone, repr, get_params, and `__sklearn_tags__` for free).
- **Why:** This pattern lets the notebook golden path keep working at every intermediate commit while the new architecture absorbs responsibilities one at a time. Users/agents who adopt `pycaret.tasks.ClassificationExperiment` today get a proper sklearn-compatible object that returns typed results and emits events — while the internals are still the 3.x implementation underneath.

## 2026-04-22 (session 2) · Event-stream logger in-process, not a tracker adapter

- **Decision:** `pycaret.logging` emits structured `Event` dataclasses through a `BaseLogger` hook surface with `subscribe(callback)` for fan-out. Default `MemoryLogger` keeps events in a thread-safe list and optionally tees to a JSONL file. mlflow/comet/wandb/dagshub stay cut (session 1); if someone wants to tee events to an external tracker in 4.1 they subclass `BaseLogger`.
- **Alternatives:** Python `logging` module with a custom formatter (text-oriented, loses structure); OpenTelemetry traces (heavy dep, overkill); re-add mlflow as a soft extra (explicitly on the kill list).
- **Why:** The React UI plus LLM agents both want structured, subscribable events — not log strings. A dataclass-based stream is the minimum viable surface that works for both consumers.

## 2026-04-22 (session 2) · Every verb returns a typed dataclass, not bare DataFrame / model

- **Decision:** `compare_models` returns `CompareResult(best, models, leaderboard, ranked_ids, events)`. `create_model` returns `CreateResult(pipeline, model_id, metrics, params, events)`. Etc. Notebook users who wrote `best = compare_models()` can reach `best = compare_models().best` or iterate the `CompareResult` directly (it implements `__iter__` / `__getitem__` for drop-in compat with `top3 = compare_models(n_select=3)` list-indexing).
- **Alternatives:** Continue returning bare lists of sklearn estimators; mutate `exp.pull()` as the only way to get metrics.
- **Why:** The UI/agent surface wants a single object whose fields are all independently useful. Breaking the `compare_models() -> list` convention is the least-change path that still gives us structured returns.

## 2026-04-22 · tbats and statsforecast demoted from `timeseries` extra

- **Decision:** Both packages were removed from the `timeseries` extra in `pyproject.toml`. BATS/TBATS model containers in `pycaret/containers/models/time_series.py` now try-import their sktime wrappers and mark themselves `inactive=False` silently if the underlying package is missing.
- **Alternatives:** Pin `numpy<2` in the extras (regresses modernity); keep statsforecast and ship a C-compiler dependency.
- **Why:**
  - `tbats` declares `numpy<2` at runtime, which is incompatible with the NumPy 2.x modernization. Users who specifically want BATS/TBATS can `pip install tbats` manually (and accept the numpy downgrade) — the container code will then light up automatically.
  - `statsforecast` has no prebuilt wheel for Python 3.14 on Windows. It isn't imported anywhere in pycaret source — it was only referenced by container id strings.

## 2026-04-22 · Strip mlflow/boto3 imports from tests rather than skip-gate them

- **Decision:** In `tests/test_{classification,regression,clustering,anomaly}.py`, delete the `MlflowClient`-based test methods outright. In `tests/test_persistence.py`, replace the file with a stub comment (all tests in it were S3/boto3-specific). Delete `tests/test_clustering_engines.py` (sklearn-intelex-only).
- **Alternatives:** Wrap each mlflow/boto3 test in `pytest.importorskip()`.
- **Why:** mlflow and boto3 are on the 4.0 kill list. Skipping would leave dead test code referencing dead features. Clean delete keeps the suite honest about what 4.0 actually ships.

## 2026-04-22 · Python / sklearn floor = 3.11 / sklearn 1.7 (transitional)

- **Decision:** `requires-python = ">=3.11"`; `scikit-learn>=1.7,<1.8`; primary dev = **Python 3.13**. CI matrix: 3.11 / 3.12 / 3.13.
- **Alternatives:** 3.10 floor; pin sklearn to 1.8 and drop `timeseries` extra.
- **Why:** Two real-world constraints discovered during Phase 0:
  - `sktime` (the `timeseries` extra) requires `scikit-learn<1.8`, so a core requirement of `>=1.8` makes the `timeseries` extra unresolvable. Pin is transitional — when sktime declares sklearn 1.8 support, we bump.
  - Python 3.14 introduces PEP 649 deferred annotations, which breaks joblib + cloudpickle pickling (see `thinking/2026-04-22_python314_pep649_blocker.md`). 3.14 stays in the classifiers as aspirational; dev and CI settle on 3.13.

## 2026-04-22 · Python / sklearn initial target (superseded same day)

- **Originally decided:** `requires-python = ">=3.11"`, `scikit-learn>=1.8`, primary dev = 3.14.
- **Superseded by the entry above** after Phase 0 smoke testing surfaced the sktime cap and the 3.14/PEP-649 blocker.

## 2026-04-22 · Build backend = hatchling; env + lock = uv

- **Decision:** Move from `setuptools` backend to `hatchling`; single source of truth is `pyproject.toml`; lockfile = `uv.lock` committed; `setup.cfg` and `MANIFEST.in` removed.
- **Alternatives:** Keep setuptools (minimal churn); `uv_build` backend (too new).
- **Why:** User asked for uv. Hatchling is uv's recommended partner, widely adopted, and eliminates the legacy `setup.cfg` dual-config pattern that's half-broken today.

## 2026-04-22 · Break backward compatibility freely except for the notebook golden path

- **Decision:** Any internal API is fair game. The only stable surface between 3.x and 4.0 is: `setup`, `create_model`, `compare_models`, `tune_model`, `plot_model`, `predict_model`, `save_model`, `load_model`, `finalize_model`, `blend_models`, `stack_models`, `ensemble_model`, `interpret_model` — per module (classification/regression/clustering/anomaly/time_series).
- **Alternatives:** Deprecation cycle with warnings.
- **Why:** 3.x has been unmaintained for ~3 years; there is no active deprecation story to respect. Deprecation shims would reintroduce the tech debt we're removing.

## 2026-04-22 · Cut mlflow/comet/wandb/dagshub; build a lean internal logger

- **Decision:** Remove the entire `pycaret/loggers/` package of external trackers. Introduce `pycaret/logging/` — file + in-memory logger with a JSON event stream, designed to be consumed by the forthcoming React UI and by LLM agents.
- **Alternatives:** Keep mlflow behind an extra.
- **Why:** The user is building the React UI on top and wants the engine to own its event stream. External trackers are out of scope for 4.0; re-add as thin adapters in 4.1 if someone needs them.

## 2026-04-22 · Cut parallel (fugue/dask/ray/distributed) entirely — no replacement

- **Decision:** Delete `pycaret/parallel/`, `pycaret/internal/parallel/`, `parallel_backend` arguments, and all `*_parallel` tests. No replacement.
- **Alternatives:** Hide behind an optional extra.
- **Why:** User: "nobody uses that anyways". Keeping it alive across the sklearn 1.8 upgrade is meaningful cost for ~zero benefit.

## 2026-04-22 · Cut yellowbrick; rewrite plots in Plotly from scratch

- **Decision:** Remove yellowbrick dep, `patches/yellowbrick.py`, `plots/yellowbrick.py`, and the 16 inline yellowbrick imports in `tabular_experiment.py`. Rewrite each plot as a Plotly function in a new `pycaret/plots/` module.
- **Alternatives:** Vendor yellowbrick.
- **Why:** Yellowbrick has not kept pace with sklearn — it's a compat blocker for the 1.8 migration. Plotly rewrites look better, match the React UI design system, and remove a dep with its own dep chain (matplotlib).

## 2026-04-22 · Cut deploy/app/api/dashboard/drift/fairness helpers from the engine

- **Decision:** Remove `create_api`, `create_app`, `create_docker`, `dashboard`, `check_drift`, `check_fairness`, `eda`, `convert_model`, `deploy_model`. Corresponding tests deleted.
- **Alternatives:** Move to separate `pycaret-extras` package.
- **Why:** These pulled in fastapi, uvicorn, gradio, docker, boto3, evidently, fairlearn, ydata-profiling, explainerdashboard, m2cgen — ~10 heavy deps for features that are either UI-duplicated or niche. If anything comes back, it goes in a separate package so core stays lean.
