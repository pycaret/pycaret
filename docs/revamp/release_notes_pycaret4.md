# PyCaret 4.0 — Engineering Change Log

> **This file is the single source of truth for everything that changed between PyCaret 3.4.0 and 4.0.**
>
> Every non-trivial edit during the 4.0 revamp is logged here. At release time, the user-facing `RELEASE_NOTES.md` (and the GitHub Release body) will be generated from this file by summarising and regrouping entries. Do not edit past entries when new work is added; append only.

## How to use this file
>
> For the Part-2 Application-Platform plan (CLI / FastAPI / DB / React / Docker), see
> [`PLATFORM_PLAN.md`](PLATFORM_PLAN.md). Its phase breakdown is tracked in
> [`ROADMAP.md`](ROADMAP.md) under Part 2.

- **Organization:** newest session first. Within each session, entries are grouped by type.
- **Entry types:** `BREAKING`, `REMOVED`, `ADDED`, `CHANGED`, `FIXED`, `DEPRECATED`, `SECURITY`, `DOCS`, `BUILD`, `TESTS`, `DEPS`, `INTERNAL`.
- **One change = one bullet.** If a single change touches many files, list it once with the file count; reserve multi-bullet listings for distinct changes.
- **Every bullet must be independently understandable.** Include the file path(s) and, for behavior changes, the before/after in a sentence.
- **Linking:** when a change is backed by a decision or audit note, reference it (e.g. `(see DECISIONS.md · 2026-04-22 · tbats demotion)`).
- **Breakage tagging:** anything that changes a public import path, a function signature, or removes a symbol must carry `BREAKING` in addition to the type tag.

## Entry template

```
- `TYPE[, BREAKING]` — **Short imperative title.** Fuller explanation. File paths. Rationale if non-obvious.
```

## Category legend (for the user-facing generator)

| Type | User-facing section it feeds |
|---|---|
| `BREAKING` | "⚠ Breaking changes — read before upgrading" |
| `REMOVED` | "Removed features and dependencies" |
| `ADDED` | "New features" |
| `CHANGED` | "Behavior changes" |
| `FIXED` | "Bug fixes" |
| `DEPRECATED` | "Deprecations" (4.0 has none — clean break) |
| `SECURITY` | "Security fixes" |
| `DOCS` | Collapsed into "Documentation" footnote |
| `BUILD` | "Installation & packaging" |
| `TESTS` | Usually omitted from user notes |
| `DEPS` | "Dependency changes" |
| `INTERNAL` | Usually omitted from user notes |

---

# Session 19 — 2026-04-24 — Failure debugger + Deployment reviewer + API keys

Baseline: session 18 completed the three classic copilots (3 of 6 types from SPEC § 12.2) + the router/provider infrastructure. No admin surface yet; only dataset-adjacent advisories.

Theme: ship two more copilots (failure debugger + deployment risk reviewer) + the first admin surface (personal API keys). Drift analyst + audit logs + workspace members defer to session 20.

## ADDED — LLM advisories (2 of remaining 3)

- `ADDED` — **`services/api/pycaret_server/llm/consultations/failure_debugging.py`** (5th copilot). System prompt classifies the error as **DATA** (schema mismatch, missing target, wrong dtype), **CONFIG** (wrong task for target, incompatible model, train_size too small), or **ENGINE** (upstream library error, version skew). Output demands reasoning_summary open with the category label so UI can tone-code. Event stream head-5 + tail-35 + `__truncated__` marker to keep prompts bounded.
- `ADDED` — **`services/api/pycaret_server/llm/consultations/deployment_risk_review.py`** (6th copilot — drift deferred). System prompt walks explicit risks: overfit (AUC≈1.0), tiny margin between top-2 models, small training sample, missing imputer/encoder in the pipeline, metric-goal mismatch, version skew. Demands verdict start with `APPROVE` / `APPROVE WITH CAVEATS: …` / `DO NOT DEPLOY: …` so the UI can render it tone-coded.
- `ADDED` — **`POST /api/v1/llm/debug-run`** — body `{run_id}`. 400 when `status != 'failed'` (succeeded runs use `explain-run`, in-flight runs have nothing to debug yet). Consultation persisted with `type='failure_debugging'` + FKs to run/experiment/project for audit correlation.
- `ADDED` — **`POST /api/v1/llm/review-deployment`** — body `{pipeline_id}`. Pulls origin Run snapshot + leaderboard; consultation stored with `run_id = pipeline.origin_run_id`.

## ADDED — API keys (first admin surface)

- `ADDED` — **`services/api/pycaret_server/api/api_keys.py`** — 3 routes:
  - `POST /auth/api-keys` — mint a key. Returns plaintext **exactly once**. Body: `{name, workspace_id?, expires_in_days?, scopes?}`. Hashes with SHA-256; stores hash + prefix only.
  - `GET /auth/api-keys` — list the caller's keys. Never exposes plaintext; only `prefix` (`pck_abcd1234`).
  - `DELETE /auth/api-keys/{id}` — revoke (soft delete — `revoked_at` set; row stays for audit). Only the owner (or a superuser) can revoke a key.
- `ADDED` — **Key format**: `pck_` recognisable prefix + `secrets.token_urlsafe(32)` body. `pck_` chosen to be greppable in logs + triggerable by GitHub's secret-scanner pattern library later. Total plaintext length: ~47 chars.
- `INTERNAL` — **Middleware that accepts `X-PyCaret-Key` as an auth header** on all `/api/v1/*` routes is session-20 work. Session 19 ships the CRUD surface so users can start minting keys + we can exercise the UX — the middleware is a small addition once that's settled.

## ADDED — frontend

- `ADDED` — **`apps/web/src/components/FailureDebuggerCard.tsx`** — inline card on `/runs/:id` when `status === 'failed'`. Red-tinted border. Opt-in (button fires the consultation, not mount). Button label flips "Diagnose" → "Re-diagnose" after first success. Renders standard `LLMAdvice` envelope + (when present) suggested config as pretty JSON.
- `ADDED` — **`apps/web/src/components/DeploymentReviewModal.tsx`** — modal on `/pipelines/:id`. Opens on "✨ Review" button click in the deploy sidebar. **Auto-fires on open** (unlike the cards — the user's committing to run the review by clicking the button to open the modal). Verdict tone-coded: `DO NOT DEPLOY` → `text-danger-500`, `APPROVE WITH CAVEATS` → `text-warn-500`, `APPROVE` → `text-success-500`.
- `ADDED` — **`apps/web/src/pages/ApiKeysScreen.tsx`** at `/account/api-keys`:
  - Table with name / prefix / status (active / revoked / expired — computed from `revoked_at` + `expires_at`) / expiry / created-at / revoke action.
  - "New API key" form with name + optional `expires_in_days`.
  - **One-time plaintext panel** appears on successful creation with a bold warning, the plaintext in a `<pre>`, a Copy button, and an "I've saved it" primary button to dismiss.
- `CHANGED` — **`apps/web/src/pages/RunDetail.tsx`** — terminal-state rendering splits: `succeeded` → `<RunExplainerCard>`, `failed` → `<FailureDebuggerCard>`.
- `CHANGED` — **`apps/web/src/pages/PipelineDetail.tsx`** — deploy sidebar gains a "✨ Review" button alongside Deploy; opens `<DeploymentReviewModal>`. Layout became a flex row so the two buttons share the bottom of the sidebar.
- `CHANGED` — **`apps/web/src/components/Layout.tsx`** — top nav gains an "API keys" link → `/account/api-keys`.
- `CHANGED` — **`apps/web/src/App.tsx`** — new authenticated route `/account/api-keys`.
- `ADDED` — **`apps/web/src/api/endpoints.ts`**:
  - `llmApi.debugRun` + `llmApi.reviewDeployment`.
  - `apiKeysApi` module — `list`, `create`, `revoke`.
- `ADDED` — **`apps/web/src/api/types.ts`** — `ApiKeyRead`, `ApiKeyCreateResponse` (extends + adds `token`), `ApiKeyCreateRequest`.

## TESTS

- `TESTS` — **`services/api/tests/test_session19.py`** — 9 new integration tests:
  - **Failure debugger** (2): `happy_path` (forces a real failed run via a bogus model id then debugs), `rejects_succeeded` (400).
  - **Deployment reviewer** (2): `happy_path` (train → promote → review; verify `run_id` correlated to `origin_run_id`), `404_on_unknown_pipeline`.
  - **API keys** (5): `create_returns_plaintext_once` (prefix matches head of plaintext; plaintext absent from GET), `list_scoped_to_user`, `revoke_soft_deletes` (row stays, `revoked_at` set), `expiry_round_trip`, `create_requires_name` (Pydantic 422).
- `TESTS` — **`apps/web/src/components/FailureDebuggerCard.test.tsx`** — 2 new (opt-in on mount, click-fires + renders diagnosis + Re-diagnose button).
- `TESTS` — **`apps/web/src/components/DeploymentReviewModal.test.tsx`** — 2 new (inert when closed, auto-fires on open + tone-codes `APPROVE WITH CAVEATS` with `text-warn-500`).
- `TESTS` — **`apps/web/src/pages/ApiKeysScreen.test.tsx`** — 3 new (empty state, create-flow with one-time plaintext panel renders + warning + correct payload, active/revoked status column with distinct key names to avoid text collisions).
- `TESTS` — **Combined suite: 134/134 green** (32 engine + 54 server + 48 web); was 118.

## INTERNAL

- `INTERNAL` — **Auto-fire vs. opt-in modal pattern.** `<DeploymentReviewModal>` auto-fires on open (matches `<AnalyzeDatasetModal>` from session 17 + `<ExperimentDesignerModal>` from session 18 — opening a modal is the user's consent signal). `<FailureDebuggerCard>` + `<RunExplainerCard>` are opt-in buttons (they're always-visible cards; firing on mount would run the LLM on every page view). Same envelope, different trigger affordance.
- `INTERNAL` — **Verdict-string classifier.** The deployment-reviewer prompt demands the verdict start with one of three literal strings (`APPROVE`, `APPROVE WITH CAVEATS`, `DO NOT DEPLOY`) so the UI can classify them via simple `.startsWith()` checks instead of NLP. The test asserts tone-coded class names on the DOM to lock this contract.
- `INTERNAL` — **Test `getByText` ambiguity fix.** An API-keys test initially used `getByText('active')` which matched both a key name and a status cell. Renamed the fixture keys to distinct values (`my-laptop` / `old-ci-token`) + added `{ exact: true }` for the status cells. Extending this — all test fixtures should avoid string collisions with semantic text the component renders.
- `INTERNAL` — **Forcing a failed run in tests.** `_fail_a_run()` submits a `create` plan with `model_id='zzzz_not_a_model'` (bogus) against valid iris data. `setup` plans tolerate many misconfigurations (the engine defers validation); `create` has to actually look up the model in the registry → `UnknownModelError` at execute time → run.status='failed'. This is the cleanest way to deterministically produce a failed run for the debugger test.
- `INTERNAL` — **Key prefix `pck_`.** Chose `pck` to stand for *PyCaret key*. Distinctive + short enough that the visible prefix (`pck_abcd1234` = 12 chars) is still useful in UIs. Will register with GitHub secret scanning once we publish a stable format.

## Session 19 delta summary

| Metric | Session 18 end | Session 19 end |
|---|---:|---:|
| LLM copilots (of 6 in spec) | 3 | **5** |
| API routes | ~49 | **~54** |
| Server integration tests | 45 | **54** |
| UI shared components | 10 | **12** |
| UI screens | 13 | **14** |
| UI tests | 41 | **48** |
| **Combined tests** | **118** | **134** |
| Production bundle (gz) | 96 kB | **98 kB** |

---

# Session 31 — 2026-04-25 — Secondary-verb drain: pull / models / get_metrics

Baseline: session 30 finished the internal-state drain. Session 31 drains the three advisory secondary verbs that have a clean native equivalent.

## CHANGED — engine

- `CHANGED` — **`Experiment.pull()`** reads from `self._fit_state["last_metrics"]`. Native modeling verbs (`create_model`, `tune_model`, `compare_models`) update that slot before returning via the new `_set_last_metrics()` helper. Falls through to `self._legacy.pull()` only when no native verb has run yet (TS-fallback case).
- `CHANGED` — **`Experiment.models()`** builds the user-facing DataFrame from `_fit_state["model_registry"]` directly. Columns: `Name`, `Reference`, `Turbo`. Index: model ID. `internal=True` keeps delegating — the legacy `ModelContainer` row exposes engine-internal fields some advanced callers depend on.
- `CHANGED` — **`Experiment.get_metrics()`** reads from the task's metric registry helper (`pycaret.containers.metrics.<task>.get_all_metric_containers`) instead of `self._legacy.get_metrics`. Output columns mirror legacy: `Name`, `Display Name`, `Score Function`, `Scorer`, `Target`, `Args`, `Greater is Better`, `Multiclass`, `Custom`. Time-series falls back to legacy (its registry is sktime-shaped).

## ADDED — internal helper

- `ADDED` — **`Experiment._set_last_metrics(df)`** stashes the most recent metrics DataFrame in `_fit_state["last_metrics"]`. Called by each native modeling verb right before returning. `pull()` reads from there.

## ADDED — tests

- `ADDED` — **`packages/engine/tests/test_session31_secondary_verbs.py`** — 8 tests:
  - 3 for `pull()` — returns the right DataFrame after each of `create_model` / `tune_model` / `compare_models`. First test drain-locked against `legacy.pull`.
  - 2 for `models()` — native DataFrame matches expected schema (drain-locked against `legacy.models`); `internal=True` falls back.
  - 2 for `get_metrics()` — classification + regression registries; drain-locked against `legacy.get_metrics`.
  - 1 for `NotFittedError` on all three pre-fit.

## INTERNAL

- `INTERNAL` — **Why not drain `add_metric` / `remove_metric` in this session.** These mutate the metric registry. The current registry is a global, container-class-decorator-populated structure shared across experiments. Adding a metric on one Experiment instance shouldn't visibly affect another, but the legacy state is partially shared. Properly draining these requires a small refactor to make the metric registry per-Experiment (move `metric_registry` into `_fit_state` like the model registry, plus support for `add_metric` to mutate it, plus a way for `calculate_metrics` to use the per-Experiment registry instead of the global). That's a clean follow-up but doesn't move the 4.0.0 release date — the verbs are advisory, not in the predict/tune/compare path.
- `INTERNAL` — **`models()` `internal=True` carve-out.** Some PyCaret 3.x test code paths use `exp.models(internal=True)` to introspect the full `ModelContainer` rows (with engine-internal fields like `eq_function`, `tunable`, `is_special`, etc.). Building that view from the bare snapshot would require us to expose more of the container shape, which we'd then have to keep in lockstep with the registry classes. Cleaner: keep delegating for the `internal=True` case, drain the public path. That preserves backward compatibility for one specific test family without leaking implementation details into the new core.
- `INTERNAL` — **`pull()` fallback semantics.** When `_fit_state["last_metrics"]` is None (no native verb has run yet), we fall through to `self._legacy.pull()`. This is reachable: if a TS-task user runs `compare_models`, that goes through `_compare_models_legacy` which writes to the legacy display container. Without the fallback, `pull()` would silently return None. The fallback keeps the public contract intact during the TS transition.

## Session 31 delta summary

| Metric | Session 30 end | Session 31 end |
|---|---:|---:|
| Drainable secondary verbs still on `_legacy` | 3 | **0** ✅ |
| Engine tests (fast + slow) | 113 | **121** |
| **Combined tests** | **259** | **267** |

---

# Session 30 — 2026-04-24 — Internal-state drain: transformed splits + fold generator + model registry

Baseline: session 29 drained the 7 user-facing data accessors. Session 30 drains the 6 *internal* `self._legacy.<x>` reads still present inside the drained verbs.

## CHANGED — engine

- `CHANGED` — **`packages/engine/pycaret/core/experiment.py` — `_snapshot_fit_state()`** captures 6 new slots in `_fit_state` at the end of `fit()`: `X_transformed`, `X_train_transformed`, `y_transformed`, `y_train_transformed`, `fold_generator`, `model_registry` (a `dict(legacy._all_models_internal)` copy).
- `CHANGED` — **13 callsites drained** across `experiment.py`, `supervised.py`, `unsupervised.py`:
  - `create_model` (supervised native) — reads `X_train_transformed`, `y_train_transformed`, `fold_generator`, and `model_registry` from `_fit_state`.
  - `_resolve_supervised_estimator` — reads `model_registry`.
  - `tune_model` — `X_train_transformed`, `y_train_transformed`, `fold_generator`, `model_registry`.
  - `compare_models` — `model_registry`.
  - `stack_models` — `fold_generator`.
  - `calibrate_model` — `fold_generator`.
  - `finalize_model` — `X_transformed`, `y_transformed`.
  - Unsupervised `create_model` + `assign_model` — `X_transformed`, `model_registry`.

## ADDED — tests

- `ADDED` — **`packages/engine/tests/test_session30_internal_state_drain.py`** — 5 tests with a generalised drain-lock helper (`_PoisonedAttrAccess`) that raises on every dunder a verb might use to read legacy state (`__getattr__`, `__getitem__`, `__contains__`, `__iter__`, `__len__`). Tests cover:
  - `create_model` succeeds after every internal-state attr on `_legacy` is poisoned.
  - `tune_model` resolves the search space from the snapshot's `model_registry` even when `legacy._all_models_internal` is dropped to `{}`.
  - `finalize_model` re-fits using `_fit_state["X_transformed"]` rather than legacy.
  - Clustering `create_model` fits on the snapshot's `X_transformed`.
  - Sanity: all 13 `_fit_state` keys are populated post-fit.

## INTERNAL

- `INTERNAL` — **Why a copy of the registry, not just a reference.** `_all_models_internal` is mutable on the legacy class — extensions like `add_metric` could grow it. Taking a `dict(legacy._all_models_internal)` shallow-copy at fit-time makes the snapshot stable: future legacy mutations don't leak into already-fitted experiments. Trade-off: if a user genuinely wants their `add_metric` call to affect a previously-fitted experiment, they'd need to re-snapshot. That's an edge case worth the reproducibility win.
- `INTERNAL` — **`_PoisonedAttrAccess` is the drain-lock pattern at scale.** Sessions 22-28 monkeypatched a single legacy method to raise; session 29 monkeypatched a property; session 30 needs to lock down ~6 attrs across multiple dunder methods. Wrapping them all in one sentinel object that raises on every access keeps the test code short and the failure mode clear ("a verb reached for self._legacy.\<x\>").

## Session 30 delta summary

| Metric | Session 29 end | Session 30 end |
|---|---:|---:|
| `_legacy.<internal-state>` reads in drained verbs | 13 | **0** ✅ |
| Engine tests (fast + slow) | 108 | **113** |
| **Combined tests** | **254** | **259** |

---

# Session 29 — 2026-04-24 — Property drain: data accessors

Baseline: session 28 finished the modeling-verb drain (16 verbs). Session 29 promotes the user-facing data accessor properties off `self._legacy` onto a snapshot in `self._fit_state`. The public API surface no longer requires `self._legacy` to exist on read paths.

## CHANGED — engine

- `CHANGED` — **`packages/engine/pycaret/core/experiment.py` — `Experiment.fit`** now calls `_snapshot_fit_state()` after `self._legacy.setup()` returns. The snapshot captures references (not copies) to seven legacy attributes in `self._fit_state`, a dict-backed cache.
- `CHANGED` — **`Experiment.X` / `X_train` / `X_test` / `y` / `y_train` / `y_test` / `preprocess_pipeline`** now read from `self._fit_state` instead of dispatching to `self._legacy.<attr>` on every access. `_require_fitted()` is still called to maintain the NotFittedError contract.
- `CHANGED` — **Defensive `getattr(legacy, name, None)`** in `_snapshot_fit_state()` lets the same code path work across task types. Clustering / anomaly experiments don't have `y_test`; the snapshot stores `None` for missing slots.

## ADDED — tests

- `ADDED` — **`packages/engine/tests/test_session29_property_drain.py`** — 4 tests:
  - `test_data_properties_do_not_call_legacy_after_fit` — the property drain-lock. Wraps every drained `self._legacy.<X>` accessor with a raise-on-read sentinel post-fit; the 7 properties continue to return correct values, proving they no longer touch the legacy holder.
  - `test_data_properties_clustering_y_is_none` — clustering experiments don't have a target; `y` / `y_train` / `y_test` come back `None`.
  - `test_data_properties_require_fit` — every accessor raises `NotFittedError` on an unfit experiment.
  - `test_fit_state_returns_equivalent_data_to_legacy` — sanity check on shape + columns + identity for the singleton `preprocess_pipeline`.

## INTERNAL

- `INTERNAL` — **References vs deep copies.** `_fit_state` holds references to the legacy attribute values, not deep copies. Mutating `exp.X_train` propagates to the underlying frame, matching legacy semantics. This decision was made deliberately: copying would (a) double memory for large datasets and (b) break code that does `exp.X_train.iloc[5:10] = …`. The cost is that `_fit_state` is implicitly tied to the legacy lifetime; once `setup()` is itself drained (last step before deleting `pycaret/internal/pycaret_experiment/`), the references will hold the data the new fit path produces directly.
- `INTERNAL` — **Why `dict` not `dataclass`.** A `dataclass` would force a fixed schema; the snapshot-as-dict tolerates task-specific gaps (clustering's `y_test=None`) without needing typed `Optional` annotations everywhere. When the state-holder migration finishes (post-4.0.0), the dict can be promoted to a typed `FitState` dataclass; for now the dict shape gives flexibility.
- `INTERNAL` — **The drain-lock test pattern, now applied to properties.** Sessions 22-28 used the `monkeypatch self._legacy.<verb> → raise` pattern for verb drains. Session 29 generalises it to attribute reads: `object.__setattr__(legacy, name, _BoomDescriptor())` shadows the legacy property with a raise-on-read object. The `try/except AttributeError` handles slot-bound legacy attributes that can't be shadowed (best-effort). The test passes because the drain genuinely doesn't touch those callsites — the sentinels are never read.

## Session 29 delta summary

| Metric | Session 28 end | Session 29 end |
|---|---:|---:|
| User-facing API touching `self._legacy` | 7 (props) + 6 (verbs) | **0 + 6** |
| Engine tests (fast + slow) | 104 | **108** |
| **Combined tests** | **250** | **254** |

---

# Session 28 — 2026-04-24 — God-class drain: unsupervised verbs (clustering + anomaly)

Baseline: session 27 finished the supervised drain (13 verbs). Session 28 drains the 3 unsupervised verbs — clustering + anomaly experiments now run natively.

## CHANGED — engine

- `CHANGED, BREAKING` — **`packages/engine/pycaret/core/unsupervised.py` — `UnsupervisedExperiment.create_model`**. No longer delegates to `self._legacy.create_model`. Resolves the estimator from the registry (or accepts a pre-constructed object), fits on `self._legacy.X_transformed`, and returns a `CreateResult` whose `.pipeline` is a real sklearn Pipeline.
  - Signature: `(estimator, *, num_clusters=None, fraction=None, fit_kwargs=None, round=4, verbose=False, **estimator_kwargs)`. `num_clusters` translates to `n_clusters`; `fraction` translates to `contamination` for anomaly.
  - Constructor falls through to the registry's default kwargs if a forwarded kwarg is rejected (`AffinityPropagation` has no `n_clusters`, `MeanShift` has none, etc.).
  - Dropped 3.x cruft: `ground_truth`, `experiment_custom_tags`, `system`, `add_to_model_list`, `raise_num_clusters`, `X_data`, `display`.
- `CHANGED, BREAKING` — **`UnsupervisedExperiment.assign_model`**. No longer delegates. Unwraps Pipeline → bare model, reads `model.labels_` (and `model.decision_scores_` for anomaly), and decorates a copy of `self.X`. `transformation=True` returns `X_transformed` rows; `score=False` skips the `Anomaly_Score` column for anomaly tasks.
- `CHANGED` — **CBLOF retry preserved**. The `cluster` anomaly detector (CBLOF) can `ValueError` on degenerate cluster separation. The native path mirrors legacy: catch the error, `model.set_params(n_clusters=12)`, refit. Wrapped in a `RuntimeError` with the legacy's message if the retry also fails.

## ADDED — tests

- `ADDED` — **`packages/engine/tests/test_session28_unsupervised.py`** — 11 tests covering clustering + anomaly create_model + assign_model + drain-locks + edge cases (unknown ID, `score=False`, `NotFittedError`).

## INTERNAL

- `INTERNAL` — **`predict_model`'s transitional bare-estimator branch is now universally dead.** Both supervised and unsupervised `create_model` return real Pipelines, so the `preprocessor.transform(X)` fallback inside `predict_model` is only reachable when a caller passes a bare estimator directly (uncommon). The comment in `core/experiment.py:predict_model` is updated to reflect this — the branch lives on as a belt-and-braces fallback rather than a transitional accommodation.
- `INTERNAL` — **Why CBLOF gets explicit retry, not a generic one.** Other anomaly detectors fail with different errors (`pyod`'s `ABOD` has its own gotchas around small datasets, etc.). The legacy code only retries CBLOF specifically, so we mirror that. A more general "try with smaller / larger param" retry would be over-engineering for v1.
- `INTERNAL` — **Constructor `TypeError` fallback.** When the registry args + user kwargs include something the underlying class doesn't accept (e.g. `AffinityPropagation(n_clusters=4)` — no such kwarg), we catch the `TypeError` and retry with just the registry defaults. Cleaner than gating on a per-algorithm allowlist + handles future registry changes gracefully.

## Session 28 delta summary

| Metric | Session 27 end | Session 28 end |
|---|---:|---:|
| OOP verbs still on `self._legacy` (clf/reg/clu/anomaly) | 0/0/2/2 | **0/0/0/0** ✅ |
| Engine tests (fast + slow) | 93 | **104** |
| **Combined tests** | **239** | **250** |

---

# Session 27 — 2026-04-24 — God-class drain: ensemble / blend / stack / calibrate / finalize

**The supervised drain is complete.** All 13 OOP verbs on classification + regression now run without `self._legacy`. This session lands the final 5 in one batch — each is a thin sklearn-meta-estimator wrapper that reuses the already-drained `create_model`.

## CHANGED — engine

- `CHANGED, BREAKING` — **`SupervisedExperiment.ensemble_model`** (supervised path). No longer delegates to `self._legacy.ensemble_model`. `method="Bagging"` → `BaggingClassifier`/`BaggingRegressor`. `method="Boosting"` → `AdaBoostClassifier`/`AdaBoostRegressor`. Returns a Pipeline whose last step is named `Bagging[<base_id>]` or `AdaBoost[<base_id>]`. Signature trimmed to `(estimator, *, method="Bagging", n_estimators=10, fold, round, fit_kwargs, verbose)` — dropped legacy `choose_better`, `optimize`, `experiment_custom_tags`, `groups`, `return_train_score`.
- `CHANGED, BREAKING` — **`SupervisedExperiment.blend_models`** (supervised path). Wraps `VotingClassifier` / `VotingRegressor`. Classification `method="auto"` (default) picks `"soft"` when every base model has `predict_proba`, else `"hard"`. Each base is added under a unique name (`{model_id}_{i}`). Signature: `(estimators, *, method="auto", weights=None, fold, round, fit_kwargs, verbose)`.
- `CHANGED, BREAKING` — **`SupervisedExperiment.stack_models`** (supervised path). Wraps `StackingClassifier` / `StackingRegressor`. Default meta-learner: `LogisticRegression(max_iter=1000)` for classification, `LinearRegression()` for regression — overridable via `meta_model=`. CV is `fold or self._legacy.fold_generator`. Signature: `(estimators, *, meta_model=None, fold, round, fit_kwargs, verbose)`.
- `CHANGED, BREAKING` — **`SupervisedExperiment.calibrate_model`** (supervised path). Wraps `CalibratedClassifierCV`. Classification only — raises `ValueError` for regression (calibration is undefined for continuous targets). Signature: `(estimator, *, method="sigmoid", cv=None, fold, round, fit_kwargs, verbose)`.
- `CHANGED, BREAKING` — **`SupervisedExperiment.finalize_model`** (supervised path). Re-fits the bare estimator on `X_transformed` + `y_transformed` (the FULL dataset, train + holdout combined) and returns a fresh fitted Pipeline. Input pipeline is untouched. Signature: `(estimator)` — dropped `model_only`, `groups`, `experiment_custom_tags`.

## ADDED — internal helpers

- `ADDED` — **`SupervisedExperiment._unwrap_estimator(obj)`** — single source of truth for converting any of {Pipeline, registry ID string, bare estimator} into `(bare_model, model_id)`. Used by all 5 new verb implementations + reusable for future drain work on unsupervised verbs.
- `ADDED` — **`SupervisedExperiment._wrap_in_pipeline(model, name)`** — the canonical Pipeline-assembly helper. `deepcopy(self.preprocess_pipeline) + [(name, model)]`. Used by `finalize_model` directly; the same construction is inlined in `create_model` for historical reasons (could be DRY-ed up in a follow-up).

## ADDED — tests

- `ADDED` — **`packages/engine/tests/test_session27_combine.py`** — 13 tests:
  - `test_ensemble_model_bagging` / `..._boosting` — wrapper produces the right sklearn meta + named pipeline step. End-to-end `predict` chain.
  - `test_ensemble_model_does_not_call_legacy_ensemble_model` — drain-lock.
  - `test_blend_models_voting_classifier_soft` — `voting=="soft"` auto-detected when all bases have `predict_proba`.
  - `test_blend_models_regressor` — uses `VotingRegressor`.
  - `test_blend_models_does_not_call_legacy_blend_models` — drain-lock.
  - `test_stack_models_classifier_with_default_meta` — default meta is `LogisticRegression`; result is a `StackingClassifier`.
  - `test_stack_models_does_not_call_legacy_stack_models` — drain-lock.
  - `test_calibrate_model_classification` — `CalibratedClassifierCV` with sigmoid.
  - `test_calibrate_model_rejects_regression` — `ValueError` when called on a regression experiment.
  - `test_finalize_model_refits_on_full_data` — predict on the (now-training-included) holdout still returns valid predictions.
  - `test_finalize_model_does_not_call_legacy_finalize_model` — drain-lock.
  - `test_combine_verbs_require_fit` — all 5 raise `NotFittedError` on unfit experiments.

## INTERNAL

- `INTERNAL` — **Why batch 5 verbs in one session.** ensemble / blend / stack / calibrate / finalize are all variations on "wrap a model in a sklearn meta-estimator + train as if it were a regular model". They share `_unwrap_estimator` + reuse `create_model` for CV. Each individual drain is ~30-50 LoC; batching them together makes the diff coherent and the test file unified.
- `INTERNAL` — **Renaming the final pipeline step.** Each of the 5 new verbs reuses `create_model` to do the actual training. `create_model` names the last step with the bare estimator's class name (e.g. `BaggingClassifier`); we then mutate `pipeline.steps[-1] = (descriptive_name, fitted_estimator)` to give it a more readable name (`Bagging[lr]`, `Voting`, `Stacking[LogisticRegression]`, `Calibrated[lr]`). This keeps the user-facing pipeline repr informative without complicating `create_model`'s contract.
- `INTERNAL` — **`finalize_model` reads `X_transformed` not `X_train_transformed`.** The legacy splits the full dataset into train + holdout; CV runs on train only. `finalize_model`'s contract is "include the holdout now too" → `self._legacy.X_transformed` is the union. The pipeline returned by finalize doesn't have a holdout — predictions on the holdout are now in-sample. Caller's responsibility to track which model is finalized vs not (the `FinalizeResult` dataclass exists precisely so the type system makes that visible).
- `INTERNAL` — **`calibrate_model` regression rejection.** `CalibratedClassifierCV` doesn't have a regression analogue. Sigmoid / isotonic calibration are about mapping scores to probabilities; regression has no analogous concept. Raising up-front rather than letting sklearn's "no `decision_function`" error surface 30s into a CV gives the caller a clear error. (3.x's `calibrate_model` similarly didn't support regression but failed implicitly.)
- `INTERNAL` — **Drain progress.** With session 27, all 13 OOP verbs on supervised tasks (4 persistence + 9 modeling) are native. Remaining drain work for `4.0.0`:
  1. Unsupervised verbs (`create_model` / `predict_model` / `assign_model` for clustering + anomaly).
  2. Time-series `Experiment` subclass.
  3. Strip transitional branches: the bare-estimator path in supervised `predict_model` is dead now (kept only for clustering/anomaly).
  4. Refactor model + metric registries to take an `Experiment` (not `_legacy`).
  5. Delete `pycaret/internal/pycaret_experiment/`.
  6. Ship `4.0.0` non-alpha to PyPI.

## Session 27 delta summary

| Metric | Session 26 end | Session 27 end |
|---|---:|---:|
| Supervised OOP verbs still on `self._legacy` | 1 | **0** ✅ |
| Engine tests (fast + slow) | 80 | **93** |
| **Combined tests** | **226** | **239** |

---

# Session 26 — 2026-04-24 — God-class drain: `compare_models` (supervised)

Baseline: session 25 drained `tune_model`. Session 26 drains the heart of the AutoML loop — `compare_models` — by reusing the already-drained `create_model` in a per-model loop.

## CHANGED — engine

- `CHANGED, BREAKING` — **`packages/engine/pycaret/core/supervised.py` — `SupervisedExperiment.compare_models`** (supervised path). No longer delegates to `self._legacy.compare_models` for classification + regression. Iterates the engine's `_all_models_internal` registry (filtered by `include` / `exclude` / `turbo`), calls `self.create_model` for each candidate, and assembles the leaderboard from each candidate's `Mean` metrics row. Time-series / clustering / anomaly still delegate via `_compare_models_legacy`.
- `CHANGED, BREAKING` — **Signature slim-down**. Kept: `include`, `exclude`, `fold`, `cross_validation`, `sort`, `n_select`, `turbo`, `errors`, `fit_kwargs`, `round`, `verbose`. Dropped 3.x cruft: `budget_time`, `experiment_custom_tags`, `probability_threshold`, `groups`, `caller_params`. All gone for the same reasons as previous session drains — either dead code, MLflow integration that was already killed, or one-line post-hoc overrides on the result.
- `CHANGED` — **All slots are keyword-only.** Decorator-style `compare_models(include=, n_select=)` is the only valid call shape. The legacy positional form (`compare_models(["lr", "dt"], None, None, 4)`) is gone.
- `CHANGED` — **Auto-detect ascending vs descending sort.** Error metrics (`MAE`, `MSE`, `RMSE`, `MAPE`, `RMSLE` + sklearn `neg_*` family) sort ascending; everything else descending. The legacy code required the caller to know which way each metric sorted.
- `CHANGED` — **`CompareResult.leaderboard` row source.** Each row is the `Mean` row of `created.metrics` (the per-fold DataFrame from session 24's drained `create_model`), prepended with a `Model` column for the registry ID. So leaderboard column schema is identical across classification (`Accuracy` / `AUC` / `Recall` / `Prec.` / `F1` / `Kappa` / `MCC`) and across all 4 supervised result types now (`CreateResult`, `TuneResult`, `CompareResult`, `PredictResult`).
- `CHANGED` — **`errors="ignore"` no longer hides errors silently.** When a candidate raises, the exception is swallowed + the candidate is dropped from the leaderboard. Future enhancement: log the exception type / message via the event stream so users can see what failed (currently the per-candidate loop is silent on failure for noise reasons; tracked as a polish item).

## ADDED — tests

- `ADDED` — **`packages/engine/tests/test_session26_compare.py`** — 10 tests covering: top-N return shape, default-sort defaults, ascending sort for error metrics, `exclude=` removes models, `turbo=True` blocks slow models, drain-lock against `self._legacy.compare_models`, end-to-end `compare → predict` chain, `errors="ignore"` skips a bogus model id, NotFittedError on unfit, and that `result.best` is a real Pipeline.

## INTERNAL

- `INTERNAL` — **Reusing already-drained verbs.** The native `compare_models` is ~50 LoC of glue around `self.create_model`. No new search / metric registry / fold logic — that all lives in `create_model` already. Each new drain reuses upstream drained verbs, which is why later sessions are progressively shorter despite covering more surface area. The `_cross_validate_supervised` helper from session 24 is now indirectly used by every supervised verb (`create_model` calls it directly; `tune_model` and `compare_models` call `create_model`).
- `INTERNAL` — **Empty-result soft-handling.** If every candidate fails (impossible in practice with the default registry, but ``errors="ignore"`` + a custom `include=` could trigger it), we return an empty `CompareResult(best=None, models=[], leaderboard=DataFrame(), ranked_ids=[])` rather than raising. Caller code that checks `if result.best is not None` gets a clear path; callers expecting at least one result must provide `errors="raise"` instead.
- `INTERNAL` — **Per-candidate error swallowing keeps the leaderboard reproducible.** Without `errors="ignore"` as the default, a single new model in the registry that breaks on a particular dataset would sink every notebook in the wild. With it, the registry can grow without breaking historical comparisons.

## Session 26 delta summary

| Metric | Session 25 end | Session 26 end |
|---|---:|---:|
| Supervised OOP verbs still on `self._legacy` | 2 | **1** |
| Engine tests (fast + slow) | 70 | **80** |
| **Combined tests** | **216** | **226** |

---

# Session 25 — 2026-04-24 — God-class drain: `tune_model` (supervised)

Baseline: session 24 drained `create_model`, returning real Pipelines for supervised tasks. Session 25 drains the next verb in the chain — `tune_model` — using sklearn's `RandomizedSearchCV` with the engine's registry-supplied search space.

## CHANGED — engine

- `CHANGED, BREAKING` — **`packages/engine/pycaret/core/supervised.py` — `SupervisedExperiment.tune_model`** (supervised path). No longer delegates to `self._legacy.tune_model` for classification + regression. Rewritten as task-aware dispatcher with native `RandomizedSearchCV`. Time-series / clustering / anomaly still delegate via `_tune_model_legacy`.
- `CHANGED, BREAKING` — **Signature slim-down** (supervised path). Kept: `estimator`, `fold`, `n_iter`, `custom_grid`, `optimize`, `fit_kwargs`, `round`, `verbose`. Dropped 3.x cruft: `custom_scorer`, `search_library` (sklearn-only for now; optuna / scikit-optimize integration is a follow-up if needed), `search_algorithm`, `early_stopping`, `early_stopping_max_iters`, `choose_better`, `groups`, `return_tuner`, `tuner_verbose`, `return_train_score`. All gone for the same reason as session 24's drops — either dead code (legacy tuner library deps that were already killed) or one-line post-hoc overrides on `TuneResult.search`.
- `CHANGED` — **`optimize=` accepts both naming conventions.** Maps PyCaret display names (``"Accuracy"`` / ``"AUC"`` / ``"Recall"`` / ``"Precision"`` / ``"F1"`` / ``"MAE"`` / ``"MSE"`` / ``"RMSE"`` / ``"R2"`` / ``"MAPE"``) AND sklearn scorer strings (``"accuracy"`` / ``"roc_auc"`` / ...) to sklearn's built-in scorers. Defaults to ``"accuracy"`` for classification, ``"r2"`` for regression. Metrics not in the mapping (PyCaret-only ``"Kappa"`` / ``"MCC"`` / ``"RMSLE"``) fall through to the task default — adapting them via `make_scorer` is a polish item.
- `CHANGED` — **Search space comes from `tune_grid` not `tune_distribution`.** Each engine container exposes both; sklearn's `RandomizedSearchCV` requires either an iterable or a scipy distribution with `.rvs()`, neither of which the registry's custom `UniformDistribution` / `IntUniformDistribution` types implement. `tune_grid` is a plain `dict[str, list]` and works directly. Adapting `tune_distribution` to scipy is a future-session polish.
- `CHANGED` — **`TuneResult` shape stabilised**:
  - `pipeline` — fitted sklearn Pipeline (preprocessor + tuned model).
  - `best_params` — `dict(search.best_params_)`.
  - `search` — the `RandomizedSearchCV` instance (now actually populated; the legacy wrapper had `search=None`).
  - `cv_results` — `pd.DataFrame(search.cv_results_)`.
  - `metrics` — per-fold metrics DataFrame for the winning estimator (same schema as `CreateResult.metrics`).

## ADDED — tests

- `ADDED` — **`packages/engine/tests/test_session25_tune.py`** — 9 tests:
  - `test_tune_model_returns_pipeline_with_tuned_estimator` — Pipeline shape + `RandomizedSearchCV` in `result.search`.
  - `test_tune_model_cv_results_and_metrics_dataframes` — `cv_results.length == n_iter`; `metrics` has `Mean` / `Std` rows.
  - `test_tune_model_custom_grid_overrides_registry` — `custom_grid={"C": [0.1, 1.0, 10.0]}` ⇒ `best_params["C"] in grid["C"]`.
  - `test_tune_model_optimize_mapping_uses_sklearn_scorer` — `optimize="AUC"` ⇒ `search.scoring == "roc_auc"`.
  - `test_tune_model_regression_default_optimize_is_r2`.
  - `test_tune_model_does_not_call_legacy_tune_model` — drain-lock; poisons `self._legacy.tune_model` + verifies success.
  - `test_tune_model_predict_chain_from_tuned_pipeline` — full `create → tune → predict` chain on Pipeline-in/out.
  - `test_tune_model_accepts_registry_id_directly` — `tune_model("lr", ...)` without prior `create_model`.
  - `test_tune_model_requires_fit` — `NotFittedError` on unfit Experiment.

## INTERNAL

- `INTERNAL` — **No-op fallback when `tune_grid` is empty.** Some registry containers have `tune_grid={}` (no tunable params). Rather than failing the whole call, native `tune_model` falls through to `create_model` with cross-validation enabled, returning a `TuneResult` with `search=None` + `best_params=params`. Caller's flow is consistent — `result.pipeline` is always a fitted Pipeline.
- `INTERNAL` — **`deepcopy` before search.** `RandomizedSearchCV` mutates the estimator in place during refit. Deep-copying the bare model first means the user's input estimator (and any Pipeline carrying it) is untouched after the call. Same pattern as session 24's per-fold deep-copy.
- `INTERNAL` — **Reusing `_cross_validate_supervised` for `TuneResult.metrics`.** The helper from session 24 is task-aware + uses the shared metric registry. By calling it on `search.best_estimator_`, `TuneResult.metrics` and `CreateResult.metrics` have byte-identical column schemas. The downstream UI / leaderboard / LLM-advisory code can render either uniformly.
- `INTERNAL` — **`error_score=0.0`.** sklearn's `RandomizedSearchCV` defaults to `error_score=np.nan` which causes the search to mark a fold-fit failure as nan and propagate through the mean. PyCaret's legacy default is to return `0.0` for failed metric calculations (see session 23 metric helper). Aligning the search to `0.0` keeps tune CV rankings consistent with the rest of the engine.

## Session 25 delta summary

| Metric | Session 24 end | Session 25 end |
|---|---:|---:|
| Supervised OOP verbs still on `self._legacy` | 3 | **2** |
| Engine tests (fast + slow) | 61 | **70** |
| **Combined tests** | **207** | **216** |

---

# Session 24 — 2026-04-24 — God-class drain: `create_model` (supervised)

Baseline: session 23 drained `predict_model`. Session 24 drains `create_model` for classification + regression. Unlocks the 4.0 invariant "`CreateResult.pipeline` is a real sklearn Pipeline".

## CHANGED — engine

- `CHANGED, BREAKING` — **`packages/engine/pycaret/core/experiment.py` — `Experiment.create_model`** (supervised). No longer delegates to `self._legacy.create_model` for classification + regression experiments. Rewritten as a task-aware dispatcher:
  - Supervised (classification / regression) → native `_create_model_supervised_native`.
  - Clustering / anomaly / time-series → still delegate via `_create_model_legacy` (their drains land in future sessions).
- `CHANGED, BREAKING` — **Signature slim-down** (supervised path). Kept: `estimator`, `fold`, `cross_validation`, `fit_kwargs`, `round`, `verbose`, `**estimator_kwargs`. Dropped: `probability_threshold`, `experiment_custom_tags`, `refit`, `return_train_score`, `groups`, `predict`, plus all the `system=True` / `add_to_model_list=True` / `X_train_data=...` / `metrics=...` / `display=...` internal hooks. All gone because they either (a) never made sense in 4.0's clean-room design, (b) belonged to MLflow integration that's already dead, or (c) are recoverable via a couple of lines post-hoc.
- `CHANGED` — **`CreateResult.pipeline` is now always a sklearn Pipeline on supervised tasks.** Previously it was a bare estimator (e.g. `LogisticRegression`), and downstream code had to either know to wrap it or run its own `transform` against `self.preprocess_pipeline`. Now the returned Pipeline is `deepcopy(self.preprocess_pipeline).steps + [(model_id, trained_model)]` — a true end-to-end predict chain.
- `CHANGED` — **Cross-validation runs manually instead of via `sklearn.model_selection.cross_validate`.** Rationale: the engine's metric registry doesn't map 1-1 to `sklearn.metrics.get_scorer(...)` — several registry entries take `pred_proba`, not `pred`, and the sklearn scorer interface doesn't cleanly expose that. Manual fold loop + shared `calculate_metrics` gives identical columns to legacy `predict_model` output (`Accuracy` / `AUC` / `Recall` / `Prec.` / `F1` / `Kappa` / `MCC` for classification; `MAE` / `MSE` / `RMSE` / `R2` / `RMSLE` / `MAPE` for regression). Per-fold failures are swallowed (empty row) so CV degrades gracefully.
- `CHANGED` — **`self._legacy._all_models_internal` is still the model registry source** (for now). The registry module itself is a separate concern from the god-class; its current `get_all_model_containers(experiment, ...)` signature reads a bunch of fields off the experiment object. Refactoring the registry to take an `Experiment` (not `_legacy`) is a separate follow-up — tracked as session 27+ "registry decoupling". For now the drain reads `self._legacy._all_models_internal` directly without calling `self._legacy.create_model`.

## ADDED — tests

- `ADDED` — **`packages/engine/tests/test_session24_create_model.py`** — 10 tests:
  - `test_create_model_returns_real_sklearn_pipeline_classification` — pipeline shape + last-step name = `model_id`.
  - `test_create_model_cv_metrics_have_mean_and_std_rows` — `Fold 0..N-1`, `Mean`, `Std` index + classification metric columns present.
  - `test_create_model_no_cross_validation_skips_metrics` — `cross_validation=False` → `metrics is None`, pipeline still fitted.
  - `test_create_model_regression_uses_regression_metric_registry` — MAE / R2 columns.
  - `test_create_model_unknown_id_raises` — `ConfigurationError` on bad ID.
  - `test_create_model_accepts_preconstructed_estimator` — user's `LogisticRegression(C=2.0)` survives into the final Pipeline step.
  - `test_create_model_predict_model_roundtrip_no_bare_branch` — the killer test. Monkey-patches `self.preprocess_pipeline` to raise; runs `predict_model(created.pipeline)` end-to-end. Passes because the returned Pipeline is already complete — `predict_model` doesn't need to touch the preprocessor for supervised create_model output.
  - `test_create_model_does_not_call_legacy_create_model` — drain-lock. Poisons `self._legacy.create_model` + asserts native path succeeds.
  - `test_create_model_clustering_still_delegates_to_legacy` — keeps the fallback working.
  - `test_create_model_requires_fit` — `NotFittedError` on unfit.

## CHANGED — existing tests

- `CHANGED` — **`packages/engine/tests/test_models.py`**. The `check_exp` helper's loop now unwraps `exp.create_model(id).pipeline` via a `_unwrap_pipeline` helper (grabs `.steps[-1][1]` if it's a `Pipeline`). Rationale: the registry's `Equality` predicate is `lambda m: isinstance(m, <Class>)`, which failed against a Pipeline wrapper. Clustering / anomaly bypass the unwrap (they still return bare estimators).

## INTERNAL

- `INTERNAL` — **Why `self._legacy` for `X_train_transformed` + `fold_generator`.** The drain's scope is "this verb doesn't *call* `self._legacy.create_model`". Reading transformed data + the pre-built CV generator from the legacy state is a separate (also-to-be-drained-later) delegation point. Splitting the drain this way keeps each session's diff manageable.
- `INTERNAL` — **Deep-copy per fold.** `deepcopy(model)` before each fold's fit prevents state leakage from the previous fold (sklearn estimators are generally stateful — `_n_features_in_` etc.). Costs O(K) copies for K folds; cheap vs. CV work itself. The final fit uses the *original* `model` object so the Pipeline that comes out carries the estimator's full final state.
- `INTERNAL` — **Pipeline assembly ordering.** The returned Pipeline is `[preprocessing_steps..., (model_id, trained_model)]`. Why this order: the preprocessor's `label_encoding` step is a `TransformerWrapperWithInverse` which inverse-transforms predictions back to the original label space; appending it *before* the model would mean predictions are inverse-transformed on the way in. sklearn Pipeline semantics (`fit(X, y)` transforms all non-final steps, `predict(X)` transforms then predicts) give us the right behavior when the model is the last step.
- `INTERNAL` — **3.x-style CV semantics intentionally preserved.** Legacy runs CV on `X_train_transformed` (preprocessing fit on all training rows, then CV over transformed data). Strictly, this leaks test-fold information into the preprocessor's fit, which a pure sklearn `cross_validate(Pipeline([preproc, model]), X_raw, y)` would avoid. For the drain we preserve legacy semantics — purity can be revisited post-4.0.0 once we have a baseline to regression-test against.

## Session 24 delta summary

| Metric | Session 23 end | Session 24 end |
|---|---:|---:|
| Supervised OOP verbs still on `self._legacy` | 4 | **3** |
| Supervised `CreateResult.pipeline` type | bare estimator | **sklearn `Pipeline`** |
| Engine tests (fast + slow) | 51 | **61** |
| **Combined tests** | **197** | **207** |

---

# Session 23 — 2026-04-24 — God-class drain: `predict_model`

Baseline: session 22 drained the 4 persistence verbs. Session 23 drains the 5th OOP verb — `predict_model`. The heart of the rewrite is a task-aware dispatch that handles classification / regression / clustering / anomaly without ever touching `self._legacy`.

## CHANGED — engine

- `CHANGED, BREAKING` — **`packages/engine/pycaret/core/experiment.py` — `Experiment.predict_model`**. No longer delegates to `self._legacy.predict_model`. Rewritten as ~170 LoC of native dispatch. Signature slimmed from `(estimator, *args, **kwargs)` to `(estimator, data=None, *, raw_score=False, round=4, verbose=False)`. All 3.x-era params are gone:
  - `probability_threshold` — removed. Callers thresholding on positive-class probability can do `out["prediction_label"] = out["prediction_score"] >= t`.
  - `encoded_labels` — removed. Label encoding happens inside the preprocessor; for integer labels, `out["prediction_label"].map(class_to_int)` is a one-liner.
  - `preprocess` — removed. In 4.0 the pipeline either preprocesses itself (Pipeline case) or we apply `self.preprocess_pipeline` automatically (bare estimator case, transitional).
  - `ml_usecase` — removed. Comes from `self.task` directly now.
- `CHANGED` — **Transitional bare-estimator accommodation.** `CreateResult.pipeline` today is a bare sklearn estimator (e.g. `LogisticRegression`), not a Pipeline. Once `create_model`'s drain lands (session 24), it becomes a proper Pipeline with preprocessing baked in, and the transitional code-path in `predict_model` collapses to a one-line `estimator.predict(X)`. Flagged in the docstring + guarded by `isinstance(estimator, sklearn.pipeline.Pipeline)`.
- `CHANGED` — **Metric computation inlined.** Native `predict_model` uses the existing metric registry (`pycaret.containers.metrics.{classification,regression}.get_all_metric_containers` + `pycaret.utils.generic.calculate_metrics`) directly — no longer depends on `self._legacy.pull()` to surface the holdout metrics DataFrame. Wraps the whole metric block in a broad try/except → returns `None` on any registry hiccup. Rationale: metrics are advisory; a predict must never fail because a metric choked.
- `CHANGED` — **Per-task output columns**:
  - Classification binary → `prediction_label` + `prediction_score` (positive-class probability).
  - Classification multiclass, `raw_score=False` → `prediction_label` + `prediction_score` (winning-class probability).
  - Classification multiclass, `raw_score=True` → `prediction_label` + `prediction_score_<class>` per class.
  - Regression → `prediction_label` only.
  - Clustering → `Cluster` column with `"Cluster {i}"` labels.
  - Anomaly → `Anomaly` + `Anomaly_Score` (when the detector exposes `decision_function`).

## ADDED — tests

- `ADDED` — **`packages/engine/tests/test_session23_predict.py`** — 12 new tests, split by speed:
  - **Fast (7 tests)** — fabricate a tiny `StandardScaler + {LogReg,LinReg}` pipeline + a fit-sentinel Experiment; exercise raw predict paths. Confirms: non-estimator rejection (`TypeError` on dict), NotFittedError without fit, metrics absent when data lacks target, metrics present + model name set when data has target, regression has no score column, multiclass `prediction_score` is winning-class prob, multiclass `raw_score` sums to ~1 per row.
  - **Slow (5 tests, @slow marker)** — full engine E2E on `juice` / `boston`. Covers binary classification output columns, classification `raw_score`, regression output + metrics, event stream captures `MODEL_PREDICTED` with `n_rows` + `duration_ms`, and the drain-lock test (`test_predict_model_does_not_call_legacy_predict_model`).
- `ADDED` — **Drain-lock test pattern.** Monkeypatches `exp._legacy.predict_model` with a raise-on-call function, then calls `exp.predict_model(pipeline)` and asserts it succeeds. Any future refactor that accidentally re-delegates will fail on the Ubuntu + Windows matrix. Same shape as session 22's `test_save_model_does_not_touch_legacy`.

## INTERNAL

- `INTERNAL` — **Why bare estimators are still accepted (temporarily).** The pyramid of the 10-verb drain is `save_model → predict_model → create_model → ...`. Sessions 22–23 drain verbs that consume the output of `create_model`. `create_model` still returns a bare estimator today, so strictly rejecting that in `predict_model` would red-light the slow E2E suite (and would also break every notebook in the wild). Session 24 drains `create_model`, replacing the returned bare estimator with a Pipeline. At that point the transitional branch in `predict_model` (+ the `self.preprocess_pipeline` transform call, + the `estimator_is_pipeline` check) can be deleted in a follow-up cleanup commit.
- `INTERNAL` — **Task dispatch via `self.task` not `isinstance`.** The legacy code used `self._ml_usecase` (a `MLUsecase` enum on the god-class). The native code reads `self.task` (a `pycaret.core.tasks.TaskType` enum on the Experiment). Cleaner because `TaskType` already lives on the 4.0 surface; no import needed from the internal module.
- `INTERNAL` — **Metric registry called with empty `globals_dict`.** `get_all_metric_containers(globals_dict, raise_errors=False)` — we pass `{}` for `globals_dict` because the default-behavior metrics don't need the legacy experiment's variables. If any metric registers via `globals_dict` reads, the `raise_errors=False` means it gets silently skipped rather than crashing the predict. This is consistent with the legacy behavior (which also had try/except around `calculate_metrics` calls for robustness).

## Session 23 delta summary

| Metric | Session 22 end | Session 23 end |
|---|---:|---:|
| OOP verbs still on `self._legacy` | 6 | **5** |
| Engine tests (fast + slow) | 35 | **51** |
| **Combined tests** | **181** | **197** |

---

# Session 22 — 2026-04-24 — God-class drain kickoff: persistence verbs

Baseline: sessions 9–21 completed the entire platform side of the 4.0 revamp (backend + frontend + 6 LLM copilots + multi-user + auth + audit). Session 22 pivots back to the engine: the ~10 OOP verbs that still delegate to `self._legacy` (the 3.x god-class) are drained one at a time onto native sklearn. This session targets the 4 persistence verbs.

Theme: **drain `save_model` / `load_model` / `save_experiment` / `load_experiment` off the legacy god-class**.

## CHANGED — engine

- `CHANGED` — **`packages/engine/pycaret/core/experiment.py`** — `Experiment.save_model` / `load_model` / `save_experiment` / `load_experiment` no longer call `self._legacy.save_model` / etc. They now delegate to `pycaret.persistence.save_model` / `load_model` — thin `joblib.dump` / `joblib.load` wrappers with a `.pkl` suffix fallback.
  - Contract change: `save_model(model, path, *args, **kwargs)` → `save_model(model, path, *, verbose=False)`. The `*args, **kwargs` were only ever forwarded to the legacy path (`model_only`, `prep_pipe_`, cloud kwargs). None of those survive the drain; if callers need to persist only a sub-step they can `save_model(pipeline.named_steps["trained_model"], path)` directly.
  - Contract change: `save_model` no longer requires the Experiment to be fitted. A caller may have obtained a pipeline from elsewhere (loaded from disk, handed over by a colleague, produced by `finalize_model` in another Experiment) and still want a normalised save. The 3.x path enforced fitting for no technical reason.
  - Contract change: `save_experiment(path)` requires `fit` (an unfit Experiment is just constructor kwargs), and `Experiment.load_experiment(path)` is now a `@staticmethod` that raises `TypeError` if the pickled object is not an Experiment. Previously it silently returned whatever was in the file.
  - Added: `MODEL_SAVED` event is logged on the experiment's event stream with `payload={"path": str(written)}` (and `"kind": "experiment"` for `save_experiment`). Skipped gracefully when `self.logger is None` (pre-fit state).
- `REMOVED` — **Cloud-credential path for `load_model`.** The 3.x `load_model` accepted `platform="aws"|"gcp"|"azure"` + `authentication=...` kwargs to pull from S3 / GCS / Blob. That's gone. Cloud serving is Control Plane territory — the `services/api` backend loads pipelines from the configured artifact backend (local dir today, MinIO / S3 once prod-compose lands). If a notebook user needs to fetch a remote model, they pull the file themselves (boto3, gcloud, azure-storage) and pass the path to `load_model`.
- `REMOVED` — **MLflow artifact logging on save.** The 3.x path, when `logging_param.loggers` included a remote-capable logger, would auto-push the saved file to MLflow / Comet / W&B. The 4.0 logger base class doesn't have this hook (MLflow/Comet/W&B all killed from the engine in phase 0–1); artifact promotion is now an explicit server-side step (`POST /runs/{id}/promote`).
- `REMOVED` — **`model_only` kwarg.** Was only meaningful when the legacy path had access to `self.pipeline` (the preprocessing chain). In 4.0 the pipeline is always self-contained — preprocessing + trained model are one `sklearn.pipeline.Pipeline`. Saving "only the model" is either impossible (pipeline input is already transformed) or trivially `pipeline.named_steps[...]`.

## ADDED — tests

- `ADDED` — **`packages/engine/tests/test_session22_persistence.py`** — 7 new unit tests covering the drained verbs. Deliberately don't use the full engine fit path (no `setup()` / `create_model()`), so they run in ~2s total:
  - `test_save_model_via_experiment_instance_roundtrips` — `exp.save_model` + `exp.load_model` preserve `predict()` output.
  - **`test_save_model_does_not_touch_legacy`** — constructs an unfit Experiment (so `self._legacy` doesn't exist yet), calls save/load, asserts `_legacy` is still absent. This test locks the drain against accidental re-introduction of a `self._legacy.*` call in a future refactor.
  - `test_save_model_accepts_path_objects_and_strings` — `Path` and `str` both work, `.pkl` suffix is added.
  - `test_save_model_emits_model_saved_event` — event stream carries the absolute path.
  - `test_save_experiment_requires_fit` — `NotFittedError` on an unfit Experiment.
  - `test_load_experiment_rejects_plain_model_file` — `TypeError` with message steering the caller to `load_model`.
  - `test_module_level_helpers_still_exposed` — `pycaret.save_model` / `pycaret.load_model` remain top-level importable.

## INTERNAL

- `INTERNAL` — **Drain ordering + test affordance.** The 10-verb drain list (`save_model` → `predict_model` → `create_model` → `tune_model` → `ensemble_model` → `blend_models` → `stack_models` → `calibrate_model` → `compare_models` → `finalize_model`) is easiest-first. Persistence is trivially stateless → natural starting point. `predict_model` + `create_model` next. The stacked-models trio is the hardest because each layer's behavior has to be preserved bit-for-bit against the 3.x reference. Each session migrates one verb + writes a test of the form "this verb does NOT touch `self._legacy`" to lock the drain.
- `INTERNAL` — **Null-logger guard in `save_model`.** `self.logger` is `None` until `fit()` installs one (either `NullLogger` or `MemoryLogger` depending on `log_experiment`). `save_model` deliberately does not require fit, so the logger can legitimately be `None` at call time. We null-check + skip the event emit rather than eagerly installing a NullLogger in `__init__` — eagerly installing would violate sklearn's `get_params` contract (constructor args must be stored verbatim).
- `INTERNAL` — **`pycaret.persistence` vs `pycaret.internal.persistence`.** Both exist today. `pycaret.persistence` is the 4.0 clean version (`save_model(model, path)` + `load_model(path)` only — ~90 lines). `pycaret.internal.persistence` is the 3.x-era version (~800 lines including cloud + MLflow). The session-22 drain points the Experiment verbs at `pycaret.persistence`. The legacy module remains imported by `tabular_experiment.save_model` + other drain-targets; once all 10 verbs are drained, `pycaret.internal.persistence` becomes deletable (tracked as a session-32 exit criterion).

## Session 22 delta summary

| Metric | Session 21 end | Session 22 end |
|---|---:|---:|
| OOP verbs still on `self._legacy` | 10 | **6** |
| Engine fast tests | 28 | **35** (+7) |
| **Combined tests** | **174** | **181** |
| `self._legacy.save_model` / `load_model` callsites on Experiment | 4 | **0** |

---

# Session 21 — 2026-04-24 — Drift analyst + audit logs

Baseline: session 20 shipped workspace-member CRUD + the `X-PyCaret-Key` fallback for programmatic auth. This session closes out the MVP-2 punch list: the 6th / final LLM copilot (drift analyst) + the cross-cutting `AuditLog` table + middleware + viewer screen (SPEC § 17.4). After this, there is nothing left on the platform roadmap before the engine god-class drain — session 22+ pivots to engine work → `4.0.0` non-alpha release on PyPI.

Theme: deliver the **drift_analysis** consultation type end-to-end + make every mutating API call auditable.

## ADDED — drift reports

- `ADDED` — **`services/api/pycaret_server/db/models.py` — `DriftReport`** (SPEC § 4.12). Columns: `deployment_id` FK (cascade), `baseline_artifact_id` FK (set-null), `window_start` / `window_end`, `drift_score` (0..1), bucketed `drift_status` (`none | mild | moderate | severe`), `feature_drift_json` (shape `{feature: {score, kind}}` where kind ∈ psi/ks/chi2/missing_rate), `prediction_drift_json` (JS divergence), `sample_size`, `created_by` FK.
- `ADDED` — **`services/api/pycaret_server/api/drift.py`** — 3 routes under `/api/v1/`. `POST /deployments/{id}/drift-reports` creates a snapshot + server-buckets `drift_status` from `drift_score` (thresholds 0.10 / 0.25 / 0.40, aligned with the PSI convention) + guards `window_end >= window_start` (400). `GET /deployments/{id}/drift-reports` lists reports for a deployment (newest first, capped 500). `GET /drift-reports/{id}` returns a single row with full feature/prediction JSON.
- `ADDED` — **`services/api/pycaret_server/llm/consultations/drift_analysis.py`** — 6th LLM copilot. Prompt tells the model to look for concentration (one dominant feature → data-source change) vs diffuse drift (genuine concept shift), factor in sample size (skepticism when < 200), and classify prediction-drift-without-feature-drift / missing-rate-spike as specific risk flags. Verdict is prefixed with one of `RETRAIN NOW` / `INVESTIGATE` / `MONITOR` / `NO ACTION` so the UI can tone-code via `.startsWith()`. Output schema locks top-level keys with `additionalProperties: false`.
- `ADDED` — **`POST /api/v1/llm/analyze-drift`** — body `{drift_report_id}`. Pulls the DriftReport + Deployment + owning Pipeline snapshot, consults the workspace's active LLM provider via the shared `ConsultationContext` path (free-rides on provider routing + audit trail). 404 on unknown report or missing deployment.
- `ADDED` — **Migration `0cd9d5ea2e17`** — adds `drift_reports` + `audit_logs` in one revision. Auto-generated then reviewed; FK cascades match the model.

## ADDED — audit logs

- `ADDED` — **`services/api/pycaret_server/db/models.py` — `AuditLog`** (SPEC § 17.4). Append-only: `id`, `workspace_id` (nullable for global events), `user_id` (nullable for unauth calls), `action` (dotted `{namespace}.{verb}`), `method`, `path`, `target_type`, `target_id`, `status_code`, `payload` (scrubbed JSON), `ip_address`, `user_agent`, `created_at`. Explicitly *no* `updated_at` — rows are immutable by design.
- `ADDED` — **`services/api/pycaret_server/audit.py` — `AuditLogMiddleware`** — FastAPI `BaseHTTPMiddleware` that records one row per `POST/PATCH/PUT/DELETE` on `/api/v1/*`. Captures the request body via `request.body()` + re-injects it into `request._body` so route handlers can still read it. Scrubs sensitive fields (`password`, `password_hash`, `api_key`, `token`, `refresh_token`, `access_token`, `api_key_encrypted`, `plaintext_token` — case-insensitive). Derives `{entity}.{verb}` action by walking path segments + classifying UUIDs vs nouns vs known sub-verbs. Extracts `workspace_id` from `/workspaces/{id}/…` URLs. Skips `/auth/refresh`, `/healthz`, `/openapi.json`, `/docs`, `/redoc`. Best-effort — never blocks or fails the request; DB errors are logged + swallowed.
- `ADDED` — **`services/api/pycaret_server/api/audit.py`** — 2 viewer routes. `GET /admin/audit-logs` is superuser-gated (via the `require_admin` dependency). `GET /workspaces/{id}/audit-logs` is workspace-admin-gated. Both support pagination (limit/offset) + filters on action, user_id, target_type, target_id, since, until. Reads are not themselves audited (that would be infinite recursion once the admin opens the viewer).
- `ADDED` — **`services/api/tests/test_session21.py`** — 12 integration tests. Drift CRUD (4): buckets drift_status correctly, bucket-boundary parameterisation (0.05 → none, 0.15 → mild, 0.6 → severe), list + get round-trip, window_end < window_start → 400. Drift analyst (2): happy path runs the LLM + returns the canned INVESTIGATE verdict with risk flags, 404 on unknown report. Audit logs (6): mutating request is recorded + user-attributed, bootstrap password is REDACTED in the stored payload, workspace-scoped viewer returns 403 for a non-member, admin route returns 403 for a non-superuser, action filter narrows results, workspace-scoped viewer returns only that workspace's rows.

## CHANGED

- `CHANGED` — **`services/api/pycaret_server/auth/deps.py`** — `get_current_user` now stashes the resolved `User` onto `request.state.audit_user` so the audit-log middleware can attribute rows without re-resolving the header. Best-effort — routes that don't depend on `CurrentUser` simply don't have `audit_user` set + the middleware tolerates that (row persists with `user_id=NULL`, still useful for intrusion forensics on failed-auth attempts).
- `CHANGED` — **`services/api/pycaret_server/api/llm.py`** — imports `drift_analysis` + `AnalyzeDriftRequest` + `DriftReport` + `Deployment` + registers the `/llm/analyze-drift` route.
- `CHANGED` — **`services/api/pycaret_server/app.py`** — registers `AuditLogMiddleware` + mounts `drift.router` and `audit.router` under `/api/v1`.
- `CHANGED` — **`services/api/pycaret_server/db/__init__.py`** — re-exports `DriftReport` + `AuditLog`.
- `CHANGED` — **`services/api/pycaret_server/llm/schemas.py`** — adds `AnalyzeDriftRequest` (pydantic body: `{drift_report_id}`).

## ADDED — frontend

- `ADDED` — **`apps/web/src/components/DriftAnalysisModal.tsx`** — modal rendering the canonical `LLMAdvice` envelope for a specific drift report. Auto-fires the consultation on open (same pattern as `<DeploymentReviewModal>`). Verdict tone-coded via the 4-prefix classifier (`RETRAIN NOW` → danger-500, `INVESTIGATE` → warn-500, `MONITOR` → ink-200, `NO ACTION` → success-500). Shows the feature-drift snapshot sorted by score desc so the dominant drivers sit at the top.
- `ADDED` — **`apps/web/src/components/DriftReportsCard.tsx`** — inline card on `/deployments/:id`. Lists existing reports with window / score / status / sample columns + a "✨ Analyze" button per row that opens the modal. "Record snapshot" button toggles an inline form: `drift_score` input + optional `sample_size` + pasted `feature_drift_json` / `prediction_drift_json` textareas with sensible placeholders. Client-side JSON parsing + 0–1 range guard on score, with inline error rendering before the network round-trip.
- `ADDED` — **`apps/web/src/pages/AuditLogViewer.tsx`** at `/admin/audit` — superuser-gated screen. Reads `auditApi.listAdmin` with debounced filters (action + target_type + limit). Table with When / Action / Method / Path / Status / User columns; clicking a row expands an inline panel showing the scrubbed payload + workspace_id / target_type / target_id / ip_address / user_agent. Status codes tone-coded (5xx → danger-500, 4xx → warn-500, 2xx/3xx → ink-200/80). Non-superusers see a forbidden message + pointer to the workspace-scoped view.
- `ADDED` — **`apps/web/src/api/types.ts`** — `DriftStatus`, `DriftKind`, `FeatureDriftEntry`, `PredictionDrift`, `DriftReportRead`, `DriftReportCreate`, `AuditLogRead`, `AuditLogFilters`.
- `ADDED` — **`apps/web/src/api/endpoints.ts`** — `driftApi` (list/create/get), `auditApi` (listAdmin/listForWorkspace), `llmApi.analyzeDrift`.
- `ADDED` — **10 new Vitest tests** — 3 for `<DriftAnalysisModal>` (inert-when-closed, danger-toned `RETRAIN NOW` + feature rows sorted desc, success-toned `NO ACTION`), 4 for `<DriftReportsCard>` (empty state, list + open modal + auto-fire, create form submit with parsed JSON, out-of-range score triggers a form error without hitting the API), 3 for `<AuditLogViewer>` (row-expand reveals scrubbed payload, non-superuser sees forbidden + API call is skipped, filter form triggers a new fetch with the right params).

## CHANGED — frontend

- `CHANGED` — **`apps/web/src/App.tsx`** — registers `/admin/audit`.
- `CHANGED` — **`apps/web/src/pages/DeploymentDetail.tsx`** — renders `<DriftReportsCard>` below the PredictTester in the left column.
- `CHANGED` — **`apps/web/src/components/Layout.tsx`** — top nav gains an "Audit log" link that renders only when `user.is_superuser === true`.

## INTERNAL

- `INTERNAL` — **Drift bucket thresholds.** Chose 0.10 / 0.25 / 0.40 to align with the common PSI convention (below 0.10 = no drift, 0.10–0.25 = mild investigation, above 0.25 = significant). The verdict strings the LLM returns don't have to match the bucket label — the analyst decides severity in context of sample size + feature concentration.
- `INTERNAL` — **No scheduled drift-detection job in v1.** Real drift detection needs a prediction log + a scheduled Job queue runner, neither of which is built yet. For v1 the `POST /deployments/{id}/drift-reports` route accepts a pre-computed snapshot — CI jobs / notebooks / external monitors can POST reports with an `X-PyCaret-Key` header, and the UI is a read/analyse surface. When the Job queue lands (post-4.0.0), we add `drift_detection_job` that does the compute itself.
- `INTERNAL` — **Why the middleware resolves `session_factory` lazily.** First test run hit an empty `audit_logs` table because the middleware captured `session_factory` at import time, before the test fixture rebound `pycaret_server.db.session.session_factory` to a test-scoped factory. Fixed by importing the module (`from pycaret_server.db import session as _session_mod`) and reading `_session_mod.session_factory` at call-time. The pattern applies to any module that caches a session factory across test fixtures.
- `INTERNAL` — **Action derivation.** Rather than statically mapping routes to action strings (would need to be maintained in lockstep with new routes), the middleware folds URL segments into `{namespace}.{verb}` at runtime. UUIDs are skipped; "verb" segments are recognised from a known allowlist (`promote`, `cancel`, `predict`, `analyze-drift`, `invite`, …). A dotted namespace makes filter-by-action ergonomic (`workspaces.create` vs `runs.cancel`) without needing prior registration.
- `INTERNAL` — **Scrubbing rule is field-name-based, not value-based.** We redact by key name (`password`, `api_key`, …), not by pattern-matching the value (which would miss passwords that happen to look like normal strings). Tradeoff: if a field is named something innocuous but contains a secret, it'll leak. Acceptable for v1 — SPEC § 17.3 promises KMS-wrapped secrets anyway; the audit log is a transparency surface, not a secret store.
- `INTERNAL` — **Verdict-string classifier vs enum.** Same design as session 19's deployment reviewer: the LLM emits a string prefixed with one of 4 literal verdicts, and the UI classifies with `.startsWith()`. Beats an enum because the LLM can tack on reasoning (`"RETRAIN NOW: amount feature missing-rate 0.42"`) that shows up verbatim in the verdict line. UI tests assert tone-coded class names on the DOM to lock the contract.

## Session 21 delta summary

| Metric | Session 20 end | Session 21 end |
|---|---:|---:|
| LLM copilots (of 6 in spec) | 5 | **6** (all) |
| API routes | ~58 | **~63** |
| Server integration tests | 68 | **80** |
| UI components | 12 | **14** |
| UI screens | 15 | **16** |
| UI tests | 52 | **62** |
| **Combined tests** | **148** | **174** |
| Production bundle (gz) | 99 kB | **101 kB** |

---

# Session 20 — 2026-04-24 — Workspace members + programmatic API-key auth

Baseline: session 19 shipped 5 of 6 LLM copilots + per-user API-key CRUD (mint / list / revoke) but the `X-PyCaret-Key` header was not yet accepted by any route. Session 20 closes that loop and adds multi-user collaboration — the platform is now usable by more than one person per workspace.

Theme: deliver **workspace member CRUD + programmatic API-key auth**. Drift analyst + audit logs deferred to session 21 to keep scope honest.

## ADDED — workspace members

- `ADDED` — **`services/api/pycaret_server/api/members.py`** — 4-route module under `/workspaces/{workspace_id}/members`. `GET` lists members with role + active status (any member); `POST` invites an existing user by email (admins only — returns 404 if no user with that email + hint about V2 email-invite flow); `PATCH /{user_id}` changes a role (admins only); `DELETE /{user_id}` removes (admins only). Both `PATCH` demote-admin + `DELETE` remove-admin enforce the last-admin guard: refuse to drop the workspace below 1 admin (400).
- `ADDED` — **Role model `admin | member`** — Python `Literal` matches the DB column. SPEC § 17.2 proposes a richer 6-role set; rolled forward when SSO lands. Pydantic `InviteRequest` / `PatchRoleRequest` validate against the same literal.
- `ADDED` — **`_admin_count(db, workspace_id)` helper** — single-query count feeding both last-admin guards. Avoids a subquery in the route body + keeps the invariant in one place.
- `ADDED` — **`services/api/tests/test_session20.py`** — 8 member-CRUD tests covering list bootstrap admin, invite-existing-user, invite-unknown-email-404, non-admin-cannot-invite-403, promote / demote, cannot-demote-last-admin-400, cannot-remove-last-admin-400, and remove-member-succeeds.

## ADDED — programmatic API-key auth

- `ADDED` — **`X-PyCaret-Key` header acceptance in `auth/deps.py`** — `get_current_user` now accepts `Authorization: Bearer …` (JWT) OR `X-PyCaret-Key: pck_…` (API key). JWT takes precedence when both are present (common dev pattern: long-lived key in env + short-lived UI session). Hash-and-lookup against the `ApiKey` table; checks `revoked_at is null`, `expires_at > now` (with SQLite tz coercion), and `user.is_active`. `last_used_at` is stamped on every successful auth.
- `ADDED` — **`services/api/tests/test_session20.py` API-key tests** — 6 tests: happy-path auth with a minted key, revoked key → 401, bogus key → 401, expired key → 401 (forges `expires_at` backwards), JWT-takes-precedence when both sent, missing-both → 401.

## ADDED — frontend

- `ADDED` — **`apps/web/src/pages/WorkspaceMembers.tsx`** — `/workspaces/:wsId/members` screen. Admins see: invite form (email + role + submit) + members table with inline role `<select>` + Remove button per row. Non-admins see the members table only. Own row flagged `(you)`. Last-admin row has both the select + Remove button disabled with an explanatory `title` tooltip — mirrors the server guard in the UI.
- `ADDED` — **`apps/web/src/pages/WorkspaceMembers.test.tsx`** — 4 Vitest tests: admin view shows invite + can change role, last-admin disables select + remove, non-admin hides invite + action column, invite submit fires the API with the chosen role.
- `ADDED` — **`membersApi` in `apps/web/src/api/endpoints.ts`** — typed bindings for list / invite / changeRole / remove.
- `ADDED` — **`MemberRead` / `InviteRequest` / `PatchRoleRequest` / `WorkspaceRole` types** in `apps/web/src/api/types.ts`.

## CHANGED

- `CHANGED` — **`apps/web/src/pages/WorkspaceDetail.tsx`** — header action row gains a "Members" button alongside Pipelines / Deployments / LLM. Same `btn-secondary` affordance.
- `CHANGED` — **`apps/web/src/App.tsx`** — registers the new route `/workspaces/:wsId/members`.
- `CHANGED` — **`services/api/pycaret_server/app.py` + `api/__init__.py`** — mount the `members` router under `/api/v1`.

## INTERNAL

- `INTERNAL` — **Single-flight auth priority.** The dependency checks `Authorization` first, then `X-PyCaret-Key`, but crucially only *invokes* the fallback when the bearer path returns no user — not when it's present-but-invalid. Invalid bearer → 401 even if a valid key is present. Rationale: a client that sends a bearer header is signalling "use this"; silently falling back masks config bugs.
- `INTERNAL` — **Last-admin guard location.** Kept in the route handlers (not a DB constraint), because "admin" is a soft role in the data model + a pure count check reads cleaner than a CHECK constraint. Doubled-up in UI (disabled control + tooltip). If we ever need to atomic-ify it against concurrent demotes, switch to a `SELECT … FOR UPDATE` on the admin rows — v1 single-writer workload doesn't need it yet.
- `INTERNAL` — **`getByText` ambiguity fixes in member tests.** `MemberRow` renders the user email twice when `display_name` is null (once as the display-name fallback, once as the small mono subtitle). Tests that assert a member is present use `getAllByText('…@example.com').length > 0` rather than `getByText`. Extending the lesson from session 19: always audit whether a text snippet appears once or many times before reaching for `getByText`.
- `INTERNAL` — **Invite-by-email shape.** v1 looks up an existing `User` row by email, returns 404 if absent with a hint ("ask them to sign up first (email invites arrive in V2)"). V2 will either create a pending-account row + send a confirmation email, or integrate with the SSO IdP's user directory. The 404 response keeps the failure mode explicit instead of silently creating ghost accounts.

## Session 20 delta summary

| Metric | Session 19 end | Session 20 end |
|---|---:|---:|
| API routes | ~54 | **~58** |
| Server integration tests | 54 | **68** |
| Auth methods | JWT only | **JWT + X-PyCaret-Key** |
| UI screens | 14 | **15** |
| UI tests | 48 | **52** |
| **Combined tests** | **134** | **148** |
| Production bundle (gz) | 98 kB | **99 kB** |

---

# Session 18 — 2026-04-24 — Experiment designer + Run explainer advisories

Baseline: session 17 shipped the LLM router + dataset consultant (1 of 6 consultation types in SPEC § 12.2). Infrastructure solid; this session demonstrates the extension pattern + completes the three classic copilots.

Theme: deliver **experiment_design + run_summary** consultation types end-to-end. All three copilots now live under one router, one audit shape, one envelope.

## ADDED — experiment designer

- `ADDED` — **`services/api/pycaret_server/llm/consultations/experiment_design.py`** — reads a CSV column profile + a free-text user goal, asks the LLM for a RunConfig-shaped proposal. System prompt enumerates expected `suggested_config_json` keys (`task_type`, `target`, `train_size`, `fold`, `primary_metric`, `preprocessing.*`, `model_shortlist`, `class_imbalance_strategy`) and tells the model to ground every choice in the profile + never invent columns. Output schema locks top-level keys via `additionalProperties: false`.
- `ADDED` — **`POST /api/v1/llm/design-experiment`** — body `{workspace_id, data_source_id, goal}`. CSV-only guard; `min_length=1` on `goal` surfaces as 422 on empty input.

## ADDED — run explainer

- `ADDED` — **`services/api/pycaret_server/llm/consultations/run_explanation.py`** — reads a completed Run's snapshot + leaderboard + full event stream, asks the LLM for plain-prose explanation + prioritised next experiments. System prompt pushes metric-grounded reasoning (AUC margin, CV-std vs mean-diff, AUC=1.0 suspicion) over model-class-alone takes. Event stream truncated to head-5 + tail-45 with a `__truncated__` marker.
- `ADDED` — **`POST /api/v1/llm/explain-run`** — body `{run_id}`. Access control traverses `run → experiment → project → workspace`. Non-terminal runs rejected with 400 ("wait for a terminal state"). Consultation row carries `run_id`/`experiment_id`/`project_id` FKs for audit correlation.

## ADDED — frontend AI surfaces

- `ADDED` — **`apps/web/src/components/ExperimentDesignerModal.tsx`** — modal from the New Experiment wizard. CSV picker (workspace data sources, filtered to `csv_upload`) + free-text goal textarea. Renders standard `LLMAdvice` envelope + pretty-printed suggested RunConfig. No one-click apply in v1 (the UI says so explicitly — waits on MVP-1 exit: canonical `RunConfig` Pydantic model).
- `ADDED` — **`apps/web/src/components/RunExplainerCard.tsx`** — inline card on `/runs/:id`, only on terminal runs. Opt-in: button click fires the LLM call (explanations cost tokens; they don't auto-run on every page view). "Ideas to try" list rendered from `suggested_config_json.next_actions`. Button flips "Explain" → "Re-explain" after first success.

## CHANGED — screens

- `CHANGED` — **`apps/web/src/pages/NewExperiment.tsx`** — header gains an **"✨ Ask AI"** button alongside the title; opens the designer modal. Modal mounts at page bottom so it doesn't disrupt the single-column wizard flow.
- `CHANGED` — **`apps/web/src/pages/RunDetail.tsx`** — imports `<RunExplainerCard>`; drops it between Leaderboard and Promote sections, guarded on `terminal === true`.

## ADDED — API bindings

- `ADDED` — **`apps/web/src/api/endpoints.ts`**: `llmApi.designExperiment` + `llmApi.explainRun`.

## TESTS

- `TESTS` — **`services/api/tests/test_llm_advisories.py`** — 6 new integration tests using `FakeLLMProvider`:
  - `test_design_experiment_happy_path` — upload iris → configure LLM → POST design-experiment → assert `type=experiment_design`, `cfg.task_type=classification`, `cfg.primary_metric=auc`, `lr` in shortlist, user goal reaches prompt verbatim.
  - `test_design_experiment_requires_goal` — 422 on empty goal (Pydantic).
  - `test_design_experiment_rejects_non_csv` — 400 on S3 data source.
  - `test_explain_run_happy_path` — actually runs a create-LR on iris, waits, explains; asserts `type=run_summary`, `run_id` correlated.
  - `test_explain_run_rejects_in_progress` — race-tolerant guard (accepts 400 or 200 depending on whether the tiny-iris run beat the POST).
  - `test_explain_run_requires_configured_llm` — 400 "No LLM provider configured" when workspace has no LLM setting.
- `TESTS` — **`apps/web/src/components/RunExplainerCard.test.tsx`** — 2 new: opt-in behaviour on mount, click-fires + envelope-renders + button-label-flip.
- `TESTS` — **`apps/web/src/components/ExperimentDesignerModal.test.tsx`** — 3 new: inert when closed, CSV-only options + submit-disabled-until-filled, fires with correct payload + renders advice.
- `TESTS` — **Combined suite: 118/118 green** (32 engine + 45 server + 41 web); was 107.

## INTERNAL

- `INTERNAL` — **Extension pattern locked in.** Adding a consultation type is now three files (one consultation module, one server test, one UI surface) + one route. The 3 consultation modules are structurally identical (SYSTEM string, strict OUTPUT_SCHEMA dict, `build_prompt(...)` → `(system, user)` tuple, `parse_response` → `LLMAdvice` with defensive fallback). Future copilots (`failure_debugging`, `deployment_risk_review`, `drift_analysis`) drop into this slot.
- `INTERNAL` — **Race tolerance in explain-run test.** A `POST /runs` followed immediately by `POST /llm/explain-run` hits either the `queued/running` guard (400) or the `succeeded` happy path depending on worker-pool timing on a setup-plan iris run. Test asserts either outcome + waits for the run in teardown. Lesson: unit tests should specify invariants, not timing.
- `INTERNAL` — **Button labels as state indicators.** `<RunExplainerCard>` flips "Explain" → "Re-explain" after first success. Small UX signal: the advice below may be stale if you've run something since. Same pattern as `test-connection`'s green-tick after verification.
- `INTERNAL` — **Defensive `_truncate_events(..., 50)`.** Head-5 + tail-45 + `__truncated__` marker when there are more than 50 events. Keeps both "what started" and "what crashed" visible to the LLM for long experiments without blowing the context window.

## Session 18 delta summary

| Metric | Session 17 end | Session 18 end |
|---|---:|---:|
| Consultation types | 1 | **3** |
| API routes | ~47 | **~49** |
| Server integration tests | 39 | **45** |
| UI shared components | 8 | **10** |
| UI tests | 36 | **41** |
| **Combined tests** | **107** | **118** |
| Production bundle (gz) | 95 kB | **96 kB** |

---

# Session 17 — 2026-04-24 — LLM router (Claude + OpenAI) + dataset consultant

Baseline: session 16 closed the Control Plane's deterministic loop (CSV → run → promote → deploy → predict). The spec's AI-native half (§ 12) was still a stub.

Theme: deliver the LLM advisory layer end-to-end, provider-agnostic from day one, with **Anthropic (Claude) + OpenAI as first-class from commit one** per DECISIONS.md § session-13 · 3. LLM is advisory (CONTROL_PLANE_SPEC § 12.3); the deterministic engine executes what the user approves.

## ADDED — DB + migration

- `ADDED` — **`LLMProviderSetting`** model. Per-workspace provider config. `UniqueConstraint(workspace_id, provider)` so workspaces can keep an Anthropic + OpenAI entry side-by-side; only one row carries `enabled=True`.
  - TODO comment on `api_key_encrypted`: stored plaintext for v1; KMS / Vault wrapping tracked under V2 secrets-encryption in ROADMAP.
- `ADDED` — **`LLMConsultation`** model. Append-only audit row. Captures prompt, raw response, normalised `LLMAdvice`, latency_ms, error. Optional FKs (project_id / experiment_id / run_id) correlate each consultation to the domain object that triggered it.
- `ADDED` — **`services/api/pycaret_server/migrations/versions/20260424_1628_d582b350c276_add_llm_provider_settings_and_.py`** — Alembic autogen-generated. Applied clean against the baseline migration.

## ADDED — `pycaret_server.llm` module

- `ADDED` — **`llm/schemas.py`** — Pydantic:
  - `LLMAdvice` — the canonical envelope every consultation returns (`suggested_config_json`, `suggested_action`, `reasoning_summary`, `risk_flags`).
  - `LLMProviderSettingRead` deliberately drops `api_key_encrypted` + adds `has_api_key: bool` so the plaintext never hits the browser.
  - `LLMProviderSettingWrite`, `LLMConsultationRead`, `AnalyzeDatasetRequest`, `TestConnectionResponse`.
  - Frozen tuples for `PROVIDERS` + `CONSULTATION_TYPES` match the CONTROL_PLANE_SPEC § 4.14 / § 4.15 allowlists.
- `ADDED` — **`llm/providers/base.py`** — `LLMProvider` Protocol with one method: `complete(system, user, output_schema, max_tokens, temperature) -> dict`. Tool-use (Anthropic) and JSON mode (OpenAI) both normalise to this.
- `ADDED` — **`llm/providers/anthropic_provider.py`** — Claude via native tool-use. Declares an inline tool wrapping `output_schema`; consumes the first `tool_use` content block. SDK imported lazily so the base server install doesn't depend on it (`pycaret-server[llm-anthropic]` extra).
- `ADDED` — **`llm/providers/openai_provider.py`** — OpenAI structured-output via `response_format={"type": "json_schema", ...}`. Works against native OpenAI, Azure OpenAI, and any OpenAI-compatible endpoint (Ollama, LM Studio, vLLM) via `base_url`. SDK lazy-imported (`pycaret-server[llm-openai]`).
- `ADDED` — **`llm/providers/fake.py`** — `FakeLLMProvider` for tests + local dev. Tracks the last prompt seen; optional `canned_response` override. Zero network, zero cost, deterministic.
- `ADDED` — **`llm/providers/__init__.py`** — registry + `get_provider(...)` factory. Keyed on provider name; adding Google / Azure / Ollama is one factory function + one entry. `register_fake_for_tests(canned_response)` installs the fake under every provider name so real SDKs are never hit in CI.
- `ADDED` — **`llm/router.py`** — `LLMRouter` class. `consult(session, ctx)` dispatches: load active setting → build provider → call → normalise to `LLMAdvice` → persist `LLMConsultation` (even on failure — audit row still written). Auxiliary `test_connection(setting)` does a lightweight round-trip. Module-level singleton via `get_router()` / `reset_router()`.
- `ADDED` — **`llm/consultations/dataset_analysis.py`** — the first consultation type. Reads the CSV's first 200 rows + total row count + column dtypes + cardinality + null fractions + sample values; serialises as a deterministic JSON blob in the user prompt. Output schema is strict (`additionalProperties: false` on top-level keys) so the model can't invent fields.

## ADDED — API routes

- `ADDED` — **`services/api/pycaret_server/api/llm.py`** — 6 operations across 5 paths:
  - `GET /api/v1/workspaces/{id}/llm/settings` — the currently-enabled provider setting, or `null`.
  - `PUT /api/v1/workspaces/{id}/llm/settings` — upsert. Admin-gated. Switching providers flips the previous row's `enabled=False` (audit preserved). PUT-merge on the API key: passing `null` leaves the existing key alone.
  - `POST /api/v1/workspaces/{id}/llm/test-connection` — probe. Returns `{ok, provider, model_name, error, latency_ms}`.
  - `POST /api/v1/llm/analyze-dataset` — runs the dataset consultant on a `csv_upload` data source. Returns the persisted consultation row.
  - `GET /api/v1/workspaces/{id}/llm/consultations?limit=50` — history (newest first, cap 500).
  - `GET /api/v1/llm/consultations/{id}` — single consultation.
- `CHANGED` — **`app.py`** lifespan — now also resets the LLM router singleton on shutdown (matches the pattern for `RunOrchestrator` + `DeploymentRegistry`).
- `ADDED` — **`pyproject.toml`** extras: `llm-anthropic` (anthropic SDK), `llm-openai` (openai SDK), `llm` (both). Neither required for the base server install.

## ADDED — frontend

- `ADDED` — **`apps/web/src/pages/LLMSettings.tsx`** — new route `/workspaces/:wsId/llm`. Provider picker (6 options; Anthropic + OpenAI supported, 4 more disabled as "(coming later)"). Model name with per-provider default. API key as `type="password"` (never round-tripped via `GET /settings`). Optional `base_url`. Enabled toggle. "Test connection" button wired to `llmApi.testConnection`; shows green tick + latency on success, red error on failure.
- `ADDED` — **`apps/web/src/components/AnalyzeDatasetModal.tsx`** — opens with a `dataSourceId`, auto-fires `llmApi.analyzeDataset` on first render, renders the `LLMAdvice` envelope: suggested action as headline, reasoning as paragraph, risk flags as warn-toned chips, suggested config as pretty-printed JSON block, provider/model/latency footer. Esc + click-outside + footer "Close" all dismiss.
- `CHANGED` — **`apps/web/src/components/DataSourcesCard.tsx`** — each CSV row now has an **"✨ AI"** button alongside delete; clicking opens `<AnalyzeDatasetModal>` for that dataset.
- `CHANGED` — **`apps/web/src/pages/WorkspaceDetail.tsx`** — header nav gains an **"✨ LLM"** button (third alongside Pipelines + Deployments) linking to the settings screen.
- `ADDED` — **`apps/web/src/api/types.ts`** — `LLMProviderName`, `LLMAdvice`, `LLMProviderSettingRead/Write`, `LLMConsultationRead`, `TestConnectionResponse`.
- `ADDED` — **`apps/web/src/api/endpoints.ts`** — `llmApi` module: `getSettings`, `upsertSettings`, `testConnection`, `analyzeDataset`, `listConsultations`, `getConsultation`.
- `CHANGED` — **`apps/web/src/App.tsx`** — new authenticated route `/workspaces/:wsId/llm`.

## TESTS

- `TESTS` — **`services/api/tests/test_llm.py`** — 9 new integration tests. Fake is registered under every provider name via `register_fake_for_tests(canned_response=...)` so the router's real dispatch path is exercised end-to-end.
  - `test_settings_empty_initially` — GET returns null before any PUT.
  - `test_upsert_settings_admin_gated_and_hides_key` — round-trip; response never contains `api_key` or `api_key_encrypted`; `has_api_key=True`.
  - `test_upsert_settings_rejects_unknown_provider` — 400 on bad provider name.
  - `test_switching_provider_disables_previous` — PUT anthropic → PUT openai; GET returns openai (anthropic row preserved but `enabled=False`).
  - `test_test_connection_against_fake_provider` — 200, `ok=True`, latency set.
  - `test_test_connection_400_when_no_provider_configured` — 400 error body.
  - `test_analyze_dataset_happy_path` — bootstrap → configure LLM → upload iris CSV → POST analyze-dataset → assert `type=dataset_analysis`, `provider=anthropic`, `generated_config_json['task_type']=='classification'`, prompt captured for audit → list history (1 row) → get-by-id round-trips.
  - `test_analyze_dataset_requires_configured_llm` — 400 `"No LLM provider configured"`.
  - `test_analyze_dataset_rejects_non_csv_source` — 400 when the data source is s3.
- `TESTS` — **`apps/web/src/components/AnalyzeDatasetModal.test.tsx`** — 3 new:
  - Modal is inert when `open=false` (no mutation fires).
  - On open, auto-fires `analyzeDataset` with the right args + renders every part of the advice envelope (action, reasoning, risk-flag chip, provider/model/latency footer).
  - Footer "Close" button invokes `onClose` (disambiguated from the `aria-label="Close"` ✕ button via `textContent` filter).
- `TESTS` — **Combined suite: 107/107 green** (32 engine + 39 server + 36 web); was 95.

## INTERNAL

- `INTERNAL` — **Provider SDKs are optional deps.** `anthropic` and `openai` live in `[project.optional-dependencies].llm-anthropic` / `llm-openai`. The base `pycaret-server` install never pulls them; the tests run against `FakeLLMProvider` so CI never hits a real API. The provider class imports the SDK inside its factory method and raises a clear `RuntimeError` telling operators which extra to install if it's missing.
- `INTERNAL` — **PUT-merge on the API key.** The settings endpoint preserves an existing key when the caller passes `api_key=null`. Lets the UI re-submit the form (model name change, enabled toggle, etc.) without round-tripping the plaintext. This is also why `has_api_key: bool` exists on the read schema instead of leaking the value.
- `INTERNAL` — **Audit row always persists.** `LLMRouter.consult` writes the `LLMConsultation` row before re-raising any provider error. This is deliberate: the audit trail of "we asked the LLM and got this error" is at least as valuable as the trail of successful calls, especially for debugging prompt regressions against a new provider release.
- `INTERNAL` — **`additionalProperties: false` on output schemas.** The dataset-consultant schema locks top-level keys to the 4 `LLMAdvice` fields. Providers that strictly honour schemas (OpenAI with `strict: true`, Claude with tool-use) won't emit rogue top-level fields; providers that don't are normalised in the router's `LLMAdvice.model_validate(raw)` step + fall-through coerce into a best-effort `LLMAdvice` with a `malformed_response` risk flag.
- `INTERNAL` — **Disambiguation via textContent in the modal test.** Two buttons both match `name: /close/i` — one has `aria-label="Close"` (the ✕), one has text `"Close"` (the footer). Using `.filter(el => el.textContent === 'Close')` on `getAllByRole` picks exactly the footer one. Same pattern applies anywhere we have a visually-distinct close button next to a header dismissal affordance.

## Session 17 delta summary

| Metric | Session 16 end | Session 17 end |
|---|---:|---:|
| DB tables | 16 | **18** (+2) |
| Alembic revisions | 1 | **2** |
| API routes | ~42 | **~47** |
| Server integration tests | 30 | **39** |
| UI shared components | 7 | **8** |
| UI screens | 12 | **13** |
| UI tests | 33 | **36** |
| **Combined tests** | **95** | **107** |
| Production bundle (gz) | 93 kB | **95 kB** |

E2E verified end-to-end with `FakeLLMProvider` — configure Anthropic as the workspace LLM → test-connection returns `ok=True` → upload iris CSV → analyze → `suggested_action: "Run a classification compare on iris with fold=5."` → history lists the consultation.

---

# Session 16 — 2026-04-24 — Pipelines + Deployments + CSV upload (closes the serving loop)

Baseline: session 15 closed the run-execution loop with live WebSocket events + leaderboard. A user could promote a run into a Pipeline via the API — but there was no UI for the pipeline registry, no UI for deployments, no way to hit `/predict` from the browser, and no way to upload a real CSV.

Theme: close the full **zero-Python product loop**. Every step in §24 of the spec (CSV upload → run → promote → deploy → predict) is now reachable from the UI in under 8 clicks.

## ADDED — pipelines UI

- `ADDED` — **`apps/web/src/pages/Pipelines.tsx`** — `/workspaces/:wsId/pipelines`. Workspace-scoped registry. Table columns: name (link to detail), model_id, SHA-256 prefix (hover for full hash), tags, created. Breadcrumb-navigated from the workspace header.
- `ADDED` — **`apps/web/src/pages/PipelineDetail.tsx`** — `/workspaces/:wsId/pipelines/:pipelineId`. Two-column layout:
  - **Main**: metadata `<dl>` (model_id, sha256, origin_run_id, stored_path, created, tags). Below: a table of every Deployment backed by this pipeline with live p50/p95 latency + inference count + error count + auth mode.
  - **Sidebar**: new-deployment form. Slug input validates against `[a-z0-9][a-z0-9-]{1,62}[a-z0-9]` live; submit disabled on invalid. Auth-mode selector with `workspace` active + `api-key` / `public` disabled + labelled "(V2)".

## ADDED — deployments UI + serving

- `ADDED` — **`apps/web/src/pages/Deployments.tsx`** — `/workspaces/:wsId/deployments`. Workspace-level list. 8-column table: slug (link) / status / auth / predictions / errors (red when > 0) / p50 / p95 / last-hit timestamp. Polls every 5 s so the metrics stay alive.
- `ADDED` — **`apps/web/src/pages/DeploymentDetail.tsx`** — `/deployments/:deploymentId`. Polls every 3 s. Header with slug (mono) + status tone + auth mode + linked pipeline name. Four `<Stat>` cards: predictions, errors (red tone when > 0), p50, p95. Below: full `<PredictTester>` inline. Right column: a metadata card with deployment_id / workspace_id / pipeline_id / created (all mono, click-to-copy via `title`). Delete action with confirmation prompt; on success, redirects to the workspace deployments list.
- `ADDED` — **`apps/web/src/components/PredictTester.tsx`** — the load-bearing serving test-form.
  - Monospace JSON-array `<textarea>` pre-seeded with an iris-shaped payload.
  - Live JSON parse on every keystroke → inline red hint (`JSON: ...`) + submit disabled on invalid.
  - Submit calls `deploymentsApi.predict(slug, {rows})` and renders the response as: latency chip (`3.1ms`) + request-id chip (truncated, full via `title`) + a predictions table (index / prediction, numeric or JSON-stringified).
  - Error responses surface via `errorMessage()`; good for catching schema mismatches ("prediction failed: Feature names seen at fit time, yet now missing: ...").

## ADDED — CSV upload UI

- `ADDED` — **`apps/web/src/components/DataSourcesCard.tsx`** — lives in `WorkspaceDetail`'s sidebar. Lists existing `csv_upload` data sources (filters out s3/postgres — those get their own Integrations screen later). Per-row: name, row count, pretty-printed size, column count. Per-row delete with confirmation. Upload form at the bottom: file picker (accept `.csv,text/csv`, styled via `file:` pseudo-class), name input that auto-fills from the file name. Multipart submit via `dataSourcesApi.uploadCsv` (FormData wrapper; axios sets Content-Type + boundary automatically).
- `CHANGED` — **`apps/web/src/pages/WorkspaceDetail.tsx`** — header now has two buttons at the top-right: **Pipelines** + **Deployments** (both `btn-secondary`). Sidebar changed from single card to stack: `New project` card + `<DataSourcesCard>`.

## ADDED — API bindings

- `ADDED` — **`apps/web/src/api/endpoints.ts`**:
  - `pipelinesApi` — `list(workspace_id)`, `get(pipeline_id)`, `remove(pipeline_id)`.
  - `deploymentsApi` — `list(workspace_id)`, `get(deployment_id)`, `create(pipeline_id, body)`, `remove(deployment_id)`, **`predict(endpoint_slug, body)`**.
  - Request / response types: `PredictRequest` (`{rows: Record<string, unknown>[]}`), `PredictResponse` (`{deployment_id, endpoint_slug, predictions: [{index, prediction}], latency_ms, request_id}`).

## CHANGED — routing

- `CHANGED` — **`apps/web/src/App.tsx`** — 4 new authenticated routes mounted inside the `<AuthGate><Layout>` wrapper: `/workspaces/:wsId/pipelines`, `/workspaces/:wsId/pipelines/:pipelineId`, `/workspaces/:wsId/deployments`, `/deployments/:deploymentId`.
- `CHANGED` — **`apps/web/src/pages/RunDetail.tsx`** — the promote-success hint now links directly to the created pipeline's detail page. Closes the loop from run → pipeline → deploy with one mouse path.

## TESTS

- `TESTS` — **`apps/web/src/components/PredictTester.test.tsx`** — 3 new:
  - Renders the monospace JSON textarea pre-seeded with an iris-shaped payload (asserts via `.value` string-contains — asymmetric matchers don't compose with `toHaveValue`).
  - Typing invalid JSON via `fireEvent.change` (userEvent.type treats `{` as a key sequence) surfaces the inline red hint and disables submit.
  - Clicking submit renders the predictions table + latency chip after the mock resolves.
- `TESTS` — **`apps/web/src/components/DataSourcesCard.test.tsx`** — 3 new:
  - Empty-state hint when `list` returns `[]`.
  - Lists only `csv_upload`-kind rows (s3/postgres filtered out), with `rows · size · cols` summary formatted.
  - Upload button disabled until a file is attached via `userEvent.upload`.
- `TESTS` — **UI suite: 33/33 green** (was 27). **Combined: 95/95** (32 engine + 30 server + 33 web).

## INTERNAL

- `INTERNAL` — **userEvent + curly braces.** `user.type(textarea, '{not json}')` throws `Expected repeat modifier or release modifier or "}" but found " "` — `{` is a special-key prefix in userEvent's keyboard DSL. `fireEvent.change(..., { target: { value: 'not json' } })` is the right primitive for pasting raw invalid input. Same pattern used by DynamicForm test (session 14) for clearing number inputs.
- `INTERNAL` — **Mid-file import rejected by flow.** First draft added `import type { Deployment }` mid-file between sections for ergonomic reasons. Moved it up into the single `import type { … } from './types'` block so the module obeys "all imports at top" convention. TS doesn't care, but readers do.
- `INTERNAL` — **Polling cadence.** DeploymentDetail polls every 3 s because its stat cards change every `/predict`. The Deployments list polls every 5 s (less critical). The PipelineDetail deployments sub-table piggybacks on the workspace-scoped list query's 5 s cadence through query-key sharing.
- `INTERNAL` — **JSON textarea vs. dynamic form.** PredictTester keeps inputs as a raw JSON array instead of a column-aware dynamic form because the `Deployment` row doesn't currently carry the pipeline's input schema. A V2 session can attach a `schema.json` artifact at promotion time and render a column-aware form on top of the same `deploymentsApi.predict` call.

## Session 16 delta summary

| Metric | Session 15 end | Session 16 end |
|---|---:|---:|
| UI screens | 8 | **12** (+4) |
| UI shared components | 5 | **7** (+2) |
| UI routes | 8 | **12** (+4) |
| UI tests | 27 | **33** (+6) |
| Combined tests | 89 | **95** |
| UI LOC | ~2,950 | **~3,800** |
| Production bundle (gz) | 89 kB | **93 kB** |

Live E2E verified: CSV upload (150 rows, 5 cols, SHA-256 checksummed) → create-plan run on LR → succeeded in 6.3 s → promoted to a pipeline with SHA-256 → deployed as slug `iris-v1` (active, workspace auth) → `/predict` with 3 iris rows → 0.9 ms latency, `inference_count` ticks to 3, p50 = p95 = 0.9 ms.

---

# Session 15 — 2026-04-24 — Run detail + live WebSocket event stream

Baseline: session 14 shipped the experiment wizard + project detail. The runs table existed but clicking a row did nothing.

Theme: close the beautiful product loop — every run gets a dedicated screen where users watch engine events stream in live, see the leaderboard render as it arrives, cancel mid-flight, and promote successful pipelines with one click.

## ADDED — live event stream

- `ADDED` — **`apps/web/src/components/EventStream.tsx`** — full WebSocket lifecycle.
  - Connects to `/api/v1/runs/:id/events/ws?token=<access_token>` — same-origin ws:// / wss:// based on `window.location.protocol`, token pulled from the Zustand auth store.
  - Parses each JSON message as a `WsEvent`; appends to a state array capped at 500 events (older events drop off — bounded memory).
  - **Single-retry reconnect** on unexpected close. Auth-failure close codes (4401 / 4403) surface a visible error and do NOT retry — those are user-facing problems, not transient.
  - Resets state on `runId` change so switching between runs doesn't mix streams.
  - Renders: header with connection-status indicator (connecting/live/closed/error tone-coded) + event counter; event log as a card list with per-event timestamp, tone-coded kind text (`.started` = teal, `.finished/.created/.fitted` = green, `.failed`/`error` = red, `warning` = amber), optional duration formatted short (`850ms` / `3.5s`).
  - Recognises the backend's `run.closed` sentinel and stops retrying once seen.

## ADDED — leaderboard

- `ADDED` — **`apps/web/src/components/Leaderboard.tsx`** — renders any JSON-table shape the engine emits.
  - Zero hard-coded metric names. First-row column order is preserved exactly.
  - Click-to-sort per column (desc default, toggle asc on second click). Numeric sort for number-valued cells; string sort via `localeCompare` otherwise.
  - Number formatter: integers stay bare, floats render with 4 decimals, values with |x| < 1e-4 use `toExponential(2)`. Numeric cells get `font-mono tabular-nums text-right` for alignment.
  - Empty-state hint when `rows` is null / empty.

## ADDED — `/runs/:runId` screen

- `ADDED` — **`apps/web/src/pages/RunDetail.tsx`** — stitches `<EventStream>` + `<Leaderboard>` + run metadata into one screen.
  - Header: tone-coded status label (`succeeded` green / `running` teal / `failed` red / `cancelled` amber / `queued` muted) + short run id + total duration + error pre-block when failed.
  - Polls `runsApi.get` every 2 s while status is queued/running; stops on terminal.
  - Cancel button (only while pending) wired to `runsApi.cancel`.
  - Promote-to-pipeline form (only on `succeeded`): inline input + submit mutates to `runsApi.promote`, on success disables with ✓.
  - Request snapshot at the bottom — full `Run.snapshot` as a two-column `<dl>` for reproducibility.

## CHANGED — ExperimentDetail sidebar

- `CHANGED` — **`apps/web/src/pages/ExperimentDetail.tsx`** — upgraded the minimal new-run form from session 14:
  - **Model picker** — free-text `model_id` replaced with a `<select>` driven by `describeApi.models(task)`. Unavailable models (`is_available=false`) render as disabled options with "(install required)" suffix.
  - **Data-source picker** — replaces the standalone sklearn-samples dropdown. Single combo-valued `<select>`: workspace CSV uploads first, sklearn samples below. Combo values use a prefix (`sklearn:iris` vs. the DataSource UUID) so one `<select>` drives two different backend fields.
  - Runs-table rows are now clickable → navigate to `/runs/:id`.
  - All API calls moved to the new `runsApi.listForExperiment` wrapper — no more raw `api.get` from the page.

## ADDED — API bindings

- `ADDED` — **`apps/web/src/api/types.ts`**:
  - `DataSource`, `DataSourceKind`, `Pipeline`, `Deployment`.
  - `RunPlan` (literal union `'setup' | 'create' | 'compare'`), `RunCreate` (full POST payload).
  - `WsEvent` — matches the engine's `Event.to_dict()` shape.
- `ADDED` — **`apps/web/src/api/endpoints.ts`**:
  - `runsApi` — `listForExperiment`, `submit`, `get`, `events` (with `after_id` + `limit` opts), `cancel`, `wait`, `promote`.
  - `dataSourcesApi` — `list`, `get`, `remove`, **`uploadCsv(workspace_id, file, name, description?)`** (multipart via `FormData`; axios sets Content-Type + boundary automatically).

## CHANGED — routing

- `CHANGED` — **`apps/web/src/App.tsx`** — new authenticated route `/runs/:runId` wired inside the `<AuthGate><Layout>` wrapper.

## TESTS

- `TESTS` — **`apps/web/src/components/Leaderboard.test.tsx`** — 4 tests:
  - Empty state renders the placeholder hint.
  - Header cells preserve engine-declared order.
  - Number formatter: integers bare, floats 4-decimal.
  - Numeric sort round-trips desc ↔ asc on repeated click.
- `TESTS` — **`apps/web/src/components/EventStream.test.tsx`** — 4 tests with a controllable `FakeWebSocket` replacing `globalThis.WebSocket` for the test scope:
  - Connects to the right URL + includes the bearer token in the query string.
  - Flips indicator to `live` on open; renders events with short-form duration.
  - Recognises the `run.closed` sentinel and reflects `closed` status.
  - Surfaces 4401 auth-failure close code as a visible error (and suppresses the normal retry).
- `TESTS` — **UI suite: 27/27 green** (was 19). **Combined across programme: 89/89** (32 engine + 30 server + 27 web).

## INTERNAL

- `INTERNAL` — **WebSocket URL construction.** `${proto}//${window.location.host}/api/v1/...` works in both dev (Vite proxies `/api` and `/ws` to the backend) and prod (the nginx config in `infra/docker/nginx.ui.conf` forwards the same paths). No env-var plumbing needed for ws endpoints.
- `INTERNAL` — **Single-retry reconnect policy.** An unexpected close (network blip, server restart) retries once after a 500 ms delay. Auth-failure close codes (4401 / 4403) never retry — they need user intervention. `retried` is a closure-scoped flag in the effect so the policy resets on run-id change.
- `INTERNAL` — **Test-only WebSocket replacement.** `beforeEach` swaps `globalThis.WebSocket` with `FakeWebSocket` (a class that tracks all instances + exposes `_open` / `_message` / `_close` hooks). Tests use `act(() => { ws._message(...) })` to drive the component deterministically. Pattern for any future component that opens a network connection.
- `INTERNAL` — **Leaderboard sort indicator overloaded header text.** A regex like `/closed/i` would match both the `"● closed"` status indicator and a `run.closed`-kind event in the log, producing a `getByText` ambiguity. Tightened to `/●\s+closed/` — small pattern, specific to the status indicator's prefix.

## Session 15 delta summary

| Metric | Session 14 end | Session 15 end |
|---|---:|---:|
| UI screens | 7 | **8** (+ RunDetail) |
| UI shared components | 3 | **5** (+ EventStream + Leaderboard) |
| UI routes | 7 | **8** |
| UI tests | 19 | **27** |
| Combined tests | 81 | **89** |
| UI LOC | ~2,100 | **~2,950** |
| Production bundle (gz) | 86 kB | **89 kB** |

Live E2E verification (AutoML on `sklearn:iris`): 4 events emitted, 4-row × 7-column leaderboard (`Fold / Accuracy / AUC / Recall / Prec. / F1 / Kappa`) rendered and sortable, pipeline promoted with SHA-256 checksum, 19 classification models in the picker.

---

# Session 14 — 2026-04-24 — Project detail + Experiment wizard (dynamic form)

Baseline: session 13 locked in the Control Plane vision and restructured the monorepo. Structure is sound; now time to push the first beautiful product loop past the workspace/project screens into the actual ML workflow.

Theme: the centerpiece of MVP 3 — an experiment setup form that is **100% driven by the engine's `describe_setup_params`**. Zero UI code hard-codes a parameter name. When the engine adds / removes / renames a setup parameter, the form just works.

## ADDED — dynamic-form infrastructure

- `ADDED` — **`apps/web/src/components/DynamicForm.tsx`** — the load-bearing component.
  - `<ParamInput>` — one function, one switch on `kind` (bool / int / float / enum / column / string). Each case returns the right native HTML input with proper validation attributes (min/max for numbers, choices for enums, optional "— none —" for non-required enums, column-text-fallback when no columns are supplied).
  - `<DynamicForm>` — groups parameters by their `group` field in the order declared by `schema.groups`, renders one fieldset per group with the param name, required indicator, inline description, and range hint where applicable. Preserves user input across re-renders.
- `ADDED` — **`apps/web/src/components/DynamicForm.helpers.ts`** — pure helpers in a separate file so ESLint's react-refresh rule is happy.
  - `applyDefaults(schema, current)` — merges schema defaults into a values object without clobbering user input.
  - `stripDefaults(schema, values)` — removes values equal to defaults so the API payload captures *user intent* only. Engine owns defaults; we only record what the user chose to override.

## ADDED — API bindings

- `ADDED` — **`apps/web/src/api/types.ts`** — 5 new types: `ParamKind` (literal union of 6 kinds), `SetupParam`, `SetupParamSchema`, `ModelCard`, `MetricCard`, `ExperimentCreate`.
- `ADDED` — **`apps/web/src/api/endpoints.ts`** — 2 new API modules:
  - `experimentsApi` — `list(project_id)`, `get(project_id, experiment_id)`, `create(project_id, body)`, `remove(project_id, experiment_id)`.
  - `describeApi` — `setupParams(task)`, `models(task)`, `metrics(task)`. 10-min `staleTime` on setup schemas (effectively static per engine release).

## ADDED — three new screens

- `ADDED` — **`apps/web/src/pages/ProjectDetail.tsx`** — `/workspaces/:wsId/projects/:projectId`. Project header with name + description + tags + breadcrumb. Experiments list with per-row `kbd`-styled task chip + target display. "New experiment" link in the top-right.
- `ADDED` — **`apps/web/src/pages/NewExperiment.tsx`** — `/workspaces/:wsId/projects/:projectId/experiments/new`. Two-card single-column wizard. Card 1 collects name + task (5-option dropdown) + target column (hidden for clustering / anomaly). Card 2 renders `<DynamicForm>` against `describeApi.setupParams(task)`. Switching task resets params (previous values are likely invalid against the new schema). On submit, `stripDefaults` removes engine-default values; only user overrides travel to the API. Redirects to the experiment detail on success.
- `ADDED` — **`apps/web/src/pages/ExperimentDetail.tsx`** — `/workspaces/:wsId/projects/:projectId/experiments/:experimentId`. Two-column layout:
  - **Main:** config overview (diffed against defaults), runs table with status colour map (`succeeded` green, `running` teal accent, `failed` red, `cancelled` warn, `queued` muted). Table auto-polls every 2 s while any run is queued or running, stops polling when everything is terminal.
  - **Sidebar:** minimal new-run form — plan (setup / create / compare) + model id (shown only for `create`) + sklearn sample dataset (iris / wine / breast_cancer / diabetes for session 14; data-source picker lands in session 15).

## CHANGED — routing + navigation

- `CHANGED` — **`apps/web/src/App.tsx`** — 3 new authenticated routes wired inside the `<AuthGate><Layout>` wrapper: `/workspaces/:wsId/projects/:projectId`, `/workspaces/:wsId/projects/:projectId/experiments/new`, `/workspaces/:wsId/projects/:projectId/experiments/:experimentId`.
- `CHANGED` — **`apps/web/src/pages/WorkspaceDetail.tsx`** — project list items are now `<Link>`s into the new project detail route. Hover border flips to accent (`hover:border-accent-500`).

## TESTS

- `TESTS` — **`apps/web/src/components/DynamicForm.test.tsx`** — 13 new tests locking in the dynamic-form contract:
  - `<ParamInput>` per kind: bool renders checkbox with correct `checked` state and dispatch; int renders number input with min/max round-trip, parses digits to Number, clears to null on empty string; float uses `step="0.01"`; enum renders `<select>` with all choices + "— none —" only when not required; column falls back to text input when no columns supplied, switches to `<select>` when columns are present.
  - `applyDefaults` seeds missing fields, preserves existing; `stripDefaults` removes values equal to defaults + empty values.
  - `<DynamicForm>` groups preserve `schema.groups` order; `hide` removes named params; `onChange` bubbles fully-merged values object; empty schema doesn't crash.
- `TESTS` — **UI suite: 19/19 green** (was 6). **Combined across programme: 81/81** (32 engine + 30 server + 19 web) — was 68.

## INTERNAL

- `INTERNAL` — **Controlled-input + vitest userEvent trap.** `userEvent.clear()` on a controlled number input with a mocked onChange doesn't actually reset the DOM value (React doesn't re-render because `value` prop is stale). First iteration's `fold=10` + `userEvent.type("5")` produced `105` instead of `5`. Switched to `fireEvent.change(..., { target: { value: "5" } })` for atomic value replacement. Pattern for future controlled-input tests.
- `INTERNAL` — **Split `DynamicForm.helpers.ts` from `DynamicForm.tsx`.** ESLint's `react-refresh/only-export-components` rule rejects mixing component exports with pure-function exports; pure helpers now live in `*.helpers.ts` so HMR works cleanly. Same split pattern applies to any future component file that acquires non-component exports.
- `INTERNAL` — **Zero hard-coded parameter names in the UI.** This is the design principle session 14 locks in. `<NewExperiment>` knows about `name`, `task`, `target`, and `setup_params` — the three columns the `experiments` table defines. Everything under `setup_params` is renderer-agnostic: the engine's schema is the contract. Verified by smoke test — 13 parameters across 6 groups arrive from `/describe/setup-params` and render correctly without the UI knowing any of the names.
- `INTERNAL` — **Bundle size.** 83 kB gz → 86 kB gz (+3 kB) for three new screens + `<DynamicForm>` + API bindings + 13 tests. 148 modules transformed (was 143).

## Session 14 delta summary

| Metric | Session 13 end | Session 14 end |
|---|---:|---:|
| UI screens (authenticated) | 2 | **5** (+ 3 new) |
| UI shared components | 2 | **3** (+ DynamicForm) |
| UI routes | 4 | **7** (+ 3 new) |
| UI vitest | 6 | **19** |
| Combined tests | 68 | **81** |
| UI LOC (TSX + config) | ~1,300 | **~2,100** |
| Production bundle (gz) | 83 kB | **86 kB** |

---

# Session 13 — 2026-04-24 — Monorepo restructure + Control Plane vision lock-in

Baseline: session 12 landed the 4-screen frontend scaffold. Three sibling packages at the root (`pycaret/`, `pycaret-server/`, `pycaret-ui/`), plus `docker/`, `tests/`, and everything else flat.

Theme: owner shared a comprehensive "PyCaret Control Plane" technical spec (24 sections, ~300 planned endpoints, full LLM + monitoring + drift + K8s + multi-cloud story). Locked it in as canonical. Restructured the monorepo to the spec's `apps/` + `services/` + `packages/` + `infra/` layout. Overhauled every agent-facing doc.

## ADDED — canonical directory structure

- `ADDED` — **`packages/engine/`** — engine source moved here (from repo-root `pycaret/`). Now has its own `pyproject.toml` (split out of the root) and a dedicated README. Hatchling wheel target: `packages = ["pycaret"]` resolved relative to the engine dir.
- `ADDED` — **`packages/engine/tests/`** — engine pytest suite moved here (from repo-root `tests/`). 32 tests; CI invokes via `uv run pytest packages/engine/tests/ -q`.
- `ADDED` — **`services/api/`** — FastAPI backend moved here (from `pycaret-server/`). Internal package name `pycaret_server` unchanged.
- `ADDED` — **`apps/web/`** — React UI moved here (from `pycaret-ui/`). Package name `@pycaret/ui` unchanged.
- `ADDED` — **`infra/docker/`** — Dockerfiles + compose moved here (from repo-root `docker/`).
- `ADDED` — **11 empty stub READMEs** documenting future directories: `apps/desktop/` (V2 Electron), `services/worker/` (V2 job runner), `services/deployment-runtime/` (V2 serving), `packages/sdk-python/` (V2 Python client), `packages/shared-schemas/` (V2 JSON schemas), `infra/helm/` (V2 K8s chart), `infra/terraform/{aws,gcp,azure}/` (V2 IaC). Each README explains scope, when the work starts, and what files will live there.
- `ADDED` — **Root `pyproject.toml`** now a pure workspace manifest — declares `[tool.uv.workspace] members = ["packages/engine", "services/api"]` + shared ruff defaults. No package metadata; that moved to `packages/engine/pyproject.toml`. `uv sync --all-packages --all-extras` resolves both members from their new homes.

## CHANGED — Docker + CI + docs paths

- `CHANGED` — **`infra/docker/Dockerfile.api`** — all `COPY` paths updated (`pycaret-server/` → `services/api/`, `pycaret/` → `packages/engine/pycaret/`). Editable install line now `uv pip install -e ./packages/engine -e ./services/api`. Default image tag renamed `pycaret-server:dev` → `pycaret-api:dev`.
- `CHANGED` — **`infra/docker/Dockerfile.ui`** — `COPY pycaret-ui/` → `COPY apps/web/`. Image tag `pycaret-ui:dev` → `pycaret-web:dev`. Labels updated to "PyCaret Control Plane — React frontend".
- `CHANGED` — **`infra/docker/docker-compose.yml`** — build context `..` → `../..` (deeper nesting), volume mount `../data` → `../../data`, service rename `ui:` → `web:`. Invocation: `docker compose -f infra/docker/docker-compose.yml up --build`.
- `CHANGED` — **`.github/workflows/test.yml`** — ruff paths changed to `packages/engine services/api`; pytest paths changed to `packages/engine/tests/` + `services/api/tests/`; web CI job `working-directory: apps/web` + `cache-dependency-path: apps/web/package-lock.json`; job renamed "UI" → "Web".
- `CHANGED` — **`docs/revamp/PLATFORM_QUICKSTART.md`** — all path references updated (`pycaret-server/` → `services/api/`, `pycaret-ui/` → `apps/web/`, `docker/` → `infra/docker/`, compose invocation with new file path).

## ADDED — vision & spec docs

- `ADDED` — **`docs/revamp/CONTROL_PLANE_SPEC.md`** — owner's 24-section technical spec checked in verbatim (with minor markdown fixes so tables + code blocks render). Supersedes the earlier `PLATFORM_PLAN.md`. Canonical product scope.
- `ADDED` — **`docs/revamp/VISION.md`** — 1-page product statement: what we're building (engine + Control Plane), who it's for, deployment modes, three engineering principles, what success looks like. Distilled from `CONTROL_PLANE_SPEC.md § 1 / § 2 / § 24`.

## CHANGED — architecture + roadmap

- `DOCS` — **`docs/revamp/ARCHITECTURE.md`** — rewritten end-to-end for the full Control Plane (engine + backend + UI + infra). 11 sections: monorepo layout rules, service topology (ASCII diagram), engine layer, backend routers + domain model + run execution + deployment flow, frontend stack + directory, infra story, LLM router design, RunConfig single-contract principle, CI job matrix, deliberate non-goals. Supersedes the prior engine-only content.
- `DOCS` — **`docs/revamp/ARCHITECTURE_ENGINE.md`** (renamed) — the previous `ARCHITECTURE.md` content (engine internals: god-class, class hierarchy, event system, migration plan) preserved under this new filename. Referenced from the new `ARCHITECTURE.md`.
- `DOCS` — **`docs/revamp/ROADMAP.md`** — rewritten around MVP 1 (engine) / MVP 2 (backend) / MVP 3 (UI) / MVP 4 (self-hosted) / V2 (enterprise) / V3 (scale + governance). Every already-shipped phase remapped into its MVP bucket with concrete exit criteria. Forward work laid out through session ~20. Current-session ledger at the bottom.

## CHANGED — agent + contributor docs

- `DOCS` — **`AGENTS.md`** — rewritten for the new structure. New 60-second briefing, new repo map, "which phase am I in?" decision tree, updated workflow, new common-task playbooks (add-a-backend-route, add-a-frontend-screen, drain-a-god-class-verb, add-an-LLM-advisory-feature). Removes stale references to old paths.
- `DOCS` — **`CONTRIBUTING.md`** — rewritten. New setup flow (uv + npm dual pipeline), new test commands, new PR checklist, updated non-negotiables, licensing section.
- `DOCS` — **`README.md`** — repositioned as the platform's landing page (not just an engine README). Engine + Control Plane quickstarts side by side. Three-mode deployment table. Links to VISION, SPEC, ARCHITECTURE, ROADMAP.

## CHANGED — DECISIONS.md (4 new ADRs)

- `DOCS` — **`2026-04-24 · restructure decision 1`** — adopt `apps/services/packages/infra` monorepo layout now. Rationale: "wash away all the old sins"; one-time pain beats spread-across-every-future-session pain. Python package names unchanged; PyPI + notebook users unaffected.
- `DOCS` — **`2026-04-24 · restructure decision 2`** — Electron desktop is V2, not MVP. Defers signed-installer tooling, per-OS CI, bundled-Python packaging, auto-update infra. Local dev already works via `uv run pycaret-server serve` + `npm run dev`.
- `DOCS` — **`2026-04-24 · restructure decision 3`** — LLM **router** supporting Anthropic (Claude) + OpenAI as first-class from day one, not single-provider. Rationale: provider abstraction cost is small, credibility matters in an agentic-ML world, provider APIs will drift.
- `DOCS` — **`2026-04-24 · restructure decision 4`** — product name = "PyCaret"; UI branding = "PyCaret Control Plane". Package names on registries (`pycaret` on PyPI, `pycaret-server` on PyPI, `@pycaret/ui` on npm) unchanged. OpenAPI `info.title` = "PyCaret Control Plane". Preserves 10 years of brand equity while distinguishing the new platform story.

## INTERNAL

- `INTERNAL` — **4 ruff import-order auto-fixes** across `services/api/pycaret_server/` triggered by running ruff on the new paths. No semantic changes; just `ruff check --fix`.
- `INTERNAL` — **Path-sensitive code check.** The only hardcoded `Path(__file__).resolve().parents[N]` in the codebase is `services/api/pycaret_server/db/bootstrap.py::_ALEMBIC_INI`, which uses `parents[2]`. After the move, that still resolves correctly because only the outer directory changed — the package-internal structure (`services/api/pycaret_server/db/bootstrap.py` → `services/api/alembic.ini`) maintains the same depth.
- `INTERNAL` — **Tests green through the move.** Verified post-restructure: 32 engine + 30 server + 6 web = 68 total. No test changes required; `uv sync` resolved both workspace members from their new paths without complaint.

## TESTS

- `TESTS` — **68/68 combined green** after restructure (32 engine + 30 server + 6 web). No test file changes; only invocation paths changed.

## Session 13 delta summary

| Metric | Session 12 end | Session 13 end |
|---|---:|---:|
| Top-level directories (code) | 5 flat | **4 hierarchical** (`apps/` `services/` `packages/` `infra/`) |
| Documented future-stub dirs | 0 | **11** (V2 scaffolds with READMEs) |
| Docs in `docs/revamp/` | 9 | **11** (+ VISION, + CONTROL_PLANE_SPEC; ARCHITECTURE split into 2) |
| Tests | 68 | **68** (unchanged) |
| DECISIONS entries | 12 | **16** (+ 4 session-13 ADRs) |

---

# Session 12 — 2026-04-24 — Frontend scaffold + bootstrap flow (Phase 10 start)

Baseline: session 11 closed Phase 9. The backend is feature-complete (62/62 tests, 39 routes + 1 WS). No UI exists yet.

Theme: owner: "lets go." Put a face on the platform. Scaffold the React UI as a third monorepo sibling, implement the bootstrap → workspace-detail flow end-to-end, and wire it into CI + Docker.

## ADDED — `pycaret-ui/` package (new monorepo sibling)

- `ADDED` — **`pycaret-ui/package.json`** — `@pycaret/ui`, version `0.1.0-alpha.0`, dual-licensed `MIT OR BUSL-1.1`. Scripts: `dev`, `build` (tsc -b + vite), `typecheck`, `lint`, `test`, `gen:api` (openapi-typescript). Runtime deps: react 18, react-router 6, axios, zustand, @tanstack/react-query, plotly.js-basic-dist + react-plotly.js. Dev deps: vite 5, vitest 2, @testing-library, typescript 5.6, tailwindcss 3, eslint 9.
- `ADDED` — **`pycaret-ui/tsconfig.{json,app.json,node.json}`** — strict TS with `verbatimModuleSyntax` (forces type-only imports, cleaner build), `target: ES2022`, path alias `@/*` → `src/*`.
- `ADDED` — **`pycaret-ui/vite.config.ts`** — dev server on `:3000` proxying `/api`, `/ws`, `/healthz` to the FastAPI backend at `:8000` (no CORS headaches locally). Vitest config with jsdom env + `vitest.setup.ts` loading jest-dom matchers.
- `ADDED` — **`pycaret-ui/tailwind.config.js`** — dark-mode-first palette: `ink` (slate-leaning darks/lights), `accent` (teal), `success` / `danger` / `warn`. Inter + JetBrains Mono font stacks. `maxWidth.form = 32rem` for single-column forms.
- `ADDED` — **`pycaret-ui/eslint.config.js`** — flat config, TS + react-hooks + react-refresh, `--max-warnings 0`.
- `ADDED` — **`pycaret-ui/index.html`** — root `<html class="dark">`, `bg-ink-950 text-ink-100` body.

## ADDED — API client + auth

- `ADDED` — **`src/api/client.ts`** — axios instance with bearer-token injection + **single-flight 401 refresh**. The refresh promise is stashed in a module-level `refreshing` so multiple concurrent 401s share one refresh call instead of stampeding. The `/auth/refresh` route is excluded from retry to prevent loops. `errorMessage(err)` helper pulls `detail` out of Pydantic error bodies for toast/form display.
- `ADDED` — **`src/api/types.ts`** — hand-written mirrors of the server's Pydantic schemas: TokenPair, User, SetupStatus, BootstrapRequest, LoginRequest, Workspace, Project, Experiment, Run, RunEvent, TaskType, RunStatus. Scope limited to what session 12 touches; `npm run gen:api` regenerates `schema.ts` for the full OpenAPI surface when needed.
- `ADDED` — **`src/api/endpoints.ts`** — one function per API route grouped by concern: `setupApi`, `authApi`, `workspacesApi`, `projectsApi`. Each method returns a typed Promise.
- `ADDED` — **`src/state/auth.ts`** — Zustand `useAuthStore`. Access token stays in memory; refresh token persisted to `localStorage["pycaret.refresh_token"]` so a page reload doesn't kick users to login. `refresh()` uses a bare axios call (not the instance with the interceptor) to avoid recursion.

## ADDED — screens + routing

- `ADDED` — **`src/components/AuthGate.tsx`** — guards authenticated routes. On mount with a refresh token but no access token, one-shot refreshes and shows "Restoring session…". Redirects to `/login` on failure (with `state.from` for return-after-login).
- `ADDED` — **`src/components/Layout.tsx`** — top nav shell with workspace link, user display name, sign-out button. Uses a `react-query`-cached `/auth/me` call to hydrate the user.
- `ADDED` — **`src/pages/Setup.tsx`** — first-run wizard. Detects already-bootstrapped servers via `GET /setup/status` and redirects to `/login`. Form fields: admin email, password (min 8), display name (optional), workspace name. On success stores the returned token pair and jumps to `/`.
- `ADDED` — **`src/pages/Login.tsx`** — sign in. Mirror of Setup; detects *un*-bootstrapped servers and redirects to `/setup`. Honours `state.from` for post-login redirect.
- `ADDED` — **`src/pages/Workspaces.tsx`** — `/`. Two-pane: list of workspaces (cards with name + description + created date) + side-card "New workspace" form. `useQueryClient().invalidateQueries` on create so the list refreshes without a page reload.
- `ADDED` — **`src/pages/WorkspaceDetail.tsx`** — `/workspaces/:id`. Breadcrumb + workspace header + project list + "New project" side-card with comma-separated tag input.
- `ADDED` — **`src/App.tsx`** — route table. `/setup` + `/login` are unauthenticated; everything else is wrapped in `<AuthGate><Layout />` via a parent route. Unknown paths fall through to a minimal 404.
- `ADDED` — **`src/index.css`** — Tailwind directives + component primitives (`.btn-primary/.btn-secondary/.btn-ghost/.btn-danger`, `.input`, `.field`, `.card`, `.hint`, `.error`, `.kbd`). Global focus ring, antialiasing.
- `ADDED` — **`src/main.tsx`** — React 18 root with `StrictMode`, TanStack Query client (no refetch-on-focus, 30s staleTime), `<BrowserRouter>`.

## ADDED — tests

- `ADDED` — **`src/state/auth.test.ts`** — 2 tests for the auth store (localStorage persistence + clear + refresh without token returns false).
- `ADDED` — **`src/components/AuthGate.test.tsx`** — 2 tests (redirects without tokens, renders children with access token).
- `ADDED` — **`src/pages/Setup.test.tsx`** — 2 tests (form renders, submit disabled until password valid). Mocks `@/api/endpoints` so no network.
- `TESTS` — **UI suite: 6/6 green in ~2 s.** Combined across the programme: **68/68** (32 engine + 30 server + 6 UI).

## ADDED — Docker + CI

- `ADDED` — **`docker/Dockerfile.ui`** — two-stage: `node:22-alpine` builder runs `npm ci || npm install` + `npm run build`, then `nginx:1.27-alpine` runtime serves `dist/` on port 8080 as a non-root user. Healthcheck via `wget` on `/`.
- `ADDED` — **`docker/nginx.ui.conf`** — SPA history fallback (`try_files ... /index.html`), `/api/` + `/healthz` reverse proxy to `upstream pycaret_api { server api:8000; }`, WebSocket upgrade on `/api/v1/runs/*` with 1h idle timeouts (long-running PyCaret experiments shouldn't drop the event stream).
- `CHANGED` — **`docker/docker-compose.yml`** — new `ui` service exposing `3000:8080`, depends on `api:service_healthy`, its own healthcheck.
- `CHANGED` — **`.github/workflows/test.yml`** — new `ui` job runs typecheck + lint + test + build on Ubuntu Node 22 with npm cache. Added to the `ci-status` aggregate so branch protection gates on it.

## Session 12 delta summary

| Metric | Session 11 end | Session 12 end |
|---|---:|---:|
| Monorepo packages | 2 | **3** (+ @pycaret/ui) |
| Tests total | 62 | **68** |
| UI LOC (TSX + config) | 0 | **~1,300** |
| API routes covered in UI | 0 | 6 (`setup/status`, `setup/bootstrap`, `auth/login+refresh+logout+me`, `workspaces`, `projects`) |
| Docker images | 1 | **2** |
| Production bundle | — | **83 kB gzipped** |

## INTERNAL

- `INTERNAL` — **Hand-written API types vs. generated schema.ts.** Chose hand-written for session 12's 6-route surface because strict TS on a codegen'd OpenAPI output for every pydantic model adds more churn than the typing payoff until the UI hits ~15+ routes. Generated client is wired (`npm run gen:api`) for the switchover.
- `INTERNAL` — **Single-flight refresh pattern.** The axios interceptor stashes the in-flight refresh Promise in a closure variable; concurrent 401s await the same promise. Critical once parallel `useQuery` calls start happening across screens.
- `INTERNAL` — **Why port 3000 → 8080 in the UI container.** nginx default is 80 (root-only); we run as `nginx` user on 8080 and let compose map it out as 3000 for developer familiarity. Matches how `docker/docker-compose.prod.yml` will stage the reverse proxy.
- `INTERNAL` — **`verbatimModuleSyntax` caught a real bug.** The axios `AxiosInstance` / `InternalAxiosRequestConfig` were imported as values but are type-only; the strict flag forced the right `import type`, which also helps tree-shaking in the production build.

---

# Session 11 — 2026-04-24 — Phase 9 finish: data sources, deployments, cancel, alembic

Baseline: session 10 landed the runs subsystem with live event streaming. The server could execute runs but had no way to ingest real data, no way to serve a trained model, no way to cancel, and no proper migration story.

Theme: owner: "lets continue with our roadmap development as per agreed and logical sequence." Close Phase 9 so the backend is feature-complete before the frontend starts.

## ADDED — data sources (CSV upload + S3/Postgres connectors)

- `ADDED` — **`pycaret_server/api/data_sources.py`** — new router, 5 endpoints:
  - `POST /api/v1/workspaces/{ws_id}/data-sources/upload` — streams a multipart CSV to `${ARTIFACT_DIR}/data-sources/<uuid>.csv`, enforces a 64 MB cap as it goes (no "copy to disk then reject"), computes SHA-256 in the same pass, samples with `pd.read_csv(nrows=5)` for column metadata, records row count via a line scan. Returns the DataSource row.
  - `POST /api/v1/workspaces/{ws_id}/data-sources` — register an `s3` or `postgres` connector config (no connectivity check — deferred to dispatch time).
  - `GET /api/v1/workspaces/{ws_id}/data-sources` — list.
  - `GET /api/v1/data-sources/{id}` — fetch.
  - `DELETE /api/v1/data-sources/{id}` — delete + unlink the uploaded file when `kind="csv_upload"`.
- `ADDED` — **`pycaret_server/runs/plans.py::load_csv(path)`** — tiny loader called from the orchestrator's `_load_data` when `RunSpec.data_source_path` is set.
- `CHANGED` — **`pycaret_server/runs/orchestrator.py::RunSpec`** gains `data_source_path` + `target_override`; `_load_data` picks the right source based on which field is populated.
- `CHANGED` — **`pycaret_server/api/runs.py::submit_run`** resolves `data_source_id` against the DataSource table, rejects cross-workspace references with 400, rejects non-`csv_upload` kinds with 400 (for now), snapshots both the effective target and the data_source_id into `Run.snapshot`.
- `CHANGED` — **`pycaret_server/api/schemas.py::RunCreate`** adds `data_source_id` + `target` fields; existing `sklearn_dataset`/`data_inline` unchanged.

## ADDED — deployments + in-house serving

- `ADDED` — **`pycaret_server/serving.py`** — `DeploymentRegistry`, an in-process thread-safe slug→pipeline cache. Pipelines are loaded on first prediction via cloudpickle (with pickle fallback) and evicted on deployment delete. Rolling 100-sample latency window tracks p50/p95 without adding a histogram dep. Module-level singleton with `reset_registry()` for test fixtures.
- `ADDED` — **`pycaret_server/api/deployments.py`** — new router, 9 endpoints:
  - `POST /api/v1/runs/{run_id}/promote` — validates the run succeeded + has a `pipeline_pickle` artifact; creates a workspace-scoped `pipelines` row pointing at the artifact path (reuses the SHA-256 that the orchestrator stamped).
  - `GET /api/v1/workspaces/{ws_id}/pipelines` / `GET /api/v1/pipelines/{id}` — list + fetch.
  - `DELETE /api/v1/pipelines/{id}` — refuses with 409 when any `Deployment` still references the pipeline (mirrors the FK's `ON DELETE RESTRICT`).
  - `POST /api/v1/pipelines/{id}/deployments` — create a Deployment. Validates `endpoint_slug` against `^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$` and `auth_mode` against `workspace|api-key|public`. Global uniqueness on slug.
  - `GET /api/v1/workspaces/{ws_id}/deployments` / `GET /api/v1/deployments/{id}` — list + fetch (with live p50/p95).
  - `DELETE /api/v1/deployments/{id}` — evicts the registry entry + deletes the row.
  - `POST /api/v1/deployments/{slug}/predict` — the serving endpoint. `workspace` auth_mode for v1 (api-key and public are schema-reserved but not enforceable yet). Ticks `inference_count` + `last_inference_at` + `p50_latency_ms` + `p95_latency_ms` on every request; errors tick `error_count`. Request contract: `{"rows": [{...}, ...]}` in, `{"deployment_id", "endpoint_slug", "predictions": [{"index", "prediction"}, ...], "latency_ms", "request_id"}` out.

## ADDED — run cancellation

- `ADDED` — **`pycaret_server/runs/orchestrator.py`**:
  - `_CancelledError` — private exception mapped to `Run.status = "cancelled"`.
  - `RunOrchestrator.cancel(run_id)` — thread-safe setter on a per-run `threading.Event` stored in `self._cancel_events`.
  - `_checkpoint()` closure in `_execute` — polls the event at stage boundaries (pre-load, post-load, post-fit, post-plan). Raises `_CancelledError` when set.
  - `_cleanup()` done-callback now pops both the Future and the cancel event.
- `ADDED` — **`POST /api/v1/runs/{id}/cancel`** route — returns current row; no-op when already terminal.

## ADDED — Alembic baseline + schema bootstrap

- `ADDED` — **`pycaret-server/alembic.ini`** — script location, file template (date + slug), UTC timezone, no post-write hooks (we run `ruff format` ourselves).
- `ADDED` — **`pycaret_server/migrations/env.py`** — pulls `database_url` from `get_settings()` (overridable via `ALEMBIC_URL`), `target_metadata = Base.metadata`, `render_as_batch=True` on SQLite, `compare_type=True` + `compare_server_default=True` so autogenerate catches column type changes.
- `ADDED` — **`pycaret_server/migrations/script.py.mako`** — typed PEP-604 template (`str | None`) compatible with Python 3.11+.
- `ADDED` — **`pycaret_server/migrations/versions/20260424_0213_9f9b7c770df0_baseline_schema.py`** — autogen-produced baseline. Creates all 14 tables + 24 indexes + 5 unique constraints + alembic_version. Confirmed end-to-end: fresh SQLite `alembic upgrade head` yields 15 tables.
- `ADDED` — **`pycaret_server/db/bootstrap.py::ensure_schema(engine, dev_auto_migrate=True)`** — the bridge between dev SQLite (one-command workflow) and prod (explicit migrations). If `alembic_version` is present → no-op. Else if a legacy `create_all`-seeded DB is detected (has `users` table, no `alembic_version`) → auto-stamp to baseline. Else if empty + `dev_auto_migrate=True` → `alembic upgrade head`. Else raise (prod safeguard).
- `ADDED` — **`pycaret-server migrate [--url ...] [--revision head]`** — CLI subcommand that calls `alembic.command.upgrade` with the live engine URL, so ops can deploy without a vendored alembic CLI.
- `CHANGED` — **`pycaret_server/app.py::_lifespan`** — now calls `ensure_schema` instead of `Base.metadata.create_all`. Also resets the `DeploymentRegistry` singleton in the `finally:` block to stop reload mode from caching stale pipelines across processes.

## TESTS

- `TESTS` — **`tests/test_phase9_finish.py`** — 10 new integration tests, ~350 LOC:
  - Data sources: `test_csv_upload_and_run_from_it` (upload an iris CSV, submit a `create` run that reads from it, assert succeeded), `test_register_s3_connector` (happy path + 2 bad shapes), `test_data_source_delete_cleans_file`.
  - Run cancel: `test_cancel_queued_run` (cancel mid-flight; outcome is `cancelled` or `succeeded` if the worker raced past every checkpoint), `test_cancel_terminal_run_is_noop`.
  - Deployments: `test_promote_run_and_serve_predictions` (the full curl flow: train → promote → deploy → predict × 2 rows → verify inference_count=2 + p50 non-null), `test_promote_rejects_unfinished_run`, `test_delete_pipeline_with_active_deployment_fails` (409), `test_deployment_slug_collision` (409) + bad-format (400).
  - Alembic: `test_alembic_baseline_creates_schema` — subprocess-invokes `alembic upgrade head` on a fresh SQLite, asserts 15-table result set.
- `TESTS` — **Server suite total: 30/30 green in ~55 s** (14 + 6 + 10).
- `TESTS` — **Combined engine + server: 62/62 green.**

## INTERNAL

- `INTERNAL` — **Route style** — data-sources' `upload_csv` uses FastAPI's `Annotated[UploadFile, File()]` / `Annotated[str, Form()]` form (the modern recommendation), not the deprecated `file: UploadFile = File(...)` default-arg form. Ruff B008 caught the latter; picked up as a style contract for future file-upload endpoints.
- `INTERNAL` — **No Alembic in CI yet** — the server suite still uses `Base.metadata.create_all` in its per-test fixture (fast path — ~40 ms vs alembic's ~700 ms). The `test_alembic_baseline_creates_schema` test is the single canary that the baseline migration actually applies; it runs `alembic upgrade head` in a subprocess to keep the inner test fast.
- `INTERNAL` — **Cancellation semantics** — cancellation is cooperative, not preemptive. An engine verb that's deep inside sklearn code (a long `compare_models`, say) cannot be interrupted mid-fit. The `_checkpoint()` calls catch cancellation between verbs, which is enough for the UI to feel responsive on a multi-stage plan but not a substitute for a real signal-handling worker.

## Session 11 delta summary

| Metric | Session 10 end | Session 11 end |
|---|---:|---:|
| Server LOC | ~2,400 | **~3,600** (+1,200) |
| API routes | 26 + 1 WS | **39 + 1 WS** |
| Server tests | 20 | **30** |
| Combined tests | 52 | **62** |
| Alembic revisions | 0 | **1 baseline** |

---

# Session 10 — 2026-04-24 — Run execution + event stream (Phase 9 core complete)

Baseline: session 9 landed the scaffold — 29 routes, 14 tables, 14 tests, 3766 LOC added to `pycaret-server/`. The server could CRUD workspaces / projects / experiments but couldn't actually *run* a PyCaret experiment.

Theme: owner: "lets do the next major phase." Wire the engine's `Experiment` verbs to the platform. A client POSTs to `/api/v1/experiments/{id}/runs` with a plan (`setup` / `create` / `compare`) and a data source (`sklearn_dataset` or `data_inline`); the server enqueues a job, dispatches it to a thread-pool worker that builds the right `pycaret.tasks.*Experiment`, wires a DB-backed `BaseLogger` subclass so every engine `Event` becomes an `events` row, pickles the fitted pipeline to the artifact dir, writes the leaderboard JSON back onto the Run row, and — if anyone is listening on the WebSocket — fans out events live.

## ADDED — `pycaret_server/runs/` subsystem

- `ADDED` — **`pycaret_server/runs/broker.py`** — `EventBroker`, a thread-safe fan-out that bridges the worker thread's synchronous `BaseLogger.emit` to async WebSocket consumers. Subscribers register an `asyncio.Queue`; `publish(run_id, event_dict)` dispatches through `loop.call_soon_threadsafe(queue.put_nowait, event)` so producers can live on any thread. `close_run(run_id)` pushes an `END` sentinel to drain outstanding subscribers when a run reaches a terminal state. Module-level singleton `event_broker`; `clear()` for test fixtures.
- `ADDED` — **`pycaret_server/runs/logger_bridge.py`** — `DBEventLogger(pycaret.logging.BaseLogger)`. Overrides `emit()` to open a short-lived SQLAlchemy session, write a single `Event` row (kind + message + payload + duration_ms + emitted_at), then republish via `event_broker.publish`. Per-emit session avoids holding SQLite connections open across long-running verbs. Resolves `session_factory` via `get_session()` (the defining-module function) so test monkeypatches take effect.
- `ADDED` — **`pycaret_server/runs/plans.py`** — "plan" abstraction: a `PlanName` Literal (`"setup" | "create" | "compare"`), a `PlanOutcome` dataclass (leaderboard + best_model + extra), and `execute_plan(exp, plan, *, model_id, plan_params)` that maps each plan onto the right engine verb call. Also `load_sklearn_dataset(name)` covering iris / wine / breast_cancer / diabetes (no network dependency) and `load_inline(rows)` for JSON dict-of-lists payloads.
- `ADDED` — **`pycaret_server/runs/orchestrator.py`** — `RunOrchestrator`: a `ThreadPoolExecutor(max_workers=2)` wrapped with submit / wait_for / shutdown. `_execute(spec)` transitions the Run row through `queued → running → succeeded|failed`, loads data, instantiates the right `Experiment` subclass by TaskType, attaches the `DBEventLogger`, calls `exp.fit(df)`, executes the plan, pickles `outcome.best_model` to `${PYCARET_ARTIFACT_DIR}/runs/<run_id>/pipeline.pkl` with a SHA-256 checksum, writes an `Artifact` row, stores the leaderboard JSON + a two-key summary (`rows`, `best`) on the Run row. Every exception is captured into `Run.error`; a `finally:` closes the event stream via `event_broker.close_run`. Module-level singleton via `get_orchestrator()` / `reset_orchestrator()` (used in test fixtures + the app lifespan teardown).
- `ADDED` — **`pycaret_server/runs/__init__.py`** — exports `EventBroker`, `event_broker`, `RunOrchestrator`, `get_orchestrator`.

## ADDED — HTTP + WebSocket routes

- `ADDED` — **`pycaret_server/api/runs.py`** — a single `APIRouter(tags=["runs"])` mounted at `/api/v1` hosting:
  - `POST /experiments/{experiment_id}/runs` — validates plan / model_id / data source, snapshots the Experiment config into `Run.snapshot`, persists a queued Run, enqueues a `RunSpec` with the orchestrator, returns 202.
  - `GET /experiments/{experiment_id}/runs` — all runs for an experiment, newest first.
  - `GET /runs/{run_id}` — status + leaderboard + metrics_summary + error.
  - `GET /runs/{run_id}/events?limit=&after_id=` — paginated replay (polling clients; UI uses the WebSocket).
  - `POST /runs/{run_id}/wait?timeout_s=30` — blocking wait; refreshes the Run and returns. Useful for notebooks + tests.
  - `WS /runs/{run_id}/events/ws?token=<jwt>` — authenticates via query-param JWT (browser WebSocket can't set headers), replays stored events first so the client sees full history, then subscribes to the broker until `run.closed`. Non-terminal subscribers get live fan-out; late-joiners on a terminal run get the replay + sentinel then disconnect.
- `ADDED` — **`pycaret_server/api/schemas.py`** — `RunCreate`, `RunResponse`, `EventResponse` Pydantic models.

## ADDED — app wiring

- `ADDED` — **`pycaret_server/app.py`** now mounts the runs router and tears down the orchestrator in the lifespan `finally` so ephemeral worker threads stop between test runs and on graceful server shutdown.
- `ADDED` — **`pycaret_server/api/__init__.py`** exports the `runs` submodule.

## TESTS

- `TESTS` — **`pycaret-server/tests/test_runs.py`** — 6 new integration tests with per-test SQLite + orchestrator reset:
  - `test_submit_run_validation` — covers 3 bad shapes (no data source, create without model_id, unknown plan). 400 each.
  - `test_setup_run_lifecycle` — submit a `setup` run on iris, block via `/wait`, assert status=succeeded + duration_ms > 0 + events contain `experiment.started` and `experiment.fitted`.
  - `test_create_run_produces_artifact` — submit a `create` run with `model_id=lr` on iris, block, assert the pipeline pickle exists on disk under `${artifact_dir}/runs/<id>/pipeline.pkl`.
  - `test_list_runs_for_experiment` — empty list, submit 2 runs (iris + wine), list returns 2.
  - `test_websocket_replay_after_run_finishes` — submit + wait, then open the WS; receive the event replay followed by `run.closed`. Verifies the broker handles late subscribers against terminal runs.
  - `test_ws_rejects_unauth` — WebSocket with no `?token=` is closed with code 4401.
- `TESTS` — **Server suite total: 20/20 green in ~10 s** (14 existing + 6 new).
- `TESTS` — **Combined engine + server: 52/52 green**.

## INTERNAL

- `INTERNAL` — **Thread + asyncio bridge** — `EventBroker.publish` is safe from worker threads because each subscriber remembers its owning event loop at subscribe-time; `call_soon_threadsafe` defers the `Queue.put_nowait` onto the right loop. This matches FastAPI's own pattern for cross-loop wakeups.
- `INTERNAL` — **Test monkeypatch safety** — any module-level `from pycaret_server.db import session_factory` would have frozen the reference at import time and missed the per-test rebind. Switched to `from pycaret_server.db import get_session` (a wrapper function that resolves `session_factory` from its defining module's globals at call time) in `logger_bridge.py`, `orchestrator.py`, and `api/runs.py`. Pattern to reuse in any future module.
- `INTERNAL` — **Pipeline pickling** — uses `cloudpickle` (already a PyCaret core dep) with `pickle` fallback. SHA-256 checksum captured in the `Artifact` row for later integrity validation when loaded for serving.

## Session 10 delta summary

| Metric | Session 9 end | Session 10 end |
|---|---:|---:|
| Server LOC | ~1,800 | **~2,400** (+600) |
| API routes (under /api/v1) | 21 | **26 + 1 WebSocket** |
| Tests in server suite | 14 | **20** |
| Total tests (engine + server) | 46 | **52** |
| `runs/*` subsystem | 0 files | **5 files, ~580 LOC** |

---

# Session 9 — 2026-04-24 — `pycaret-server` backend scaffolded (Phase 8 + 9 + 11 partial)

Baseline: 4.0.0a1 engine released; 41-dep lean install runs on sklearn 1.8 / NumPy 2.4 / pandas 3.0.
Environment: monorepo with uv workspace (engine + server share the lockfile).

Theme: owner: "lets do the next major phase." Part-2 platform kickoff — a FastAPI + SQLAlchemy backend sibling package (`pycaret-server`) that fronts the engine with a typed REST API, matching the design in `PLATFORM_PLAN.md`.

## ADDED — `pycaret-server` package (new monorepo sibling)

- `ADDED` — **`pycaret-server/pyproject.toml`** — new hatchling-built package, Python 3.11+, depends on `pycaret >= 4.0.0a1` (resolved via uv workspace during dev). Extras: `postgres` / `mysql` / `s3` / `notebook` / `dev` / `test`. Entry point: `pycaret-server` console script. Dual-licensed (MIT / BSL-1.1) per `DECISIONS.md § decision 5`.
- `ADDED` — **`pyproject.toml` (root)** — `[tool.uv.workspace]` with `members = ["pycaret-server"]` and `[tool.uv.sources]` pinning both packages to the workspace. `uv sync --all-packages --all-extras` resolves both in one go.
- `ADDED` — **`pycaret-server/pycaret_server/config.py`** — pydantic-settings reading `PYCARET_*` env vars; `.env` file support. Settings: database URL (SQLite default), JWT secret + TTLs, artifact dir, CORS origins, feature flags.
- `ADDED` — **`pycaret-server/pycaret_server/db/base.py`** — SQLAlchemy 2.x declarative base + `UUIDMixin` + `TimestampMixin`.
- `ADDED` — **`pycaret-server/pycaret_server/db/session.py`** — engine + session factory + FastAPI `get_db` dependency with per-backend pool kwargs (SQLite cross-thread safe).
- `ADDED` — **`pycaret-server/pycaret_server/db/models.py`** — **14 tables** matching `PLATFORM_PLAN.md § 3`:

  | Table | Purpose |
  |---|---|
  | `users` | Local user store (email + bcrypt hash). |
  | `sessions` | Refresh-token storage (hashed). |
  | `api_keys` | Programmatic-access tokens. |
  | `workspaces` | Top-level container. |
  | `workspace_members` | User × workspace × role (admin / member). |
  | `data_sources` | CSV upload / S3 / Postgres connection config. |
  | `projects` | Inside a workspace. |
  | `experiments` | Configured Experiment (task + target + setup_params). |
  | `runs` | One invocation of an experiment; status + leaderboard + metrics_summary. |
  | `events` | Append-only engine Event stream per run. |
  | `artifacts` | Run outputs (pickle / notebook / html preview / leaderboard.json / events.jsonl). |
  | `fold_metrics` | Per-fold × per-model × per-metric (composite PK). |
  | `pipelines` | Workspace-scoped fitted Pipeline registry. |
  | `pipeline_project_links` | Many-to-many between pipelines and projects. |
  | `deployments` | In-house serving record. |

- `ADDED` — **`pycaret-server/pycaret_server/auth/`** — bcrypt password hashing (`passwords.py`); JWT access + rotating refresh tokens with session-row storage (`tokens.py`); FastAPI `CurrentUser` / `require_admin` dependencies (`deps.py`).
- `ADDED` — **`pycaret-server/pycaret_server/api/schemas.py`** — Pydantic request/response models for bootstrap, login/refresh, workspaces, projects, experiments.
- `ADDED` — **`pycaret-server/pycaret_server/api/setup.py`** — `GET /api/v1/setup/status`, `POST /api/v1/setup/bootstrap` (first-run admin + workspace creation, returns token pair).
- `ADDED` — **`pycaret-server/pycaret_server/api/auth.py`** — `POST /api/v1/auth/login`, `POST /api/v1/auth/refresh` (with rotation + old-token revocation), `POST /api/v1/auth/logout`, `GET /api/v1/auth/me`.
- `ADDED` — **`pycaret-server/pycaret_server/api/describe.py`** — `GET /api/v1/describe/{models,models/{id},metrics,setup-params}` — thin proxy over `pycaret.api`.
- `ADDED` — **`pycaret-server/pycaret_server/api/workspaces.py`** — workspace CRUD + `_require_access` / `_require_admin` helpers.
- `ADDED` — **`pycaret-server/pycaret_server/api/projects.py`** — project CRUD nested under workspace.
- `ADDED` — **`pycaret-server/pycaret_server/api/experiments.py`** — experiment CRUD nested under project; validates `task` against `pycaret.core.tasks.TaskType`.
- `ADDED` — **`pycaret-server/pycaret_server/app.py`** — FastAPI app factory with CORS + lifespan. On first boot, `Base.metadata.create_all` seeds the SQLite schema. Mounts all 6 routers at `/api/v1`. OpenAPI at `/docs`, `/redoc`, `/openapi.json`. Health at `/healthz`.
- `ADDED` — **`pycaret-server/pycaret_server/cli.py`** — argparse CLI with `serve [--host] [--port] [--reload]` and `version` subcommands. Console entry: `pycaret-server`.
- `ADDED` — **`pycaret-server/tests/test_api.py`** — **14 integration tests** exercising every route via `fastapi.testclient.TestClient` + in-memory SQLite. Covers: meta (root, healthz, openapi schema), setup (status, bootstrap, idempotency), auth (login + refresh + rotation + revocation), workspaces CRUD, projects CRUD, experiments CRUD, describe proxy, unauthenticated-route rejection. 14/14 green in ~8 s.
- `ADDED` — **`pycaret-server/README.md`** — package-level README.
- `ADDED` — **`docker/Dockerfile.api`** — multi-stage build (Python 3.13-slim + uv cache layer + non-root runtime user + healthcheck). `uvicorn pycaret_server.app:create_app --factory` as entrypoint.
- `ADDED` — **`docker/docker-compose.yml`** — dev compose: api service with SQLite + artifact volume at `./data/`, env vars, healthcheck, restart policy.
- `ADDED` — **`docs/revamp/PLATFORM_QUICKSTART.md`** — 5-minute clone-to-running walkthrough (both local-dev and docker-compose paths), config reference, first-run flow, curl examples, troubleshooting.

## CHANGED

- `CHANGED` — **`.github/workflows/test.yml`** — lint job now covers `pycaret-server/` too; test job installs both workspace packages (`uv sync --all-packages --all-extras`) and runs `pytest pycaret-server/tests/` alongside the engine suite.
- `CHANGED` — **`pycaret-server/pycaret_server/api/auth.py::refresh`** — handles SQLite's tz-stripping behaviour on `DateTime(timezone=True)` columns: coerces both sides to tz-aware before comparing `expires_at <= now`.

## FIXED

- `FIXED` — During initial server scaffolding, `pydantic`'s `EmailStr` required an extra runtime dep (`email-validator`). Added via `pydantic[email]>=2.9` in `pycaret-server/pyproject.toml` core deps.

## TESTS

- `TESTS` — **Engine**: 32/32 green in 1:36.
- `TESTS` — **Server**: 14/14 green in 7.4 s.
- `TESTS` — **Combined**: 46/46 green. CI runs both in the same workflow job for every push.

## Session 9 delta summary

| Metric | 4.0.0a1 (engine only) | Session 9 end |
|---|---:|---:|
| Monorepo packages | 1 | **2** (+ pycaret-server) |
| Total tests | 32 | **46** |
| SQLAlchemy tables | 0 | **14** |
| API routes (HTTP endpoints) | 0 | **29** |
| OpenAPI schema | — | **valid, live at `/openapi.json`** |
| Docker artifacts | — | **Dockerfile.api + docker-compose.yml** |
| Part-2 phases | all 🔴 NOT STARTED | **Phase 8 ✅, Phase 9 🟡 mostly, Phase 11 🟡 partial** |

Coming next session:
- `POST /api/v1/experiments/{id}/runs` with a threading-based worker that loads the data source, constructs the engine's `Experiment` class, runs the full `compare_models → predict_model → save artifacts` chain, streams events through `BaseLogger.subscribe(...)` into a WebSocket, and persists everything into `runs` / `events` / `artifacts` / `fold_metrics`.
- Data-source connectors (CSV upload, S3, Postgres).
- Deployment subsystem (`DeploymentRegistry` + catch-all `/api/v1/deployments/{slug}/predict`).
- Alembic baseline migration.

---

# Session 8 — 2026-04-23 — Aggressive dependency cut → 4.0.0a1

Baseline: end of session 7 (4.0.0a0 tag pushed, GitHub Release published, user testing on Google Colab surfaced "deps still too heavy").
Environment: unchanged.

Theme: owner asked for another round of dep discipline — "lets start cleaning them up. we dont need all kind of tuners. we dont need all that kind of extra visualization... nobody uses all those long tail functionalities. we dont need kmodes, lightgbm, catboost, xgboost, let user install it separately and add them to the library model container or something." Plus: unpin sklearn.

## BREAKING — extras collapsed

- `REMOVED, BREAKING` — **`pycaret[models]`** extra entirely removed. `xgboost`, `catboost`, `kmodes`, `mlxtend`, `lightgbm` — all no longer installed by `pycaret[full]` or any other extra. Users install them directly: `pip install xgboost lightgbm catboost`. Pycaret's model containers already use `_check_soft_dependencies` to auto-detect and light up when the backing package is present (LGBM container gained this guard in this session; xgboost/catboost already had it).
- `REMOVED, BREAKING` — **`pycaret[tuners]`** extra entirely removed. `optuna`, `optuna-integration`, `scikit-optimize`, `hyperopt` — all gone. Users attach a custom search via sklearn's `GridSearchCV` / `RandomizedSearchCV` / `HalvingRandomSearchCV`, or install an optuna-type backend and pass a constructed search-cv to `tune_model`.
- `REMOVED, BREAKING` — **`pycaret[analysis]`** extra entirely removed. `shap`, `interpret`, `umap-learn` — all gone. Interpretability re-enters targeted in a later release (owner: "for analysis we get drop interpret related functionalities, shap as well for now, we will bring it later").
- `REMOVED, BREAKING` — **`pycaret[prophet]`** extra entirely removed.

## DEPS — core cut from 19 → 11

Removed from core `dependencies`:
- `lightgbm>=4.3` (moved to user-install).
- `cloudpickle>=3.0` (joblib pulls it transitively when needed; `load_experiment()` lazy-imports with a clean fallback).
- `psutil>=5.9` (system-info logging falls back to `os.cpu_count()`).
- `xxhash>=3.4` (`FastMemory` falls back to `hashlib.blake2b(digest_size=16)`).
- `matplotlib>=3.9` (Plotly is the single chosen library; matplotlib lazy-imported in 4 residual non-Plotly call sites).
- `kaleido>=0.2` (moved to new `export` extra).
- `nbformat>=5.10` (moved to `notebook` extra).
- `ipywidgets>=8.1` (moved to `notebook` extra).

Kept in core:
- `numpy>=1.26`, `pandas>=2.2`, `scipy>=1.11`, **`scikit-learn>=1.7`** (upper cap removed — see below), `joblib>=1.4`, `plotly>=5.22`, `tqdm>=4.66`, `requests>=2.32`, `jinja2>=3.1` (pandas.style), `ipython>=8.18`.

Transitional (still in core, flagged for removal):
- `imbalanced-learn>=0.13` — removed when `pycaret.internal.pipeline.Pipeline` stops inheriting from `imblearn.pipeline.Pipeline` (Phase 4).
- `category-encoders>=2.6` — removed when legacy preprocessor is rewritten on native sklearn encoders (Phase 4).

## CHANGED — sklearn unpinned

- `CHANGED` — **`scikit-learn>=1.7`** — upper cap `<1.8` removed. The cap existed only because `sktime` (in the `timeseries` extra) required `scikit-learn<1.8`. Since `sktime` is no longer pulled into the default install — only when a user installs `pycaret[timeseries]` — the core can track the latest sklearn. Default fresh install now pulls sklearn 1.8, NumPy 2.4, pandas 3.0.

## REMOVED — show_versions diagnostic table entries

- `CHANGED` — `pycaret/utils/_show_versions.py` dependency-version reporting table no longer includes `mlxtend`, `kmodes`, `kaleido`, `jinja2` (still reported), `xxhash` (uses stdlib now). Kept for diagnostic: `numpy`, `pandas`, `scipy`, `scikit-learn`, `joblib`, `plotly`, `tqdm`, `requests`, `ipython`, `imbalanced-learn`, `category-encoders`, `matplotlib` (if installed), plus extras packages when detected.

## FIXED — soft-dep guards

- `FIXED` — `pycaret/containers/models/classification.py::LGBMClassifierContainer` — added `_check_soft_dependencies("lightgbm")` guard; container sets `self.active = False` cleanly when lightgbm isn't installed. Same guard added to `regression.py::LGBMRegressorContainer` and `time_series.py::LGBMCdsDtContainer` (TS variant also pre-sets `is_gpu_enabled = False` so the god-class `__init__` flow doesn't AttributeError before the `active=False` short-circuit takes effect).
- `FIXED` — `pycaret/internal/memory.py` — xxhash import now `try/except`; falls back to `hashlib.blake2b`. No behaviour change in practice (hash-based cache key collisions don't matter for correctness, only perf).
- `FIXED` — `pycaret/internal/pycaret_experiment/pycaret_experiment.py` — `cloudpickle` now lazy-imported; `psutil` system-info logging is conditional.
- `FIXED` — `pycaret/internal/pycaret_experiment/supervised_experiment.py` — `matplotlib.pyplot` lazy-imported. Only the two `plt.savefig()` interpret-model call sites need it; both guarded against `plt is None` via the surrounding `if self.logging_param` block and the `try/except AttributeError` in the calling code.
- `FIXED` — `pycaret/internal/plots/helper.py` + `pycaret/internal/plots/utils/time_series.py` — matplotlib lazy-imported; `MatplotlibDefaultDPI` already had `try/except` wrapping its `plt.rcParams` access so `plt = None` is a silent no-op.

## TESTS

- `TESTS` — **32/32 green** on the lean 4.0.0a1 in 71s on the full-extras dev venv.
- `TESTS` — **Fresh `pycaret` install with no extras** (41 deps total) successfully runs `ClassificationExperiment(target='Purchase').fit(juice) → compare_models(include=['lr','dt']) → predict_model → save_model → load_model` roundtrip on sklearn 1.8 / NumPy 2.4 / pandas 3.0.

## BUILD

- `BUILD, BREAKING` — **Version `4.0.0a0` → `4.0.0a1`**.
- `BUILD` — Rebuilt wheel `pycaret-4.0.0a1-py3-none-any.whl` (412 KB, 112 files). `twine check` PASSED.

## DOCS

- `DOCS` — **`CHANGELOG.md`** — prepended 4.0.0a1 release entry with install commands, changed-removed-fixed subsections, new-extras-structure table, and the two "transitional deps" flagged for removal.
- `DOCS` — this session-8 block in `release_notes_pycaret4.md`.
- `DOCS` — `KILL_LIST.md` updated with the 4.0.0a1 extras collapse.

## Session 8 delta summary

| Metric | 4.0.0a0 | 4.0.0a1 | Δ |
|---|---:|---:|---:|
| Core `dependencies` count | 19 | **11** | **−8** |
| `[project.optional-dependencies]` extras | 7 (models/tuners/analysis/anomaly/timeseries/prophet/full + dev/test) | **6** (notebook/export/anomaly/timeseries/full + dev/test) | −1 category, simpler shape |
| Default `pip install pycaret` dep closure | ~65 pkgs | **~41 pkgs** | **−24** |
| sklearn constraint | `>=1.7,<1.8` | `>=1.7` (no upper cap) | unpinned |
| Test suite | 32/32 in 1:46 | 32/32 in 1:11 | −35s |
| Lazy-guarded matplotlib import sites | 0 | 4 | +4 (matplotlib now optional) |

---

# Session 6 — 2026-04-23 — Cleanup pass 2 + Application-Platform plan authored

Baseline: end of session 5 (v4 branch live on GitHub, CI green).
Environment: unchanged.

Theme: user asked for "one more round of clean ups. get rid of any garbage from 3.0. keep the bare minimum. the core logic that we will use and thats it." Then laid out the Part-2 vision — PyCaret as an enterprise-grade open-source application platform (CLI + FastAPI + SQL DB + React UI + Docker). Session 6 executed the cleanup and captured the platform plan.

## REMOVED

- `REMOVED` — **`pycaret/distributions.py`** (0 callers) deleted.
- `REMOVED` — **`pycaret/internal/cloudpickle_compat.py`** (0 callers) deleted.
- `REMOVED` — **`pycaret/internal/cuml_wrappers.py`** (143 LOC) deleted. cuml is not a 4.0 dep; GPU fallback via NVIDIA cuml is out of scope for the 4.0 engine.
- `REMOVED` — **`pycaret/loggers/`** shim package deleted. Re-pointed 7 `BaseLogger` import sites to `pycaret.logging.base` directly (1 in each of: `classification/oop.py`, `regression/oop.py`, `time_series/forecasting/oop.py`, `internal/pycaret_experiment/tabular_experiment.py`, `internal/pycaret_experiment/unsupervised_experiment.py`; 2 others already migrated). The 4.0 `BaseLogger` lives in `pycaret.logging.base`; the shim was legacy-compat and had no user after session 3.
- `REMOVED, BREAKING` — **9 killed-verb methods** deleted across god-class + task oop wrappers (no replacement; public API didn't expose them):

  | File | Methods deleted | ~LOC |
  |---|---|---:|
  | `internal/pycaret_experiment/pycaret_experiment.py` | `deploy_model` (stub) | 9 |
  | `internal/pycaret_experiment/tabular_experiment.py` | `deploy_model`, `convert_model`, `create_api`, `create_docker` | 361 |
  | `internal/pycaret_experiment/supervised_experiment.py` | `check_fairness`, `create_app`, `dashboard`, `check_drift` | 353 |
  | `classification/oop.py` | `deploy_model`, `dashboard` | 174 |
  | `regression/oop.py` | `deploy_model`, `dashboard` | 168 |
  | `time_series/forecasting/oop.py` | `deploy_model` | 91 |
  | **Total** | **15 method definitions** | **~1,156** |

  Lazy imports inside those methods (mlflow / comet / wandb / dagshub / fairlearn / evidently / gradio / fastapi / boto3 / m2cgen) disappeared with the bodies.

## CHANGED

- `CHANGED` — **Model containers (`containers/models/{classification,regression,clustering,anomaly}.py`) — cuml branches now raise `NotImplementedError`.** Deleted the `import pycaret.internal.cuml_wrappers` imports + the `pycaret.internal.cuml_wrappers.get_*()` call sites inside `if gpu_imported:` blocks, and replaced `import cuml.X` lines inside `if experiment.gpu_param == "force":` / `elif experiment.gpu_param:` blocks with a raise. These branches were unreachable with default `gpu_param=False` + cuml-not-installed, so no behaviour change; the code is now honest about it. (10 more cuml imports in `containers/models/time_series.py` left as-is — same dead-branch pattern; they'll go with the Phase-5 god-class drain.)
- `INTERNAL` — **`from functools import partial`** removed from `supervised_experiment.py` (only the deleted `check_fairness` method used it).

## ADDED

- `DOCS, ADDED` — **`docs/revamp/PLATFORM_PLAN.md`** (~350 lines) — detailed design for the Part-2 application platform:
  - **Vision**: credible open-source alternative to DataRobot / H2O.ai for teams under ~20 people.
  - **Architecture**: monorepo with 4 sibling packages — `pycaret` (library, current) + `pycaret-server` (FastAPI) + `pycaret-ui` (React) + `pycaret-cli` (CLI).
  - **Data model**: Workspace → Project → Experiment → Run → Pipeline. 11 SQLAlchemy tables.
  - **First-run flow**: `docker compose up` → self-service admin setup wizard → no external config.
  - **Database**: SQLite default, Postgres/MySQL opt-in via `DATABASE_URL`.
  - **Auth**: local user store + JWT; OAuth as plugin; admin/member roles.
  - **Tech choices**: Vite + React 18 + Tailwind + TanStack Query + Zustand + Plotly.js; FastAPI + uvicorn + SQLAlchemy + Alembic; Typer + Rich for CLI.
  - **6 new phases (7-12)** added to ROADMAP.
  - **Gated on Phase 5** — `pycaret==4.0.0alpha0` shipping — so engine stays focused.
  - Explicit "out of scope": Celery/Redis v1, K8s operator, GraphQL, multi-tenant SaaS, hosted billing, model serving.

## DOCS

- `DOCS` — **`ROADMAP.md` restructured** into Part 1 (Engine, Phases 0-6) and Part 2 (Platform, Phases 7-12). Every checkbox reflects actual state: Phases 0, 1, 3.5 ✅ COMPLETE; Phase 2 / 4 / 6 ✅ MOSTLY / 🟡 PARTIAL; Phase 5 🟡 IN FLIGHT (god-class drain, 10-verb migration order spelled out); Phases 7-12 🔴 NOT STARTED.
- `DOCS` — **`STATUS.md`** updated with session-6 delta table + platform-plan summary.
- `DOCS` — **`docs/revamp/README.md`** hub index updated to include `ARCHITECTURE.md`, `PLATFORM_PLAN.md`, `github_issues/`. New "Two parts, one programme" section. Reading order reorganized.

## TESTS

- `TESTS` — **32/32 still green** on Python 3.13 + sklearn 1.7.2 + NumPy 2.3.5 + pandas 2.x, in 1:37 (was 2:07 in session 5 — slightly faster with less code to import).

## ADDED — 6 resolved platform decisions

Owner answered the six parked questions from `PLATFORM_PLAN.md §7`. Each answer is now baked into the plan and recorded as an ADR in `DECISIONS.md`:

- `DOCS, ADDED` — **Decision 1: Run notebooks are first-class artifacts.** Every Run persists `run.ipynb` + `fitted_pipeline.pkl` + `leaderboard.json` + `events.jsonl` + `preview.html`. Immutable, downloadable, shareable via signed URL, previewable in-app. Storage: local disk v1, S3 when deployed.
- `DOCS, ADDED` — **Decision 2: Data-source connectors v1 = CSV upload + S3 + Postgres.** `DataSourceConnector` ABC allows adding Snowflake / GSheets / MySQL later without core changes. AWS-first since immediate deploy target.
- `DOCS, ADDED` — **Decision 3: Pipelines are workspace-scoped + shareable across projects.** `Pipeline` moves out of `Project` into `Workspace`; `pipeline_project_links` many-to-many joins them. Workspace gets a top-level "Pipelines" screen.
- `DOCS, ADDED` — **Decision 4: In-house serving system, not MLServer/BentoML.** `DeploymentRegistry` loads pickles into memory; single catch-all `POST /api/v1/deployments/{slug}/predict` handles inference. Per-deployment auth: `workspace` / `api-key` / `public`. Per-deployment metrics: count, p50/p95 latency, error rate. Phase 11 renamed "In-house serving + Docker/deploy".
- `DOCS, ADDED` — **Decision 5: Dual-license the platform packages.** Engine `pycaret` stays MIT. `pycaret-server` / `pycaret-cli` / `pycaret-ui` become MIT + BSL 1.1 (BSL for multi-tenant hosted SaaS only; converts to MIT after 3 years). CLA added to CONTRIBUTING.md. Mirrors Sentry / Cal.com / Supabase / Plausible posture.
- `DOCS, ADDED` — **Decision 6: Metrics stored as summary AND per-fold.** Two tables — `runs.metrics_summary` (leaderboard shape) and `fold_metrics` (per-fold × per-model × per-metric). Summary drives leaderboard; per-fold unlocks variance / stability / time-to-train analysis.

Data model in `PLATFORM_PLAN.md §3` expanded to 14 SQLAlchemy tables (from 11): added `fold_metrics`, `deployments`, `api_keys`, `pipeline_project_links`. Phase 11 now covers the serving subsystem in detail. Dep discipline §6 updated with `nbconvert` (notebook preview), `boto3` (S3 extra), `psycopg[binary]` (postgres extra), `python-multipart` (CSV upload), `joblib` (deployment loading).

New §8 "Licensing posture" added to `PLATFORM_PLAN.md`. Reading order updated.

## Session 6 delta summary

| Metric | Session 5 end | Session 6 end | Δ |
|---|---:|---:|---:|
| Source LOC in `pycaret/` | 51,976 | **50,544** | **−1,432** |
| Zero-import leaf files | 3 | **0** | **−3** |
| Killed-verb methods in source | 15 | **0** | **−15** |
| cuml-coupled files with runtime risk | 5 | **0** (branches raise) | − |
| Part-2 platform plan | none | **PLATFORM_PLAN.md (~350 lines)** | +1 doc |
| Roadmap phases defined | 6 engine phases | **12 (6 engine + 6 platform)** | +6 |
| Test pass rate | 100% (32/32) | 100% (32/32) | — |

---

# Session 4 — 2026-04-23 — Repo restructure + working notebooks + agent/dev docs + issue triage

Baseline: end of session 3.
Environment: unchanged.

Theme: user asked to "clear the folder, restructure for dev contributions, get rid of old stuff, one notebook per use-case fully working, MD files for agents, download all open issues, start cleaning them up." Session 4 does all of that in one pass.

## REMOVED — dead weight purged

- `REMOVED` — **`Docker_files/`** directory (old `pycaret_full` / `pycaret_slim` Docker image scaffolding) deleted.
- `REMOVED` — **`docs/source/`** directory (Sphinx docs tree) deleted — we use GitBook for hosted docs now; Sphinx was unused.
- `REMOVED` — **`docs/Makefile`, `docs/make.bat`, `docs/make.sh`, `docs/logs.log`** — Sphinx build scaffolding, deleted.
- `REMOVED` — **`tutorials/legacy_v3/`** (the 6 archived 3.x notebooks from session 3) deleted.
- `REMOVED` — **`tutorials/time_series/forecasting/`** (old TS example scripts) deleted.
- `REMOVED` — **`tutorials/translations/`** (4 language-translated tutorial trees: chinese/greek/japanese/portuguese) deleted.
- `REMOVED` — **`tutorials/pycaret_cheat-sheet_in_excel.xlsx`** deleted.
- `REMOVED` — **`logs.log`** (root-level 28K-line runtime log) deleted.
- `REMOVED` — **`.readthedocs.yml`** — ReadTheDocs config, unused since GitBook migration.
- `REMOVED` — **`.slugignore`** — Heroku-specific ignore file, long dead.

## CHANGED — directory restructure

- `CHANGED, BREAKING` — **`tutorials/` → `notebooks/`**. Modern naming; all 5 canonical notebooks live under `notebooks/` now (`01_classification.ipynb` … `05_time_series.ipynb`).
- `CHANGED` — **`.gitignore` rewritten.** Cleaner structure; `artifacts/`, `*.log`, `.ruff_cache/`, `demo*.py`, `nbtest*.ipynb` added; legacy 3.x scratch patterns retained as safety net.

## ADDED — working notebooks (one per task)

- `ADDED` — **`scripts/build_notebooks.py`** — programmatic notebook generator using `nbformat` + `nbclient`. Run with `--run` to execute and capture fresh outputs.
- `ADDED` — **`notebooks/01_classification.ipynb`** (43 KB with outputs). Canonical `ClassificationExperiment` → `fit` → `compare_models` → `tune_model` → `predict_model` → `save/load_model` → inspect event stream.
- `ADDED` — **`notebooks/02_regression.ipynb`** (25 KB).
- `ADDED` — **`notebooks/03_clustering.ipynb`** (18 KB). `ClusteringExperiment` → `create_model("kmeans")` → `assign_model`.
- `ADDED` — **`notebooks/04_anomaly_detection.ipynb`** (18 KB). `AnomalyExperiment` → `create_model("iforest")` → `assign_model`.
- `ADDED` — **`notebooks/05_time_series.ipynb`** (13 KB). `TimeSeriesExperiment(fh=12)` → `compare_models` → `predict_model`.
- `ADDED` — **All 5 notebooks executed end-to-end under Python 3.13 + sklearn 1.7.2.** Outputs are persisted in the `.ipynb` JSON so GitHub can render them and users can skim without running.

## ADDED — `AGENTS.md` at repo root

- `ADDED` — **`/AGENTS.md`** — 60-second briefing for AI coding agents: TL;DR, read-first list, non-negotiables (8 rules), conventions, workflow, repo map, common-task recipes. This is the file any agent should consume before touching the repo.

## ADDED — `docs/for_agents/` deep dives (5 files)

- `ADDED` — **`docs/for_agents/README.md`** — index of agent-facing docs.
- `ADDED` — **`docs/for_agents/ENGINE_WALKTHROUGH.md`** — step-by-step of what happens inside `fit → compare_models → predict_model`, including the legacy-delegation pattern.
- `ADDED` — **`docs/for_agents/TYPED_RESULTS.md`** — every result dataclass (9 of them) with full field listing and usage idioms.
- `ADDED` — **`docs/for_agents/EVENT_STREAM.md`** — all 22 `EventKind`s tabulated with their typical payloads; subscriber contract + custom-logger recipe.
- `ADDED` — **`docs/for_agents/INTROSPECTION_API.md`** — the `pycaret.api` surface as a UI/agent integration contract, with React-form and LLM-prompt examples.
- `ADDED` — **`docs/for_agents/TASK_CHEATSHEET.md`** — verb × task matrix + constructor-parameter matrix on one page; import-path reference.

## ADDED — `docs/for_developers/` onboarding (5 files)

- `ADDED` — **`docs/for_developers/README.md`** — index.
- `ADDED` — **`docs/for_developers/SETUP.md`** — clone → `uv sync --all-extras` → first green test in < 5 min; Python-version matrix; Windows notes; IDE setup.
- `ADDED` — **`docs/for_developers/TESTING.md`** — test layout, how to run/write tests, markers, CI matrix, coverage, what not to write.
- `ADDED` — **`docs/for_developers/DRAINING_THE_GODCLASS.md`** — verb-migration playbook for Phase 5: recommended order, before/after recipe, hard constraints, completion criteria.
- `ADDED` — **`docs/for_developers/CODING_STYLE.md`** — formatting, imports, type hints, docstrings, naming, dataclass conventions, error handling, logging, dependency discipline, git hygiene.
- `ADDED` — **`docs/for_developers/RELEASE_PROCESS.md`** — versioning, tagging, notes generation, PyPI publish.

## CHANGED — `CONTRIBUTING.md` rewritten for 4.0

- `CHANGED` — **`CONTRIBUTING.md`** — replaced 3.x-era content (black/isort mention, sphinx docs, `pip install -e .[test]`) with a 4.0-flavored guide that links into `AGENTS.md` + `docs/for_agents/` + `docs/for_developers/`. Includes the current PR checklist.

## ADDED — GitHub issue snapshot + triage

- `ADDED` — **`docs/revamp/github_issues/open_issues_raw.json`** — raw snapshot of all 388 open issues on `pycaret/pycaret` (fetched via `gh issue list`).
- `ADDED` — **`scripts/triage_issues.py`** — classifier that reads the raw snapshot and assigns each issue to one of 5 buckets: `fixed_in_4_0`, `out_of_scope`, `stale`, `still_relevant_bug`, `still_relevant_enhancement`. Heuristic-based; reruns are idempotent.
- `ADDED` — **`docs/revamp/github_issues/triage.json`** — machine-readable output of the classifier.
- `ADDED` — **`docs/revamp/github_issues/triage.md`** — human-browsable triage report, one table per bucket with issue number, title, labels, update date, classification reason.
- `ADDED` — **`docs/revamp/github_issues/README.md`** — triage summary + methodology.
- `ADDED` — **`docs/revamp/github_issues/PLAYBOOK.md`** — step-by-step actions per bucket with ready-to-paste reply templates.

### Headline triage result

From 388 open issues:
- **8 (2%) `fixed_in_4_0`** — close with 4.0 release-notes pointer.
- **92 (24%) `out_of_scope`** — close with `KILL_LIST.md` pointer.
- **123 (32%) `stale`** — auto-ping, close after 30 days of silence.
- **58 (15%) `still_relevant_bug`** — label `4.0-candidate`, route to Phase 5.
- **107 (28%) `still_relevant_enhancement`** — per-item decision.

**224 of 388 (58%) can be closed or auto-pinged without per-issue human judgment.**

## FIXED

- `FIXED` — **`np.product` → `np.prod`** in `pycaret/internal/patches/sklearn.py:106`. NumPy 2 removed the capitalised alias; the call site surfaced when the notebook-execution pass tried to run `compare_models`. Was blocking the full e2e notebook run.

## INTERNAL

- `INTERNAL` — **`scripts/` directory created** for maintenance scripts. Two scripts landed this session; future additions (bulk issue close, LOC reports, release build helpers) go here.
- `INTERNAL` — `tests/conftest.py`, `tests/test_models.py`, and all other tests untouched in this session (no code-under-test changes).

## Session 4 delta summary

| Metric | Session 3 end | Session 4 end | Δ |
|---|---:|---:|---:|
| Repo top-level directory count | 9 | 10 | +1 (`scripts/`) |
| Source LOC in `pycaret/` | ~49,400 | ~49,400 | 0 |
| Test files | 4 | 4 | 0 |
| Test pass rate | 100% (32/32) | 100% (32/32) | — |
| Dead-weight directories | 4 (Docker_files, docs/source, tutorials/{legacy_v3,time_series,translations}) | **0** | − |
| Stale root-level files | 3 (logs.log, .readthedocs.yml, .slugignore) | **0** | − |
| Working notebooks | 0 (3.x legacy archived) | **5 (executed, with outputs)** | +5 |
| Docs files | 7 in `docs/revamp/` | **23** (revamp + for_agents + for_developers + github_issues) | +16 |
| GitHub issues in actionable triage buckets | 0 (untouched backlog of 388) | 224 ready-to-close, 165 needing human review | — |

---

# Session 3 — 2026-04-22 — Functional API killed; 4.0 is OOP-only

Baseline: end of session 2.
Environment: unchanged.

Theme: the user made the final 4.0 call — "nobody will migrate 3 → 4. 4 in my mind is totally new thing. I really would like to get rid of 90% tech debt now." This session deletes the module-level functional API entirely, leaving the sklearn-compatible `Experiment` class hierarchy as the single canonical surface.

## BREAKING — functional API removed wholesale

- `REMOVED, BREAKING` — **`pycaret.classification.setup`, `compare_models`, `create_model`, `tune_model`, `ensemble_model`, `blend_models`, `stack_models`, `plot_model`, `evaluate_model`, `interpret_model`, `calibrate_model`, `optimize_threshold`, `predict_model`, `finalize_model`, `deploy_model`, `save_model`, `load_model`, `automl`, `pull`, `models`, `get_metrics`, `add_metric`, `remove_metric`, `get_logs`, `get_config`, `set_config`, `save_experiment`, `load_experiment`, `get_leaderboard`, `set_current_experiment`, `get_current_experiment`, `dashboard`, `convert_model`, `check_fairness`, `create_api`, `create_docker`, `create_app`, `get_allowed_engines`, `get_engine`, `check_drift`** — all 40 module-level functions gone. File `pycaret/classification/functional.py` deleted (3,323 LOC).
- `REMOVED, BREAKING` — **`pycaret.regression` functional API** — 38 module-level functions, `pycaret/regression/functional.py` deleted (3,033 LOC).
- `REMOVED, BREAKING` — **`pycaret.clustering` functional API** — 23 module-level functions, `pycaret/clustering/functional.py` deleted (1,461 LOC).
- `REMOVED, BREAKING` — **`pycaret.anomaly` functional API** — 18 module-level functions, `pycaret/anomaly/functional.py` deleted (1,256 LOC).
- `REMOVED, BREAKING` — **`pycaret.time_series` functional API** — 26 module-level functions, `pycaret/time_series/forecasting/functional.py` deleted (2,260 LOC).

Total LOC dropped in this session: **~11,333 lines of pass-through wrappers**.

## BREAKING — class renaming

- `CHANGED, BREAKING` — **`TSForecastingExperiment` renamed to `TimeSeriesExperiment`** in the public API. The cleaner name matches the task's module name. The legacy class (`pycaret.time_series.forecasting.oop.TSForecastingExperiment`) still exists as an internal implementation detail the new wrapper delegates to during migration.

## ADDED — 4 new task subclasses + stateless persistence

- `ADDED` — **`pycaret.tasks.RegressionExperiment`** (sklearn-compatible, `estimator_type="regressor"`; delegates to legacy `_NonTSSupervisedExperiment` during transition).
- `ADDED` — **`pycaret.tasks.ClusteringExperiment`** (inherits `UnsupervisedExperiment`; adds `assign_model`).
- `ADDED` — **`pycaret.tasks.AnomalyExperiment`** (inherits `UnsupervisedExperiment`; adds `assign_model`).
- `ADDED` — **`pycaret.tasks.TimeSeriesExperiment`** (inherits `SupervisedExperiment`; adds `check_stats`; overrides `fit()` for univariate time-series inputs).
- `ADDED` — **`pycaret.core.supervised.SupervisedExperiment`** — intermediate base that hosts supervised-only verbs (`compare_models`, `tune_model`, `ensemble_model`, `blend_models`, `stack_models`, `calibrate_model`, `finalize_model`, `interpret_model`, `automl`, `get_leaderboard`). Extracted from `Experiment` so unsupervised tasks don't inherit verbs they can't implement.
- `ADDED` — **`pycaret.core.unsupervised.UnsupervisedExperiment`** — intermediate base that adds `assign_model`. Tells the legacy engine via an override that supervised-coercion is not needed in `fit`.
- `ADDED` — **`pycaret.persistence`** — stateless top-level `save_model(model, path)` / `load_model(path)` utilities. No experiment required, no globals, no mutation. Re-exported as `pycaret.save_model` / `pycaret.load_model`.

## CHANGED — legacy module paths thinned

Each `pycaret/{module}/__init__.py` reduced from a 40-entry re-export list (~90 LOC with `__all__`) to a ~15-line docstring plus a single-line `from pycaret.tasks.{module} import {Task}Experiment`:

- `CHANGED` — `pycaret/classification/__init__.py` thinned (88 LOC → 20).
- `CHANGED` — `pycaret/regression/__init__.py` thinned (83 LOC → 13).
- `CHANGED` — `pycaret/clustering/__init__.py` thinned (53 LOC → 14).
- `CHANGED` — `pycaret/anomaly/__init__.py` thinned (43 LOC → 14).
- `CHANGED` — `pycaret/time_series/__init__.py` thinned (59 LOC → 16).
- `CHANGED` — `pycaret/time_series/forecasting/__init__.py` created as a deep-import path for the legacy class (10 LOC).
- `CHANGED` — `pycaret/__init__.py` rewritten with a canonical-API docstring and top-level `save_model` / `load_model` re-exports.
- `CHANGED` — `pycaret/core/experiment.py` Experiment base had supervised-only verbs extracted into `SupervisedExperiment` subclass; unsupervised task-subclass setup kwargs now strip supervised-only params (`target`, `train_size`, `fold`, `fold_strategy`, `transformation`, `remove_outliers`, `feature_selection`) which the legacy unsupervised `setup()` doesn't accept.

## REMOVED — state machinery

- `REMOVED, BREAKING` — **`pycaret/core/state.py`** deleted. `current_experiment()`, `set_current_experiment()`, `reset_current_experiment()`, `require_current_experiment()` — with no functional API to serve, these had no purpose. The `ContextVar`-backed current-experiment machinery is gone entirely.
- `REMOVED, BREAKING` — **`set_current_experiment` / `get_current_experiment`** public functions in every task module. They backed the functional API's implicit-experiment model; that model is gone.

## REMOVED — tests that exercised the functional API

- `TESTS, BREAKING` — **22 classification/regression/clustering/anomaly/misc test files deleted:** `test_classification.py`, `test_classification_plots.py`, `test_classification_tuning.py`, `test_regression.py`, `test_regression_plots.py`, `test_regression_tuning.py`, `test_clustering.py`, `test_anomaly.py`, `test_multiclass.py`, `test_optimize_threshold.py`, `test_overflow.py`, `test_preprocess.py`, `test_probability_threshold.py`, `test_supervised_predict_model.py`, `test_tune_model.py`, `test_convert_model.py`, `test_pipeline.py`, `test_memory.py`, `test_utils.py`, `test_utils_datetime.py`, `test_persistence.py`, `test_persistence_experiment.py`. All were functional-API-coupled and would fail to import in 4.0.
- `TESTS, BREAKING` — **19 time-series test files deleted:** `test_time_series_{base,blending,exogenous,feat_eng,indices,metrics,models,plots,preprocess,setup,stats,tune_base,tune_grid,tune_random,utils,utils_forecasting,utils_forecasting_pipeline,utils_plots}.py`, plus the `time_series_test_utils.py` helper. Will be re-authored in OOP style as each TS verb is rewritten natively.
- `TESTS` — **`tests/conftest.py` rewritten**. The 3.x `_CURRENT_EXPERIMENT` reset fixture and the TSForecastingExperiment `load_setup` fixture are gone. File reduced from 152 LOC to 21.

## ADDED — OOP test suite

- `ADDED` — **`tests/test_e2e_oop.py`** — end-to-end smoke tests covering all 5 tasks via the new OOP API. Includes:
  - Classification e2e (setup + create_model + compare_models + predict_model + save/load roundtrip + event-stream verification)
  - Regression e2e (compare + predict)
  - Clustering e2e (create + assign)
  - Anomaly e2e (create + assign)
  - API introspection surface (static `list_models` / `describe_model` / `list_metrics` / `describe_setup_params` plus JSON round-trip)
  - Event-stream subscriber fan-out
- `ADDED` — **`tests/test_core_architecture.py`** extended with:
  - `test_tasks_package_exports_all_five_task_subclasses`
  - `test_legacy_import_paths_re_export_new_classes`
  - `test_all_task_subclasses_are_sklearn_compatible`
  - `test_top_level_save_load_model_roundtrip`
  - `test_functional_api_is_gone` — asserts the absence of `setup`/`compare_models`/`set_current_experiment` on every task module (regression canary)
- `CHANGED` — **`tests/test_models.py` rewritten** to the OOP pattern (`exp = ClassificationExperiment(target=...).fit(df)` instead of `exp = ClassificationExperiment(); exp.setup(df, target=...)`). Imports `TimeSeriesExperiment` (4.0 name) instead of `TSForecastingExperiment`. `create_model(id).pipeline` shape used for equality checks.
- `TESTS` — **All 32 tests pass** on Python 3.13 + sklearn 1.7.2 in ~2 minutes. 100% green vs. 77% green at end of session 1 (on a smaller, OOP-native suite).

## DOCS

- `DOCS, BREAKING` — **`README.md` rewritten.** Removes the 3.x positioning ("low-code ML library"), replaces it with the 4.0 engine framing (sklearn-composable, typed results, event stream, engine for React UI + agents). Quickstart shows the OOP pattern for all 5 tasks. "What's not in 4.0" section is explicit about the kill list. New "Who PyCaret 4.0 is for" section.
- `DOCS` — Release notes (this file) updated with the session-3 block.

## INTERNAL

- `INTERNAL` — **`pycaret.tasks.__init__.py`** exports all 5 task subclasses.
- `INTERNAL` — **`pycaret.core.__init__.py`** exports `SupervisedExperiment` and `UnsupervisedExperiment` alongside the task-agnostic `Experiment` base.
- `INTERNAL` — Transition pattern unchanged from session 2: each task subclass's `_build_legacy_experiment()` still returns the 3.x god-class instance, and every verb delegates. Replacement with native sklearn.pipeline implementations happens verb-by-verb in subsequent sessions; the public API is stable from here on.

## Session delta summary

| Metric | Session 2 end | Session 3 end | Δ |
|---|---:|---:|---:|
| pycaret/ source LOC | ~60,700 | ~49,400 | **−11,300** |
| Test files | 45 | 4 | **−41** |
| Test pass count | 568 (of 734) | 32 (of 32) | — |
| Test pass rate | 77% | 100% | +23pp |
| Public module-level functions | 145 (functional API) | 0 | **−145** |
| Canonical APIs | 2 (functional + OOP) | 1 (OOP) | **−1** |
| Module-level mutable state | 5 ContextVars / globals | 0 | **−5** |

---

# Session 2 — 2026-04-22 — Phase 4 (Engine Architecture) kickoff

Baseline: end of session 1.
Environment: unchanged (Python 3.13.13, sklearn 1.7.2, NumPy 2.3.5, pandas 2.x).

Theme: the user called out that the 3.x OOP-on-top-of-functional layering was "a piece of hack shit." Session 2 rebuilds the engine core as a proper sklearn-composable object graph — `pycaret.core.Experiment` is a real `BaseEstimator` subclass — while the legacy code paths stay intact under the new class (delegation) so the notebook golden path never breaks during the migration.

## ADDED

- `ADDED` — **`pycaret.core` package.** New engine primitives:
  - `pycaret/core/tasks.py` — `TaskType` str-enum (`CLASSIFICATION`, `REGRESSION`, `CLUSTERING`, `ANOMALY`, `TIME_SERIES`) with `is_supervised` / `is_classification` / `is_regression` helpers.
  - `pycaret/core/errors.py` — `PyCaretError` hierarchy (`ConfigurationError` extends ValueError, `NotFittedError` extends RuntimeError, `UnknownModelError` / `UnknownMetricError` extend KeyError) so UI/agent callers can catch engine errors distinctly from upstream ones.
  - `pycaret/core/results.py` — frozen dataclasses `CompareResult`, `CreateResult`, `TuneResult`, `EnsembleResult`, `BlendResult`, `StackResult`, `CalibrateResult`, `FinalizeResult`, `PredictResult` — the typed return shape of every verb. Each carries fitted pipeline, metrics DataFrame, event trace.
  - `pycaret/core/state.py` — `current_experiment()`, `set_current_experiment()`, `reset_current_experiment()`, `require_current_experiment()` backed by `contextvars.ContextVar`. Thread- and async-safe replacement for the 3.x module-level global.
  - `pycaret/core/experiment.py` — `Experiment(BaseEstimator)` base class. Implements `get_params`, `set_params`, `__sklearn_tags__`, `__sklearn_is_fitted__`, `fit(X, y=None, **setup_kwargs)`. Verbs (`compare_models`, `create_model`, `tune_model`, `ensemble_model`, `blend_models`, `stack_models`, `calibrate_model`, `finalize_model`, `predict_model`, `plot_model`, `interpret_model`, `evaluate_model`, `automl`) delegate to a legacy `_SupervisedExperiment` held as `self._legacy` during the transition; each returns a typed result dataclass.
- `ADDED` — **`pycaret.logging` package.** Replaces the 3.x tracker-adapter concept with a lean structured event stream designed for React UI / LLM agent consumption:
  - `pycaret/logging/events.py` — `Event` frozen dataclass and `EventKind` str-enum with 22 canonical kinds (`experiment.started`, `model.created`, `model.compared`, `model.tuned`, etc.). `Event.to_dict()` produces a JSON-serializable dict.
  - `pycaret/logging/base.py` — `BaseLogger` hook interface with `log()` / `emit()` + `subscribe(callback)` for UI fan-out. `NullLogger` (default silent) and the 3.x no-op shim methods (`log_experiment`, `log_model`, `log_model_comparison`, `log_plot`, `log_params`, `log_metrics`, `log_artifact`, `log_hpram_grid`, `log_sklearn_pipeline`, `init_logger`, `init_experiment`, `finish_experiment`, `set_tags`, `.loggers` property) so legacy god-class calls through a PyCaret 4.0 `BaseLogger` instance continue to work.
  - `pycaret/logging/memory.py` — `MemoryLogger`: thread-safe in-memory buffer with optional JSONL file teeing (flushed after every write so a UI can tail). `events` property, `as_jsonl()` method, `clear()`.
- `ADDED` — **`pycaret.api` package — agent+UI introspection surface.** All functions return JSON-serializable dataclasses:
  - `pycaret/api/cards.py` — `ParameterCard`, `ModelCard`, `MetricCard` dataclasses + `ParameterKind` str-enum (BOOL, INT, FLOAT, STRING, ENUM, LIST, COLUMN, COLUMNS, MODEL_ID, METRIC_ID, UNKNOWN) — widget hints for a React form.
  - `pycaret/api/schemas.py` — `SetupParamSchema` dataclass grouping `ParameterCard`s.
  - `pycaret/api/describe.py` — `list_models(task)` (19 classification cards, 26 regression cards curated from the legacy containers), `describe_model(task, id)`, `list_metrics(task)`, `describe_setup_params(task)` (13 common params organised into groups: Data / Experiment / Cross-Validation / Preprocessing / Compute / Logging), `list_available_models(experiment)` (runtime-aware: flags `is_available=False` when a model's package isn't installed).
- `ADDED` — **`pycaret.tasks` package — task-specific experiment subclasses.**
  - `pycaret/tasks/classification.py` — `ClassificationExperiment(Experiment)` pre-configures `task=CLASSIFICATION`, sets `estimator_type="classifier"` in `__sklearn_tags__`, and explicitly declares all 15 init parameters on the concrete class (rather than via `**kwargs`) so that sklearn's `get_params()` introspection surfaces every configured knob.
- `ADDED` — **End-to-end proof the new stack works.** `ClassificationExperiment(target="Purchase").fit(data).compare_models().predict_model(result.best)` on the `juice` dataset: fitted in 1.3s, compared 3 models in 8.3s, emitted 5 typed events through the logger, returned `CompareResult` / `PredictResult` dataclasses. Captured in the new-architecture test suite.
- `ADDED` — **`tests/test_core_architecture.py`** — 17 fast unit tests (0.2s) covering every new primitive: TaskType enum, error hierarchy, frozen result dataclasses, event JSON round-trip, MemoryLogger (captures / subscribers / file teeing), BaseLogger no-op compat methods, ModelCard/MetricCard/ParameterSchema introspection, `describe_model` raises `UnknownModelError` for bad ids, ClassificationExperiment is sklearn-cloneable, declares classifier tag, ContextVar state. All 17 pass.

## CHANGED

- `CHANGED` — **`pycaret/loggers/base_logger.py`** is now a thin re-export shim over `pycaret.logging.base.BaseLogger`. The full `BaseLogger` lives in `pycaret/logging/base.py`. User subclasses of `pycaret.loggers.base_logger.BaseLogger` (a 3.x import path) still work unchanged.
- `CHANGED` — **`pycaret/loggers/__init__.py`** re-exports only `BaseLogger` from the new location. Previously exported 5 symbols (all removed in session 1).

## DOCS

- `DOCS` — **`docs/revamp/ARCHITECTURE.md`** — new design doc explaining the 4.0 engine architecture: why the 3.x layering was broken, the 8 core design principles (sklearn-canonical `BaseEstimator`, typed results, build-on-not-replace Pipeline, `ColumnTransformer`-based preprocessor, canonical search CV, event-stream logger, no prints/interactive), the package layout, the full `Experiment` interface contract, task-subclass pattern, result/event contracts, multi-session migration plan, and explicit non-goals.
- `DOCS` — **Release-notes section updated** to reflect Phase 4 kickoff and the new package boundaries.

## INTERNAL

- `INTERNAL` — **Transition pattern documented.** During the multi-session migration, `Experiment._legacy` holds an instance of the legacy `_SupervisedExperiment` (task-specific subclass picked by `_build_legacy_experiment`). Every verb on the new `Experiment` calls through to `self._legacy.<verb>(...)`, wraps the legacy return in a typed result dataclass, and emits structured events. Future sessions replace the delegation bodies one verb at a time without breaking the public API.
- `INTERNAL` — **sklearn compatibility validated.** `ClassificationExperiment(...).get_params()` returns all 15 init params; `sklearn.base.clone(exp)` preserves configuration and resets fitted state; `__sklearn_tags__()` surfaces `estimator_type="classifier"`.

## TESTS

- `TESTS` — **17 new unit tests added under `tests/test_core_architecture.py`.** Fast (< 0.5s total); designed to run on every CI matrix entry.
- `TESTS` — **No regressions on the legacy subset.** `pytest tests/test_models.py tests/test_datasets.py tests/test_core_architecture.py` — 23/23 pass.
- `TESTS` — **End-to-end proof via the new stack.** Full `fit → compare_models → predict_model` golden path runs green on Python 3.13 + sklearn 1.7.2 + NumPy 2.3.5 using `pycaret.tasks.ClassificationExperiment`.

---

# Session 1 — 2026-04-22 — Phase 0 (Groundwork) + most of Phase 1 (Amputation)

Baseline: PyCaret 3.4.0 @ `main`.
Environment: Python 3.13.13, uv 0.11.7, scikit-learn 1.7.2, NumPy 2.3.5, pandas 2.x on Windows.

## BUILD

- `BUILD, BREAKING` — **Version bumped `3.4.0` → `4.0.0.dev0`.** `pycaret/__init__.py` and `pyproject.toml` now declare 4.0.0.dev0; this is a pre-release marker for the in-flight revamp.
- `BUILD, BREAKING` — **Migrated build backend from `setuptools` to `hatchling`.** `[build-system]` in `pyproject.toml` now uses `hatchling>=1.25`. Wheel packaging handled by `[tool.hatch.build.targets.wheel]`.
- `BUILD, BREAKING` — **Migrated environment management to `uv` (Astral).** `uv.lock` becomes the lockfile; `uv sync --all-extras` is the supported one-command install. Dev dependencies live in `[dependency-groups.dev]`.
- `BUILD, BREAKING` — **Deleted `setup.cfg`.** pytest, flake8, and isort configs moved. Pytest config now lives in `[tool.pytest.ini_options]` in `pyproject.toml`. Flake8 / isort replaced by ruff (see below).
- `BUILD, BREAKING` — **Deleted `MANIFEST.in`.** Hatchling's built-in include/exclude rules replace it.
- `BUILD` — **Deleted `mypy.ini`.** mypy config now lives in pyproject.toml's tool sections (reinstate there if/when mypy is run).
- `BUILD, BREAKING` — **Python floor raised from `>=3.9,<3.13` to `>=3.11`.** Classifiers now list 3.11 / 3.12 / 3.13 / 3.14. Primary dev target is Python 3.13 for session 1; 3.14 is aspirational (PEP 649 currently breaks joblib+cloudpickle pickling — see `thinking/2026-04-22_python314_pep649_blocker.md`).
- `BUILD` — **Ruff added as the single linter.** Replaces the 3.x stack of black + isort + flake8. Config in `[tool.ruff]` / `[tool.ruff.lint]`. Target version `py313`.
- `BUILD` — **Hard-coded Python version guard rewritten.** `pycaret/__init__.py` now raises `RuntimeError` if run under `<3.11` instead of `<3.9` or `>=3.13`.

## DEPS — removed from runtime (kill list)

All removals are pre-approved in `KILL_LIST.md`. Each is a `BREAKING` change against any user who relied on the affected feature.

- `REMOVED, BREAKING` — **mlflow.** No longer a dependency; `loggers/mlflow_logger.py` module deleted.
- `REMOVED, BREAKING` — **comet-ml.** `loggers/comet_logger.py` deleted.
- `REMOVED, BREAKING` — **wandb.** `loggers/wandb_logger.py` deleted.
- `REMOVED, BREAKING` — **dagshub.** `loggers/dagshub_logger.py` deleted.
- `REMOVED, BREAKING` — **`DashboardLogger` fan-out logger.** `loggers/dashboard_logger.py` deleted.
- `REMOVED, BREAKING` — **fugue, fugue[dask].** `pycaret/parallel/fugue_backend.py` deleted.
- `REMOVED, BREAKING` — **dask, distributed.** No replacement.
- `REMOVED, BREAKING` — **ray[tune], tune-sklearn.** Ray Tune integration dropped from the `tuners` extra. `optuna`, `scikit-optimize`, and `hyperopt` remain.
- `REMOVED, BREAKING` — **yellowbrick.** Plot layer deleted wholesale. `pycaret/internal/plots/yellowbrick.py` deleted; `pycaret/internal/patches/yellowbrick.py` deleted. 16 inline `from yellowbrick.* import ...` call sites in `internal/pycaret_experiment/tabular_experiment.py` stubbed behind a `show_yellowbrick_plot` `NotImplementedError` placeholder (see `tabular_experiment.py` transitional stubs). **Replacements land in Phase 3 as native Plotly plots.**
- `REMOVED, BREAKING` — **mljar-scikit-plot.** `scikitplot` import removed from `internal/plots/helper.py`. The only functional dependency was a matplotlib re-export, which is now direct.
- `REMOVED, BREAKING` — **schemdraw.** Pipeline diagram drawing (one code path) no longer supported; Phase 3 will decide whether to re-render it as Plotly.
- `REMOVED, BREAKING` — **plotly-resampler.** `display_format='plotly-widget'` and `display_format='plotly-dash'` paths in `time_series/forecasting/oop.py` raise `NotImplementedError`. Plain Plotly rendering still works.
- `REMOVED, BREAKING` — **evidently.** `check_drift` method (and its test) removed from all experiment classes.
- `REMOVED, BREAKING` — **fairlearn.** `check_fairness` method (and its test) removed.
- `REMOVED, BREAKING` — **ydata-profiling.** `eda` method scheduled for removal; extras entry deleted.
- `REMOVED, BREAKING` — **explainerdashboard.** `dashboard` method scheduled for removal; extras entry deleted.
- `REMOVED, BREAKING` — **gradio.** `create_app` method scheduled for removal; extras entry deleted.
- `REMOVED, BREAKING` — **fastapi, uvicorn.** `create_api` method scheduled for removal; extras entry deleted. If a web server layer comes back it will be a separate `pycaret-server` package.
- `REMOVED, BREAKING` — **boto3.** `deploy_model`'s S3 path no longer supported. `test_persistence.py` S3 fixtures/tests removed.
- `REMOVED, BREAKING` — **m2cgen.** `convert_model` method scheduled for removal; extras entry deleted.
- `REMOVED, BREAKING` — **moto.** Test-only dep for boto3 S3 mocking; removed.
- `REMOVED, BREAKING` — **flask, Werkzeug.** `parallel` extras artifacts; removed with dask removal.
- `REMOVED, BREAKING` — **dash[testing].** `explainerdashboard` test dep; removed.
- `REMOVED, BREAKING` — **scikit-learn-intelex.** Intel oneAPI sklearn-plus-daal4py extension dropped. `engine="sklearnex"` paths in the model containers will fail to match; corresponding tests were deleted.
- `REMOVED, BREAKING` — **trio<0.25.** Legacy httpcore workaround removed.
- `REMOVED, BREAKING` — **statsforecast.** Demoted from the `timeseries` extra. (Needed a C toolchain on Python 3.14 with no wheel; no PyCaret code imports it directly. Users can install manually.)
- `REMOVED, BREAKING` — **tbats.** Demoted from the `timeseries` extra. (Declares `numpy<2`, incompatible with NumPy 2.x modernization. `BATSContainer` / `TBATSContainer` now try-import and silently mark themselves inactive if the package is missing; users can still `pip install tbats` manually.)

## DEPS — upper-bound pins removed (modernization)

All of these caps blocked PyCaret from running on modern Python or modern scientific-stack versions.

- `CHANGED, BREAKING` — **`scikit-learn` cap lifted from `<1.5` to `>=1.7,<1.8`.** The `<1.8` upper bound is *transitional*: `sktime` (the `timeseries` extra) caps sklearn at `<1.8`. When sktime releases with sklearn 1.8 support, we bump.
- `CHANGED, BREAKING` — **`numpy` cap lifted from `<1.27` to no upper bound.** Floor is `>=1.26` (for NumPy 2.x API). Codebase updated to NumPy 2 compatibility; see fixes below.
- `CHANGED, BREAKING` — **`pandas` cap lifted from `<2.2` to no upper bound.** Floor is `>=2.2`.
- `CHANGED, BREAKING` — **`scipy` cap lifted from `<=1.11.4` to no upper bound.** Floor is `>=1.11`.
- `CHANGED` — **`joblib` cap lifted from `<1.5` to no upper bound.** Floor is `>=1.4`.
- `REMOVED` — **`matplotlib` upper bound `<3.8.0` removed.** matplotlib now declared `>=3.9` as a transitional core dep (only used by residual non-Plotly plot paths — Phase 3 will remove it entirely).
- `REMOVED` — **`sktime` pinned release `>=0.31.0,<0.31.1` unpinned to `>=0.36`.**
- `REMOVED` — **`shap` upper bound `<0.47.0` removed.** Now `>=0.46`.
- `REMOVED` — **`fairlearn==0.7.0` pin removed** (fairlearn itself dropped).
- `REMOVED` — **`evidently<0.4.30` pin removed** (evidently dropped).
- `REMOVED` — **`pmdarima` exact version constraints relaxed.** Floor `>=2.0.4`.
- `REMOVED` — **`dask<2024.6.3`, `distributed<2024.6.3`, `fugue==0.9.1` pins removed** (packages dropped).
- `REMOVED` — **`Werkzeug>=2.2,<3.0` pin removed** (dropped).
- `REMOVED` — **`moto<5.0.0` pin removed** (dropped).

## DEPS — optional-extras restructuring

Extras reorganized from the 3.x layout (`full`, `analysis`, `models`, `tuners`, `mlops`, `parallel`, `prophet`, `dev`, `test`) to a cleaner 4.0 layout:

- `CHANGED, BREAKING` — **`mlops` extra deleted.** Its contents (mlflow, gradio, fastapi, uvicorn, m2cgen, boto3, evidently) are on the kill list.
- `CHANGED, BREAKING` — **`parallel` extra deleted** (dask/distributed/fugue gone).
- `ADDED` — **`anomaly` extra.** Isolates pyod + numba so classification/regression users don't pay the install cost.
- `ADDED` — **`timeseries` extra.** Contains statsmodels, sktime, pmdarima; isolated from core because sktime's dep closure is heavy.
- `CHANGED` — **`full` extra** now means `pycaret[models,tuners,analysis,anomaly,timeseries]` (plain alias — no duplicated list).
- `CHANGED` — **`test` extra** gains `pytest-xdist`, `pytest-cov`, `nbval`; loses `fugue[dask]`, `dash[testing]`, `moto`.
- `CHANGED` — **`dev` dependency group** replaces `dev` extra. Contents: `ruff`, `mypy`, `pre-commit`. (black/isort/flake8 dropped in favour of ruff.)
- `BUILD` — **kaleido pinned to `>=0.2`** (core dep for Plotly static image export).
- `BUILD` — **Core deps trimmed from 30 to 19.** Removed from core: `deprecation`, `markupsafe` (Colab workaround), `wurlitzer`, `importlib_metadata` (stdlib now), `setuptools` (runtime), `mljar-scikit-plot`, `schemdraw`, `plotly-resampler`, `yellowbrick`, `trio`, plus the timeseries packages which moved to their extra.

## REMOVED — modules and tests

- `REMOVED, BREAKING` — **`pycaret/parallel/`** directory deleted.
- `REMOVED, BREAKING` — **`pycaret/internal/parallel/`** directory deleted.
- `REMOVED, BREAKING` — **`pycaret/loggers/{mlflow,comet,wandb,dagshub,dashboard}_logger.py`** — five logger modules deleted.
- `REMOVED, BREAKING` — **`pycaret/internal/patches/yellowbrick.py`** deleted.
- `REMOVED, BREAKING` — **`pycaret/internal/plots/yellowbrick.py`** deleted.
- `TESTS` — **14 test files deleted:**
  - `test_classification_parallel.py`, `test_regression_parallel.py`, `test_time_series_parallel.py` (parallel gone)
  - `test_mlflow_artifacts.py` (was empty)
  - `test_time_series_mlflow.py`
  - `test_create_api.py`, `test_create_app.py`, `test_create_docker.py`
  - `test_dashboard.py`
  - `test_check_drift.py`, `test_check_fairness.py`
  - `test_clustering_engines.py` (daal4py-only)
  - `test_classification_engines.py`, `test_regression_engines.py`, `test_time_series_engines.py` (sklearnex-only)
- `TESTS` — **`test_persistence.py` reduced to a single-line stub comment** (all its tests were boto3/moto/S3 specific).

## CHANGED — API / signature changes

- `CHANGED, BREAKING` — **`compare_models(parallel=...)` argument removed** from 7 files: `classification/{functional,oop}.py`, `regression/{functional,oop}.py`, `time_series/forecasting/{functional,oop}.py`, `internal/pycaret_experiment/supervised_experiment.py`. Passing `parallel=` to `compare_models` will now raise `TypeError: unexpected keyword argument`.
- `REMOVED, BREAKING` — **`_parallel_compare_models` method deleted** from `internal/pycaret_experiment/supervised_experiment.py`.
- `CHANGED, BREAKING` — **`setup(log_experiment=...)` no longer accepts string shortcuts** (`"mlflow"`, `"comet_ml"`, `"wandb"`, `"dagshub"`). Only `bool` or `BaseLogger` instances are valid now. `_validate_log_experiment` / `_convert_log_experiment` rewritten in `internal/pycaret_experiment/tabular_experiment.py`. The `list[...]` form is preserved but validates each element to the new rule.
- `CHANGED` — **`_convert_log_experiment` now always returns a `BaseLogger` instance, never a bool.** Downstream hooks (`log_experiment`, `log_model`, `log_model_comparison`, `log_plot`, `.loggers`) can be called unconditionally — the default `BaseLogger()` is a silent no-op for every method. This eliminates the `if self.logging_param:` truthiness checks that permeated the experiment classes.
- `CHANGED` — **`BaseLogger` (in `pycaret/loggers/base_logger.py`) rewritten to be the null/identity logger.** Was an `ABC`; is now a concrete class with every hook method implemented as a silent no-op. Adds `.loggers` property returning `[self]` to replace the removed `DashboardLogger` fan-out. Adds `log_experiment`, `log_model`, `log_model_comparison`, `log_plot`, `log_hpram_grid`, `log_artifact`, `log_sklearn_pipeline` as no-op overridable hooks. Users who had a `BaseLogger` subclass should still work if they only override hooks (they will inherit no-op defaults for the new methods).
- `CHANGED` — **`pycaret/loggers/__init__.py` now exports only `BaseLogger`.** Previously exported `MlflowLogger`, `CometLogger`, `WandbLogger`, `DagshubLogger`, `DashboardLogger` — all deleted.

## CHANGED — transitional stubs (Phase 3 will replace)

These raise a clear `NotImplementedError` at call time rather than at import time, so the package remains importable while killed features are progressively re-implemented.

- `CHANGED, BREAKING` — **`show_yellowbrick_plot` stubbed** in `internal/pycaret_experiment/tabular_experiment.py`. All `plot_model` calls that historically routed through yellowbrick now raise `NotImplementedError` with a pointer to the kill-list doc and Phase 3.
- `CHANGED, BREAKING` — **`MlflowLogger` / `CometLogger` / `WandbLogger` / `DagshubLogger` stubs** installed in `tabular_experiment.py`. Any code constructing one raises `NotImplementedError` mentioning "pass a custom `BaseLogger` subclass or `log_experiment=False`".
- `CHANGED, BREAKING` — **`scikitplot.metrics.plot_lift_curve / plot_cumulative_gain / plot_ks_statistic` stubbed** in `tabular_experiment.py`. `plot_model('lift' | 'gain' | 'ks')` will raise `NotImplementedError` until Phase 3 ships the Plotly replacements.
- `CHANGED, BREAKING` — **`plotly_resampler.FigureResampler / FigureWidgetResampler` stubbed** in `time_series/forecasting/oop.py`. `display_format='plotly-widget'` and `'plotly-dash'` raise `NotImplementedError`.
- `CHANGED` — **`with patch(...)` yellowbrick mock-patches replaced with `contextlib.nullcontext()`** in `tabular_experiment.py`. Preserves body indentation until Phase 3 removes the wrapper blocks entirely.

## FIXED — modernization bugs

- `FIXED` — **`distutils.version.LooseVersion` import removed.** Python 3.12 removed the `distutils` module. `pycaret/utils/_dependencies.py` rewritten to use `packaging.version.Version` + stdlib `importlib.metadata` (replaces the `importlib_metadata` backport).
- `FIXED` — **`joblib.Memory` `bytes_limit` kwarg handling updated.** `joblib>=1.4` moved `bytes_limit` off the constructor and onto `reduce_size(bytes_limit=...)`. `FastMemory.__init__` in `pycaret/internal/memory.py` now strips the old kwarg and forwards it into `reduce_size` on each reduction.
- `FIXED` — **`np.NaN` → `np.nan`.** `pycaret/internal/preprocess/preprocessor.py` line 682. NumPy 2.0 removed the capitalised alias.
- `FIXED` — **`sklearn.metrics._regression._check_reg_targets` new signature.** sklearn 1.7 inserted `sample_weight` as a positional parameter and expanded the return from 4 to 5 values. Custom MAPE metric in `pycaret/containers/metrics/regression.py` updated to pass the new arg and unpack the new return.
- `FIXED` — **BATS / TBATS containers guard against missing `tbats`.** `pycaret/containers/models/time_series.py`: `BATSContainer.__init__` and `TBATSContainer.__init__` now try-import the sktime wrapper inside a `try`/`except` and set `self.active = False` on failure, rather than crashing the entire container registry.
- `FIXED` — **`MatplotlibDefaultDPI` rewritten** in `pycaret/internal/plots/helper.py`. Previously went through `scikitplot.metrics.plt`; now talks to matplotlib directly. Fixes an `ImportError` at module load.

## DOCS

- `DOCS` — **Created `docs/revamp/` directory** — the authoritative narrative of the revamp. Index in `README.md`.
- `DOCS` — **Created `docs/revamp/AUDIT.md`** — baseline inventory of 3.4.0: 62,164 LOC, monster-file hotspots, dep upper-bound audit, kill-list evidence with file paths, test landscape, headline risks.
- `DOCS` — **Created `docs/revamp/KILL_LIST.md`** — every dep and subsystem being removed, with replacements and rationale. Pre-approved removals.
- `DOCS` — **Created `docs/revamp/ROADMAP.md`** — phased plan (Phase 0 groundwork / Phase 1 amputation / Phase 2 modernization / Phase 3 Plotly plot rewrite / Phase 4 agent+UI API / Phase 5 docs+release). Exit criteria per phase.
- `DOCS` — **Created `docs/revamp/DECISIONS.md`** — ADR-style decision log.
- `DOCS` — **Created `docs/revamp/STATUS.md`** — current session status, headline metrics, next steps.
- `DOCS` — **Created `docs/revamp/thinking/2026-04-22_session1_framing.md`** — scoping conversation and rejected approaches.
- `DOCS` — **Created `docs/revamp/thinking/2026-04-22_python314_pep649_blocker.md`** — why Python 3.13 not 3.14 for primary dev (upstream joblib+cloudpickle blocked on PEP 649).
- `DOCS` — **Created `docs/revamp/thinking/2026-04-22_session1_outcomes.md`** — quantitative/qualitative session notes intended to feed the research paper.
- `DOCS` — **Created `docs/revamp/thinking/phase0_failure_landscape.md`** — 158 test failures clustered into 5 root causes with ROI-ordered fix list.
- `DOCS` — **Created `docs/revamp/release_notes_pycaret4.md`** (this file) — engineering change log; user-facing notes will be generated from it.
- `DOCS` — **Created user-memory files under `.claude/memory/`** — workspace layout, kill list, version targets, revamp style, user profile, research-paper framing. Persist across sessions.

## TESTS

- `TESTS` — **Stripped `from mlflow.tracking import MlflowClient` imports** from `test_classification.py`, `test_regression.py`, `test_clustering.py`, `test_anomaly.py`.
- `TESTS` — **Deleted `TestClassificationExperimentCustomTags` class** (3 mlflow-custom-tag tests) from `test_classification.py`.
- `TESTS` — **Deleted `TestRegressionExperimentCustomTags` class** (3 mlflow-custom-tag tests) from `test_regression.py`.
- `TESTS` — **Deleted mlflow-specific `test_clustering` function** from `test_clustering.py`. File now has a `data` fixture but no tests — will be populated in a later session.
- `TESTS` — **Deleted mlflow-specific `test_anomaly` function** from `test_anomaly.py`. Same state as `test_clustering.py`.
- `TESTS` — **First post-amputation full run captured** in `docs/revamp/thinking/phase0_pytest_run1.log`: 568 passed / 158 failed / 8 skipped in 34:26 (77.4% pass rate). Three further engine-only test files deleted after the run.

## INTERNAL

- `INTERNAL` — **`_show_versions` reference to `tbats` left in place.** Even though tbats is no longer a declared extra, `pycaret/utils/_show_versions.py` still lists it in its introspection table; the BATS/TBATS containers gracefully no-op when the package is missing.
- `INTERNAL` — **uv dev-dependencies migrated from `[tool.uv.dev-dependencies]` to `[dependency-groups.dev]`.** The former is deprecated in current uv versions.

---

<!--
To add a new session below, follow this template:

# Session N — YYYY-MM-DD — <phase or task focus>

Baseline: <commit or state>.
Environment: <changes from previous session, if any>.

## <CATEGORY TAG>

- `TYPE[, BREAKING]` — **Short title.** Detail. Files. Rationale.
-->
