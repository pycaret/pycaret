/**
 * Trial Detail — `/runs/:runId/trials/:trialId`.
 *
 * One candidate model from a `compare_models` / `search` run, presented as
 * a self-contained model dashboard:
 *
 *   [breadcrumb] · h1 · pill (Rank · Best · Promoted) · [Download · Promote]
 *   [KPI strip — every numeric metric, primary metric ★, Δ vs best]
 *   [Tabs: Overview | Pipeline | Hyperparameters | Plots | Artifact]
 *
 * - Overview: sklearn's native nested-box pipeline diagram (the same
 *   one Jupyter shows with `set_config(display='diagram')`) + run
 *   context block (dataset, target, fold, train_size, session_id) +
 *   "vs Best" deltas if this isn't the winning trial.
 * - Pipeline: explicit vertical step-flow with class + module +
 *   per-step params accordion. Complementary to Overview's diagram.
 * - Hyperparameters: sortable table of the *final estimator* params.
 * - Plots: model diagnostics, sub-grouped by category (Diagnostics /
 *   Explainability / Curves / Raw) — every plot generated on-demand
 *   from the trial pickle. No promote required.
 * - Artifact: file metadata + sha256 + on-disk path.
 *
 * Download is fetched via axios (responseType: 'blob') so the auth
 * interceptor attaches the bearer token; a plain `<a download>` skips
 * headers and 401s.
 */

import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import Plot from 'react-plotly.js';
import {
  describeApi,
  experimentsApi,
  pipelinesApi,
  plotsApi,
  projectsApi,
  runsApi,
  workspacesApi,
} from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import { BackButton } from '@/components/BackButton';
import { Dialog } from '@/components/Dialog';
import {
  EnsembleDialog,
  TuneDialog,
} from '@/components/TrialActionDialog';
import { DeployFromPipelineDialog } from '@/components/DeployFromPipelineDialog';
import { EventLogDrawer } from '@/components/EventLogDrawer';
import { PlotlyFigure } from '@/components/PlotlyFigure';
import { PipelineDiagram } from '@/components/PipelineDiagram';
import type { PipelineNode, PlotEnvelope, TaskType } from '@/api/types';

const ASCENDING_METRICS = new Set(['MAE', 'MSE', 'RMSE', 'RMSLE', 'MAPE', 'TT (Sec)']);

const PLOT_TITLES: Record<string, string> = {
  confusion_matrix: 'Confusion matrix',
  roc_curve: 'ROC curve',
  pr_curve: 'Precision–Recall',
  calibration_curve: 'Calibration',
  threshold_curve: 'Threshold sweep',
  lift_curve: 'Lift',
  gain_curve: 'Gain',
  class_distribution: 'Class distribution',
  feature_importance: 'Feature importance',
  permutation_importance: 'Permutation importance',
  residuals: 'Residuals vs. predicted',
  residuals_distribution: 'Residuals distribution',
  prediction_error: 'Predicted vs. actual',
  learning_curve: 'Learning curve',
  cluster_distribution: 'Cluster sizes',
  elbow_curve: 'Elbow curve',
  silhouette_curve: 'Silhouette score',
  silhouette_plot: 'Silhouette per cluster',
  embedding_2d: '2-D embedding',
  score_distribution: 'Anomaly score distribution',
  anomaly_map: 'Anomaly map',
};

type Tab =
  | 'overview'
  | 'pipeline'
  | 'params'
  | 'plots'
  | 'predict'
  | 'validation'
  | 'artifact';
type PlotCategory = 'diagnostics' | 'explainability' | 'curves' | 'raw';

// Mirrors apps/web/src/pages/ModelCard.tsx so behaviour stays consistent
// across the two model-detail surfaces.
const PLOT_CATEGORIES: Record<string, Record<PlotCategory, string[]>> = {
  classification: {
    diagnostics: ['confusion_matrix', 'roc_curve', 'pr_curve', 'class_distribution'],
    explainability: ['feature_importance', 'permutation_importance'],
    curves: ['calibration_curve', 'threshold_curve', 'lift_curve', 'gain_curve'],
    raw: [],
  },
  regression: {
    diagnostics: ['prediction_error', 'residuals', 'residuals_distribution'],
    explainability: ['feature_importance', 'permutation_importance'],
    curves: ['learning_curve'],
    raw: [],
  },
  clustering: {
    diagnostics: ['cluster_distribution', 'silhouette_plot'],
    explainability: [],
    curves: ['elbow_curve', 'silhouette_curve'],
    raw: ['embedding_2d'],
  },
  anomaly: {
    diagnostics: ['score_distribution'],
    explainability: [],
    curves: [],
    raw: ['anomaly_map'],
  },
  time_series: {
    diagnostics: ['forecast', 'residual_diagnostics'],
    explainability: [],
    curves: ['decomposition', 'acf', 'pacf'],
    raw: [],
  },
};

const CATEGORY_LABEL: Record<PlotCategory, string> = {
  diagnostics: 'Diagnostics',
  explainability: 'Explainability',
  curves: 'Curves',
  raw: 'Raw',
};

// Snapshot keys we want to surface on the Overview page, in display
// order. Anything else stays under the Artifact tab's "raw snapshot"
// dump (which we keep terse — most users don't need the long tail).
const SNAPSHOT_FIELDS: { key: string; label: string }[] = [
  { key: 'task', label: 'Task' },
  { key: 'target', label: 'Target' },
  { key: 'sklearn_dataset', label: 'Dataset (sklearn)' },
  { key: 'data_source_id', label: 'Dataset' },
  { key: 'plan', label: 'Plan' },
  { key: 'train_size', label: 'Train size' },
  { key: 'fold', label: 'CV folds' },
  { key: 'session_id', label: 'Seed' },
];

// ─────────────────────────────────────────────── helpers

function formatBytes(n: number | null | undefined): string {
  if (n == null) return '—';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(2)} MB`;
}

function formatMetric(v: unknown): string {
  if (typeof v === 'number') return v.toFixed(4);
  if (v == null) return '—';
  return String(v);
}

function formatParam(v: unknown): string {
  if (v == null) return 'None';
  if (typeof v === 'string') return v || "''";
  if (typeof v === 'number' || typeof v === 'boolean') return String(v);
  try {
    const json = JSON.stringify(v);
    return json.length > 80 ? `${json.slice(0, 77)}…` : json;
  } catch {
    return String(v);
  }
}

function shortModuleHint(mod: string): string {
  // sklearn.linear_model._logistic → sklearn.linear_model
  if (!mod) return '';
  const parts = mod.split('.');
  if (parts.length <= 2) return mod;
  return `${parts[0]}.${parts[1]}`;
}

// ─────────────────────────────────────────────── page

export function TrialDetail() {
  const { runId = '', trialId = '' } = useParams<{
    runId: string;
    trialId: string;
  }>();
  const qc = useQueryClient();

  const trial = useQuery({
    queryKey: ['runs', runId, 'trials', trialId],
    queryFn: () => runsApi.trial(runId, trialId),
    enabled: !!runId && !!trialId,
  });

  const run = useQuery({
    queryKey: ['runs', runId],
    queryFn: () => runsApi.get(runId),
    enabled: !!runId,
  });

  const task = (run.data?.snapshot as { task?: TaskType } | null)?.task ??
    (trial.data?.task as TaskType | undefined);
  const models = useQuery({
    queryKey: ['describe', 'models', task],
    queryFn: () => describeApi.models(task!),
    enabled: !!task,
    staleTime: 10 * 60 * 1000,
  });

  const friendlyName = useMemo(() => {
    const id = trial.data?.model_id;
    if (!id) return '';
    const card = (models.data ?? []).find((m) => m.id === id);
    return card?.name ?? id;
  }, [models.data, trial.data?.model_id]);

  const [tab, setTab] = useState<Tab>('overview');

  // For "vs Best" deltas: pull all trials of this run and locate the
  // winner. Cheap — same query the parent run page uses.
  const trialsList = useQuery({
    queryKey: ['runs', runId, 'trials'],
    queryFn: () => runsApi.trials(runId),
    enabled: !!runId,
    staleTime: 60_000,
  });

  // ─── compare dialog ─────────────────────────────────────────────
  const [compareOpen, setCompareOpen] = useState(false);

  // ─── trial action dialogs (tune / ensemble) ────────────────────
  const [tuneOpen, setTuneOpen] = useState(false);
  const [ensembleOpen, setEnsembleOpen] = useState(false);

  // ─── event log drawer ──────────────────────────────────────────
  // Same drawer the run page uses. Tune / Ensemble pop it open on
  // submit so the user immediately sees their action's events
  // streaming — same UX as the compare run.
  const [logOpen, setLogOpen] = useState(false);

  // ─── registry-promote dialog (Phase 7) ─────────────────────────
  // Distinct from the legacy ``Promote`` (which creates a Pipeline
  // row). This one creates a versioned RegisteredModelVersion via
  // the Phase 7 registry. Pre-existing legacy Promote stays for
  // now — both buttons live side-by-side so users can choose.

  // ─── promote dialog ─────────────────────────────────────────────
  const [promoteOpen, setPromoteOpen] = useState(false);
  const [promoteName, setPromoteName] = useState('');
  const [promoteDesc, setPromoteDesc] = useState('');
  const promote = useMutation({
    mutationFn: () =>
      runsApi.trialPromote(runId, trialId, {
        name: promoteName.trim(),
        description: promoteDesc.trim() || undefined,
      }),
    onSuccess: () => {
      setPromoteOpen(false);
      setPromoteName('');
      setPromoteDesc('');
      // Refresh: this trial (back-link), the trials list (pill on the
      // sibling rows), and any cached pipeline list (so the parent run
      // page's "Promoted pipelines" section picks up the new row
      // without a full reload).
      qc.invalidateQueries({ queryKey: ['runs', runId, 'trials'] });
      qc.invalidateQueries({ queryKey: ['runs', runId, 'trials', trialId] });
      qc.invalidateQueries({ queryKey: ['pipelines'] });
    },
  });

  const unpromote = useMutation({
    mutationFn: () => runsApi.trialUnpromote(runId, trialId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['runs', runId, 'trials'] });
      qc.invalidateQueries({ queryKey: ['runs', runId, 'trials', trialId] });
      qc.invalidateQueries({ queryKey: ['pipelines'] });
    },
  });

  // ─── download via blob (auth-bearing) ──────────────────────────
  const download = useMutation({
    mutationFn: async () => {
      const t = trial.data;
      if (!t) return;
      const safe = (t.model_id || 'trial').replace(/[^a-zA-Z0-9._-]/g, '_');
      await runsApi.trialDownload(runId, trialId, `${safe}.pkl`);
    },
  });

  // "vs Best" deltas — best is the trial flagged is_best, not just
  // rank 1, since leaderboard order can disagree in edge cases.
  // MUST be called above any early returns to satisfy hooks rules.
  const bestMetrics = useMemo(() => {
    const items = trialsList.data?.items ?? [];
    const best = items.find((x) => x.is_best);
    return best?.metrics ?? null;
  }, [trialsList.data]);

  if (trial.isPending) {
    return <div className="card text-sm text-ink-500">Loading trial…</div>;
  }
  if (trial.error || !trial.data) {
    return (
      <div className="card text-sm text-danger-600">
        Could not load trial: {errorMessage(trial.error)}
      </div>
    );
  }

  const t = trial.data;
  const numericMetrics = Object.entries(t.metrics).filter(
    ([, v]) => typeof v === 'number',
  ) as [string, number][];
  const primaryMetric = numericMetrics[0]?.[0] ?? '';
  const finalEstimatorParams = Object.entries(t.params ?? {}).sort(([a], [b]) =>
    a.localeCompare(b),
  );

  const tabs: { id: Tab; label: string; count?: number }[] = [
    { id: 'overview', label: 'Overview' },
    { id: 'pipeline', label: 'Pipeline', count: t.pipeline_steps?.length || 0 },
    { id: 'params', label: 'Hyperparameters', count: finalEstimatorParams.length },
    { id: 'plots', label: 'Plots', count: t.available_plots?.length || 0 },
    { id: 'predict', label: 'Predict' },
    { id: 'validation', label: 'Validation' },
    { id: 'artifact', label: 'Artifact' },
  ];

  return (
    <div className="space-y-8">
      {/* ─── Header ─── */}
      <header>
        <BackButton to={`/runs/${runId}`} />
        <Lineage run={run.data} runId={runId} trialModelId={t.model_id} />
        <div className="flex items-end justify-between gap-4 flex-wrap">
          <div>
            <div className="flex items-center gap-3 flex-wrap">
              <h1 className="h-page">{friendlyName || t.model_id}</h1>
              <span
                className={`pill-${t.is_best ? 'success' : 'neutral'} capitalize`}
              >
                Rank #{t.rank}
                {t.is_best && ' · Best'}
              </span>
              {t.kind && t.kind !== 'compare' && (
                <span
                  className={`${TRIAL_KIND_CHIP[t.kind]} capitalize`}
                  title={
                    t.parent_trial_ids && t.parent_trial_ids.length > 0
                      ? `Built from ${t.parent_trial_ids.length} source trial${t.parent_trial_ids.length === 1 ? '' : 's'}`
                      : t.kind
                  }
                >
                  {t.kind}
                </span>
              )}
              {t.fitted_pipeline_id && (
                <span className="pill-accent">Promoted</span>
              )}
            </div>
            <p className="mt-2 text-sm text-ink-500 font-mono">{t.model_id}</p>
            {t.parent_trial_ids && t.parent_trial_ids.length > 0 && (
              <ParentLineage
                runId={runId}
                parentTrialIds={t.parent_trial_ids}
                kind={t.kind}
              />
            )}
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            <ModelCardExportButton trial={t} modelName={friendlyName || t.model_id} />
            <button
              type="button"
              className="btn-secondary inline-flex items-center gap-2"
              onClick={() => setLogOpen(true)}
              title="Live engine event log for this run"
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden
              >
                <path d="M4 6h16M4 12h16M4 18h10" />
              </svg>
              Event log
            </button>
            <button
              type="button"
              className="btn-secondary"
              onClick={() => setCompareOpen(true)}
              disabled={(trialsList.data?.items?.length ?? 0) <= 1}
              title="Compare this trial with another candidate"
            >
              Compare
            </button>
            <button
              type="button"
              className="btn-secondary"
              onClick={() => setTuneOpen(true)}
              disabled={!t.has_artifact || run.data?.status !== 'succeeded'}
              title="Random-search the estimator hyperparameter space"
            >
              Tune
            </button>
            <button
              type="button"
              className="btn-secondary"
              onClick={() => setEnsembleOpen(true)}
              disabled={!t.has_artifact || run.data?.status !== 'succeeded'}
              title="Bag or boost this estimator"
            >
              Ensemble
            </button>
            <button
              type="button"
              className="btn-secondary"
              onClick={() => download.mutate()}
              disabled={!t.has_artifact || download.isPending}
              title={
                !t.has_artifact ? 'No stored pipeline for this trial' : undefined
              }
            >
              {download.isPending ? 'Downloading…' : 'Download pipeline'}
            </button>
            {t.fitted_pipeline_id ? (
              <>
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => {
                    if (
                      window.confirm(
                        'Withdraw this promotion? The pipeline row will be removed (only if no deployments reference it).',
                      )
                    ) {
                      unpromote.mutate();
                    }
                  }}
                  disabled={unpromote.isPending}
                  title="Clear the back-link and delete the registry entry"
                >
                  {unpromote.isPending ? 'Withdrawing…' : 'Withdraw'}
                </button>
                <button
                  type="button"
                  className="btn-primary"
                  disabled
                  title="Already promoted — open the pipeline page or Withdraw"
                >
                  Promoted ✓
                </button>
              </>
            ) : (
              <button
                type="button"
                className="btn-primary"
                onClick={() => setPromoteOpen(true)}
                disabled={!t.has_artifact || run.data?.status !== 'succeeded'}
                title="Creates a Pipeline + a new Model Registry version"
              >
                Promote
              </button>
            )}
          </div>
        </div>
        {unpromote.error && (
          <p className="mt-2 text-xs text-danger-600">
            Withdraw failed: {errorMessage(unpromote.error)}
          </p>
        )}
        {download.error && (
          <p className="mt-2 text-xs text-danger-600">
            Download failed: {errorMessage(download.error)}
          </p>
        )}
      </header>

      {/* ─── KPI strip ─── */}
      {numericMetrics.length > 0 && (
        <section>
          <div
            className="grid gap-3"
            style={{
              gridTemplateColumns:
                'repeat(auto-fit, minmax(160px, 1fr))',
            }}
          >
            {numericMetrics.slice(0, 10).map(([k, v]) => (
              <div
                key={k}
                className="rounded-xl border border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 px-4 py-3"
              >
                <div className="text-xs font-medium text-ink-500 uppercase tracking-wider">
                  {k}
                  {k === primaryMetric && (
                    <span className="ml-1 text-accent-500" title="Primary metric">
                      ★
                    </span>
                  )}
                </div>
                <div className="mt-1 text-2xl font-semibold text-ink-900 dark:text-ink-50 tabular-nums">
                  {formatMetric(v)}
                </div>
                <div className="mt-0.5 text-[10px] text-ink-400">
                  {ASCENDING_METRICS.has(k) ? 'lower is better' : 'higher is better'}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ─── Tabs ─── */}
      <section>
        <div className="border-b border-ink-200 dark:border-ink-800 mb-6">
          <nav className="flex gap-6 -mb-px" aria-label="Trial sections">
            {tabs.map((tt) => (
              <button
                key={tt.id}
                type="button"
                onClick={() => setTab(tt.id)}
                className={`relative pb-3 text-sm font-medium transition-colors ${
                  tab === tt.id
                    ? 'text-ink-900 dark:text-ink-50'
                    : 'text-ink-500 hover:text-ink-700 dark:hover:text-ink-300'
                }`}
              >
                {tt.label}
                {typeof tt.count === 'number' && (
                  <span
                    className={`ml-1.5 text-xs ${
                      tab === tt.id ? 'text-ink-500' : 'text-ink-400'
                    }`}
                  >
                    ({tt.count})
                  </span>
                )}
                {tab === tt.id && (
                  <span className="absolute -bottom-px left-0 right-0 h-0.5 bg-accent-500" />
                )}
              </button>
            ))}
          </nav>
        </div>

        {tab === 'overview' && (
          <OverviewTab
            runId={runId}
            trialId={trialId}
            tree={t.pipeline_tree ?? null}
            snapshot={t.run_snapshot ?? {}}
            metrics={t.metrics}
            bestMetrics={bestMetrics}
            isBest={t.is_best}
            notes={t.notes ?? null}
            fittedPipelineId={t.fitted_pipeline_id ?? null}
            modelName={friendlyName || t.model_id}
            run={run.data}
            kind={t.kind ?? 'compare'}
            params={t.params ?? {}}
            parentTrialIds={t.parent_trial_ids ?? []}
          />
        )}
        {tab === 'pipeline' && <PipelineTab steps={t.pipeline_steps ?? []} />}
        {tab === 'params' && <ParamsTab entries={finalEstimatorParams} />}
        {tab === 'plots' && (
          <PlotsTab
            runId={runId}
            trialId={trialId}
            availablePlots={t.available_plots ?? []}
            task={t.task ?? task ?? null}
          />
        )}
        {tab === 'predict' && (
          <PredictTab
            runId={runId}
            trialId={trialId}
            snapshot={t.run_snapshot ?? {}}
            inputSchema={t.input_schema ?? null}
          />
        )}
        {tab === 'validation' && (
          <ValidationTab
            runId={runId}
            trialId={trialId}
            task={t.task ?? task ?? null}
          />
        )}
        {tab === 'artifact' && <ArtifactTab trial={t} />}
      </section>

      {/* ─── Promote dialog — creates Pipeline + RegisteredModelVersion ─── */}
      <Dialog
        open={promoteOpen}
        onClose={() => setPromoteOpen(false)}
        title="Promote trial"
        description="Creates a workspace Pipeline + a new Model Registry version. Same name reuses the existing model and bumps the version."
      >
        <form
          className="space-y-4"
          onSubmit={(e) => {
            e.preventDefault();
            if (promoteName.trim()) promote.mutate();
          }}
        >
          <div>
            <label className="block text-xs text-ink-500 mb-1">Name</label>
            <input
              className="input"
              value={promoteName}
              onChange={(e) => setPromoteName(e.target.value)}
              placeholder="e.g. iris-classifier"
              required
            />
          </div>
          <div>
            <label className="block text-xs text-ink-500 mb-1">
              Description (optional)
            </label>
            <textarea
              className="input"
              rows={2}
              value={promoteDesc}
              onChange={(e) => setPromoteDesc(e.target.value)}
            />
          </div>
          {promote.error && (
            <p className="text-xs text-danger-600">
              {errorMessage(promote.error)}
            </p>
          )}
          <div className="flex items-center justify-end gap-2 pt-2">
            <button
              type="button"
              className="btn-secondary"
              onClick={() => setPromoteOpen(false)}
              data-dialog-close
            >
              Cancel
            </button>
            <button
              type="submit"
              className="btn-primary"
              disabled={!promoteName.trim() || promote.isPending}
            >
              {promote.isPending ? 'Promoting…' : 'Promote'}
            </button>
          </div>
        </form>
      </Dialog>

      {/* ─── Compare picker ─── */}
      <CompareDialog
        open={compareOpen}
        onClose={() => setCompareOpen(false)}
        runId={runId}
        currentTrialId={trialId}
        items={trialsList.data?.items ?? []}
      />

      {/* ─── Tune / Ensemble — both produce a new trial in this run.
          On submit we pop the event-log drawer right here so the user
          sees the tuning events streaming live (same UX as compare). The
          new trial row arrives via the polling on the trials query +
          the run page's trials table when they navigate back. ─── */}
      <TuneDialog
        open={tuneOpen}
        onClose={() => setTuneOpen(false)}
        onSubmitted={() => setLogOpen(true)}
        runId={runId}
        trialId={trialId}
        modelLabel={friendlyName || t.model_id}
        task={t.task}
      />
      <EnsembleDialog
        open={ensembleOpen}
        onClose={() => setEnsembleOpen(false)}
        onSubmitted={() => setLogOpen(true)}
        runId={runId}
        trialId={trialId}
        modelLabel={friendlyName || t.model_id}
      />

      {/* ─── Live event log drawer for this run. Anchored to the
          trial page so the user doesn't have to leave to watch the
          tune / ensemble action play out. ─── */}
      <EventLogDrawer
        runId={runId}
        open={logOpen}
        onClose={() => setLogOpen(false)}
      />
    </div>
  );
}

// ──────────────────────────────────────────────────────────── tabs

interface Step {
  index: number;
  name: string;
  class: string;
  module: string;
  params: Record<string, unknown>;
  is_estimator: boolean;
}

// ─── Overview tab ─────────────────────────────────────────────────

function OverviewTab({
  runId,
  trialId,
  tree,
  snapshot,
  metrics,
  bestMetrics,
  isBest,
  notes,
  fittedPipelineId,
  modelName: _modelName,
  run: _run,
  kind,
  params,
  parentTrialIds,
}: {
  runId: string;
  trialId: string;
  tree: PipelineNode | null;
  snapshot: Record<string, unknown>;
  metrics: Record<string, number | string>;
  bestMetrics: Record<string, number> | null;
  isBest: boolean;
  notes: string | null;
  fittedPipelineId: string | null;
  modelName: string;
  run: { workspace_id?: string | null } | undefined;
  kind: string;
  params: Record<string, unknown>;
  parentTrialIds: string[];
}) {
  // Build a per-metric delta vs the winner. Only show when this trial
  // isn't the best and the winner has a comparable numeric value.
  const deltas = useMemo(() => {
    if (isBest || !bestMetrics) return null;
    const out: { name: string; mine: number; best: number; delta: number; pct: number }[] = [];
    for (const [k, v] of Object.entries(metrics)) {
      if (typeof v !== 'number') continue;
      const b = bestMetrics[k];
      if (typeof b !== 'number' || b === 0) continue;
      const delta = v - b;
      const pct = (delta / Math.abs(b)) * 100;
      out.push({ name: k, mine: v, best: b, delta, pct });
    }
    return out;
  }, [metrics, bestMetrics, isBest]);

  type SnapshotEntry = { key: string; label: string; value: unknown };
  const snapshotEntries: SnapshotEntry[] = SNAPSHOT_FIELDS.map(({ key, label }) => {
    const v = snapshot[key];
    return v == null || v === '' ? null : ({ key, label, value: v } as SnapshotEntry);
  }).filter((x): x is SnapshotEntry => x !== null);

  const dataSourceWsId = _run?.workspace_id ?? null;

  return (
    <div className="space-y-6">
      {/* Follow-on visualizations — only render when the trial actually
          came from a tune/blend/stack action. */}
      {kind === 'tuned' && <TuneHistorySection params={params} />}
      {(kind === 'blended' || kind === 'stacked') && (
        <EnsembleContributionSection
          runId={runId}
          trialId={trialId}
          parentTrialIds={parentTrialIds}
          ownMetrics={metrics}
          kind={kind}
        />
      )}

      {/* Native React pipeline diagram. */}
      <section>
        <div className="flex items-baseline justify-between mb-3 gap-3 flex-wrap">
          <h3 className="h-section">Pipeline</h3>
          <p className="text-xs text-ink-500">
            Color-coded by step category. Click any leaf to inspect its params.
          </p>
        </div>
        <PipelineDiagram tree={tree} />
      </section>

      {/* Promote history — only when this trial has been promoted. */}
      {fittedPipelineId && (
        <section>
          <h3 className="h-section mb-3">Promotion</h3>
          <PromoteHistoryCard
            pipelineId={fittedPipelineId}
            workspaceId={_run?.workspace_id ?? null}
          />
        </section>
      )}

      {/* Notes — free-form annotation, autosaved on blur. */}
      <section>
        <h3 className="h-section mb-3">Notes</h3>
        <NotesCard runId={runId} trialId={trialId} initialNotes={notes} />
      </section>

      {/* Run context — what was actually fitted. */}
      {snapshotEntries.length > 0 && (
        <section>
          <h3 className="h-section mb-3">Run context</h3>
          <div className="card">
            <dl className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-y-3 gap-x-6 text-sm">
              {snapshotEntries.map(({ key, label, value }) => {
                const isDataset =
                  key === 'data_source_id' &&
                  typeof value === 'string' &&
                  dataSourceWsId;
                return (
                  <div key={label}>
                    <dt className="text-xs text-ink-500 uppercase tracking-wide">
                      {label}
                    </dt>
                    <dd className="mt-1 font-mono text-xs text-ink-900 dark:text-ink-50 break-all">
                      {isDataset ? (
                        <Link
                          to={`/workspaces/${dataSourceWsId}/datasets/${value}/profile`}
                          className="text-accent-600 hover:underline"
                        >
                          {formatParam(value)} →
                        </Link>
                      ) : (
                        formatParam(value)
                      )}
                    </dd>
                  </div>
                );
              })}
            </dl>
          </div>
        </section>
      )}

      {/* vs Best metric deltas — only when this isn't the winner. */}
      {deltas && deltas.length > 0 && (
        <section>
          <h3 className="h-section mb-3">
            vs Best
            <span className="ml-2 text-xs text-ink-400 font-normal">
              (this trial − rank #1)
            </span>
          </h3>
          <div className="card overflow-hidden p-0">
            <table className="w-full text-sm">
              <thead className="bg-white text-ink-500 dark:bg-ink-900 border-b border-ink-200 dark:border-ink-800">
                <tr>
                  <th className="px-4 py-2.5 text-left font-medium">Metric</th>
                  <th className="px-4 py-2.5 text-right font-medium tabular-nums">
                    This trial
                  </th>
                  <th className="px-4 py-2.5 text-right font-medium tabular-nums">
                    Best
                  </th>
                  <th className="px-4 py-2.5 text-right font-medium tabular-nums">
                    Δ
                  </th>
                </tr>
              </thead>
              <tbody>
                {deltas.map(({ name, mine, best, delta, pct }) => {
                  const lowerIsBetter = ASCENDING_METRICS.has(name);
                  // delta vs best: negative on a higher-is-better metric is bad,
                  // positive on a lower-is-better metric is bad.
                  const ahead = lowerIsBetter ? delta < 0 : delta > 0;
                  const equal = delta === 0;
                  return (
                    <tr
                      key={name}
                      className="border-t border-ink-200 dark:border-ink-800"
                    >
                      <td className="px-4 py-2 font-mono text-xs text-ink-700 dark:text-ink-300">
                        {name}
                      </td>
                      <td className="px-4 py-2 text-right font-mono tabular-nums text-ink-900 dark:text-ink-50">
                        {mine.toFixed(4)}
                      </td>
                      <td className="px-4 py-2 text-right font-mono tabular-nums text-ink-500">
                        {best.toFixed(4)}
                      </td>
                      <td
                        className={`px-4 py-2 text-right font-mono tabular-nums ${
                          equal
                            ? 'text-ink-500'
                            : ahead
                              ? 'text-success-600'
                              : 'text-danger-600'
                        }`}
                      >
                        {delta >= 0 ? '+' : ''}
                        {delta.toFixed(4)}
                        <span className="ml-2 text-[10px] text-ink-400">
                          ({pct >= 0 ? '+' : ''}
                          {pct.toFixed(2)}%)
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}

function PipelineTab({ steps }: { steps: Step[] }) {
  const [openIdx, setOpenIdx] = useState<number | null>(null);

  if (steps.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 px-6 py-10 text-center">
        <p className="text-sm text-ink-500">
          Pipeline structure unavailable — the trial pickle could not be inspected.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {steps.map((step, idx) => {
        const open = openIdx === idx;
        const paramCount = Object.keys(step.params || {}).length;
        return (
          <div key={`${step.index}-${step.name}`} className="relative">
            {/* Connecting line between cards (skip after the last) */}
            {idx < steps.length - 1 && (
              <div
                className="absolute left-6 top-full h-2 w-px bg-ink-300 dark:bg-ink-700"
                aria-hidden
              />
            )}
            <div
              className={`group rounded-lg border bg-white dark:bg-ink-900 transition-shadow ${
                step.is_estimator
                  ? 'border-accent-300 dark:border-accent-500/40 ring-1 ring-accent-300/30 dark:ring-accent-500/15'
                  : 'border-ink-200 dark:border-ink-800'
              } hover:shadow-sm`}
            >
              <button
                type="button"
                onClick={() => setOpenIdx(open ? null : idx)}
                className="w-full flex items-center gap-3 px-4 py-3 text-left"
                aria-expanded={open}
              >
                {/* Index dot */}
                <div
                  className={`flex-shrink-0 h-8 w-8 rounded-full flex items-center justify-center text-xs font-semibold tabular-nums ${
                    step.is_estimator
                      ? 'bg-accent-500 text-white'
                      : 'bg-ink-100 text-ink-700 dark:bg-ink-800 dark:text-ink-200'
                  }`}
                >
                  {step.index + 1}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-mono text-xs text-ink-500">
                      {step.name}
                    </span>
                    <span className="text-sm font-medium text-ink-900 dark:text-ink-50 truncate">
                      {step.class}
                    </span>
                    {step.is_estimator && (
                      <span className="pill-accent">estimator</span>
                    )}
                  </div>
                  <div className="text-xs text-ink-400 font-mono truncate">
                    {shortModuleHint(step.module)}
                  </div>
                </div>
                <div className="text-xs text-ink-500 flex-shrink-0 flex items-center gap-2">
                  <span>{paramCount} param{paramCount === 1 ? '' : 's'}</span>
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className={`transition-transform ${open ? 'rotate-180' : ''}`}
                    aria-hidden
                  >
                    <polyline points="6 9 12 15 18 9" />
                  </svg>
                </div>
              </button>
              {open && (
                <div className="border-t border-ink-200 dark:border-ink-800 px-4 py-3 bg-ink-50/50 dark:bg-ink-950/30">
                  {paramCount === 0 ? (
                    <p className="text-xs text-ink-500">
                      No params on this step.
                    </p>
                  ) : (
                    <table className="w-full text-xs">
                      <tbody>
                        {Object.entries(step.params)
                          .sort(([a], [b]) => a.localeCompare(b))
                          .map(([k, v]) => (
                            <tr
                              key={k}
                              className="border-b border-ink-200/60 dark:border-ink-800/60 last:border-0"
                            >
                              <td className="py-1.5 pr-4 font-mono text-ink-600 dark:text-ink-400 align-top w-1/3">
                                {k}
                              </td>
                              <td className="py-1.5 font-mono text-ink-900 dark:text-ink-50 break-all">
                                {formatParam(v)}
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  )}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ParamsTab({ entries }: { entries: [string, unknown][] }) {
  if (entries.length === 0) {
    return (
      <p className="text-sm text-ink-500">
        No estimator params extracted (older runs predate trial-level params).
      </p>
    );
  }
  return (
    <div className="card overflow-hidden p-0">
      <table className="w-full text-sm">
        <thead className="bg-white text-ink-500 dark:bg-ink-900 border-b border-ink-200 dark:border-ink-800">
          <tr>
            <th className="px-4 py-2.5 text-left font-medium w-1/3">
              Parameter
            </th>
            <th className="px-4 py-2.5 text-left font-medium">Value</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([k, v]) => (
            <tr
              key={k}
              className="border-t border-ink-200 dark:border-ink-800 hover:bg-ink-50 dark:hover:bg-ink-950/40"
            >
              <td className="px-4 py-2 font-mono text-xs text-ink-700 dark:text-ink-300">
                {k}
              </td>
              <td className="px-4 py-2 font-mono text-xs text-ink-900 dark:text-ink-50 break-all">
                {formatParam(v)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── Plots tab ────────────────────────────────────────────────────

function PlotsTab({
  runId,
  trialId,
  availablePlots,
  task,
}: {
  runId: string;
  trialId: string;
  availablePlots: string[];
  task: string | null;
}) {
  // Fall back to the registry only if the detail endpoint didn't surface
  // available plots (e.g. older row, missing task).
  const registry = useQuery({
    queryKey: ['plots', 'registry'],
    queryFn: () => plotsApi.registry(),
    enabled: availablePlots.length === 0 && !!task,
    staleTime: 60 * 60 * 1000,
  });

  const kinds: string[] = useMemo(
    () =>
      availablePlots.length > 0
        ? availablePlots
        : (registry.data?.tasks?.[task ?? ''] ?? []),
    [availablePlots, registry.data, task],
  );

  // Group plots into the same buckets ModelCard uses, then drop empty
  // buckets. We only show kinds that are actually registered for the
  // task — that way a model the engine doesn't yet plot for renders an
  // empty section instead of a broken card.
  const groups = useMemo<{ key: PlotCategory; label: string; kinds: string[] }[]>(() => {
    const dispatch = task ? PLOT_CATEGORIES[task] : null;
    if (!dispatch) {
      return kinds.length > 0
        ? [{ key: 'diagnostics', label: 'Plots', kinds }]
        : [];
    }
    const present = new Set(kinds);
    const out: { key: PlotCategory; label: string; kinds: string[] }[] = [];
    (Object.keys(dispatch) as PlotCategory[]).forEach((cat) => {
      const filtered = dispatch[cat].filter((k) => present.has(k));
      if (filtered.length > 0) {
        out.push({ key: cat, label: CATEGORY_LABEL[cat], kinds: filtered });
      }
    });
    // Catch any kinds the engine reports but we haven't categorised — show
    // them under "Other" so nothing silently disappears as the registry grows.
    const known = new Set(out.flatMap((g) => g.kinds));
    const orphans = kinds.filter((k) => !known.has(k));
    if (orphans.length > 0) {
      out.push({ key: 'raw', label: 'Other', kinds: orphans });
    }
    return out;
  }, [kinds, task]);

  const [active, setActive] = useState<PlotCategory>('diagnostics');
  const visible = groups.find((g) => g.key === active) ?? groups[0];

  if (!task) {
    return (
      <p className="text-sm text-ink-500">
        Cannot derive plots — the run snapshot is missing the task type.
      </p>
    );
  }
  if (groups.length === 0) {
    return (
      <p className="text-sm text-ink-500">
        No plots registered for task <code>{task}</code>.
      </p>
    );
  }

  return (
    <div>
      {groups.length > 1 && (
        <div className="inline-flex rounded-md border border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 p-0.5 text-xs mb-4">
          {groups.map((g) => (
            <button
              key={g.key}
              type="button"
              onClick={() => setActive(g.key)}
              className={`px-3 py-1.5 rounded font-medium transition-colors ${
                (visible?.key ?? 'diagnostics') === g.key
                  ? 'bg-ink-900 text-white dark:bg-white dark:text-ink-900'
                  : 'text-ink-600 hover:text-ink-900 dark:text-ink-400 dark:hover:text-ink-50'
              }`}
            >
              {g.label}
              <span className="ml-1.5 text-ink-400">
                ({g.kinds.length})
              </span>
            </button>
          ))}
        </div>
      )}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {(visible?.kinds ?? []).map((k) => (
          <TrialPlotCard
            key={k}
            runId={runId}
            trialId={trialId}
            kind={k}
          />
        ))}
      </div>
    </div>
  );
}

function TrialPlotCard({
  runId,
  trialId,
  kind,
}: {
  runId: string;
  trialId: string;
  kind: string;
}) {
  const q = useQuery<PlotEnvelope, Error>({
    queryKey: ['runs', runId, 'trials', trialId, 'plots', kind],
    queryFn: () => runsApi.trialPlot(runId, trialId, kind),
    staleTime: 60_000,
    retry: false,
  });

  // Drop the in-figure title — we render our own card title so the
  // Plotly title doesn't double up or eat vertical space.
  const figure = useMemo(() => {
    if (!q.data?.figure) return q.data?.figure;
    const layout = { ...(q.data.figure.layout ?? {}) };
    delete (layout as Record<string, unknown>).title;
    return { ...q.data.figure, layout };
  }, [q.data]);

  return (
    <PlotlyFigure
      figure={figure}
      loading={q.isLoading}
      error={q.error ?? undefined}
      onRetry={() => q.refetch()}
      title={PLOT_TITLES[kind] ?? kind}
      height={300}
      hideToolbar
    />
  );
}

// ─── Artifact tab ─────────────────────────────────────────────────

interface ArtifactTabTrial {
  size_bytes: number | null;
  sha256: string | null;
  stored_path: string | null;
  created_at: string | null;
}

function ArtifactTab({ trial }: { trial: ArtifactTabTrial }) {
  return (
    <div className="card">
      <dl className="grid grid-cols-1 sm:grid-cols-3 gap-y-3 gap-x-6 text-sm">
        <div>
          <dt className="text-xs text-ink-500 uppercase tracking-wide">Size</dt>
          <dd className="mt-1 font-mono text-ink-900 dark:text-ink-50">
            {formatBytes(trial.size_bytes)}
          </dd>
        </div>
        <div className="sm:col-span-2">
          <dt className="text-xs text-ink-500 uppercase tracking-wide">
            SHA-256
          </dt>
          <dd className="mt-1 font-mono text-xs text-ink-700 dark:text-ink-300 break-all">
            {trial.sha256 ?? '—'}
          </dd>
        </div>
        <div className="sm:col-span-3">
          <dt className="text-xs text-ink-500 uppercase tracking-wide">
            Stored path
          </dt>
          <dd className="mt-1 font-mono text-xs text-ink-700 dark:text-ink-300 break-all">
            {trial.stored_path ?? '—'}
          </dd>
        </div>
        {trial.created_at && (
          <div className="sm:col-span-3">
            <dt className="text-xs text-ink-500 uppercase tracking-wide">
              Created
            </dt>
            <dd className="mt-1 text-xs text-ink-700 dark:text-ink-300">
              {new Date(trial.created_at).toLocaleString()}
            </dd>
          </div>
        )}
      </dl>
    </div>
  );
}

// ─── Lineage breadcrumb ──────────────────────────────────────────

function Lineage({
  run,
  runId,
  trialModelId,
}: {
  run:
    | {
        workspace_id?: string | null;
        project_id?: string | null;
        experiment_id?: string | null;
      }
    | undefined;
  runId: string;
  trialModelId: string;
}) {
  const wsId = run?.workspace_id ?? null;
  const projId = run?.project_id ?? null;
  const expId = run?.experiment_id ?? null;

  const ws = useQuery({
    queryKey: ['workspaces', wsId],
    queryFn: () => workspacesApi.get(wsId!),
    enabled: !!wsId,
    staleTime: 5 * 60 * 1000,
  });
  const proj = useQuery({
    queryKey: ['projects', wsId, projId],
    queryFn: () => projectsApi.get(wsId!, projId!),
    enabled: !!wsId && !!projId,
    staleTime: 5 * 60 * 1000,
  });
  const exp = useQuery({
    queryKey: ['experiments', projId, expId],
    queryFn: () => experimentsApi.get(projId!, expId!),
    enabled: !!projId && !!expId,
    staleTime: 5 * 60 * 1000,
  });

  return (
    <nav className="text-xs text-ink-500 mb-2 flex items-center flex-wrap gap-y-0.5">
      <Link to="/" className="hover:text-ink-900 dark:hover:text-ink-50">
        Workspaces
      </Link>
      {wsId && (
        <>
          <span className="mx-1.5 text-ink-300">/</span>
          <Link
            to={`/workspaces/${wsId}`}
            className="hover:text-ink-900 dark:hover:text-ink-50"
          >
            {ws.data?.name ?? wsId.slice(0, 8)}
          </Link>
        </>
      )}
      {wsId && projId && (
        <>
          <span className="mx-1.5 text-ink-300">/</span>
          <Link
            to={`/workspaces/${wsId}/projects/${projId}`}
            className="hover:text-ink-900 dark:hover:text-ink-50"
          >
            {proj.data?.name ?? projId.slice(0, 8)}
          </Link>
        </>
      )}
      {wsId && projId && expId && (
        <>
          <span className="mx-1.5 text-ink-300">/</span>
          <Link
            to={`/workspaces/${wsId}/projects/${projId}/experiments/${expId}`}
            className="hover:text-ink-900 dark:hover:text-ink-50"
          >
            {exp.data?.name ?? expId.slice(0, 8)}
          </Link>
        </>
      )}
      <span className="mx-1.5 text-ink-300">/</span>
      <Link
        to={`/runs/${runId}`}
        className="hover:text-ink-900 dark:hover:text-ink-50 font-mono"
      >
        Run · {runId.slice(0, 8)}
      </Link>
      <span className="mx-1.5 text-ink-300">/</span>
      <span className="text-ink-700 dark:text-ink-300 font-mono">
        Trial · {trialModelId}
      </span>
    </nav>
  );
}

// ─── Promote history badge ──────────────────────────────────────

function PromoteHistoryCard({
  pipelineId,
  workspaceId,
}: {
  pipelineId: string;
  workspaceId: string | null;
}) {
  const pipe = useQuery({
    queryKey: ['pipelines', pipelineId],
    queryFn: () => pipelinesApi.get(pipelineId),
    enabled: !!pipelineId,
  });
  const [deployOpen, setDeployOpen] = useState(false);

  if (pipe.isPending) {
    return <div className="card text-sm text-ink-500">Looking up pipeline…</div>;
  }
  if (!pipe.data) {
    return null;
  }
  const p = pipe.data;
  return (
    <>
      <div className="card flex items-center justify-between gap-4 flex-wrap">
        <div>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="pill-accent">Promoted</span>
            <span className="text-sm font-semibold text-ink-900 dark:text-ink-50">
              {p.name}
            </span>
            <span className="text-xs text-ink-500 font-mono">v{p.version}</span>
          </div>
          {p.description && (
            <p className="mt-1 text-xs text-ink-500">{p.description}</p>
          )}
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {workspaceId && p.registered_model_id && (
            <Link
              to={`/workspaces/${workspaceId}/models/${p.registered_model_id}`}
              className="btn-ghost text-xs"
            >
              Open in registry →
            </Link>
          )}
          <button
            type="button"
            className="btn-primary text-sm"
            onClick={() => setDeployOpen(true)}
          >
            Deploy
          </button>
        </div>
      </div>
      <DeployFromPipelineDialog
        open={deployOpen}
        onClose={() => setDeployOpen(false)}
        pipelineId={p.id}
        pipelineName={p.name}
      />
    </>
  );
}

// ─── Notes editor ───────────────────────────────────────────────

function NotesCard({
  runId,
  trialId,
  initialNotes,
}: {
  runId: string;
  trialId: string;
  initialNotes: string | null;
}) {
  const qc = useQueryClient();
  const [value, setValue] = useState(initialNotes ?? '');
  const [dirty, setDirty] = useState(false);
  const [saved, setSaved] = useState(false);

  // Reset when the upstream trial query changes (e.g. switching trials).
  useEffect(() => {
    setValue(initialNotes ?? '');
    setDirty(false);
  }, [initialNotes]);

  const save = useMutation({
    mutationFn: () =>
      runsApi.trialPatch(runId, trialId, { notes: value || null }),
    onSuccess: () => {
      setDirty(false);
      setSaved(true);
      window.setTimeout(() => setSaved(false), 1500);
      qc.invalidateQueries({ queryKey: ['runs', runId, 'trials', trialId] });
    },
  });

  return (
    <div className="card">
      <textarea
        className="input"
        rows={4}
        placeholder="Add notes about this candidate — what worked, what to revisit, why you'd promote it…"
        value={value}
        onChange={(e) => {
          setValue(e.target.value);
          setDirty(true);
        }}
      />
      <div className="mt-2 flex items-center justify-between gap-2">
        <p className="text-[11px] text-ink-400">
          Markdown allowed. Saved to the trial row.
        </p>
        <div className="flex items-center gap-2">
          {save.error && (
            <span className="text-xs text-danger-600">
              {errorMessage(save.error)}
            </span>
          )}
          {saved && !dirty && (
            <span className="text-xs text-success-600">Saved.</span>
          )}
          <button
            type="button"
            className="btn-secondary text-sm"
            onClick={() => save.mutate()}
            disabled={!dirty || save.isPending}
          >
            {save.isPending ? 'Saving…' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Model card export ──────────────────────────────────────────

interface ExportTrial {
  model_id: string;
  rank: number;
  is_best: boolean;
  metrics: Record<string, number | string>;
  params: Record<string, unknown>;
  pipeline_steps?: Array<{
    name: string;
    class: string;
    module: string;
    is_estimator: boolean;
  }>;
  run_snapshot?: Record<string, unknown>;
  sha256: string | null;
  size_bytes: number | null;
  notes: string | null;
}

function buildModelCardMarkdown(t: ExportTrial, modelName: string): string {
  const lines: string[] = [];
  lines.push(`# Model card — ${modelName}`);
  lines.push('');
  lines.push(`- **Model id:** \`${t.model_id}\``);
  lines.push(`- **Rank:** #${t.rank}${t.is_best ? ' (best)' : ''}`);
  if (t.sha256) lines.push(`- **SHA-256:** \`${t.sha256}\``);
  if (t.size_bytes != null) lines.push(`- **Pipeline size:** ${t.size_bytes} bytes`);
  lines.push('');
  if (t.run_snapshot && Object.keys(t.run_snapshot).length > 0) {
    lines.push('## Run context');
    lines.push('');
    for (const { key, label } of SNAPSHOT_FIELDS) {
      const v = t.run_snapshot[key];
      if (v == null || v === '') continue;
      lines.push(`- **${label}:** ${formatParam(v)}`);
    }
    lines.push('');
  }
  lines.push('## Metrics');
  lines.push('');
  lines.push('| Metric | Value |');
  lines.push('|---|---|');
  for (const [k, v] of Object.entries(t.metrics)) {
    if (typeof v === 'number') lines.push(`| ${k} | ${v.toFixed(4)} |`);
  }
  lines.push('');
  if (t.pipeline_steps && t.pipeline_steps.length > 0) {
    lines.push('## Pipeline');
    lines.push('');
    for (const s of t.pipeline_steps) {
      lines.push(
        `- \`${s.name}\` — ${s.class}${s.is_estimator ? ' _(estimator)_' : ''} ` +
          `from \`${s.module}\``,
      );
    }
    lines.push('');
  }
  if (t.params && Object.keys(t.params).length > 0) {
    lines.push('## Estimator hyperparameters');
    lines.push('');
    lines.push('| Parameter | Value |');
    lines.push('|---|---|');
    for (const [k, v] of Object.entries(t.params).sort(([a], [b]) =>
      a.localeCompare(b),
    )) {
      lines.push(`| \`${k}\` | \`${formatParam(v)}\` |`);
    }
    lines.push('');
  }
  if (t.notes) {
    lines.push('## Notes');
    lines.push('');
    lines.push(t.notes);
    lines.push('');
  }
  return lines.join('\n');
}

function ModelCardExportButton({
  trial,
  modelName,
}: {
  trial: ExportTrial;
  modelName: string;
}) {
  const handleExport = () => {
    const md = buildModelCardMarkdown(trial, modelName);
    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const safe = (modelName || 'model').replace(/[^a-zA-Z0-9._-]/g, '_');
    a.href = url;
    a.download = `${safe}-model-card.md`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  };
  return (
    <button
      type="button"
      className="btn-secondary"
      onClick={handleExport}
      title="Download a markdown model card with metrics, params, and pipeline structure"
    >
      Export model card
    </button>
  );
}

// ─── Compare picker ─────────────────────────────────────────────

interface TrialListItem {
  id: string;
  model_id: string;
  rank: number | null;
  is_best: boolean;
}

function CompareDialog({
  open,
  onClose,
  runId,
  currentTrialId,
  items,
}: {
  open: boolean;
  onClose: () => void;
  runId: string;
  currentTrialId: string;
  items: TrialListItem[];
}) {
  const others = items.filter((x) => x.id !== currentTrialId);
  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="Compare with another trial"
      description="Pick a candidate and we'll open a side-by-side comparison view."
    >
      {others.length === 0 ? (
        <p className="text-sm text-ink-500">
          This run only has one trial — nothing to compare against.
        </p>
      ) : (
        <ul className="divide-y divide-ink-200 dark:divide-ink-800 max-h-80 overflow-y-auto">
          {others.map((x) => (
            <li key={x.id}>
              <Link
                to={`/runs/${runId}/compare?a=${currentTrialId}&b=${x.id}`}
                onClick={onClose}
                className="flex items-center justify-between gap-3 py-2 px-1 hover:bg-ink-50 dark:hover:bg-ink-950/40 rounded"
              >
                <div>
                  <div className="text-sm text-ink-900 dark:text-ink-50">
                    {x.model_id}
                  </div>
                  <div className="text-xs text-ink-500">
                    Rank #{x.rank}
                    {x.is_best && ' · Best'}
                  </div>
                </div>
                <span className="text-xs text-accent-600">→</span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </Dialog>
  );
}

// ─── Predict tab ────────────────────────────────────────────────

function PredictTab({
  runId,
  trialId,
  snapshot,
  inputSchema,
}: {
  runId: string;
  trialId: string;
  snapshot: Record<string, unknown>;
  inputSchema: {
    columns: Array<{ name: string; dtype: string }>;
    sample_row: Record<string, unknown>;
    target: string | null;
  } | null;
}) {
  const target =
    inputSchema?.target ?? ((snapshot.target as string | undefined) ?? null);

  // Seed the textarea with a real holdout row so the user can press
  // "Run prediction" with no edits and see immediate results.
  const seedJson = useMemo(() => {
    if (inputSchema?.sample_row && Object.keys(inputSchema.sample_row).length > 0) {
      return JSON.stringify([inputSchema.sample_row], null, 2);
    }
    if (inputSchema?.columns?.length) {
      const empty: Record<string, unknown> = {};
      for (const c of inputSchema.columns) empty[c.name] = '';
      return JSON.stringify([empty], null, 2);
    }
    return JSON.stringify([{ feature_a: 0, feature_b: 0 }], null, 2);
  }, [inputSchema]);

  const [raw, setRaw] = useState<string>(seedJson);

  // Re-seed if the schema arrives after first render.
  useEffect(() => {
    setRaw(seedJson);
  }, [seedJson]);

  const predict = useMutation({
    mutationFn: async () => {
      let rows: Record<string, unknown>[];
      try {
        const parsed = JSON.parse(raw);
        rows = Array.isArray(parsed) ? parsed : [parsed];
      } catch (e) {
        throw new Error(
          `Invalid JSON: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
      return runsApi.trialPredict(runId, trialId, { rows });
    },
  });

  const result = predict.data;
  const colCount = inputSchema?.columns.length ?? 0;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <div className="card">
        <div className="flex items-center justify-between mb-2 gap-2 flex-wrap">
          <h3 className="text-sm font-semibold text-ink-900 dark:text-ink-50">
            Input rows
          </h3>
          <p className="text-[11px] text-ink-500">
            {colCount > 0 && <>{colCount} feature{colCount === 1 ? '' : 's'}</>}
            {target && (
              <>
                {colCount > 0 && ' · '}target <code className="font-mono">{target}</code> excluded
              </>
            )}
          </p>
        </div>
        {inputSchema?.columns && inputSchema.columns.length > 0 && (
          <details className="mb-2">
            <summary className="text-[11px] text-ink-500 cursor-pointer hover:text-ink-700 dark:hover:text-ink-300">
              Schema
            </summary>
            <div className="mt-1 flex flex-wrap gap-1">
              {inputSchema.columns.map((c) => (
                <span
                  key={c.name}
                  className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-ink-50 dark:bg-ink-950/40 border border-ink-200 dark:border-ink-800 text-[11px] font-mono"
                >
                  <span className="text-ink-700 dark:text-ink-300">{c.name}</span>
                  <span className="text-ink-400">{c.dtype}</span>
                </span>
              ))}
            </div>
          </details>
        )}
        <textarea
          className="input font-mono text-xs"
          rows={14}
          value={raw}
          onChange={(e) => setRaw(e.target.value)}
          spellCheck={false}
        />
        <div className="mt-3 flex items-center justify-between gap-2">
          <button
            type="button"
            className="btn-ghost text-xs"
            onClick={() => setRaw(seedJson)}
          >
            Reset to sample row
          </button>
          <button
            type="button"
            className="btn-primary"
            onClick={() => predict.mutate()}
            disabled={predict.isPending}
          >
            {predict.isPending ? 'Predicting…' : 'Run prediction'}
          </button>
        </div>
        {predict.error && (
          <p className="mt-2 text-xs text-danger-600">
            {errorMessage(predict.error)}
          </p>
        )}
      </div>

      <div className="card">
        <h3 className="text-sm font-semibold text-ink-900 dark:text-ink-50 mb-2">
          Predictions
        </h3>
        {!result ? (
          <p className="text-xs text-ink-500">
            No predictions yet — the input is pre-seeded with a real holdout
            row, so you can press "Run prediction" and see what this model
            outputs immediately.
          </p>
        ) : (
          <PredictResultTable result={result} />
        )}
      </div>
    </div>
  );
}

function PredictResultTable({
  result,
}: {
  result: {
    predictions: (string | number | boolean | null)[];
    probabilities?: number[][];
    classes?: (string | number | boolean | null)[];
    scores?: number[];
  };
}) {
  const hasProba = Array.isArray(result.probabilities) && result.probabilities.length > 0;
  const classes = result.classes ?? [];
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="text-ink-500 border-b border-ink-200 dark:border-ink-800">
          <tr>
            <th className="px-2 py-1.5 text-left font-medium">#</th>
            <th className="px-2 py-1.5 text-left font-medium">Prediction</th>
            {hasProba &&
              classes.map((c, i) => (
                <th key={i} className="px-2 py-1.5 text-right font-medium tabular-nums">
                  P({String(c)})
                </th>
              ))}
            {!hasProba && result.scores && (
              <th className="px-2 py-1.5 text-right font-medium tabular-nums">
                Score
              </th>
            )}
          </tr>
        </thead>
        <tbody>
          {result.predictions.map((p, i) => (
            <tr
              key={i}
              className="border-t border-ink-200 dark:border-ink-800"
            >
              <td className="px-2 py-1.5 text-ink-500 tabular-nums">{i + 1}</td>
              <td className="px-2 py-1.5 font-mono text-xs text-ink-900 dark:text-ink-50">
                {String(p)}
              </td>
              {hasProba &&
                result.probabilities![i]?.map((pp, j) => (
                  <td
                    key={j}
                    className="px-2 py-1.5 text-right font-mono tabular-nums text-ink-700 dark:text-ink-300"
                  >
                    {pp.toFixed(3)}
                  </td>
                ))}
              {!hasProba && result.scores && (
                <td className="px-2 py-1.5 text-right font-mono tabular-nums text-ink-700 dark:text-ink-300">
                  {result.scores[i]?.toFixed(4) ?? '—'}
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── Validation tab (CV + Cohorts) ─────────────────────────────

function ValidationTab({
  runId,
  trialId,
  task,
}: {
  runId: string;
  trialId: string;
  task: string | null;
}) {
  const [sub, setSub] = useState<'cv' | 'cohorts'>('cv');
  return (
    <div>
      <div className="inline-flex rounded-md border border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 p-0.5 text-xs mb-4">
        {(['cv', 'cohorts'] as const).map((k) => (
          <button
            key={k}
            type="button"
            onClick={() => setSub(k)}
            className={`px-3 py-1.5 rounded font-medium transition-colors ${
              sub === k
                ? 'bg-ink-900 text-white dark:bg-white dark:text-ink-900'
                : 'text-ink-600 hover:text-ink-900 dark:text-ink-400 dark:hover:text-ink-50'
            }`}
          >
            {k === 'cv' ? 'Cross-validation' : 'Cohort metrics'}
          </button>
        ))}
      </div>
      {sub === 'cv' && <CVPanel runId={runId} trialId={trialId} task={task} />}
      {sub === 'cohorts' && <CohortPanel runId={runId} trialId={trialId} />}
    </div>
  );
}

function CVPanel({
  runId,
  trialId,
  task,
}: {
  runId: string;
  trialId: string;
  task: string | null;
}) {
  const [folds, setFolds] = useState(5);
  const cv = useQuery({
    queryKey: ['runs', runId, 'trials', trialId, 'cv', folds],
    queryFn: () => runsApi.trialCv(runId, trialId, folds),
    enabled: !!runId && !!trialId,
    retry: false,
  });

  if (task && task !== 'classification' && task !== 'regression') {
    return (
      <p className="text-sm text-ink-500">
        Per-fold CV not supported for task <code>{task}</code>.
      </p>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <p className="text-xs text-ink-500">
          Recomputed on demand from the trial pickle + run holdout. Stratified
          for classification, plain K-Fold for regression. Uses the run's seed
          so the splits match.
        </p>
        <label className="text-xs text-ink-500 inline-flex items-center gap-2">
          Folds
          <select
            className="input py-1 text-sm w-auto"
            value={folds}
            onChange={(e) => setFolds(Number(e.target.value))}
          >
            {[3, 5, 7, 10].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
      </div>
      {cv.isPending && (
        <div className="card text-sm text-ink-500">Running cross-validation…</div>
      )}
      {cv.error && (
        <div className="card text-sm text-danger-600">
          {errorMessage(cv.error)}
        </div>
      )}
      {cv.data && (
        <>
          <div className="card overflow-x-auto p-0">
            <table className="w-full text-sm">
              <thead className="bg-white text-ink-500 dark:bg-ink-900 border-b border-ink-200 dark:border-ink-800">
                <tr>
                  <th className="px-3 py-2 text-left font-medium">Fold</th>
                  {cv.data.scoring.map((m) => (
                    <th
                      key={m}
                      className="px-3 py-2 text-right font-medium tabular-nums"
                    >
                      {m}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {cv.data.rows.map((row) => (
                  <tr
                    key={row.fold}
                    className="border-t border-ink-200 dark:border-ink-800"
                  >
                    <td className="px-3 py-2 font-mono text-xs text-ink-700 dark:text-ink-300">
                      #{row.fold}
                    </td>
                    {cv.data.scoring.map((m) => (
                      <td
                        key={m}
                        className="px-3 py-2 text-right font-mono tabular-nums text-ink-900 dark:text-ink-50"
                      >
                        {typeof row[m] === 'number' ? row[m].toFixed(4) : '—'}
                      </td>
                    ))}
                  </tr>
                ))}
                <tr className="border-t border-ink-200 dark:border-ink-800 bg-ink-50/50 dark:bg-ink-950/30">
                  <td className="px-3 py-2 text-xs font-semibold text-ink-700 dark:text-ink-300">
                    mean ± std
                  </td>
                  {cv.data.scoring.map((m) => {
                    const s = cv.data.summary[m];
                    return (
                      <td
                        key={m}
                        className="px-3 py-2 text-right font-mono tabular-nums text-ink-900 dark:text-ink-50"
                      >
                        {s ? `${s.mean.toFixed(4)} ± ${s.std.toFixed(4)}` : '—'}
                      </td>
                    );
                  })}
                </tr>
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

function CohortPanel({ runId, trialId }: { runId: string; trialId: string }) {
  const [column, setColumn] = useState<string>('');
  // First fetch with empty column will 400 — only enable when chosen.
  const cohorts = useQuery({
    queryKey: ['runs', runId, 'trials', trialId, 'cohorts', column],
    queryFn: () => runsApi.trialCohorts(runId, trialId, column),
    enabled: !!column,
    retry: false,
  });

  // Discovery probe: empty column → backend returns just available_columns.
  const probe = useQuery({
    queryKey: ['runs', runId, 'trials', trialId, 'cohorts', '__probe__'],
    queryFn: () => runsApi.trialCohorts(runId, trialId, ''),
    enabled: !!runId && !!trialId,
    retry: false,
  });
  const availableColumns: string[] = (probe.data?.available_columns ??
    cohorts.data?.available_columns ??
    []) as string[];

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 flex-wrap">
        <label className="text-xs text-ink-500 inline-flex items-center gap-2">
          Slice by column
          <select
            className="input py-1 text-sm w-auto min-w-[12rem]"
            value={column}
            onChange={(e) => setColumn(e.target.value)}
          >
            <option value="">Pick a column…</option>
            {availableColumns.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>
        <p className="text-xs text-ink-500">
          Numeric columns are bucketed into quartiles automatically.
        </p>
      </div>
      {cohorts.isPending && column && (
        <div className="card text-sm text-ink-500">Slicing…</div>
      )}
      {cohorts.error && (
        <div className="card text-sm text-danger-600">
          {errorMessage(cohorts.error)}
        </div>
      )}
      {cohorts.data && cohorts.data.rows.length > 0 && (
        <div className="card overflow-x-auto p-0">
          <table className="w-full text-sm">
            <thead className="bg-white text-ink-500 dark:bg-ink-900 border-b border-ink-200 dark:border-ink-800">
              <tr>
                <th className="px-3 py-2 text-left font-medium">{column}</th>
                <th className="px-3 py-2 text-right font-medium tabular-nums">
                  n
                </th>
                {Object.keys(cohorts.data.rows[0].metrics).map((m) => (
                  <th
                    key={m}
                    className="px-3 py-2 text-right font-medium tabular-nums"
                  >
                    {m}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {cohorts.data.rows.map((row) => (
                <tr
                  key={row.value}
                  className="border-t border-ink-200 dark:border-ink-800"
                >
                  <td className="px-3 py-2 font-mono text-xs text-ink-700 dark:text-ink-300">
                    {row.value}
                  </td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums text-ink-900 dark:text-ink-50">
                    {row.n}
                  </td>
                  {Object.entries(row.metrics).map(([k, v]) => (
                    <td
                      key={k}
                      className="px-3 py-2 text-right font-mono tabular-nums text-ink-700 dark:text-ink-300"
                    >
                      {v.toFixed(4)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {cohorts.data && cohorts.data.rows.length === 0 && (
        <p className="text-sm text-ink-500">No cohorts produced for this column.</p>
      )}
    </div>
  );
}

// ─── Tune optimization history ───────────────────────────────────
//
// The orchestrator stashes the search history on a tuned trial's params
// under `_cv_history` (mean_test_score + std_test_score + params per
// iteration) and the best params dict under `_best_params`. Both are
// optional — pre-session-27 tuned trials won't have them.

interface CvHistory {
  mean_test_score: (number | null)[];
  std_test_score: (number | null)[];
  params: Array<Record<string, unknown>>;
}

function TuneHistorySection({ params }: { params: Record<string, unknown> }) {
  const history = params._cv_history as CvHistory | undefined;
  const bestParams = params._best_params as
    | Record<string, unknown>
    | undefined;

  if (!history && !bestParams) return null;

  // Compute running-best for the orange overlay — what the search
  // *would* have returned if it stopped at each iteration.
  const xs: number[] = [];
  const meanY: number[] = [];
  const stdHi: number[] = [];
  const stdLo: number[] = [];
  const runningBest: number[] = [];
  if (history?.mean_test_score) {
    let best = Number.NEGATIVE_INFINITY;
    history.mean_test_score.forEach((v, i) => {
      if (typeof v !== 'number') return;
      xs.push(i + 1);
      meanY.push(v);
      const s = history.std_test_score?.[i];
      const std = typeof s === 'number' ? s : 0;
      stdHi.push(v + std);
      stdLo.push(v - std);
      if (v > best) best = v;
      runningBest.push(best);
    });
  }

  return (
    <section>
      <h3 className="h-section mb-3">Tuning history</h3>
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-4">
        <div className="card p-3">
          {xs.length > 0 ? (
            <Plot
              data={[
                // Confidence band (std)
                {
                  x: [...xs, ...[...xs].reverse()],
                  y: [...stdHi, ...[...stdLo].reverse()],
                  fill: 'toself',
                  fillcolor: 'rgba(56, 189, 248, 0.12)',
                  line: { color: 'transparent' },
                  hoverinfo: 'skip',
                  showlegend: false,
                  type: 'scatter',
                },
                // Per-iteration mean score (dots + line)
                {
                  x: xs,
                  y: meanY,
                  mode: 'lines+markers',
                  name: 'CV score',
                  line: { color: 'rgb(56, 189, 248)', width: 2 },
                  marker: {
                    color: 'rgb(56, 189, 248)',
                    size: 6,
                  },
                  hovertemplate: 'iter %{x}<br>score %{y:.4f}<extra></extra>',
                },
                // Running best — the orange "if we stopped here" line
                {
                  x: xs,
                  y: runningBest,
                  mode: 'lines',
                  name: 'Best so far',
                  line: {
                    color: 'rgb(249, 115, 22)',
                    width: 2,
                    dash: 'dash',
                  },
                  hovertemplate:
                    'best by iter %{x}<br>%{y:.4f}<extra></extra>',
                },
              ]}
              layout={{
                height: 280,
                margin: { l: 44, r: 12, t: 8, b: 36 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {
                  family: 'Inter, system-ui, sans-serif',
                  size: 11,
                  color: '#52525b',
                },
                xaxis: {
                  title: { text: 'Iteration', standoff: 6 },
                  gridcolor: '#e4e4e7',
                  zerolinecolor: '#e4e4e7',
                },
                yaxis: {
                  title: { text: 'CV score', standoff: 6 },
                  gridcolor: '#e4e4e7',
                  zerolinecolor: '#e4e4e7',
                  tickformat: '.3f',
                },
                legend: {
                  orientation: 'h',
                  y: 1.1,
                  x: 0,
                },
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: '100%' }}
              useResizeHandler
            />
          ) : (
            <p className="text-sm text-ink-500 px-2 py-8 text-center">
              No per-iteration CV history was stashed for this trial.
            </p>
          )}
        </div>
        <aside className="card">
          <h4 className="text-sm font-semibold text-ink-900 dark:text-ink-50 mb-2">
            Best hyperparameters
          </h4>
          {bestParams && Object.keys(bestParams).length > 0 ? (
            <dl className="space-y-1.5 text-xs">
              {Object.entries(bestParams)
                .sort(([a], [b]) => a.localeCompare(b))
                .map(([k, v]) => (
                  <div
                    key={k}
                    className="flex items-baseline justify-between gap-3 border-b border-ink-100 dark:border-ink-800/60 pb-1 last:border-0"
                  >
                    <dt className="font-mono text-ink-500">{k}</dt>
                    <dd className="font-mono text-ink-900 dark:text-ink-50 truncate">
                      {String(v)}
                    </dd>
                  </div>
                ))}
            </dl>
          ) : (
            <p className="text-xs text-ink-500">
              Best params unavailable for this trial.
            </p>
          )}
        </aside>
      </div>
    </section>
  );
}

// ─── Blend / Stack contribution chart ────────────────────────────
//
// For ensemble trials, render a side-by-side comparison of every parent
// trial's primary metric against the new ensemble's. Helps users see
// whether the combo actually beat its bases (the whole point of
// ensembling) without flipping back to the leaderboard.

function EnsembleContributionSection({
  runId,
  trialId: _trialId,
  parentTrialIds,
  ownMetrics,
  kind,
}: {
  runId: string;
  trialId: string;
  parentTrialIds: string[];
  ownMetrics: Record<string, number | string>;
  kind: string;
}) {
  const trialsList = useQuery({
    queryKey: ['runs', runId, 'trials'],
    queryFn: () => runsApi.trials(runId),
    enabled: !!runId,
    staleTime: 30_000,
  });

  const parents = useMemo(() => {
    const byId: Record<string, { id: string; model_id: string; metrics: Record<string, number | string> }> = {};
    for (const t of trialsList.data?.items ?? []) byId[t.id] = t;
    return parentTrialIds.map((id) => byId[id]).filter(Boolean);
  }, [trialsList.data, parentTrialIds]);

  if (parents.length === 0) return null;

  // Use the first numeric metric on the ensemble as the comparison axis.
  const metricKey = Object.entries(ownMetrics).find(
    ([, v]) => typeof v === 'number',
  )?.[0];
  if (!metricKey) return null;

  const ownScore = ownMetrics[metricKey] as number;
  const rows = parents.map((p) => ({
    label: p.model_id,
    value:
      typeof p.metrics[metricKey] === 'number'
        ? (p.metrics[metricKey] as number)
        : 0,
  }));
  rows.push({ label: kind === 'blended' ? 'Blend' : 'Stack', value: ownScore });

  // Highlight the ensemble row + flag whether it beat the best base.
  const bestBase = parents.reduce<number>((acc, p) => {
    const v = p.metrics[metricKey];
    return typeof v === 'number' && v > acc ? v : acc;
  }, Number.NEGATIVE_INFINITY);
  const winning = ownScore >= bestBase;

  return (
    <section>
      <div className="flex items-baseline justify-between mb-3 gap-3 flex-wrap">
        <h3 className="h-section">
          {kind === 'blended' ? 'Blend contribution' : 'Stack contribution'}
        </h3>
        <span
          className={
            winning
              ? 'pill-success text-[10px] uppercase tracking-wide'
              : 'pill-warn text-[10px] uppercase tracking-wide'
          }
        >
          {winning
            ? `Beat best base by ${(ownScore - bestBase).toFixed(4)}`
            : `Trails best base by ${(bestBase - ownScore).toFixed(4)}`}
        </span>
      </div>
      <div className="card p-3">
        <Plot
          data={[
            {
              type: 'bar',
              orientation: 'h',
              x: rows.map((r) => r.value),
              y: rows.map((r) => r.label),
              marker: {
                color: rows.map((_, i) =>
                  i === rows.length - 1
                    ? 'rgb(34, 197, 94)'
                    : 'rgb(168, 85, 247)',
                ),
              },
              text: rows.map((r) => r.value.toFixed(4)),
              textposition: 'outside',
              cliponaxis: false,
              hovertemplate: '%{y}<br>%{x:.4f}<extra></extra>',
            },
          ]}
          layout={{
            height: Math.max(220, rows.length * 36 + 60),
            margin: { l: 8, r: 60, t: 12, b: 36 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {
              family: 'Inter, system-ui, sans-serif',
              size: 12,
              color: '#52525b',
            },
            xaxis: {
              title: { text: metricKey, standoff: 6 },
              gridcolor: '#e4e4e7',
              zerolinecolor: '#e4e4e7',
              tickformat: '.3f',
              automargin: true,
              range: (() => {
                const v = rows.map((r) => r.value);
                const min = Math.min(...v, 0);
                const max = Math.max(...v, 1);
                const span = max - min || Math.abs(max) || 1;
                return [min - span * 0.02, max + span * 0.12];
              })(),
            },
            yaxis: { autorange: 'reversed', automargin: true },
            bargap: 0.35,
            showlegend: false,
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%' }}
          useResizeHandler
        />
      </div>
    </section>
  );
}

// ─── Trial kind chip + lineage ───────────────────────────────────

const TRIAL_KIND_CHIP: Record<string, string> = {
  compare: 'pill-neutral',
  tuned: 'pill-accent',
  ensembled:
    'pill bg-violet-50 text-violet-700 dark:bg-violet-500/15 dark:text-violet-300',
  blended:
    'pill bg-cyan-50 text-cyan-700 dark:bg-cyan-500/15 dark:text-cyan-300',
  stacked:
    'pill bg-fuchsia-50 text-fuchsia-700 dark:bg-fuchsia-500/15 dark:text-fuchsia-300',
  manual: 'pill-neutral',
};

function ParentLineage({
  runId,
  parentTrialIds,
  kind,
}: {
  runId: string;
  parentTrialIds: string[];
  kind: string | undefined;
}) {
  // Resolve each parent's display name lazily. The trials list is already
  // cached on the run page so this is usually a cache hit.
  const trialsList = useQuery({
    queryKey: ['runs', runId, 'trials'],
    queryFn: () => runsApi.trials(runId),
    enabled: !!runId,
    staleTime: 30_000,
  });
  const byId = useMemo(() => {
    const m: Record<string, { id: string; model_id: string }> = {};
    for (const t of trialsList.data?.items ?? []) m[t.id] = t;
    return m;
  }, [trialsList.data]);

  const verb =
    kind === 'tuned'
      ? 'Tuned from'
      : kind === 'ensembled'
        ? 'Ensembled from'
        : kind === 'blended'
          ? 'Blended from'
          : kind === 'stacked'
            ? 'Stacked from'
            : 'Derived from';
  return (
    <p className="mt-2 text-xs text-ink-500 flex flex-wrap items-center gap-1.5">
      <span>{verb}</span>
      {parentTrialIds.map((pid, i) => {
        const p = byId[pid];
        const label = p?.model_id ?? pid.slice(0, 8);
        return (
          <span key={pid} className="inline-flex items-center gap-1">
            <Link
              to={`/runs/${runId}/trials/${pid}`}
              className="px-1.5 py-0.5 rounded bg-ink-100 dark:bg-ink-800 font-mono text-[11px] text-ink-700 dark:text-ink-300 hover:text-accent-600 hover:underline"
            >
              {label}
            </Link>
            {i < parentTrialIds.length - 1 && (
              <span className="text-ink-300">·</span>
            )}
          </span>
        );
      })}
    </p>
  );
}
