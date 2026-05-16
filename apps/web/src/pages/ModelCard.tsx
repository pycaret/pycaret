/**
 * Model Card — per-model deep-dive at /runs/:runId/model-card.
 *
 * Layout:
 *   [breadcrumb] · h1 · description
 *   [KPI strip from leaderboard]
 *   [Tabs: Diagnostics | Explainability | Curves | Raw]
 *   [grid of <PlotlyFigure> cards]
 *
 * Pre-flight: the Plot endpoints all 404 if the run hasn't been
 * "promoted" yet — i.e., the trained pipeline isn't yet a Pipeline
 * row in the workspace's registry. We detect that via a probe on
 * the first plot kind. If the probe 404s with the well-known
 * "No saved pipeline" detail, we replace the entire plots area
 * with a single empty state + "Promote this run" button (modal).
 */
import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';

import { plotsApi, runsApi } from '../api/endpoints';
import { errorMessage } from '../api/client';
import { BackButton } from '../components/BackButton';
import { PlotlyFigure } from '../components/PlotlyFigure';
import { Dialog } from '../components/Dialog';
import type { PlotEnvelope, PlotRegistry } from '../api/types';

type Tab = 'diagnostics' | 'explainability' | 'curves' | 'raw';

const TAB_DISPATCH: Record<string, Record<Tab, string[]>> = {
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
  forecast: 'Forecast vs. actual',
  decomposition: 'Decomposition',
  acf: 'Autocorrelation',
  pacf: 'Partial autocorrelation',
  residual_diagnostics: 'Residual diagnostics',
};

function isNoPipelineError(err: unknown): boolean {
  const msg = errorMessage(err).toLowerCase();
  return (
    msg.includes('no saved pipeline') ||
    msg.includes('promote') ||
    msg.includes('not yet promoted')
  );
}

// ─── Plot card ────────────────────────────────────────────────────

function PlotCard({ runId, kind }: { runId: string; kind: string }) {
  const q = useQuery<PlotEnvelope, Error>({
    queryKey: ['runs', runId, 'plots', kind],
    queryFn: () => plotsApi.forRun(runId, kind),
    enabled: !!runId,
    staleTime: 60_000,
    retry: false,
  });

  return (
    <PlotlyFigure
      figure={q.data?.figure}
      loading={q.isLoading}
      error={q.error ?? undefined}
      onRetry={() => q.refetch()}
      title={PLOT_TITLES[kind] ?? kind}
    />
  );
}

// ─── KPI strip ────────────────────────────────────────────────────

function KpiStrip({ leaderboard }: { leaderboard: unknown }) {
  const tiles = useMemo<Array<{ label: string; value: string }>>(() => {
    const arr = Array.isArray(leaderboard) ? leaderboard : null;
    const top = arr && arr.length > 0 ? arr[0] : null;
    if (!top || typeof top !== 'object') return [];
    const out: Array<{ label: string; value: string }> = [];
    for (const [k, v] of Object.entries(top)) {
      if (k === 'Model') continue;
      if (typeof v === 'number') {
        out.push({ label: k, value: Number.isInteger(v) ? String(v) : v.toFixed(4) });
      }
      if (out.length >= 6) break;
    }
    return out;
  }, [leaderboard]);

  if (tiles.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 bg-white dark:bg-ink-900 px-4 py-3 text-sm text-ink-500">
        No metric summary yet — promote the run or finish training to populate.
      </div>
    );
  }

  return (
    <div
      className="grid gap-px rounded-xl overflow-hidden bg-ink-200 dark:bg-ink-800 border border-ink-200 dark:border-ink-800"
      style={{ gridTemplateColumns: `repeat(${tiles.length}, minmax(0, 1fr))` }}
    >
      {tiles.map((t) => (
        <div key={t.label} className="bg-white dark:bg-ink-900 px-4 py-3">
          <div className="text-xs font-medium text-ink-500 uppercase tracking-wider">
            {t.label}
          </div>
          <div className="mt-1 text-2xl font-semibold text-ink-900 dark:text-ink-50 tabular-nums">
            {t.value}
          </div>
        </div>
      ))}
    </div>
  );
}

// ─── Promote dialog ───────────────────────────────────────────────

function PromoteDialog({
  runId,
  open,
  onClose,
  onSuccess,
}: {
  runId: string;
  open: boolean;
  onClose: () => void;
  onSuccess: () => void;
}) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');

  useEffect(() => {
    if (!open) {
      setName('');
      setDescription('');
    }
  }, [open]);

  const promote = useMutation({
    mutationFn: () =>
      runsApi.promote(runId, {
        name: name.trim(),
        description: description.trim() || undefined,
      }),
    onSuccess: () => {
      onSuccess();
      onClose();
    },
  });

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="Promote this run"
      description="Save the trained pipeline into the workspace's pipeline registry. After promotion you can render diagnostics, deploy, and serve predictions."
    >
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (name.trim()) promote.mutate();
        }}
        className="space-y-4"
      >
        <div>
          <label className="field" htmlFor="promote-name">
            Pipeline name <span className="text-ink-400 font-normal">*</span>
          </label>
          <input
            id="promote-name"
            className="input"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. juice-baseline"
            autoFocus
            required
          />
        </div>
        <div>
          <label className="field" htmlFor="promote-desc">
            Description
          </label>
          <textarea
            id="promote-desc"
            className="input resize-none"
            rows={2}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Notes about this version, optional."
          />
        </div>
        {promote.error && <p className="error">{errorMessage(promote.error)}</p>}
        <div className="flex items-center justify-end gap-2 pt-2">
          <button type="button" onClick={onClose} className="btn-ghost">
            Cancel
          </button>
          <button
            type="submit"
            className="btn-primary"
            disabled={promote.isPending || !name.trim()}
          >
            {promote.isPending ? 'Promoting…' : 'Promote run'}
          </button>
        </div>
      </form>
    </Dialog>
  );
}

// ─── Empty state for "no pipeline" ────────────────────────────────

function NoPipelineEmptyState({
  onPromote,
  runId,
}: {
  onPromote: () => void;
  runId: string;
}) {
  return (
    <div className="rounded-xl bg-white dark:bg-ink-900 border border-dashed border-ink-300 dark:border-ink-700 px-6 py-14 text-center">
      <div className="mx-auto h-12 w-12 rounded-xl bg-accent-50 dark:bg-accent-500/15 text-accent-600 dark:text-accent-400 flex items-center justify-center mb-4">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
          <path d="M12 2v6 M5 9l7-7 7 7 M5 13v6a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-6" />
        </svg>
      </div>
      <h3 className="text-base font-semibold text-ink-900 dark:text-ink-50">
        This run hasn't been promoted yet
      </h3>
      <p className="mt-2 text-sm text-ink-500 max-w-xl mx-auto">
        The Model Card needs a saved pipeline. Promoting registers the fitted
        model into the workspace's pipeline registry — required for rendering
        diagnostics, deploying, and serving predictions.
      </p>
      <div className="mt-6 flex items-center justify-center gap-2">
        <button type="button" onClick={onPromote} className="btn-primary">
          Promote this run
        </button>
        <Link to={`/runs/${runId}`} className="btn-ghost">
          Back to run
        </Link>
      </div>
    </div>
  );
}

// ─── Main page ────────────────────────────────────────────────────

export function ModelCard() {
  const qc = useQueryClient();
  const { runId = '' } = useParams<{ runId: string }>();
  const [tab, setTab] = useState<Tab>('diagnostics');
  const [promoteOpen, setPromoteOpen] = useState(false);

  const run = useQuery({
    queryKey: ['runs', runId],
    queryFn: () => runsApi.get(runId),
    enabled: !!runId,
  });

  const registry = useQuery<PlotRegistry, Error>({
    queryKey: ['plots', 'registry'],
    queryFn: () => plotsApi.registry(),
    staleTime: 5 * 60_000,
  });

  const taskKey = useMemo(() => {
    const r = run.data;
    if (!r) return '';
    const snap = (r.snapshot ?? {}) as Record<string, unknown>;
    const task =
      (snap.task as string | undefined) ??
      (snap.task_type as string | undefined) ??
      '';
    return task.toLowerCase();
  }, [run.data]);

  const dispatch = TAB_DISPATCH[taskKey] ?? null;

  const supported = useMemo(() => {
    const set = new Set(registry.data?.tasks?.[taskKey] ?? []);
    if (!dispatch) return null;
    return {
      diagnostics: dispatch.diagnostics.filter((k) => set.has(k)),
      explainability: dispatch.explainability.filter((k) => set.has(k)),
      curves: dispatch.curves.filter((k) => set.has(k)),
      raw: dispatch.raw.filter((k) => set.has(k)),
    } as Record<Tab, string[]>;
  }, [dispatch, registry.data, taskKey]);

  const tabs = useMemo(() => {
    if (!supported) return [];
    return (
      [
        { key: 'diagnostics' as const, label: 'Diagnostics' },
        { key: 'explainability' as const, label: 'Explainability' },
        { key: 'curves' as const, label: 'Curves' },
        { key: 'raw' as const, label: 'Raw' },
      ]
    )
      .filter(({ key }) => supported[key].length > 0)
      .map(({ key, label }) => ({ key, label, kinds: supported[key] }));
  }, [supported]);

  const activeKinds = supported?.[tab] ?? [];

  // Pre-flight: probe the first available plot to detect the
  // "no promoted pipeline" situation without rendering 4 broken cards.
  const probeKind = activeKinds[0] ?? '';
  const probe = useQuery<PlotEnvelope, Error>({
    queryKey: ['runs', runId, 'plots', probeKind],
    queryFn: () => plotsApi.forRun(runId, probeKind),
    enabled: !!runId && !!probeKind,
    staleTime: 60_000,
    retry: false,
  });

  const needsPromotion = probe.isError && isNoPipelineError(probe.error);

  const handlePromoteSuccess = () => {
    // Re-fetch all plots for this run + the registry.
    qc.invalidateQueries({ queryKey: ['runs', runId] });
    qc.invalidateQueries({
      predicate: (q) => {
        const k = q.queryKey;
        return Array.isArray(k) && k[0] === 'runs' && k[1] === runId && k[2] === 'plots';
      },
    });
  };

  return (
    <div className="space-y-8">
      {/* ─── Hero ─────────────────────────────────────────────── */}
      <header className="space-y-2">
        <BackButton />
        <nav className="text-xs text-ink-500">
          <Link to="/" className="hover:text-ink-900 dark:hover:text-ink-50 transition-colors">
            Workspaces
          </Link>
          <span className="mx-1.5 text-ink-300 dark:text-ink-700">/</span>
          <Link
            to={`/runs/${runId}`}
            className="hover:text-ink-900 dark:hover:text-ink-50 transition-colors"
          >
            Run {runId.slice(0, 8)}…
          </Link>
          <span className="mx-1.5 text-ink-300 dark:text-ink-700">/</span>
          <span className="text-ink-700 dark:text-ink-300">Model card</span>
        </nav>
        <div className="flex items-start justify-between gap-6">
          <div className="min-w-0">
            <h1 className="h-page">Model card</h1>
            <p className="mt-2 text-sm text-ink-500">
              {run.data?.status === 'succeeded'
                ? `Diagnostics for the model produced by run ${runId.slice(0, 8)}.`
                : 'Awaiting run completion…'}
            </p>
          </div>
          {!needsPromotion && (
            <Link to={`/runs/${runId}`} className="btn-secondary shrink-0">
              ← Back to run
            </Link>
          )}
        </div>
      </header>

      {/* ─── KPI strip ────────────────────────────────────────── */}
      <KpiStrip leaderboard={run.data?.leaderboard} />

      {/* ─── Body: empty state or plots grid ─────────────────── */}
      {needsPromotion ? (
        <NoPipelineEmptyState
          runId={runId}
          onPromote={() => setPromoteOpen(true)}
        />
      ) : tabs.length > 0 ? (
        <>
          {/* Tabs */}
          <div
            role="tablist"
            className="flex gap-1 border-b border-ink-200 dark:border-ink-800"
          >
            {tabs.map(({ key, label }) => (
              <button
                key={key}
                role="tab"
                aria-selected={tab === key}
                onClick={() => setTab(key)}
                className={`px-3.5 py-2 text-sm font-medium transition-colors border-b-2 -mb-px ${
                  tab === key
                    ? 'border-ink-900 dark:border-ink-50 text-ink-900 dark:text-ink-50'
                    : 'border-transparent text-ink-500 hover:text-ink-900 dark:hover:text-ink-50'
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Plot grid */}
          <div className="grid gap-4 grid-cols-1 lg:grid-cols-2">
            {activeKinds.map((kind) => (
              <PlotCard key={kind} runId={runId} kind={kind} />
            ))}
          </div>
        </>
      ) : (
        <div className="card text-sm text-ink-500">
          {!taskKey
            ? 'Loading run details…'
            : `No diagnostic plots available yet for task type "${taskKey}".`}
        </div>
      )}

      <PromoteDialog
        runId={runId}
        open={promoteOpen}
        onClose={() => setPromoteOpen(false)}
        onSuccess={handlePromoteSuccess}
      />
    </div>
  );
}
