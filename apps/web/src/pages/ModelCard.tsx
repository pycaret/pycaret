/**
 * Model Card — per-model deep-dive screen.
 *
 * Lives at /runs/:runId/model-card. Shows:
 *  - Header: model id, run status, promotion CTA, share link.
 *  - KPI strip: top metrics from the leaderboard for the chosen model.
 *  - Tabs: Diagnostics | Explainability | Curves | Raw.
 *  - Each tab is a grid of `<PlotlyFigure>` cards driven by the
 *    /runs/:runId/plots/:kind endpoint.
 *
 * The plot kinds shown depend on the run's task type — derived from
 * the registry response so we don't hard-code per-task lists here.
 */

import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';

import { plotsApi, runsApi } from '../api/endpoints';
import { PlotlyFigure } from '../components/PlotlyFigure';
import type { PlotEnvelope, PlotRegistry } from '../api/types';

type Tab = 'diagnostics' | 'explainability' | 'curves' | 'raw';

interface TabConfig {
  label: string;
  kinds: string[];
}

/**
 * Map a task-type (lowercase) to per-tab plot kinds. Source-of-truth
 * lives in the API's plot registry; this dispatch decides which kinds
 * land on which tab.
 */
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

function PlotCard({ runId, kind }: { runId: string; kind: string }) {
  const q = useQuery<PlotEnvelope, Error>({
    queryKey: ['runs', runId, 'plots', kind],
    queryFn: () => plotsApi.forRun(runId, kind),
    enabled: !!runId,
    staleTime: 60_000,
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

function KpiStrip({ leaderboard }: { leaderboard: unknown }) {
  // leaderboard is loosely typed (JSON from the API). Try to pull the
  // first row's metric columns as KPI tiles. Falls back to an empty
  // strip when the shape doesn't match.
  const tiles = useMemo<Array<{ label: string; value: string }>>(() => {
    const arr = Array.isArray(leaderboard) ? leaderboard : null;
    const top = arr && arr.length > 0 ? arr[0] : null;
    if (!top || typeof top !== 'object') return [];
    const out: Array<{ label: string; value: string }> = [];
    for (const [k, v] of Object.entries(top)) {
      if (k === 'Model') continue;
      if (typeof v === 'number') {
        out.push({ label: k, value: v.toFixed(4) });
      }
      if (out.length >= 6) break;
    }
    return out;
  }, [leaderboard]);

  if (tiles.length === 0) {
    return (
      <div
        style={{
          padding: 12,
          background: 'rgba(91,141,239,0.06)',
          borderRadius: 12,
          fontSize: 12,
          color: '#64748B',
        }}
      >
        No metric summary yet — promote the run or finish training to populate.
      </div>
    );
  }

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: `repeat(${Math.min(tiles.length, 6)}, minmax(120px, 1fr))`,
        gap: 12,
      }}
    >
      {tiles.map((t) => (
        <div
          key={t.label}
          className="card"
          style={{ display: 'flex', flexDirection: 'column', gap: 4, padding: 12 }}
        >
          <div style={{ fontSize: 11, textTransform: 'uppercase', color: '#64748B' }}>
            {t.label}
          </div>
          <div style={{ fontSize: 22, fontWeight: 700, color: '#0F172A' }}>{t.value}</div>
        </div>
      ))}
    </div>
  );
}

export function ModelCard() {
  const { runId = '' } = useParams<{ runId: string }>();
  const [tab, setTab] = useState<Tab>('diagnostics');

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
    const task = (snap.task as string | undefined) ?? (snap.task_type as string | undefined) ?? '';
    return task.toLowerCase();
  }, [run.data]);

  const dispatch = TAB_DISPATCH[taskKey] ?? null;

  // Filter to plot kinds the API actually advertises for this task —
  // protects against client/server skew.
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

  const tabs: Array<{ key: Tab; cfg: TabConfig }> = useMemo(() => {
    if (!supported) return [];
    return (
      [
        { key: 'diagnostics', label: 'Diagnostics' },
        { key: 'explainability', label: 'Explainability' },
        { key: 'curves', label: 'Curves' },
        { key: 'raw', label: 'Raw' },
      ] as const
    )
      .filter(({ key }) => supported[key].length > 0)
      .map(({ key, label }) => ({ key, cfg: { label, kinds: supported[key] } }));
  }, [supported]);

  const activeKinds = supported?.[tab] ?? [];

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <nav style={{ fontSize: 12, color: '#94A3B8' }}>
          <Link to="/" style={{ color: 'inherit' }}>
            Workspaces
          </Link>{' '}
          /{' '}
          <Link to={`/runs/${runId}`} style={{ color: 'inherit' }}>
            Run {runId.slice(0, 8)}…
          </Link>{' '}
          / Model card
        </nav>
        <h1 style={{ fontSize: 24, fontWeight: 700, color: '#0F172A', margin: 0 }}>
          Model card
        </h1>
        <p style={{ color: '#64748B', fontSize: 13, margin: 0 }}>
          {run.data?.status === 'succeeded'
            ? `Diagnostics for the model produced by run ${runId.slice(0, 8)}.`
            : 'Awaiting run completion…'}
        </p>
      </header>

      <KpiStrip leaderboard={run.data?.leaderboard} />

      {tabs.length > 0 ? (
        <>
          <div
            role="tablist"
            style={{
              display: 'flex',
              gap: 4,
              borderBottom: '1px solid rgba(148, 163, 184, 0.2)',
            }}
          >
            {tabs.map(({ key, cfg }) => (
              <button
                key={key}
                role="tab"
                aria-selected={tab === key}
                onClick={() => setTab(key)}
                style={{
                  padding: '8px 14px',
                  fontSize: 13,
                  fontWeight: 500,
                  background: 'transparent',
                  color: tab === key ? '#5B8DEF' : '#64748B',
                  border: 'none',
                  borderBottom:
                    tab === key ? '2px solid #5B8DEF' : '2px solid transparent',
                  cursor: 'pointer',
                  transition: 'color 0.15s',
                }}
              >
                {cfg.label}
              </button>
            ))}
          </div>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(420px, 1fr))',
              gap: 16,
            }}
          >
            {activeKinds.map((kind) => (
              <PlotCard key={kind} runId={runId} kind={kind} />
            ))}
          </div>
        </>
      ) : (
        <div className="card">
          <p style={{ color: '#64748B', fontSize: 13, margin: 0 }}>
            {!taskKey
              ? "Loading run details…"
              : `No diagnostic plots available yet for task type "${taskKey}". Promote the run to generate them.`}
          </p>
        </div>
      )}
    </div>
  );
}
