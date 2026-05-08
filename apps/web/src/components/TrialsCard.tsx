/**
 * Trials card — leaderboard view of every candidate model from a
 * ``compare_models`` / ``automl`` / ``search`` run.
 *
 * Two view modes:
 *   - **Table** (default): sortable columns, pretty model names, primary
 *     metric highlighted with an inline progress bar.
 *   - **Chart**: horizontal bar chart of the selected metric, models on
 *     the y-axis sorted by score.
 */

import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import Plot from 'react-plotly.js';
import { describeApi, runsApi } from '@/api/endpoints';
import type { TaskType } from '@/api/types';

export interface TrialsCardProps {
  runId: string;
  workspaceId?: string;
}

// Metrics where smaller is better. Used for sort + chart axis direction.
const ASCENDING_METRICS = new Set(['MAE', 'MSE', 'RMSE', 'RMSLE', 'MAPE', 'TT (Sec)']);

export function TrialsCard({ runId, workspaceId }: TrialsCardProps) {
  const [view, setView] = useState<'table' | 'chart'>('table');

  // Pull trials + the parent run (we need the task to resolve model names).
  const trials = useQuery({
    queryKey: ['runs', runId, 'trials'],
    queryFn: () => runsApi.trials(runId),
  });
  const run = useQuery({
    queryKey: ['runs', runId],
    queryFn: () => runsApi.get(runId),
  });

  const task = (run.data?.snapshot as { task?: TaskType } | null)?.task;
  const models = useQuery({
    queryKey: ['describe', 'models', task],
    queryFn: () => describeApi.models(task!),
    enabled: !!task,
    staleTime: 10 * 60 * 1000,
  });

  // model_id → friendly name (fallbacks to model_id when not found).
  const nameOf = useMemo(() => {
    const m: Record<string, string> = {};
    for (const card of models.data ?? []) m[card.id] = card.name;
    return (id: string) => m[id] ?? id;
  }, [models.data]);

  if (trials.isPending) {
    return (
      <section>
        <h2 className="h-section mb-3">Trials</h2>
        <div className="card text-sm text-ink-500">Loading trials…</div>
      </section>
    );
  }
  if (trials.error) {
    return (
      <section>
        <h2 className="h-section mb-3">Trials</h2>
        <div className="card text-sm text-danger-600">Could not load trials.</div>
      </section>
    );
  }
  const items = trials.data?.items ?? [];
  if (items.length === 0) {
    return (
      <section>
        <h2 className="h-section mb-3">Trials</h2>
        <p className="text-sm text-ink-500">
          No trials yet — runs of plan{' '}
          <code className="font-mono text-xs">compare</code> or{' '}
          <code className="font-mono text-xs">search</code> persist one per
          candidate.
        </p>
      </section>
    );
  }

  // Numeric metric columns, in stable order (skip TT which is timing not skill).
  const allMetricKeys = Object.keys(items[0].metrics).filter(
    (k) => typeof items[0].metrics[k] === 'number',
  );
  const metricKeys = allMetricKeys;
  const primaryMetric = metricKeys[0] ?? 'Accuracy';

  return (
    <section>
      <div className="flex items-baseline justify-between mb-4 gap-3">
        <div>
          <h2 className="h-section">
            Trials{' '}
            <span className="text-ink-400 font-normal">({items.length})</span>
          </h2>
          <p className="text-xs text-ink-500 mt-0.5">
            Every candidate model the engine tried, ranked by{' '}
            <span className="font-mono">{primaryMetric}</span>.
          </p>
        </div>
        <ViewToggle view={view} onChange={setView} />
      </div>

      {view === 'table' ? (
        <TrialsTable
          items={items}
          metricKeys={metricKeys}
          primaryMetric={primaryMetric}
          nameOf={nameOf}
          workspaceId={workspaceId}
        />
      ) : (
        <TrialsChart
          items={items}
          metricKeys={metricKeys}
          primaryMetric={primaryMetric}
          nameOf={nameOf}
        />
      )}
    </section>
  );
}

// ─── View toggle ───────────────────────────────────────────────────

function ViewToggle({
  view,
  onChange,
}: {
  view: 'table' | 'chart';
  onChange: (v: 'table' | 'chart') => void;
}) {
  return (
    <div className="inline-flex rounded-md border border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 p-0.5 text-xs">
      {(['table', 'chart'] as const).map((v) => (
        <button
          key={v}
          onClick={() => onChange(v)}
          className={`px-2.5 py-1 rounded font-medium transition-colors ${
            view === v
              ? 'bg-ink-900 text-white dark:bg-white dark:text-ink-900'
              : 'text-ink-600 hover:text-ink-900 dark:text-ink-400 dark:hover:text-ink-50'
          }`}
        >
          {v === 'table' ? 'Table' : 'Chart'}
        </button>
      ))}
    </div>
  );
}

// ─── Table view ────────────────────────────────────────────────────

interface TrialItem {
  id: string;
  model_id: string;
  rank: number;
  metrics: Record<string, number | string>;
  is_best: boolean;
  fitted_pipeline_id: string | null;
}

function TrialsTable({
  items,
  metricKeys,
  primaryMetric,
  nameOf,
  workspaceId,
}: {
  items: TrialItem[];
  metricKeys: string[];
  primaryMetric: string;
  nameOf: (id: string) => string;
  workspaceId?: string;
}) {
  // For the inline bar visualisation on the primary metric.
  const ascending = ASCENDING_METRICS.has(primaryMetric);
  const primaryValues = items
    .map((t) => t.metrics[primaryMetric])
    .filter((v): v is number => typeof v === 'number');
  const maxV = Math.max(...primaryValues, 1);
  const minV = Math.min(...primaryValues, 0);

  return (
    <div className="card overflow-hidden p-0">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-white text-ink-500 dark:bg-ink-900 border-b border-ink-200 dark:border-ink-800">
            <tr>
              <th className="px-4 py-2.5 text-left font-medium w-10">#</th>
              <th className="px-4 py-2.5 text-left font-medium">Model</th>
              {metricKeys.map((k) => (
                <th
                  key={k}
                  className="px-4 py-2.5 text-right font-medium tabular-nums"
                >
                  {k === primaryMetric ? (
                    <span className="text-ink-900 dark:text-ink-50">{k}</span>
                  ) : (
                    k
                  )}
                </th>
              ))}
              <th className="px-4 py-2.5 text-left font-medium w-20">Pipeline</th>
            </tr>
          </thead>
          <tbody>
            {items.map((t) => {
              const primary = t.metrics[primaryMetric];
              const ratio =
                typeof primary === 'number' && maxV !== minV
                  ? ascending
                    ? 1 - (primary - minV) / (maxV - minV)
                    : (primary - minV) / (maxV - minV)
                  : 0;
              return (
                <tr
                  key={t.id}
                  className={`border-t border-ink-200 dark:border-ink-800 ${
                    t.is_best
                      ? 'bg-success-50/50 dark:bg-success-500/5'
                      : 'hover:bg-ink-50 dark:hover:bg-ink-950/40'
                  }`}
                >
                  <td className="px-4 py-2.5 text-ink-500 tabular-nums">
                    {t.rank}
                    {t.is_best && (
                      <span className="ml-1 text-warn-500" title="Best">
                        ★
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-2.5">
                    <div className="font-medium text-ink-900 dark:text-ink-50">
                      {nameOf(t.model_id)}
                    </div>
                    <div className="text-xs text-ink-400 font-mono">
                      {t.model_id}
                    </div>
                  </td>
                  {metricKeys.map((k) => {
                    const v = t.metrics[k];
                    const isPrimary = k === primaryMetric;
                    return (
                      <td
                        key={k}
                        className={`px-4 py-2.5 text-right tabular-nums ${
                          isPrimary
                            ? 'text-ink-900 dark:text-ink-50 font-medium'
                            : 'text-ink-700 dark:text-ink-300'
                        }`}
                      >
                        {isPrimary && typeof v === 'number' ? (
                          <div className="inline-flex items-center gap-2 justify-end">
                            <div
                              className="h-1.5 w-16 rounded-full bg-ink-100 dark:bg-ink-800 overflow-hidden"
                              aria-hidden
                            >
                              <div
                                className="h-full bg-accent-500"
                                style={{
                                  width: `${Math.max(2, ratio * 100)}%`,
                                }}
                              />
                            </div>
                            <span className="font-mono">{v.toFixed(4)}</span>
                          </div>
                        ) : typeof v === 'number' ? (
                          <span className="font-mono">{v.toFixed(4)}</span>
                        ) : (
                          <span className="text-ink-400">{String(v)}</span>
                        )}
                      </td>
                    );
                  })}
                  <td className="px-4 py-2.5">
                    {t.fitted_pipeline_id && workspaceId ? (
                      <Link
                        className="text-xs text-accent-600 hover:underline"
                        to={`/workspaces/${workspaceId}/pipelines/${t.fitted_pipeline_id}`}
                      >
                        view →
                      </Link>
                    ) : (
                      <span className="text-xs text-ink-400">—</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── Chart view ────────────────────────────────────────────────────

function TrialsChart({
  items,
  metricKeys,
  primaryMetric,
  nameOf,
}: {
  items: TrialItem[];
  metricKeys: string[];
  primaryMetric: string;
  nameOf: (id: string) => string;
}) {
  const [metric, setMetric] = useState(primaryMetric);

  const sorted = useMemo(() => {
    const ascending = ASCENDING_METRICS.has(metric);
    return [...items]
      .filter((t) => typeof t.metrics[metric] === 'number')
      .sort((a, b) => {
        const va = a.metrics[metric] as number;
        const vb = b.metrics[metric] as number;
        return ascending ? va - vb : vb - va;
      });
  }, [items, metric]);

  const labels = sorted.map((t) => nameOf(t.model_id));
  const values = sorted.map((t) => t.metrics[metric] as number);
  const colors = sorted.map((t) =>
    t.is_best ? 'rgb(34, 197, 94)' : 'rgb(91, 141, 239)',
  );

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-3 gap-3">
        <label className="text-xs text-ink-500 inline-flex items-center gap-2">
          Metric
          <select
            className="input py-1 text-sm w-auto"
            value={metric}
            onChange={(e) => setMetric(e.target.value)}
          >
            {metricKeys.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        </label>
        <p className="text-xs text-ink-500">
          {ASCENDING_METRICS.has(metric)
            ? 'Lower is better'
            : 'Higher is better'}
        </p>
      </div>
      <Plot
        data={[
          {
            type: 'bar',
            orientation: 'h',
            x: values,
            y: labels,
            marker: { color: colors },
            text: values.map((v) => v.toFixed(4)),
            textposition: 'auto',
            hovertemplate: '%{y}<br>%{x:.4f}<extra></extra>',
          },
        ]}
        layout={{
          height: Math.max(280, sorted.length * 28 + 80),
          margin: { l: 160, r: 24, t: 8, b: 32 },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: {
            family: 'Inter, system-ui, sans-serif',
            size: 12,
            color: '#52525b',
          },
          xaxis: {
            gridcolor: '#e4e4e7',
            zerolinecolor: '#e4e4e7',
            tickformat: '.3f',
          },
          yaxis: {
            autorange: 'reversed',
            automargin: true,
          },
          showlegend: false,
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%' }}
        useResizeHandler
      />
    </div>
  );
}
