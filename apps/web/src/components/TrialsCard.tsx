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
import { Link, useNavigate } from 'react-router-dom';
import Plot from 'react-plotly.js';
import { describeApi, runsApi } from '@/api/endpoints';
import type { TaskType, TrialKind } from '@/api/types';
import {
  BlendDialog,
  StackDialog,
} from '@/components/TrialActionDialog';

export interface TrialsCardProps {
  runId: string;
  workspaceId?: string;
  /** Fires when a follow-on action (blend/stack) has been kicked off.
   *  The parent typically uses this to open the event-log drawer so the
   *  user sees live progress immediately. */
  onActionSubmitted?: () => void;
}

// Metrics where smaller is better. Used for sort + chart axis direction.
const ASCENDING_METRICS = new Set(['MAE', 'MSE', 'RMSE', 'RMSLE', 'MAPE', 'TT (Sec)']);

export function TrialsCard({ runId, workspaceId, onActionSubmitted }: TrialsCardProps) {
  const [view, setView] = useState<'table' | 'chart'>('table');
  // Set of trial ids selected via the row checkboxes — drives the
  // Blend / Stack bulk-action bar.
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [blendOpen, setBlendOpen] = useState(false);
  const [stackOpen, setStackOpen] = useState(false);

  // Pull trials + the parent run (we need the task to resolve model names).
  const trials = useQuery({
    queryKey: ['runs', runId, 'trials'],
    // Poll while follow-on actions might be in flight — the worker writes
    // a new Trial row on completion and we want it to show up without a
    // manual refresh. 4s feels live without hammering the API.
    refetchInterval: 4000,
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

  const toggleSelected = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };
  const clearSelected = () => setSelected(new Set());

  // Derive items + selection slices BEFORE any early returns so the
  // useMemo hook list stays stable across renders (React's hooks-rules
  // rejects calling a hook conditionally — adding one after a guard
  // that returns early on the first render and not the second is the
  // classic black-screen footgun).
  const items = trials.data?.items ?? [];
  const selectableItems = useMemo(
    () => items.filter((t) => t.has_artifact !== false),
    [items],
  );
  const selectedIds = useMemo(
    () => Array.from(selected).filter((id) => items.some((t) => t.id === id)),
    [selected, items],
  );
  const canBlendStack = selectedIds.length >= 2;

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
      <div className="flex items-baseline justify-between mb-4 gap-3 flex-wrap">
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
        <div className="flex items-center gap-2">
          <button
            type="button"
            className="btn-secondary text-sm"
            disabled={
              selectableItems.length < 2 || run.data?.status !== 'succeeded'
            }
            onClick={() => setBlendOpen(true)}
            title="Voting ensemble across selected trials"
          >
            Blend models
          </button>
          <button
            type="button"
            className="btn-secondary text-sm"
            disabled={
              selectableItems.length < 2 || run.data?.status !== 'succeeded'
            }
            onClick={() => setStackOpen(true)}
            title="Stack with a meta-learner"
          >
            Stack models
          </button>
          <ViewToggle view={view} onChange={setView} />
        </div>
      </div>

      {/* Bulk-action bar — kept as a power-user shortcut when rows are
          pre-checked. Header buttons above are the primary discoverable
          path; this just lets a user who already ticked rows skip the
          modal's source picker. */}
      {selected.size > 0 && (
        <div className="mb-3 flex items-center justify-between gap-3 rounded-lg border border-accent-200 dark:border-accent-500/30 bg-accent-50/60 dark:bg-accent-500/5 px-3 py-2 text-xs">
          <div className="flex items-center gap-2">
            <span className="font-medium text-ink-900 dark:text-ink-50">
              {selected.size} selected
            </span>
            <button
              type="button"
              className="text-ink-500 hover:text-ink-900 dark:hover:text-ink-50 underline-offset-2 hover:underline"
              onClick={clearSelected}
            >
              clear
            </button>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              className="btn-secondary text-xs"
              disabled={!canBlendStack || run.data?.status !== 'succeeded'}
              onClick={() => setBlendOpen(true)}
              title="Voting ensemble across the selected trials"
            >
              Blend
            </button>
            <button
              type="button"
              className="btn-primary text-xs"
              disabled={!canBlendStack || run.data?.status !== 'succeeded'}
              onClick={() => setStackOpen(true)}
              title="Stack with a meta-learner"
            >
              Stack
            </button>
          </div>
        </div>
      )}

      {view === 'table' ? (
        <TrialsTable
          items={items}
          metricKeys={metricKeys}
          primaryMetric={primaryMetric}
          nameOf={nameOf}
          workspaceId={workspaceId}
          runId={runId}
          selected={selected}
          onToggle={toggleSelected}
          selectableItems={selectableItems}
        />
      ) : (
        <TrialsChart
          items={items}
          metricKeys={metricKeys}
          primaryMetric={primaryMetric}
          nameOf={nameOf}
        />
      )}

      <BlendDialog
        open={blendOpen}
        onClose={() => setBlendOpen(false)}
        onSubmitted={() => {
          clearSelected();
          onActionSubmitted?.();
        }}
        runId={runId}
        availableTrials={selectableItems}
        initialSelectedIds={selectedIds}
        nameOf={nameOf}
        primaryMetric={primaryMetric}
      />
      <StackDialog
        open={stackOpen}
        onClose={() => setStackOpen(false)}
        onSubmitted={() => {
          clearSelected();
          onActionSubmitted?.();
        }}
        runId={runId}
        availableTrials={selectableItems}
        initialSelectedIds={selectedIds}
        nameOf={nameOf}
        primaryMetric={primaryMetric}
        task={task ?? null}
      />
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
  rank: number | null;
  metrics: Record<string, number | string>;
  is_best: boolean;
  fitted_pipeline_id: string | null;
  has_artifact?: boolean;
  kind?: TrialKind;
  status?:
    | 'queued'
    | 'running'
    | 'succeeded'
    | 'failed'
    | 'cancelled';
  duration_ms?: number | null;
  error?: string | null;
  parent_trial_ids?: string[];
}

// Tone for the kind chip — compare reads neutral, follow-on actions get
// distinct accent colours so the eye groups them in a busy leaderboard.
const KIND_CHIP: Record<TrialKind, string> = {
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

function TrialsTable({
  items,
  metricKeys,
  primaryMetric,
  nameOf,
  workspaceId,
  runId,
  selected,
  onToggle,
  selectableItems,
}: {
  items: TrialItem[];
  metricKeys: string[];
  primaryMetric: string;
  nameOf: (id: string) => string;
  workspaceId?: string;
  runId: string;
  selected: Set<string>;
  onToggle: (id: string) => void;
  selectableItems: TrialItem[];
}) {
  const navigate = useNavigate();
  // For the inline bar visualisation on the primary metric.
  const ascending = ASCENDING_METRICS.has(primaryMetric);
  const primaryValues = items
    .map((t) => t.metrics[primaryMetric])
    .filter((v): v is number => typeof v === 'number');
  const maxV = Math.max(...primaryValues, 1);
  const minV = Math.min(...primaryValues, 0);

  const allSelectableSelected =
    selectableItems.length > 0 &&
    selectableItems.every((t) => selected.has(t.id));
  const someSelected = selectableItems.some((t) => selected.has(t.id));

  const toggleAll = () => {
    if (allSelectableSelected) {
      // Clear everything that's currently selected.
      for (const t of selectableItems) {
        if (selected.has(t.id)) onToggle(t.id);
      }
    } else {
      // Add anything missing.
      for (const t of selectableItems) {
        if (!selected.has(t.id)) onToggle(t.id);
      }
    }
  };

  return (
    <div className="card overflow-hidden p-0">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-white text-ink-500 dark:bg-ink-900 border-b border-ink-200 dark:border-ink-800">
            <tr>
              <th className="px-3 py-2.5 text-left font-medium w-8">
                <input
                  type="checkbox"
                  aria-label="Select all"
                  className="cursor-pointer accent-accent-500"
                  checked={allSelectableSelected}
                  ref={(el) => {
                    if (el) el.indeterminate = !allSelectableSelected && someSelected;
                  }}
                  onChange={toggleAll}
                />
              </th>
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
              const trialHref = `/runs/${runId}/trials/${t.id}`;
              return (
                <tr
                  key={t.id}
                  role="link"
                  tabIndex={0}
                  onClick={() => navigate(trialHref)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      navigate(trialHref);
                    }
                  }}
                  className={`cursor-pointer border-t border-ink-200 dark:border-ink-800 ${
                    selected.has(t.id)
                      ? 'bg-accent-50/60 dark:bg-accent-500/10'
                      : t.is_best
                        ? 'bg-success-50/50 dark:bg-success-500/5 hover:bg-success-50'
                        : 'hover:bg-ink-50 dark:hover:bg-ink-950/40'
                  }`}
                >
                  <td
                    className="px-3 py-2.5 align-middle"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <input
                      type="checkbox"
                      aria-label={`Select ${t.model_id}`}
                      className="cursor-pointer accent-accent-500"
                      checked={selected.has(t.id)}
                      onChange={() => onToggle(t.id)}
                      disabled={t.has_artifact === false}
                      title={
                        t.has_artifact === false
                          ? 'No stored pipeline — cannot use as a blend/stack source'
                          : 'Select for blend / stack'
                      }
                    />
                  </td>
                  <td className="px-4 py-2.5 text-ink-500 tabular-nums">
                    {t.rank}
                    {t.is_best && (
                      <span className="ml-1 text-warn-500" title="Best">
                        ★
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-2.5">
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <span className="font-medium text-ink-900 dark:text-ink-50">
                        {nameOf(t.model_id)}
                      </span>
                      {t.kind && t.kind !== 'compare' && (
                        <span
                          className={`${KIND_CHIP[t.kind]} text-[10px] capitalize`}
                          title={
                            t.parent_trial_ids && t.parent_trial_ids.length > 0
                              ? `${t.kind} from ${t.parent_trial_ids.length} source${t.parent_trial_ids.length === 1 ? '' : 's'}`
                              : t.kind
                          }
                        >
                          {t.kind}
                        </span>
                      )}
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
                        to={`/workspaces/${workspaceId}/models`}
                        onClick={(e) => e.stopPropagation()}
                        title="View this trial's promoted version in the Model registry"
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
            // Render values to the right of each bar so digits never sit
            // on top of the colored fill where contrast was unreadable.
            text: values.map((v) => v.toFixed(4)),
            textposition: 'outside',
            cliponaxis: false,
            insidetextanchor: 'end',
            textfont: {
              family: 'Inter, system-ui, sans-serif',
              size: 11,
              color: '#27272a',
            },
            hovertemplate: '%{y}<br>%{x:.4f}<extra></extra>',
          },
        ]}
        layout={{
          // Generous per-row height so labels don't pancake on big leaderboards.
          height: Math.max(320, sorted.length * 36 + 120),
          // automargin handles l/r so the longest model name never clips.
          margin: { l: 8, r: 56, t: 12, b: 48 },
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
            automargin: true,
            // Pad the right edge so the outside text labels don't run
            // past the chart frame on tight viewports.
            range: (() => {
              const max = Math.max(...values, 0);
              const min = Math.min(...values, 0);
              const span = max - min || Math.abs(max) || 1;
              return [min - span * 0.02, max + span * 0.12];
            })(),
            title: { text: metric, standoff: 6 },
          },
          yaxis: {
            autorange: 'reversed',
            automargin: true,
            tickfont: { size: 12 },
          },
          showlegend: false,
          bargap: 0.35,
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%' }}
        useResizeHandler
      />
    </div>
  );
}
