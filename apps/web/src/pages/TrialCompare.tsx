/**
 * Trial comparison — `/runs/:runId/compare?a=<trialId>&b=<trialId>`.
 *
 * Side-by-side view of two trials from the same run:
 *   [breadcrumb back to run]
 *   [headers: Trial A vs Trial B + per-trial Open buttons]
 *   [Metrics diff table — each metric row, who wins highlighted]
 *   [Pipeline diagrams stacked side-by-side]
 *   [Plots overlay — confusion matrix + ROC for both trials]
 *
 * The whole route is read-only — promote/notes happen on the per-trial
 * detail page. This is a "spot the difference" surface, nothing more.
 */

import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { describeApi, runsApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import { BackButton } from '@/components/BackButton';
import { PipelineDiagram } from '@/components/PipelineDiagram';
import { PlotlyFigure } from '@/components/PlotlyFigure';
import type { TaskType } from '@/api/types';

const ASCENDING_METRICS = new Set(['MAE', 'MSE', 'RMSE', 'RMSLE', 'MAPE', 'TT (Sec)']);

const SIDE_BY_SIDE_PLOTS: Record<string, string[]> = {
  classification: ['confusion_matrix', 'roc_curve', 'pr_curve'],
  regression: ['prediction_error', 'residuals'],
  clustering: ['silhouette_plot'],
  anomaly: ['score_distribution'],
};

export function TrialCompare() {
  const { runId = '' } = useParams<{ runId: string }>();
  const [params] = useSearchParams();
  const a = params.get('a') ?? '';
  const b = params.get('b') ?? '';

  const trialA = useQuery({
    queryKey: ['runs', runId, 'trials', a],
    queryFn: () => runsApi.trial(runId, a),
    enabled: !!runId && !!a,
  });
  const trialB = useQuery({
    queryKey: ['runs', runId, 'trials', b],
    queryFn: () => runsApi.trial(runId, b),
    enabled: !!runId && !!b,
  });

  const task = (trialA.data?.task ?? trialB.data?.task ?? null) as TaskType | null;
  const models = useQuery({
    queryKey: ['describe', 'models', task],
    queryFn: () => describeApi.models(task!),
    enabled: !!task,
    staleTime: 10 * 60 * 1000,
  });
  const friendly = (id: string | undefined): string => {
    if (!id) return '';
    const m = (models.data ?? []).find((x) => x.id === id);
    return m?.name ?? id;
  };

  const metrics = useMemo(() => {
    const A = trialA.data;
    const B = trialB.data;
    if (!A || !B) return [];
    const keys = new Set<string>();
    for (const k of Object.keys(A.metrics)) keys.add(k);
    for (const k of Object.keys(B.metrics)) keys.add(k);
    const out: { name: string; av: number | null; bv: number | null }[] = [];
    for (const k of keys) {
      const av = typeof A.metrics[k] === 'number' ? (A.metrics[k] as number) : null;
      const bv = typeof B.metrics[k] === 'number' ? (B.metrics[k] as number) : null;
      if (av == null && bv == null) continue;
      out.push({ name: k, av, bv });
    }
    return out;
  }, [trialA.data, trialB.data]);

  if (!a || !b) {
    return (
      <div className="card text-sm text-danger-600">
        Comparison requires <code>?a=&lt;trialId&gt;&amp;b=&lt;trialId&gt;</code>.
      </div>
    );
  }
  if (trialA.error || trialB.error) {
    return (
      <div className="card text-sm text-danger-600">
        {errorMessage(trialA.error || trialB.error)}
      </div>
    );
  }

  const A = trialA.data;
  const B = trialB.data;
  const plotKinds = task ? SIDE_BY_SIDE_PLOTS[task] ?? [] : [];

  return (
    <div className="space-y-8">
      <header>
        <BackButton to={`/runs/${runId}`} />
        <nav className="text-xs text-ink-500 mb-2">
          <Link to={`/runs/${runId}`} className="hover:text-ink-900 dark:hover:text-ink-50 font-mono">
            Run · {runId.slice(0, 8)}
          </Link>
          <span className="mx-1.5 text-ink-300">/</span>
          <span className="text-ink-700 dark:text-ink-300">Compare</span>
        </nav>
        <h1 className="h-page">Compare trials</h1>
      </header>

      <section className="grid grid-cols-2 gap-4">
        <TrialCard
          runId={runId}
          trialId={a}
          modelName={friendly(A?.model_id)}
          rank={A?.rank ?? null}
          isBest={A?.is_best ?? false}
        />
        <TrialCard
          runId={runId}
          trialId={b}
          modelName={friendly(B?.model_id)}
          rank={B?.rank ?? null}
          isBest={B?.is_best ?? false}
        />
      </section>

      {metrics.length > 0 && (
        <section>
          <h2 className="h-section mb-3">Metrics</h2>
          <div className="card overflow-x-auto p-0">
            <table className="w-full text-sm">
              <thead className="bg-white text-ink-500 dark:bg-ink-900 border-b border-ink-200 dark:border-ink-800">
                <tr>
                  <th className="px-4 py-2.5 text-left font-medium">Metric</th>
                  <th className="px-4 py-2.5 text-right font-medium tabular-nums">A</th>
                  <th className="px-4 py-2.5 text-right font-medium tabular-nums">B</th>
                  <th className="px-4 py-2.5 text-right font-medium tabular-nums">Δ (B − A)</th>
                  <th className="px-4 py-2.5 text-center font-medium">Winner</th>
                </tr>
              </thead>
              <tbody>
                {metrics.map(({ name, av, bv }) => {
                  const lower = ASCENDING_METRICS.has(name);
                  const delta =
                    av != null && bv != null ? bv - av : null;
                  let winner: 'a' | 'b' | 'tie' | null = null;
                  if (av != null && bv != null) {
                    if (av === bv) winner = 'tie';
                    else if (lower) winner = av < bv ? 'a' : 'b';
                    else winner = av > bv ? 'a' : 'b';
                  }
                  return (
                    <tr
                      key={name}
                      className="border-t border-ink-200 dark:border-ink-800"
                    >
                      <td className="px-4 py-2 font-mono text-xs text-ink-700 dark:text-ink-300">
                        {name}
                      </td>
                      <td
                        className={`px-4 py-2 text-right font-mono tabular-nums ${
                          winner === 'a' ? 'text-success-600 font-semibold' : 'text-ink-900 dark:text-ink-50'
                        }`}
                      >
                        {av != null ? av.toFixed(4) : '—'}
                      </td>
                      <td
                        className={`px-4 py-2 text-right font-mono tabular-nums ${
                          winner === 'b' ? 'text-success-600 font-semibold' : 'text-ink-900 dark:text-ink-50'
                        }`}
                      >
                        {bv != null ? bv.toFixed(4) : '—'}
                      </td>
                      <td className="px-4 py-2 text-right font-mono tabular-nums text-ink-500">
                        {delta != null
                          ? `${delta >= 0 ? '+' : ''}${delta.toFixed(4)}`
                          : '—'}
                      </td>
                      <td className="px-4 py-2 text-center text-xs">
                        {winner === 'a' && (
                          <span className="pill-success">A</span>
                        )}
                        {winner === 'b' && (
                          <span className="pill-success">B</span>
                        )}
                        {winner === 'tie' && (
                          <span className="pill-neutral">tie</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section>
        <h2 className="h-section mb-3">Pipeline structure</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <PipelineDiagram tree={A?.pipeline_tree ?? null} />
          <PipelineDiagram tree={B?.pipeline_tree ?? null} />
        </div>
      </section>

      {plotKinds.length > 0 && (
        <section>
          <h2 className="h-section mb-3">Diagnostics</h2>
          <div className="space-y-6">
            {plotKinds.map((kind) => (
              <div key={kind} className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <ComparePlot
                  runId={runId}
                  trialId={a}
                  kind={kind}
                  label={`A · ${friendly(A?.model_id) || A?.model_id || 'a'}`}
                />
                <ComparePlot
                  runId={runId}
                  trialId={b}
                  kind={kind}
                  label={`B · ${friendly(B?.model_id) || B?.model_id || 'b'}`}
                />
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

function TrialCard({
  runId,
  trialId,
  modelName,
  rank,
  isBest,
}: {
  runId: string;
  trialId: string;
  modelName: string;
  rank: number | null;
  isBest: boolean;
}) {
  return (
    <div className="card">
      <div className="flex items-baseline justify-between gap-2 flex-wrap">
        <div>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm font-semibold text-ink-900 dark:text-ink-50">
              {modelName || trialId.slice(0, 8)}
            </span>
            {rank != null && (
              <span className={`pill-${isBest ? 'success' : 'neutral'}`}>
                Rank #{rank}
                {isBest && ' · Best'}
              </span>
            )}
          </div>
          <p className="text-xs text-ink-500 font-mono mt-1">{trialId}</p>
        </div>
        <Link
          to={`/runs/${runId}/trials/${trialId}`}
          className="text-xs text-accent-600 hover:underline"
        >
          Open →
        </Link>
      </div>
    </div>
  );
}

function ComparePlot({
  runId,
  trialId,
  kind,
  label,
}: {
  runId: string;
  trialId: string;
  kind: string;
  label: string;
}) {
  const q = useQuery({
    queryKey: ['runs', runId, 'trials', trialId, 'plots', kind],
    queryFn: () => runsApi.trialPlot(runId, trialId, kind),
    staleTime: 60_000,
    retry: false,
  });
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
      title={label}
      height={300}
      hideToolbar
    />
  );
}
