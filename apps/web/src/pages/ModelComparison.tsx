/**
 * Model Comparison — `/workspaces/:wsId/compare`.
 *
 * Pick two pipelines from the workspace registry and compare them
 * side-by-side: parameters, metrics-summary diff, and (when both runs
 * shared a task) shared diagnostic plots in a 2-column grid.
 */

import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';

import { pipelinesApi, plotsApi, runsApi } from '../api/endpoints';
import { PlotlyFigure } from '../components/PlotlyFigure';
import type { PlotEnvelope } from '../api/types';

function PlotPair({ runA, runB, kind, title }: { runA: string; runB: string; kind: string; title: string }) {
  const a = useQuery<PlotEnvelope, Error>({
    queryKey: ['runs', runA, 'plots', kind],
    queryFn: () => plotsApi.forRun(runA, kind),
    enabled: !!runA,
    staleTime: 60_000,
  });
  const b = useQuery<PlotEnvelope, Error>({
    queryKey: ['runs', runB, 'plots', kind],
    queryFn: () => plotsApi.forRun(runB, kind),
    enabled: !!runB,
    staleTime: 60_000,
  });
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
      <PlotlyFigure
        figure={a.data?.figure}
        loading={a.isLoading}
        error={a.error ?? undefined}
        onRetry={() => a.refetch()}
        title={`A: ${title}`}
      />
      <PlotlyFigure
        figure={b.data?.figure}
        loading={b.isLoading}
        error={b.error ?? undefined}
        onRetry={() => b.refetch()}
        title={`B: ${title}`}
      />
    </div>
  );
}

const COMPARABLE_KINDS: Record<string, string[]> = {
  classification: ['confusion_matrix', 'roc_curve', 'pr_curve', 'feature_importance'],
  regression: ['prediction_error', 'residuals', 'feature_importance'],
};

export function ModelComparison() {
  const { wsId = '' } = useParams<{ wsId: string }>();
  const [pipeAId, setPipeAId] = useState<string>('');
  const [pipeBId, setPipeBId] = useState<string>('');

  const pipelines = useQuery({
    queryKey: ['pipelines', wsId],
    queryFn: () => pipelinesApi.list(wsId),
    enabled: !!wsId,
  });

  const pipeA = pipelines.data?.find((p) => p.id === pipeAId);
  const pipeB = pipelines.data?.find((p) => p.id === pipeBId);

  const runA = useQuery({
    queryKey: ['runs', pipeA?.origin_run_id],
    queryFn: () => runsApi.get(pipeA!.origin_run_id!),
    enabled: !!pipeA?.origin_run_id,
  });
  const runB = useQuery({
    queryKey: ['runs', pipeB?.origin_run_id],
    queryFn: () => runsApi.get(pipeB!.origin_run_id!),
    enabled: !!pipeB?.origin_run_id,
  });

  const sharedTask = useMemo(() => {
    const ta = ((runA.data?.snapshot ?? {}) as Record<string, unknown>).task as string | undefined;
    const tb = ((runB.data?.snapshot ?? {}) as Record<string, unknown>).task as string | undefined;
    if (!ta || !tb) return '';
    return ta.toLowerCase() === tb.toLowerCase() ? ta.toLowerCase() : '';
  }, [runA.data, runB.data]);

  const comparableKinds = COMPARABLE_KINDS[sharedTask] ?? [];

  // Metric diff: prefer leaderboard's first row.
  const metricRows = useMemo(() => {
    const ma = (Array.isArray(runA.data?.leaderboard) ? runA.data!.leaderboard![0] : null) as
      | Record<string, unknown>
      | null;
    const mb = (Array.isArray(runB.data?.leaderboard) ? runB.data!.leaderboard![0] : null) as
      | Record<string, unknown>
      | null;
    if (!ma || !mb) return [];
    const keys = Array.from(new Set([...Object.keys(ma), ...Object.keys(mb)])).filter(
      (k) => k !== 'Model',
    );
    return keys.map((k) => {
      const av = ma[k];
      const bv = mb[k];
      const numeric = typeof av === 'number' && typeof bv === 'number';
      const delta = numeric ? (bv as number) - (av as number) : null;
      return { key: k, a: av, b: bv, delta };
    });
  }, [runA.data, runB.data]);

  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <nav style={{ fontSize: 12, color: '#94A3B8' }}>
          <Link to={`/workspaces/${wsId}/home`} style={{ color: 'inherit' }}>
            Workspace
          </Link>{' '}
          / Model comparison
        </nav>
        <h1 style={{ fontSize: 24, fontWeight: 700, color: '#0F172A', margin: 0 }}>
          A / B model comparison
        </h1>
        <p style={{ color: '#64748B', fontSize: 13, margin: 0 }}>
          Pick two pipelines from the registry and diff their metrics + diagnostics.
        </p>
      </header>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 12,
        }}
      >
        {([
          ['A', pipeAId, setPipeAId],
          ['B', pipeBId, setPipeBId],
        ] as const).map(([label, value, setter]) => (
          <div key={label} className="card" style={{ padding: 14 }}>
            <div style={{ fontSize: 12, color: '#64748B', marginBottom: 4 }}>Pipeline {label}</div>
            <select
              value={value}
              onChange={(e) => setter(e.target.value)}
              className="input"
              style={{ width: '100%' }}
            >
              <option value="">— Select a pipeline —</option>
              {pipelines.data?.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} {p.model_id ? `(${p.model_id})` : ''}
                </option>
              ))}
            </select>
          </div>
        ))}
      </div>

      {pipeA && pipeB && (
        <div className="card">
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>
            Metrics diff
          </div>
          {metricRows.length === 0 ? (
            <div style={{ color: '#94A3B8', fontSize: 13 }}>
              No leaderboard rows on at least one of the runs.
            </div>
          ) : (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ color: '#64748B', fontSize: 11, textTransform: 'uppercase', textAlign: 'left' }}>
                  <th style={{ padding: '8px 12px' }}>Metric</th>
                  <th style={{ padding: '8px 12px' }}>A</th>
                  <th style={{ padding: '8px 12px' }}>B</th>
                  <th style={{ padding: '8px 12px' }}>Δ (B − A)</th>
                </tr>
              </thead>
              <tbody>
                {metricRows.map((r) => (
                  <tr key={r.key} style={{ borderTop: '1px solid rgba(148,163,184,0.15)' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 500 }}>{r.key}</td>
                    <td style={{ padding: '8px 12px', fontVariantNumeric: 'tabular-nums' }}>
                      {typeof r.a === 'number' ? (r.a as number).toFixed(4) : String(r.a ?? '—')}
                    </td>
                    <td style={{ padding: '8px 12px', fontVariantNumeric: 'tabular-nums' }}>
                      {typeof r.b === 'number' ? (r.b as number).toFixed(4) : String(r.b ?? '—')}
                    </td>
                    <td
                      style={{
                        padding: '8px 12px',
                        fontVariantNumeric: 'tabular-nums',
                        color:
                          typeof r.delta === 'number'
                            ? r.delta > 0
                              ? '#22C55E'
                              : r.delta < 0
                                ? '#EF4444'
                                : '#64748B'
                            : '#94A3B8',
                      }}
                    >
                      {typeof r.delta === 'number' ? r.delta.toFixed(4) : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {pipeA && pipeB && sharedTask && comparableKinds.length > 0 && pipeA.origin_run_id && pipeB.origin_run_id && (
        <div className="space-y-4">
          {comparableKinds.map((kind) => (
            <PlotPair
              key={kind}
              runA={pipeA.origin_run_id!}
              runB={pipeB.origin_run_id!}
              kind={kind}
              title={kind.replace(/_/g, ' ')}
            />
          ))}
        </div>
      )}
    </div>
  );
}
