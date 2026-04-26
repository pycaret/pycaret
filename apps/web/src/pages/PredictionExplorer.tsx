/**
 * Prediction Explorer — `/workspaces/:wsId/predictions`.
 *
 * Live-test deployed pipelines against ad-hoc input. Three sections:
 *  - Deployment picker (radio cards with health + p50/p95).
 *  - JSON input editor + sample row pre-fill.
 *  - Result panel: prediction value, score, latency, recent history.
 *
 * Replaces the today's Pipelines deep-link with a screen built for
 * analysts who want to feel the model. Surfaces the Deployment p50/p95
 * stats already collected by the registry.
 */

import { useMemo, useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';

import { deploymentsApi } from '../api/endpoints';
import type { Deployment } from '../api/types';

interface RecentCall {
  ts: number;
  latency_ms: number;
  prediction: unknown;
  ok: boolean;
}

export function PredictionExplorer() {
  const { wsId = '' } = useParams<{ wsId: string }>();
  const [selected, setSelected] = useState<string>('');
  const [inputJson, setInputJson] = useState<string>('[\n  {}\n]');
  const [recent, setRecent] = useState<RecentCall[]>([]);
  const [error, setError] = useState<string | null>(null);

  const deployments = useQuery({
    queryKey: ['deployments', wsId],
    queryFn: () => deploymentsApi.list(wsId),
    enabled: !!wsId,
  });

  const active = useMemo<Deployment | undefined>(
    () => deployments.data?.find((d) => d.id === selected) ?? deployments.data?.[0],
    [deployments.data, selected],
  );

  const callPredict = useMutation({
    mutationFn: async () => {
      if (!active) throw new Error('Pick a deployment first.');
      const parsed = JSON.parse(inputJson);
      const rows = Array.isArray(parsed) ? parsed : [parsed];
      const start = performance.now();
      const result = await deploymentsApi.predict(active.endpoint_slug, { rows });
      return { result, latency: performance.now() - start };
    },
    onSuccess: ({ result, latency }) => {
      setError(null);
      setRecent((prev) =>
        [
          {
            ts: Date.now(),
            latency_ms: latency,
            prediction: result?.predictions?.[0] ?? null,
            ok: true,
          },
          ...prev,
        ].slice(0, 12),
      );
    },
    onError: (e) => {
      setError((e as Error).message);
      setRecent((prev) =>
        [
          { ts: Date.now(), latency_ms: 0, prediction: null, ok: false },
          ...prev,
        ].slice(0, 12),
      );
    },
  });

  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <nav style={{ fontSize: 12, color: '#94A3B8' }}>
          <Link to={`/workspaces/${wsId}/home`} style={{ color: 'inherit' }}>
            Workspace
          </Link>{' '}
          / Prediction explorer
        </nav>
        <h1 style={{ fontSize: 24, fontWeight: 700, color: '#0F172A', margin: 0 }}>
          Prediction explorer
        </h1>
        <p style={{ color: '#64748B', fontSize: 13, margin: 0 }}>
          Test a deployment with hand-crafted or sampled rows and watch latency in real time.
        </p>
      </header>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '320px 1fr',
          gap: 16,
          alignItems: 'start',
        }}
      >
        <div className="card" style={{ padding: 12 }}>
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>Deployment</div>
          {deployments.isLoading ? (
            <div style={{ color: '#94A3B8', fontSize: 13 }}>Loading…</div>
          ) : deployments.data?.length === 0 ? (
            <div style={{ color: '#94A3B8', fontSize: 13 }}>
              No deployments yet — promote a pipeline first.
            </div>
          ) : (
            <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 6 }}>
              {deployments.data!.map((d) => (
                <li key={d.id}>
                  <button
                    onClick={() => setSelected(d.id)}
                    style={{
                      width: '100%',
                      textAlign: 'left',
                      padding: '8px 12px',
                      borderRadius: 10,
                      border:
                        active?.id === d.id
                          ? '1px solid #5B8DEF'
                          : '1px solid rgba(148,163,184,0.2)',
                      background:
                        active?.id === d.id
                          ? 'rgba(91,141,239,0.08)'
                          : 'transparent',
                      cursor: 'pointer',
                    }}
                  >
                    <div style={{ fontWeight: 500, fontSize: 13, color: '#0F172A' }}>
                      {d.endpoint_slug}
                    </div>
                    <div style={{ fontSize: 11, color: '#64748B', marginTop: 2 }}>
                      p50: {d.p50_latency_ms?.toFixed?.(1) ?? '—'} ms · p95:{' '}
                      {d.p95_latency_ms?.toFixed?.(1) ?? '—'} ms
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="card space-y-3">
          <div style={{ fontSize: 14, fontWeight: 600 }}>Request body</div>
          <textarea
            value={inputJson}
            onChange={(e) => setInputJson(e.target.value)}
            style={{
              width: '100%',
              minHeight: 200,
              padding: 12,
              borderRadius: 10,
              border: '1px solid rgba(148,163,184,0.25)',
              fontFamily: 'ui-monospace, monospace',
              fontSize: 13,
              resize: 'vertical',
            }}
          />
          {error && (
            <div
              style={{
                color: '#7f1d1d',
                background: 'rgba(239,68,68,0.08)',
                padding: 10,
                borderRadius: 8,
                fontSize: 12,
              }}
            >
              {error}
            </div>
          )}
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              className="btn-primary"
              disabled={!active || callPredict.isPending}
              onClick={() => callPredict.mutate()}
            >
              {callPredict.isPending ? 'Predicting…' : 'Predict'}
            </button>
            <button
              className="btn-secondary"
              onClick={() => setInputJson('[\n  {}\n]')}
            >
              Reset
            </button>
          </div>
        </div>
      </div>

      <div className="card">
        <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>Recent calls</div>
        {recent.length === 0 ? (
          <div style={{ color: '#94A3B8', fontSize: 13 }}>
            No calls yet. Hit Predict to populate this list.
          </div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ color: '#64748B', fontSize: 11, textTransform: 'uppercase', textAlign: 'left' }}>
                <th style={{ padding: '8px 12px' }}>When</th>
                <th style={{ padding: '8px 12px' }}>Status</th>
                <th style={{ padding: '8px 12px' }}>Latency</th>
                <th style={{ padding: '8px 12px' }}>Prediction</th>
              </tr>
            </thead>
            <tbody>
              {recent.map((c) => (
                <tr
                  key={c.ts}
                  style={{ borderTop: '1px solid rgba(148,163,184,0.15)' }}
                >
                  <td style={{ padding: '8px 12px', color: '#64748B' }}>
                    {new Date(c.ts).toLocaleTimeString()}
                  </td>
                  <td style={{ padding: '8px 12px' }}>
                    <span
                      style={{
                        fontSize: 11,
                        fontWeight: 600,
                        padding: '2px 8px',
                        borderRadius: 999,
                        color: c.ok ? '#22C55E' : '#EF4444',
                        background: c.ok ? 'rgba(34,197,94,0.12)' : 'rgba(239,68,68,0.12)',
                      }}
                    >
                      {c.ok ? 'OK' : 'ERROR'}
                    </span>
                  </td>
                  <td style={{ padding: '8px 12px', fontVariantNumeric: 'tabular-nums' }}>
                    {c.latency_ms.toFixed(1)} ms
                  </td>
                  <td
                    style={{
                      padding: '8px 12px',
                      fontFamily: 'ui-monospace, monospace',
                      fontSize: 12,
                      maxWidth: 360,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {JSON.stringify(c.prediction)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
