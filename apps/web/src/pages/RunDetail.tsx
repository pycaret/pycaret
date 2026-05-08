/**
 * /runs/:runId — dedicated run detail screen.
 *
 * Three concerns:
 *  1. Status header + actions (cancel if pending, promote if succeeded).
 *  2. Live event stream (WebSocket) + leaderboard (when available).
 *  3. Config snapshot — exactly what was submitted, for reproducibility.
 *
 * Polls the run row every 2s while status is queued/running, so the header
 * + leaderboard stay fresh even if the WebSocket drops. Once the run is
 * terminal, polling stops.
 */

import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import { runsApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import { EventStream } from '@/components/EventStream';
import { RunExplainerCard } from '@/components/RunExplainerCard';
import { FailureDebuggerCard } from '@/components/FailureDebuggerCard';
import { TrialsCard } from '@/components/TrialsCard';
import { RunRunningCard } from '@/components/RunRunningCard';
import type { Run } from '@/api/types';

const STATUS_PILL: Record<string, string> = {
  queued: 'pill-neutral',
  running: 'pill-accent',
  succeeded: 'pill-success',
  failed: 'pill-danger',
  cancelled: 'pill-warn',
};

function formatDuration(ms: number | null | undefined): string {
  if (ms == null) return '—';
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${(s - m * 60).toFixed(0)}s`;
}

function isPending(r: Run | undefined): boolean {
  return !!r && (r.status === 'queued' || r.status === 'running');
}

export function RunDetail() {
  const { runId = '' } = useParams<{ runId: string }>();
  const qc = useQueryClient();

  const run = useQuery({
    queryKey: ['runs', runId],
    queryFn: () => runsApi.get(runId),
    enabled: !!runId,
    refetchInterval: (q) => (isPending(q.state.data) ? 2000 : false),
  });

  const cancel = useMutation({
    mutationFn: () => runsApi.cancel(runId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['runs', runId] }),
  });

  const [promoteName, setPromoteName] = useState('');
  const promote = useMutation({
    mutationFn: () => runsApi.promote(runId, { name: promoteName.trim() }),
    onSuccess: () => setPromoteName(''),
  });

  const r = run.data;
  const snapshot = (r?.snapshot ?? {}) as Record<string, unknown>;
  const snapshotEntries = Object.entries(snapshot);
  const terminal = r && (r.status === 'succeeded' || r.status === 'failed' || r.status === 'cancelled');

  return (
    <div className="space-y-8">
      <header>
        <nav className="text-xs text-ink-500 mb-2">
          <Link to="/" className="hover:text-ink-900 dark:hover:text-ink-50">
            Workspaces
          </Link>
          <span className="mx-1.5 text-ink-300">/</span>
          <span className="text-ink-700 dark:text-ink-300 font-mono">
            Run · {runId}
          </span>
        </nav>
        <div className="flex items-end justify-between gap-4">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="h-page">Run</h1>
              {r?.status && (
                <span className={`${STATUS_PILL[r.status] ?? 'pill-neutral'} capitalize`}>
                  {r.status}
                </span>
              )}
            </div>
            {r?.duration_ms != null && (
              <p className="mt-2 text-sm text-ink-500">
                ran for{' '}
                <span className="font-mono text-ink-700 dark:text-ink-300">
                  {formatDuration(r.duration_ms)}
                </span>
              </p>
            )}
            {r?.error && (
              <pre className="mt-3 card text-xs whitespace-pre-wrap text-danger-600 dark:text-danger-500">
                {r.error}
              </pre>
            )}
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {r?.status === 'succeeded' && (
              <Link to={`/runs/${runId}/model-card`} className="btn-secondary">
                Open model card →
              </Link>
            )}
            {r?.status === 'succeeded' &&
              ((r.snapshot ?? {}) as Record<string, unknown>).task === 'time_series' && (
                <Link to={`/runs/${runId}/forecast`} className="btn-secondary">
                  Forecast workbench
                </Link>
              )}
            {r && isPending(r) && (
              <button
                className="btn-danger"
                onClick={() => cancel.mutate()}
                disabled={cancel.isPending}
              >
                {cancel.isPending ? 'Cancelling…' : 'Cancel'}
              </button>
            )}
          </div>
        </div>
        {run.error && <p className="error mt-2">{errorMessage(run.error)}</p>}
        {cancel.error && <p className="error mt-2">{errorMessage(cancel.error)}</p>}
      </header>

      {/* ────────── While pending: animated progress card */}
      {r && isPending(r) && (
        <RunRunningCard
          runId={runId}
          status={r.status as 'queued' | 'running'}
          startedAt={r.started_at ?? r.created_at}
        />
      )}

      {/* ────────── Live event stream — collapses behind the progress card */}
      <section>
        {r && <EventStream runId={runId} />}
      </section>

      {/* ────────── Trials / leaderboard — succeeded runs */}
      {r?.status === 'succeeded' && runId && <TrialsCard runId={runId} />}

      {/* ────────── AI explainer (succeeded runs) / debugger (failed runs) */}
      {r?.status === 'succeeded' && runId && <RunExplainerCard runId={runId} />}
      {r?.status === 'failed' && runId && <FailureDebuggerCard runId={runId} />}

      {/* ────────── Promote */}
      {r?.status === 'succeeded' && (
        <section>
          <h2 className="h-section mb-3">Promote fitted pipeline</h2>
          <div className="card flex items-end gap-3">
            <div className="flex-1">
              <label className="field" htmlFor="promote-name">
                Pipeline name
              </label>
              <input
                id="promote-name"
                className="input"
                value={promoteName}
                onChange={(e) => setPromoteName(e.target.value)}
                placeholder="e.g. churn-model-v1"
                disabled={promote.isPending || promote.isSuccess}
              />
              <p className="hint mt-1">
                Promotes the fitted pipeline into the workspace registry so it can be
                deployed behind a slug.
              </p>
            </div>
            <button
              className="btn-primary"
              onClick={() => promote.mutate()}
              disabled={
                !promoteName.trim() || promote.isPending || promote.isSuccess
              }
            >
              {promote.isPending
                ? 'Promoting…'
                : promote.isSuccess
                  ? 'Promoted ✓'
                  : 'Promote'}
            </button>
          </div>
          {promote.error && <p className="error mt-2">{errorMessage(promote.error)}</p>}
          {promote.data && (
            <p className="hint mt-2">
              Created pipeline{' '}
              <Link
                to={`/workspaces/${promote.data.workspace_id}/pipelines/${promote.data.id}`}
                className="font-mono text-accent-600 hover:underline"
              >
                {promote.data.name}
              </Link>
              . Visit its page to deploy it behind a slug.
            </p>
          )}
        </section>
      )}

      {/* ────────── Snapshot */}
      <section>
        <h2 className="h-section">Request snapshot</h2>
        <p className="text-xs text-ink-500 mb-3">
          Frozen copy of every parameter the engine received when this run was
          submitted. Use it to reproduce the result.
        </p>
        {snapshotEntries.length === 0 ? (
          <p className="text-sm text-ink-500">Snapshot not available.</p>
        ) : (
          <dl className="card divide-y divide-ink-100 dark:divide-ink-800 p-0">
            {snapshotEntries.map(([k, v]) => (
              <div
                key={k}
                className="grid grid-cols-3 gap-4 px-4 py-2.5 text-sm"
              >
                <dt className="text-ink-500 font-mono text-xs col-span-1 truncate">
                  {k}
                </dt>
                <dd className="text-ink-900 dark:text-ink-50 font-mono text-xs col-span-2 break-all">
                  {v == null
                    ? <span className="text-ink-400">—</span>
                    : typeof v === 'object'
                      ? JSON.stringify(v)
                      : String(v)}
                </dd>
              </div>
            ))}
          </dl>
        )}
      </section>
    </div>
  );
}
