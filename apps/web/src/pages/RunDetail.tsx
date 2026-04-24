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
import { Leaderboard } from '@/components/Leaderboard';
import type { Run } from '@/api/types';

const STATUS_TONE: Record<string, string> = {
  queued: 'text-ink-200/60',
  running: 'text-accent-400',
  succeeded: 'text-success-500',
  failed: 'text-danger-500',
  cancelled: 'text-warn-500',
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

function asLeaderboardRows(
  lb: Run['leaderboard'],
): Record<string, unknown>[] | null {
  if (!lb) return null;
  if (Array.isArray(lb)) return lb;
  return null;
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
        <nav className="text-xs text-ink-200/60 mb-2">
          <Link to="/" className="hover:text-ink-100">
            Workspaces
          </Link>
          <span className="mx-1">/</span>
          <span>Run {runId.slice(0, 8)}…</span>
        </nav>
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold flex items-center gap-3">
              Run
              <span className={STATUS_TONE[r?.status ?? 'queued'] ?? ''}>
                {r?.status ?? '…'}
              </span>
            </h1>
            <p className="mt-1 text-sm text-ink-200/70">
              <span className="font-mono text-xs text-ink-200/50">{runId}</span>
              {r?.duration_ms != null && (
                <>
                  {' '}
                  • ran for{' '}
                  <span className="font-mono">{formatDuration(r.duration_ms)}</span>
                </>
              )}
            </p>
            {r?.error && (
              <pre className="mt-3 card text-xs whitespace-pre-wrap text-danger-500">
                {r.error}
              </pre>
            )}
          </div>
          <div className="flex items-center gap-2 shrink-0">
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

      {/* ────────── Live stream (takes the full width; leaderboard stacks below) */}
      <section>
        {r && <EventStream runId={runId} />}
      </section>

      {/* ────────── Leaderboard */}
      <section>
        <h2 className="text-sm font-medium text-ink-100 mb-3">Leaderboard</h2>
        <Leaderboard rows={asLeaderboardRows(r?.leaderboard ?? null)} />
      </section>

      {/* ────────── Promote */}
      {r?.status === 'succeeded' && (
        <section>
          <h2 className="text-sm font-medium text-ink-100 mb-3">Promote fitted pipeline</h2>
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
                className="font-mono text-accent-400 hover:underline"
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
        <h2 className="text-sm font-medium text-ink-100 mb-3">Request snapshot</h2>
        {snapshotEntries.length === 0 ? (
          <p className="hint">Snapshot not available.</p>
        ) : (
          <dl className="card grid gap-2 md:grid-cols-2">
            {snapshotEntries.map(([k, v]) => (
              <div key={k} className="flex justify-between gap-4">
                <dt className="text-sm text-ink-200/70 font-mono">{k}</dt>
                <dd className="text-sm text-ink-100 font-mono text-right break-all">
                  {typeof v === 'object' && v !== null
                    ? JSON.stringify(v)
                    : String(v ?? '—')}
                </dd>
              </div>
            ))}
          </dl>
        )}
      </section>

      {terminal && (
        <p className="hint text-center">
          Run reached terminal state <code>{r?.status}</code>. The event stream has
          closed.
        </p>
      )}
    </div>
  );
}
