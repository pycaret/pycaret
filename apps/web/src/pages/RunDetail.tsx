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

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { pipelinesApi, runsApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import { BackButton } from '@/components/BackButton';
import { EventLogDrawer } from '@/components/EventLogDrawer';
import { RunExplainerCard } from '@/components/RunExplainerCard';
import { FailureDebuggerCard } from '@/components/FailureDebuggerCard';
import { TrialsCard } from '@/components/TrialsCard';
import { RunRunningCard } from '@/components/RunRunningCard';
import { WorkerLoadCard } from '@/components/WorkerLoadCard';
import { TrainingChart } from '@/components/TrainingChart';
import { DeployFromPipelineDialog } from '@/components/DeployFromPipelineDialog';
import { useEffect, useState } from 'react';
import type { Run, Pipeline } from '@/api/types';

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

  const r = run.data;
  const snapshot = (r?.snapshot ?? {}) as Record<string, unknown>;
  const snapshotEntries = Object.entries(snapshot);

  // Event log drawer state — auto-opens once when the run starts running
  // so users see live activity without having to discover the button.
  // Open the drawer on the very first render when ``?log=1`` is in the
  // URL — callers from TrialDetail (tune / ensemble submit) navigate
  // here with that flag so the user immediately sees their action's
  // events stream into the run's log without having to click anything.
  const [searchParams, setSearchParams] = useSearchParams();
  const [logOpen, setLogOpen] = useState(
    () => searchParams.get('log') === '1',
  );
  const [autoOpened, setAutoOpened] = useState(false);
  useEffect(() => {
    if (r && !autoOpened && r.status === 'running') {
      setAutoOpened(true);
      setLogOpen(true);
    }
  }, [r, autoOpened]);
  // Strip the ``?log=1`` flag from the URL once we've consumed it so
  // subsequent navigations don't reopen the drawer.
  useEffect(() => {
    if (searchParams.get('log') === '1') {
      const next = new URLSearchParams(searchParams);
      next.delete('log');
      setSearchParams(next, { replace: true });
    }
    // run once on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="space-y-8">
      <header>
        <BackButton />
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
            <button
              type="button"
              className="btn-secondary inline-flex items-center gap-2"
              onClick={() => setLogOpen(true)}
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
              {r && isPending(r) && (
                <span className="ml-0.5 inline-flex h-1.5 w-1.5 rounded-full bg-accent-500 animate-pulse" />
              )}
            </button>
            {r?.status === 'succeeded' && (
              <Link
                to={`/runs/${runId}/model-card`}
                className="btn-secondary"
                title="Plots + metrics for the run's best trial"
              >
                Open best trial →
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

      {/* ────────── Live worker load — per-trial status grid + ETA.
           Renders while trials are in flight AND after they finish so
           the user has a summary of per-trial timings. */}
      {runId && <WorkerLoadCard runId={runId} />}

      {/* ────────── Live training chart — per-trial primary metric
           as the engine emits ``model.created`` events. */}
      {runId && <TrainingChart runId={runId} />}

      {/* ────────── Trials / leaderboard — succeeded runs */}
      {r?.status === 'succeeded' && runId && (
        <TrialsCard
          runId={runId}
          onActionSubmitted={() => setLogOpen(true)}
        />
      )}

      {/* ────────── AI explainer (succeeded runs) / debugger (failed runs) */}
      {r?.status === 'succeeded' && runId && <RunExplainerCard runId={runId} />}
      {r?.status === 'failed' && runId && <FailureDebuggerCard runId={runId} />}

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

      {/* ────────── Promoted pipelines (when any) — sits below the
          request snapshot since it's a downstream artifact, not part of
          the run's own state. */}
      {r?.status === 'succeeded' && r.workspace_id && (
        <PromotedPipelinesSection runId={runId} workspaceId={r.workspace_id} />
      )}

      {/* ────────── Event log drawer — slides in from the right. */}
      <EventLogDrawer
        runId={runId}
        open={logOpen}
        onClose={() => setLogOpen(false)}
      />
    </div>
  );
}

// ─── Promoted pipelines for this run ─────────────────────────────

function PromotedPipelinesSection({
  runId,
  workspaceId,
}: {
  runId: string;
  workspaceId: string;
}) {
  const list = useQuery({
    queryKey: ['pipelines', 'by-workspace', workspaceId],
    queryFn: () => pipelinesApi.list(workspaceId),
    enabled: !!workspaceId,
    staleTime: 30_000,
  });
  const fromThisRun = (list.data ?? []).filter(
    (p) => p.origin_run_id === runId,
  );

  const [deployTarget, setDeployTarget] = useState<Pipeline | null>(null);

  if (list.isPending) return null;
  if (fromThisRun.length === 0) {
    return (
      <section>
        <h2 className="h-section mb-3">Promoted versions</h2>
        <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 px-6 py-8 text-center">
          <p className="text-sm text-ink-500">
            No trial from this run has been promoted yet. Open any trial in the
            leaderboard above and click <span className="font-medium">Promote</span>{' '}
            to register it on the Model registry.
          </p>
        </div>
      </section>
    );
  }

  // Each Pipeline (session-56+) carries the registered_model_id of its
  // matching RegisteredModel — link there directly. Pre-session-56 rows
  // (no registry mirror) fall back to the workspace registry list.
  const registryHref = (p: Pipeline) =>
    p.registered_model_id
      ? `/workspaces/${workspaceId}/models/${p.registered_model_id}`
      : `/workspaces/${workspaceId}/models`;

  return (
    <section>
      <div className="flex items-baseline justify-between mb-3 gap-2 flex-wrap">
        <h2 className="h-section">
          Promoted versions{' '}
          <span className="text-ink-400 font-normal">({fromThisRun.length})</span>
        </h2>
      </div>
      <ul className="rounded-xl bg-white dark:bg-ink-900 border border-ink-200 dark:border-ink-800 shadow-soft-1 divide-y divide-ink-100 dark:divide-ink-800">
        {fromThisRun.map((p) => (
          <li
            key={p.id}
            className="px-4 py-3 flex items-center justify-between gap-3 flex-wrap"
          >
            <div className="min-w-0 flex items-center gap-3">
              <span className="pill-accent">v{p.version}</span>
              <div className="min-w-0">
                <Link
                  to={registryHref(p)}
                  className="text-sm font-medium text-ink-900 dark:text-ink-50 hover:underline truncate"
                >
                  {p.name}
                </Link>
                {p.model_id && (
                  <span className="ml-2 text-xs text-ink-500 font-mono">
                    {p.model_id}
                  </span>
                )}
                {p.description && (
                  <p className="text-xs text-ink-500 mt-0.5 truncate">
                    {p.description}
                  </p>
                )}
              </div>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <Link to={registryHref(p)} className="btn-ghost text-xs">
                Open in registry →
              </Link>
              <button
                type="button"
                className="btn-primary text-sm"
                onClick={() => setDeployTarget(p)}
              >
                Deploy
              </button>
            </div>
          </li>
        ))}
      </ul>
      {deployTarget && (
        <DeployFromPipelineDialog
          open
          onClose={() => setDeployTarget(null)}
          pipelineId={deployTarget.id}
          pipelineName={deployTarget.name}
        />
      )}
    </section>
  );
}
