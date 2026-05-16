/**
 * Live worker-load card for a parallel Run.
 *
 * Phase 0-v2: a ``compare`` Run owns N Trials, each picked up
 * independently by a worker. This card polls ``/runs/:id/trials`` and
 * renders:
 *
 *   - A concurrency gauge — "X of Y trials running in parallel"
 *   - A per-Trial status grid — every algorithm with its own pill
 *   - A completion bar — succeeded / total
 *   - An ETA — projected finish time from completed-trial durations
 *
 * Poll cadence is 1500ms while ANY trial is still queued/running,
 * stops once everything is terminal.
 */

import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { runsApi } from '@/api/endpoints';

type TrialStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';

interface TrialLite {
  id: string;
  model_id: string;
  status: TrialStatus;
  duration_ms: number | null;
  rank: number | null;
  metrics: Record<string, number>;
}

const STATUS_DOT: Record<TrialStatus, string> = {
  queued: 'bg-ink-300 dark:bg-ink-700',
  running:
    'bg-accent-500 ring-2 ring-accent-200 dark:ring-accent-500/20 animate-pulse',
  succeeded: 'bg-success-500',
  failed: 'bg-danger-500',
  cancelled: 'bg-amber-400',
};

const STATUS_TEXT: Record<TrialStatus, string> = {
  queued: 'text-ink-500',
  running: 'text-accent-700 dark:text-accent-300 font-medium',
  succeeded: 'text-success-700 dark:text-success-400',
  failed: 'text-danger-700 dark:text-danger-400',
  cancelled: 'text-amber-700 dark:text-amber-400',
};

function fmtMs(ms: number | null | undefined): string {
  if (!ms) return '—';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${(s - m * 60).toFixed(0)}s`;
}

export function WorkerLoadCard({ runId }: { runId: string }) {
  // Poll while anything's still in flight; stop polling once all
  // trials are terminal (saves bandwidth + DB hits).
  const trials = useQuery({
    queryKey: ['runs', runId, 'trials', 'worker-load'],
    queryFn: () => runsApi.trials(runId),
    refetchInterval: (q) => {
      const items = (q.state.data?.items ?? []) as TrialLite[];
      if (items.length === 0) return 1500;
      const stillRunning = items.some(
        (t) => t.status === 'queued' || t.status === 'running',
      );
      return stillRunning ? 1500 : false;
    },
  });

  const stats = useMemo(() => {
    const items = (trials.data?.items ?? []) as TrialLite[];
    const total = items.length;
    const running = items.filter((t) => t.status === 'running').length;
    const succeeded = items.filter((t) => t.status === 'succeeded').length;
    const failed = items.filter((t) => t.status === 'failed').length;
    const queued = items.filter((t) => t.status === 'queued').length;
    const completed = succeeded + failed;
    const completedDurations = items
      .filter((t) => t.duration_ms != null && t.status === 'succeeded')
      .map((t) => t.duration_ms as number);
    const avgMs =
      completedDurations.length > 0
        ? completedDurations.reduce((a, b) => a + b, 0) /
          completedDurations.length
        : null;
    // Naive ETA: assume the remaining queued+running trials each take
    // about ``avgMs``, divided by current concurrency (or 1 if nothing
    // is running yet).
    const concurrency = Math.max(running, 1);
    const remaining = queued + running;
    const etaMs =
      avgMs && remaining > 0 ? (avgMs * remaining) / concurrency : null;
    return {
      total,
      running,
      succeeded,
      failed,
      queued,
      completed,
      avgMs,
      etaMs,
      pctComplete: total > 0 ? (completed / total) * 100 : 0,
    };
  }, [trials.data]);

  if (trials.isLoading) {
    return <div className="card text-sm text-ink-500">Loading worker activity…</div>;
  }

  const items = (trials.data?.items ?? []) as TrialLite[];
  if (items.length === 0) return null;

  const allTerminal = stats.queued === 0 && stats.running === 0;

  return (
    <section className="card space-y-4">
      <header className="flex items-baseline justify-between gap-4">
        <div>
          <h3 className="h-section">
            {allTerminal ? 'Compare summary' : 'Workers'}
          </h3>
          <p className="muted small mt-1">
            {allTerminal
              ? `Finished — ${stats.succeeded} succeeded${stats.failed ? `, ${stats.failed} failed` : ''}.`
              : `${stats.running} running · ${stats.queued} queued · ${stats.completed}/${stats.total} done`}
          </p>
        </div>
        {!allTerminal && stats.etaMs != null && (
          <div className="text-right">
            <p className="muted small">ETA</p>
            <p className="text-sm font-medium tabular-nums">
              ~{fmtMs(stats.etaMs)} remaining
            </p>
          </div>
        )}
      </header>

      {/* Progress bar */}
      <div>
        <div className="flex justify-between text-xs muted mb-1">
          <span>{Math.round(stats.pctComplete)}% complete</span>
          <span>
            avg per trial: <span className="tabular-nums">{fmtMs(stats.avgMs)}</span>
          </span>
        </div>
        <div className="h-2 w-full rounded-full bg-ink-100 dark:bg-ink-800 overflow-hidden">
          <div
            className="h-full bg-accent-500 transition-all duration-500"
            style={{ width: `${stats.pctComplete}%` }}
          />
        </div>
      </div>

      {/* Per-trial grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
        {items.map((t) => (
          <TrialChip key={t.id} trial={t} />
        ))}
      </div>
    </section>
  );
}

function TrialChip({ trial }: { trial: TrialLite }) {
  const primary = (() => {
    const m = trial.metrics ?? {};
    for (const k of ['Accuracy', 'AUC', 'F1', 'R2'] as const) {
      if (typeof m[k] === 'number') return { name: k, value: m[k] };
    }
    return null;
  })();

  return (
    <div className="flex items-center gap-2.5 rounded-md border border-ink-200 dark:border-ink-800 px-3 py-2">
      <span
        aria-hidden
        className={`h-2.5 w-2.5 rounded-full shrink-0 ${STATUS_DOT[trial.status]}`}
      />
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium truncate">{trial.model_id}</p>
        <p className={`text-xs truncate ${STATUS_TEXT[trial.status]}`}>
          {trial.status === 'succeeded' && primary
            ? `${primary.name} ${primary.value.toFixed(3)}`
            : trial.status === 'running'
              ? 'training…'
              : trial.status === 'queued'
                ? 'waiting'
                : trial.status === 'failed'
                  ? 'failed'
                  : trial.status}
        </p>
      </div>
      {trial.status === 'succeeded' && trial.duration_ms != null && (
        <span className="text-xs muted tabular-nums shrink-0">
          {fmtMs(trial.duration_ms)}
        </span>
      )}
    </div>
  );
}
