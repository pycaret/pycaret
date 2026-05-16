/**
 * Live "run is in flight" card — fed by the engine event stream.
 *
 * The engine emits per-model `model.compared` events while
 * `compare_models` runs. We poll `/runs/:id/events?tail=true` every
 * 1.5s and use those events to render:
 *
 *   - A stage line (latest event with friendly text).
 *   - A real progress bar (succeeded / total candidates) when compare is
 *     in flight, plus an ETA based on the average per-model duration.
 *   - A live mini-leaderboard of the models trained so far, ranked by
 *     the run's primary sort metric so the user sees the leader emerge.
 *
 * No more skeleton-strip placeholder — actual data lands as soon as the
 * first model finishes.
 */

import { useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { runsApi } from '@/api/endpoints';
import type { RunEvent } from '@/api/types';

const STATUS_PHRASE: Record<string, string> = {
  queued: 'Queued — waiting for a worker',
  running: 'Training models',
};

const KIND_PHRASE: Record<string, string> = {
  'experiment.started': 'Starting experiment',
  'experiment.fitted': 'Preprocessing complete',
  'model.compare.started': 'Comparing models',
  'model.create.started': 'Training model',
  'model.tune.started': 'Tuning hyperparameters',
  'model.tuned': 'Tuning complete',
  'model.compare.finished': 'Ranking models',
  'model.predicted': 'Generating predictions',
};

function fmtElapsed(start: string | null | undefined): string {
  if (!start) return '—';
  // Some servers emit ISO without a timezone suffix; JS then parses it as
  // local time and the diff comes out negative when the user is east of
  // UTC. Treat ambiguous values as UTC by appending Z when no offset is
  // present (timezone-aware ISO already has +/- or Z and is left alone).
  const normalised = /[zZ]|[+-]\d{2}:?\d{2}$/.test(start) ? start : `${start}Z`;
  const ms = Date.now() - Date.parse(normalised);
  if (!Number.isFinite(ms) || ms < 0) return '—';
  return fmtDuration(ms);
}

function fmtDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s - m * 60}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m - h * 60}m`;
}

interface CompareProgress {
  total: number;
  succeededCount: number;
  failedCount: number;
  /** Avg ms per succeeded model — drives the ETA. */
  avgDurationMs: number | null;
}

/** Walk the event tail and pull out enough to render the progress bar + ETA. */
function deriveProgress(events: RunEvent[] | undefined): CompareProgress | null {
  if (!events || events.length === 0) return null;
  let total = 0;
  let succeeded = 0;
  let failed = 0;
  let durationSum = 0;

  for (const e of events) {
    if (e.kind === 'model.compare.started') {
      const p = (e.payload ?? {}) as { n_candidates?: number };
      if (typeof p.n_candidates === 'number') total = p.n_candidates;
    } else if (e.kind === 'model.compared') {
      const p = (e.payload ?? {}) as { status?: string };
      if (p.status === 'failed') {
        failed += 1;
      } else {
        succeeded += 1;
        if (typeof e.duration_ms === 'number') durationSum += e.duration_ms;
      }
    }
  }
  if (total === 0 && succeeded === 0 && failed === 0) return null;
  return {
    total,
    succeededCount: succeeded,
    failedCount: failed,
    avgDurationMs: succeeded > 0 ? durationSum / succeeded : null,
  };
}

export interface RunRunningCardProps {
  runId: string;
  status: 'queued' | 'running';
  startedAt: string | null | undefined;
}

export function RunRunningCard({ runId, status, startedAt }: RunRunningCardProps) {
  // Re-render each second so the elapsed counter ticks.
  const [, force] = useState(0);
  useEffect(() => {
    const id = setInterval(() => force((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, []);

  const events = useQuery({
    queryKey: ['runs', runId, 'events', 'tail'],
    queryFn: () => runsApi.events(runId, { limit: 200, tail: true }),
    refetchInterval: 1500,
  });

  const lastEvent = events.data?.[events.data.length - 1];
  const stage = lastEvent
    ? (KIND_PHRASE[lastEvent.kind] ?? lastEvent.message ?? lastEvent.kind)
    : (STATUS_PHRASE[status] ?? 'Working…');

  const progress = useMemo(() => deriveProgress(events.data), [events.data]);
  const completed =
    (progress?.succeededCount ?? 0) + (progress?.failedCount ?? 0);
  const total = progress?.total ?? 0;
  const pct = total > 0 ? Math.min(100, (completed / total) * 100) : 0;
  const eta =
    progress && total > 0 && completed > 0 && completed < total && progress.avgDurationMs
      ? progress.avgDurationMs * (total - completed)
      : null;

  return (
    <section>
      <div className="card relative overflow-hidden">
        {/* Subtle animated stripe behind the content */}
        <div className="absolute inset-x-0 top-0 h-0.5 bg-gradient-to-r from-transparent via-accent-500 to-transparent animate-pulse" />

        <div className="flex items-start gap-4">
          <span className="relative shrink-0 mt-1">
            <span className="block h-2.5 w-2.5 rounded-full bg-accent-500 animate-ping absolute inset-0" />
            <span className="block h-2.5 w-2.5 rounded-full bg-accent-500" />
          </span>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-ink-900 dark:text-ink-50">
              {stage}
            </p>
            {lastEvent?.message && lastEvent.message !== stage && (
              <p className="mt-1 text-xs text-ink-500 truncate">
                {lastEvent.message}
              </p>
            )}
          </div>
          <div className="text-right shrink-0">
            <p className="text-[11px] uppercase tracking-wider text-ink-500 font-medium">
              Elapsed
            </p>
            <p className="text-sm font-mono tabular-nums text-ink-700 dark:text-ink-300">
              {fmtElapsed(startedAt)}
            </p>
          </div>
        </div>

        {/* Progress + ETA — shown once compare has started. */}
        {progress && total > 0 && (
          <div className="mt-5">
            <div className="flex items-baseline justify-between text-xs text-ink-500 mb-1.5">
              <span>
                <span className="text-ink-900 dark:text-ink-50 font-medium tabular-nums">
                  {completed}
                </span>{' '}
                of {total} models trained
                {progress.failedCount > 0 && (
                  <span className="ml-2 text-danger-600">
                    ({progress.failedCount} failed)
                  </span>
                )}
              </span>
              {eta !== null && (
                <span className="tabular-nums">
                  ~{fmtDuration(eta)} remaining
                </span>
              )}
            </div>
            <div className="h-2 rounded-full bg-ink-100 dark:bg-ink-800 overflow-hidden">
              <div
                className="h-full bg-accent-500 transition-all duration-500"
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        )}

        {/* Pre-compare placeholder — keeps the card from collapsing while
            the first events are still flowing. Once the progress bar above
            takes over there's no need for skeletons. */}
        {!progress && (
          <div className="mt-5 space-y-2">
            {Array.from({ length: 4 }).map((_, i) => (
              <div
                key={i}
                className="h-4 rounded bg-ink-100 dark:bg-ink-800 animate-pulse"
                style={{ animationDelay: `${i * 100}ms`, opacity: 1 - i * 0.15 }}
              />
            ))}
          </div>
        )}

        <p className="mt-4 text-xs text-ink-500">
          The leaderboard lands below as soon as the run finishes. Open the
          event log drawer for live per-model details — the run continues
          server-side if you close this tab.
        </p>
      </div>
    </section>
  );
}
