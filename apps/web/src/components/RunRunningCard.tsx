/**
 * Animated "run is in flight" card.
 *
 * Replaces the empty-leaderboard placeholder while the run is queued or
 * running. Shows a pulsing indicator, the current stage (latest event
 * kind), elapsed time since submit, and a hint to wait.
 *
 * Polls /runs/:id/events every 2s for the latest event so the user sees
 * "fitting Logistic Regression…" → "fitting Random Forest…" → etc.
 */

import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { runsApi } from '@/api/endpoints';

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
  const ms = Date.now() - Date.parse(start);
  if (ms < 1000) return `${ms}ms`;
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${s - m * 60}s`;
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

  // Pull the latest few events to show the current stage.
  const events = useQuery({
    queryKey: ['runs', runId, 'events', 'tail'],
    queryFn: () => runsApi.events(runId, { limit: 10 }),
    refetchInterval: 1500,
  });

  const lastEvent = events.data?.[events.data.length - 1];
  const stage = lastEvent
    ? (KIND_PHRASE[lastEvent.kind] ?? lastEvent.message ?? lastEvent.kind)
    : (STATUS_PHRASE[status] ?? 'Working…');

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

        {/* Skeleton table: hints at where the leaderboard will land */}
        <div className="mt-5 space-y-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <div
              key={i}
              className="h-4 rounded bg-ink-100 dark:bg-ink-800 animate-pulse"
              style={{ animationDelay: `${i * 100}ms`, opacity: 1 - i * 0.15 }}
            />
          ))}
        </div>

        <p className="mt-4 text-xs text-ink-500">
          The full leaderboard will appear here once every model has finished
          training. You can keep this tab open or come back later — the run
          continues server-side.
        </p>
      </div>
    </section>
  );
}
