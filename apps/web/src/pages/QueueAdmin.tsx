/**
 * /admin/queues — Phase 14 queue + worker visibility.
 *
 * Two cards side-by-side: per-queue depth (queued / running / 1h
 * throughput) and active workers (id, in-flight jobs, last lock).
 * Polls every 5s while open so the dashboard reads "live" without
 * the user needing to refresh.
 */

import { useQuery } from '@tanstack/react-query';
import { BackButton } from '@/components/BackButton';
import { queueAdminApi } from '@/api/endpoints';

function ageSeconds(iso: string | null): number | null {
  if (!iso) return null;
  return Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 1000));
}

function ageLabel(iso: string | null): string {
  const s = ageSeconds(iso);
  if (s === null) return '—';
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  return `${Math.round(s / 3600)}h ago`;
}

export function QueueAdmin() {
  const queues = useQuery({
    queryKey: ['admin', 'queues'],
    queryFn: queueAdminApi.queues,
    refetchInterval: 5000,
  });
  const workers = useQuery({
    queryKey: ['admin', 'workers'],
    queryFn: queueAdminApi.workers,
    refetchInterval: 5000,
  });
  const system = useQuery({
    queryKey: ['admin', 'system'],
    queryFn: queueAdminApi.system,
    refetchInterval: 15000,
  });

  return (
    <div className="space-y-6">
      <header>
        <BackButton />
        <h1 className="h-page mt-2">Queues & workers</h1>
        <p className="muted small">
          Phase 14 control surface. Per-class queue depth and active
          worker locks. Auto-refreshes every 5s.
        </p>
      </header>

      <section>
        <h2 className="h-section mb-3">System capabilities</h2>
        {system.isLoading && (
          <div className="card text-sm text-ink-500">Loading…</div>
        )}
        {system.data && (
          <div className="card grid gap-3 md:grid-cols-3 text-sm">
            <div>
              <div className="muted small">Runs backend</div>
              <div className="font-mono mt-0.5">{system.data.runs_backend}</div>
              {system.data.runs_backend === 'redis' && (
                <div className="mt-1 text-xs">
                  Redis{' '}
                  {system.data.redis.healthy ? (
                    <span className="pill-success">healthy</span>
                  ) : (
                    <span className="pill-danger">unreachable</span>
                  )}
                  {system.data.redis.error && (
                    <span className="ml-2 text-danger-600">
                      {system.data.redis.error}
                    </span>
                  )}
                </div>
              )}
            </div>
            <div>
              <div className="muted small">GPU</div>
              {system.data.gpu.available ? (
                <>
                  <div className="font-mono mt-0.5">
                    {system.data.gpu.count}× device
                    {system.data.gpu.count === 1 ? '' : 's'}
                  </div>
                  <div className="text-xs mt-1 text-ink-500 truncate">
                    {system.data.gpu.devices.join(', ')}
                  </div>
                  <div className="text-xs mt-1 muted">
                    detected via{' '}
                    <code className="font-mono">{system.data.gpu.source}</code>
                  </div>
                </>
              ) : (
                <>
                  <div className="font-mono mt-0.5 text-ink-500">
                    none detected
                  </div>
                  {system.data.gpu.error && (
                    <div className="text-xs mt-1 text-ink-500">
                      {system.data.gpu.error}
                    </div>
                  )}
                </>
              )}
            </div>
            <div>
              <div className="muted small">Worker queues (env)</div>
              <div className="flex flex-wrap gap-1 mt-0.5">
                {system.data.worker_queues.map((q) => (
                  <span
                    key={q}
                    className={
                      q === 'gpu' && !system.data?.gpu.available
                        ? 'pill-warn'
                        : 'pill-neutral'
                    }
                    title={
                      q === 'gpu' && !system.data?.gpu.available
                        ? 'No GPU available — gpu queue jobs will be released'
                        : undefined
                    }
                  >
                    {q}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </section>

      <section>
        <h2 className="h-section mb-3">Queues</h2>
        {queues.isLoading && (
          <div className="card text-sm text-ink-500">Loading…</div>
        )}
        {queues.error && (
          <div className="card text-sm text-danger-600">
            {String((queues.error as Error).message)}
          </div>
        )}
        {queues.data && queues.data.queues.length === 0 && (
          <div className="card text-sm text-ink-500">
            No queues have seen traffic yet. Submit a run to populate the
            default queue.
          </div>
        )}
        {queues.data && queues.data.queues.length > 0 && (
          <div className="card overflow-hidden p-0">
            <table className="w-full text-sm">
              <thead className="bg-white text-ink-500 dark:bg-ink-900 border-b border-ink-200 dark:border-ink-800">
                <tr>
                  <th className="px-4 py-2 text-left font-medium">Queue</th>
                  <th className="px-4 py-2 text-right font-medium">Queued</th>
                  <th className="px-4 py-2 text-right font-medium">Running</th>
                  <th className="px-4 py-2 text-right font-medium">Failed</th>
                  <th className="px-4 py-2 text-right font-medium">
                    Succeeded
                  </th>
                  <th className="px-4 py-2 text-right font-medium">
                    Throughput / hr
                  </th>
                </tr>
              </thead>
              <tbody>
                {queues.data.queues.map((q) => (
                  <tr
                    key={q.name}
                    className="border-t border-ink-200 dark:border-ink-800"
                  >
                    <td className="px-4 py-2 font-medium">{q.name}</td>
                    <td className="px-4 py-2 text-right tabular-nums">
                      {q.queued > 0 ? (
                        <span className="pill-accent">{q.queued}</span>
                      ) : (
                        q.queued
                      )}
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums">
                      {q.running}
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums">
                      {q.failed > 0 ? (
                        <span className="text-danger-600">{q.failed}</span>
                      ) : (
                        q.failed
                      )}
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums">
                      {q.succeeded}
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums">
                      {q.recent_throughput_1h}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section>
        <h2 className="h-section mb-3">Active workers</h2>
        {workers.isLoading && (
          <div className="card text-sm text-ink-500">Loading…</div>
        )}
        {workers.data && workers.data.workers.length === 0 && (
          <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 p-8 text-center">
            <p className="text-sm font-semibold text-ink-900 dark:text-ink-50">
              No workers currently holding a lock
            </p>
            <p className="mt-1 text-xs text-ink-500">
              Workers only appear here while they're processing a job.
              Start one with <code>pycaret-server worker</code>.
            </p>
          </div>
        )}
        {workers.data && workers.data.workers.length > 0 && (
          <div className="card overflow-hidden p-0">
            <table className="w-full text-sm">
              <thead className="bg-white text-ink-500 dark:bg-ink-900 border-b border-ink-200 dark:border-ink-800">
                <tr>
                  <th className="px-4 py-2 text-left font-medium">Worker</th>
                  <th className="px-4 py-2 text-right font-medium">
                    In flight
                  </th>
                  <th className="px-4 py-2 text-left font-medium">
                    Last lock
                  </th>
                </tr>
              </thead>
              <tbody>
                {workers.data.workers.map((w) => (
                  <tr
                    key={w.worker_id}
                    className="border-t border-ink-200 dark:border-ink-800"
                  >
                    <td className="px-4 py-2 font-mono text-xs">
                      {w.worker_id}
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums">
                      {w.running_jobs}
                    </td>
                    <td className="px-4 py-2 text-xs text-ink-500">
                      {ageLabel(w.last_lock_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
