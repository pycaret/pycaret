/**
 * /workspaces/:wsId/deployments — workspace-level deployments list.
 *
 * Shows every active / paused / archived deployment in the workspace, with
 * at-a-glance p50 / p95 latency, inference count, and error count.
 */

import { useQuery } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import { deploymentsApi, workspacesApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';

const STATUS_PILL: Record<string, 'success' | 'warn' | 'neutral'> = {
  active: 'success',
  paused: 'warn',
  archived: 'neutral',
};

export function Deployments() {
  const { wsId = '' } = useParams<{ wsId: string }>();

  const ws = useQuery({
    queryKey: ['workspaces', wsId],
    queryFn: () => workspacesApi.get(wsId),
    enabled: !!wsId,
  });
  const deployments = useQuery({
    queryKey: ['deployments', 'by-workspace', wsId],
    queryFn: () => deploymentsApi.list(wsId),
    enabled: !!wsId,
    // Latency metrics update on every /predict; a 5 s poll keeps them alive-ish.
    refetchInterval: 5000,
  });

  return (
    <div className="space-y-8">
      <header>
        <nav className="text-xs text-ink-500 mb-2">
          <Link to="/" className="hover:text-ink-900">
            Workspaces
          </Link>
          <span className="mx-1">/</span>
          <Link to={`/workspaces/${wsId}`} className="hover:text-ink-900">
            {ws.data?.name ?? '…'}
          </Link>
          <span className="mx-1">/</span>
          <span>Deployments</span>
        </nav>
        <h1 className="text-xl font-semibold">Deployments</h1>
        <p className="mt-1 text-sm text-ink-500">
          Slug-addressable prediction endpoints backed by promoted pipelines.
        </p>
      </header>

      {deployments.isLoading && <p className="hint">Loading…</p>}
      {deployments.error && <p className="error">{errorMessage(deployments.error)}</p>}

      {deployments.data && deployments.data.length === 0 && (
        <div className="card text-sm text-ink-500">
          No deployments yet. Go to{' '}
          <Link to={`/workspaces/${wsId}/pipelines`} className="text-accent-600 hover:underline">
            Pipelines
          </Link>{' '}
          and deploy one.
        </div>
      )}

      {deployments.data && deployments.data.length > 0 && (
        <div className="rounded-xl bg-white dark:bg-ink-900 border border-ink-200 dark:border-ink-800 shadow-soft-1 overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-ink-50 dark:bg-ink-950 text-ink-600 dark:text-ink-400 border-b border-ink-200 dark:border-ink-800">
              <tr>
                <th className="px-4 py-2.5 text-left text-xs font-semibold uppercase tracking-wider">Slug</th>
                <th className="px-4 py-2.5 text-left text-xs font-semibold uppercase tracking-wider">Status</th>
                <th className="px-4 py-2.5 text-left text-xs font-semibold uppercase tracking-wider">Auth</th>
                <th className="px-4 py-2.5 text-right text-xs font-semibold uppercase tracking-wider">Predictions</th>
                <th className="px-4 py-2.5 text-right text-xs font-semibold uppercase tracking-wider">Errors</th>
                <th className="px-4 py-2.5 text-right text-xs font-semibold uppercase tracking-wider">p50</th>
                <th className="px-4 py-2.5 text-right text-xs font-semibold uppercase tracking-wider">p95</th>
                <th className="px-4 py-2.5 text-left text-xs font-semibold uppercase tracking-wider">Last hit</th>
              </tr>
            </thead>
            <tbody className="text-ink-800 dark:text-ink-100">
              {deployments.data.map((d) => (
                <tr
                  key={d.id}
                  className="border-t border-ink-100 dark:border-ink-800 hover:bg-ink-50 dark:hover:bg-ink-800/40 transition-colors"
                >
                  <td className="px-4 py-2.5">
                    <Link
                      to={`/deployments/${d.id}`}
                      className="text-accent-600 dark:text-accent-400 hover:text-accent-700 hover:underline font-mono text-xs"
                    >
                      {d.endpoint_slug}
                    </Link>
                  </td>
                  <td className="px-4 py-2.5">
                    <span className={`pill-${STATUS_PILL[d.status] ?? 'neutral'}`}>
                      {d.status}
                    </span>
                  </td>
                  <td className="px-4 py-2.5 text-xs">{d.auth_mode}</td>
                  <td className="px-4 py-2.5 text-right font-mono text-xs tabular-nums">
                    {d.inference_count}
                  </td>
                  <td className={`px-4 py-2.5 text-right font-mono text-xs tabular-nums ${d.error_count > 0 ? 'text-danger-600 dark:text-danger-500 font-medium' : ''}`}>
                    {d.error_count}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-xs tabular-nums">
                    {d.p50_latency_ms?.toFixed(1) ?? '—'}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-xs tabular-nums">
                    {d.p95_latency_ms?.toFixed(1) ?? '—'}
                  </td>
                  <td className="px-4 py-2.5 text-xs text-ink-500">
                    {d.last_inference_at
                      ? new Date(d.last_inference_at).toLocaleString()
                      : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
