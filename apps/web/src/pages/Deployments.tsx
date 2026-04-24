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

const STATUS_TONE: Record<string, string> = {
  active: 'text-success-500',
  paused: 'text-warn-500',
  archived: 'text-ink-200/60',
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
        <nav className="text-xs text-ink-200/60 mb-2">
          <Link to="/" className="hover:text-ink-100">
            Workspaces
          </Link>
          <span className="mx-1">/</span>
          <Link to={`/workspaces/${wsId}`} className="hover:text-ink-100">
            {ws.data?.name ?? '…'}
          </Link>
          <span className="mx-1">/</span>
          <span>Deployments</span>
        </nav>
        <h1 className="text-xl font-semibold">Deployments</h1>
        <p className="mt-1 text-sm text-ink-200/70">
          Slug-addressable prediction endpoints backed by promoted pipelines.
        </p>
      </header>

      {deployments.isLoading && <p className="hint">Loading…</p>}
      {deployments.error && <p className="error">{errorMessage(deployments.error)}</p>}

      {deployments.data && deployments.data.length === 0 && (
        <div className="card text-sm text-ink-200/70">
          No deployments yet. Go to{' '}
          <Link to={`/workspaces/${wsId}/pipelines`} className="text-accent-400 hover:underline">
            Pipelines
          </Link>{' '}
          and deploy one.
        </div>
      )}

      {deployments.data && deployments.data.length > 0 && (
        <div className="card overflow-hidden p-0">
          <table className="w-full text-sm">
            <thead className="bg-ink-800 text-ink-200/70">
              <tr>
                <th className="px-4 py-2 text-left font-medium">Slug</th>
                <th className="px-4 py-2 text-left font-medium">Status</th>
                <th className="px-4 py-2 text-left font-medium">Auth</th>
                <th className="px-4 py-2 text-right font-medium">Predictions</th>
                <th className="px-4 py-2 text-right font-medium">Errors</th>
                <th className="px-4 py-2 text-right font-medium">p50</th>
                <th className="px-4 py-2 text-right font-medium">p95</th>
                <th className="px-4 py-2 text-left font-medium">Last hit</th>
              </tr>
            </thead>
            <tbody>
              {deployments.data.map((d) => (
                <tr key={d.id} className="border-t border-ink-800 hover:bg-ink-800/50">
                  <td className="px-4 py-2">
                    <Link
                      to={`/deployments/${d.id}`}
                      className="text-accent-400 hover:underline font-mono"
                    >
                      {d.endpoint_slug}
                    </Link>
                  </td>
                  <td className={`px-4 py-2 ${STATUS_TONE[d.status] ?? ''}`}>
                    {d.status}
                  </td>
                  <td className="px-4 py-2 font-mono text-xs">{d.auth_mode}</td>
                  <td className="px-4 py-2 text-right font-mono text-xs tabular-nums">
                    {d.inference_count}
                  </td>
                  <td className={`px-4 py-2 text-right font-mono text-xs tabular-nums ${d.error_count > 0 ? 'text-danger-500' : ''}`}>
                    {d.error_count}
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-xs tabular-nums">
                    {d.p50_latency_ms?.toFixed(1) ?? '—'}
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-xs tabular-nums">
                    {d.p95_latency_ms?.toFixed(1) ?? '—'}
                  </td>
                  <td className="px-4 py-2 text-xs text-ink-200/60">
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
