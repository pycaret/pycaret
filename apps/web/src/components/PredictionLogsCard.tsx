/**
 * Recent prediction-logs panel for a deployment.
 *
 * Backend: GET /api/v1/deployments/{id}/prediction-logs
 *   query params: limit, offset, status_filter ('ok' | 'error')
 *
 * Pagination is single-page (limit=25) to keep the panel compact;
 * power users can hit the API directly for deeper history.
 */

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { deploymentsApi } from '@/api/endpoints';

export interface PredictionLogsCardProps {
  deploymentId: string;
}

const PAGE_SIZE = 25;

export function PredictionLogsCard({ deploymentId }: PredictionLogsCardProps) {
  const [statusFilter, setStatusFilter] = useState<'all' | 'ok' | 'error'>('all');

  const { data, isPending, error, refetch, isFetching } = useQuery({
    queryKey: ['deployments', deploymentId, 'logs', statusFilter],
    queryFn: () =>
      deploymentsApi.predictionLogs(deploymentId, {
        limit: PAGE_SIZE,
        offset: 0,
        status_filter: statusFilter === 'all' ? undefined : statusFilter,
      }),
    refetchInterval: 5000,
  });

  return (
    <section>
      <div className="flex items-center justify-between mb-3">
        <h2 className="h-section">Recent predictions</h2>
        <div className="flex items-center gap-2">
          <select
            className="input py-1 text-xs w-auto"
            value={statusFilter}
            onChange={(e) =>
              setStatusFilter(e.target.value as 'all' | 'ok' | 'error')
            }
          >
            <option value="all">All</option>
            <option value="ok">Successful</option>
            <option value="error">Errors</option>
          </select>
          <button
            className="btn-secondary text-xs py-1.5 px-2.5"
            onClick={() => refetch()}
            disabled={isFetching}
          >
            {isFetching ? '…' : 'Refresh'}
          </button>
        </div>
      </div>

      {isPending ? (
        <div className="card text-sm text-ink-500">Loading…</div>
      ) : error ? (
        <div className="card text-sm text-danger-600">
          Could not load prediction logs.
        </div>
      ) : !data || data.items.length === 0 ? (
        <p className="text-sm text-ink-500">
          No predictions yet for this deployment. Use the tester above to send
          one.
        </p>
      ) : (
        <div className="card overflow-hidden p-0">
          <table className="w-full text-sm">
            <thead className="bg-white text-ink-500 dark:bg-ink-900">
              <tr>
                <th className="px-4 py-2 text-left font-medium">When</th>
                <th className="px-4 py-2 text-left font-medium">Status</th>
                <th className="px-4 py-2 text-right font-medium">Rows</th>
                <th className="px-4 py-2 text-right font-medium">Latency</th>
                <th className="px-4 py-2 text-left font-medium">Request ID</th>
                <th className="px-4 py-2 text-left font-medium">Error</th>
              </tr>
            </thead>
            <tbody>
              {data.items.map((log) => (
                <tr
                  key={log.id}
                  className="border-t border-ink-200 dark:border-ink-800 hover:bg-ink-50 dark:hover:bg-ink-950/40"
                >
                  <td className="px-4 py-2 text-xs text-ink-700 dark:text-ink-300 whitespace-nowrap">
                    {new Date(log.created_at).toLocaleString()}
                  </td>
                  <td className="px-4 py-2">
                    {log.status === 'ok' ? (
                      <span className="pill-success">ok</span>
                    ) : (
                      <span className="pill-danger">error</span>
                    )}
                  </td>
                  <td className="px-4 py-2 text-right tabular-nums text-ink-700 dark:text-ink-300">
                    {log.n_rows}
                  </td>
                  <td className="px-4 py-2 text-right tabular-nums text-ink-700 dark:text-ink-300">
                    {log.latency_ms != null
                      ? `${log.latency_ms.toFixed(1)}ms`
                      : '—'}
                  </td>
                  <td
                    className="px-4 py-2 font-mono text-xs text-ink-500 max-w-[12rem] truncate"
                    title={log.request_id}
                  >
                    {log.request_id.slice(0, 8)}…
                  </td>
                  <td
                    className="px-4 py-2 text-xs text-danger-600 max-w-[20rem] truncate"
                    title={log.error ?? ''}
                  >
                    {log.error ?? ''}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="text-xs text-ink-500 mt-2">
        Showing latest {PAGE_SIZE}. Auto-refreshes every 5 s.
      </p>
    </section>
  );
}
