/**
 * /workspaces/:wsId/pipelines — workspace-scoped registry of fitted pipelines.
 *
 * A Pipeline row is the immutable, deployable artifact produced when a
 * succeeded Run is promoted. From here a user can inspect each pipeline and
 * promote it into one or more Deployments (distinct, slug-addressable
 * endpoints).
 */

import { useQuery } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import { pipelinesApi, workspacesApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';

export function Pipelines() {
  const { wsId = '' } = useParams<{ wsId: string }>();

  const ws = useQuery({
    queryKey: ['workspaces', wsId],
    queryFn: () => workspacesApi.get(wsId),
    enabled: !!wsId,
  });
  const pipelines = useQuery({
    queryKey: ['pipelines', wsId],
    queryFn: () => pipelinesApi.list(wsId),
    enabled: !!wsId,
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
          <span>Pipelines</span>
        </nav>
        <h1 className="text-xl font-semibold">Pipelines</h1>
        <p className="mt-1 text-sm text-ink-500">
          Workspace-scoped registry of fitted pipelines. Promote a succeeded run
          from its detail page to add a new entry here.
        </p>
      </header>

      {pipelines.isLoading && <p className="hint">Loading…</p>}
      {pipelines.error && <p className="error">{errorMessage(pipelines.error)}</p>}

      {pipelines.data && pipelines.data.length === 0 && (
        <div className="card text-sm text-ink-500">
          No pipelines yet. Complete a run and click <strong>Promote</strong> on its
          detail page.
        </div>
      )}

      {pipelines.data && pipelines.data.length > 0 && (
        <div className="card overflow-hidden p-0">
          <table className="w-full text-sm">
            <thead className="bg-white text-ink-500">
              <tr>
                <th className="px-4 py-2 text-left font-medium">Name</th>
                <th className="px-4 py-2 text-left font-medium">Model</th>
                <th className="px-4 py-2 text-left font-medium">SHA-256</th>
                <th className="px-4 py-2 text-left font-medium">Tags</th>
                <th className="px-4 py-2 text-left font-medium">Created</th>
              </tr>
            </thead>
            <tbody>
              {pipelines.data.map((p) => (
                <tr
                  key={p.id}
                  className="border-t border-ink-200 hover:bg-ink-50"
                >
                  <td className="px-4 py-2">
                    <Link
                      to={`/workspaces/${wsId}/pipelines/${p.id}`}
                      className="text-accent-400 hover:underline"
                    >
                      {p.name}
                    </Link>
                    {p.description && (
                      <p className="text-xs text-ink-500 mt-0.5">{p.description}</p>
                    )}
                  </td>
                  <td className="px-4 py-2 font-mono text-xs">
                    {p.model_id ?? '—'}
                  </td>
                  <td className="px-4 py-2 font-mono text-xs text-ink-500" title={p.sha256 ?? ''}>
                    {p.sha256 ? `${p.sha256.slice(0, 8)}…` : '—'}
                  </td>
                  <td className="px-4 py-2">
                    <div className="flex flex-wrap gap-1">
                      {p.tags.map((t) => (
                        <span key={t} className="kbd">
                          {t}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td className="px-4 py-2 text-xs text-ink-500">
                    {new Date(p.created_at).toLocaleDateString()}
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
