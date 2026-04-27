import { useQuery } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import { experimentsApi, projectsApi, workspacesApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';

/** /workspaces/:wsId/projects/:projectId — project detail: experiments list + New Experiment. */
export function ProjectDetail() {
  const { wsId = '', projectId = '' } = useParams<{ wsId: string; projectId: string }>();

  const ws = useQuery({
    queryKey: ['workspaces', wsId],
    queryFn: () => workspacesApi.get(wsId),
    enabled: !!wsId,
  });
  const project = useQuery({
    queryKey: ['projects', wsId, projectId],
    queryFn: () => projectsApi.get(wsId, projectId),
    enabled: !!wsId && !!projectId,
  });
  const experiments = useQuery({
    queryKey: ['experiments', projectId],
    queryFn: () => experimentsApi.list(projectId),
    enabled: !!projectId,
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
          <span>{project.data?.name ?? '…'}</span>
        </nav>
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold">{project.data?.name ?? 'Loading…'}</h1>
            {project.data?.description && (
              <p className="text-sm text-ink-500 mt-1">{project.data.description}</p>
            )}
            {project.data && project.data.tags.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1">
                {project.data.tags.map((t) => (
                  <span key={t} className="kbd">
                    {t}
                  </span>
                ))}
              </div>
            )}
          </div>
          <Link
            to={`/workspaces/${wsId}/projects/${projectId}/experiments/new`}
            className="btn-primary"
          >
            New experiment
          </Link>
        </div>
      </header>

      <section>
        <header className="mb-4 flex items-baseline justify-between">
          <h2 className="font-medium">Experiments</h2>
          <span className="hint">{experiments.data?.length ?? 0} total</span>
        </header>

        {experiments.isLoading && <p className="hint">Loading…</p>}
        {experiments.error && <p className="error">{errorMessage(experiments.error)}</p>}

        {experiments.data && experiments.data.length === 0 && (
          <div className="card text-sm text-ink-500">
            No experiments yet. Create your first one →
          </div>
        )}

        <ul className="grid gap-3">
          {experiments.data?.map((e) => (
            <li key={e.id}>
              <Link
                to={`/workspaces/${wsId}/projects/${projectId}/experiments/${e.id}`}
                className="card block hover:border-accent-500 transition-colors"
              >
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <h3 className="font-medium text-ink-900">{e.name}</h3>
                    <p className="mt-1 text-sm text-ink-500">
                      <span className="kbd">{e.task}</span>
                      {e.target && (
                        <>
                          {' '}
                          target: <span className="font-mono">{e.target}</span>
                        </>
                      )}
                    </p>
                  </div>
                  <p className="text-xs text-ink-400 shrink-0">
                    {new Date(e.created_at).toLocaleDateString()}
                  </p>
                </div>
              </Link>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
