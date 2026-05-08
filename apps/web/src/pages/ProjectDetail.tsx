/**
 * /workspaces/:wsId/projects/:projectId — project detail.
 *
 * Layout:
 *   - Breadcrumb + project header with primary "New experiment" CTA.
 *   - Data sources section (CSVs available to experiments here).
 *   - Experiments list.
 *
 * Data sources are workspace-scoped at the data layer, but conceptually
 * they "live" inside a project — uploads happen from this page so users
 * don't have to context-switch to a workspace settings screen first.
 */

import { useQuery } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import { experimentsApi, projectsApi, workspacesApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import { DataSourcesSection } from '@/components/DataSourcesSection';

export function ProjectDetail() {
  const { wsId = '', projectId = '' } = useParams<{
    wsId: string;
    projectId: string;
  }>();

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
      {/* ─── Header ──────────────────────────────────────────── */}
      <header className="space-y-2">
        <nav className="text-xs text-ink-500">
          <Link to="/" className="hover:text-ink-900 dark:hover:text-ink-50">
            Workspaces
          </Link>
          <span className="mx-1.5 text-ink-300">/</span>
          <Link
            to={`/workspaces/${wsId}`}
            className="hover:text-ink-900 dark:hover:text-ink-50"
          >
            {ws.data?.name ?? '…'}
          </Link>
          <span className="mx-1.5 text-ink-300">/</span>
          <span className="text-ink-700 dark:text-ink-300">
            {project.data?.name ?? '…'}
          </span>
        </nav>
        <div className="flex items-end justify-between gap-4">
          <div>
            <h1 className="h-page">{project.data?.name ?? 'Loading…'}</h1>
            {project.data?.description && (
              <p className="mt-2 text-sm text-ink-500">{project.data.description}</p>
            )}
            {project.data && project.data.tags.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-1">
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
            className="btn-primary shrink-0"
          >
            <PlusIcon />
            New experiment
          </Link>
        </div>
      </header>

      {/* ─── Data sources ────────────────────────────────────── */}
      <DataSourcesSection workspaceId={wsId} />

      {/* ─── Experiments ─────────────────────────────────────── */}
      <section>
        <header className="mb-4 flex items-baseline justify-between">
          <h2 className="h-section">
            Experiments{' '}
            <span className="text-ink-400 font-normal">
              ({experiments.data?.length ?? 0})
            </span>
          </h2>
        </header>

        {experiments.isLoading && (
          <div className="card text-sm text-ink-500">Loading…</div>
        )}
        {experiments.error && (
          <div className="card text-sm text-danger-600">
            {errorMessage(experiments.error)}
          </div>
        )}

        {experiments.data && experiments.data.length === 0 && (
          <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 p-10 text-center">
            <div className="mx-auto h-10 w-10 rounded-lg bg-ink-100 dark:bg-ink-800 text-ink-500 flex items-center justify-center mb-3">
              <BeakerIcon />
            </div>
            <h3 className="text-base font-semibold text-ink-900 dark:text-ink-50">
              No experiments yet
            </h3>
            <p className="mt-1.5 text-sm text-ink-500 max-w-sm mx-auto">
              Pick a task, point it at a data source, and PyCaret runs the rest.
            </p>
            <Link
              to={`/workspaces/${wsId}/projects/${projectId}/experiments/new`}
              className="btn-primary mt-5 inline-flex"
            >
              <PlusIcon />
              Create your first experiment
            </Link>
          </div>
        )}

        {experiments.data && experiments.data.length > 0 && (
          <ul className="rounded-xl bg-white dark:bg-ink-900 border border-ink-200 dark:border-ink-800 shadow-soft-1 divide-y divide-ink-100 dark:divide-ink-800">
            {experiments.data.map((e) => (
              <li key={e.id}>
                <Link
                  to={`/workspaces/${wsId}/projects/${projectId}/experiments/${e.id}`}
                  className="px-4 py-3 flex items-center justify-between gap-3 hover:bg-ink-50 dark:hover:bg-ink-950/40 transition-colors"
                >
                  <div className="min-w-0 flex items-center gap-3">
                    <span className="h-8 w-8 rounded-md bg-ink-100 dark:bg-ink-800 text-ink-500 flex items-center justify-center shrink-0">
                      <BeakerIcon />
                    </span>
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-ink-900 dark:text-ink-50 truncate">
                        {e.name}
                      </p>
                      <p className="mt-0.5 text-xs text-ink-500 truncate">
                        <span className="pill-neutral mr-1.5">{e.task}</span>
                        {e.target && (
                          <>
                            target:{' '}
                            <span className="font-mono">{e.target}</span>
                          </>
                        )}
                      </p>
                    </div>
                  </div>
                  <span className="text-xs text-ink-400 shrink-0 tabular-nums">
                    {new Date(e.created_at).toLocaleDateString()}
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

const stroke = {
  width: '14',
  height: '14',
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: '1.75',
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  'aria-hidden': true,
};
const PlusIcon = () => (
  <svg {...stroke}>
    <path d="M12 5v14" />
    <path d="M5 12h14" />
  </svg>
);
const BeakerIcon = () => (
  <svg {...stroke}>
    <path d="M9 3h6" />
    <path d="M10 3v8L4 21h16L14 11V3" />
  </svg>
);
