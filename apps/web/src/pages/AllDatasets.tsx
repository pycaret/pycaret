/**
 * /workspaces/:wsId/datasets — workspace-level list of every DataSource.
 *
 * DataSources are workspace-scoped (db/models.py:161 — "Registered
 * CSV/S3/Postgres source a Project can point at"). They're shared across
 * projects; this page is the canonical "browse all data" entry point.
 *
 * The per-source version history + EDA still live under
 * /workspaces/:wsId/datasets/:dataSourceId{,/profile}. This page is the
 * index that links into both.
 *
 * Implementation is a thin wrapper over the existing DataSourcesSection
 * component (already used inside ProjectDetail) so behaviour stays in
 * lockstep with the project-page version.
 */

import { Link, useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { workspacesApi } from '@/api/endpoints';
import { DataSourcesSection } from '@/components/DataSourcesSection';

export function AllDatasets() {
  const { wsId = '' } = useParams<{ wsId: string }>();

  const ws = useQuery({
    queryKey: ['workspaces', wsId],
    queryFn: () => workspacesApi.get(wsId),
    enabled: !!wsId,
  });

  return (
    <div className="space-y-6">
      <header>
        <nav className="text-xs text-ink-500 mb-2">
          <Link to="/" className="hover:text-ink-900 dark:hover:text-ink-50">
            Workspaces
          </Link>
          <span className="mx-1">/</span>
          <Link
            to={`/workspaces/${wsId}`}
            className="hover:text-ink-900 dark:hover:text-ink-50"
          >
            {ws.data?.name ?? '…'}
          </Link>
          <span className="mx-1">/</span>
          <span>Datasets</span>
        </nav>
        <h1 className="h-page">Datasets</h1>
        <p className="muted small mt-1">
          CSV uploads and registered remote sources for this workspace.
          Experiments in any project can reference these.
        </p>
      </header>

      <DataSourcesSection workspaceId={wsId} />
    </div>
  );
}
