/**
 * /workspaces/:wsId/datasets/:dataSourceId — Phase 4 dataset version
 * history.
 *
 * Each DataSource accumulates Datasets (immutable snapshots) over
 * time. The "Refresh" button re-introspects the DataSource via its
 * driver and writes a new version row with the captured schema +
 * sample.
 */

import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useParams } from 'react-router-dom';
import { BackButton } from '@/components/BackButton';
import { dataSourcesApi, datasetsApi } from '@/api/endpoints';
import type { Dataset } from '@/api/types';

export function Datasets() {
  const { dataSourceId = '' } = useParams<{
    wsId: string;
    dataSourceId: string;
  }>();
  const qc = useQueryClient();
  const [selectedVersion, setSelectedVersion] = useState<string | null>(null);

  const dataSource = useQuery({
    queryKey: ['data-source', dataSourceId],
    queryFn: () => dataSourcesApi.get(dataSourceId),
    enabled: !!dataSourceId,
  });

  const versions = useQuery({
    queryKey: ['datasets', 'for-source', dataSourceId],
    queryFn: () => datasetsApi.list(dataSourceId),
    enabled: !!dataSourceId,
  });

  const refresh = useMutation({
    mutationFn: () => datasetsApi.refresh(dataSourceId),
    onSuccess: (data) => {
      setSelectedVersion(data.id);
      qc.invalidateQueries({ queryKey: ['datasets', 'for-source', dataSourceId] });
    },
  });

  const active =
    versions.data?.find((v) => v.id === selectedVersion) ??
    versions.data?.[0] ??
    null;

  return (
    <div className="space-y-6">
      <header className="flex items-baseline justify-between">
        <div>
          <BackButton />
          <h1 className="h-page mt-2">
            {dataSource.data?.name ?? 'Dataset versions'}
          </h1>
          <p className="muted small">
            DataSource <code>{dataSourceId.slice(0, 8)}…</code> ·{' '}
            <span className="pill-neutral">{dataSource.data?.kind}</span>
          </p>
        </div>
        <button
          type="button"
          className="btn-primary"
          onClick={() => refresh.mutate()}
          disabled={refresh.isPending}
        >
          {refresh.isPending ? 'Refreshing…' : 'Refresh now'}
        </button>
      </header>

      {refresh.error && (
        <div className="card text-sm text-danger-600">
          Refresh failed: {(refresh.error as Error).message}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <section className="md:col-span-1">
          <h2 className="h-section mb-3">
            Versions ({versions.data?.length ?? 0})
          </h2>
          {versions.isLoading && (
            <div className="card text-sm text-ink-500">Loading…</div>
          )}
          {versions.data && versions.data.length === 0 && (
            <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 p-6 text-center">
              <p className="text-sm text-ink-500">
                No snapshots yet. Hit "Refresh now".
              </p>
            </div>
          )}
          {versions.data && versions.data.length > 0 && (
            <ul className="space-y-2">
              {versions.data.map((v) => (
                <li key={v.id}>
                  <button
                    type="button"
                    className={`w-full text-left card p-3 ${
                      active?.id === v.id
                        ? 'ring-2 ring-accent-500'
                        : 'hover:bg-ink-50 dark:hover:bg-ink-950/40'
                    }`}
                    onClick={() => setSelectedVersion(v.id)}
                  >
                    <p className="text-sm font-medium">v{v.version}</p>
                    <p className="text-xs text-ink-500 mt-1">
                      {v.row_count != null ? `${v.row_count} rows` : '— rows'} ·{' '}
                      {v.byte_count != null
                        ? `${Math.round(v.byte_count / 1024)} kB`
                        : '— kB'}
                    </p>
                    <p className="text-xs text-ink-500 mt-1">
                      {v.created_at
                        ? new Date(v.created_at).toLocaleString()
                        : '—'}
                    </p>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="md:col-span-2">
          {active ? (
            <DatasetDetail dataset={active} />
          ) : (
            <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 p-8 text-center">
              <p className="text-sm text-ink-500">
                Pick a version to inspect.
              </p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

function DatasetDetail({ dataset }: { dataset: Dataset }) {
  const columns = dataset.schema_json?.columns ?? [];
  const sample = dataset.schema_json?.sample_rows ?? [];

  return (
    <div className="space-y-4">
      <section className="card">
        <h3 className="text-sm font-semibold">Schema</h3>
        {columns.length === 0 ? (
          <p className="text-xs text-ink-500 mt-2">No columns captured.</p>
        ) : (
          <div className="overflow-x-auto mt-2">
            <table className="w-full text-xs">
              <thead className="text-ink-500">
                <tr>
                  <th className="text-left px-2 py-1 font-medium">Name</th>
                  <th className="text-left px-2 py-1 font-medium">Dtype</th>
                  <th className="text-left px-2 py-1 font-medium">Nullable</th>
                  <th className="text-left px-2 py-1 font-medium">Sample</th>
                </tr>
              </thead>
              <tbody>
                {columns.map((c, i) => (
                  <tr
                    key={i}
                    className="border-t border-ink-200 dark:border-ink-800"
                  >
                    <td className="px-2 py-1 font-medium">{c.name}</td>
                    <td className="px-2 py-1">
                      <code className="text-xs">{c.dtype}</code>
                    </td>
                    <td className="px-2 py-1">
                      {c.nullable ? 'yes' : 'no'}
                    </td>
                    <td className="px-2 py-1 text-ink-500 truncate max-w-[200px]">
                      {String(c.sample ?? '—')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {sample.length > 0 && (
        <section className="card">
          <h3 className="text-sm font-semibold">
            Sample rows ({sample.length})
          </h3>
          <div className="overflow-x-auto mt-2">
            <table className="w-full text-xs">
              <thead className="text-ink-500">
                <tr>
                  {Object.keys(sample[0]).map((k) => (
                    <th key={k} className="text-left px-2 py-1 font-medium">
                      {k}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sample.map((row, i) => (
                  <tr
                    key={i}
                    className="border-t border-ink-200 dark:border-ink-800"
                  >
                    {Object.keys(sample[0]).map((k) => (
                      <td
                        key={k}
                        className="px-2 py-1 truncate max-w-[160px]"
                      >
                        {String((row as Record<string, unknown>)[k] ?? '—')}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
