/**
 * Model Library section: per-workspace, per-task model registry view.
 *
 * Lists every model the engine ships for a given task, with toggleable
 * ``enabled`` flags. Lazy-seeded on first read of a workspace + task pair.
 *
 * v1: enabling/disabling is informational + UI-driven; engine-side
 * enforcement (filtering ``compare_models``) lands in V2.
 */

import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { modelLibraryApi, type ModelLibraryRow } from '@/api/endpoints';

export interface ModelLibrarySectionProps {
  workspaceId: string;
}

const TASKS = ['classification', 'regression', 'clustering', 'anomaly', 'time_series'];

export function ModelLibrarySection({ workspaceId }: ModelLibrarySectionProps) {
  const [task, setTask] = useState<string>('classification');
  const qc = useQueryClient();

  const { data, isPending } = useQuery({
    queryKey: ['workspaces', workspaceId, 'model-library', task],
    queryFn: () => modelLibraryApi.list(workspaceId, task),
    enabled: !!workspaceId,
  });

  const patch = useMutation({
    mutationFn: (vars: { id: string; enabled: boolean }) =>
      modelLibraryApi.patch(workspaceId, vars.id, { enabled: vars.enabled }),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ['workspaces', workspaceId, 'model-library'] }),
  });

  const sync = useMutation({
    mutationFn: () => modelLibraryApi.sync(workspaceId, task),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ['workspaces', workspaceId, 'model-library'] }),
  });

  return (
    <section>
      <div className="flex items-end justify-between gap-4 mb-4">
        <div>
          <h2 className="h-section">Model library</h2>
          <p className="text-xs text-ink-500 mt-1">
            Editable mirror of the engine's model registry. Engine-side
            filtering by these toggles ships in V2.
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <select
            className="input py-1.5 text-sm w-auto"
            value={task}
            onChange={(e) => setTask(e.target.value)}
          >
            {TASKS.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
          <button
            className="btn-secondary text-xs py-1.5 px-2.5"
            onClick={() => sync.mutate()}
            disabled={sync.isPending}
          >
            {sync.isPending ? 'Syncing…' : 'Sync from engine'}
          </button>
        </div>
      </div>

      {isPending ? (
        <div className="card text-sm text-ink-500">Loading…</div>
      ) : !data || data.items.length === 0 ? (
        <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 p-8 text-center text-sm text-ink-500">
          No models for {task}.
        </div>
      ) : (
        <div className="card overflow-hidden p-0">
          <table className="w-full text-sm">
            <thead className="bg-white text-ink-500 dark:bg-ink-900">
              <tr>
                <th className="px-4 py-2 text-left font-medium">Model ID</th>
                <th className="px-4 py-2 text-left font-medium">Name</th>
                <th className="px-4 py-2 text-center font-medium">Enabled</th>
                <th className="px-4 py-2 text-left font-medium">Custom params</th>
              </tr>
            </thead>
            <tbody>
              {data.items.map((row: ModelLibraryRow) => (
                <tr
                  key={row.id}
                  className="border-t border-ink-200 dark:border-ink-800 hover:bg-ink-50 dark:hover:bg-ink-950/40"
                >
                  <td className="px-4 py-2 font-mono text-ink-900 dark:text-ink-50">
                    {row.model_id}
                  </td>
                  <td className="px-4 py-2 text-ink-700 dark:text-ink-300">
                    {row.name}
                  </td>
                  <td className="px-4 py-2 text-center">
                    <input
                      type="checkbox"
                      checked={row.enabled}
                      disabled={patch.isPending}
                      onChange={(e) =>
                        patch.mutate({ id: row.id, enabled: e.target.checked })
                      }
                    />
                  </td>
                  <td className="px-4 py-2 text-xs font-mono text-ink-500">
                    {row.custom_params ? (
                      JSON.stringify(row.custom_params)
                    ) : (
                      <span className="text-ink-400">(defaults)</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
