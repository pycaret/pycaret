/**
 * /workspaces/:wsId/models — Phase 7 model registry list.
 *
 * Sister page to ``/workspaces/:wsId/models/:id`` for version
 * history. Lists every RegisteredModel in the workspace with a
 * "current version" pill (production / staging / archived).
 */

import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import { BackButton } from '@/components/BackButton';
import { Dialog } from '@/components/Dialog';
import { registryApi } from '@/api/endpoints';

export function RegisteredModels() {
  const { wsId = '' } = useParams<{ wsId: string }>();
  const qc = useQueryClient();
  const [newOpen, setNewOpen] = useState(false);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');

  const list = useQuery({
    queryKey: ['registered-models', wsId],
    queryFn: () => registryApi.list(wsId),
    enabled: !!wsId,
  });

  const create = useMutation({
    mutationFn: () =>
      registryApi.create(wsId, {
        name: name.trim(),
        description: description.trim() || undefined,
      }),
    onSuccess: () => {
      setNewOpen(false);
      setName('');
      setDescription('');
      qc.invalidateQueries({ queryKey: ['registered-models', wsId] });
    },
  });

  return (
    <div className="space-y-6">
      <header className="flex items-baseline justify-between">
        <div>
          <BackButton />
          <h1 className="h-page mt-2">Registered models</h1>
          <p className="muted small">
            Named, versioned references a Deployment points at. Phase 7.
          </p>
        </div>
        <button
          type="button"
          className="btn-primary"
          onClick={() => setNewOpen(true)}
        >
          New model
        </button>
      </header>

      {list.isLoading && (
        <div className="card text-sm text-ink-500">Loading…</div>
      )}
      {list.data && list.data.length === 0 && (
        <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 p-10 text-center">
          <h3 className="text-base font-semibold">No registered models yet</h3>
          <p className="mt-1 text-sm text-ink-500">
            Promote a successful Trial to create the first version.
          </p>
        </div>
      )}
      {list.data && list.data.length > 0 && (
        <div className="card overflow-hidden p-0">
          <table className="w-full text-sm">
            <thead className="bg-white text-ink-500 dark:bg-ink-900 border-b border-ink-200 dark:border-ink-800">
              <tr>
                <th className="px-4 py-2 text-left font-medium">Name</th>
                <th className="px-4 py-2 text-left font-medium">
                  Description
                </th>
                <th className="px-4 py-2 text-left font-medium">
                  Current version
                </th>
                <th className="px-4 py-2 text-left font-medium">Created</th>
                <th className="px-4 py-2 text-right" />
              </tr>
            </thead>
            <tbody>
              {list.data.map((m) => (
                <tr
                  key={m.id}
                  className="border-t border-ink-200 dark:border-ink-800 hover:bg-ink-50 dark:hover:bg-ink-950/40"
                >
                  <td className="px-4 py-2 font-medium">
                    <Link
                      to={`/workspaces/${wsId}/models/${m.id}`}
                      className="hover:underline"
                    >
                      {m.name}
                    </Link>
                  </td>
                  <td className="px-4 py-2 text-ink-700 dark:text-ink-300">
                    {m.description ?? <span className="text-ink-400">—</span>}
                  </td>
                  <td className="px-4 py-2">
                    {m.current_version_id ? (
                      <code className="text-xs">
                        {m.current_version_id.slice(0, 8)}…
                      </code>
                    ) : (
                      <span className="text-ink-400">—</span>
                    )}
                  </td>
                  <td className="px-4 py-2 text-xs text-ink-500">
                    {m.created_at
                      ? new Date(m.created_at).toLocaleDateString()
                      : '—'}
                  </td>
                  <td className="px-4 py-2 text-right">
                    <Link
                      to={`/workspaces/${wsId}/models/${m.id}`}
                      className="text-accent-700 hover:text-accent-800 dark:text-accent-300"
                    >
                      Open →
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <Dialog
        open={newOpen}
        onClose={() => setNewOpen(false)}
        title="New registered model"
        size="sm"
      >
        <div className="space-y-3">
          <label className="field">
            <span className="muted small">Name</span>
            <input
              className="input w-full"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="iris-classifier"
              autoFocus
            />
          </label>
          <label className="field">
            <span className="muted small">Description (optional)</span>
            <textarea
              className="input w-full"
              rows={3}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
            />
          </label>
          <div className="flex justify-end gap-2">
            <button
              type="button"
              className="btn-secondary"
              onClick={() => setNewOpen(false)}
            >
              Cancel
            </button>
            <button
              type="button"
              className="btn-primary"
              disabled={!name.trim() || create.isPending}
              onClick={() => create.mutate()}
            >
              {create.isPending ? 'Creating…' : 'Create'}
            </button>
          </div>
        </div>
      </Dialog>
    </div>
  );
}
