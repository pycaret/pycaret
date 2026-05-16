/**
 * /workspaces/:wsId/secrets — Phase 4 secret store.
 *
 * Encrypted-at-rest key/value pairs scoped to a workspace. Used by
 * Connections (DB passwords), Git repositories (PATs), and any future
 * driver that needs a credential.
 *
 * The plaintext value is never shown after creation — we display a
 * ``****abcd`` last4 hint instead. To rotate, delete + re-create.
 */

import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useParams } from 'react-router-dom';
import { BackButton } from '@/components/BackButton';
import { Dialog } from '@/components/Dialog';
import { secretsApi } from '@/api/endpoints';

const KIND_OPTIONS = [
  { value: 'opaque', label: 'Opaque (generic)' },
  { value: 'db_password', label: 'Database password' },
  { value: 'api_key', label: 'API key' },
  { value: 'git_pat', label: 'Git PAT' },
  { value: 's3_secret', label: 'S3 secret key' },
];

export function Secrets() {
  const { wsId = '' } = useParams<{ wsId: string }>();
  const qc = useQueryClient();
  const [createOpen, setCreateOpen] = useState(false);
  const [name, setName] = useState('');
  const [value, setValue] = useState('');
  const [kind, setKind] = useState('opaque');

  const list = useQuery({
    queryKey: ['secrets', wsId],
    queryFn: () => secretsApi.list(wsId),
    enabled: !!wsId,
  });

  const create = useMutation({
    mutationFn: () =>
      secretsApi.create(wsId, { name: name.trim(), value, kind }),
    onSuccess: () => {
      setCreateOpen(false);
      setName('');
      setValue('');
      setKind('opaque');
      qc.invalidateQueries({ queryKey: ['secrets', wsId] });
    },
  });

  const remove = useMutation({
    mutationFn: (secretId: string) => secretsApi.delete(wsId, secretId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['secrets', wsId] }),
  });

  return (
    <div className="space-y-6">
      <header className="flex items-baseline justify-between">
        <div>
          <BackButton />
          <h1 className="h-page mt-2">Secrets</h1>
          <p className="muted small">
            Encrypted-at-rest credentials scoped to this workspace. Used
            by Connections, Git repositories, and other drivers.
          </p>
        </div>
        <button
          type="button"
          className="btn-primary"
          onClick={() => setCreateOpen(true)}
        >
          New secret
        </button>
      </header>

      {list.isLoading && (
        <div className="card text-sm text-ink-500">Loading…</div>
      )}
      {list.data && list.data.length === 0 && (
        <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 p-10 text-center">
          <h3 className="text-base font-semibold">No secrets yet</h3>
          <p className="mt-1 text-sm text-ink-500">
            Store API keys, DB passwords, and Git PATs here so drivers
            can reach them without inlining credentials in connection
            configs.
          </p>
        </div>
      )}
      {list.data && list.data.length > 0 && (
        <div className="card overflow-hidden p-0">
          <table className="w-full text-sm">
            <thead className="bg-white text-ink-500 dark:bg-ink-900 border-b border-ink-200 dark:border-ink-800">
              <tr>
                <th className="px-4 py-2 text-left font-medium">Name</th>
                <th className="px-4 py-2 text-left font-medium">Kind</th>
                <th className="px-4 py-2 text-left font-medium">Value</th>
                <th className="px-4 py-2 text-left font-medium">Created</th>
                <th className="px-4 py-2 text-right" />
              </tr>
            </thead>
            <tbody>
              {list.data.map((s) => (
                <tr
                  key={s.id}
                  className="border-t border-ink-200 dark:border-ink-800 hover:bg-ink-50 dark:hover:bg-ink-950/40"
                >
                  <td className="px-4 py-2 font-medium">{s.name}</td>
                  <td className="px-4 py-2">
                    <span className="pill-neutral">{s.kind}</span>
                  </td>
                  <td className="px-4 py-2 font-mono text-xs text-ink-500">
                    {s.last4 ? `••••${s.last4}` : '••••'}
                  </td>
                  <td className="px-4 py-2 text-xs text-ink-500">
                    {s.created_at
                      ? new Date(s.created_at).toLocaleDateString()
                      : '—'}
                  </td>
                  <td className="px-4 py-2 text-right">
                    <button
                      type="button"
                      className="text-xs text-danger-600 hover:underline"
                      onClick={() => {
                        if (
                          confirm(
                            `Delete secret "${s.name}"? This cannot be undone.`,
                          )
                        ) {
                          remove.mutate(s.id);
                        }
                      }}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <Dialog
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        title="New secret"
        size="sm"
      >
        <div className="space-y-3">
          <label className="field">
            <span className="muted small">Name</span>
            <input
              className="input w-full"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="prod-postgres-password"
              autoFocus
            />
          </label>
          <label className="field">
            <span className="muted small">Kind</span>
            <select
              className="input w-full"
              value={kind}
              onChange={(e) => setKind(e.target.value)}
            >
              {KIND_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span className="muted small">Value (not echoed back after save)</span>
            <input
              className="input w-full font-mono text-xs"
              type="password"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              placeholder="••••••••"
            />
          </label>
          <p className="muted small">
            Stored encrypted with the server's Fernet key. Only{' '}
            <code>••••{value.slice(-4)}</code> shows in the list view.
          </p>
          <div className="flex justify-end gap-2">
            <button
              type="button"
              className="btn-secondary"
              onClick={() => setCreateOpen(false)}
            >
              Cancel
            </button>
            <button
              type="button"
              className="btn-primary"
              disabled={!name.trim() || !value || create.isPending}
              onClick={() => create.mutate()}
            >
              {create.isPending ? 'Saving…' : 'Save'}
            </button>
          </div>
          {create.error && (
            <p className="text-xs text-danger-600">
              {(create.error as Error).message}
            </p>
          )}
        </div>
      </Dialog>
    </div>
  );
}
