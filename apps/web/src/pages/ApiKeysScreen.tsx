/**
 * /account/api-keys — personal API-key management.
 *
 * Not workspace-scoped: keys are issued against the current user. A workspace
 * scope can be attached at creation time for fine-grained access control;
 * middleware that enforces it is session-20 work.
 *
 * UX: list of existing keys with prefix + expiry + revoked-at. "New key"
 * button opens an inline form; on success a confirmation panel shows the
 * plaintext ONCE with a copy button + a clear warning that it won't be
 * shown again.
 */

import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { apiKeysApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import type { ApiKeyCreateResponse } from '@/api/types';

function isActive(k: {
  revoked_at: string | null;
  expires_at: string | null;
}): boolean {
  if (k.revoked_at) return false;
  if (k.expires_at && new Date(k.expires_at) < new Date()) return false;
  return true;
}

export function ApiKeysScreen() {
  const qc = useQueryClient();
  const keys = useQuery({
    queryKey: ['api-keys'],
    queryFn: () => apiKeysApi.list(),
  });

  const [name, setName] = useState('');
  const [expiresInDays, setExpiresInDays] = useState<string>('');
  // Holds the one-time plaintext after a successful create.
  const [justCreated, setJustCreated] = useState<ApiKeyCreateResponse | null>(null);

  const create = useMutation({
    mutationFn: () =>
      apiKeysApi.create({
        name: name.trim(),
        expires_in_days: expiresInDays ? Number.parseInt(expiresInDays, 10) : null,
      }),
    onSuccess: (resp) => {
      setJustCreated(resp);
      setName('');
      setExpiresInDays('');
      qc.invalidateQueries({ queryKey: ['api-keys'] });
    },
  });

  const revoke = useMutation({
    mutationFn: (id: string) => apiKeysApi.revoke(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['api-keys'] }),
  });

  return (
    <div className="max-w-3xl space-y-8">
      <header>
        <nav className="text-xs text-ink-200/60 mb-2">
          <Link to="/" className="hover:text-ink-100">
            Workspaces
          </Link>
          <span className="mx-1">/</span>
          <span>API keys</span>
        </nav>
        <h1 className="text-xl font-semibold">API keys</h1>
        <p className="mt-1 text-sm text-ink-200/70">
          Personal programmatic-access tokens for CI pipelines + scripts. Revoking
          a key soft-deletes it; the audit trail stays. Middleware that accepts
          keys via <code className="font-mono">X-PyCaret-Key</code> lands in
          session 20.
        </p>
      </header>

      {/* ────────── one-time plaintext panel */}
      {justCreated && (
        <div className="card border-accent-500/50 space-y-3">
          <h2 className="text-sm font-medium text-accent-400">
            ✓ API key created — copy it now
          </h2>
          <p className="hint">
            This plaintext will <strong>never</strong> be shown again. Store it in
            your secret manager now.
          </p>
          <pre className="bg-ink-950 border border-ink-800 rounded p-3 font-mono text-xs text-ink-100 overflow-x-auto">
            {justCreated.token}
          </pre>
          <div className="flex items-center justify-end gap-2">
            <button
              className="btn-secondary"
              onClick={() => {
                navigator.clipboard?.writeText(justCreated.token);
              }}
            >
              Copy
            </button>
            <button className="btn-primary" onClick={() => setJustCreated(null)}>
              I've saved it
            </button>
          </div>
        </div>
      )}

      {/* ────────── create form */}
      <div className="card space-y-4">
        <h2 className="text-sm font-medium text-ink-100">New API key</h2>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (name.trim()) create.mutate();
          }}
          className="space-y-4"
        >
          <div>
            <label className="field" htmlFor="key-name">
              Name <span className="text-danger-500">*</span>
            </label>
            <input
              id="key-name"
              className="input"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="ci-bot, laptop-local, ..."
              required
            />
          </div>
          <div>
            <label className="field" htmlFor="key-expiry">
              Expires in (days)
            </label>
            <input
              id="key-expiry"
              type="number"
              min={1}
              max={3650}
              className="input"
              value={expiresInDays}
              onChange={(e) => setExpiresInDays(e.target.value)}
              placeholder="leave blank for no expiry"
            />
          </div>
          {create.error && <p className="error">{errorMessage(create.error)}</p>}
          <button
            type="submit"
            className="btn-primary"
            disabled={create.isPending || !name.trim()}
          >
            {create.isPending ? 'Creating…' : 'Create key'}
          </button>
        </form>
      </div>

      {/* ────────── existing keys */}
      <section>
        <header className="mb-3 flex items-baseline justify-between">
          <h2 className="text-sm font-medium text-ink-100">Your keys</h2>
          <span className="hint">{keys.data?.length ?? 0} total</span>
        </header>

        {keys.isLoading && <p className="hint">Loading…</p>}
        {keys.error && <p className="error">{errorMessage(keys.error)}</p>}

        {keys.data && keys.data.length === 0 && (
          <div className="card text-sm text-ink-200/70">
            No API keys yet. Create one above.
          </div>
        )}

        {keys.data && keys.data.length > 0 && (
          <div className="card p-0 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-ink-800 text-ink-200/70">
                <tr>
                  <th className="px-4 py-2 text-left font-medium">Name</th>
                  <th className="px-4 py-2 text-left font-medium">Prefix</th>
                  <th className="px-4 py-2 text-left font-medium">Status</th>
                  <th className="px-4 py-2 text-left font-medium">Expires</th>
                  <th className="px-4 py-2 text-left font-medium">Created</th>
                  <th className="px-4 py-2 text-right font-medium">Action</th>
                </tr>
              </thead>
              <tbody>
                {keys.data.map((k) => (
                  <tr key={k.id} className="border-t border-ink-800 hover:bg-ink-800/50">
                    <td className="px-4 py-2 font-medium text-ink-100">{k.name}</td>
                    <td className="px-4 py-2 font-mono text-xs text-ink-200/70">
                      {k.prefix}…
                    </td>
                    <td
                      className={`px-4 py-2 ${
                        isActive(k)
                          ? 'text-success-500'
                          : 'text-ink-200/60'
                      }`}
                    >
                      {k.revoked_at ? 'revoked' : isActive(k) ? 'active' : 'expired'}
                    </td>
                    <td className="px-4 py-2 text-xs text-ink-200/60">
                      {k.expires_at
                        ? new Date(k.expires_at).toLocaleDateString()
                        : '—'}
                    </td>
                    <td className="px-4 py-2 text-xs text-ink-200/60">
                      {new Date(k.created_at).toLocaleDateString()}
                    </td>
                    <td className="px-4 py-2 text-right">
                      {!k.revoked_at && (
                        <button
                          className="btn-ghost text-xs"
                          onClick={() => {
                            if (
                              window.confirm(`Revoke key "${k.name}"? This cannot be undone.`)
                            ) {
                              revoke.mutate(k.id);
                            }
                          }}
                          disabled={revoke.isPending}
                        >
                          Revoke
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
