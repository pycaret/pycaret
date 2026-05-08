/**
 * /workspaces/:wsId/webhooks — outgoing event-hook subscriptions.
 *
 * v1: list / create / toggle / delete / test. Each subscription gets a
 * shared secret used for HMAC-SHA256 signing of delivery bodies. The
 * server stores the secret encrypted at rest.
 */

import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useParams } from 'react-router-dom';
import { webhooksApi, type Webhook } from '@/api/endpoints';
import { errorMessage } from '@/api/client';

const EVENT_TYPES = [
  'run.succeeded',
  'run.failed',
  'run.cancelled',
  'deployment.created',
  'deployment.deleted',
  'deployment.rollback',
  'drift.alert',
  'schedule.failed',
];

export function Webhooks() {
  const { wsId } = useParams<{ wsId: string }>();
  const workspaceId = wsId ?? '';
  const qc = useQueryClient();

  const list = useQuery({
    queryKey: ['workspaces', workspaceId, 'webhooks'],
    queryFn: () => webhooksApi.list(workspaceId),
    enabled: !!workspaceId,
  });

  const patch = useMutation({
    mutationFn: (vars: { id: string; enabled: boolean }) =>
      webhooksApi.patch(vars.id, { enabled: vars.enabled }),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ['workspaces', workspaceId, 'webhooks'] }),
  });

  const remove = useMutation({
    mutationFn: (id: string) => webhooksApi.remove(id),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ['workspaces', workspaceId, 'webhooks'] }),
  });

  const test = useMutation({
    mutationFn: (id: string) => webhooksApi.test(id),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ['workspaces', workspaceId, 'webhooks'] }),
  });

  return (
    <div className="space-y-8">
      <header>
        <h1 className="h-page">Webhooks</h1>
        <p className="mt-2 text-sm text-ink-500">
          Outgoing HTTP POSTs fired on platform events. HMAC-SHA256 signed
          via the configured secret (sent as{' '}
          <code className="font-mono text-xs">X-PyCaret-Signature</code>).
        </p>
      </header>

      <NewWebhookForm workspaceId={workspaceId} />

      <section>
        <h2 className="h-section mb-4">Subscriptions</h2>
        {list.isPending ? (
          <div className="card text-sm text-ink-500">Loading…</div>
        ) : !list.data || list.data.items.length === 0 ? (
          <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 p-8 text-center text-sm text-ink-500">
            No webhooks yet — subscribe to platform events to integrate Slack,
            PagerDuty, or your own service.
          </div>
        ) : (
          <div className="card overflow-hidden p-0">
            <table className="w-full text-sm">
              <thead className="bg-white text-ink-500 dark:bg-ink-900">
                <tr>
                  <th className="px-4 py-2 text-left font-medium">URL</th>
                  <th className="px-4 py-2 text-left font-medium">Events</th>
                  <th className="px-4 py-2 text-center font-medium">Secret</th>
                  <th className="px-4 py-2 text-left font-medium">Last fired</th>
                  <th className="px-4 py-2 text-center font-medium">Enabled</th>
                  <th className="px-4 py-2 text-right font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {list.data.items.map((w: Webhook) => (
                  <tr
                    key={w.id}
                    className="border-t border-ink-200 dark:border-ink-800 hover:bg-ink-50 dark:hover:bg-ink-950/40"
                  >
                    <td
                      className="px-4 py-2 font-mono text-xs text-ink-800 dark:text-ink-200 max-w-md truncate"
                      title={w.url}
                    >
                      {w.url}
                    </td>
                    <td className="px-4 py-2 text-xs">
                      <div className="flex flex-wrap gap-1">
                        {w.event_types.map((e) => (
                          <span key={e} className="kbd">
                            {e}
                          </span>
                        ))}
                      </div>
                    </td>
                    <td className="px-4 py-2 text-center">
                      {w.has_secret ? (
                        <span className="pill-success" title="Encrypted at rest">
                          signed
                        </span>
                      ) : (
                        <span className="pill-warn" title="No HMAC secret">
                          unsigned
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-2 text-xs">
                      {w.last_fired_at ? (
                        <div className="flex items-center gap-2">
                          {w.last_status_code && w.last_status_code < 400 ? (
                            <span className="pill-success">
                              HTTP {w.last_status_code}
                            </span>
                          ) : (
                            <span className="pill-danger">
                              HTTP {w.last_status_code ?? '—'}
                            </span>
                          )}
                          <span className="text-ink-500">
                            {new Date(w.last_fired_at).toLocaleString()}
                          </span>
                        </div>
                      ) : (
                        <span className="text-ink-400">never</span>
                      )}
                    </td>
                    <td className="px-4 py-2 text-center">
                      <input
                        type="checkbox"
                        checked={w.enabled}
                        disabled={patch.isPending}
                        onChange={(e) =>
                          patch.mutate({ id: w.id, enabled: e.target.checked })
                        }
                      />
                    </td>
                    <td className="px-4 py-2 text-right text-xs">
                      <button
                        className="text-accent-600 hover:underline"
                        disabled={test.isPending}
                        onClick={() => test.mutate(w.id)}
                      >
                        Test
                      </button>
                      <span className="text-ink-300 mx-1">·</span>
                      <button
                        className="text-danger-600 hover:underline"
                        onClick={() => {
                          if (confirm(`Delete ${w.url}?`)) remove.mutate(w.id);
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
      </section>
    </div>
  );
}

function NewWebhookForm({ workspaceId }: { workspaceId: string }) {
  const qc = useQueryClient();
  const [url, setUrl] = useState('');
  const [secret, setSecret] = useState('');
  const [selected, setSelected] = useState<Set<string>>(
    new Set(['run.succeeded', 'drift.alert']),
  );

  const create = useMutation({
    mutationFn: () =>
      webhooksApi.create(workspaceId, {
        url,
        event_types: Array.from(selected),
        secret: secret || undefined,
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['workspaces', workspaceId, 'webhooks'] });
      setUrl('');
      setSecret('');
    },
  });

  return (
    <section className="card">
      <h2 className="text-sm font-medium text-ink-900 mb-3">New webhook</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div>
          <label className="field" htmlFor="url">Target URL</label>
          <input
            id="url"
            className="input"
            placeholder="https://example.com/hook"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
        </div>
        <div>
          <label className="field" htmlFor="secret">
            Shared secret (optional, ≥ 8 chars)
          </label>
          <input
            id="secret"
            className="input"
            type="password"
            value={secret}
            onChange={(e) => setSecret(e.target.value)}
            placeholder="used for HMAC signature"
          />
        </div>
        <div className="md:col-span-2">
          <div className="field mb-2">Event types</div>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-1.5">
            {EVENT_TYPES.map((e) => {
              const checked = selected.has(e);
              return (
                <label
                  key={e}
                  className="flex items-center gap-2 text-xs cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => {
                      const next = new Set(selected);
                      if (checked) next.delete(e);
                      else next.add(e);
                      setSelected(next);
                    }}
                  />
                  <code className="text-ink-700">{e}</code>
                </label>
              );
            })}
          </div>
        </div>
      </div>
      <div className="mt-3 flex items-center gap-3">
        <button
          className="btn-primary"
          disabled={
            !url.trim() ||
            selected.size === 0 ||
            (secret.length > 0 && secret.length < 8) ||
            create.isPending
          }
          onClick={() => create.mutate()}
        >
          {create.isPending ? 'Creating…' : 'Create webhook'}
        </button>
        {create.error && (
          <p className="error">{errorMessage(create.error)}</p>
        )}
      </div>
    </section>
  );
}
