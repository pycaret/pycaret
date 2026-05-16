/**
 * /workspaces/:wsId/connections — Phase 4 catalog connections.
 *
 * A Connection = "how to reach a backend" (Postgres host/port,
 * Snowflake account, S3 bucket). The credentials live in a linked
 * Secret. DataSources reference a Connection + the specific table /
 * file to expose.
 *
 * Per-kind config schemas are inlined here for v1; future cuts can
 * pull them from a `/connections/schema/{kind}` endpoint instead.
 */

import { useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useParams } from 'react-router-dom';
import { BackButton } from '@/components/BackButton';
import { Dialog } from '@/components/Dialog';
import { connectionsApi, secretsApi } from '@/api/endpoints';
import type { Connection } from '@/api/types';

type ConfigField = {
  key: string;
  label: string;
  placeholder?: string;
  type?: 'text' | 'number';
};

const KIND_SCHEMAS: Record<string, ConfigField[]> = {
  postgres: [
    { key: 'host', label: 'Host', placeholder: 'localhost' },
    { key: 'port', label: 'Port', placeholder: '5432', type: 'number' },
    { key: 'database', label: 'Database', placeholder: 'pycaret' },
    { key: 'user', label: 'User', placeholder: 'pycaret' },
    { key: 'schema', label: 'Schema (default: public)', placeholder: 'public' },
  ],
  // CSV uploads don't need a connection — they're DataSources directly.
  csv_upload: [
    { key: 'path', label: 'URI', placeholder: 'file:///path/to.csv' },
  ],
};

const KIND_OPTIONS = [
  { value: 'postgres', label: 'PostgreSQL' },
  { value: 'csv_upload', label: 'CSV file (URI)' },
];

export function Connections() {
  const { wsId = '' } = useParams<{ wsId: string }>();
  const qc = useQueryClient();
  const [createOpen, setCreateOpen] = useState(false);
  const [browseConn, setBrowseConn] = useState<Connection | null>(null);

  const list = useQuery({
    queryKey: ['connections', wsId],
    queryFn: () => connectionsApi.list(wsId),
    enabled: !!wsId,
  });

  const testConn = useMutation({
    mutationFn: (cid: string) => connectionsApi.test(wsId, cid),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['connections', wsId] }),
  });

  const remove = useMutation({
    mutationFn: (cid: string) => connectionsApi.delete(wsId, cid),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['connections', wsId] }),
  });

  return (
    <div className="space-y-6">
      <header className="flex items-baseline justify-between">
        <div>
          <BackButton />
          <h1 className="h-page mt-2">Connections</h1>
          <p className="muted small">
            Credentialed links to data backends. Phase 4. DataSources
            reference a Connection + a specific table.
          </p>
        </div>
        <button
          type="button"
          className="btn-primary"
          onClick={() => setCreateOpen(true)}
        >
          New connection
        </button>
      </header>

      {list.isLoading && (
        <div className="card text-sm text-ink-500">Loading…</div>
      )}
      {list.data && list.data.length === 0 && (
        <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 p-10 text-center">
          <h3 className="text-base font-semibold">No connections</h3>
          <p className="mt-1 text-sm text-ink-500">
            Register a Postgres or S3 backend so DataSources can target
            it.
          </p>
        </div>
      )}
      {list.data && list.data.length > 0 && (
        <ul className="space-y-2">
          {list.data.map((c) => (
            <ConnectionRow
              key={c.id}
              connection={c}
              onTest={() => testConn.mutate(c.id)}
              onBrowse={() => setBrowseConn(c)}
              onDelete={() => {
                if (
                  confirm(`Delete connection "${c.name}"? DataSources pointing at it stay but won't resolve.`)
                ) {
                  remove.mutate(c.id);
                }
              }}
              testing={testConn.isPending && testConn.variables === c.id}
            />
          ))}
        </ul>
      )}

      <NewConnectionDialog
        open={createOpen}
        wsId={wsId}
        onClose={() => setCreateOpen(false)}
        onSaved={() => {
          setCreateOpen(false);
          qc.invalidateQueries({ queryKey: ['connections', wsId] });
        }}
      />

      {browseConn && (
        <BrowseTablesDialog
          wsId={wsId}
          connection={browseConn}
          onClose={() => setBrowseConn(null)}
          onRegistered={() => {
            // Invalidate the workspace data-sources so the project
            // detail page picks the new row up immediately.
            qc.invalidateQueries({ queryKey: ['data-sources', wsId] });
          }}
        />
      )}
    </div>
  );
}

function ConnectionRow({
  connection,
  onTest,
  onBrowse,
  onDelete,
  testing,
}: {
  connection: Connection;
  onTest: () => void;
  onBrowse: () => void;
  onDelete: () => void;
  testing: boolean;
}) {
  const status = connection.last_test_status;
  return (
    <li className="card flex items-start gap-4">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-medium">{connection.name}</span>
          <span className="pill-neutral">{connection.kind}</span>
          {status === 'ok' && <span className="pill-success">ok</span>}
          {status === 'error' && (
            <span className="pill bg-rose-50 text-rose-700 dark:bg-rose-500/15 dark:text-rose-300">
              error
            </span>
          )}
        </div>
        <p className="text-xs text-ink-500 mt-1">
          {Object.entries(connection.config)
            .filter(([k]) => k !== 'password')
            .map(([k, v]) => `${k}=${String(v)}`)
            .join(' · ') || 'no config'}
        </p>
        {connection.last_test_error && (
          <p className="text-xs text-danger-600 mt-1">
            {connection.last_test_error}
          </p>
        )}
        {connection.last_tested_at && (
          <p className="text-xs text-ink-500 mt-1">
            tested {new Date(connection.last_tested_at).toLocaleString()}
          </p>
        )}
      </div>
      <div className="flex flex-col gap-1 shrink-0">
        <button
          type="button"
          className="btn-secondary text-xs"
          onClick={onTest}
          disabled={testing}
        >
          {testing ? 'Testing…' : 'Test'}
        </button>
        <button
          type="button"
          className="btn-primary text-xs"
          onClick={onBrowse}
          disabled={connection.last_test_status !== 'ok'}
          title={
            connection.last_test_status === 'ok'
              ? 'Browse tables and register as a DataSource'
              : 'Test the connection first'
          }
        >
          Browse tables
        </button>
        <button
          type="button"
          className="text-xs text-danger-600 hover:underline"
          onClick={onDelete}
        >
          Delete
        </button>
      </div>
    </li>
  );
}

function BrowseTablesDialog({
  wsId,
  connection,
  onClose,
  onRegistered,
}: {
  wsId: string;
  connection: Connection;
  onClose: () => void;
  onRegistered: () => void;
}) {
  const [selected, setSelected] = useState('');
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [registered, setRegistered] = useState<string | null>(null);

  const tables = useQuery({
    queryKey: ['connections', wsId, connection.id, 'tables'],
    queryFn: () => connectionsApi.tables(wsId, connection.id),
  });

  const register = useMutation({
    mutationFn: () =>
      connectionsApi.registerAsDataSource(wsId, {
        connection_id: connection.id,
        table: selected,
        name: name.trim() || undefined,
        description: description.trim() || undefined,
      }),
    onSuccess: (ds) => {
      setRegistered(String((ds as { name?: string }).name ?? selected));
      onRegistered();
    },
  });

  // Default the DataSource name to ``<connection>:<table>`` so it
  // reads self-documenting in the data-sources list.
  const defaultName = selected ? `${connection.name}:${selected}` : '';

  return (
    <Dialog
      open={true}
      onClose={onClose}
      title={`Browse tables · ${connection.name}`}
      size="md"
    >
      <div className="space-y-4">
        {tables.isLoading && (
          <div className="text-sm text-ink-500">Listing tables…</div>
        )}
        {tables.error && (
          <div className="text-sm text-danger-600">
            {(tables.error as Error).message}
          </div>
        )}
        {tables.data && tables.data.tables.length === 0 && (
          <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 p-6 text-center">
            <p className="text-sm text-ink-500">
              No tables exposed by this connection.
            </p>
          </div>
        )}
        {tables.data && tables.data.tables.length > 0 && (
          <div>
            <label className="field" htmlFor="table">
              Table
            </label>
            <select
              id="table"
              className="input w-full"
              value={selected}
              onChange={(e) => {
                setSelected(e.target.value);
                setName('');
              }}
            >
              <option value="">— pick a table —</option>
              {tables.data.tables.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
            <p className="hint mt-1">
              {tables.data.tables.length} tables in this connection.
            </p>
          </div>
        )}

        {selected && (
          <>
            <div>
              <label className="field" htmlFor="ds-name">
                DataSource name
              </label>
              <input
                id="ds-name"
                className="input w-full"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder={defaultName}
              />
            </div>
            <div>
              <label className="field" htmlFor="ds-desc">
                Description (optional)
              </label>
              <textarea
                id="ds-desc"
                className="input w-full"
                rows={2}
                value={description}
                onChange={(e) => setDescription(e.target.value)}
              />
            </div>
          </>
        )}

        {registered && (
          <div className="rounded-md bg-success-50 dark:bg-success-500/15 px-3 py-2 text-sm text-success-700 dark:text-success-300">
            Registered as <strong>{registered}</strong>. It now appears in the
            data-sources picker.
          </div>
        )}
        {register.error && (
          <p className="error">{(register.error as Error).message}</p>
        )}

        <div className="flex justify-end gap-2 pt-2">
          <button type="button" className="btn-secondary" onClick={onClose}>
            {registered ? 'Close' : 'Cancel'}
          </button>
          {!registered && (
            <button
              type="button"
              className="btn-primary"
              disabled={!selected || register.isPending}
              onClick={() => register.mutate()}
            >
              {register.isPending ? 'Registering…' : 'Register as DataSource'}
            </button>
          )}
        </div>
      </div>
    </Dialog>
  );
}

function NewConnectionDialog({
  open,
  wsId,
  onClose,
  onSaved,
}: {
  open: boolean;
  wsId: string;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [name, setName] = useState('');
  const [kind, setKind] = useState('postgres');
  const [config, setConfig] = useState<Record<string, string>>({});
  const [secretId, setSecretId] = useState('');

  const secrets = useQuery({
    queryKey: ['secrets', wsId],
    queryFn: () => secretsApi.list(wsId),
    enabled: !!wsId && open,
  });

  const schema = KIND_SCHEMAS[kind] ?? [];

  const save = useMutation({
    mutationFn: () => {
      // Coerce numeric fields.
      const cfg: Record<string, unknown> = {};
      for (const f of schema) {
        const raw = config[f.key];
        if (raw === undefined || raw === '') continue;
        cfg[f.key] = f.type === 'number' ? Number(raw) : raw;
      }
      return connectionsApi.create(wsId, {
        name: name.trim(),
        kind,
        config: cfg,
        secret_id: secretId || undefined,
      });
    },
    onSuccess: () => {
      setName('');
      setConfig({});
      setSecretId('');
      onSaved();
    },
  });

  const dbSecrets = useMemo(
    () =>
      (secrets.data ?? []).filter(
        (s) => kind !== 'postgres' || s.kind === 'db_password' || s.kind === 'opaque',
      ),
    [secrets.data, kind],
  );

  return (
    <Dialog open={open} onClose={onClose} title="New connection" size="md">
      <div className="space-y-3">
        <label className="field">
          <span className="muted small">Name</span>
          <input
            className="input w-full"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="prod-postgres"
            autoFocus
          />
        </label>
        <label className="field">
          <span className="muted small">Kind</span>
          <select
            className="input w-full"
            value={kind}
            onChange={(e) => {
              setKind(e.target.value);
              setConfig({});
            }}
          >
            {KIND_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
        {schema.map((f) => (
          <label key={f.key} className="field">
            <span className="muted small">{f.label}</span>
            <input
              className="input w-full"
              type={f.type ?? 'text'}
              placeholder={f.placeholder}
              value={config[f.key] ?? ''}
              onChange={(e) =>
                setConfig((s) => ({ ...s, [f.key]: e.target.value }))
              }
            />
          </label>
        ))}
        {kind === 'postgres' && (
          <label className="field">
            <span className="muted small">Password secret</span>
            <select
              className="input w-full"
              value={secretId}
              onChange={(e) => setSecretId(e.target.value)}
            >
              <option value="">— pick a secret —</option>
              {dbSecrets.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
            {dbSecrets.length === 0 && (
              <span className="muted small">
                No secrets yet — create one under Secrets first.
              </span>
            )}
          </label>
        )}
        {save.error && (
          <p className="text-xs text-danger-600">
            {(save.error as Error).message}
          </p>
        )}
        <div className="flex justify-end gap-2">
          <button type="button" className="btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="btn-primary"
            disabled={!name.trim() || save.isPending}
            onClick={() => save.mutate()}
          >
            {save.isPending ? 'Saving…' : 'Create'}
          </button>
        </div>
      </div>
    </Dialog>
  );
}
