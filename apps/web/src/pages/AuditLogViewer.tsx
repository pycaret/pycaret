/**
 * /admin/audit — installation-wide audit log viewer, superuser only.
 *
 * A simple table with action / method / path / status / user / workspace
 * columns plus a filter bar (action + target_type + limit). Clicking a
 * row opens its scrubbed payload inline.
 *
 * Workspace-admin-scoped view lives at
 * /workspaces/:wsId/audit and reuses the same table, filtered server-side.
 */

import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { auditApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import { useAuthStore } from '@/state/auth';
import type { AuditLogRead } from '@/api/types';

export function AuditLogViewer() {
  const me = useAuthStore((s) => s.user);
  const [action, setAction] = useState('');
  const [targetType, setTargetType] = useState('');
  const [limit, setLimit] = useState(100);
  const [expanded, setExpanded] = useState<string | null>(null);

  const filters = useMemo(
    () => ({
      action: action.trim() || undefined,
      target_type: targetType.trim() || undefined,
      limit,
    }),
    [action, targetType, limit],
  );

  const rows = useQuery({
    queryKey: ['audit-logs', filters],
    queryFn: () => auditApi.listAdmin(filters),
    enabled: !!me?.is_superuser,
  });

  if (me && !me.is_superuser) {
    return (
      <div className="space-y-4">
        <h1 className="text-xl font-semibold">Audit log</h1>
        <p className="error">
          Superuser privilege required to view the installation-wide audit log.
          Workspace admins see a scoped log at{' '}
          <code className="font-mono text-xs">/workspaces/:id/audit</code>.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <header>
        <nav className="text-xs text-ink-500 mb-2">
          <Link to="/" className="hover:text-ink-900">
            Home
          </Link>
          <span className="mx-1">/</span>
          <span>Admin</span>
          <span className="mx-1">/</span>
          <span>Audit log</span>
        </nav>
        <h1 className="text-xl font-semibold">Audit log</h1>
        <p className="mt-1 text-sm text-ink-500">
          Every mutating API call (POST / PATCH / PUT / DELETE) on /api/v1/* is
          recorded here with the caller + scrubbed payload. Reads are not
          logged.
        </p>
      </header>

      <section className="card">
        <form
          className="grid gap-3 md:grid-cols-[1fr_1fr_10rem_auto]"
          onSubmit={(e) => {
            e.preventDefault();
            rows.refetch();
          }}
        >
          <div>
            <label className="field" htmlFor="f-action">
              action
            </label>
            <input
              id="f-action"
              className="input"
              placeholder="e.g. workspaces.create"
              value={action}
              onChange={(e) => setAction(e.target.value)}
            />
          </div>
          <div>
            <label className="field" htmlFor="f-target">
              target_type
            </label>
            <input
              id="f-target"
              className="input"
              placeholder="e.g. workspace"
              value={targetType}
              onChange={(e) => setTargetType(e.target.value)}
            />
          </div>
          <div>
            <label className="field" htmlFor="f-limit">
              limit
            </label>
            <input
              id="f-limit"
              className="input"
              type="number"
              min={1}
              max={500}
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value) || 100)}
            />
          </div>
          <div className="flex items-end">
            <button type="submit" className="btn-secondary">
              Apply
            </button>
          </div>
        </form>
      </section>

      {rows.isLoading && <p className="hint">Loading…</p>}
      {rows.error && <p className="error">{errorMessage(rows.error)}</p>}

      {rows.data && (
        <section className="card p-0 overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-white text-ink-500">
              <tr>
                <th className="px-4 py-2 text-left font-medium">When</th>
                <th className="px-4 py-2 text-left font-medium">Action</th>
                <th className="px-4 py-2 text-left font-medium">Method</th>
                <th className="px-4 py-2 text-left font-medium">Path</th>
                <th className="px-4 py-2 text-left font-medium">Status</th>
                <th className="px-4 py-2 text-left font-medium">User</th>
              </tr>
            </thead>
            <tbody>
              {rows.data.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-4 py-8 text-center hint">
                    No matching audit rows.
                  </td>
                </tr>
              )}
              {rows.data.map((row) => (
                <RowCells
                  key={row.id}
                  row={row}
                  expanded={expanded === row.id}
                  onToggle={() =>
                    setExpanded((prev) => (prev === row.id ? null : row.id))
                  }
                />
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}

function RowCells({
  row,
  expanded,
  onToggle,
}: {
  row: AuditLogRead;
  expanded: boolean;
  onToggle: () => void;
}) {
  const tone =
    row.status_code && row.status_code >= 500
      ? 'text-danger-500'
      : row.status_code && row.status_code >= 400
        ? 'text-warn-500'
        : 'text-ink-600';
  return (
    <>
      <tr
        className="border-t border-ink-200 hover:bg-ink-50 cursor-pointer"
        onClick={onToggle}
      >
        <td className="px-4 py-2 text-xs text-ink-500 font-mono whitespace-nowrap">
          {new Date(row.created_at).toLocaleString()}
        </td>
        <td className="px-4 py-2 font-mono text-xs text-ink-900">{row.action}</td>
        <td className="px-4 py-2 font-mono text-xs text-ink-500">
          {row.method}
        </td>
        <td
          className="px-4 py-2 font-mono text-xs text-ink-500 truncate max-w-md"
          title={row.path}
        >
          {row.path}
        </td>
        <td className={`px-4 py-2 font-mono text-xs ${tone}`}>
          {row.status_code ?? '—'}
        </td>
        <td className="px-4 py-2 font-mono text-xs text-ink-500 truncate max-w-xs">
          {row.user_id ?? '—'}
        </td>
      </tr>
      {expanded && (
        <tr className="border-t border-ink-200 bg-white/80">
          <td colSpan={6} className="px-4 py-3">
            <div className="grid gap-3 md:grid-cols-2">
              <div>
                <p className="text-xs uppercase tracking-wider text-ink-500 mb-1">
                  Payload (scrubbed)
                </p>
                <pre className="bg-white border border-ink-200 rounded p-2 font-mono text-xs text-ink-900 overflow-x-auto">
                  {row.payload ? JSON.stringify(row.payload, null, 2) : '(none)'}
                </pre>
              </div>
              <div className="space-y-1 text-xs text-ink-500 font-mono">
                <p>
                  workspace_id: <span className="text-ink-900">{row.workspace_id ?? '—'}</span>
                </p>
                <p>
                  target_type:{' '}
                  <span className="text-ink-900">{row.target_type ?? '—'}</span>
                </p>
                <p>
                  target_id: <span className="text-ink-900">{row.target_id ?? '—'}</span>
                </p>
                <p>
                  ip_address: <span className="text-ink-900">{row.ip_address ?? '—'}</span>
                </p>
                <p
                  className="truncate"
                  title={row.user_agent ?? undefined}
                >
                  user_agent: <span className="text-ink-900">{row.user_agent ?? '—'}</span>
                </p>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}
