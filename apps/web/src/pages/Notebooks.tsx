/**
 * /workspaces/:wsId/projects/:projectId/notebooks — Phase 8 notebook
 * list. Click into one to launch a JupyterLab session.
 *
 * The launcher is iframe-based — when a session starts in ``docker``
 * mode the manager returns a token-bearing URL that we drop straight
 * into the iframe. In ``local`` mode the API returns a
 * ``pycaret://local-notebook`` URL the iframe can't load; we render a
 * "backend unavailable" placeholder with a hint instead.
 */

import { useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useParams } from 'react-router-dom';
import { notebooksApi } from '@/api/endpoints';
import { BackButton } from '@/components/BackButton';
import { Dialog } from '@/components/Dialog';
import type { Notebook, NotebookSessionStart } from '@/api/types';

export function Notebooks() {
  const { projectId = '' } = useParams<{
    wsId: string;
    projectId: string;
  }>();
  const qc = useQueryClient();
  const [createOpen, setCreateOpen] = useState(false);
  const [newName, setNewName] = useState('');
  const [newKernel, setNewKernel] = useState('python3');
  const [activeSession, setActiveSession] = useState<NotebookSessionStart | null>(
    null,
  );

  const list = useQuery({
    queryKey: ['notebooks', 'for-project', projectId],
    queryFn: () => notebooksApi.forProject(projectId),
    enabled: !!projectId,
  });

  const create = useMutation({
    mutationFn: () =>
      notebooksApi.create(projectId, {
        name: newName.trim(),
        kernel: newKernel,
      }),
    onSuccess: () => {
      setCreateOpen(false);
      setNewName('');
      qc.invalidateQueries({ queryKey: ['notebooks', 'for-project', projectId] });
    },
  });

  const startSession = useMutation({
    mutationFn: (notebookId: string) => notebooksApi.startSession(notebookId),
    onSuccess: (data) => setActiveSession(data),
  });

  return (
    <div className="space-y-6">
      <header className="flex items-baseline justify-between">
        <div>
          <BackButton />
          <h1 className="h-page mt-2">Notebooks</h1>
          <p className="muted small">
            JupyterLab containers spawned per session. Phase 8.
          </p>
        </div>
        <button
          type="button"
          className="btn-primary"
          onClick={() => setCreateOpen(true)}
        >
          New notebook
        </button>
      </header>

      {list.isLoading && <div className="card text-sm text-ink-500">Loading…</div>}
      {list.data && list.data.length === 0 && (
        <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 p-10 text-center">
          <h3 className="text-base font-semibold">No notebooks yet</h3>
          <p className="mt-1 text-sm text-ink-500">
            Spawn a JupyterLab session bound to this project.
          </p>
          <button
            type="button"
            className="btn-primary mt-5"
            onClick={() => setCreateOpen(true)}
          >
            New notebook
          </button>
        </div>
      )}
      {list.data && list.data.length > 0 && (
        <div className="card overflow-hidden p-0">
          <table className="w-full text-sm">
            <thead className="bg-white text-ink-500 dark:bg-ink-900 border-b border-ink-200 dark:border-ink-800">
              <tr>
                <th className="px-4 py-2 text-left font-medium">Name</th>
                <th className="px-4 py-2 text-left font-medium">Kernel</th>
                <th className="px-4 py-2 text-left font-medium">Modified</th>
                <th className="px-4 py-2 text-right font-medium" />
              </tr>
            </thead>
            <tbody>
              {list.data.map((n) => (
                <NotebookRow
                  key={n.id}
                  notebook={n}
                  isStarting={
                    startSession.isPending && startSession.variables === n.id
                  }
                  onLaunch={() => startSession.mutate(n.id)}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}

      <Dialog
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        title="New notebook"
        size="sm"
      >
        <div className="space-y-3">
          <label className="field">
            <span className="muted small">Name</span>
            <input
              className="input w-full"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="exploration.ipynb"
              autoFocus
            />
          </label>
          <label className="field">
            <span className="muted small">Kernel</span>
            <select
              className="input w-full"
              value={newKernel}
              onChange={(e) => setNewKernel(e.target.value)}
            >
              <option value="python3">python3</option>
              <option value="ir">R (ir)</option>
              <option value="julia-1.9">Julia 1.9</option>
            </select>
          </label>
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
              disabled={!newName.trim() || create.isPending}
              onClick={() => create.mutate()}
            >
              {create.isPending ? 'Creating…' : 'Create'}
            </button>
          </div>
        </div>
      </Dialog>

      <NotebookSessionDialog
        session={activeSession}
        onClose={() => setActiveSession(null)}
      />
    </div>
  );
}

function NotebookRow({
  notebook,
  isStarting,
  onLaunch,
}: {
  notebook: Notebook;
  isStarting: boolean;
  onLaunch: () => void;
}) {
  return (
    <tr className="border-t border-ink-200 dark:border-ink-800 hover:bg-ink-50 dark:hover:bg-ink-950/40">
      <td className="px-4 py-2 font-medium">{notebook.name}</td>
      <td className="px-4 py-2 text-ink-700 dark:text-ink-300">
        <span className="pill-neutral">{notebook.kernel}</span>
      </td>
      <td className="px-4 py-2 text-xs text-ink-500">
        {notebook.last_modified_at
          ? new Date(notebook.last_modified_at).toLocaleString()
          : '—'}
      </td>
      <td className="px-4 py-2 text-right">
        <button
          type="button"
          className="btn-secondary"
          onClick={onLaunch}
          disabled={isStarting}
        >
          {isStarting ? 'Starting…' : 'Open'}
        </button>
      </td>
    </tr>
  );
}

function NotebookSessionDialog({
  session,
  onClose,
}: {
  session: NotebookSessionStart | null;
  onClose: () => void;
}) {
  const url = session?.url ?? '';
  const isLocalStub = url.startsWith('pycaret://');
  const externalUrl = useMemo(
    () => (isLocalStub ? null : url),
    [url, isLocalStub],
  );

  return (
    <Dialog
      open={session !== null}
      onClose={onClose}
      title={`Session ${session?.id.slice(0, 8) ?? ''}…`}
      size="lg"
    >
      {session === null ? null : isLocalStub ? (
        <div className="rounded-xl bg-ink-50 dark:bg-ink-950 p-6 text-sm space-y-2">
          <p className="font-semibold">Notebook backend unavailable.</p>
          <p className="muted">
            The server is running with{' '}
            <code>PYCARET_NOTEBOOK_BACKEND=local</code>. To open a real
            JupyterLab session, set the backend to <code>docker</code>{' '}
            and restart the API.
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          <p className="text-xs text-ink-500">
            Embedded JupyterLab — direct link:{' '}
            <a
              href={externalUrl ?? '#'}
              target="_blank"
              rel="noreferrer"
              className="underline"
            >
              open in new tab
            </a>
          </p>
          <iframe
            src={externalUrl ?? 'about:blank'}
            className="w-full h-[70vh] rounded-lg border border-ink-200 dark:border-ink-800"
            title="notebook session"
          />
        </div>
      )}
    </Dialog>
  );
}

// Helper to keep the BackButton link sane when this page is opened
// from the project detail page. Exported for unit tests.
export function notebooksPathFor(wsId: string, projectId: string): string {
  return `/workspaces/${wsId}/projects/${projectId}/notebooks`;
}
