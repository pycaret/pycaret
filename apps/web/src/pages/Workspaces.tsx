/**
 * /  — list of workspaces the current user belongs to.
 *
 * Modal-based create flow (no permanent right-sidebar form).
 * Empty state with primary CTA, otherwise a 3-column responsive
 * grid of workspace cards.
 */
import { useEffect, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { workspacesApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import { Dialog } from '@/components/Dialog';

export function Workspaces() {
  const list = useQuery({ queryKey: ['workspaces'], queryFn: workspacesApi.list });
  const [createOpen, setCreateOpen] = useState(false);

  const count = list.data?.length ?? 0;

  return (
    <div className="space-y-8">
      {/* ─── Hero ─────────────────────────────────────────────── */}
      <header className="flex items-end justify-between gap-6">
        <div>
          <h1 className="h-page">Workspaces</h1>
          <p className="mt-2 text-sm text-ink-500">
            A workspace groups projects, datasets, pipelines, and deployments
            for a team or environment.
          </p>
        </div>
        {count > 0 && (
          <button
            type="button"
            onClick={() => setCreateOpen(true)}
            className="btn-primary shrink-0"
          >
            <PlusIcon />
            New workspace
          </button>
        )}
      </header>

      {list.isLoading && (
        <div className="card text-sm text-ink-500">Loading…</div>
      )}
      {list.error && (
        <div className="card text-sm text-danger-600">
          {errorMessage(list.error)}
        </div>
      )}

      {list.data && count === 0 && (
        <EmptyState
          icon={<HomeIcon />}
          title="No workspaces yet"
          description="Create your first workspace to start organizing projects, datasets, and ML experiments."
          cta={{
            label: 'Create your first workspace',
            onClick: () => setCreateOpen(true),
          }}
        />
      )}

      {count > 0 && (
        <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 auto-rows-fr">
          {list.data?.map((w) => (
            <li key={w.id} className="h-full">
              <Link
                to={`/workspaces/${w.id}`}
                className="block group h-full rounded-xl bg-white dark:bg-ink-900 border border-ink-200 dark:border-ink-800 shadow-soft-1 p-4 hover:border-ink-300 dark:hover:border-ink-700 hover:shadow-soft-2 transition-all"
              >
                <div className="flex items-start gap-3 h-full">
                  <span className="h-9 w-9 rounded-md bg-accent-50 dark:bg-accent-500/15 text-accent-600 dark:text-accent-400 flex items-center justify-center shrink-0">
                    <HomeIcon />
                  </span>
                  <div className="min-w-0 flex-1 flex flex-col">
                    <h3 className="text-sm font-semibold text-ink-900 dark:text-ink-50 truncate group-hover:text-accent-700 dark:group-hover:text-accent-400 transition-colors">
                      {w.name}
                    </h3>
                    {w.description ? (
                      <p className="mt-1 text-xs text-ink-500 line-clamp-2">
                        {w.description}
                      </p>
                    ) : (
                      <p className="mt-1 text-xs text-ink-400 italic">
                        No description
                      </p>
                    )}
                    <p className="mt-auto pt-2 text-xs text-ink-400">
                      Created {new Date(w.created_at).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              </Link>
            </li>
          ))}
        </ul>
      )}

      <NewWorkspaceDialog
        open={createOpen}
        onClose={() => setCreateOpen(false)}
      />
    </div>
  );
}

// ─── Empty-state primitive ────────────────────────────────────────

function EmptyState({
  icon,
  title,
  description,
  cta,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
  cta: { label: string; onClick: () => void };
}) {
  return (
    <div className="rounded-xl bg-white dark:bg-ink-900 border border-dashed border-ink-300 dark:border-ink-700 px-6 py-14 text-center">
      <div className="mx-auto h-12 w-12 rounded-xl bg-accent-50 dark:bg-accent-500/15 text-accent-600 dark:text-accent-400 flex items-center justify-center mb-4">
        {icon}
      </div>
      <h3 className="text-base font-semibold text-ink-900 dark:text-ink-50">
        {title}
      </h3>
      <p className="mt-2 text-sm text-ink-500 max-w-md mx-auto">
        {description}
      </p>
      <button type="button" onClick={cta.onClick} className="btn-primary mt-6">
        <PlusIcon />
        {cta.label}
      </button>
    </div>
  );
}

// ─── New-workspace dialog ─────────────────────────────────────────

function NewWorkspaceDialog({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');

  useEffect(() => {
    if (!open) {
      setName('');
      setDescription('');
    }
  }, [open]);

  const create = useMutation({
    mutationFn: () =>
      workspacesApi.create({ name, description: description || undefined }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['workspaces'] });
      onClose();
    },
  });

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="New workspace"
      description="A workspace groups projects, datasets, pipelines, and deployments — usually one per team or environment."
    >
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (name.trim()) create.mutate();
        }}
        className="space-y-4"
      >
        <div>
          <label className="field" htmlFor="ws-name">
            Name <span className="text-ink-400 font-normal">*</span>
          </label>
          <input
            id="ws-name"
            className="input"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. production"
            autoFocus
            required
          />
        </div>
        <div>
          <label className="field" htmlFor="ws-desc">
            Description
          </label>
          <textarea
            id="ws-desc"
            className="input resize-none"
            rows={3}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="What's this workspace for?"
          />
        </div>
        {create.error && <p className="error">{errorMessage(create.error)}</p>}
        <div className="flex items-center justify-end gap-2 pt-2">
          <button type="button" onClick={onClose} className="btn-ghost">
            Cancel
          </button>
          <button
            type="submit"
            className="btn-primary"
            disabled={create.isPending || !name.trim()}
          >
            {create.isPending ? 'Creating…' : 'Create workspace'}
          </button>
        </div>
      </form>
    </Dialog>
  );
}

// ─── Icons ────────────────────────────────────────────────────────

const sx = {
  width: '14',
  height: '14',
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: '2',
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  'aria-hidden': true,
};

function PlusIcon() {
  return (
    <svg {...sx}>
      <path d="M12 5v14 M5 12h14" />
    </svg>
  );
}
function HomeIcon() {
  return (
    <svg {...{ ...sx, width: '18', height: '18' }}>
      <path d="M3 9.5 12 3l9 6.5V21H3z" />
      <path d="M9 21V12h6v9" />
    </svg>
  );
}
