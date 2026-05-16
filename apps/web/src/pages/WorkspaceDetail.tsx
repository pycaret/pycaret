/**
 * /workspaces/:id — workspace landing page.
 *
 * Hero (name + description + primary CTA) → stats strip → Projects
 * section → Data sources section. Create flows are **modals**, not
 * always-visible side panels — the page stays readable when you're
 * scanning, and the empty state has a clear single CTA.
 */
import { useEffect, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import {
  dataSourcesApi,
  deploymentsApi,
  pipelinesApi,
  projectsApi,
  workspacesApi,
} from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import { Dialog } from '@/components/Dialog';
import { DataSourcesSection } from '@/components/DataSourcesSection';

export function WorkspaceDetail() {
  const { id = '' } = useParams<{ id: string }>();

  const ws = useQuery({
    queryKey: ['workspaces', id],
    queryFn: () => workspacesApi.get(id),
    enabled: !!id,
  });
  const projects = useQuery({
    queryKey: ['projects', id],
    queryFn: () => projectsApi.list(id),
    enabled: !!id,
  });
  const dataSources = useQuery({
    queryKey: ['data-sources', id],
    queryFn: () => dataSourcesApi.list(id),
    enabled: !!id,
  });
  const pipelines = useQuery({
    queryKey: ['pipelines', 'by-workspace', id],
    queryFn: () => pipelinesApi.list(id),
    enabled: !!id,
  });
  const deployments = useQuery({
    queryKey: ['deployments', 'by-workspace', id],
    queryFn: () => deploymentsApi.list(id),
    enabled: !!id,
  });

  const [newProjectOpen, setNewProjectOpen] = useState(false);

  const projectCount = projects.data?.length ?? 0;
  const csvCount = (dataSources.data ?? []).filter(
    (d) => d.kind === 'csv_upload',
  ).length;
  const pipelineCount = pipelines.data?.length ?? 0;
  const deploymentCount = deployments.data?.length ?? 0;

  return (
    <div className="space-y-10">
      {/* ─── Hero ─────────────────────────────────────────────── */}
      <header className="space-y-4">
        <nav className="text-xs text-ink-500">
          <Link to="/" className="hover:text-ink-900 dark:text-ink-50 transition-colors">
            Workspaces
          </Link>
          <span className="mx-1.5 text-ink-300">/</span>
          <span className="text-ink-700 dark:text-ink-300">{ws.data?.name ?? '…'}</span>
        </nav>
        <div className="flex items-start justify-between gap-6">
          <div className="min-w-0">
            <h1 className="h-page truncate">
              {ws.data?.name ?? 'Loading…'}
            </h1>
            {ws.data?.description && (
              <p className="mt-2 text-sm text-ink-600 dark:text-ink-400 dark:text-ink-500 max-w-2xl">
                {ws.data.description}
              </p>
            )}
          </div>
          <button
            type="button"
            onClick={() => setNewProjectOpen(true)}
            className="btn-primary shrink-0"
          >
            <PlusIcon />
            New project
          </button>
        </div>
      </header>

      {/* ─── Stats strip ─────────────────────────────────────── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-px rounded-xl overflow-hidden bg-ink-200 dark:bg-ink-800 border border-ink-200 dark:border-ink-800">
        <Stat label="Projects" value={projectCount} />
        <Stat label="Data sources" value={csvCount} />
        <Stat
          label="Model versions"
          value={pipelineCount}
          to={`/workspaces/${id}/models`}
        />
        <Stat
          label="Deployments"
          value={deploymentCount}
          to={`/workspaces/${id}/deployments`}
        />
      </div>

      {/* ─── Projects ────────────────────────────────────────── */}
      <section>
        <header className="mb-3">
          <h2 className="h-section">Projects</h2>
          <p className="text-sm text-ink-500 mt-0.5">
            Each project groups experiments, runs, and pipelines around one ML problem.
          </p>
        </header>

        {projects.isLoading && (
          <div className="card text-sm text-ink-500">Loading…</div>
        )}
        {projects.error && (
          <div className="card text-sm text-danger-600">
            {errorMessage(projects.error)}
          </div>
        )}

        {projects.data && projectCount === 0 && (
          <EmptyState
            icon={<FolderIcon />}
            title="No projects yet"
            description="A project groups experiments, runs, pipelines, and deployments around one ML problem. Create your first one to get started."
            cta={{
              label: 'Create your first project',
              onClick: () => setNewProjectOpen(true),
            }}
          />
        )}

        {projectCount > 0 && (
          <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 auto-rows-fr">
            {projects.data?.map((p) => (
              <li key={p.id} className="h-full">
                <Link
                  to={`/workspaces/${id}/projects/${p.id}`}
                  className="block h-full group rounded-xl bg-white dark:bg-ink-900 border border-ink-200 dark:border-ink-800 shadow-soft-1 p-4 hover:border-ink-300 dark:hover:border-ink-700 hover:shadow-soft-2 transition-all"
                >
                  <div className="flex items-start gap-3 h-full">
                    <span className="h-9 w-9 rounded-md bg-accent-50 dark:bg-accent-500/15 text-accent-600 dark:text-accent-400 flex items-center justify-center shrink-0">
                      <FolderIcon />
                    </span>
                    <div className="min-w-0 flex-1 flex flex-col">
                      <h3 className="text-sm font-semibold text-ink-900 dark:text-ink-50 truncate group-hover:text-accent-700 dark:group-hover:text-accent-400 transition-colors">
                        {p.name}
                      </h3>
                      {p.description ? (
                        <p className="mt-1 text-xs text-ink-500 line-clamp-2">
                          {p.description}
                        </p>
                      ) : (
                        <p className="mt-1 text-xs text-ink-400 italic">
                          No description
                        </p>
                      )}
                      <div className="mt-auto pt-3 flex items-center justify-between gap-3 text-[11px] text-ink-500">
                        <span title={new Date(p.created_at).toLocaleString()}>
                          Created {fmtRelativeDate(p.created_at)}
                        </span>
                        {p.tags.length > 0 && (
                          <span className="flex flex-wrap gap-1 justify-end">
                            {p.tags.slice(0, 3).map((t) => (
                              <span key={t} className="pill-neutral">
                                {t}
                              </span>
                            ))}
                            {p.tags.length > 3 && (
                              <span className="text-ink-400">+{p.tags.length - 3}</span>
                            )}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </section>

      {/* ─── Data sources ────────────────────────────────────── */}
      {id && <DataSourcesSection workspaceId={id} />}

      {/* ─── New-project modal ─────────────────────────────── */}
      <NewProjectDialog
        workspaceId={id}
        open={newProjectOpen}
        onClose={() => setNewProjectOpen(false)}
      />
    </div>
  );
}

// ─── Relative-date helper ─────────────────────────────────────────

function fmtRelativeDate(iso: string): string {
  const then = new Date(iso).getTime();
  if (!Number.isFinite(then)) return '—';
  const diffMs = Date.now() - then;
  const sec = Math.round(diffMs / 1000);
  if (sec < 60) return 'just now';
  const min = Math.round(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.round(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const days = Math.round(hr / 24);
  if (days < 7) return `${days}d ago`;
  if (days < 30) return `${Math.round(days / 7)}w ago`;
  return new Date(iso).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}

// ─── Stats card ────────────────────────────────────────────────────

function Stat({
  label,
  value,
  hint,
  to,
}: {
  label: string;
  value: number | string;
  hint?: string;
  to?: string;
}) {
  const inner = (
    <>
      <div className="text-xs font-medium text-ink-500 uppercase tracking-wider">
        {label}
      </div>
      <div className="mt-1 text-2xl font-semibold text-ink-900 dark:text-ink-50 tabular-nums">
        {value}
      </div>
      {hint && <div className="text-xs text-ink-400 dark:text-ink-500 mt-0.5">{hint}</div>}
    </>
  );
  if (to) {
    return (
      <Link
        to={to}
        className="bg-white dark:bg-ink-900 px-4 py-4 hover:bg-ink-50 dark:hover:bg-ink-800/40 transition-colors"
      >
        {inner}
      </Link>
    );
  }
  return (
    <div className="bg-white dark:bg-ink-900 px-4 py-4">
      {inner}
    </div>
  );
}

// ─── Empty state primitive ────────────────────────────────────────

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
    <div className="rounded-xl bg-white dark:bg-ink-900 border border-dashed border-ink-300 dark:border-ink-700 px-6 py-12 text-center">
      <div className="mx-auto h-12 w-12 rounded-xl bg-accent-50 text-accent-600 flex items-center justify-center mb-4">
        {icon}
      </div>
      <h3 className="text-sm font-semibold text-ink-900 dark:text-ink-50">{title}</h3>
      <p className="mt-1.5 text-sm text-ink-500 max-w-md mx-auto">
        {description}
      </p>
      <button
        type="button"
        onClick={cta.onClick}
        className="btn-primary mt-6"
      >
        <PlusIcon />
        {cta.label}
      </button>
    </div>
  );
}

// ─── New-project dialog ──────────────────────────────────────────

function NewProjectDialog({
  workspaceId,
  open,
  onClose,
}: {
  workspaceId: string;
  open: boolean;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [tagsInput, setTagsInput] = useState('');

  // Reset state on close.
  useEffect(() => {
    if (!open) {
      setName('');
      setDescription('');
      setTagsInput('');
    }
  }, [open]);

  const create = useMutation({
    mutationFn: () =>
      projectsApi.create(workspaceId, {
        name,
        description: description || undefined,
        tags: tagsInput
          .split(',')
          .map((t) => t.trim())
          .filter(Boolean),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['projects', workspaceId] });
      onClose();
    },
  });

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="New project"
      description="A project groups experiments, runs, pipelines, and deployments around one ML problem."
    >
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (name.trim()) create.mutate();
        }}
        className="space-y-4"
      >
        <div>
          <label className="field" htmlFor="p-name">
            Name <span className="text-ink-400 dark:text-ink-500 font-normal">*</span>
          </label>
          <input
            id="p-name"
            className="input"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. juice-classification"
            autoFocus
            required
          />
        </div>
        <div>
          <label className="field" htmlFor="p-desc">
            Description
          </label>
          <textarea
            id="p-desc"
            className="input resize-none"
            rows={3}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="What's this project about?"
          />
        </div>
        <div>
          <label className="field" htmlFor="p-tags">
            Tags
          </label>
          <input
            id="p-tags"
            className="input"
            value={tagsInput}
            onChange={(e) => setTagsInput(e.target.value)}
            placeholder="comma, separated"
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
            {create.isPending ? 'Creating…' : 'Create project'}
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
function FolderIcon() {
  return (
    <svg {...{ ...sx, width: '18', height: '18' }}>
      <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
    </svg>
  );
}
