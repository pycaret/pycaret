import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import { projectsApi, workspacesApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import { DataSourcesCard } from '@/components/DataSourcesCard';

/** /workspaces/:id  — workspace summary + project list + create. */
export function WorkspaceDetail() {
  const { id = '' } = useParams<{ id: string }>();
  const qc = useQueryClient();

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

  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [tagsInput, setTagsInput] = useState('');

  const create = useMutation({
    mutationFn: () =>
      projectsApi.create(id, {
        name,
        description: description || undefined,
        tags: tagsInput
          .split(',')
          .map((t) => t.trim())
          .filter(Boolean),
      }),
    onSuccess: () => {
      setName('');
      setDescription('');
      setTagsInput('');
      qc.invalidateQueries({ queryKey: ['projects', id] });
    },
  });

  return (
    <div className="space-y-8">
      <header>
        <nav className="text-xs text-ink-200/60 mb-2">
          <Link to="/" className="hover:text-ink-100">
            Workspaces
          </Link>
          <span className="mx-1">/</span>
          <span>{ws.data?.name ?? '…'}</span>
        </nav>
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold">{ws.data?.name ?? 'Loading…'}</h1>
            {ws.data?.description && (
              <p className="text-sm text-ink-200/70 mt-1">{ws.data.description}</p>
            )}
          </div>
          {id && (
            <div className="flex items-center gap-2 text-sm shrink-0">
              <Link
                to={`/workspaces/${id}/pipelines`}
                className="btn-secondary"
              >
                Pipelines
              </Link>
              <Link
                to={`/workspaces/${id}/deployments`}
                className="btn-secondary"
              >
                Deployments
              </Link>
            </div>
          )}
        </div>
      </header>

      <div className="grid gap-8 md:grid-cols-[1fr_auto]">
        <section>
          <header className="mb-4 flex items-baseline justify-between">
            <h2 className="font-medium">Projects</h2>
            <span className="hint">{projects.data?.length ?? 0} total</span>
          </header>

          {projects.isLoading && <p className="hint">Loading…</p>}
          {projects.error && <p className="error">{errorMessage(projects.error)}</p>}

          {projects.data && projects.data.length === 0 && (
            <div className="card text-sm text-ink-200/70">
              No projects yet. Create one on the right →
            </div>
          )}

          <ul className="grid gap-3">
            {projects.data?.map((p) => (
              <li key={p.id}>
                <Link
                  to={`/workspaces/${id}/projects/${p.id}`}
                  className="card block hover:border-accent-500 transition-colors"
                >
                  <h3 className="font-medium text-ink-100">{p.name}</h3>
                  {p.description && (
                    <p className="mt-1 text-sm text-ink-200/70">{p.description}</p>
                  )}
                  {p.tags.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {p.tags.map((t) => (
                        <span key={t} className="kbd">
                          {t}
                        </span>
                      ))}
                    </div>
                  )}
                </Link>
              </li>
            ))}
          </ul>
        </section>

        <aside className="md:w-80 space-y-4">
          <div className="card">
            <h2 className="text-sm font-medium text-ink-100 mb-4">New project</h2>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                if (name.trim()) create.mutate();
              }}
              className="space-y-4"
            >
              <div>
                <label className="field" htmlFor="p-name">
                  Name
                </label>
                <input
                  id="p-name"
                  className="input"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
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
              <button
                type="submit"
                className="btn-primary w-full"
                disabled={create.isPending || !name.trim()}
              >
                {create.isPending ? 'Creating…' : 'Create'}
              </button>
            </form>
          </div>
          {id && <DataSourcesCard workspaceId={id} />}
        </aside>
      </div>
    </div>
  );
}
