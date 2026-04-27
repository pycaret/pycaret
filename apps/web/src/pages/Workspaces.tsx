import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { workspacesApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';

/** /  — list of workspaces the current user belongs to, plus a create form. */
export function Workspaces() {
  const qc = useQueryClient();
  const list = useQuery({ queryKey: ['workspaces'], queryFn: workspacesApi.list });

  const [name, setName] = useState('');
  const [description, setDescription] = useState('');

  const create = useMutation({
    mutationFn: () => workspacesApi.create({ name, description: description || undefined }),
    onSuccess: () => {
      setName('');
      setDescription('');
      qc.invalidateQueries({ queryKey: ['workspaces'] });
    },
  });

  return (
    <div className="grid gap-8 md:grid-cols-[1fr_auto]">
      <section>
        <header className="mb-4 flex items-baseline justify-between">
          <h1 className="text-xl font-semibold">Workspaces</h1>
          <span className="hint">{list.data?.length ?? 0} total</span>
        </header>

        {list.isLoading && <p className="hint">Loading…</p>}
        {list.error && <p className="error">{errorMessage(list.error)}</p>}

        {list.data && list.data.length === 0 && (
          <div className="card text-sm text-ink-500">
            No workspaces yet. Create your first one →
          </div>
        )}

        <ul className="grid gap-3">
          {list.data?.map((w) => (
            <li key={w.id}>
              <Link
                to={`/workspaces/${w.id}`}
                className="card block hover:border-accent-500 transition-colors"
              >
                <h3 className="font-medium text-ink-900">{w.name}</h3>
                {w.description && (
                  <p className="mt-1 text-sm text-ink-500">{w.description}</p>
                )}
                <p className="mt-2 text-xs text-ink-400">
                  Created {new Date(w.created_at).toLocaleDateString()}
                </p>
              </Link>
            </li>
          ))}
        </ul>
      </section>

      <aside className="md:w-80">
        <div className="card">
          <h2 className="text-sm font-medium text-ink-900 mb-4">New workspace</h2>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (name.trim()) create.mutate();
            }}
            className="space-y-4"
          >
            <div>
              <label className="field" htmlFor="ws-name">
                Name
              </label>
              <input
                id="ws-name"
                className="input"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
              />
            </div>
            <div>
              <label className="field" htmlFor="ws-desc">
                Description (optional)
              </label>
              <textarea
                id="ws-desc"
                className="input resize-none"
                rows={3}
                value={description}
                onChange={(e) => setDescription(e.target.value)}
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
      </aside>
    </div>
  );
}
