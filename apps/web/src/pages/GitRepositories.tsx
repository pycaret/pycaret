/**
 * /workspaces/:wsId/git — Phase 5 Git repository management.
 *
 * Per-workspace list of linked Git repos with an inline "Publish"
 * button that fires ``POST /git-repositories/{id}/publish`` and shows
 * the resulting commit SHA inline.
 */

import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useParams } from 'react-router-dom';
import { BackButton } from '@/components/BackButton';
import { Dialog } from '@/components/Dialog';
import { gitApi, projectsApi, secretsApi } from '@/api/endpoints';
import type { GitProvider, GitRepository, PublishResult } from '@/api/types';

const PROVIDER_OPTIONS: { value: GitProvider; label: string }[] = [
  { value: 'github', label: 'GitHub' },
  { value: 'gitlab', label: 'GitLab' },
  { value: 'gitea', label: 'Gitea' },
  { value: 'bitbucket', label: 'Bitbucket' },
];

export function GitRepositories() {
  const { wsId = '' } = useParams<{ wsId: string }>();
  const qc = useQueryClient();
  const [createOpen, setCreateOpen] = useState(false);
  const [lastPublish, setLastPublish] = useState<PublishResult | null>(null);

  const list = useQuery({
    queryKey: ['git-repos', wsId],
    queryFn: () => gitApi.list(wsId),
    enabled: !!wsId,
  });

  const remove = useMutation({
    mutationFn: (id: string) => gitApi.delete(wsId, id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['git-repos', wsId] }),
  });

  const publish = useMutation({
    mutationFn: (id: string) => gitApi.publish(id),
    onSuccess: (data) => {
      setLastPublish(data);
      qc.invalidateQueries({ queryKey: ['git-repos', wsId] });
    },
  });

  const patch = useMutation({
    mutationFn: (vars: { id: string; body: Parameters<typeof gitApi.patch>[2] }) =>
      gitApi.patch(wsId, vars.id, vars.body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['git-repos', wsId] }),
  });

  return (
    <div className="space-y-6">
      <header className="flex items-baseline justify-between">
        <div>
          <BackButton />
          <h1 className="h-page mt-2">Git repositories</h1>
          <p className="muted small">
            Project ↔ Git repo sync. Phase 5. Publishes Experiments /
            Trials / Runs as YAML manifests; artifact URIs only (never
            raw model bytes).
          </p>
        </div>
        <button
          type="button"
          className="btn-primary"
          onClick={() => setCreateOpen(true)}
        >
          Link repository
        </button>
      </header>

      {lastPublish && (
        <div
          className={`card text-sm ${
            lastPublish.ok
              ? 'border-success-400'
              : 'border-danger-400'
          }`}
        >
          {lastPublish.ok ? (
            <p>
              Published to <code>{lastPublish.sha?.slice(0, 12)}…</code>
            </p>
          ) : (
            <p className="text-danger-600">
              Publish failed: {lastPublish.error}
            </p>
          )}
        </div>
      )}

      {list.isLoading && (
        <div className="card text-sm text-ink-500">Loading…</div>
      )}
      {list.data && list.data.length === 0 && (
        <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 p-10 text-center">
          <h3 className="text-base font-semibold">No repositories linked</h3>
          <p className="mt-1 text-sm text-ink-500">
            Add a GitHub / GitLab / Gitea / Bitbucket repo and assign
            it to a project to enable manifest sync.
          </p>
        </div>
      )}
      {list.data && list.data.length > 0 && (
        <ul className="space-y-2">
          {list.data.map((r) => (
            <RepoRow
              key={r.id}
              repo={r}
              onPublish={() => {
                setLastPublish(null);
                publish.mutate(r.id);
              }}
              onDelete={() => {
                if (confirm(`Unlink "${r.url}"?`)) {
                  remove.mutate(r.id);
                }
              }}
              onToggleAutoPublish={(value) =>
                patch.mutate({ id: r.id, body: { auto_publish: value } })
              }
              publishing={publish.isPending && publish.variables === r.id}
              patching={patch.isPending && patch.variables?.id === r.id}
            />
          ))}
        </ul>
      )}

      <NewRepoDialog
        open={createOpen}
        wsId={wsId}
        onClose={() => setCreateOpen(false)}
        onSaved={() => {
          setCreateOpen(false);
          qc.invalidateQueries({ queryKey: ['git-repos', wsId] });
        }}
      />
    </div>
  );
}

function RepoRow({
  repo,
  onPublish,
  onDelete,
  onToggleAutoPublish,
  publishing,
  patching,
}: {
  repo: GitRepository;
  onPublish: () => void;
  onDelete: () => void;
  onToggleAutoPublish: (value: boolean) => void;
  publishing: boolean;
  patching: boolean;
}) {
  return (
    <li className="card flex items-start gap-4">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-medium truncate">{repo.url}</span>
          <span className="pill-neutral">{repo.provider}</span>
          {!repo.enabled && <span className="pill-neutral">disabled</span>}
          {repo.auto_publish && (
            <span className="pill bg-accent-50 text-accent-700 dark:bg-accent-500/15 dark:text-accent-300">
              auto-publish
            </span>
          )}
          {repo.last_push_status === 'ok' && (
            <span className="pill-success">last ok</span>
          )}
          {repo.last_push_status === 'error' && (
            <span className="pill bg-rose-50 text-rose-700 dark:bg-rose-500/15 dark:text-rose-300">
              last error
            </span>
          )}
        </div>
        <p className="text-xs text-ink-500 mt-1">
          branch <code>{repo.default_branch}</code>
          {repo.path_prefix ? ` · prefix ${repo.path_prefix}` : ''}
          {repo.project_id ? ` · project ${repo.project_id.slice(0, 8)}…` : ''}
        </p>
        {repo.last_push_at && (
          <p className="text-xs text-ink-500 mt-1">
            last pushed {new Date(repo.last_push_at).toLocaleString()}
            {repo.last_push_sha ? ` (${repo.last_push_sha.slice(0, 12)}…)` : ''}
          </p>
        )}
        {repo.last_push_error && (
          <p className="text-xs text-danger-600 mt-1">{repo.last_push_error}</p>
        )}
        <label className="mt-2 inline-flex items-center gap-2 text-xs text-ink-600 dark:text-ink-400">
          <input
            type="checkbox"
            checked={repo.auto_publish}
            disabled={patching || !repo.project_id || !repo.enabled}
            onChange={(e) => onToggleAutoPublish(e.target.checked)}
          />
          Auto-publish on every Run completion
          {!repo.project_id && (
            <span className="text-ink-400">(needs a project link)</span>
          )}
        </label>
      </div>
      <div className="flex flex-col gap-1 shrink-0">
        <button
          type="button"
          className="btn-secondary text-xs"
          onClick={onPublish}
          disabled={publishing || !repo.project_id || !repo.enabled}
          title={
            !repo.project_id
              ? 'Bind the repo to a project before publishing'
              : !repo.enabled
                ? 'Repo is disabled'
                : 'Publish now'
          }
        >
          {publishing ? 'Publishing…' : 'Publish'}
        </button>
        <button
          type="button"
          className="text-xs text-danger-600 hover:underline"
          onClick={onDelete}
        >
          Unlink
        </button>
      </div>
    </li>
  );
}

function NewRepoDialog({
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
  const [provider, setProvider] = useState<GitProvider>('github');
  const [url, setUrl] = useState('');
  const [defaultBranch, setDefaultBranch] = useState('main');
  const [pathPrefix, setPathPrefix] = useState('');
  const [projectId, setProjectId] = useState('');
  const [secretId, setSecretId] = useState('');
  const [autoPublish, setAutoPublish] = useState(true);

  const projects = useQuery({
    queryKey: ['projects', wsId],
    queryFn: () => projectsApi.list(wsId),
    enabled: !!wsId && open,
  });
  const secrets = useQuery({
    queryKey: ['secrets', wsId],
    queryFn: () => secretsApi.list(wsId),
    enabled: !!wsId && open,
  });

  const save = useMutation({
    mutationFn: () =>
      gitApi.create(wsId, {
        provider,
        url: url.trim(),
        default_branch: defaultBranch.trim() || 'main',
        path_prefix: pathPrefix.trim() || undefined,
        project_id: projectId || undefined,
        secret_id: secretId || undefined,
        auto_publish: autoPublish && !!projectId,
      }),
    onSuccess: () => {
      setUrl('');
      setPathPrefix('');
      setProjectId('');
      setSecretId('');
      setAutoPublish(true);
      onSaved();
    },
  });

  return (
    <Dialog open={open} onClose={onClose} title="Link Git repository" size="md">
      <div className="space-y-3">
        <label className="field">
          <span className="muted small">Provider</span>
          <select
            className="input w-full"
            value={provider}
            onChange={(e) => setProvider(e.target.value as GitProvider)}
          >
            {PROVIDER_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span className="muted small">Clone URL</span>
          <input
            className="input w-full"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://github.com/owner/repo.git"
            autoFocus
          />
        </label>
        <div className="grid grid-cols-2 gap-2">
          <label className="field">
            <span className="muted small">Default branch</span>
            <input
              className="input w-full"
              value={defaultBranch}
              onChange={(e) => setDefaultBranch(e.target.value)}
            />
          </label>
          <label className="field">
            <span className="muted small">Path prefix (optional)</span>
            <input
              className="input w-full"
              value={pathPrefix}
              onChange={(e) => setPathPrefix(e.target.value)}
              placeholder="pycaret/"
            />
          </label>
        </div>
        <label className="field">
          <span className="muted small">Project</span>
          <select
            className="input w-full"
            value={projectId}
            onChange={(e) => setProjectId(e.target.value)}
          >
            <option value="">(none — publish via API only)</option>
            {(projects.data ?? []).map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span className="muted small">PAT secret</span>
          <select
            className="input w-full"
            value={secretId}
            onChange={(e) => setSecretId(e.target.value)}
          >
            <option value="">(public repo — no PAT)</option>
            {(secrets.data ?? [])
              .filter((s) => s.kind === 'git_pat' || s.kind === 'opaque')
              .map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
          </select>
        </label>
        <label className="inline-flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={autoPublish}
            disabled={!projectId}
            onChange={(e) => setAutoPublish(e.target.checked)}
          />
          Auto-publish on every Run completion
          {!projectId && (
            <span className="text-xs text-ink-400">
              (select a project above to enable)
            </span>
          )}
        </label>
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
            disabled={!url.trim() || save.isPending}
            onClick={() => save.mutate()}
          >
            {save.isPending ? 'Saving…' : 'Link'}
          </button>
        </div>
      </div>
    </Dialog>
  );
}
