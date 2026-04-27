/**
 * /workspaces/:wsId/pipelines/:pipelineId
 *
 * Pipeline metadata + the "deploy" action. Each pipeline can back multiple
 * Deployment rows (think dev/staging/prod slugs pointing at the same
 * immutable artifact). This screen lists the existing deployments and
 * provides a small form to create a new one.
 */

import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import {
  deploymentsApi,
  pipelinesApi,
  workspacesApi,
} from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import { DeploymentReviewModal } from '@/components/DeploymentReviewModal';
import type { Deployment } from '@/api/types';

const SLUG_RE = /^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$/;

const STATUS_TONE: Record<string, string> = {
  active: 'text-success-500',
  paused: 'text-warn-500',
  archived: 'text-ink-500',
};

export function PipelineDetail() {
  const { wsId = '', pipelineId = '' } = useParams<{
    wsId: string;
    pipelineId: string;
  }>();
  const qc = useQueryClient();

  const ws = useQuery({
    queryKey: ['workspaces', wsId],
    queryFn: () => workspacesApi.get(wsId),
    enabled: !!wsId,
  });
  const pipeline = useQuery({
    queryKey: ['pipelines', pipelineId],
    queryFn: () => pipelinesApi.get(pipelineId),
    enabled: !!pipelineId,
  });
  const deployments = useQuery({
    queryKey: ['deployments', 'by-workspace', wsId],
    queryFn: () => deploymentsApi.list(wsId),
    enabled: !!wsId,
  });
  const pipelineDeployments: Deployment[] = (deployments.data ?? []).filter(
    (d) => d.pipeline_id === pipelineId,
  );

  // ────────── deploy form
  const [slug, setSlug] = useState('');
  const [reviewing, setReviewing] = useState(false);
  const [authMode, setAuthMode] = useState<'workspace' | 'api-key' | 'public'>(
    'workspace',
  );

  const slugValid = SLUG_RE.test(slug);

  const deploy = useMutation({
    mutationFn: () =>
      deploymentsApi.create(pipelineId, {
        endpoint_slug: slug,
        auth_mode: authMode,
      }),
    onSuccess: () => {
      setSlug('');
      qc.invalidateQueries({ queryKey: ['deployments'] });
    },
  });

  return (
    <div className="space-y-8">
      <header>
        <nav className="text-xs text-ink-500 mb-2">
          <Link to="/" className="hover:text-ink-900">
            Workspaces
          </Link>
          <span className="mx-1">/</span>
          <Link to={`/workspaces/${wsId}`} className="hover:text-ink-900">
            {ws.data?.name ?? '…'}
          </Link>
          <span className="mx-1">/</span>
          <Link
            to={`/workspaces/${wsId}/pipelines`}
            className="hover:text-ink-900"
          >
            Pipelines
          </Link>
          <span className="mx-1">/</span>
          <span>{pipeline.data?.name ?? '…'}</span>
        </nav>
        <h1 className="text-xl font-semibold">
          {pipeline.data?.name ?? 'Loading…'}
        </h1>
        {pipeline.data?.description && (
          <p className="text-sm text-ink-500 mt-1">
            {pipeline.data.description}
          </p>
        )}
      </header>

      <div className="grid gap-8 md:grid-cols-[1fr_22rem]">
        <div className="space-y-8">
          {/* ────────── metadata */}
          <section>
            <h2 className="text-sm font-medium text-ink-900 mb-3">Metadata</h2>
            {pipeline.error && <p className="error">{errorMessage(pipeline.error)}</p>}
            {pipeline.data && (
              <dl className="card grid gap-2 md:grid-cols-2">
                <Row k="model_id" v={pipeline.data.model_id ?? '—'} mono />
                <Row k="sha256" v={pipeline.data.sha256 ?? '—'} mono title />
                <Row k="origin_run_id" v={pipeline.data.origin_run_id ?? '—'} mono />
                <Row k="stored_path" v={pipeline.data.stored_path} mono />
                <Row k="created" v={new Date(pipeline.data.created_at).toLocaleString()} />
                <Row k="tags" v={pipeline.data.tags.join(', ') || '—'} />
              </dl>
            )}
          </section>

          {/* ────────── existing deployments */}
          <section>
            <header className="mb-3 flex items-baseline justify-between">
              <h2 className="text-sm font-medium text-ink-900">Deployments</h2>
              <span className="hint">{pipelineDeployments.length} total</span>
            </header>
            {deployments.isLoading && <p className="hint">Loading…</p>}
            {pipelineDeployments.length === 0 && !deployments.isLoading && (
              <div className="card text-sm text-ink-500">
                This pipeline hasn't been deployed yet. Use the panel on the right →
              </div>
            )}
            {pipelineDeployments.length > 0 && (
              <div className="card overflow-hidden p-0">
                <table className="w-full text-sm">
                  <thead className="bg-white text-ink-500">
                    <tr>
                      <th className="px-4 py-2 text-left font-medium">Slug</th>
                      <th className="px-4 py-2 text-left font-medium">Status</th>
                      <th className="px-4 py-2 text-left font-medium">Auth</th>
                      <th className="px-4 py-2 text-right font-medium">Predictions</th>
                      <th className="px-4 py-2 text-right font-medium">p50</th>
                      <th className="px-4 py-2 text-right font-medium">p95</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pipelineDeployments.map((d) => (
                      <tr key={d.id} className="border-t border-ink-200 hover:bg-ink-50">
                        <td className="px-4 py-2">
                          <Link
                            to={`/deployments/${d.id}`}
                            className="text-accent-400 hover:underline font-mono"
                          >
                            {d.endpoint_slug}
                          </Link>
                        </td>
                        <td className={`px-4 py-2 ${STATUS_TONE[d.status] ?? ''}`}>
                          {d.status}
                        </td>
                        <td className="px-4 py-2 font-mono text-xs">{d.auth_mode}</td>
                        <td className="px-4 py-2 text-right font-mono text-xs tabular-nums">
                          {d.inference_count}
                        </td>
                        <td className="px-4 py-2 text-right font-mono text-xs tabular-nums">
                          {d.p50_latency_ms?.toFixed(1) ?? '—'}
                        </td>
                        <td className="px-4 py-2 text-right font-mono text-xs tabular-nums">
                          {d.p95_latency_ms?.toFixed(1) ?? '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </div>

        {/* ────────── deploy form */}
        <aside>
          <div className="card space-y-4">
            <h2 className="text-sm font-medium text-ink-900">New deployment</h2>

            <div>
              <label className="field" htmlFor="slug">
                Endpoint slug <span className="text-danger-500">*</span>
              </label>
              <input
                id="slug"
                className="input"
                value={slug}
                onChange={(e) => setSlug(e.target.value.toLowerCase())}
                placeholder="iris-v1"
                required
              />
              <p className="hint mt-1">
                Lowercase letters, digits, dashes. 3–64 chars. Must be globally unique.
              </p>
              {slug && !slugValid && (
                <p className="error mt-1">
                  Invalid — must match{' '}
                  <code className="font-mono">[a-z0-9][a-z0-9-]{'{1,62}'}[a-z0-9]</code>.
                </p>
              )}
            </div>

            <div>
              <label className="field" htmlFor="auth">
                Auth mode
              </label>
              <select
                id="auth"
                className="input"
                value={authMode}
                onChange={(e) =>
                  setAuthMode(e.target.value as 'workspace' | 'api-key' | 'public')
                }
              >
                <option value="workspace">workspace — JWT of a workspace member</option>
                <option value="api-key" disabled>
                  api-key — (V2)
                </option>
                <option value="public" disabled>
                  public — (V2)
                </option>
              </select>
            </div>

            {deploy.error && <p className="error">{errorMessage(deploy.error)}</p>}

            <div className="flex gap-2">
              <button
                type="button"
                className="btn-secondary"
                onClick={() => setReviewing(true)}
                title="Run a pre-deploy risk review via the LLM"
              >
                ✨ Review
              </button>
              <button
                className="btn-primary flex-1"
                disabled={!slugValid || deploy.isPending}
                onClick={() => deploy.mutate()}
              >
                {deploy.isPending ? 'Deploying…' : 'Deploy'}
              </button>
            </div>
          </div>
        </aside>
      </div>

      {pipeline.data && (
        <DeploymentReviewModal
          pipelineId={pipelineId}
          pipelineName={pipeline.data.name}
          open={reviewing}
          onClose={() => setReviewing(false)}
        />
      )}
    </div>
  );
}

function Row({
  k,
  v,
  mono,
  title,
}: {
  k: string;
  v: string;
  mono?: boolean;
  title?: boolean;
}) {
  return (
    <div className="flex justify-between gap-4">
      <dt className="text-sm text-ink-500 font-mono shrink-0">{k}</dt>
      <dd
        className={`text-sm text-ink-900 text-right break-all ${mono ? 'font-mono' : ''}`}
        title={title ? v : undefined}
      >
        {v}
      </dd>
    </div>
  );
}
