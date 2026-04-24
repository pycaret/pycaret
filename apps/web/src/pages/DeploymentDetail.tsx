/**
 * /deployments/:deploymentId — single deployment view.
 *
 * Layout:
 *  - Header: slug + status + auth mode + short metadata.
 *  - Left: live metrics (inference count, errors, p50/p95, last hit) + the
 *    PredictTester for hitting the endpoint interactively.
 *  - Right: sticky actions (delete for v1; pause / rollback later).
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { deploymentsApi, pipelinesApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import { PredictTester } from '@/components/PredictTester';

const STATUS_TONE: Record<string, string> = {
  active: 'text-success-500',
  paused: 'text-warn-500',
  archived: 'text-ink-200/60',
};

export function DeploymentDetail() {
  const { deploymentId = '' } = useParams<{ deploymentId: string }>();
  const nav = useNavigate();
  const qc = useQueryClient();

  const deployment = useQuery({
    queryKey: ['deployments', deploymentId],
    queryFn: () => deploymentsApi.get(deploymentId),
    enabled: !!deploymentId,
    refetchInterval: 3000, // keep metrics counters live
  });
  const pipeline = useQuery({
    queryKey: ['pipelines', deployment.data?.pipeline_id],
    queryFn: () => pipelinesApi.get(deployment.data!.pipeline_id),
    enabled: !!deployment.data?.pipeline_id,
  });

  const remove = useMutation({
    mutationFn: () => deploymentsApi.remove(deploymentId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['deployments'] });
      if (deployment.data?.workspace_id) {
        nav(`/workspaces/${deployment.data.workspace_id}/deployments`, {
          replace: true,
        });
      } else {
        nav('/', { replace: true });
      }
    },
  });

  const d = deployment.data;

  return (
    <div className="space-y-8">
      <header>
        <nav className="text-xs text-ink-200/60 mb-2">
          <Link to="/" className="hover:text-ink-100">
            Workspaces
          </Link>
          <span className="mx-1">/</span>
          {d ? (
            <>
              <Link
                to={`/workspaces/${d.workspace_id}/deployments`}
                className="hover:text-ink-100"
              >
                Deployments
              </Link>
              <span className="mx-1">/</span>
            </>
          ) : null}
          <span className="font-mono">{d?.endpoint_slug ?? '…'}</span>
        </nav>
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold font-mono">
              {d?.endpoint_slug ?? 'Loading…'}
            </h1>
            {d && (
              <p className="mt-1 text-sm text-ink-200/70">
                <span className={STATUS_TONE[d.status] ?? ''}>{d.status}</span>
                {' · '}
                <span className="font-mono">{d.auth_mode}</span>
                {pipeline.data && (
                  <>
                    {' · pipeline '}
                    <Link
                      to={`/workspaces/${d.workspace_id}/pipelines/${d.pipeline_id}`}
                      className="text-accent-400 hover:underline"
                    >
                      {pipeline.data.name}
                    </Link>
                  </>
                )}
              </p>
            )}
          </div>
          <button
            className="btn-danger shrink-0"
            onClick={() => {
              if (
                window.confirm(
                  `Delete deployment "${d?.endpoint_slug}"? The underlying pipeline is not deleted.`,
                )
              ) {
                remove.mutate();
              }
            }}
            disabled={!d || remove.isPending}
          >
            {remove.isPending ? 'Deleting…' : 'Delete'}
          </button>
        </div>
        {deployment.error && (
          <p className="error mt-2">{errorMessage(deployment.error)}</p>
        )}
        {remove.error && <p className="error mt-2">{errorMessage(remove.error)}</p>}
      </header>

      {d && (
        <div className="grid gap-8 md:grid-cols-[1fr_22rem]">
          <div className="space-y-8">
            {/* ────────── metrics */}
            <section>
              <h2 className="text-sm font-medium text-ink-100 mb-3">Metrics</h2>
              <dl className="grid gap-3 grid-cols-2 md:grid-cols-4">
                <Stat k="predictions" v={d.inference_count.toString()} />
                <Stat
                  k="errors"
                  v={d.error_count.toString()}
                  tone={d.error_count > 0 ? 'danger' : undefined}
                />
                <Stat
                  k="p50 latency"
                  v={d.p50_latency_ms != null ? `${d.p50_latency_ms.toFixed(1)}ms` : '—'}
                />
                <Stat
                  k="p95 latency"
                  v={d.p95_latency_ms != null ? `${d.p95_latency_ms.toFixed(1)}ms` : '—'}
                />
              </dl>
              <p className="hint mt-3">
                Metrics are rolling over the last 100 predictions.{' '}
                {d.last_inference_at
                  ? `Last hit ${new Date(d.last_inference_at).toLocaleString()}.`
                  : 'No predictions yet.'}
              </p>
            </section>

            {/* ────────── test form */}
            <PredictTester endpointSlug={d.endpoint_slug} />
          </div>

          {/* ────────── right column — metadata */}
          <aside className="space-y-4">
            <div className="card space-y-2 text-sm">
              <h3 className="text-sm font-medium text-ink-100">Metadata</h3>
              <MetaRow k="deployment_id" v={d.id} />
              <MetaRow k="workspace_id" v={d.workspace_id} />
              <MetaRow k="pipeline_id" v={d.pipeline_id} />
              <MetaRow k="created" v={new Date(d.created_at).toLocaleString()} />
            </div>
          </aside>
        </div>
      )}
    </div>
  );
}

function Stat({
  k,
  v,
  tone,
}: {
  k: string;
  v: string;
  tone?: 'danger';
}) {
  return (
    <div className="card">
      <p className="text-xs uppercase tracking-wider text-ink-200/60">{k}</p>
      <p
        className={`mt-1 font-mono text-2xl tabular-nums ${
          tone === 'danger' ? 'text-danger-500' : 'text-ink-100'
        }`}
      >
        {v}
      </p>
    </div>
  );
}

function MetaRow({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-3">
      <span className="text-xs text-ink-200/60 font-mono shrink-0">{k}</span>
      <span className="text-xs text-ink-100 font-mono text-right break-all" title={v}>
        {v}
      </span>
    </div>
  );
}
