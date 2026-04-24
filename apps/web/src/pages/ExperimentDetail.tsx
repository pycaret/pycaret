/**
 * /workspaces/:wsId/projects/:projectId/experiments/:experimentId
 *
 * Two-column layout:
 *  - Main: experiment header, config summary, runs list (status + timing + link).
 *  - Sidebar: "New run" form (plan + model + data source).
 *
 * The run form is deliberately minimal in session 14 (only sklearn_dataset
 * built-ins). Session 15 expands it with data-source selection + proper
 * live event streaming on the run-detail screen.
 */

import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import { experimentsApi, projectsApi, workspacesApi } from '@/api/endpoints';
import { errorMessage, api } from '@/api/client';
import type { Run } from '@/api/types';

type RunPlan = 'setup' | 'create' | 'compare';

function formatDuration(ms: number | null): string {
  if (!ms) return '—';
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${(s - m * 60).toFixed(0)}s`;
}

const STATUS_COLOR: Record<string, string> = {
  queued: 'text-ink-200/60',
  running: 'text-accent-400',
  succeeded: 'text-success-500',
  failed: 'text-danger-500',
  cancelled: 'text-warn-500',
};

export function ExperimentDetail() {
  const {
    wsId = '',
    projectId = '',
    experimentId = '',
  } = useParams<{ wsId: string; projectId: string; experimentId: string }>();
  const qc = useQueryClient();

  const ws = useQuery({
    queryKey: ['workspaces', wsId],
    queryFn: () => workspacesApi.get(wsId),
    enabled: !!wsId,
  });
  const project = useQuery({
    queryKey: ['projects', wsId, projectId],
    queryFn: () => projectsApi.get(wsId, projectId),
    enabled: !!wsId && !!projectId,
  });
  const experiment = useQuery({
    queryKey: ['experiments', projectId, experimentId],
    queryFn: () => experimentsApi.get(projectId, experimentId),
    enabled: !!projectId && !!experimentId,
  });

  // Runs list. Polls while anything is queued/running so the table stays fresh.
  const runs = useQuery({
    queryKey: ['runs', 'for-experiment', experimentId],
    queryFn: () =>
      api.get<Run[]>(`/experiments/${experimentId}/runs`).then((r) => r.data),
    enabled: !!experimentId,
    refetchInterval: (q) => {
      const data = q.state.data;
      if (!data) return false;
      const pending = data.some((r) => r.status === 'queued' || r.status === 'running');
      return pending ? 2000 : false;
    },
  });

  // New run submit form (basic — session 15 replaces with full RunConfig UI).
  const [plan, setPlan] = useState<RunPlan>('compare');
  const [modelId, setModelId] = useState('lr');
  const [dataset, setDataset] = useState('iris');

  const submitRun = useMutation({
    mutationFn: () =>
      api
        .post<Run>(`/experiments/${experimentId}/runs`, {
          plan,
          model_id: plan === 'create' ? modelId : null,
          sklearn_dataset: dataset,
        })
        .then((r) => r.data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['runs', 'for-experiment', experimentId] });
    },
  });

  const setupEntries = Object.entries(experiment.data?.setup_params ?? {});

  return (
    <div className="space-y-8">
      <header>
        <nav className="text-xs text-ink-200/60 mb-2">
          <Link to="/" className="hover:text-ink-100">
            Workspaces
          </Link>
          <span className="mx-1">/</span>
          <Link to={`/workspaces/${wsId}`} className="hover:text-ink-100">
            {ws.data?.name ?? '…'}
          </Link>
          <span className="mx-1">/</span>
          <Link
            to={`/workspaces/${wsId}/projects/${projectId}`}
            className="hover:text-ink-100"
          >
            {project.data?.name ?? '…'}
          </Link>
          <span className="mx-1">/</span>
          <span>{experiment.data?.name ?? '…'}</span>
        </nav>
        <h1 className="text-xl font-semibold">
          {experiment.data?.name ?? 'Loading…'}
        </h1>
        {experiment.data && (
          <p className="mt-1 text-sm text-ink-200/70">
            <span className="kbd">{experiment.data.task}</span>
            {experiment.data.target && (
              <>
                {' '}
                target: <span className="font-mono">{experiment.data.target}</span>
              </>
            )}
          </p>
        )}
      </header>

      <div className="grid gap-8 md:grid-cols-[1fr_20rem]">
        {/* ────────── Main */}
        <div className="space-y-8">
          {/* Config overview */}
          <section>
            <h2 className="text-sm font-medium text-ink-100 mb-3">Setup parameters</h2>
            {setupEntries.length === 0 ? (
              <p className="hint">
                Using engine defaults for every parameter (none overridden).
              </p>
            ) : (
              <dl className="card grid gap-2 md:grid-cols-2">
                {setupEntries.map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-4">
                    <dt className="text-sm text-ink-200/70 font-mono">{k}</dt>
                    <dd className="text-sm text-ink-100 font-mono">
                      {typeof v === 'object' ? JSON.stringify(v) : String(v)}
                    </dd>
                  </div>
                ))}
              </dl>
            )}
          </section>

          {/* Runs list */}
          <section>
            <header className="mb-4 flex items-baseline justify-between">
              <h2 className="font-medium">Runs</h2>
              <span className="hint">{runs.data?.length ?? 0} total</span>
            </header>

            {runs.isLoading && <p className="hint">Loading…</p>}
            {runs.error && <p className="error">{errorMessage(runs.error)}</p>}

            {runs.data && runs.data.length === 0 && (
              <div className="card text-sm text-ink-200/70">
                No runs yet. Submit one from the right panel →
              </div>
            )}

            {runs.data && runs.data.length > 0 && (
              <div className="card overflow-hidden p-0">
                <table className="w-full text-sm">
                  <thead className="bg-ink-800 text-ink-200/70">
                    <tr>
                      <th className="px-4 py-2 text-left font-medium">Status</th>
                      <th className="px-4 py-2 text-left font-medium">Plan</th>
                      <th className="px-4 py-2 text-left font-medium">Dataset</th>
                      <th className="px-4 py-2 text-left font-medium">Duration</th>
                      <th className="px-4 py-2 text-left font-medium">Created</th>
                    </tr>
                  </thead>
                  <tbody>
                    {runs.data.map((r) => {
                      const snapshot = r.snapshot ?? {};
                      const planText =
                        (snapshot as { plan?: string }).plan ?? '—';
                      const dsText =
                        (snapshot as { sklearn_dataset?: string }).sklearn_dataset ??
                        (snapshot as { data_source_id?: string }).data_source_id ??
                        '—';
                      return (
                        <tr
                          key={r.id}
                          className="border-t border-ink-800 hover:bg-ink-800/50"
                        >
                          <td className="px-4 py-2">
                            <span className={STATUS_COLOR[r.status] ?? ''}>
                              {r.status}
                            </span>
                          </td>
                          <td className="px-4 py-2 font-mono text-xs">{planText}</td>
                          <td className="px-4 py-2 font-mono text-xs">{dsText}</td>
                          <td className="px-4 py-2 font-mono text-xs">
                            {formatDuration(r.duration_ms)}
                          </td>
                          <td className="px-4 py-2 text-xs text-ink-200/60">
                            {new Date(r.created_at).toLocaleString()}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </div>

        {/* ────────── Sidebar: new run */}
        <aside>
          <div className="card space-y-4">
            <h2 className="text-sm font-medium text-ink-100">New run</h2>

            <div>
              <label className="field" htmlFor="plan">
                Plan
              </label>
              <select
                id="plan"
                className="input"
                value={plan}
                onChange={(e) => setPlan(e.target.value as RunPlan)}
              >
                <option value="setup">setup — preprocess only</option>
                <option value="create">create — train one model</option>
                <option value="compare">compare — leaderboard</option>
              </select>
            </div>

            {plan === 'create' && (
              <div>
                <label className="field" htmlFor="model">
                  Model id
                </label>
                <input
                  id="model"
                  className="input"
                  value={modelId}
                  onChange={(e) => setModelId(e.target.value)}
                  placeholder="lr"
                />
                <p className="hint mt-1">
                  PyCaret model id (e.g. <code className="font-mono">lr</code>,{' '}
                  <code className="font-mono">rf</code>).
                </p>
              </div>
            )}

            <div>
              <label className="field" htmlFor="dataset">
                Sample dataset
              </label>
              <select
                id="dataset"
                className="input"
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
              >
                <option value="iris">iris (classification)</option>
                <option value="wine">wine (classification)</option>
                <option value="breast_cancer">breast_cancer (classification)</option>
                <option value="diabetes">diabetes (regression)</option>
              </select>
              <p className="hint mt-1">
                Built-in sklearn datasets for quick experiments. Custom CSV uploads
                come next session.
              </p>
            </div>

            {submitRun.error && (
              <p className="error">{errorMessage(submitRun.error)}</p>
            )}

            <button
              className="btn-primary w-full"
              disabled={submitRun.isPending}
              onClick={() => submitRun.mutate()}
            >
              {submitRun.isPending ? 'Submitting…' : 'Submit run'}
            </button>
            <p className="hint">Runs execute in the background. The list refreshes automatically.</p>
          </div>
        </aside>
      </div>
    </div>
  );
}
