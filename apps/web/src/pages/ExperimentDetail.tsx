/**
 * /workspaces/:wsId/projects/:projectId/experiments/:experimentId
 *
 * Two-column layout:
 *  - Main: experiment header, config summary, runs list (linked to /runs/:id).
 *  - Sidebar: "New run" form — plan + (model picker when plan=create) +
 *    data-source picker (workspace CSVs first, sklearn samples as fallback).
 *
 * The sidebar form's model picker is driven by `describeApi.models(task)` so
 * it's always task-appropriate. The data-source picker pulls registered
 * CSV uploads for the workspace; if there are none, it falls back to the
 * built-in sklearn sample datasets (useful for a fresh install demo).
 */

import { useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import {
  dataSourcesApi,
  describeApi,
  experimentsApi,
  projectsApi,
  runsApi,
  workspacesApi,
} from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import type { RunCreate, RunPlan } from '@/api/types';

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

/** Sklearn sample datasets kept alive for a zero-data-source demo. */
const SKLEARN_SAMPLES = [
  { value: 'sklearn:iris', label: 'iris (classification)' },
  { value: 'sklearn:wine', label: 'wine (classification)' },
  { value: 'sklearn:breast_cancer', label: 'breast_cancer (classification)' },
  { value: 'sklearn:diabetes', label: 'diabetes (regression)' },
];

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
    queryFn: () => runsApi.listForExperiment(experimentId),
    enabled: !!experimentId,
    refetchInterval: (q) => {
      const data = q.state.data;
      if (!data) return false;
      const pending = data.some((r) => r.status === 'queued' || r.status === 'running');
      return pending ? 2000 : false;
    },
  });

  // Workspace data sources (CSV uploads).
  const dataSources = useQuery({
    queryKey: ['data-sources', wsId],
    queryFn: () => dataSourcesApi.list(wsId),
    enabled: !!wsId,
  });

  // Models available for this task (drives the create-plan model picker).
  const models = useQuery({
    queryKey: ['describe', 'models', experiment.data?.task],
    queryFn: () => describeApi.models(experiment.data!.task),
    enabled: !!experiment.data?.task,
    staleTime: 10 * 60 * 1000,
  });

  // ────────── new-run form state
  const [plan, setPlan] = useState<RunPlan>('compare');
  const [modelId, setModelId] = useState('lr');
  // The source picker uses a single combo-value string: either a data-source
  // UUID or `sklearn:<name>` for the built-in samples. This keeps one `<select>`
  // driving two different backend fields without juggling extra state.
  const [source, setSource] = useState<string>('sklearn:iris');

  // Default to the first real CSV source when the list arrives.
  const sources = useMemo(() => {
    const csvs = (dataSources.data ?? []).filter((d) => d.kind === 'csv_upload');
    return [
      ...csvs.map((d) => ({ value: d.id, label: `${d.name} (CSV)` })),
      ...SKLEARN_SAMPLES,
    ];
  }, [dataSources.data]);

  const submitRun = useMutation({
    mutationFn: () => {
      const body: RunCreate = {
        plan,
        model_id: plan === 'create' ? modelId : null,
      };
      if (source.startsWith('sklearn:')) {
        body.sklearn_dataset = source.slice('sklearn:'.length);
      } else {
        body.data_source_id = source;
      }
      return runsApi.submit(experimentId, body);
    },
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
                          className="border-t border-ink-800 hover:bg-ink-800/50 cursor-pointer"
                          onClick={() => {
                            window.location.href = `/runs/${r.id}`;
                          }}
                        >
                          <td className="px-4 py-2">
                            <Link
                              to={`/runs/${r.id}`}
                              className={STATUS_COLOR[r.status] ?? ''}
                              onClick={(e) => e.stopPropagation()}
                            >
                              {r.status}
                            </Link>
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
                  Model
                </label>
                <select
                  id="model"
                  className="input"
                  value={modelId}
                  onChange={(e) => setModelId(e.target.value)}
                  disabled={models.isLoading || !models.data}
                >
                  {(models.data ?? []).map((m) => (
                    <option key={m.id} value={m.id} disabled={!m.is_available}>
                      {m.id} — {m.name}
                      {!m.is_available ? ' (install required)' : ''}
                    </option>
                  ))}
                </select>
                {models.error && <p className="error mt-1">{errorMessage(models.error)}</p>}
              </div>
            )}

            <div>
              <label className="field" htmlFor="source">
                Data source
              </label>
              <select
                id="source"
                className="input"
                value={source}
                onChange={(e) => setSource(e.target.value)}
              >
                {sources.map((s) => (
                  <option key={s.value} value={s.value}>
                    {s.label}
                  </option>
                ))}
              </select>
              <p className="hint mt-1">
                CSV uploads appear at the top of this list. Sklearn samples are
                handy for demos.
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
            <p className="hint">
              Runs execute in the background. Click any row in the table to watch
              the event stream live.
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}
