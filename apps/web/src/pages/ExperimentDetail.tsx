/**
 * /workspaces/:wsId/projects/:projectId/experiments/:experimentId
 *
 * Two-column layout:
 *  - Main: experiment header, setup summary, runs table.
 *  - Sidebar: "New run" form. Plan + (model picker for plan=create) +
 *    data-source picker.
 *
 * Data-source defaulting (the user-facing fix):
 *   1. If the experiment has previous runs, use whatever the most recent
 *      one ran against (CSV upload or sklearn sample).
 *   2. Otherwise, default to the first CSV in this workspace.
 *   3. Sklearn samples only show in the dropdown when no CSVs are
 *      registered (so a one-click demo still works), or as fallback when
 *      a previous run used one.
 */

import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useNavigate, useParams } from 'react-router-dom';
import {
  dataSourcesApi,
  describeApi,
  experimentsApi,
  projectsApi,
  runsApi,
  workspacesApi,
} from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import { Dialog } from '@/components/Dialog';
import type { Run, RunCreate, RunPlan } from '@/api/types';

function formatDuration(ms: number | null): string {
  if (!ms) return '—';
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${(s - m * 60).toFixed(0)}s`;
}

const STATUS_PILL: Record<string, string> = {
  queued: 'pill-neutral',
  running: 'pill-accent',
  succeeded: 'pill-success',
  failed: 'pill-danger',
  cancelled: 'pill-warn',
};

/** Sklearn sample datasets — surfaced only when no CSVs are registered. */
const SKLEARN_SAMPLES = [
  { value: 'sklearn:iris', label: 'iris (classification)' },
  { value: 'sklearn:wine', label: 'wine (classification)' },
  { value: 'sklearn:breast_cancer', label: 'breast_cancer (classification)' },
  { value: 'sklearn:diabetes', label: 'diabetes (regression)' },
];

/** Pick a sensible default source for the next run from prior runs + CSVs. */
function pickDefaultSource(
  runs: Run[] | undefined,
  csvs: { id: string; name: string }[],
): string {
  if (runs && runs.length > 0) {
    // Prefer the most recent run's data source if it's still around.
    for (const r of runs) {
      const snap = (r.snapshot ?? {}) as {
        data_source_id?: string;
        sklearn_dataset?: string;
      };
      if (snap.data_source_id && csvs.find((d) => d.id === snap.data_source_id)) {
        return snap.data_source_id;
      }
      if (snap.sklearn_dataset) {
        return `sklearn:${snap.sklearn_dataset}`;
      }
    }
  }
  if (csvs.length > 0) return csvs[0].id;
  return SKLEARN_SAMPLES[0].value;
}

export function ExperimentDetail() {
  const {
    wsId = '',
    projectId = '',
    experimentId = '',
  } = useParams<{ wsId: string; projectId: string; experimentId: string }>();
  const qc = useQueryClient();
  const nav = useNavigate();
  const [runDialogOpen, setRunDialogOpen] = useState(false);

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

  const runs = useQuery({
    queryKey: ['runs', 'for-experiment', experimentId],
    queryFn: () => runsApi.listForExperiment(experimentId),
    enabled: !!experimentId,
    refetchInterval: (q) => {
      const data = q.state.data;
      if (!data) return false;
      const pending = data.some(
        (r) => r.status === 'queued' || r.status === 'running',
      );
      return pending ? 2000 : false;
    },
  });

  const dataSources = useQuery({
    queryKey: ['data-sources', wsId],
    queryFn: () => dataSourcesApi.list(wsId),
    enabled: !!wsId,
  });

  const models = useQuery({
    queryKey: ['describe', 'models', experiment.data?.task],
    queryFn: () => describeApi.models(experiment.data!.task),
    enabled: !!experiment.data?.task,
    staleTime: 10 * 60 * 1000,
  });

  // ────────── new-run form state
  const [plan, setPlan] = useState<RunPlan>('compare');
  const [modelId, setModelId] = useState('lr');
  const [source, setSource] = useState<string>('');

  const csvs = useMemo(
    () => (dataSources.data ?? []).filter((d) => d.kind === 'csv_upload'),
    [dataSources.data],
  );

  // Resolve sensible default source whenever data lands.
  useEffect(() => {
    if (source) return; // user already chose
    if (dataSources.isLoading || runs.isLoading) return;
    setSource(
      pickDefaultSource(
        runs.data,
        csvs.map((d) => ({ id: d.id, name: d.name })),
      ),
    );
  }, [source, dataSources.isLoading, runs.isLoading, runs.data, csvs]);

  // The dropdown lists CSV uploads first. Sklearn samples appear only when
  // there are zero CSVs (true demo case) OR when a prior run used one (so
  // the user can re-pick the same demo source they ran before).
  const sourceOptions = useMemo(() => {
    const csvOpts = csvs.map((d) => ({
      value: d.id,
      label: `${d.name}`,
      kind: 'csv' as const,
    }));
    const usedSklearn = new Set<string>();
    for (const r of runs.data ?? []) {
      const v = (r.snapshot as { sklearn_dataset?: string } | null)
        ?.sklearn_dataset;
      if (v) usedSklearn.add(v);
    }
    const showSklearn = csvOpts.length === 0 || usedSklearn.size > 0;
    const sklearnOpts = showSklearn
      ? SKLEARN_SAMPLES.map((s) => ({
          value: s.value,
          label: s.label,
          kind: 'sklearn' as const,
        }))
      : [];
    return { csvOpts, sklearnOpts };
  }, [csvs, runs.data]);

  // Map data_source_id → friendly name for the runs table.
  const sourceLabelMap = useMemo(() => {
    const m: Record<string, string> = {};
    for (const d of csvs) m[d.id] = d.name;
    return m;
  }, [csvs]);

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
      if (plan === 'search') {
        body.plan_params = {
          variants: [
            {},
            { normalize: true },
            { normalize: true, transformation: true },
          ],
        };
      }
      return runsApi.submit(experimentId, body);
    },
    onSuccess: (run) => {
      qc.invalidateQueries({
        queryKey: ['runs', 'for-experiment', experimentId],
      });
      setRunDialogOpen(false);
      nav(`/runs/${run.id}`);
    },
  });

  const setupEntries = Object.entries(experiment.data?.setup_params ?? {});
  const noCsvs = !dataSources.isLoading && csvs.length === 0;

  return (
    <div className="space-y-8">
      {/* ─── Header ──────────────────────────────────────────── */}
      <header className="space-y-2">
        <nav className="text-xs text-ink-500">
          <Link to="/" className="hover:text-ink-900 dark:hover:text-ink-50">
            Workspaces
          </Link>
          <span className="mx-1.5 text-ink-300">/</span>
          <Link
            to={`/workspaces/${wsId}`}
            className="hover:text-ink-900 dark:hover:text-ink-50"
          >
            {ws.data?.name ?? '…'}
          </Link>
          <span className="mx-1.5 text-ink-300">/</span>
          <Link
            to={`/workspaces/${wsId}/projects/${projectId}`}
            className="hover:text-ink-900 dark:hover:text-ink-50"
          >
            {project.data?.name ?? '…'}
          </Link>
          <span className="mx-1.5 text-ink-300">/</span>
          <span className="text-ink-700 dark:text-ink-300">
            {experiment.data?.name ?? '…'}
          </span>
        </nav>
        <div className="flex items-end justify-between gap-4">
          <div>
            <h1 className="h-page">{experiment.data?.name ?? 'Loading…'}</h1>
            {experiment.data && (
              <div className="mt-2 flex items-center gap-2">
                <span className="pill-neutral">{experiment.data.task}</span>
                {experiment.data.target && (
                  <span className="text-sm text-ink-500">
                    target:{' '}
                    <span className="font-mono text-ink-700 dark:text-ink-300">
                      {experiment.data.target}
                    </span>
                  </span>
                )}
              </div>
            )}
          </div>
          <button
            className="btn-primary shrink-0"
            onClick={() => setRunDialogOpen(true)}
            disabled={!experiment.data}
          >
            <PlusIcon />
            New run
          </button>
        </div>
      </header>

      {/* ─── Main (single column) ────────────────────────────── */}
      <div className="space-y-8">
        {/* Setup parameters */}
        <section>
          <h2 className="h-section mb-3">Setup parameters</h2>
          {setupEntries.length === 0 ? (
            <p className="text-sm text-ink-500">
              Using engine defaults for every parameter (none overridden).
            </p>
          ) : (
            <dl className="card grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-2">
              {setupEntries.map(([k, v]) => (
                <div
                  key={k}
                  className="flex items-baseline justify-between gap-3 text-sm"
                >
                  <dt className="text-ink-500 font-mono text-xs">{k}</dt>
                  <dd className="text-ink-900 dark:text-ink-50 font-mono text-xs truncate">
                    {typeof v === 'object' ? JSON.stringify(v) : String(v)}
                  </dd>
                </div>
              ))}
            </dl>
          )}
        </section>

        {/* Runs */}
        <section>
          <header className="mb-4 flex items-baseline justify-between">
            <h2 className="h-section">
              Runs{' '}
              <span className="text-ink-400 font-normal">
                ({runs.data?.length ?? 0})
              </span>
            </h2>
          </header>

          {runs.isLoading && (
            <div className="card text-sm text-ink-500">Loading…</div>
          )}
          {runs.error && (
            <div className="card text-sm text-danger-600">
              {errorMessage(runs.error)}
            </div>
          )}

          {runs.data && runs.data.length === 0 && (
            <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 p-10 text-center">
              <h3 className="text-base font-semibold text-ink-900 dark:text-ink-50">
                No runs yet
              </h3>
              <p className="mt-1.5 text-sm text-ink-500">
                Click <span className="font-medium text-ink-700">New run</span> to
                submit your first one.
              </p>
              <button
                className="btn-primary mt-5 inline-flex"
                onClick={() => setRunDialogOpen(true)}
              >
                <PlusIcon />
                New run
              </button>
            </div>
          )}

            {runs.data && runs.data.length > 0 && (
              <div className="card overflow-hidden p-0">
                <table className="w-full text-sm">
                  <thead className="bg-white text-ink-500 dark:bg-ink-900">
                    <tr>
                      <th className="px-4 py-2 text-left font-medium">Status</th>
                      <th className="px-4 py-2 text-left font-medium">Plan</th>
                      <th className="px-4 py-2 text-left font-medium">Dataset</th>
                      <th className="px-4 py-2 text-right font-medium">Duration</th>
                      <th className="px-4 py-2 text-left font-medium">Created</th>
                    </tr>
                  </thead>
                  <tbody>
                    {runs.data.map((r) => {
                      const snapshot = (r.snapshot ?? {}) as {
                        plan?: string;
                        sklearn_dataset?: string;
                        data_source_id?: string;
                      };
                      const planText = snapshot.plan ?? '—';
                      const dsText = snapshot.sklearn_dataset
                        ? snapshot.sklearn_dataset
                        : snapshot.data_source_id
                          ? sourceLabelMap[snapshot.data_source_id] ??
                            `${snapshot.data_source_id.slice(0, 8)}…`
                          : '—';
                      return (
                        <tr
                          key={r.id}
                          className="border-t border-ink-200 dark:border-ink-800 hover:bg-ink-50 dark:hover:bg-ink-950/40"
                        >
                          <td className="px-4 py-2">
                            <Link to={`/runs/${r.id}`}>
                              <span
                                className={`${STATUS_PILL[r.status] ?? 'pill-neutral'} capitalize`}
                              >
                                {r.status}
                              </span>
                            </Link>
                          </td>
                          <td className="px-4 py-2">
                            <span className="pill-neutral">{planText}</span>
                          </td>
                          <td className="px-4 py-2 text-ink-700 dark:text-ink-300">
                            {dsText}
                          </td>
                          <td className="px-4 py-2 text-right tabular-nums text-ink-700 dark:text-ink-300">
                            {formatDuration(r.duration_ms)}
                          </td>
                          <td className="px-4 py-2 text-xs text-ink-500 whitespace-nowrap">
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

      {/* ─── New-run dialog (opens from header CTA) ──────────── */}
      <Dialog
        open={runDialogOpen}
        onClose={() => !submitRun.isPending && setRunDialogOpen(false)}
        title="New run"
        description={`Submit another run of "${experiment.data?.name ?? ''}".`}
        size="md"
      >
        <div className="space-y-4">
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
              <option value="compare">compare — leaderboard</option>
              <option value="create">create — train one model</option>
              <option value="search">search — preprocess × models</option>
              <option value="setup">setup — preprocess only</option>
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
                  <option
                    key={m.id}
                    value={m.id}
                    disabled={!m.is_available}
                  >
                    {m.id} — {m.name}
                    {!m.is_available ? ' (install required)' : ''}
                  </option>
                ))}
              </select>
              {models.error && (
                <p className="text-sm text-danger-600 mt-1">
                  {errorMessage(models.error)}
                </p>
              )}
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
              disabled={dataSources.isLoading}
            >
              {sourceOptions.csvOpts.length > 0 && (
                <optgroup label="Workspace CSVs">
                  {sourceOptions.csvOpts.map((s) => (
                    <option key={s.value} value={s.value}>
                      {s.label}
                    </option>
                  ))}
                </optgroup>
              )}
              {sourceOptions.sklearnOpts.length > 0 && (
                <optgroup label="Built-in samples">
                  {sourceOptions.sklearnOpts.map((s) => (
                    <option key={s.value} value={s.value}>
                      {s.label}
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
            {noCsvs ? (
              <p className="text-xs text-ink-500 mt-1.5">
                No CSVs in this workspace yet.{' '}
                <Link
                  to={`/workspaces/${wsId}/projects/${projectId}`}
                  className="text-accent-600 hover:underline"
                  onClick={() => setRunDialogOpen(false)}
                >
                  Upload one →
                </Link>
              </p>
            ) : (
              <p className="text-xs text-ink-500 mt-1.5">
                Defaults to whatever the previous run used.
              </p>
            )}
          </div>

          {submitRun.error && (
            <p className="text-sm text-danger-600">
              {errorMessage(submitRun.error)}
            </p>
          )}

          <div className="flex items-center justify-end gap-2 pt-2">
            <button
              className="btn-secondary"
              onClick={() => setRunDialogOpen(false)}
              disabled={submitRun.isPending}
            >
              Cancel
            </button>
            <button
              className="btn-primary"
              disabled={submitRun.isPending || !source}
              onClick={() => submitRun.mutate()}
            >
              {submitRun.isPending ? 'Submitting…' : 'Submit run'}
            </button>
          </div>
        </div>
      </Dialog>
    </div>
  );
}

const PlusIcon = () => (
  <svg
    width="14"
    height="14"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.75"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden
  >
    <path d="M12 5v14" />
    <path d="M5 12h14" />
  </svg>
);
