/**
 * Create-experiment wizard at /workspaces/:wsId/projects/:projectId/experiments/new.
 *
 * Two stages in a single single-column form:
 *  1. Identity: name, task, target column.
 *  2. Setup parameters: rendered entirely from `describe_setup_params(task)`
 *     via <DynamicForm>. Zero hard-coded parameter names here — if the
 *     engine adds `transformation_method: "quantile" | "yeo-johnson"`
 *     tomorrow, this form picks it up without a code change.
 *
 * Submit → POST /projects/:id/experiments → navigate to the experiment
 * detail view where the user picks a data source and fires the first run.
 */

import { useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useNavigate, useParams } from 'react-router-dom';
import {
  describeApi,
  experimentsApi,
  projectsApi,
  workspacesApi,
} from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import type { TaskType } from '@/api/types';
import { DynamicForm } from '@/components/DynamicForm';
import {
  applyDefaults,
  stripDefaults,
  type ParamValues,
} from '@/components/DynamicForm.helpers';

const TASKS: { value: TaskType; label: string }[] = [
  { value: 'classification', label: 'Classification' },
  { value: 'regression', label: 'Regression' },
  { value: 'clustering', label: 'Clustering' },
  { value: 'anomaly', label: 'Anomaly detection' },
  { value: 'time_series', label: 'Time series' },
];

function requiresTarget(task: TaskType): boolean {
  // Unsupervised tasks don't use a target column.
  return task !== 'clustering' && task !== 'anomaly';
}

export function NewExperiment() {
  const { wsId = '', projectId = '' } = useParams<{ wsId: string; projectId: string }>();
  const nav = useNavigate();
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

  const [name, setName] = useState('');
  const [task, setTask] = useState<TaskType>('classification');
  const [target, setTarget] = useState('');
  const [params, setParams] = useState<ParamValues>({});

  // Engine-driven schema. Reloads whenever the task changes.
  const schema = useQuery({
    queryKey: ['describe', 'setup-params', task],
    queryFn: () => describeApi.setupParams(task),
    // Setup schemas are effectively static per release — long cache.
    staleTime: 10 * 60 * 1000,
  });

  // When the schema arrives, seed sensible defaults (but keep anything the
  // user has already typed).
  const seededParams = useMemo(() => {
    if (!schema.data) return params;
    return applyDefaults(schema.data, params);
  }, [schema.data, params]);

  const create = useMutation({
    mutationFn: () => {
      if (!schema.data) throw new Error('setup schema not loaded');
      return experimentsApi.create(projectId, {
        name: name.trim(),
        task,
        target: requiresTarget(task) ? target.trim() || null : null,
        // Strip defaults so we don't pin the experiment to today's values.
        // The engine owns defaults; we only record user intent.
        setup_params: stripDefaults(schema.data, seededParams),
      });
    },
    onSuccess: (exp) => {
      qc.invalidateQueries({ queryKey: ['experiments', projectId] });
      nav(`/workspaces/${wsId}/projects/${projectId}/experiments/${exp.id}`, {
        replace: true,
      });
    },
  });

  const canSubmit =
    name.trim().length > 0 &&
    (!requiresTarget(task) || target.trim().length > 0) &&
    !create.isPending &&
    schema.data;

  return (
    <div className="max-w-2xl mx-auto space-y-8">
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
          <span>New experiment</span>
        </nav>
        <h1 className="text-xl font-semibold">New experiment</h1>
        <p className="text-sm text-ink-200/70 mt-1">
          Configure the task + preprocessing. You'll pick a data source + run plan on
          the next screen.
        </p>
      </header>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (canSubmit) create.mutate();
        }}
        className="space-y-8"
      >
        {/* ────────── Identity */}
        <section className="card space-y-5">
          <h2 className="text-sm font-medium text-ink-100">Identity</h2>

          <div>
            <label className="field" htmlFor="name">
              Name <span className="text-danger-500">*</span>
            </label>
            <input
              id="name"
              className="input"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="baseline"
              required
            />
          </div>

          <div>
            <label className="field" htmlFor="task">
              Task <span className="text-danger-500">*</span>
            </label>
            <select
              id="task"
              className="input"
              value={task}
              onChange={(e) => {
                setTask(e.target.value as TaskType);
                // Reset params when switching tasks — the schema changes
                // and previous values are likely invalid.
                setParams({});
              }}
            >
              {TASKS.map((t) => (
                <option key={t.value} value={t.value}>
                  {t.label}
                </option>
              ))}
            </select>
            <p className="hint mt-1">
              Determines which models + metrics are available and which preprocessing
              steps apply.
            </p>
          </div>

          {requiresTarget(task) && (
            <div>
              <label className="field" htmlFor="target">
                Target column <span className="text-danger-500">*</span>
              </label>
              <input
                id="target"
                className="input"
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                placeholder="e.g. churn"
              />
              <p className="hint mt-1">
                Column name in the dataset that holds the label / prediction target.
              </p>
            </div>
          )}
        </section>

        {/* ────────── Setup parameters (dynamic, engine-driven) */}
        <section className="card">
          <h2 className="text-sm font-medium text-ink-100 mb-6">Setup parameters</h2>

          {schema.isLoading && (
            <p className="hint">Loading schema from the engine…</p>
          )}
          {schema.error && <p className="error">{errorMessage(schema.error)}</p>}
          {schema.data && (
            <DynamicForm
              schema={schema.data}
              values={seededParams}
              onChange={setParams}
              // `target` is collected above in the Identity section; don't re-ask.
              hide={['target']}
              disabled={create.isPending}
            />
          )}
        </section>

        {create.error && <p className="error">{errorMessage(create.error)}</p>}

        <div className="flex items-center justify-end gap-3">
          <Link
            to={`/workspaces/${wsId}/projects/${projectId}`}
            className="btn-ghost"
          >
            Cancel
          </Link>
          <button type="submit" className="btn-primary" disabled={!canSubmit}>
            {create.isPending ? 'Creating…' : 'Create experiment'}
          </button>
        </div>
      </form>
    </div>
  );
}
