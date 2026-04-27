/**
 * Create-experiment page — opinionated, single-screen, end-to-end.
 *
 * /workspaces/:wsId/projects/:projectId/experiments/new
 *
 * Sections (top to bottom):
 *   1. Identity & task — name, task pills (visual), description
 *   2. Data source — inline pick from existing CSVs + Upload button
 *      with column auto-detection of target
 *   3. Target column — populated from selected data source columns
 *   4. Run plan — Compare / Create / Setup with sensible defaults
 *   5. Advanced configuration — collapsible DynamicForm
 *
 * Sticky summary card at the bottom shows a live preview of what will
 * happen + the primary "Create & run" action that submits both
 * experimentsApi.create AND runsApi.submit, then routes to /runs/:id.
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
import type { DataSource, RunPlan, TaskType } from '@/api/types';
import { DynamicForm } from '@/components/DynamicForm';
import {
  applyDefaults,
  stripDefaults,
  type ParamValues,
} from '@/components/DynamicForm.helpers';
import { ExperimentDesignerModal } from '@/components/ExperimentDesignerModal';

// ─── Task definitions ─────────────────────────────────────────────

interface TaskOption {
  value: TaskType;
  label: string;
  description: string;
  icon: React.ReactNode;
  needsTarget: boolean;
}

const TASKS: TaskOption[] = [
  {
    value: 'classification',
    label: 'Classification',
    description: 'Predict a category or label.',
    icon: <ClassifyIcon />,
    needsTarget: true,
  },
  {
    value: 'regression',
    label: 'Regression',
    description: 'Predict a numeric value.',
    icon: <RegressIcon />,
    needsTarget: true,
  },
  {
    value: 'clustering',
    label: 'Clustering',
    description: 'Discover groups in unlabeled data.',
    icon: <ClusterIcon />,
    needsTarget: false,
  },
  {
    value: 'anomaly',
    label: 'Anomaly detection',
    description: 'Flag unusual rows.',
    icon: <AnomalyIcon />,
    needsTarget: false,
  },
  {
    value: 'time_series',
    label: 'Time series',
    description: 'Forecast a value over time.',
    icon: <TimeSeriesIcon />,
    needsTarget: true,
  },
];

const TARGET_HINT_PATTERNS = [
  /^target$/i,
  /^label$/i,
  /^y$/i,
  /^class$/i,
  /^outcome$/i,
  /^churn$/i,
  /^purchase$/i,
  /^medv$/i,
  /^price$/i,
];

function autoDetectTarget(columns: string[] | undefined): string | null {
  if (!columns || columns.length === 0) return null;
  for (const pat of TARGET_HINT_PATTERNS) {
    const hit = columns.find((c) => pat.test(c));
    if (hit) return hit;
  }
  // Fallback: last column (sklearn convention)
  return columns[columns.length - 1] ?? null;
}

interface DataSourceConfig {
  rows?: number;
  size_bytes?: number;
  columns?: string[];
}

function getCfg(d: DataSource): DataSourceConfig {
  return (d.config ?? {}) as DataSourceConfig;
}

// ─── Main page ────────────────────────────────────────────────────

export function NewExperiment() {
  const { wsId = '', projectId = '' } = useParams<{
    wsId: string;
    projectId: string;
  }>();
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
  const dataSources = useQuery({
    queryKey: ['data-sources', wsId],
    queryFn: () => dataSourcesApi.list(wsId),
    enabled: !!wsId,
  });

  const csvs = useMemo(
    () => (dataSources.data ?? []).filter((d) => d.kind === 'csv_upload'),
    [dataSources.data],
  );

  // ── State
  const [name, setName] = useState('');
  const [task, setTask] = useState<TaskType>('classification');
  const [target, setTarget] = useState('');
  const [dataSourceId, setDataSourceId] = useState<string>('');
  const [plan, setPlan] = useState<RunPlan>('compare');
  const [params, setParams] = useState<ParamValues>({});
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [askAI, setAskAI] = useState(false);

  const taskCfg = TASKS.find((t) => t.value === task)!;
  const needsTarget = taskCfg.needsTarget;

  // Auto-pick the first CSV when the list loads.
  useEffect(() => {
    if (!dataSourceId && csvs.length > 0) {
      setDataSourceId(csvs[0].id);
    }
  }, [csvs, dataSourceId]);

  // Auto-detect target when data source / task changes.
  const selectedDS = csvs.find((d) => d.id === dataSourceId);
  const cols = selectedDS ? getCfg(selectedDS).columns : undefined;
  useEffect(() => {
    if (!needsTarget) return;
    if (target) return;
    const guess = autoDetectTarget(cols);
    if (guess) setTarget(guess);
  }, [cols, needsTarget, target]);

  // Engine-driven schema for advanced config.
  const schema = useQuery({
    queryKey: ['describe', 'setup-params', task],
    queryFn: () => describeApi.setupParams(task),
    staleTime: 10 * 60 * 1000,
  });
  const seededParams = useMemo(() => {
    if (!schema.data) return params;
    return applyDefaults(schema.data, params);
  }, [schema.data, params]);

  // ── Submit: create experiment + start a run, in one go.
  const submit = useMutation({
    mutationFn: async () => {
      if (!schema.data) throw new Error('Setup schema not loaded yet.');
      const exp = await experimentsApi.create(projectId, {
        name: name.trim(),
        task,
        target: needsTarget ? target.trim() || null : null,
        setup_params: stripDefaults(schema.data, seededParams),
      });
      const run = await runsApi.submit(exp.id, {
        plan,
        data_source_id: dataSourceId || null,
        target: needsTarget ? target.trim() || null : null,
      });
      return { exp, run };
    },
    onSuccess: ({ run }) => {
      qc.invalidateQueries({ queryKey: ['experiments', projectId] });
      nav(`/runs/${run.id}`, { replace: true });
    },
  });

  const canSubmit =
    name.trim().length > 0 &&
    (!needsTarget || target.trim().length > 0) &&
    dataSourceId !== '' &&
    schema.data &&
    !submit.isPending;

  return (
    <div className="space-y-8 pb-32">
      {/* ─── Hero ─────────────────────────────────────────────── */}
      <header className="space-y-2">
        <nav className="text-xs text-ink-500">
          <Link to="/" className="hover:text-ink-900 dark:hover:text-ink-50 transition-colors">
            Workspaces
          </Link>
          <span className="mx-1.5 text-ink-300 dark:text-ink-700">/</span>
          <Link
            to={`/workspaces/${wsId}`}
            className="hover:text-ink-900 dark:hover:text-ink-50 transition-colors"
          >
            {ws.data?.name ?? '…'}
          </Link>
          <span className="mx-1.5 text-ink-300 dark:text-ink-700">/</span>
          <Link
            to={`/workspaces/${wsId}/projects/${projectId}`}
            className="hover:text-ink-900 dark:hover:text-ink-50 transition-colors"
          >
            {project.data?.name ?? '…'}
          </Link>
          <span className="mx-1.5 text-ink-300 dark:text-ink-700">/</span>
          <span className="text-ink-700 dark:text-ink-300">New experiment</span>
        </nav>
        <div className="flex items-start justify-between gap-6">
          <div>
            <h1 className="h-page">New experiment</h1>
            <p className="mt-2 text-sm text-ink-500 max-w-2xl">
              Configure the run, pick a dataset, and launch — all on one screen.
            </p>
          </div>
          <button
            type="button"
            onClick={() => setAskAI(true)}
            className="btn-secondary shrink-0"
            title="Ask AI to propose a configuration"
          >
            <SparkIcon />
            Ask AI
          </button>
        </div>
      </header>

      {/* ═══ STEP 1 — Identity & task ═══════════════════════════ */}
      <Section
        index="1"
        title="Identity & task"
        description="Give it a name and tell us what kind of model you want."
      >
        <div className="space-y-5">
          <div>
            <label className="field" htmlFor="name">
              Experiment name <span className="text-ink-400 font-normal">*</span>
            </label>
            <input
              id="name"
              className="input max-w-md"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. baseline"
              required
            />
          </div>

          <div>
            <div className="field mb-2">Task</div>
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {TASKS.map((t) => (
                <TaskPill
                  key={t.value}
                  task={t}
                  selected={task === t.value}
                  onClick={() => {
                    setTask(t.value);
                    setParams({});
                  }}
                />
              ))}
            </div>
          </div>
        </div>
      </Section>

      {/* ═══ STEP 2 — Data source ═══════════════════════════════ */}
      <Section
        index="2"
        title="Data source"
        description="Pick a CSV from this workspace. Each experiment runs against one dataset."
      >
        {dataSources.isLoading && (
          <div className="text-sm text-ink-500">Loading data sources…</div>
        )}
        {csvs.length === 0 && !dataSources.isLoading && (
          <div className="rounded-xl bg-white dark:bg-ink-900 border border-dashed border-ink-300 dark:border-ink-700 px-6 py-10 text-center">
            <p className="text-sm text-ink-700 dark:text-ink-300 mb-3">
              No data sources in this workspace yet.
            </p>
            <Link
              to={`/workspaces/${wsId}`}
              className="btn-secondary"
            >
              Upload a CSV →
            </Link>
          </div>
        )}
        {csvs.length > 0 && (
          <div className="grid gap-2 md:grid-cols-2">
            {csvs.map((d) => (
              <DataSourceCard
                key={d.id}
                ds={d}
                selected={dataSourceId === d.id}
                onSelect={() => setDataSourceId(d.id)}
              />
            ))}
          </div>
        )}

        {needsTarget && (
          <div className="mt-6">
            <label className="field" htmlFor="target">
              Target column <span className="text-ink-400 font-normal">*</span>
            </label>
            {cols && cols.length > 0 ? (
              <select
                id="target"
                className="input max-w-md"
                value={target}
                onChange={(e) => setTarget(e.target.value)}
              >
                <option value="">— pick a column —</option>
                {cols.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            ) : (
              <input
                id="target"
                className="input max-w-md"
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                placeholder="e.g. churn"
              />
            )}
            <p className="hint mt-1.5">
              {target && cols?.includes(target)
                ? '✓ Auto-detected from column names. Override above if needed.'
                : 'The column the model will learn to predict.'}
            </p>
          </div>
        )}
      </Section>

      {/* ═══ STEP 3 — Run plan ══════════════════════════════════ */}
      <Section
        index="3"
        title="Run plan"
        description="What should the engine do when this experiment runs?"
      >
        <div className="space-y-2">
          <PlanOption
            value="compare"
            current={plan}
            onSelect={setPlan}
            label="Compare models"
            description="Train + cross-validate every model for this task and rank them by metric. The standard first run."
            recommended
          />
          <PlanOption
            value="setup"
            current={plan}
            onSelect={setPlan}
            label="Setup only"
            description="Run preprocessing, materialise the train/test split, then stop. Useful for sanity-checking the pipeline before training."
          />
          <PlanOption
            value="create"
            current={plan}
            onSelect={setPlan}
            label="Create a single model"
            description="Train one model end-to-end. Picks the engine's default for this task. Pick a specific model after the first run."
          />
        </div>
      </Section>

      {/* ═══ STEP 4 — Advanced (collapsed) ══════════════════════ */}
      <Section
        index="4"
        title="Advanced configuration"
        description="Preprocessing, sampling, training knobs. Defaults are sensible — only open this if you need to tune."
        collapsible
        open={advancedOpen}
        onToggle={() => setAdvancedOpen((v) => !v)}
      >
        {advancedOpen && (
          <>
            {schema.isLoading && (
              <p className="text-sm text-ink-500">Loading schema from the engine…</p>
            )}
            {schema.error && (
              <p className="error">{errorMessage(schema.error)}</p>
            )}
            {schema.data && (
              <DynamicForm
                schema={schema.data}
                values={seededParams}
                onChange={setParams}
                hide={['target']}
                disabled={submit.isPending}
                columns={cols}
              />
            )}
          </>
        )}
      </Section>

      {/* ═══ Sticky summary + CTA ═══════════════════════════════ */}
      <div className="fixed bottom-0 left-0 right-0 z-30 border-t border-ink-200 dark:border-ink-800 bg-white/85 dark:bg-ink-950/85 backdrop-blur-md md:left-60 lg:left-64">
        <div className="mx-auto max-w-6xl px-6 py-3 flex items-center justify-between gap-4">
          <SummaryStrip
            name={name}
            task={taskCfg.label}
            target={needsTarget ? target : null}
            dsName={selectedDS?.name}
            dsRows={selectedDS ? getCfg(selectedDS).rows : null}
            plan={plan}
          />
          <div className="flex items-center gap-2 shrink-0">
            <Link
              to={`/workspaces/${wsId}/projects/${projectId}`}
              className="btn-ghost"
            >
              Cancel
            </Link>
            <button
              type="button"
              className="btn-primary"
              onClick={() => submit.mutate()}
              disabled={!canSubmit}
            >
              {submit.isPending ? 'Creating…' : 'Create & run'}
              <ArrowRightIcon />
            </button>
          </div>
        </div>
        {submit.error && (
          <div className="mx-auto max-w-6xl px-6 pb-2 text-xs text-danger-600">
            {errorMessage(submit.error)}
          </div>
        )}
      </div>

      <ExperimentDesignerModal
        workspaceId={wsId}
        open={askAI}
        onClose={() => setAskAI(false)}
      />
    </div>
  );
}

// ─── Section wrapper ──────────────────────────────────────────────

function Section({
  index,
  title,
  description,
  children,
  collapsible,
  open,
  onToggle,
}: {
  index: string;
  title: string;
  description: string;
  children: React.ReactNode;
  collapsible?: boolean;
  open?: boolean;
  onToggle?: () => void;
}) {
  return (
    <section className="rounded-xl bg-white dark:bg-ink-900 border border-ink-200 dark:border-ink-800 shadow-soft-1 overflow-hidden">
      <header
        className={`flex items-start gap-4 px-6 py-4 ${
          collapsible
            ? 'cursor-pointer select-none hover:bg-ink-50 dark:hover:bg-ink-800/40 transition-colors'
            : ''
        }`}
        onClick={collapsible ? onToggle : undefined}
        role={collapsible ? 'button' : undefined}
        aria-expanded={collapsible ? open : undefined}
      >
        <span className="h-7 w-7 rounded-full bg-ink-100 dark:bg-ink-800 text-ink-700 dark:text-ink-300 text-xs font-semibold flex items-center justify-center shrink-0 mt-0.5">
          {index}
        </span>
        <div className="min-w-0 flex-1">
          <h2 className="text-base font-semibold text-ink-900 dark:text-ink-50">
            {title}
          </h2>
          <p className="text-sm text-ink-500 mt-0.5">{description}</p>
        </div>
        {collapsible && (
          <span
            className={`text-ink-500 mt-1 transition-transform ${open ? 'rotate-90' : ''}`}
          >
            <ChevronRightIcon />
          </span>
        )}
      </header>
      {(!collapsible || open) && (
        <div className="px-6 pb-6 pt-2 border-t border-ink-100 dark:border-ink-800">
          {children}
        </div>
      )}
    </section>
  );
}

// ─── Task pill ────────────────────────────────────────────────────

function TaskPill({
  task,
  selected,
  onClick,
}: {
  task: TaskOption;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`text-left rounded-lg p-3 border-2 transition-all ${
        selected
          ? 'border-accent-500 bg-accent-50 dark:bg-accent-500/10'
          : 'border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 hover:border-ink-300 dark:hover:border-ink-700'
      }`}
    >
      <div className="flex items-start gap-3">
        <span
          className={`h-8 w-8 rounded-md flex items-center justify-center shrink-0 ${
            selected
              ? 'bg-accent-500 text-white'
              : 'bg-ink-100 dark:bg-ink-800 text-ink-600 dark:text-ink-400'
          }`}
        >
          {task.icon}
        </span>
        <div className="min-w-0">
          <div
            className={`text-sm font-semibold ${
              selected
                ? 'text-accent-700 dark:text-accent-300'
                : 'text-ink-900 dark:text-ink-50'
            }`}
          >
            {task.label}
          </div>
          <div className="text-xs text-ink-500 mt-0.5">{task.description}</div>
        </div>
      </div>
    </button>
  );
}

// ─── Data source card ────────────────────────────────────────────

function DataSourceCard({
  ds,
  selected,
  onSelect,
}: {
  ds: DataSource;
  selected: boolean;
  onSelect: () => void;
}) {
  const cfg = getCfg(ds);
  return (
    <button
      type="button"
      onClick={onSelect}
      className={`text-left rounded-lg p-3 border-2 transition-all ${
        selected
          ? 'border-accent-500 bg-accent-50 dark:bg-accent-500/10'
          : 'border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 hover:border-ink-300 dark:hover:border-ink-700'
      }`}
    >
      <div className="flex items-start gap-3">
        <span
          className={`h-8 w-8 rounded-md flex items-center justify-center shrink-0 ${
            selected
              ? 'bg-accent-500 text-white'
              : 'bg-ink-100 dark:bg-ink-800 text-ink-600 dark:text-ink-400'
          }`}
        >
          <DataIcon />
        </span>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-semibold text-ink-900 dark:text-ink-50 truncate">
            {ds.name}
          </div>
          <div className="text-xs text-ink-500 mt-0.5 tabular-nums">
            {cfg.rows != null && <>{cfg.rows.toLocaleString()} rows · </>}
            {cfg.columns?.length != null && <>{cfg.columns.length} cols</>}
          </div>
        </div>
        {selected && (
          <span className="text-accent-600 dark:text-accent-400 shrink-0">
            <CheckIcon />
          </span>
        )}
      </div>
    </button>
  );
}

// ─── Run plan option ─────────────────────────────────────────────

function PlanOption({
  value,
  current,
  onSelect,
  label,
  description,
  recommended,
}: {
  value: RunPlan;
  current: RunPlan;
  onSelect: (v: RunPlan) => void;
  label: string;
  description: string;
  recommended?: boolean;
}) {
  const selected = value === current;
  return (
    <button
      type="button"
      onClick={() => onSelect(value)}
      className={`w-full text-left rounded-lg p-3 border-2 flex items-start gap-3 transition-all ${
        selected
          ? 'border-accent-500 bg-accent-50 dark:bg-accent-500/10'
          : 'border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 hover:border-ink-300 dark:hover:border-ink-700'
      }`}
    >
      <span
        className={`h-4 w-4 rounded-full border-2 flex items-center justify-center mt-0.5 shrink-0 ${
          selected ? 'border-accent-500' : 'border-ink-300 dark:border-ink-700'
        }`}
      >
        {selected && <span className="h-2 w-2 rounded-full bg-accent-500" />}
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-ink-900 dark:text-ink-50">
            {label}
          </span>
          {recommended && <span className="pill-accent">Recommended</span>}
        </div>
        <p className="text-xs text-ink-500 mt-0.5">{description}</p>
      </div>
    </button>
  );
}

// ─── Summary strip ───────────────────────────────────────────────

function SummaryStrip({
  name,
  task,
  target,
  dsName,
  dsRows,
  plan,
}: {
  name: string;
  task: string;
  target: string | null;
  dsName?: string;
  dsRows?: number | null;
  plan: RunPlan;
}) {
  const hasName = name.trim().length > 0;
  const planLabel =
    plan === 'compare' ? 'Compare models' : plan === 'setup' ? 'Setup only' : 'Single model';
  return (
    <div className="min-w-0 flex-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
      <span className="text-ink-500">Will create:</span>
      <span className="pill-neutral">
        {hasName ? name : 'unnamed'}
      </span>
      <span className="text-ink-300 dark:text-ink-700">·</span>
      <span className="pill-accent">{task}</span>
      {target && (
        <>
          <span className="text-ink-300 dark:text-ink-700">·</span>
          <span className="text-ink-700 dark:text-ink-300">
            target=<span className="font-mono">{target}</span>
          </span>
        </>
      )}
      {dsName && (
        <>
          <span className="text-ink-300 dark:text-ink-700">·</span>
          <span className="text-ink-700 dark:text-ink-300 truncate">
            {dsName}
            {dsRows != null && (
              <span className="text-ink-500 tabular-nums"> ({dsRows.toLocaleString()} rows)</span>
            )}
          </span>
        </>
      )}
      <span className="text-ink-300 dark:text-ink-700">·</span>
      <span className="pill-neutral">{planLabel}</span>
    </div>
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
const lg = { ...sx, width: '18', height: '18' };

function ClassifyIcon() {
  return (
    <svg {...lg}>
      <rect x="3" y="3" width="7" height="7" rx="1" />
      <rect x="14" y="3" width="7" height="7" rx="1" />
      <rect x="3" y="14" width="7" height="7" rx="1" />
      <rect x="14" y="14" width="7" height="7" rx="1" />
    </svg>
  );
}
function RegressIcon() {
  return (
    <svg {...lg}>
      <path d="M3 17l4-7 5 4 5-9 4 7" />
    </svg>
  );
}
function ClusterIcon() {
  return (
    <svg {...lg}>
      <circle cx="6" cy="6" r="2" />
      <circle cx="18" cy="6" r="2" />
      <circle cx="6" cy="18" r="2" />
      <circle cx="18" cy="18" r="2" />
      <circle cx="12" cy="12" r="2" />
    </svg>
  );
}
function AnomalyIcon() {
  return (
    <svg {...lg}>
      <path d="M12 9v4 M12 17h.01" />
      <circle cx="12" cy="12" r="9" />
    </svg>
  );
}
function TimeSeriesIcon() {
  return (
    <svg {...lg}>
      <path d="M3 12h3l2-7 4 14 2-7h7" />
    </svg>
  );
}
function DataIcon() {
  return (
    <svg {...sx}>
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M3 5v14a9 3 0 0 0 18 0V5" />
      <path d="M3 12a9 3 0 0 0 18 0" />
    </svg>
  );
}
function SparkIcon() {
  return (
    <svg {...sx}>
      <path d="M12 3v3 M12 18v3 M3 12h3 M18 12h3" />
      <path d="M5.6 5.6l2.1 2.1 M16.3 16.3l2.1 2.1 M5.6 18.4l2.1-2.1 M16.3 7.7l2.1-2.1" />
    </svg>
  );
}
function CheckIcon() {
  return (
    <svg {...sx}>
      <path d="M20 6L9 17l-5-5" />
    </svg>
  );
}
function ChevronRightIcon() {
  return (
    <svg {...sx}>
      <path d="m9 18 6-6-6-6" />
    </svg>
  );
}
function ArrowRightIcon() {
  return (
    <svg {...sx}>
      <path d="M5 12h14 M12 5l7 7-7 7" />
    </svg>
  );
}
