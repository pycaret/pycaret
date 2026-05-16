/**
 * /workspaces/:wsId/projects/:projectId/analyses — Phase 11
 * statistical computing surface.
 *
 * Three modes in one page:
 *
 * 1. **List** (default) — every saved Analysis in this project + a
 *    "Quick run" launcher that drops you into mode 2.
 * 2. **Wizard** — pick a kind, fill in a typed form (column pickers
 *    + alpha + procedure-specific knobs), preview against a transient
 *    run, save when happy.
 * 3. **Result** — for an executed Analysis, render the uniform
 *    envelope (test statistic / p-value / effect size + interpretation
 *    + table + Plotly figure).
 */

import { useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useParams } from 'react-router-dom';
import {
  analysesApi,
  dataSourcesApi,
} from '@/api/endpoints';
import { BackButton } from '@/components/BackButton';
import { PlotlyFigure } from '@/components/PlotlyFigure';
import type { AnalysisKind, AnalysisResult } from '@/api/types';

const KIND_LABELS: Record<AnalysisKind, string> = {
  ttest: 'Two-sample t-test',
  welch_ttest: "Welch's t-test",
  paired_ttest: 'Paired t-test',
  mannwhitney: 'Mann–Whitney U',
  anova_oneway: 'One-way ANOVA',
  kruskal: 'Kruskal–Wallis',
  chi2: 'Chi-square independence',
  ols: 'OLS regression',
  kaplan_meier: 'Kaplan–Meier survival',
  logrank: 'Log-rank test',
  cox_ph: 'Cox proportional hazards',
  arima: 'ARIMA forecast',
  prophet: 'Prophet forecast',
};

const KIND_CATEGORIES: { label: string; kinds: AnalysisKind[] }[] = [
  { label: 'Compare two groups', kinds: ['ttest', 'welch_ttest', 'paired_ttest', 'mannwhitney'] },
  { label: 'Compare many groups', kinds: ['anova_oneway', 'kruskal'] },
  { label: 'Categorical association', kinds: ['chi2'] },
  { label: 'Regression', kinds: ['ols'] },
  { label: 'Survival', kinds: ['kaplan_meier', 'logrank', 'cox_ph'] },
  { label: 'Forecasting', kinds: ['arima', 'prophet'] },
];

type Mode =
  | { kind: 'list' }
  | { kind: 'wizard'; analysisKind: AnalysisKind | null }
  | { kind: 'result'; analysisId: string };

export function Analyses() {
  const { wsId = '', projectId = '' } = useParams<{
    wsId: string;
    projectId: string;
  }>();
  const [mode, setMode] = useState<Mode>({ kind: 'list' });

  return (
    <div className="space-y-6">
      <header>
        <BackButton />
        <h1 className="h-page mt-2">Analyses</h1>
        <p className="muted small">
          Statistical procedures over a DataSource — t-tests, ANOVA,
          regression, survival, forecasting. Phase 11.
        </p>
      </header>

      {mode.kind === 'list' && (
        <AnalysisList
          projectId={projectId}
          onNew={(k) => setMode({ kind: 'wizard', analysisKind: k })}
          onOpen={(id) => setMode({ kind: 'result', analysisId: id })}
        />
      )}
      {mode.kind === 'wizard' && (
        <AnalysisWizard
          projectId={projectId}
          wsId={wsId}
          initialKind={mode.analysisKind}
          onCancel={() => setMode({ kind: 'list' })}
          onSaved={(id) => setMode({ kind: 'result', analysisId: id })}
        />
      )}
      {mode.kind === 'result' && (
        <AnalysisResultPane
          analysisId={mode.analysisId}
          onBack={() => setMode({ kind: 'list' })}
        />
      )}
    </div>
  );
}

// ─────────────────────────────────────────── list

function AnalysisList({
  projectId,
  onNew,
  onOpen,
}: {
  projectId: string;
  onNew: (k: AnalysisKind | null) => void;
  onOpen: (id: string) => void;
}) {
  const list = useQuery({
    queryKey: ['analyses', 'for-project', projectId],
    queryFn: () => analysesApi.forProject(projectId),
    enabled: !!projectId,
  });

  return (
    <div className="space-y-6">
      <section>
        <h2 className="h-section mb-3">Start a new analysis</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {KIND_CATEGORIES.map((cat) => (
            <div
              key={cat.label}
              className="card"
            >
              <p className="text-sm font-semibold mb-2">{cat.label}</p>
              <div className="flex flex-wrap gap-2">
                {cat.kinds.map((k) => (
                  <button
                    key={k}
                    type="button"
                    className="pill-neutral hover:bg-accent-50 dark:hover:bg-accent-500/15 hover:text-accent-700 dark:hover:text-accent-300"
                    onClick={() => onNew(k)}
                  >
                    {KIND_LABELS[k]}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h2 className="h-section mb-3">
          Saved analyses ({list.data?.length ?? 0})
        </h2>
        {list.isLoading && (
          <div className="card text-sm text-ink-500">Loading…</div>
        )}
        {list.data && list.data.length === 0 && (
          <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 p-8 text-center">
            <p className="text-sm text-ink-500">
              Nothing saved yet — pick a procedure above to start.
            </p>
          </div>
        )}
        {list.data && list.data.length > 0 && (
          <div className="card overflow-hidden p-0">
            <table className="w-full text-sm">
              <thead className="bg-white text-ink-500 dark:bg-ink-900 border-b border-ink-200 dark:border-ink-800">
                <tr>
                  <th className="px-4 py-2 text-left font-medium">Name</th>
                  <th className="px-4 py-2 text-left font-medium">Kind</th>
                  <th className="px-4 py-2 text-left font-medium">Created</th>
                  <th className="px-4 py-2 text-right" />
                </tr>
              </thead>
              <tbody>
                {list.data.map((a) => (
                  <tr
                    key={a.id}
                    className="border-t border-ink-200 dark:border-ink-800 hover:bg-ink-50 dark:hover:bg-ink-950/40 cursor-pointer"
                    onClick={() => onOpen(a.id)}
                  >
                    <td className="px-4 py-2 font-medium">{a.name}</td>
                    <td className="px-4 py-2">
                      <span className="pill-neutral">{KIND_LABELS[a.kind] ?? a.kind}</span>
                    </td>
                    <td className="px-4 py-2 text-xs text-ink-500">
                      {a.created_at
                        ? new Date(a.created_at).toLocaleString()
                        : '—'}
                    </td>
                    <td className="px-4 py-2 text-right">
                      <span className="text-ink-400">→</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

// ─────────────────────────────────────────── wizard

const KIND_PARAM_FIELDS: Record<AnalysisKind, { name: string; label: string; placeholder?: string }[]> = {
  ttest: [
    { name: 'grouping_column', label: 'Grouping column' },
    { name: 'measure_column', label: 'Measure column' },
  ],
  welch_ttest: [
    { name: 'grouping_column', label: 'Grouping column' },
    { name: 'measure_column', label: 'Measure column' },
  ],
  paired_ttest: [
    { name: 'column_a', label: 'Column A' },
    { name: 'column_b', label: 'Column B' },
  ],
  mannwhitney: [
    { name: 'grouping_column', label: 'Grouping column' },
    { name: 'measure_column', label: 'Measure column' },
  ],
  anova_oneway: [
    { name: 'grouping_column', label: 'Grouping column' },
    { name: 'measure_column', label: 'Measure column' },
  ],
  kruskal: [
    { name: 'grouping_column', label: 'Grouping column' },
    { name: 'measure_column', label: 'Measure column' },
  ],
  chi2: [
    { name: 'column_a', label: 'Column A' },
    { name: 'column_b', label: 'Column B' },
  ],
  ols: [
    { name: 'response', label: 'Response column' },
    { name: 'predictors', label: 'Predictors (comma-separated)' },
  ],
  kaplan_meier: [
    { name: 'time_column', label: 'Time column' },
    { name: 'event_column', label: 'Event column (0/1)' },
    { name: 'group_column', label: 'Group column (optional)' },
  ],
  logrank: [
    { name: 'time_column', label: 'Time column' },
    { name: 'event_column', label: 'Event column (0/1)' },
    { name: 'group_column', label: 'Group column' },
  ],
  cox_ph: [
    { name: 'time_column', label: 'Time column' },
    { name: 'event_column', label: 'Event column (0/1)' },
    { name: 'covariates', label: 'Covariates (comma-separated)' },
  ],
  arima: [
    { name: 'column', label: 'Series column' },
    { name: 'order', label: 'Order (p,d,q)', placeholder: '1,1,1' },
    { name: 'horizon', label: 'Horizon', placeholder: '10' },
  ],
  prophet: [
    { name: 'date_column', label: 'Date column' },
    { name: 'value_column', label: 'Value column' },
    { name: 'horizon_days', label: 'Horizon (days)', placeholder: '30' },
  ],
};

function parseParam(name: string, raw: string): unknown {
  // Comma-separated lists.
  if (name === 'predictors' || name === 'covariates') {
    return raw
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);
  }
  if (name === 'order') {
    return raw
      .split(',')
      .map((s) => parseInt(s.trim(), 10))
      .filter((v) => !Number.isNaN(v));
  }
  if (name === 'horizon' || name === 'horizon_days') {
    const n = Number(raw);
    return Number.isFinite(n) ? n : raw;
  }
  return raw;
}

function AnalysisWizard({
  projectId,
  wsId,
  initialKind,
  onCancel,
  onSaved,
}: {
  projectId: string;
  wsId: string;
  initialKind: AnalysisKind | null;
  onCancel: () => void;
  onSaved: (id: string) => void;
}) {
  const [kind, setKind] = useState<AnalysisKind>(initialKind ?? 'ttest');
  const [name, setName] = useState('');
  const [dataSourceId, setDataSourceId] = useState('');
  const [fieldValues, setFieldValues] = useState<Record<string, string>>({});
  const [preview, setPreview] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const dataSources = useQuery({
    queryKey: ['data-sources', wsId],
    queryFn: () => dataSourcesApi.list(wsId),
    enabled: !!wsId,
  });

  const fields = KIND_PARAM_FIELDS[kind];
  const params = useMemo<Record<string, unknown>>(
    () =>
      Object.fromEntries(
        fields.map((f) => [f.name, parseParam(f.name, fieldValues[f.name] ?? '')]),
      ),
    [fields, fieldValues],
  );

  const previewMutation = useMutation({
    mutationFn: async () =>
      analysesApi.runOnce({
        kind,
        params,
        data_source_id: dataSourceId || undefined,
      }),
    onSuccess: (data) => {
      setPreview(data.result);
      setError(null);
    },
    onError: (e) => {
      setError((e as Error).message);
      setPreview(null);
    },
  });

  const saveMutation = useMutation({
    mutationFn: async () =>
      analysesApi.create(projectId, {
        name: name.trim(),
        kind,
        params,
        data_source_id: dataSourceId || undefined,
      }),
    onSuccess: (data) => onSaved(data.id),
  });

  const csvs = useMemo(
    () => (dataSources.data ?? []).filter((d) => d.kind === 'csv_upload'),
    [dataSources.data],
  );

  return (
    <div className="space-y-6">
      <header className="flex items-baseline justify-between">
        <h2 className="h-section">
          New analysis · {KIND_LABELS[kind]}
        </h2>
        <button type="button" className="btn-secondary" onClick={onCancel}>
          Cancel
        </button>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <section className="card space-y-4">
          <label className="field">
            <span className="muted small">Procedure</span>
            <select
              className="input w-full"
              value={kind}
              onChange={(e) => {
                setKind(e.target.value as AnalysisKind);
                setFieldValues({});
                setPreview(null);
              }}
            >
              {(Object.keys(KIND_LABELS) as AnalysisKind[]).map((k) => (
                <option key={k} value={k}>
                  {KIND_LABELS[k]}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span className="muted small">Name</span>
            <input
              className="input w-full"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="height-by-treatment"
            />
          </label>

          <label className="field">
            <span className="muted small">Data source</span>
            <select
              className="input w-full"
              value={dataSourceId}
              onChange={(e) => setDataSourceId(e.target.value)}
            >
              <option value="">(pick a CSV upload)</option>
              {csvs.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name}
                </option>
              ))}
            </select>
          </label>

          <div className="border-t border-ink-200 dark:border-ink-800 pt-3">
            <p className="muted small mb-2">Parameters</p>
            <div className="space-y-2">
              {fields.map((f) => (
                <label key={f.name} className="field">
                  <span className="muted small">{f.label}</span>
                  <input
                    className="input w-full"
                    value={fieldValues[f.name] ?? ''}
                    placeholder={f.placeholder}
                    onChange={(e) =>
                      setFieldValues((s) => ({
                        ...s,
                        [f.name]: e.target.value,
                      }))
                    }
                  />
                </label>
              ))}
            </div>
          </div>

          <div className="flex gap-2 pt-2">
            <button
              type="button"
              className="btn-secondary"
              onClick={() => previewMutation.mutate()}
              disabled={!dataSourceId || previewMutation.isPending}
            >
              {previewMutation.isPending ? 'Running…' : 'Preview'}
            </button>
            <button
              type="button"
              className="btn-primary"
              onClick={() => saveMutation.mutate()}
              disabled={
                !name.trim() ||
                !dataSourceId ||
                saveMutation.isPending
              }
            >
              {saveMutation.isPending ? 'Saving…' : 'Save'}
            </button>
          </div>

          {error && (
            <div className="text-xs text-danger-600">{error}</div>
          )}
        </section>

        <section>
          {preview ? (
            <AnalysisResultCard result={preview} />
          ) : (
            <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 p-8 text-center">
              <p className="text-sm text-ink-500">
                Fill in the parameters and hit Preview to see the result.
              </p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────── result pane

function AnalysisResultPane({
  analysisId,
  onBack,
}: {
  analysisId: string;
  onBack: () => void;
}) {
  const qc = useQueryClient();
  const analysis = useQuery({
    queryKey: ['analyses', 'detail', analysisId],
    queryFn: () => analysesApi.get(analysisId),
  });
  const results = useQuery({
    queryKey: ['analyses', 'results', analysisId],
    queryFn: () => analysesApi.results(analysisId),
  });
  const run = useMutation({
    mutationFn: () => analysesApi.run(analysisId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['analyses', 'results', analysisId] });
    },
  });

  const latest = results.data?.[0]?.metrics ?? null;

  return (
    <div className="space-y-6">
      <header className="flex items-baseline justify-between">
        <div>
          <button
            type="button"
            className="text-sm text-ink-500 hover:text-ink-900 dark:hover:text-ink-50"
            onClick={onBack}
          >
            ← back to analyses
          </button>
          <h2 className="h-section mt-1">
            {analysis.data?.name ?? '…'}{' '}
            <span className="text-ink-400 font-normal">
              · {KIND_LABELS[analysis.data?.kind ?? 'ttest']}
            </span>
          </h2>
        </div>
        <button
          type="button"
          className="btn-primary"
          onClick={() => run.mutate()}
          disabled={run.isPending}
        >
          {run.isPending ? 'Running…' : 'Run again'}
        </button>
      </header>

      {latest ? (
        <AnalysisResultCard result={latest} />
      ) : (
        <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 p-8 text-center">
          <p className="text-sm text-ink-500">
            No results yet. Hit Run to execute the analysis.
          </p>
        </div>
      )}

      <section>
        <h3 className="h-section mb-3">
          History ({results.data?.length ?? 0})
        </h3>
        {results.data && results.data.length > 0 && (
          <div className="card overflow-hidden p-0">
            <table className="w-full text-sm">
              <thead className="bg-white text-ink-500 dark:bg-ink-900 border-b border-ink-200 dark:border-ink-800">
                <tr>
                  <th className="px-4 py-2 text-left font-medium">When</th>
                  <th className="px-4 py-2 text-right font-medium">Stat</th>
                  <th className="px-4 py-2 text-right font-medium">p-value</th>
                  <th className="px-4 py-2 text-right font-medium">
                    Duration
                  </th>
                </tr>
              </thead>
              <tbody>
                {results.data.map((r) => (
                  <tr
                    key={r.run_id}
                    className="border-t border-ink-200 dark:border-ink-800"
                  >
                    <td className="px-4 py-2 text-xs text-ink-500">
                      {r.started_at
                        ? new Date(r.started_at).toLocaleString()
                        : '—'}
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums">
                      {r.metrics?.test_statistic !== null && r.metrics?.test_statistic !== undefined
                        ? r.metrics.test_statistic.toFixed(4)
                        : '—'}
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums">
                      {r.metrics?.p_value !== null && r.metrics?.p_value !== undefined
                        ? r.metrics.p_value.toExponential(2)
                        : '—'}
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums">
                      {r.duration_ms ? `${Math.round(r.duration_ms)}ms` : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

// ─────────────────────────────────────────── result card

function AnalysisResultCard({ result }: { result: AnalysisResult }) {
  return (
    <div className="card space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat label="test statistic" value={result.test_statistic} />
        <Stat
          label="p-value"
          value={result.p_value}
          format={(v) => (v < 0.001 ? v.toExponential(2) : v.toFixed(4))}
        />
        <Stat
          label={result.effect_size_name ?? 'effect size'}
          value={result.effect_size}
        />
        <Stat
          label="95% CI"
          value={
            result.ci_low !== null && result.ci_high !== null
              ? `${result.ci_low.toFixed(3)} … ${result.ci_high.toFixed(3)}`
              : null
          }
        />
      </div>
      {result.interpretation && (
        <p className="text-sm border-l-2 border-accent-500 pl-3 text-ink-700 dark:text-ink-300">
          {result.interpretation}
        </p>
      )}
      {result.figure && (
        <div className="rounded-lg border border-ink-200 dark:border-ink-800 p-2">
          <PlotlyFigure figure={result.figure} />
        </div>
      )}
      {result.table.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-ink-500">
              <tr>
                {Object.keys(result.table[0]).map((c) => (
                  <th key={c} className="px-2 py-1 text-left font-medium">
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.table.map((row, i) => (
                <tr key={i} className="border-t border-ink-200 dark:border-ink-800">
                  {Object.keys(result.table[0]).map((c) => (
                    <td key={c} className="px-2 py-1 tabular-nums">
                      {formatCell(row[c])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
  format,
}: {
  label: string;
  value: number | string | null;
  format?: (v: number) => string;
}) {
  return (
    <div>
      <p className="muted small">{label}</p>
      <p className="text-base font-semibold tabular-nums mt-1">
        {value === null || value === undefined
          ? '—'
          : typeof value === 'number'
            ? format
              ? format(value)
              : Number.isFinite(value)
                ? value.toFixed(4)
                : '—'
            : value}
      </p>
    </div>
  );
}

function formatCell(v: unknown): string {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number') {
    return Number.isFinite(v) ? v.toFixed(4) : '—';
  }
  return String(v);
}
