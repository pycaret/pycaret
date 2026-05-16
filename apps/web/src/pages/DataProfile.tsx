/**
 * Data Profile / EDA dashboard — `/workspaces/:wsId/datasets/:dataSourceId/profile`.
 *
 * One-stop interactive overview of any uploaded CSV. The whole dashboard is
 * fed by a single `GET /data-sources/:id/profile` call (rich JSON: shape +
 * per-column stats + histograms + correlations + warnings + sample rows),
 * so initial render is one round-trip and all tabs are instant after that.
 *
 * Layout:
 *   [breadcrumb · h1 · upload date]
 *   [KPI strip: Rows · Cols · Missing% · Duplicates · Memory · Numeric · Categorical · DateTime]
 *   [Tabs: Overview | Columns | Correlations | Missing | Quality | Sample]
 *
 * Each tab is self-contained — Overview shows the at-a-glance summary,
 * Columns is the deep-dive (search + per-column histograms / value counts),
 * Correlations shows the heatmap, Missing breaks down null patterns, Quality
 * surfaces warnings, Sample shows the head() with type chips.
 */

import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import Plot from 'react-plotly.js';
import { dataSourcesApi, workspacesApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import { BackButton } from '@/components/BackButton';
import type {
  DatasetColumn,
  DatasetColumnKind,
  DatasetProfile,
} from '@/api/endpoints';

type Tab = 'overview' | 'columns' | 'correlations' | 'missing' | 'quality' | 'sample';

// ─── Helpers ────────────────────────────────────────────────────

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function formatNumber(n: number, digits = 4): string {
  if (!Number.isFinite(n)) return '—';
  if (Math.abs(n) >= 10000) return n.toLocaleString();
  if (Number.isInteger(n)) return String(n);
  return n.toFixed(digits);
}

function formatPct(n: number): string {
  if (!Number.isFinite(n)) return '—';
  if (n === 0) return '0%';
  if (n < 0.001) return `<0.1%`;
  return `${(n * 100).toFixed(n < 0.01 ? 2 : 1)}%`;
}

const KIND_STYLE: Record<DatasetColumnKind, { chip: string; dot: string; label: string }> = {
  numeric: {
    chip: 'bg-sky-50 text-sky-700 dark:bg-sky-500/15 dark:text-sky-300',
    dot: 'bg-sky-500',
    label: 'numeric',
  },
  categorical: {
    chip: 'bg-violet-50 text-violet-700 dark:bg-violet-500/15 dark:text-violet-300',
    dot: 'bg-violet-500',
    label: 'categorical',
  },
  datetime: {
    chip: 'bg-cyan-50 text-cyan-700 dark:bg-cyan-500/15 dark:text-cyan-300',
    dot: 'bg-cyan-500',
    label: 'datetime',
  },
  boolean: {
    chip: 'bg-amber-50 text-amber-700 dark:bg-amber-500/15 dark:text-amber-300',
    dot: 'bg-amber-500',
    label: 'boolean',
  },
  text: {
    chip: 'bg-fuchsia-50 text-fuchsia-700 dark:bg-fuchsia-500/15 dark:text-fuchsia-300',
    dot: 'bg-fuchsia-500',
    label: 'text',
  },
};

const SEVERITY_STYLE: Record<string, string> = {
  info: 'bg-sky-50 text-sky-700 dark:bg-sky-500/15 dark:text-sky-300',
  warn: 'bg-amber-50 text-amber-700 dark:bg-amber-500/15 dark:text-amber-300',
  error: 'bg-danger-50 text-danger-700 dark:bg-danger-500/15 dark:text-danger-300',
};

// ─── Page ───────────────────────────────────────────────────────

export function DataProfile() {
  const { wsId = '', dataSourceId = '' } = useParams<{
    wsId: string;
    dataSourceId: string;
  }>();
  const [tab, setTab] = useState<Tab>('overview');

  const ds = useQuery({
    queryKey: ['datasets', dataSourceId],
    queryFn: () => dataSourcesApi.get(dataSourceId),
    enabled: !!dataSourceId,
  });

  const ws = useQuery({
    queryKey: ['workspaces', wsId],
    queryFn: () => workspacesApi.get(wsId),
    enabled: !!wsId,
    staleTime: 5 * 60 * 1000,
  });

  const profile = useQuery({
    queryKey: ['datasets', dataSourceId, 'profile'],
    queryFn: () => dataSourcesApi.profile(dataSourceId, 25),
    enabled: !!dataSourceId,
    staleTime: 60_000,
    retry: false,
  });

  const tabs: { id: Tab; label: string; count?: number }[] = [
    { id: 'overview', label: 'Overview' },
    { id: 'columns', label: 'Columns', count: profile.data?.columns.length },
    {
      id: 'correlations',
      label: 'Correlations',
      count: profile.data?.correlations?.columns.length,
    },
    { id: 'missing', label: 'Missing' },
    { id: 'quality', label: 'Quality', count: profile.data?.warnings.length },
    { id: 'sample', label: 'Sample' },
  ];

  return (
    <div className="space-y-8">
      <header>
        <BackButton />
        <nav className="text-xs text-ink-500 mb-2 flex items-center flex-wrap gap-y-0.5">
          <Link to="/" className="hover:text-ink-900 dark:hover:text-ink-50">
            Workspaces
          </Link>
          {wsId && (
            <>
              <span className="mx-1.5 text-ink-300">/</span>
              <Link
                to={`/workspaces/${wsId}`}
                className="hover:text-ink-900 dark:hover:text-ink-50"
              >
                {ws.data?.name ?? wsId.slice(0, 8)}
              </Link>
            </>
          )}
          <span className="mx-1.5 text-ink-300">/</span>
          <span className="text-ink-700 dark:text-ink-300">Data profile</span>
        </nav>
        <div className="flex items-end justify-between gap-4 flex-wrap">
          <div>
            <h1 className="h-page">{ds.data?.name ?? 'Data profile'}</h1>
            {ds.data?.description && (
              <p className="mt-1 text-sm text-ink-500">{ds.data.description}</p>
            )}
          </div>
        </div>
      </header>

      {profile.isPending && (
        <div className="card text-sm text-ink-500">Profiling dataset…</div>
      )}
      {profile.error && (
        <div className="card text-sm text-danger-600">
          {errorMessage(profile.error)}
        </div>
      )}

      {profile.data && (
        <>
          <KpiStrip profile={profile.data} />

          <section>
            <div className="border-b border-ink-200 dark:border-ink-800 mb-6">
              <nav className="flex gap-6 -mb-px overflow-x-auto" aria-label="EDA sections">
                {tabs.map((tt) => (
                  <button
                    key={tt.id}
                    type="button"
                    onClick={() => setTab(tt.id)}
                    className={`relative pb-3 text-sm font-medium whitespace-nowrap transition-colors ${
                      tab === tt.id
                        ? 'text-ink-900 dark:text-ink-50'
                        : 'text-ink-500 hover:text-ink-700 dark:hover:text-ink-300'
                    }`}
                  >
                    {tt.label}
                    {typeof tt.count === 'number' && (
                      <span
                        className={`ml-1.5 text-xs ${
                          tab === tt.id ? 'text-ink-500' : 'text-ink-400'
                        }`}
                      >
                        ({tt.count})
                      </span>
                    )}
                    {tab === tt.id && (
                      <span className="absolute -bottom-px left-0 right-0 h-0.5 bg-accent-500" />
                    )}
                  </button>
                ))}
              </nav>
            </div>

            {tab === 'overview' && <OverviewTab profile={profile.data} onSeeColumns={() => setTab('columns')} />}
            {tab === 'columns' && <ColumnsTab profile={profile.data} />}
            {tab === 'correlations' && <CorrelationsTab profile={profile.data} />}
            {tab === 'missing' && <MissingTab profile={profile.data} />}
            {tab === 'quality' && <QualityTab profile={profile.data} />}
            {tab === 'sample' && <SampleTab profile={profile.data} />}
          </section>
        </>
      )}
    </div>
  );
}

// ─── KPI strip ───────────────────────────────────────────────

function KpiStrip({ profile }: { profile: DatasetProfile }) {
  const tc = profile.type_counts;
  const tiles: { label: string; value: string; hint?: string }[] = [
    { label: 'Rows', value: profile.shape.rows.toLocaleString() },
    { label: 'Columns', value: profile.shape.cols.toLocaleString() },
    {
      label: 'Missing',
      value: formatPct(profile.missing_pct),
      hint: `${profile.missing_total.toLocaleString()} cells`,
    },
    {
      label: 'Duplicates',
      value: profile.duplicates.toLocaleString(),
      hint: profile.duplicates ? 'rows' : 'clean',
    },
    { label: 'Memory', value: formatBytes(profile.memory_bytes) },
    { label: 'Numeric', value: String(tc.numeric ?? 0) },
    { label: 'Categorical', value: String((tc.categorical ?? 0) + (tc.text ?? 0)) },
    { label: 'Datetime', value: String(tc.datetime ?? 0) },
  ];
  return (
    <div
      className="grid gap-3"
      style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))' }}
    >
      {tiles.map((t) => (
        <div
          key={t.label}
          className="rounded-xl border border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 px-4 py-3"
        >
          <div className="text-xs font-medium text-ink-500 uppercase tracking-wider">
            {t.label}
          </div>
          <div className="mt-1 text-2xl font-semibold text-ink-900 dark:text-ink-50 tabular-nums">
            {t.value}
          </div>
          {t.hint && <div className="mt-0.5 text-[10px] text-ink-400">{t.hint}</div>}
        </div>
      ))}
    </div>
  );
}

// ─── Overview tab ────────────────────────────────────────────

function OverviewTab({
  profile,
  onSeeColumns,
}: {
  profile: DatasetProfile;
  onSeeColumns: () => void;
}) {
  const totalCols = profile.shape.cols;
  const tc = profile.type_counts;
  const typeRows: { kind: DatasetColumnKind; count: number; pct: number }[] = (
    [
      { kind: 'numeric', count: tc.numeric ?? 0, pct: 0 },
      { kind: 'categorical', count: tc.categorical ?? 0, pct: 0 },
      { kind: 'text', count: tc.text ?? 0, pct: 0 },
      { kind: 'datetime', count: tc.datetime ?? 0, pct: 0 },
      { kind: 'boolean', count: tc.boolean ?? 0, pct: 0 },
    ] as { kind: DatasetColumnKind; count: number; pct: number }[]
  )
    .map((r) => ({ ...r, pct: r.count / Math.max(1, totalCols) }))
    .filter((r) => r.count > 0);

  const topMissing = useMemo(
    () =>
      [...profile.columns]
        .filter((c) => c.missing > 0)
        .sort((a, b) => b.missing_pct - a.missing_pct)
        .slice(0, 5),
    [profile.columns],
  );

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      <section className="lg:col-span-2 space-y-4">
        <div className="card">
          <h3 className="text-sm font-semibold text-ink-900 dark:text-ink-50 mb-3">
            Column types
          </h3>
          <div className="space-y-2.5">
            {typeRows.map((r) => (
              <div key={r.kind} className="flex items-center gap-3">
                <span className={`inline-block h-2 w-2 rounded-full ${KIND_STYLE[r.kind].dot}`} />
                <span className="text-sm text-ink-900 dark:text-ink-50 capitalize w-28 flex-shrink-0">
                  {KIND_STYLE[r.kind].label}
                </span>
                <div className="flex-1 h-2 rounded-full bg-ink-100 dark:bg-ink-800 overflow-hidden">
                  <div
                    className={`h-full ${KIND_STYLE[r.kind].dot}`}
                    style={{ width: `${Math.max(2, r.pct * 100)}%` }}
                  />
                </div>
                <span className="text-xs font-mono text-ink-500 tabular-nums w-20 text-right">
                  {r.count} · {formatPct(r.pct)}
                </span>
              </div>
            ))}
          </div>
        </div>

        {topMissing.length > 0 && (
          <div className="card">
            <div className="flex items-baseline justify-between mb-3">
              <h3 className="text-sm font-semibold text-ink-900 dark:text-ink-50">
                Top columns by missing
              </h3>
              <button
                type="button"
                onClick={onSeeColumns}
                className="text-xs text-accent-600 hover:underline"
              >
                See all →
              </button>
            </div>
            <div className="space-y-2.5">
              {topMissing.map((c) => (
                <div key={c.name} className="flex items-center gap-3">
                  <span className="text-sm font-mono text-ink-700 dark:text-ink-300 truncate w-40 flex-shrink-0">
                    {c.name}
                  </span>
                  <div className="flex-1 h-2 rounded-full bg-ink-100 dark:bg-ink-800 overflow-hidden">
                    <div
                      className="h-full bg-amber-500"
                      style={{ width: `${Math.max(2, c.missing_pct * 100)}%` }}
                    />
                  </div>
                  <span className="text-xs font-mono text-ink-500 tabular-nums w-24 text-right">
                    {c.missing.toLocaleString()} · {formatPct(c.missing_pct)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </section>

      <section className="space-y-4">
        <WarningsCard warnings={profile.warnings} compact />
        {profile.correlations && profile.correlations.columns.length >= 2 && (
          <TopCorrelations correlations={profile.correlations} max={5} />
        )}
      </section>
    </div>
  );
}

// ─── Columns tab ─────────────────────────────────────────────

function ColumnsTab({ profile }: { profile: DatasetProfile }) {
  const [search, setSearch] = useState('');
  const [selected, setSelected] = useState<string>(profile.columns[0]?.name ?? '');
  const [kindFilter, setKindFilter] = useState<DatasetColumnKind | 'all'>('all');

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    return profile.columns.filter((c) => {
      if (kindFilter !== 'all' && c.kind !== kindFilter) return false;
      if (!q) return true;
      return c.name.toLowerCase().includes(q);
    });
  }, [profile.columns, search, kindFilter]);

  const current = profile.columns.find((c) => c.name === selected) ?? profile.columns[0];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-4">
      <aside className="card p-0 overflow-hidden flex flex-col" style={{ maxHeight: '70vh' }}>
        <div className="p-3 border-b border-ink-200 dark:border-ink-800 space-y-2">
          <input
            className="input text-sm"
            placeholder="Search columns…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <div className="flex flex-wrap gap-1">
            {(['all', 'numeric', 'categorical', 'text', 'datetime', 'boolean'] as const).map(
              (k) => (
                <button
                  key={k}
                  type="button"
                  onClick={() => setKindFilter(k)}
                  className={`px-2 py-0.5 rounded-full text-[11px] font-medium ${
                    kindFilter === k
                      ? 'bg-ink-900 text-white dark:bg-white dark:text-ink-900'
                      : 'bg-ink-100 text-ink-600 dark:bg-ink-800 dark:text-ink-400'
                  }`}
                >
                  {k === 'all' ? 'All' : KIND_STYLE[k].label}
                </button>
              ),
            )}
          </div>
        </div>
        <ul className="overflow-y-auto flex-1 divide-y divide-ink-200 dark:divide-ink-800">
          {filtered.map((c) => (
            <li key={c.name}>
              <button
                type="button"
                onClick={() => setSelected(c.name)}
                className={`w-full text-left px-3 py-2 hover:bg-ink-50 dark:hover:bg-ink-950/40 ${
                  selected === c.name
                    ? 'bg-ink-50 dark:bg-ink-950/40'
                    : ''
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-sm font-mono text-ink-900 dark:text-ink-50 truncate">
                    {c.name}
                  </span>
                  <span className={`pill text-[10px] ${KIND_STYLE[c.kind].chip}`}>
                    {KIND_STYLE[c.kind].label}
                  </span>
                </div>
                <div className="text-[11px] text-ink-500 mt-0.5">
                  {c.unique.toLocaleString()} unique · {formatPct(c.missing_pct)} missing
                </div>
              </button>
            </li>
          ))}
          {filtered.length === 0 && (
            <li className="px-3 py-8 text-center text-xs text-ink-500">
              No columns match.
            </li>
          )}
        </ul>
      </aside>

      <main>
        {current ? <ColumnDetail column={current} totalRows={profile.shape.rows} /> : null}
      </main>
    </div>
  );
}

function ColumnDetail({
  column,
  totalRows,
}: {
  column: DatasetColumn;
  totalRows: number;
}) {
  const stats = column.stats ?? {};
  const isNumeric = column.kind === 'numeric';
  const isCategorical = column.kind === 'categorical' || column.kind === 'text' || column.kind === 'boolean';

  const kpis: { label: string; value: string }[] = [
    { label: 'Type', value: `${KIND_STYLE[column.kind].label} (${column.dtype})` },
    {
      label: 'Missing',
      value: `${column.missing.toLocaleString()} (${formatPct(column.missing_pct)})`,
    },
    {
      label: 'Unique',
      value: `${column.unique.toLocaleString()} (${formatPct(column.cardinality_pct)})`,
    },
    {
      label: 'Filled',
      value: `${(totalRows - column.missing).toLocaleString()} of ${totalRows.toLocaleString()}`,
    },
  ];

  return (
    <div className="space-y-4">
      <div className="card">
        <div className="flex items-baseline justify-between gap-2 flex-wrap">
          <div>
            <h2 className="text-base font-semibold text-ink-900 dark:text-ink-50 font-mono">
              {column.name}
            </h2>
            <p className="text-xs text-ink-500 mt-0.5">
              {column.is_id_like && (
                <span className="pill-warn mr-1">ID-like</span>
              )}
              {column.is_constant && (
                <span className="pill-warn mr-1">Constant</span>
              )}
              <span className={`pill ${KIND_STYLE[column.kind].chip}`}>
                {KIND_STYLE[column.kind].label}
              </span>
            </p>
          </div>
        </div>
        <div
          className="mt-4 grid gap-2"
          style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))' }}
        >
          {kpis.map((k) => (
            <div
              key={k.label}
              className="rounded-md border border-ink-200 dark:border-ink-800 px-3 py-2"
            >
              <div className="text-[10px] uppercase tracking-wide text-ink-500">
                {k.label}
              </div>
              <div className="text-sm font-mono text-ink-900 dark:text-ink-50 mt-0.5">
                {k.value}
              </div>
            </div>
          ))}
        </div>
      </div>

      {isNumeric && column.histogram.length > 0 && (
        <div className="card">
          <h3 className="text-sm font-semibold text-ink-900 dark:text-ink-50 mb-2">
            Distribution
          </h3>
          <ColumnHistogram column={column} />
        </div>
      )}

      {isNumeric && column.stats && (
        <div className="card">
          <h3 className="text-sm font-semibold text-ink-900 dark:text-ink-50 mb-3">
            Statistics
          </h3>
          <div
            className="grid gap-3"
            style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))' }}
          >
            {(
              [
                ['min', 'Min'],
                ['q1', 'Q1'],
                ['median', 'Median'],
                ['q3', 'Q3'],
                ['max', 'Max'],
                ['mean', 'Mean'],
                ['std', 'Std'],
                ['skew', 'Skew'],
                ['kurtosis', 'Kurtosis'],
                ['zeros_pct', 'Zeros'],
                ['outliers_iqr', 'Outliers (IQR)'],
              ] as const
            ).map(([k, label]) => {
              const v = stats[k];
              if (v == null) return null;
              const display =
                k === 'zeros_pct' && typeof v === 'number'
                  ? formatPct(v)
                  : typeof v === 'number'
                    ? formatNumber(v)
                    : String(v);
              return (
                <div
                  key={k}
                  className="rounded-md border border-ink-200 dark:border-ink-800 px-3 py-2"
                >
                  <div className="text-[10px] uppercase tracking-wide text-ink-500">
                    {label}
                  </div>
                  <div className="text-sm font-mono text-ink-900 dark:text-ink-50 mt-0.5">
                    {display}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {isCategorical && column.top_values.length > 0 && (
        <div className="card">
          <h3 className="text-sm font-semibold text-ink-900 dark:text-ink-50 mb-3">
            Top values
          </h3>
          <div className="space-y-2">
            {column.top_values.map((v) => (
              <div key={v.value} className="flex items-center gap-3">
                <span className="text-sm font-mono text-ink-700 dark:text-ink-300 truncate w-40 flex-shrink-0">
                  {v.value}
                </span>
                <div className="flex-1 h-2 rounded-full bg-ink-100 dark:bg-ink-800 overflow-hidden">
                  <div
                    className={`h-full ${KIND_STYLE[column.kind].dot}`}
                    style={{ width: `${Math.max(2, v.pct * 100)}%` }}
                  />
                </div>
                <span className="text-xs font-mono text-ink-500 tabular-nums w-24 text-right">
                  {v.count.toLocaleString()} · {formatPct(v.pct)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {column.sample_values.length > 0 && (
        <div className="card">
          <h3 className="text-sm font-semibold text-ink-900 dark:text-ink-50 mb-2">
            Sample values
          </h3>
          <div className="flex flex-wrap gap-1">
            {column.sample_values.map((v, i) => (
              <span
                key={i}
                className="inline-flex items-center px-1.5 py-0.5 rounded bg-ink-50 dark:bg-ink-950/40 border border-ink-200 dark:border-ink-800 text-[11px] font-mono text-ink-700 dark:text-ink-300"
              >
                {String(v)}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ColumnHistogram({ column }: { column: DatasetColumn }) {
  const x = column.histogram.map((b) => (b.bin_start + b.bin_end) / 2);
  const y = column.histogram.map((b) => b.count);
  const widths = column.histogram.map((b) => Math.max(0, b.bin_end - b.bin_start));
  return (
    <Plot
      data={[
        {
          type: 'bar',
          x,
          y,
          width: widths,
          marker: { color: 'rgb(56, 189, 248)' },
          hovertemplate: '%{y} rows<br>[%{x}]<extra></extra>',
        },
      ]}
      layout={{
        height: 240,
        margin: { l: 36, r: 12, t: 8, b: 32 },
        bargap: 0.04,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Inter, system-ui, sans-serif', size: 11, color: '#52525b' },
        xaxis: { gridcolor: '#e4e4e7', zerolinecolor: '#e4e4e7' },
        yaxis: { gridcolor: '#e4e4e7', zerolinecolor: '#e4e4e7' },
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%' }}
      useResizeHandler
    />
  );
}

// ─── Correlations tab ───────────────────────────────────────

function CorrelationsTab({ profile }: { profile: DatasetProfile }) {
  if (!profile.correlations || profile.correlations.columns.length < 2) {
    return (
      <p className="text-sm text-ink-500">
        Need at least two numeric columns to compute correlations.
      </p>
    );
  }
  const { columns, matrix } = profile.correlations;

  return (
    <div className="space-y-4">
      <div className="card">
        <h3 className="text-sm font-semibold text-ink-900 dark:text-ink-50 mb-3">
          Pearson correlation
        </h3>
        <Plot
          data={[
            {
              type: 'heatmap',
              x: columns,
              y: columns,
              z: matrix,
              colorscale: [
                [0, 'rgb(220, 38, 38)'],
                [0.5, 'rgb(245, 245, 245)'],
                [1, 'rgb(37, 99, 235)'],
              ],
              zmin: -1,
              zmax: 1,
              hoverongaps: false,
              hovertemplate: '%{y} × %{x}<br>r = %{z:.3f}<extra></extra>',
            },
          ]}
          layout={{
            height: Math.max(360, columns.length * 26 + 80),
            margin: { l: 120, r: 16, t: 12, b: 100 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { family: 'Inter, system-ui, sans-serif', size: 11, color: '#52525b' },
            xaxis: { tickangle: -45 },
            yaxis: { autorange: 'reversed' },
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%' }}
          useResizeHandler
        />
      </div>
      <TopCorrelations correlations={profile.correlations} max={20} />
    </div>
  );
}

function TopCorrelations({
  correlations,
  max,
}: {
  correlations: { columns: string[]; matrix: number[][] };
  max: number;
}) {
  const pairs = useMemo(() => {
    const out: { a: string; b: string; r: number }[] = [];
    const cols = correlations.columns;
    for (let i = 0; i < cols.length; i++) {
      for (let j = i + 1; j < cols.length; j++) {
        const r = correlations.matrix[i]?.[j] ?? 0;
        if (Math.abs(r) > 0.001) {
          out.push({ a: cols[i], b: cols[j], r });
        }
      }
    }
    return out.sort((p, q) => Math.abs(q.r) - Math.abs(p.r)).slice(0, max);
  }, [correlations, max]);

  if (pairs.length === 0) {
    return null;
  }

  return (
    <div className="card overflow-hidden p-0">
      <div className="px-4 py-3 border-b border-ink-200 dark:border-ink-800">
        <h3 className="text-sm font-semibold text-ink-900 dark:text-ink-50">
          Strongest correlations
        </h3>
      </div>
      <table className="w-full text-sm">
        <thead className="bg-white text-ink-500 dark:bg-ink-900 border-b border-ink-200 dark:border-ink-800">
          <tr>
            <th className="px-4 py-2 text-left font-medium">A</th>
            <th className="px-4 py-2 text-left font-medium">B</th>
            <th className="px-4 py-2 text-right font-medium tabular-nums">r</th>
          </tr>
        </thead>
        <tbody>
          {pairs.map((p) => (
            <tr
              key={`${p.a}__${p.b}`}
              className="border-t border-ink-200 dark:border-ink-800"
            >
              <td className="px-4 py-2 font-mono text-xs text-ink-700 dark:text-ink-300">
                {p.a}
              </td>
              <td className="px-4 py-2 font-mono text-xs text-ink-700 dark:text-ink-300">
                {p.b}
              </td>
              <td
                className={`px-4 py-2 text-right font-mono tabular-nums ${
                  p.r > 0
                    ? 'text-sky-600 dark:text-sky-400'
                    : 'text-rose-600 dark:text-rose-400'
                }`}
              >
                {p.r >= 0 ? '+' : ''}
                {p.r.toFixed(3)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── Missing tab ────────────────────────────────────────────

function MissingTab({ profile }: { profile: DatasetProfile }) {
  const sorted = useMemo(
    () => [...profile.columns].sort((a, b) => b.missing_pct - a.missing_pct),
    [profile.columns],
  );
  const withMissing = sorted.filter((c) => c.missing > 0);

  return (
    <div className="space-y-4">
      <div
        className="grid gap-3"
        style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))' }}
      >
        <Tile
          label="Total cells"
          value={(profile.shape.rows * profile.shape.cols).toLocaleString()}
        />
        <Tile label="Missing cells" value={profile.missing_total.toLocaleString()} />
        <Tile label="Missing %" value={formatPct(profile.missing_pct)} />
        <Tile
          label="Columns with missing"
          value={`${withMissing.length} of ${profile.shape.cols}`}
        />
      </div>
      {withMissing.length === 0 ? (
        <div className="card text-sm text-ink-500">
          No missing values across any column. Clean dataset.
        </div>
      ) : (
        <div className="card">
          <h3 className="text-sm font-semibold text-ink-900 dark:text-ink-50 mb-3">
            Missing rate by column
          </h3>
          <div className="space-y-2.5">
            {withMissing.map((c) => (
              <div key={c.name} className="flex items-center gap-3">
                <span className="text-sm font-mono text-ink-700 dark:text-ink-300 truncate w-48 flex-shrink-0">
                  {c.name}
                </span>
                <div className="flex-1 h-2 rounded-full bg-ink-100 dark:bg-ink-800 overflow-hidden">
                  <div
                    className="h-full bg-amber-500"
                    style={{ width: `${Math.max(2, c.missing_pct * 100)}%` }}
                  />
                </div>
                <span className="text-xs font-mono text-ink-500 tabular-nums w-28 text-right">
                  {c.missing.toLocaleString()} · {formatPct(c.missing_pct)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Tile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 px-4 py-3">
      <div className="text-xs font-medium text-ink-500 uppercase tracking-wider">{label}</div>
      <div className="mt-1 text-2xl font-semibold text-ink-900 dark:text-ink-50 tabular-nums">
        {value}
      </div>
    </div>
  );
}

// ─── Quality tab ───────────────────────────────────────────

function QualityTab({ profile }: { profile: DatasetProfile }) {
  return <WarningsCard warnings={profile.warnings} compact={false} />;
}

function WarningsCard({
  warnings,
  compact,
}: {
  warnings: DatasetProfile['warnings'];
  compact: boolean;
}) {
  // React rules-of-hooks: hooks must run on every render in the same order.
  // Compute grouped BEFORE any conditional early return.
  const grouped = useMemo(() => {
    const out: Record<string, DatasetProfile['warnings']> = {};
    for (const w of warnings) {
      const key = w.severity;
      (out[key] ??= []).push(w);
    }
    return out;
  }, [warnings]);
  if (warnings.length === 0) {
    return (
      <div className="card text-sm text-ink-500">
        No quality issues detected. Dataset looks clean.
      </div>
    );
  }
  const order: ('error' | 'warn' | 'info')[] = ['error', 'warn', 'info'];

  return (
    <div className="space-y-3">
      {order.map((sev) => {
        const list = grouped[sev];
        if (!list || list.length === 0) return null;
        return (
          <div key={sev} className="card p-0 overflow-hidden">
            <div className="px-4 py-2.5 border-b border-ink-200 dark:border-ink-800 flex items-center gap-2">
              <span className={`pill ${SEVERITY_STYLE[sev]} capitalize`}>{sev}</span>
              <span className="text-xs text-ink-500">
                {list.length} issue{list.length === 1 ? '' : 's'}
              </span>
            </div>
            <ul className="divide-y divide-ink-200 dark:divide-ink-800">
              {(compact ? list.slice(0, 5) : list).map((w, i) => (
                <li key={i} className="px-4 py-2.5 flex items-baseline gap-3">
                  {w.column && (
                    <span className="text-xs font-mono text-ink-500 truncate flex-shrink-0 max-w-[10rem]">
                      {w.column}
                    </span>
                  )}
                  <span className="text-sm text-ink-900 dark:text-ink-50 flex-1">
                    {w.message}
                  </span>
                  <span className="text-[10px] uppercase tracking-wide text-ink-400 flex-shrink-0">
                    {w.kind.replace(/_/g, ' ')}
                  </span>
                </li>
              ))}
              {compact && list.length > 5 && (
                <li className="px-4 py-2 text-[11px] text-ink-500">
                  + {list.length - 5} more
                </li>
              )}
            </ul>
          </div>
        );
      })}
    </div>
  );
}

// ─── Sample tab ────────────────────────────────────────────

function SampleTab({ profile }: { profile: DatasetProfile }) {
  const cols = profile.columns;
  const rows = profile.sample;
  if (rows.length === 0) {
    return <p className="text-sm text-ink-500">No sample rows available.</p>;
  }
  return (
    <div className="card overflow-x-auto p-0">
      <table className="w-full text-sm">
        <thead className="bg-white text-ink-500 dark:bg-ink-900 border-b border-ink-200 dark:border-ink-800">
          <tr>
            <th className="px-3 py-2 text-left font-medium w-12 sticky left-0 bg-white dark:bg-ink-900">
              #
            </th>
            {cols.map((c) => (
              <th
                key={c.name}
                className="px-3 py-2 text-left font-medium whitespace-nowrap"
              >
                <div className="text-ink-900 dark:text-ink-50 font-mono text-xs">
                  {c.name}
                </div>
                <div className="mt-0.5">
                  <span className={`pill text-[10px] ${KIND_STYLE[c.kind].chip}`}>
                    {KIND_STYLE[c.kind].label}
                  </span>
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={i}
              className="border-t border-ink-200 dark:border-ink-800"
            >
              <td className="px-3 py-2 text-xs text-ink-500 tabular-nums sticky left-0 bg-white dark:bg-ink-900">
                {i + 1}
              </td>
              {cols.map((c) => {
                const v = row[c.name];
                return (
                  <td
                    key={c.name}
                    className="px-3 py-2 font-mono text-xs text-ink-900 dark:text-ink-50 whitespace-nowrap"
                  >
                    {v == null ? (
                      <span className="text-ink-400 italic">null</span>
                    ) : (
                      String(v)
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

