/**
 * DatasetExploreModal — fast, interactive EDA modal.
 *
 * Opens when the user clicks a dataset row body in DataSourcesSection.
 * One API call (GET /data-sources/{id}/profile?sample_rows=50) drives
 * the whole experience.
 *
 * Layout (viewport-bounded — left + right both scroll internally):
 *
 *   ┌──────────────────────────────────────────────────────────────┐
 *   │ Health pills (rows · cols · memory · dups · missing · types) │
 *   │ Heuristic chips (click → jump to column)                     │
 *   ├──────────────────────────┬───────────────────────────────────┤
 *   │ Sample (50)              │ [Columns] [Overview] [Correlations]│
 *   │  scroll x+y internally   │                                   │
 *   │  click header → select   │ Columns tab:                      │
 *   │  selected col highlights │   filter input                    │
 *   │                          │   column list (scrollable)        │
 *   │                          │     - dtype badge                 │
 *   │                          │     - "looks-like" tags           │
 *   │                          │     - sparkline (numeric)         │
 *   │                          │     - missing-% bar               │
 *   │                          │   selected column detail (scroll) │
 *   │                          │     - stats grid                  │
 *   │                          │     - histogram OR top-values     │
 *   │                          │     - top-5 correlated (numeric)  │
 *   │                          │                                   │
 *   │                          │ Overview tab:                     │
 *   │                          │   - dtype distribution bar        │
 *   │                          │   - top-N missing columns         │
 *   │                          │   - all warnings (full messages)  │
 *   │                          │                                   │
 *   │                          │ Correlations tab:                 │
 *   │                          │   - top-K strongest pairs         │
 *   │                          │   - mini heatmap (SVG)            │
 *   └──────────────────────────┴───────────────────────────────────┘
 *
 * Bonus ideas baked in:
 *   - "Target candidate" badge for low-cardinality numeric/boolean.
 *   - "ID-like" / "constant" / "high-cardinality" badges.
 *   - Click a correlated-column row → jumps the picker to that column.
 *   - Mini sparkline inline in the picker for numeric cols.
 *   - Column search box for navigating wide tables.
 */

import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Dialog } from './Dialog';
import { dataSourcesApi, type DatasetColumn, type DatasetProfile } from '@/api/endpoints';

export interface DatasetExploreModalProps {
  dataSourceId: string;
  dataSourceName: string;
  open: boolean;
  onClose: () => void;
}

const KIND_PALETTE: Record<string, string> = {
  numeric: 'bg-sky-500/15 text-sky-700 dark:text-sky-300',
  categorical: 'bg-violet-500/15 text-violet-700 dark:text-violet-300',
  datetime: 'bg-amber-500/15 text-amber-700 dark:text-amber-300',
  boolean: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300',
  text: 'bg-ink-500/15 text-ink-700 dark:text-ink-300',
};

function fmtBytes(n?: number | null): string {
  if (n == null) return '—';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} kB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function fmtNum(n: number | string | null | undefined): string {
  if (n == null) return '—';
  if (typeof n === 'string') return n;
  if (!Number.isFinite(n)) return '—';
  if (Math.abs(n) >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
  if (Math.abs(n) >= 1) return n.toLocaleString(undefined, { maximumFractionDigits: 4 });
  return n.toLocaleString(undefined, { maximumSignificantDigits: 4 });
}

// Heuristic role hints — informational, not authoritative. The engine
// doesn't consume these yet; they're surfaced to help the user pick a
// target / drop ID columns / catch leakage before they configure a run.
function roleHints(c: DatasetColumn, rowCount: number): Array<{ label: string; tone: string; title: string }> {
  const out: Array<{ label: string; tone: string; title: string }> = [];
  if (c.is_constant) {
    out.push({ label: 'drop?', tone: 'amber', title: 'Constant column — every value is the same.' });
  }
  if (c.is_id_like) {
    out.push({ label: 'id', tone: 'amber', title: 'ID-like — unique per row, almost certainly not a feature.' });
  }
  if (c.kind === 'boolean' || (c.kind === 'numeric' && c.unique > 0 && c.unique <= 10)) {
    out.push({ label: 'target?', tone: 'sky', title: 'Low-cardinality numeric / boolean — common target shape.' });
  }
  if (c.kind === 'categorical' && c.unique > 50 && c.cardinality_pct > 80) {
    out.push({ label: 'leak?', tone: 'rose', title: 'High-cardinality categorical close to row count — possible leakage.' });
  }
  if (c.missing_pct > 50) {
    out.push({ label: 'sparse', tone: 'amber', title: `${c.missing_pct.toFixed(0)}% missing.` });
  }
  // unused param — keep signature stable for future heuristics
  void rowCount;
  return out;
}

const HINT_TONE: Record<string, string> = {
  amber: 'bg-amber-500/15 text-amber-700 dark:text-amber-300',
  sky: 'bg-sky-500/15 text-sky-700 dark:text-sky-300',
  rose: 'bg-rose-500/15 text-rose-700 dark:text-rose-300',
};

export function DatasetExploreModal({
  dataSourceId,
  dataSourceName,
  open,
  onClose,
}: DatasetExploreModalProps) {
  const profile = useQuery({
    queryKey: ['data-source', dataSourceId, 'profile-explore'],
    queryFn: () => dataSourcesApi.profile(dataSourceId, 50),
    enabled: open && !!dataSourceId,
    staleTime: 30_000,
  });

  return (
    <Dialog
      open={open}
      onClose={onClose}
      size="full"
      noBodyPadding
      title={dataSourceName}
      description="Interactive exploration — single-call EDA over the latest snapshot."
    >
      <div className="flex-1 min-h-0 flex flex-col">
        {profile.isLoading && (
          <div className="flex-1 grid place-items-center text-sm text-ink-500">
            Loading profile…
          </div>
        )}
        {profile.error && (
          <div className="flex-1 grid place-items-center text-sm text-danger-600">
            Profile failed: {(profile.error as Error).message}
          </div>
        )}
        {profile.data && <ExploreBody profile={profile.data} />}
      </div>
    </Dialog>
  );
}

// ─── Body ────────────────────────────────────────────────────────────

type RightTab = 'columns' | 'overview' | 'correlations';

function ExploreBody({ profile }: { profile: DatasetProfile }) {
  const [selectedCol, setSelectedCol] = useState<string | null>(
    profile.columns[0]?.name ?? null,
  );
  const [tab, setTab] = useState<RightTab>('columns');

  const selected = profile.columns.find((c) => c.name === selectedCol) ?? null;

  return (
    <div className="flex-1 min-h-0 flex flex-col">
      {/* Top header row — health + warnings */}
      <div className="px-6 pt-3 pb-3 space-y-2 shrink-0 border-b border-ink-100 dark:border-ink-800">
        <HealthStrip profile={profile} />
        <WarningsStrip
          warnings={profile.warnings}
          onJump={(c) => {
            setTab('columns');
            setSelectedCol(c);
          }}
        />
      </div>

      {/* Body — flex split (not grid; grid rows default to content-height
          which lets children grow past the parent). 7/5 width split with
          each pane being its own flex column that scrolls internally. */}
      <div className="flex-1 min-h-0 flex">
        {/* Left — sample rows */}
        <section className="flex flex-col min-w-0 min-h-0 border-r border-ink-100 dark:border-ink-800" style={{ flex: '7 1 0%' }}>
          <div className="px-4 pt-2 pb-1 shrink-0 flex items-baseline justify-between">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-ink-500">
              Sample rows ({profile.sample.length})
            </h3>
            {selected && (
              <span className="text-[11px] text-ink-500">
                Click a column header to inspect
              </span>
            )}
          </div>
          <div className="flex-1 overflow-auto px-4 pb-4">
            <table className="text-xs min-w-full">
              <thead className="bg-ink-50 dark:bg-ink-950/40 sticky top-0 z-10">
                <tr>
                  {profile.columns.map((c) => (
                    <th
                      key={c.name}
                      onClick={() => {
                        setTab('columns');
                        setSelectedCol(c.name);
                      }}
                      className={`px-2 py-1.5 text-left font-medium border-b border-ink-200 dark:border-ink-800 cursor-pointer hover:bg-ink-100 dark:hover:bg-ink-800 whitespace-nowrap ${
                        selectedCol === c.name
                          ? 'bg-accent-500/10 dark:bg-accent-500/20 text-accent-700 dark:text-accent-300'
                          : 'text-ink-700 dark:text-ink-300'
                      }`}
                      title={`${c.dtype} · ${c.kind}`}
                    >
                      {c.name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {profile.sample.map((row, i) => (
                  <tr key={i} className="border-t border-ink-100 dark:border-ink-900">
                    {profile.columns.map((c) => {
                      const v = (row as Record<string, unknown>)[c.name];
                      const isSel = selectedCol === c.name;
                      return (
                        <td
                          key={c.name}
                          className={`px-2 py-1 truncate max-w-[180px] ${
                            isSel
                              ? 'bg-accent-500/5 dark:bg-accent-500/10 text-ink-900 dark:text-ink-50'
                              : 'text-ink-700 dark:text-ink-300'
                          }`}
                          title={v == null ? 'null' : String(v)}
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
        </section>

        {/* Right — tabbed pane */}
        <section className="flex flex-col min-w-0 min-h-0" style={{ flex: '5 1 0%' }}>
          <div className="px-4 pt-2 shrink-0 flex items-center gap-1 border-b border-ink-100 dark:border-ink-800">
            <TabButton active={tab === 'columns'} onClick={() => setTab('columns')}>
              Columns
              <span className="ml-1 text-[10px] text-ink-400">({profile.columns.length})</span>
            </TabButton>
            <TabButton active={tab === 'overview'} onClick={() => setTab('overview')}>
              Overview
              {profile.warnings.length > 0 && (
                <span className="ml-1 text-[10px] px-1 rounded bg-amber-500/15 text-amber-700 dark:text-amber-300">
                  {profile.warnings.length}
                </span>
              )}
            </TabButton>
            <TabButton
              active={tab === 'correlations'}
              onClick={() => setTab('correlations')}
              disabled={!profile.correlations}
              title={!profile.correlations ? 'No numeric columns to correlate' : undefined}
            >
              Correlations
            </TabButton>
          </div>

          <div className="flex-1 min-h-0 flex flex-col">
            {tab === 'columns' && (
              <ColumnsTab
                profile={profile}
                selected={selected}
                onSelect={setSelectedCol}
              />
            )}
            {tab === 'overview' && <OverviewTab profile={profile} onJump={(c) => { setTab('columns'); setSelectedCol(c); }} />}
            {tab === 'correlations' && profile.correlations && (
              <CorrelationsTab
                corr={profile.correlations}
                onJump={(c) => { setTab('columns'); setSelectedCol(c); }}
              />
            )}
          </div>
        </section>
      </div>
    </div>
  );
}

// ─── Tab button primitive ──────────────────────────────────────────

function TabButton({
  active,
  onClick,
  disabled,
  title,
  children,
}: {
  active: boolean;
  onClick: () => void;
  disabled?: boolean;
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`px-3 py-2 text-xs font-medium transition-colors -mb-px border-b-2 ${
        disabled
          ? 'text-ink-300 dark:text-ink-700 cursor-not-allowed border-transparent'
          : active
            ? 'text-ink-900 dark:text-ink-50 border-accent-500'
            : 'text-ink-500 hover:text-ink-900 dark:hover:text-ink-50 border-transparent'
      }`}
    >
      {children}
    </button>
  );
}

// ─── Columns tab ────────────────────────────────────────────────────

function ColumnsTab({
  profile,
  selected,
  onSelect,
}: {
  profile: DatasetProfile;
  selected: DatasetColumn | null;
  onSelect: (name: string) => void;
}) {
  const [search, setSearch] = useState('');
  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return profile.columns;
    return profile.columns.filter((c) => c.name.toLowerCase().includes(q));
  }, [profile.columns, search]);

  // Single scroll container for the whole tab: filter + (short, capped)
  // column picker + selected-column detail flow inline. The picker has a
  // small internal scroll only because at 19+ columns the full list would
  // dominate; the detail card always sits directly below it.
  return (
    <div className="flex-1 min-h-0 overflow-y-auto">
      <div className="px-4 pt-2 pb-3 space-y-3">
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Filter columns…"
          className="input text-xs"
        />

        <div className="rounded-lg border border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 max-h-[180px] overflow-y-auto">
          <ul className="divide-y divide-ink-100 dark:divide-ink-900">
            {filtered.map((c) => (
              <li key={c.name}>
                <button
                  type="button"
                  onClick={() => onSelect(c.name)}
                  className={`w-full text-left px-3 py-1.5 flex items-center gap-2 hover:bg-ink-50 dark:hover:bg-ink-800 transition-colors ${
                    selected?.name === c.name
                      ? 'bg-accent-500/10 dark:bg-accent-500/20'
                      : ''
                  }`}
                >
                  <span
                    className={`text-[10px] uppercase font-semibold px-1.5 py-0.5 rounded shrink-0 ${
                      KIND_PALETTE[c.kind] ?? KIND_PALETTE.text
                    }`}
                  >
                    {c.kind.slice(0, 3)}
                  </span>
                  <span className="text-xs font-medium text-ink-900 dark:text-ink-50 truncate flex-1">
                    {c.name}
                  </span>
                  {c.kind === 'numeric' && c.histogram.length > 0 && (
                    <Sparkline bins={c.histogram} />
                  )}
                  <MissingBar pct={c.missing_pct} />
                </button>
              </li>
            ))}
            {filtered.length === 0 && (
              <li className="px-3 py-4 text-center text-xs text-ink-500">
                No columns match "{search}".
              </li>
            )}
          </ul>
        </div>

        {selected ? (
          <ColumnDetail column={selected} profile={profile} onJump={onSelect} />
        ) : (
          <div className="text-xs text-ink-500 text-center py-6">
            Select a column to inspect.
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Overview tab ───────────────────────────────────────────────────

function OverviewTab({
  profile,
  onJump,
}: {
  profile: DatasetProfile;
  onJump: (col: string) => void;
}) {
  const missingRanked = useMemo(
    () =>
      [...profile.columns]
        .filter((c) => c.missing > 0)
        .sort((a, b) => b.missing_pct - a.missing_pct)
        .slice(0, 8),
    [profile.columns],
  );

  const totalCols = profile.columns.length;
  const typeBars = Object.entries(profile.type_counts).map(([k, v]) => ({
    kind: k,
    count: v,
    pct: totalCols > 0 ? (v / totalCols) * 100 : 0,
  }));

  return (
    <div className="flex-1 min-h-0 overflow-y-auto px-4 py-3 space-y-4">
      <section>
        <h4 className="text-[10px] uppercase tracking-wider text-ink-500 mb-2">
          Type distribution
        </h4>
        <div className="rounded-lg border border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 p-3 space-y-1.5">
          {typeBars.map((t) => (
            <div key={t.kind} className="text-xs">
              <div className="flex items-center justify-between mb-0.5">
                <span
                  className={`text-[10px] uppercase font-semibold px-1.5 py-0.5 rounded ${
                    KIND_PALETTE[t.kind] ?? KIND_PALETTE.text
                  }`}
                >
                  {t.kind}
                </span>
                <span className="text-ink-500 tabular-nums">
                  {t.count} · {t.pct.toFixed(0)}%
                </span>
              </div>
              <div className="h-1.5 rounded bg-ink-100 dark:bg-ink-800 overflow-hidden">
                <div
                  className="h-full bg-accent-500/70"
                  style={{ width: `${t.pct}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h4 className="text-[10px] uppercase tracking-wider text-ink-500 mb-2">
          Top {missingRanked.length || 'none'} columns by missing %
        </h4>
        {missingRanked.length === 0 ? (
          <div className="rounded-lg border border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 p-3 text-xs text-ink-500">
            ✓ No missing values anywhere.
          </div>
        ) : (
          <div className="rounded-lg border border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 divide-y divide-ink-100 dark:divide-ink-900">
            {missingRanked.map((c) => (
              <button
                key={c.name}
                type="button"
                onClick={() => onJump(c.name)}
                className="w-full text-left px-3 py-1.5 hover:bg-ink-50 dark:hover:bg-ink-800 text-xs"
              >
                <div className="flex items-center justify-between mb-0.5">
                  <span className="font-medium text-ink-900 dark:text-ink-50 truncate">
                    {c.name}
                  </span>
                  <span className="text-ink-500 tabular-nums">
                    {c.missing.toLocaleString()} ({c.missing_pct.toFixed(1)}%)
                  </span>
                </div>
                <MissingBar pct={c.missing_pct} wide />
              </button>
            ))}
          </div>
        )}
      </section>

      <section>
        <h4 className="text-[10px] uppercase tracking-wider text-ink-500 mb-2">
          Heuristic warnings ({profile.warnings.length})
        </h4>
        {profile.warnings.length === 0 ? (
          <div className="rounded-lg border border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 p-3 text-xs text-ink-500">
            ✓ Nothing flagged.
          </div>
        ) : (
          <ul className="space-y-1.5">
            {profile.warnings.map((w, i) => {
              const tone =
                w.severity === 'error'
                  ? 'border-danger-300 bg-danger-50 dark:bg-danger-950/30 text-danger-700 dark:text-danger-300'
                  : w.severity === 'warn'
                    ? 'border-amber-300 bg-amber-50 dark:bg-amber-950/30 text-amber-800 dark:text-amber-200'
                    : 'border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 text-ink-700 dark:text-ink-300';
              return (
                <li
                  key={i}
                  className={`text-xs rounded-md border px-3 py-2 ${tone} ${
                    w.column ? 'cursor-pointer hover:opacity-90' : ''
                  }`}
                  onClick={() => w.column && onJump(w.column)}
                >
                  <div className="flex items-center justify-between mb-0.5">
                    <span className="font-semibold uppercase tracking-wider text-[10px]">
                      {w.kind}
                    </span>
                    {w.column && (
                      <code className="text-[10px] font-mono opacity-80">
                        {w.column}
                      </code>
                    )}
                  </div>
                  <p>{w.message}</p>
                </li>
              );
            })}
          </ul>
        )}
      </section>
    </div>
  );
}

// ─── Correlations tab ──────────────────────────────────────────────

function CorrelationsTab({
  corr,
  onJump,
}: {
  corr: NonNullable<DatasetProfile['correlations']>;
  onJump: (col: string) => void;
}) {
  const pairs = useMemo(() => {
    const out: Array<{ a: string; b: string; r: number }> = [];
    for (let i = 0; i < corr.columns.length; i++) {
      for (let j = i + 1; j < corr.columns.length; j++) {
        const r = corr.matrix[i][j];
        if (Number.isFinite(r)) {
          out.push({ a: corr.columns[i], b: corr.columns[j], r });
        }
      }
    }
    out.sort((x, y) => Math.abs(y.r) - Math.abs(x.r));
    return out.slice(0, 10);
  }, [corr]);

  // Mini heatmap config
  const N = corr.columns.length;
  const cellSize = Math.max(8, Math.min(20, Math.floor(280 / Math.max(N, 1))));
  const heatmapSize = cellSize * N;

  return (
    <div className="flex-1 min-h-0 overflow-y-auto px-4 py-3 space-y-4">
      <section>
        <h4 className="text-[10px] uppercase tracking-wider text-ink-500 mb-2">
          Top {pairs.length} strongest correlations
        </h4>
        <ul className="space-y-1.5">
          {pairs.map((p, i) => {
            const mag = Math.abs(p.r);
            const tone =
              mag > 0.7
                ? 'bg-emerald-500/80'
                : mag > 0.4
                  ? 'bg-emerald-500/50'
                  : 'bg-ink-400/60';
            return (
              <li key={i} className="text-xs">
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => onJump(p.a)}
                    className="font-medium text-ink-900 dark:text-ink-50 truncate hover:underline max-w-[35%] text-left"
                  >
                    {p.a}
                  </button>
                  <span className="text-ink-400">×</span>
                  <button
                    type="button"
                    onClick={() => onJump(p.b)}
                    className="font-medium text-ink-900 dark:text-ink-50 truncate hover:underline max-w-[35%] text-left"
                  >
                    {p.b}
                  </button>
                  <span className="ml-auto tabular-nums text-ink-500 w-12 text-right">
                    {p.r >= 0 ? '+' : ''}
                    {p.r.toFixed(2)}
                  </span>
                </div>
                <div className="mt-0.5 h-1.5 rounded bg-ink-100 dark:bg-ink-800 overflow-hidden">
                  <div
                    className={`h-full ${tone}`}
                    style={{
                      width: `${mag * 100}%`,
                      marginLeft: p.r < 0 ? `${(1 - mag) * 100}%` : 0,
                    }}
                  />
                </div>
              </li>
            );
          })}
        </ul>
      </section>

      <section>
        <h4 className="text-[10px] uppercase tracking-wider text-ink-500 mb-2">
          Heatmap ({N} × {N})
        </h4>
        <div className="rounded-lg border border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 p-2 inline-block max-w-full overflow-auto">
          <svg width={heatmapSize} height={heatmapSize} className="block">
            {corr.matrix.map((row, i) =>
              row.map((v, j) => (
                <rect
                  key={`${i}-${j}`}
                  x={j * cellSize}
                  y={i * cellSize}
                  width={cellSize}
                  height={cellSize}
                  fill={correlationColor(v)}
                >
                  <title>
                    {corr.columns[i]} × {corr.columns[j]} = {v.toFixed(3)}
                  </title>
                </rect>
              )),
            )}
          </svg>
        </div>
        <div className="mt-2 flex items-center gap-2 text-[10px] text-ink-500">
          <span className="inline-block w-4 h-2 rounded" style={{ background: correlationColor(-1) }} /> -1
          <span className="inline-block w-4 h-2 rounded" style={{ background: correlationColor(0) }} /> 0
          <span className="inline-block w-4 h-2 rounded" style={{ background: correlationColor(1) }} /> +1
        </div>
      </section>
    </div>
  );
}

function correlationColor(r: number): string {
  // Diverging: red (neg) → white (0) → blue (pos). Magnitude → saturation.
  const mag = Math.min(1, Math.abs(r));
  const a = mag * 0.85;
  if (r >= 0) {
    return `rgba(14, 165, 233, ${a})`; // sky-500
  } else {
    return `rgba(244, 63, 94, ${a})`; // rose-500
  }
}

// ─── Health strip (top) ─────────────────────────────────────────────

function HealthStrip({ profile }: { profile: DatasetProfile }) {
  const tiles: Array<{ label: string; value: string; tone?: string }> = [
    { label: 'Rows', value: profile.shape.rows.toLocaleString() },
    { label: 'Columns', value: profile.shape.cols.toLocaleString() },
    { label: 'Memory', value: fmtBytes(profile.memory_bytes) },
    {
      label: 'Duplicates',
      value: profile.duplicates.toLocaleString(),
      tone: profile.duplicates > 0 ? 'amber' : undefined,
    },
    {
      label: 'Missing',
      value: `${profile.missing_pct.toFixed(1)}%`,
      tone:
        profile.missing_pct > 50
          ? 'red'
          : profile.missing_pct > 20
            ? 'amber'
            : undefined,
    },
  ];

  return (
    <div className="flex flex-wrap items-stretch gap-2">
      {tiles.map((t) => (
        <div
          key={t.label}
          className={`rounded-md border px-3 py-1 ${
            t.tone === 'amber'
              ? 'border-amber-300 bg-amber-50 dark:bg-amber-950/30'
              : t.tone === 'red'
                ? 'border-danger-300 bg-danger-50 dark:bg-danger-950/30'
                : 'border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900'
          }`}
        >
          <p className="text-[10px] uppercase tracking-wider text-ink-500 leading-tight">
            {t.label}
          </p>
          <p className="text-sm font-semibold text-ink-900 dark:text-ink-50 leading-tight">
            {t.value}
          </p>
        </div>
      ))}
      <div className="rounded-md border border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 px-3 py-1 flex items-center gap-2">
        <p className="text-[10px] uppercase tracking-wider text-ink-500">Types</p>
        <div className="flex items-center gap-1">
          {Object.entries(profile.type_counts).map(([k, v]) => (
            <span
              key={k}
              className={`text-[10px] uppercase font-semibold px-1.5 py-0.5 rounded ${
                KIND_PALETTE[k] ?? KIND_PALETTE.text
              }`}
              title={`${v} ${k} column${v === 1 ? '' : 's'}`}
            >
              {k.slice(0, 3)}·{v}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Warnings chip strip (top) ─────────────────────────────────────

function WarningsStrip({
  warnings,
  onJump,
}: {
  warnings: DatasetProfile['warnings'];
  onJump: (col: string) => void;
}) {
  if (warnings.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="text-[10px] uppercase tracking-wider text-ink-500 mr-1">
        Heuristics
      </span>
      {warnings.map((w, i) => {
        const tone =
          w.severity === 'error'
            ? 'bg-danger-500/15 text-danger-700 dark:text-danger-300'
            : w.severity === 'warn'
              ? 'bg-amber-500/15 text-amber-700 dark:text-amber-300'
              : 'bg-ink-500/10 text-ink-700 dark:text-ink-300';
        return (
          <button
            key={i}
            type="button"
            onClick={() => w.column && onJump(w.column)}
            className={`text-[11px] px-2 py-0.5 rounded ${tone} ${
              w.column ? 'hover:opacity-80 cursor-pointer' : 'cursor-default'
            }`}
            title={w.message}
          >
            {w.column ? `${w.column}: ${w.kind}` : w.kind}
          </button>
        );
      })}
    </div>
  );
}

// ─── Per-column detail ─────────────────────────────────────────────

function ColumnDetail({
  column,
  profile,
  onJump,
}: {
  column: DatasetColumn;
  profile: DatasetProfile;
  onJump: (col: string) => void;
}) {
  const stats = column.stats ?? {};
  const hints = roleHints(column, profile.shape.rows);

  const statRows: Array<[string, number | string | null | undefined]> = [
    ['Unique', column.unique],
    ['Cardinality', `${column.cardinality_pct.toFixed(1)}%`],
    ['Missing', `${column.missing} (${column.missing_pct.toFixed(1)}%)`],
    ['Dtype', column.dtype],
  ];
  if (column.kind === 'numeric') {
    statRows.push(
      ['Min', stats.min ?? null],
      ['Max', stats.max ?? null],
      ['Mean', stats.mean ?? null],
      ['Median', stats.median ?? null],
      ['Std', stats.std ?? null],
      ['Q1', stats.q1 ?? null],
      ['Q3', stats.q3 ?? null],
      ['Skew', stats.skew ?? null],
      ['Kurt.', stats.kurtosis ?? null],
      ['Zeros %', stats.zeros_pct ?? null],
      ['Outliers (IQR)', stats.outliers_iqr ?? null],
    );
  }

  return (
    <div className="rounded-lg border border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 p-3 space-y-3">
      <div className="flex items-center gap-2 flex-wrap">
        <span
          className={`text-[10px] uppercase font-semibold px-1.5 py-0.5 rounded ${
            KIND_PALETTE[column.kind] ?? KIND_PALETTE.text
          }`}
        >
          {column.kind}
        </span>
        <h4 className="text-sm font-semibold text-ink-900 dark:text-ink-50 truncate flex-1 min-w-0">
          {column.name}
        </h4>
        {hints.map((h, i) => (
          <span
            key={i}
            className={`text-[10px] px-1.5 py-0.5 rounded ${HINT_TONE[h.tone]}`}
            title={h.title}
          >
            {h.label}
          </span>
        ))}
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
        {statRows.map(([k, v]) => (
          <div
            key={k}
            className="flex items-center justify-between border-b border-ink-100 dark:border-ink-900 py-0.5"
          >
            <span className="text-ink-500">{k}</span>
            <span className="font-medium text-ink-900 dark:text-ink-50 truncate ml-2 tabular-nums">
              {fmtNum(v as number | string | null | undefined)}
            </span>
          </div>
        ))}
      </div>

      {/* Histogram (numeric) */}
      {column.kind === 'numeric' && column.histogram.length > 0 && (
        <div>
          <p className="text-[10px] uppercase tracking-wider text-ink-500 mb-1">
            Distribution
          </p>
          <Histogram bins={column.histogram} />
        </div>
      )}

      {/* Top values (non-numeric) */}
      {column.kind !== 'numeric' && column.top_values.length > 0 && (
        <div>
          <p className="text-[10px] uppercase tracking-wider text-ink-500 mb-1">
            Top {column.top_values.length} values
          </p>
          <TopValues rows={column.top_values} />
        </div>
      )}

      {/* Most correlated (numeric only, with matrix) */}
      {column.kind === 'numeric' && profile.correlations && (
        <CorrelatedColumns
          column={column.name}
          corr={profile.correlations}
          onJump={onJump}
        />
      )}
    </div>
  );
}

// ─── Histogram (used in detail) ────────────────────────────────────

function Histogram({ bins }: { bins: DatasetColumn['histogram'] }) {
  const max = Math.max(...bins.map((b) => b.count), 1);
  return (
    <div className="flex items-end gap-0.5 h-16 px-1 rounded bg-ink-50 dark:bg-ink-950/40 pt-1">
      {bins.map((b, i) => {
        const h = Math.max(2, Math.round((b.count / max) * 100));
        return (
          <div
            key={i}
            className="flex-1 bg-accent-500/70 hover:bg-accent-500 rounded-t transition-colors"
            style={{ height: `${h}%` }}
            title={`[${fmtNum(b.bin_start)}, ${fmtNum(b.bin_end)}) — ${b.count}`}
          />
        );
      })}
    </div>
  );
}

// ─── Inline sparkline (column picker) ──────────────────────────────

function Sparkline({ bins }: { bins: DatasetColumn['histogram'] }) {
  const max = Math.max(...bins.map((b) => b.count), 1);
  return (
    <div className="flex items-end gap-px h-3 w-14 shrink-0">
      {bins.slice(0, 14).map((b, i) => {
        const h = Math.max(1, Math.round((b.count / max) * 100));
        return (
          <div
            key={i}
            className="flex-1 bg-sky-400/70 dark:bg-sky-500/60 rounded-sm"
            style={{ height: `${h}%` }}
          />
        );
      })}
    </div>
  );
}

// ─── Top values bar list ───────────────────────────────────────────

function TopValues({ rows }: { rows: DatasetColumn['top_values'] }) {
  const max = Math.max(...rows.map((r) => r.pct), 1);
  return (
    <ul className="space-y-1">
      {rows.map((r) => (
        <li key={r.value} className="text-xs">
          <div className="flex items-center justify-between mb-0.5">
            <span
              className="font-medium text-ink-900 dark:text-ink-50 truncate max-w-[60%]"
              title={r.value}
            >
              {r.value || <em className="text-ink-400">empty</em>}
            </span>
            <span className="text-ink-500 tabular-nums">
              {r.count.toLocaleString()} · {r.pct.toFixed(1)}%
            </span>
          </div>
          <div className="h-1.5 rounded bg-ink-100 dark:bg-ink-800 overflow-hidden">
            <div
              className="h-full bg-violet-500/80"
              style={{ width: `${(r.pct / max) * 100}%` }}
            />
          </div>
        </li>
      ))}
    </ul>
  );
}

// ─── Correlated columns (per-column detail) ────────────────────────

function CorrelatedColumns({
  column,
  corr,
  onJump,
}: {
  column: string;
  corr: NonNullable<DatasetProfile['correlations']>;
  onJump: (col: string) => void;
}) {
  const idx = corr.columns.indexOf(column);
  if (idx < 0) return null;
  const row = corr.matrix[idx];
  const ranked = corr.columns
    .map((name, i) => ({ name, r: row[i] }))
    .filter((x) => x.name !== column && Number.isFinite(x.r))
    .sort((a, b) => Math.abs(b.r) - Math.abs(a.r))
    .slice(0, 5);

  if (ranked.length === 0) return null;

  return (
    <div>
      <p className="text-[10px] uppercase tracking-wider text-ink-500 mb-1">
        Top correlated
      </p>
      <ul className="space-y-1">
        {ranked.map((x) => {
          const mag = Math.abs(x.r);
          const tone =
            mag > 0.7
              ? 'bg-emerald-500/80'
              : mag > 0.4
                ? 'bg-emerald-500/50'
                : 'bg-ink-400/60';
          const sign = x.r >= 0 ? '+' : '';
          return (
            <li key={x.name} className="text-xs">
              <button
                type="button"
                onClick={() => onJump(x.name)}
                className="w-full flex items-center gap-2 hover:bg-ink-50 dark:hover:bg-ink-800 rounded px-1 py-0.5"
              >
                <span className="truncate font-medium text-ink-800 dark:text-ink-200 max-w-[55%] text-left">
                  {x.name}
                </span>
                <div className="flex-1 h-1.5 rounded bg-ink-100 dark:bg-ink-800 overflow-hidden">
                  <div
                    className={`h-full ${tone}`}
                    style={{
                      width: `${mag * 100}%`,
                      marginLeft: x.r < 0 ? `${(1 - mag) * 100}%` : 0,
                    }}
                  />
                </div>
                <span className="tabular-nums text-ink-500 w-12 text-right">
                  {sign}
                  {x.r.toFixed(2)}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

// ─── Missing-% bar (column picker + overview) ──────────────────────

function MissingBar({ pct, wide }: { pct: number; wide?: boolean }) {
  const display = Math.max(0, Math.min(100, pct));
  if (display === 0) {
    return (
      <span className={`text-[10px] text-ink-400 ${wide ? 'w-12' : 'w-10'} text-right shrink-0`}>
        0%
      </span>
    );
  }
  const tone =
    display > 50 ? 'bg-danger-500' : display > 20 ? 'bg-amber-500' : 'bg-ink-400';
  return (
    <div className={`flex items-center gap-1 ${wide ? 'w-full' : 'w-12'} shrink-0`}>
      <div className="flex-1 h-1 rounded bg-ink-100 dark:bg-ink-800 overflow-hidden">
        <div className={`h-full ${tone}`} style={{ width: `${display}%` }} />
      </div>
      {!wide && (
        <span className="text-[9px] text-ink-500 tabular-nums w-6 text-right">
          {display.toFixed(0)}%
        </span>
      )}
    </div>
  );
}
