/**
 * Modal that lets users pick a target column by visually scanning a sample
 * of the dataset. Backed by the dataset profile endpoint — same data the
 * EDA dashboard uses, so column kinds + sample values are accurate.
 *
 * UX:
 *   - Top: searchable list of columns with kind chips + missing/unique
 *     counters.
 *   - Right: a sample-rows preview where the picked column is highlighted
 *     across every row so users see what kind of values live in it.
 *   - Bottom: confirm button writes the selection back to the parent.
 */

import { useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { dataSourcesApi } from '@/api/endpoints';
import { Dialog } from './Dialog';
import type { DatasetColumn, DatasetColumnKind } from '@/api/endpoints';

const KIND_CHIP: Record<DatasetColumnKind, string> = {
  numeric: 'bg-sky-50 text-sky-700 dark:bg-sky-500/15 dark:text-sky-300',
  categorical:
    'bg-violet-50 text-violet-700 dark:bg-violet-500/15 dark:text-violet-300',
  datetime: 'bg-cyan-50 text-cyan-700 dark:bg-cyan-500/15 dark:text-cyan-300',
  boolean: 'bg-amber-50 text-amber-700 dark:bg-amber-500/15 dark:text-amber-300',
  text: 'bg-fuchsia-50 text-fuchsia-700 dark:bg-fuchsia-500/15 dark:text-fuchsia-300',
};

export interface ColumnPickerModalProps {
  open: boolean;
  onClose: () => void;
  dataSourceId: string;
  dataSourceName?: string | null;
  selected: string;
  onSelect: (column: string) => void;
}

export function ColumnPickerModal({
  open,
  onClose,
  dataSourceId,
  dataSourceName,
  selected,
  onSelect,
}: ColumnPickerModalProps) {
  const profile = useQuery({
    queryKey: ['datasets', dataSourceId, 'profile'],
    queryFn: () => dataSourcesApi.profile(dataSourceId, 12),
    enabled: open && !!dataSourceId,
    staleTime: 60_000,
  });

  // Local pick — only commits to the parent on confirm so the user can
  // explore without accidentally clobbering their previous answer.
  const [pick, setPick] = useState<string>(selected);
  useEffect(() => {
    if (open) setPick(selected);
  }, [open, selected]);

  const [search, setSearch] = useState('');
  const filteredCols = useMemo(() => {
    const cols = profile.data?.columns ?? [];
    const q = search.trim().toLowerCase();
    if (!q) return cols;
    return cols.filter((c) => c.name.toLowerCase().includes(q));
  }, [profile.data, search]);

  const sampleCols = profile.data?.columns ?? [];
  const sampleRows = profile.data?.sample ?? [];

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="Pick the target column"
      description={
        dataSourceName
          ? `Scanning ${dataSourceName} — pick the column the model should learn to predict.`
          : 'Pick the column the model should learn to predict.'
      }
      size="xl"
      footer={
        <div className="flex items-center justify-between gap-3 w-full">
          <p className="text-xs text-ink-500">
            {pick ? (
              <>
                Selected:{' '}
                <span className="font-mono text-ink-900 dark:text-ink-50">
                  {pick}
                </span>
              </>
            ) : (
              'No column picked yet'
            )}
          </p>
          <div className="flex items-center gap-2">
            <button
              type="button"
              className="btn-secondary"
              onClick={onClose}
              data-dialog-close
            >
              Cancel
            </button>
            <button
              type="button"
              className="btn-primary"
              disabled={!pick}
              onClick={() => onSelect(pick)}
            >
              Use this column
            </button>
          </div>
        </div>
      }
    >
      {profile.isPending && (
        <p className="text-sm text-ink-500">Profiling dataset…</p>
      )}
      {profile.error && (
        <p className="text-sm text-danger-600">
          Could not load dataset profile.
        </p>
      )}
      {profile.data && (
        <div className="grid grid-cols-1 md:grid-cols-[260px_1fr] gap-4 max-h-[60vh]">
          {/* ─── Column list ─── */}
          <aside className="border border-ink-200 dark:border-ink-800 rounded-md flex flex-col overflow-hidden">
            <input
              className="input rounded-none border-0 border-b border-ink-200 dark:border-ink-800 text-sm"
              placeholder="Search columns…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            <ul className="overflow-y-auto flex-1 divide-y divide-ink-200 dark:divide-ink-800">
              {filteredCols.length === 0 ? (
                <li className="px-3 py-6 text-center text-xs text-ink-500">
                  No columns match.
                </li>
              ) : (
                filteredCols.map((c) => (
                  <li key={c.name}>
                    <button
                      type="button"
                      onClick={() => setPick(c.name)}
                      className={`w-full text-left px-3 py-2 hover:bg-ink-50 dark:hover:bg-ink-950/40 ${
                        pick === c.name
                          ? 'bg-accent-50 dark:bg-accent-500/10'
                          : ''
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-sm font-mono text-ink-900 dark:text-ink-50 truncate">
                          {c.name}
                        </span>
                        <span
                          className={`pill text-[10px] capitalize ${KIND_CHIP[c.kind]}`}
                        >
                          {c.kind}
                        </span>
                      </div>
                      <div className="text-[11px] text-ink-500 mt-0.5">
                        {c.unique.toLocaleString()} unique
                        {c.missing > 0 && (
                          <span className="ml-1.5 text-amber-600 dark:text-amber-500">
                            · {Math.round(c.missing_pct * 100)}% missing
                          </span>
                        )}
                      </div>
                    </button>
                  </li>
                ))
              )}
            </ul>
          </aside>

          {/* ─── Right pane: highlighted sample rows ─── */}
          <main className="border border-ink-200 dark:border-ink-800 rounded-md overflow-hidden flex flex-col">
            <div className="px-3 py-2 border-b border-ink-200 dark:border-ink-800 text-xs text-ink-500 flex items-center justify-between">
              <span>
                Sample rows — the picked column is highlighted so you can
                eyeball value ranges.
              </span>
              <span className="tabular-nums">{sampleRows.length} shown</span>
            </div>
            <div className="overflow-auto flex-1">
              <table className="w-full text-xs">
                <thead className="bg-white dark:bg-ink-900 border-b border-ink-200 dark:border-ink-800">
                  <tr>
                    <th className="px-2 py-1.5 text-left font-medium text-ink-500 w-8">
                      #
                    </th>
                    {sampleCols.map((c) => (
                      <th
                        key={c.name}
                        className={`px-2 py-1.5 text-left font-medium whitespace-nowrap ${
                          pick === c.name
                            ? 'bg-accent-50 dark:bg-accent-500/10'
                            : ''
                        }`}
                      >
                        <button
                          type="button"
                          onClick={() => setPick(c.name)}
                          className="text-left font-mono text-[11px] text-ink-700 dark:text-ink-300 hover:text-accent-600"
                        >
                          {c.name}
                        </button>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sampleRows.map((row, i) => (
                    <tr
                      key={i}
                      className="border-t border-ink-200 dark:border-ink-800"
                    >
                      <td className="px-2 py-1.5 text-ink-400 tabular-nums">
                        {i + 1}
                      </td>
                      {sampleCols.map((c) => {
                        const v = (row as Record<string, unknown>)[c.name];
                        const isPicked = pick === c.name;
                        return (
                          <td
                            key={c.name}
                            className={`px-2 py-1.5 font-mono whitespace-nowrap ${
                              isPicked
                                ? 'bg-accent-50/70 dark:bg-accent-500/10 text-accent-800 dark:text-accent-300 font-semibold'
                                : 'text-ink-700 dark:text-ink-300'
                            }`}
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
            {/* Picked column quick stats */}
            {pick && sampleCols.find((c) => c.name === pick) && (
              <PickedColumnSummary
                col={sampleCols.find((c) => c.name === pick)!}
              />
            )}
          </main>
        </div>
      )}
    </Dialog>
  );
}

function PickedColumnSummary({ col }: { col: DatasetColumn }) {
  return (
    <div className="border-t border-ink-200 dark:border-ink-800 px-3 py-2.5 bg-ink-50/50 dark:bg-ink-950/40 text-xs">
      <div className="flex items-center gap-2 flex-wrap">
        <span className={`pill text-[10px] capitalize ${KIND_CHIP[col.kind]}`}>
          {col.kind}
        </span>
        <span className="font-mono text-ink-700 dark:text-ink-300">
          {col.name}
        </span>
        <span className="text-ink-500">
          · {col.unique.toLocaleString()} unique
        </span>
        {col.missing > 0 && (
          <span className="text-amber-600">
            · {col.missing.toLocaleString()} missing
          </span>
        )}
      </div>
      {col.kind === 'numeric' && col.stats && (
        <div className="mt-1 text-ink-500 font-mono">
          min {String(col.stats.min)} · median {String(col.stats.median)} · max{' '}
          {String(col.stats.max)}
        </div>
      )}
      {(col.kind === 'categorical' || col.kind === 'boolean') &&
        col.top_values.length > 0 && (
          <div className="mt-1 flex flex-wrap gap-1">
            {col.top_values.slice(0, 5).map((v) => (
              <span
                key={v.value}
                className="inline-flex items-center px-1.5 py-0.5 rounded bg-white dark:bg-ink-900 border border-ink-200 dark:border-ink-800 font-mono text-[10px] text-ink-700 dark:text-ink-300"
              >
                {v.value}
                <span className="ml-1 text-ink-400 tabular-nums">
                  {Math.round(v.pct * 100)}%
                </span>
              </span>
            ))}
          </div>
        )}
    </div>
  );
}
