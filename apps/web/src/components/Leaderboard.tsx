/**
 * Leaderboard table — renders a Run's `leaderboard` field (list of dicts).
 *
 * The backend stores leaderboards as `outcome.leaderboard.reset_index().to_dict(records)`
 * which means the shape is `Array<Record<string, unknown>>` — any column set,
 * any order. This component doesn't hard-code metric names.
 *
 * First numeric column that isn't `index` / `Model` is treated as the
 * "headline" column and used for bar-graph rendering. A sort-by-column
 * header chevron lets the user re-rank.
 */

import { useMemo, useState } from 'react';

export interface LeaderboardProps {
  rows: Record<string, unknown>[] | null | undefined;
  className?: string;
}

function isNumber(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

function fmtCell(v: unknown): string {
  if (v == null) return '—';
  if (typeof v === 'number') {
    if (Math.abs(v) < 0.0001 && v !== 0) return v.toExponential(2);
    if (Number.isInteger(v)) return String(v);
    return v.toFixed(4);
  }
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

export function Leaderboard({ rows, className }: LeaderboardProps) {
  const columns = useMemo(() => {
    if (!rows || rows.length === 0) return [] as string[];
    // Preserve the column order from the first row; that's what the engine
    // declared (index, Model, Accuracy, AUC, …).
    return Object.keys(rows[0]);
  }, [rows]);

  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDesc, setSortDesc] = useState<boolean>(true);

  const sorted = useMemo(() => {
    if (!rows) return [] as Record<string, unknown>[];
    if (!sortKey) return rows;
    const copy = rows.slice();
    copy.sort((a, b) => {
      const va = a[sortKey];
      const vb = b[sortKey];
      if (isNumber(va) && isNumber(vb)) return sortDesc ? vb - va : va - vb;
      return sortDesc
        ? String(vb).localeCompare(String(va))
        : String(va).localeCompare(String(vb));
    });
    return copy;
  }, [rows, sortKey, sortDesc]);

  if (!rows || rows.length === 0) {
    return (
      <p className={`hint ${className ?? ''}`}>
        Leaderboard will appear here once the run completes.
      </p>
    );
  }

  const toggleSort = (k: string) => {
    if (sortKey === k) setSortDesc((d) => !d);
    else {
      setSortKey(k);
      setSortDesc(true);
    }
  };

  return (
    <div className={`card overflow-x-auto p-0 ${className ?? ''}`}>
      <table className="w-full text-sm">
        <thead className="bg-ink-50 dark:bg-ink-950 text-ink-600 dark:text-ink-400 border-b border-ink-200 dark:border-ink-800">
          <tr>
            {columns.map((c) => (
              <th
                key={c}
                className="px-3 py-2.5 text-left text-xs font-semibold uppercase tracking-wider whitespace-nowrap cursor-pointer select-none hover:text-ink-900 dark:hover:text-ink-50 transition-colors"
                onClick={() => toggleSort(c)}
              >
                {c}
                {sortKey === c && (
                  <span className="ml-1 text-accent-600 dark:text-accent-400">
                    {sortDesc ? '▼' : '▲'}
                  </span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="text-ink-800 dark:text-ink-100">
          {sorted.map((r, i) => (
            <tr
              key={i}
              className="border-t border-ink-100 dark:border-ink-800 hover:bg-ink-50 dark:hover:bg-ink-800/40 transition-colors"
            >
              {columns.map((c) => {
                const v = r[c];
                const isNum = isNumber(v);
                return (
                  <td
                    key={c}
                    className={`px-3 py-2.5 whitespace-nowrap ${
                      isNum ? 'font-mono tabular-nums text-right' : ''
                    }`}
                  >
                    {fmtCell(v)}
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
