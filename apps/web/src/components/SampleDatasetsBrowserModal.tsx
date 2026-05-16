/**
 * SampleDatasetsBrowserModal — pick a bundled CSV and register it as
 * a workspace DataSource.
 *
 * Backend pair:
 *   GET  /sample-datasets                                   — catalog
 *   POST /workspaces/{ws}/data-sources/from-sample          — register
 *
 * UX:
 *   - Search box (matches name + task).
 *   - Task-bucket filter chips ("All", "Classification", "Regression",
 *     "Clustering", "NLP", "Anomaly", "Association").
 *   - Card grid; each card shows task badge, target, shape, missing flag,
 *     and a primary "Add" button. Clicking Add registers it as a workspace
 *     DataSource and shows a per-card "added ✓" state — modal stays open so
 *     the user can grab multiple in one pass.
 *
 * No infra dependency — these files ship with the repo at
 * `<repo>/datasets/`. Connection-based sources (Postgres / S3) reuse the
 * existing Connections page; this modal is a shortcut for "I just want to
 * play with one of the canonical demos."
 */

import { useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Dialog } from './Dialog';
import { sampleDatasetsApi, type SampleDataset } from '@/api/endpoints';
import { errorMessage } from '@/api/client';

export interface SampleDatasetsBrowserModalProps {
  workspaceId: string;
  open: boolean;
  onClose: () => void;
}

function fmtBytes(n: number | null | undefined): string {
  if (n == null) return '—';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} kB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function bucketOf(task: string): string {
  const t = task.toLowerCase();
  if (t.startsWith('classification')) return 'Classification';
  if (t.startsWith('regression')) return 'Regression';
  if (t.startsWith('clustering')) return 'Clustering';
  if (t.includes('nlp')) return 'NLP';
  if (t.startsWith('anomaly')) return 'Anomaly';
  if (t.includes('association')) return 'Association';
  return 'Other';
}

const BUCKET_PALETTE: Record<string, string> = {
  Classification: 'bg-sky-500/15 text-sky-700 dark:text-sky-300',
  Regression: 'bg-violet-500/15 text-violet-700 dark:text-violet-300',
  Clustering: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300',
  NLP: 'bg-rose-500/15 text-rose-700 dark:text-rose-300',
  Anomaly: 'bg-amber-500/15 text-amber-700 dark:text-amber-300',
  Association: 'bg-indigo-500/15 text-indigo-700 dark:text-indigo-300',
  Other: 'bg-ink-500/15 text-ink-700 dark:text-ink-300',
};

export function SampleDatasetsBrowserModal({
  workspaceId,
  open,
  onClose,
}: SampleDatasetsBrowserModalProps) {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      size="full"
      noBodyPadding
      title="Sample datasets"
      description="Bundled public datasets — click Add to register a sample as a workspace DataSource. The CSV stays on disk; the workspace just keeps a pointer."
    >
      <Body workspaceId={workspaceId} />
    </Dialog>
  );
}

function Body({ workspaceId }: { workspaceId: string }) {
  const catalog = useQuery({
    queryKey: ['sample-datasets'],
    queryFn: () => sampleDatasetsApi.list(),
    staleTime: 5 * 60 * 1000,
  });

  if (catalog.isLoading) {
    return (
      <div className="flex-1 min-h-0 grid place-items-center text-sm text-ink-500">
        Loading catalog…
      </div>
    );
  }
  if (catalog.error) {
    return (
      <div className="flex-1 min-h-0 grid place-items-center text-sm text-danger-600">
        Catalog failed: {errorMessage(catalog.error)}
      </div>
    );
  }
  if (!catalog.data) return null;
  return <Grid workspaceId={workspaceId} all={catalog.data} />;
}

function Grid({
  workspaceId,
  all,
}: {
  workspaceId: string;
  all: SampleDataset[];
}) {
  const qc = useQueryClient();
  const [search, setSearch] = useState('');
  const [bucket, setBucket] = useState<string>('All');
  // Track per-sample registration state for the "added ✓" UI without doing a
  // separate query per card.
  const [added, setAdded] = useState<Record<string, boolean>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});

  const register = useMutation({
    mutationFn: (sample_name: string) =>
      sampleDatasetsApi.register(workspaceId, sample_name),
    onSuccess: (_data, sample_name) => {
      setAdded((s) => ({ ...s, [sample_name]: true }));
      setErrors((e) => {
        const { [sample_name]: _, ...rest } = e;
        return rest;
      });
      qc.invalidateQueries({ queryKey: ['data-sources', workspaceId] });
    },
    onError: (err, sample_name) => {
      setErrors((e) => ({ ...e, [sample_name]: errorMessage(err) }));
    },
  });

  const buckets = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const s of all) {
      const b = bucketOf(s.default_task);
      counts[b] = (counts[b] ?? 0) + 1;
    }
    return [
      { name: 'All', count: all.length },
      ...Object.entries(counts)
        .sort((a, b) => b[1] - a[1])
        .map(([name, count]) => ({ name, count })),
    ];
  }, [all]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    return all.filter((s) => {
      if (bucket !== 'All' && bucketOf(s.default_task) !== bucket) return false;
      if (!q) return true;
      return (
        s.name.toLowerCase().includes(q) ||
        s.default_task.toLowerCase().includes(q) ||
        (s.target_1 ?? '').toLowerCase().includes(q)
      );
    });
  }, [all, bucket, search]);

  return (
    <div className="flex-1 min-h-0 flex flex-col">
      {/* Header: search + bucket chips */}
      <div className="px-6 pt-3 pb-3 space-y-2 shrink-0 border-b border-ink-100 dark:border-ink-800">
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search by name, task, or target column…"
          className="input text-sm"
        />
        <div className="flex flex-wrap items-center gap-1.5">
          {buckets.map((b) => (
            <button
              key={b.name}
              type="button"
              onClick={() => setBucket(b.name)}
              className={`text-xs px-2.5 py-1 rounded-full transition-colors ${
                bucket === b.name
                  ? 'bg-ink-900 dark:bg-ink-50 text-white dark:text-ink-900'
                  : 'bg-ink-100 dark:bg-ink-800 text-ink-700 dark:text-ink-300 hover:bg-ink-200 dark:hover:bg-ink-700'
              }`}
            >
              {b.name}
              <span className="ml-1 opacity-60">{b.count}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Grid */}
      <div className="flex-1 min-h-0 overflow-y-auto px-6 py-4">
        {filtered.length === 0 ? (
          <div className="text-center text-sm text-ink-500 py-12">
            No datasets match "{search}".
          </div>
        ) : (
          <ul className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {filtered.map((s) => {
              const b = bucketOf(s.default_task);
              const tone = BUCKET_PALETTE[b] ?? BUCKET_PALETTE.Other;
              const isAdded = added[s.name];
              const err = errors[s.name];
              const isPending =
                register.isPending && register.variables === s.name;
              return (
                <li
                  key={s.name}
                  className="rounded-lg border border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 p-3 flex flex-col gap-2"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0 flex-1">
                      <h3 className="text-sm font-semibold text-ink-900 dark:text-ink-50 truncate">
                        {s.name}
                      </h3>
                      <p className="text-[11px] text-ink-500 truncate" title={s.default_task}>
                        {s.default_task}
                      </p>
                    </div>
                    <span
                      className={`text-[10px] uppercase font-semibold px-1.5 py-0.5 rounded shrink-0 ${tone}`}
                    >
                      {b}
                    </span>
                  </div>

                  <div className="text-xs text-ink-600 dark:text-ink-400 grid grid-cols-2 gap-x-2 gap-y-0.5">
                    <span className="text-ink-500">Rows</span>
                    <span className="tabular-nums font-medium text-right">
                      {s.instances.toLocaleString()}
                    </span>
                    <span className="text-ink-500">Cols</span>
                    <span className="tabular-nums font-medium text-right">
                      {s.attributes}
                    </span>
                    <span className="text-ink-500">Size</span>
                    <span className="tabular-nums font-medium text-right">
                      {fmtBytes(s.size_bytes)}
                    </span>
                    {s.target_1 && (
                      <>
                        <span className="text-ink-500">Target</span>
                        <span className="font-mono text-[11px] truncate text-right" title={s.target_1}>
                          {s.target_1}
                        </span>
                      </>
                    )}
                  </div>

                  <div className="flex items-center gap-1.5 flex-wrap">
                    {s.has_missing && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-700 dark:text-amber-300">
                        missing values
                      </span>
                    )}
                    {!s.available && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-danger-500/15 text-danger-700 dark:text-danger-300">
                        not on disk
                      </span>
                    )}
                  </div>

                  {err && (
                    <p className="text-[11px] text-danger-600 dark:text-danger-400">
                      {err}
                    </p>
                  )}

                  <button
                    type="button"
                    disabled={!s.available || isAdded || isPending}
                    onClick={() => register.mutate(s.name)}
                    className={`mt-auto text-xs px-3 py-1.5 rounded font-medium transition-colors ${
                      isAdded
                        ? 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 cursor-default'
                        : !s.available
                          ? 'bg-ink-100 dark:bg-ink-800 text-ink-400 dark:text-ink-600 cursor-not-allowed'
                          : 'bg-ink-900 dark:bg-ink-50 text-white dark:text-ink-900 hover:opacity-90'
                    }`}
                  >
                    {isAdded
                      ? '✓ Added to workspace'
                      : isPending
                        ? 'Adding…'
                        : 'Add to workspace'}
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}
