/**
 * Phase 0 trial-centric view embedded inside the Experiment detail page.
 *
 * Lists every Trial that lives under an Experiment — not just the trials
 * grouped under one compare-invocation Run. This is the surface that
 * matches the Phase 0 mental model: a Trial is the persistent pipeline
 * candidate; the compare-batch grouping is an annotation, not a parent.
 *
 * Trials are grouped visually by ``created_by_action_id`` so the visual
 * "this came from one compare click" cue is preserved. Each row deep-links
 * into the legacy trial-detail URL via the back-compat shim until Phase 1
 * lands a native trial-detail route.
 */

import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { trialsApi } from '@/api/endpoints';
import type { Trial, TrialKind, TrialStatus } from '@/api/types';

const STATUS_PILL: Record<TrialStatus, string> = {
  queued: 'pill-neutral',
  running: 'pill-accent',
  succeeded: 'pill-success',
  failed:
    'pill bg-rose-50 text-rose-700 dark:bg-rose-500/15 dark:text-rose-300',
  cancelled:
    'pill bg-amber-50 text-amber-700 dark:bg-amber-500/15 dark:text-amber-300',
};

function TrialStatusPill({ status }: { status: TrialStatus }) {
  return <span className={STATUS_PILL[status]}>{status}</span>;
}

const KIND_CHIP: Record<TrialKind, string> = {
  compare: 'pill-neutral',
  tuned: 'pill-accent',
  ensembled:
    'pill bg-violet-50 text-violet-700 dark:bg-violet-500/15 dark:text-violet-300',
  blended:
    'pill bg-cyan-50 text-cyan-700 dark:bg-cyan-500/15 dark:text-cyan-300',
  stacked:
    'pill bg-fuchsia-50 text-fuchsia-700 dark:bg-fuchsia-500/15 dark:text-fuchsia-300',
  manual: 'pill-neutral',
};

const KIND_OPTIONS: { value: TrialKind | ''; label: string }[] = [
  { value: '', label: 'All kinds' },
  { value: 'compare', label: 'Compare' },
  { value: 'tuned', label: 'Tuned' },
  { value: 'ensembled', label: 'Ensembled' },
  { value: 'blended', label: 'Blended' },
  { value: 'stacked', label: 'Stacked' },
  { value: 'manual', label: 'Manual' },
];

function pickPrimaryMetric(t: Trial): { name: string; value: number } | null {
  const m = t.metrics ?? {};
  for (const name of ['Accuracy', 'AUC', 'F1', 'R2'] as const) {
    const v = m[name];
    if (typeof v === 'number' && Number.isFinite(v)) {
      return { name, value: v };
    }
  }
  for (const name of ['MAE', 'RMSE', 'MSE'] as const) {
    const v = m[name];
    if (typeof v === 'number' && Number.isFinite(v)) {
      return { name, value: v };
    }
  }
  return null;
}

function formatAge(iso: string | null): string {
  if (!iso) return '—';
  const ms = Date.now() - new Date(iso).getTime();
  if (ms < 60_000) return `${Math.round(ms / 1000)}s ago`;
  if (ms < 3_600_000) return `${Math.round(ms / 60_000)}m ago`;
  if (ms < 86_400_000) return `${Math.round(ms / 3_600_000)}h ago`;
  return `${Math.round(ms / 86_400_000)}d ago`;
}

interface Props {
  experimentId: string;
  // ``workspaceId`` / ``projectId`` are accepted for forward-compat
  // (a future "Open in workspace context" link) but the table's Open
  // button routes through the flat ``/runs/:runId/trials/:trialId``
  // path that's already wired in App.tsx, so we don't need them here.
  workspaceId?: string;
  projectId?: string;
}

/**
 * Render the trials list for one Experiment.
 *
 * Wraps `trialsApi.forExperiment` with a kind-filter. Grouped by
 * parent Run so trials born from one user click read as a unit.
 */
export function ExperimentTrialsCard({
  experimentId,
}: Props) {
  const [kind, setKind] = useState<TrialKind | ''>('');

  const trials = useQuery({
    queryKey: ['trials', 'for-experiment', experimentId, kind || 'all'],
    queryFn: () =>
      trialsApi.forExperiment(
        experimentId,
        kind ? { kind: kind as TrialKind } : undefined,
      ),
    enabled: !!experimentId,
    staleTime: 10_000,
  });

  // Group by parent Run id (Phase 0-v2 invariant: every Trial has
  // a run_id). Orphan / pre-revert rows fall into `__none__`.
  const groups = useMemo(() => {
    const items = trials.data?.items ?? [];
    const map = new Map<string, Trial[]>();
    for (const t of items) {
      const k = t.run_id ?? '__none__';
      const arr = map.get(k) ?? [];
      arr.push(t);
      map.set(k, arr);
    }
    return Array.from(map.entries()).sort(([, a], [, b]) => {
      const aLatest = a[0]?.created_at ?? '';
      const bLatest = b[0]?.created_at ?? '';
      return bLatest.localeCompare(aLatest);
    });
  }, [trials.data]);

  const total = trials.data?.items.length ?? 0;

  return (
    <section>
      <header className="mb-4 flex items-baseline justify-between">
        <h2 className="h-section">
          Trials{' '}
          <span className="text-ink-400 font-normal">({total})</span>
        </h2>
        <select
          value={kind}
          onChange={(e) => setKind(e.target.value as TrialKind | '')}
          aria-label="Filter trials by kind"
          className="input-sm"
        >
          {KIND_OPTIONS.map((o) => (
            <option key={o.value || 'all'} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </header>

      {trials.isLoading && (
        <div className="card text-sm text-ink-500">Loading trials…</div>
      )}
      {trials.error && (
        <div className="card text-sm text-danger-600">
          {String((trials.error as Error).message)}
        </div>
      )}

      {!trials.isLoading && !trials.error && total === 0 && (
        <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 p-10 text-center">
          <h3 className="text-base font-semibold text-ink-900 dark:text-ink-50">
            No trials yet
          </h3>
          <p className="mt-1.5 text-sm text-ink-500">
            Run a compare or create plan above to produce pipeline candidates.
          </p>
        </div>
      )}

      {total > 0 && (
        <div className="space-y-6">
          {groups.map(([groupKey, items]) => (
            <div key={groupKey} className="space-y-2">
              <p className="text-xs text-ink-500">
                {groupKey === '__none__'
                  ? 'Standalone'
                  : `Run · ${groupKey.slice(0, 8)}…`}{' '}
                <span className="mx-1">·</span>
                {items.length} trial{items.length === 1 ? '' : 's'}
              </p>
              <div className="card overflow-hidden p-0">
                <table className="w-full text-sm">
                  <thead className="bg-white text-ink-500 dark:bg-ink-900 border-b border-ink-200 dark:border-ink-800">
                    <tr>
                      <th className="px-4 py-2 text-left font-medium">Name</th>
                      <th className="px-4 py-2 text-left font-medium">Kind</th>
                      <th className="px-4 py-2 text-left font-medium">Status</th>
                      <th className="px-4 py-2 text-left font-medium">
                        Primary metric
                      </th>
                      <th className="px-4 py-2 text-left font-medium">Age</th>
                      <th className="px-4 py-2" />
                    </tr>
                  </thead>
                  <tbody>
                    {items.map((t) => {
                      const m = pickPrimaryMetric(t);
                      const linkRunId = t.run_id ?? '';
                      return (
                        <tr
                          key={t.id}
                          className="border-t border-ink-200 dark:border-ink-800 hover:bg-ink-50 dark:hover:bg-ink-950/40"
                        >
                          <td className="px-4 py-2 font-medium text-ink-900 dark:text-ink-50">
                            {t.name ?? t.model_id}
                          </td>
                          <td className="px-4 py-2">
                            <span className={KIND_CHIP[t.kind]}>{t.kind}</span>
                          </td>
                          <td className="px-4 py-2">
                            <TrialStatusPill status={t.status} />
                          </td>
                          <td className="px-4 py-2 tabular-nums text-ink-700 dark:text-ink-300">
                            {m ? (
                              <>
                                <span className="text-ink-500">{m.name}</span>{' '}
                                <span className="font-medium">
                                  {m.value.toFixed(4)}
                                </span>
                              </>
                            ) : (
                              <span className="text-ink-400">—</span>
                            )}
                          </td>
                          <td className="px-4 py-2 text-xs text-ink-500 whitespace-nowrap">
                            {formatAge(t.created_at)}
                          </td>
                          <td className="px-4 py-2 text-right">
                            {linkRunId ? (
                              <Link
                                to={`/runs/${linkRunId}/trials/${t.id}`}
                                className="text-accent-700 hover:text-accent-800 dark:text-accent-300 dark:hover:text-accent-200"
                              >
                                Open
                              </Link>
                            ) : null}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
