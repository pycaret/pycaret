/**
 * Workspace home — `/workspaces/:wsId/home`.
 *
 * Replaces the bare workspace landing screen with a real cockpit:
 *  - KPI strip: active experiments / runs in last 7d / pipelines /
 *    deployments healthy.
 *  - Recent runs feed (status pill + duration).
 *  - Quick links: New experiment, Datasets, Compare, Drift, Predictions.
 *
 * Data sources: existing `experimentsApi.list`, `runsApi.list`,
 * `pipelinesApi.list`, `deploymentsApi.list`. No new endpoints.
 */

import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';

import { deploymentsApi, experimentsApi, pipelinesApi, projectsApi, runsApi, workspacesApi } from '../api/endpoints';

function StatusPill({ status }: { status: string }) {
  const palette: Record<string, [string, string]> = {
    succeeded: ['#22C55E', 'rgba(34,197,94,0.12)'],
    running: ['#5B8DEF', 'rgba(91,141,239,0.12)'],
    queued: ['#94A3B8', 'rgba(148,163,184,0.12)'],
    failed: ['#EF4444', 'rgba(239,68,68,0.12)'],
    cancelled: ['#F59E0B', 'rgba(245,158,11,0.12)'],
  };
  const [fg, bg] = palette[status] ?? ['#64748B', 'rgba(100,116,139,0.12)'];
  return (
    <span
      style={{
        fontSize: 11,
        fontWeight: 600,
        padding: '2px 8px',
        borderRadius: 999,
        color: fg,
        background: bg,
        textTransform: 'capitalize',
      }}
    >
      {status}
    </span>
  );
}

export function WorkspaceHome() {
  const { wsId = '' } = useParams<{ wsId: string }>();

  const ws = useQuery({
    queryKey: ['workspace', wsId],
    queryFn: () => workspacesApi.get(wsId),
    enabled: !!wsId,
  });
  const projects = useQuery({
    queryKey: ['projects', wsId],
    queryFn: () => projectsApi.list(wsId),
    enabled: !!wsId,
  });
  const pipelines = useQuery({
    queryKey: ['pipelines', wsId],
    queryFn: () => pipelinesApi.list(wsId),
    enabled: !!wsId,
  });
  const deployments = useQuery({
    queryKey: ['deployments', wsId],
    queryFn: () => deploymentsApi.list(wsId),
    enabled: !!wsId,
  });

  // Aggregate experiments across projects.
  const experiments = useQuery({
    queryKey: ['experiments-all', wsId, projects.data?.map((p) => p.id).join(',')],
    queryFn: async () => {
      if (!projects.data) return [];
      const lists = await Promise.all(
        projects.data.map((p) => experimentsApi.list(p.id)),
      );
      return lists.flat();
    },
    enabled: !!projects.data && projects.data.length > 0,
  });

  const recentRuns = useQuery({
    queryKey: ['runs-recent', experiments.data?.map((e) => e.id).join(',')],
    queryFn: async () => {
      if (!experiments.data) return [];
      const lists = await Promise.all(
        experiments.data.slice(0, 8).map((e) =>
          runsApi.listForExperiment(e.id).catch(() => []),
        ),
      );
      return lists
        .flat()
        .sort((a: { created_at: string }, b: { created_at: string }) =>
          b.created_at > a.created_at ? 1 : -1,
        )
        .slice(0, 12);
    },
    enabled: !!experiments.data,
  });

  const last7dCount = useMemo(() => {
    if (!recentRuns.data) return 0;
    const cutoff = Date.now() - 7 * 24 * 60 * 60 * 1000;
    return recentRuns.data.filter((r: { created_at: string }) => Date.parse(r.created_at) >= cutoff).length;
  }, [recentRuns.data]);

  const tiles = [
    { label: 'Projects', value: projects.data?.length ?? '—' },
    { label: 'Experiments', value: experiments.data?.length ?? '—' },
    { label: 'Runs (7d)', value: last7dCount },
    { label: 'Pipelines', value: pipelines.data?.length ?? '—' },
    { label: 'Deployments', value: deployments.data?.length ?? '—' },
  ];

  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <nav style={{ fontSize: 12, color: '#94A3B8' }}>
          <Link to="/" style={{ color: 'inherit' }}>Workspaces</Link> / {ws.data?.name ?? ''}
        </nav>
        <h1 style={{ fontSize: 24, fontWeight: 700, color: '#0F172A', margin: 0 }}>
          {ws.data?.name ?? 'Workspace'}
        </h1>
        <p style={{ color: '#64748B', fontSize: 13, margin: 0 }}>
          Overview of activity, models, and deployments.
        </p>
      </header>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${tiles.length}, minmax(140px, 1fr))`,
          gap: 12,
        }}
      >
        {tiles.map((t) => (
          <div key={t.label} className="card" style={{ padding: 14 }}>
            <div style={{ fontSize: 11, textTransform: 'uppercase', color: '#64748B' }}>
              {t.label}
            </div>
            <div style={{ fontSize: 26, fontWeight: 700, color: '#0F172A', marginTop: 4 }}>
              {t.value}
            </div>
          </div>
        ))}
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '2fr 1fr',
          gap: 16,
          alignItems: 'start',
        }}
      >
        <div className="card">
          <h2 className="h-section mb-3">Recent runs</h2>
          {recentRuns.isLoading ? (
            <div className="text-sm text-ink-500">Loading…</div>
          ) : (recentRuns.data?.length ?? 0) === 0 ? (
            <div className="text-sm text-ink-500">
              No runs yet. Start with{' '}
              <Link to={`/workspaces/${wsId}`} className="text-accent-600 hover:text-accent-700">
                a new experiment
              </Link>
              .
            </div>
          ) : (
            <ul className="space-y-1.5">
              {recentRuns.data!.map((r: { id: string; status: string; duration_ms: number | null }) => (
                <li
                  key={r.id}
                  className="flex items-center justify-between gap-3 px-3 py-2 rounded-md hover:bg-ink-50 dark:hover:bg-ink-800/50 transition-colors"
                >
                  <div className="flex items-center gap-3 min-w-0 flex-1">
                    <StatusPill status={r.status} />
                    <Link
                      to={`/runs/${r.id}`}
                      className="text-sm font-mono text-ink-800 dark:text-ink-100 hover:text-accent-700 dark:hover:text-accent-400 truncate"
                      title={r.id}
                    >
                      {r.id}
                    </Link>
                  </div>
                  <div className="text-xs text-ink-500 tabular-nums shrink-0">
                    {r.duration_ms != null
                      ? `${(r.duration_ms / 1000).toFixed(1)}s`
                      : '—'}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="card">
          <h2 className="h-section mb-3">Shortcuts</h2>
          <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 8 }}>
            {[
              { to: `/workspaces/${wsId}`, label: 'Datasets & projects' },
              { to: `/workspaces/${wsId}/pipelines`, label: 'Pipelines registry' },
              { to: `/workspaces/${wsId}/deployments`, label: 'Deployments' },
              { to: `/workspaces/${wsId}/predictions`, label: 'Prediction explorer' },
              { to: `/workspaces/${wsId}/compare`, label: 'Model comparison' },
              { to: `/workspaces/${wsId}/drift`, label: 'Drift dashboard' },
              { to: `/workspaces/${wsId}/llm`, label: 'LLM settings' },
              { to: `/workspaces/${wsId}/members`, label: 'Members' },
            ].map((s) => (
              <li key={s.to}>
                <Link
                  to={s.to}
                  style={{
                    display: 'block',
                    padding: '8px 12px',
                    borderRadius: 10,
                    fontSize: 13,
                    color: '#5B8DEF',
                    background: 'rgba(91,141,239,0.06)',
                    textDecoration: 'none',
                  }}
                >
                  → {s.label}
                </Link>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
