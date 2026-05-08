/**
 * Workspace home — `/workspaces/:wsId/home`.
 *
 * Cockpit-style landing page with KPI strip, recent runs, and a modern
 * shortcuts column. No inline styles — all design tokens via Tailwind +
 * existing primitive classes.
 */

import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';

import {
  deploymentsApi,
  experimentsApi,
  pipelinesApi,
  projectsApi,
  runsApi,
  workspacesApi,
} from '@/api/endpoints';

const STATUS_PILL: Record<string, string> = {
  succeeded: 'pill-success',
  running: 'pill-accent',
  queued: 'pill-neutral',
  failed: 'pill-danger',
  cancelled: 'pill-warn',
};

function StatusPill({ status }: { status: string }) {
  const cls = STATUS_PILL[status] ?? 'pill-neutral';
  return <span className={`${cls} capitalize`}>{status}</span>;
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
        .sort((a, b) => (b.created_at > a.created_at ? 1 : -1))
        .slice(0, 12);
    },
    enabled: !!experiments.data,
  });

  const last7dCount = useMemo(() => {
    if (!recentRuns.data) return 0;
    const cutoff = Date.now() - 7 * 24 * 60 * 60 * 1000;
    return recentRuns.data.filter((r) => Date.parse(r.created_at) >= cutoff).length;
  }, [recentRuns.data]);

  const tiles = [
    { label: 'Projects', value: projects.data?.length ?? '—' },
    { label: 'Experiments', value: experiments.data?.length ?? '—' },
    { label: 'Runs (7d)', value: last7dCount },
    { label: 'Pipelines', value: pipelines.data?.length ?? '—' },
    { label: 'Deployments', value: deployments.data?.length ?? '—' },
  ];

  const shortcuts: ShortcutRow[] = [
    {
      to: `/workspaces/${wsId}`,
      label: 'Datasets & projects',
      desc: 'Upload CSVs, organise projects',
      icon: <FolderIcon />,
    },
    {
      to: `/workspaces/${wsId}/pipelines`,
      label: 'Pipelines registry',
      desc: 'Promoted, fitted pipelines',
      icon: <PipelineIcon />,
    },
    {
      to: `/workspaces/${wsId}/deployments`,
      label: 'Deployments',
      desc: 'Live serving endpoints',
      icon: <DeployIcon />,
    },
    {
      to: `/workspaces/${wsId}/predictions`,
      label: 'Prediction explorer',
      desc: 'Hit any deployment with JSON',
      icon: <ZapIcon />,
    },
    {
      to: `/workspaces/${wsId}/compare`,
      label: 'Model comparison',
      desc: 'Diff runs side-by-side',
      icon: <CompareIcon />,
    },
    {
      to: `/workspaces/${wsId}/drift`,
      label: 'Drift dashboard',
      desc: 'Distribution shift over time',
      icon: <DriftIcon />,
    },
    {
      to: `/workspaces/${wsId}/schedules`,
      label: 'Schedules',
      desc: 'Cron-style monitors + retraining',
      icon: <ClockIcon />,
    },
    {
      to: `/workspaces/${wsId}/webhooks`,
      label: 'Webhooks',
      desc: 'Outgoing event hooks',
      icon: <HookIcon />,
    },
    {
      to: `/workspaces/${wsId}/llm`,
      label: 'LLM settings',
      desc: 'Configure AI provider',
      icon: <SparkIcon />,
    },
    {
      to: `/workspaces/${wsId}/members`,
      label: 'Members',
      desc: 'Roles + invitations',
      icon: <UsersIcon />,
    },
  ];

  return (
    <div className="space-y-8">
      {/* ─── Header ──────────────────────────────────────────── */}
      <header className="space-y-1">
        <nav className="text-xs text-ink-500">
          <Link to="/" className="hover:text-ink-900 dark:hover:text-ink-50">
            Workspaces
          </Link>
          <span className="mx-1.5 text-ink-300">/</span>
          <span className="text-ink-700 dark:text-ink-300">{ws.data?.name ?? ''}</span>
        </nav>
        <h1 className="h-page">{ws.data?.name ?? 'Workspace'}</h1>
        <p className="text-sm text-ink-500">
          Overview of activity, models, and deployments.
        </p>
      </header>

      {/* ─── KPI strip ───────────────────────────────────────── */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
        {tiles.map((t) => (
          <div key={t.label} className="card-tight">
            <p className="text-[11px] uppercase tracking-wider text-ink-500 font-medium">
              {t.label}
            </p>
            <p className="mt-1 text-2xl font-semibold tabular-nums text-ink-900 dark:text-ink-50">
              {t.value}
            </p>
          </div>
        ))}
      </div>

      {/* ─── Recent runs + Shortcuts ─────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Recent runs spans 2 cols on lg+ */}
        <section className="card lg:col-span-2">
          <h2 className="h-section mb-4">Recent runs</h2>
          {recentRuns.isLoading ? (
            <p className="text-sm text-ink-500">Loading…</p>
          ) : (recentRuns.data?.length ?? 0) === 0 ? (
            <p className="text-sm text-ink-500">
              No runs yet. Start with{' '}
              <Link
                to={`/workspaces/${wsId}`}
                className="text-accent-600 hover:underline"
              >
                a new experiment
              </Link>
              .
            </p>
          ) : (
            <ul className="-mx-2">
              {recentRuns.data!.map((r) => (
                <li key={r.id}>
                  <Link
                    to={`/runs/${r.id}`}
                    className="flex items-center justify-between gap-3 px-2 py-1.5 rounded-md hover:bg-ink-50 dark:hover:bg-ink-800/40 transition-colors group"
                  >
                    <div className="flex items-center gap-3 min-w-0 flex-1">
                      <StatusPill status={r.status} />
                      <span
                        className="text-sm font-mono text-ink-700 dark:text-ink-300 group-hover:text-ink-900 dark:group-hover:text-ink-50 truncate"
                        title={r.id}
                      >
                        {r.id}
                      </span>
                    </div>
                    <span className="text-xs text-ink-500 tabular-nums shrink-0">
                      {r.duration_ms != null
                        ? `${(r.duration_ms / 1000).toFixed(1)}s`
                        : '—'}
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </section>

        {/* Shortcuts — modern: small icon + label + description, subtle hover */}
        <section className="card">
          <h2 className="h-section mb-4">Shortcuts</h2>
          <ul className="-mx-2">
            {shortcuts.map((s) => (
              <li key={s.to}>
                <Link
                  to={s.to}
                  className="flex items-center gap-3 px-2 py-2 rounded-md hover:bg-ink-50 dark:hover:bg-ink-800/40 transition-colors group"
                >
                  <span className="shrink-0 text-ink-400 group-hover:text-ink-700 dark:group-hover:text-ink-200 transition-colors">
                    {s.icon}
                  </span>
                  <span className="flex-1 min-w-0">
                    <span className="block text-sm font-medium text-ink-800 dark:text-ink-100 group-hover:text-ink-900 dark:group-hover:text-ink-50">
                      {s.label}
                    </span>
                    <span className="block text-xs text-ink-500 truncate">{s.desc}</span>
                  </span>
                  <span className="shrink-0 text-ink-300 group-hover:text-ink-500 dark:group-hover:text-ink-400 opacity-0 group-hover:opacity-100 transition-opacity">
                    <ChevronRightIcon />
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
}

// ─── Shortcut data shape ──────────────────────────────────────────

interface ShortcutRow {
  to: string;
  label: string;
  desc: string;
  icon: React.ReactNode;
}

// ─── Inline 16px icons (lucide-style, currentColor) ───────────────

const stroke = {
  width: '16',
  height: '16',
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: '1.75',
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  'aria-hidden': true,
};

const FolderIcon = () => (
  <svg {...stroke}>
    <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
  </svg>
);
const PipelineIcon = () => (
  <svg {...stroke}>
    <circle cx="6" cy="6" r="2" />
    <circle cx="6" cy="18" r="2" />
    <circle cx="18" cy="12" r="2" />
    <path d="M8 6h6a2 2 0 0 1 2 2v2" />
    <path d="M8 18h6a2 2 0 0 0 2-2v-2" />
  </svg>
);
const DeployIcon = () => (
  <svg {...stroke}>
    <path d="M5 12 12 5l7 7" />
    <path d="M12 5v14" />
  </svg>
);
const ZapIcon = () => (
  <svg {...stroke}>
    <path d="M13 2 4 14h7l-1 8 9-12h-7z" />
  </svg>
);
const CompareIcon = () => (
  <svg {...stroke}>
    <path d="M3 12h18" />
    <path d="M9 6 3 12l6 6" />
    <path d="m15 6 6 6-6 6" />
  </svg>
);
const DriftIcon = () => (
  <svg {...stroke}>
    <path d="M3 17l4-7 5 4 5-9 4 7" />
  </svg>
);
const ClockIcon = () => (
  <svg {...stroke}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 7v5l3 2" />
  </svg>
);
const HookIcon = () => (
  <svg {...stroke}>
    <path d="M18 6V3" />
    <path d="M18 6a4 4 0 0 0-4 4v6a3 3 0 1 1-6 0v-2" />
    <circle cx="18" cy="3" r="1" />
  </svg>
);
const SparkIcon = () => (
  <svg {...stroke}>
    <path d="M12 3v3" />
    <path d="M12 18v3" />
    <path d="M3 12h3" />
    <path d="M18 12h3" />
    <path d="M5.6 5.6l2.1 2.1" />
    <path d="M16.3 16.3l2.1 2.1" />
    <path d="M5.6 18.4l2.1-2.1" />
    <path d="M16.3 7.7l2.1-2.1" />
  </svg>
);
const UsersIcon = () => (
  <svg {...stroke}>
    <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
    <circle cx="9" cy="7" r="4" />
    <path d="M22 21v-2a4 4 0 0 0-3-3.87" />
    <path d="M16 3.13a4 4 0 0 1 0 7.75" />
  </svg>
);
const ChevronRightIcon = () => (
  <svg {...stroke}>
    <path d="m9 18 6-6-6-6" />
  </svg>
);
