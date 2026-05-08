/**
 * /workspaces/:wsId/admin — workspace admin hub.
 *
 * Centralises the per-workspace admin surfaces:
 *   - members        → /workspaces/:wsId/members
 *   - LLM provider   → /workspaces/:wsId/llm
 *   - audit log      → /workspaces/:wsId/audit-logs (mounted via existing route)
 *   - model library  → renders ModelLibrarySection inline
 *
 * Hub-style page; deeper editing happens on the dedicated screens.
 */

import { useQuery } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import { membersApi, llmApi } from '@/api/endpoints';
import { ModelLibrarySection } from '@/components/ModelLibrarySection';

export function AdminWorkspace() {
  const { wsId } = useParams<{ wsId: string }>();
  const workspaceId = wsId ?? '';

  const { data: members } = useQuery({
    queryKey: ['workspaces', workspaceId, 'members'],
    queryFn: () => membersApi.list(workspaceId),
    enabled: !!workspaceId,
  });
  const { data: llm } = useQuery({
    queryKey: ['workspaces', workspaceId, 'llm', 'settings'],
    queryFn: () => llmApi.getSettings(workspaceId),
    enabled: !!workspaceId,
  });

  return (
    <div className="space-y-8">
      <header>
        <h1 className="h-page">Workspace administration</h1>
        <p className="mt-2 text-sm text-ink-500">
          Membership, LLM provider, model library, and audit trail for this
          workspace.
        </p>
      </header>

      <section className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <AdminTile
          title="Members"
          stat={members ? `${members.length}` : '—'}
          desc="Invite + manage roles"
          href={`/workspaces/${workspaceId}/members`}
          icon={<UsersIcon />}
        />
        <AdminTile
          title="LLM provider"
          stat={llm ? llm.provider : 'not set'}
          desc={
            llm
              ? llm.has_api_key
                ? 'Key encrypted at rest'
                : 'No API key on file'
              : 'Configure Anthropic / OpenAI'
          }
          href={`/workspaces/${workspaceId}/llm`}
          icon={<SparkIcon />}
        />
        <AdminTile
          title="Audit log"
          stat="immutable"
          desc="Every mutating call recorded"
          href={`/workspaces/${workspaceId}/audit-logs`}
          icon={<ShieldIcon />}
        />
      </section>

      <ModelLibrarySection workspaceId={workspaceId} />
    </div>
  );
}

function AdminTile({
  title,
  stat,
  desc,
  href,
  icon,
}: {
  title: string;
  stat: string;
  desc: string;
  href: string;
  icon: React.ReactNode;
}) {
  return (
    <Link
      to={href}
      className="card-tight block group transition-colors hover:border-ink-300 dark:hover:border-ink-700"
    >
      <div className="flex items-start gap-3">
        <span className="shrink-0 h-8 w-8 rounded-lg bg-ink-100 dark:bg-ink-800 text-ink-500 dark:text-ink-400 group-hover:text-ink-900 dark:group-hover:text-ink-50 flex items-center justify-center transition-colors">
          {icon}
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-[11px] uppercase tracking-wider text-ink-500 font-medium">
            {title}
          </p>
          <p className="mt-0.5 text-base font-semibold text-ink-900 dark:text-ink-50 truncate">
            {stat}
          </p>
          <p className="mt-1 text-xs text-ink-500 truncate">{desc}</p>
        </div>
      </div>
    </Link>
  );
}

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
const UsersIcon = () => (
  <svg {...stroke}>
    <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
    <circle cx="9" cy="7" r="4" />
    <path d="M22 21v-2a4 4 0 0 0-3-3.87" />
    <path d="M16 3.13a4 4 0 0 1 0 7.75" />
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
const ShieldIcon = () => (
  <svg {...stroke}>
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
  </svg>
);
