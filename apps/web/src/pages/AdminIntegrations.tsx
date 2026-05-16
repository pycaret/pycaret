/**
 * /workspaces/:wsId/admin/integrations — third-party integrations hub.
 *
 * Hub page that routes to each integration's own management screen:
 *   - LLM provider (Anthropic / OpenAI) → /llm
 *   - Webhooks → /webhooks (HMAC-signed event delivery)
 *   - Object storage → /data sources (per-DS S3 / Postgres / HTTP)
 *   - SSO / SAML / OAuth — V2 (non-clickable, honestly labelled)
 */

import { useQuery } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import { llmApi } from '@/api/endpoints';

export function AdminIntegrations() {
  const { wsId } = useParams<{ wsId: string }>();
  const workspaceId = wsId ?? '';

  const { data: llm } = useQuery({
    queryKey: ['workspaces', workspaceId, 'llm', 'settings'],
    queryFn: () => llmApi.getSettings(workspaceId),
    enabled: !!workspaceId,
  });

  // Webhooks aren't part of integrations grid yet — the count is loaded only
  // to update the wire copy on the Notifications card. Cheap and useful.

  return (
    <div className="space-y-8">
      <header>
        <h1 className="h-page">Integrations</h1>
        <p className="mt-2 text-sm text-ink-500">
          Third-party services connected to this workspace.
        </p>
      </header>

      <section className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <IntegrationCard
          title="LLM provider"
          status={llm ? `${llm.provider} · ${llm.model_name}` : 'Not configured'}
          state={llm ? 'connected' : 'not_set'}
          href={`/workspaces/${workspaceId}/llm`}
          desc="Powers the 6 AI advisories. Keys encrypted at rest."
          actionLabel={llm ? 'Manage' : 'Connect'}
        />
        <IntegrationCard
          title="Webhooks"
          status="Outgoing event hooks"
          state="ready"
          href={`/workspaces/${workspaceId}/webhooks`}
          desc="HMAC-signed POSTs on run / deployment / drift events."
          actionLabel="Configure"
        />
        <IntegrationCard
          title="Object storage"
          status="Per-data-source"
          state="ready"
          href={`/workspaces/${workspaceId}`}
          desc="S3 connectors live on each data source. Workspace defaults are V2."
          actionLabel="View data sources"
        />
        <IntegrationCard
          title="Single sign-on"
          status="Coming in V2"
          state="planned"
          href={null}
          desc="SAML / OAuth / LDAP providers. Today, password + JWT only."
          actionLabel="Roadmap"
        />
      </section>
    </div>
  );
}

function IntegrationCard({
  title,
  status,
  state,
  href,
  desc,
  actionLabel,
}: {
  title: string;
  status: string;
  state: 'connected' | 'ready' | 'not_set' | 'planned';
  href: string | null;
  desc: string;
  actionLabel: string;
}) {
  const stateBadge = {
    connected: <span className="pill-success">connected</span>,
    ready: <span className="pill-neutral">available</span>,
    not_set: <span className="pill-warn">not set</span>,
    planned: <span className="pill-neutral">planned</span>,
  }[state];

  const inner = (
    <div className="card-tight h-full flex flex-col">
      <div className="flex items-center justify-between mb-2">
        <h2 className="h-section">{title}</h2>
        {stateBadge}
      </div>
      <p className="text-sm text-ink-800 dark:text-ink-200 font-mono">{status}</p>
      <p className="text-xs text-ink-500 mt-2 flex-1">{desc}</p>
      <span className="text-xs text-accent-600 dark:text-accent-400 mt-3 inline-flex items-center gap-1">
        {actionLabel}
        <ArrowRightIcon />
      </span>
    </div>
  );
  if (!href) {
    return <div className="opacity-60 cursor-not-allowed">{inner}</div>;
  }
  return (
    <Link
      to={href}
      className="block group transition-colors hover:border-ink-300 dark:hover:border-ink-700 [&>*]:hover:border-ink-300 [&>*]:dark:hover:border-ink-700"
    >
      {inner}
    </Link>
  );
}

const ArrowRightIcon = () => (
  <svg
    width="12"
    height="12"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden
  >
    <path d="M5 12h14" />
    <path d="m12 5 7 7-7 7" />
  </svg>
);
