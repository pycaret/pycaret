/**
 * Floating AI Advisor widget — surfaces recent LLM consultations and the
 * configure/connect path when no provider is set.
 *
 * Mounts on the Layout so every authenticated route gets the same affordance.
 *
 * Behaviour:
 *   - Shows a floating button at bottom-right.
 *   - Click → side panel slides in.
 *   - Panel content depends on workspace context (read from the URL via
 *     ``useParams``); falls back to "configure provider" prompt if none.
 *   - Lists the 5 most recent consultations across the workspace; each row
 *     deep-links to the screen that triggered it.
 */

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link, useLocation, useParams } from 'react-router-dom';
import { llmApi } from '@/api/endpoints';

const TYPE_LABELS: Record<string, string> = {
  'analyze-dataset': 'Dataset consultant',
  'design-experiment': 'Experiment designer',
  'explain-run': 'Run explainer',
  'debug-run': 'Failure debugger',
  'review-deployment': 'Deployment reviewer',
  'analyze-drift': 'Drift analyst',
};

function _wsIdFromPath(pathname: string): string | null {
  const m = pathname.match(/^\/workspaces\/([^/]+)/);
  return m?.[1] ?? null;
}

export function AIAdvisorWidget() {
  const [open, setOpen] = useState(false);
  const location = useLocation();
  const params = useParams<{ wsId?: string; id?: string }>();
  const workspaceId =
    params.wsId ?? params.id ?? _wsIdFromPath(location.pathname);

  return (
    <>
      <button
        className={`fixed bottom-5 right-5 z-40 inline-flex items-center gap-2 rounded-full border px-3.5 py-2 text-sm font-medium shadow-soft-2 transition-colors
          ${
            open
              ? 'bg-ink-900 text-white border-ink-900 dark:bg-white dark:text-ink-900 dark:border-white'
              : 'bg-white text-ink-800 border-ink-200 hover:bg-ink-50 dark:bg-ink-900 dark:text-ink-100 dark:border-ink-800 dark:hover:bg-ink-800'
          }`}
        onClick={() => setOpen((v) => !v)}
        aria-label="Toggle AI Advisor"
      >
        <SparkIcon />
        <span>AI</span>
      </button>
      {open && (
        <AdvisorPanel
          workspaceId={workspaceId}
          onClose={() => setOpen(false)}
        />
      )}
    </>
  );
}

const SparkIcon = () => (
  <svg
    width="14"
    height="14"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.75"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden
  >
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

function AdvisorPanel({
  workspaceId,
  onClose,
}: {
  workspaceId: string | null;
  onClose: () => void;
}) {
  const { data: settings, isPending: loadingSettings } = useQuery({
    queryKey: ['workspaces', workspaceId, 'llm', 'settings'],
    queryFn: () => llmApi.getSettings(workspaceId ?? ''),
    enabled: !!workspaceId,
  });
  const { data: consultations, isPending: loadingC } = useQuery({
    queryKey: ['workspaces', workspaceId, 'llm', 'consultations', 'recent'],
    queryFn: () => llmApi.listConsultations(workspaceId ?? '', 5),
    enabled: !!workspaceId && !!settings,
  });

  return (
    <div
      className="fixed inset-y-0 right-0 z-50 w-[380px] bg-white dark:bg-ink-900 shadow-soft-3 border-l border-ink-200 dark:border-ink-800 flex flex-col"
      role="complementary"
      aria-label="AI Advisor panel"
    >
      <header className="flex items-center justify-between px-5 py-4 border-b border-ink-200 dark:border-ink-800">
        <h2 className="h-section">AI Advisor</h2>
        <button
          className="text-ink-500 hover:text-ink-900 dark:hover:text-ink-50 transition-colors text-lg leading-none"
          onClick={onClose}
          aria-label="Close"
        >
          ×
        </button>
      </header>

      <div className="flex-1 overflow-y-auto px-5 py-4 space-y-5">
        {!workspaceId ? (
          <p className="hint">
            Open a workspace to use the AI Advisor — recent insights are scoped per workspace.
          </p>
        ) : loadingSettings ? (
          <p className="hint">Checking provider…</p>
        ) : !settings ? (
          <ConnectPrompt workspaceId={workspaceId} />
        ) : (
          <>
            <section>
              <p className="text-xs uppercase tracking-wider text-ink-500 mb-1">
                Provider
              </p>
              <p className="text-sm font-mono text-ink-900">
                {settings.provider} · {settings.model_name}
              </p>
              <p className="hint mt-1">
                {settings.has_api_key
                  ? 'Key encrypted at rest.'
                  : 'No API key set — advisories disabled.'}
              </p>
            </section>

            <section>
              <p className="text-xs uppercase tracking-wider text-ink-500 mb-2">
                Recent consultations
              </p>
              {loadingC ? (
                <p className="hint">Loading…</p>
              ) : !consultations || consultations.length === 0 ? (
                <p className="hint">
                  No advisories run yet. Try the dataset consultant on a CSV upload, or the
                  experiment designer on{' '}
                  <Link
                    to={`/workspaces/${workspaceId}`}
                    onClick={onClose}
                    className="text-accent-600"
                  >
                    a new experiment
                  </Link>
                  .
                </p>
              ) : (
                <ul className="space-y-2">
                  {consultations.map((c) => (
                    <li
                      key={c.id}
                      className="card text-xs"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-ink-900">
                          {TYPE_LABELS[c.type] ?? c.type}
                        </span>
                        <span className="text-ink-500">
                          {c.latency_ms ? `${c.latency_ms.toFixed(0)}ms` : '—'}
                        </span>
                      </div>
                      <p className="text-ink-600 mt-1 line-clamp-2">
                        {c.response_json?.reasoning_summary ?? c.error ?? ''}
                      </p>
                      <p className="text-ink-400 font-mono mt-1">
                        {new Date(c.created_at).toLocaleString()}
                      </p>
                    </li>
                  ))}
                </ul>
              )}
            </section>

            <section>
              <p className="text-xs uppercase tracking-wider text-ink-500 mb-2">Quick jump</p>
              <ul className="space-y-1 text-sm">
                <li>
                  <Link
                    to={`/workspaces/${workspaceId}`}
                    onClick={onClose}
                    className="text-accent-600 hover:underline"
                  >
                    → Workspace home (dataset consultant + designer)
                  </Link>
                </li>
                <li>
                  <Link
                    to={`/workspaces/${workspaceId}/llm`}
                    onClick={onClose}
                    className="text-accent-600 hover:underline"
                  >
                    → Provider settings
                  </Link>
                </li>
                <li>
                  <Link
                    to={`/workspaces/${workspaceId}/audit-logs`}
                    onClick={onClose}
                    className="text-accent-600 hover:underline"
                  >
                    → Full consultation history (audit log)
                  </Link>
                </li>
              </ul>
            </section>
          </>
        )}
      </div>
    </div>
  );
}

function ConnectPrompt({ workspaceId }: { workspaceId: string }) {
  return (
    <div className="space-y-3">
      <p className="text-sm text-ink-700">
        No LLM provider configured for this workspace yet.
      </p>
      <p className="hint">
        The 6 AI advisories — dataset consultant, experiment designer, run explainer,
        failure debugger, deployment reviewer, drift analyst — all route through the
        provider you configure here.
      </p>
      <Link
        to={`/workspaces/${workspaceId}/llm`}
        className="btn-primary inline-block"
      >
        Connect a provider
      </Link>
    </div>
  );
}
