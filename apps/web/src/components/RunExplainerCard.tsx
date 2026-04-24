/**
 * Inline "Explain this run" card for the RunDetail screen.
 *
 * Deliberately a *card*, not a modal — the explanation is long-form prose
 * that sits alongside the leaderboard rather than covering it. Users open
 * it by clicking the button; the LLM call only fires on demand so we
 * don't run (and bill) every run view.
 *
 * Only rendered on terminal-state runs (succeeded / failed / cancelled);
 * the parent screen guards that.
 */

import { useMutation } from '@tanstack/react-query';
import { llmApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';

export interface RunExplainerCardProps {
  runId: string;
}

export function RunExplainerCard({ runId }: RunExplainerCardProps) {
  const explain = useMutation({
    mutationFn: () => llmApi.explainRun({ run_id: runId }),
  });

  const advice = explain.data?.response_json;
  const nextActions =
    (advice?.suggested_config_json?.next_actions as string[] | undefined) ?? [];

  return (
    <section className="card space-y-4">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-sm font-medium text-ink-100">✨ AI explanation</h2>
          <p className="hint mt-1">
            Reads the leaderboard + event stream and proposes next experiments.
            Advisory — your call what to run next.
          </p>
        </div>
        <button
          className="btn-secondary shrink-0"
          onClick={() => explain.mutate()}
          disabled={explain.isPending}
        >
          {explain.isPending ? 'Thinking…' : advice ? 'Re-explain' : 'Explain'}
        </button>
      </header>

      {explain.error && <p className="error">{errorMessage(explain.error)}</p>}

      {advice && (
        <div className="space-y-4 border-t border-ink-800 pt-4">
          {advice.suggested_action && (
            <div>
              <h3 className="text-xs uppercase tracking-wider text-ink-200/60 mb-1">
                Next step
              </h3>
              <p className="text-ink-100">{advice.suggested_action}</p>
            </div>
          )}

          {advice.reasoning_summary && (
            <div>
              <h3 className="text-xs uppercase tracking-wider text-ink-200/60 mb-1">
                What happened
              </h3>
              <p className="text-sm text-ink-200/80 whitespace-pre-wrap">
                {advice.reasoning_summary}
              </p>
            </div>
          )}

          {nextActions.length > 0 && (
            <div>
              <h3 className="text-xs uppercase tracking-wider text-ink-200/60 mb-1">
                Ideas to try
              </h3>
              <ul className="text-sm text-ink-200/80 list-disc list-inside space-y-1">
                {nextActions.map((a) => (
                  <li key={a}>
                    <code className="font-mono text-xs text-ink-100">{a}</code>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {advice.risk_flags.length > 0 && (
            <div>
              <h3 className="text-xs uppercase tracking-wider text-ink-200/60 mb-1">
                Flags
              </h3>
              <div className="flex flex-wrap gap-1">
                {advice.risk_flags.map((r) => (
                  <span key={r} className="kbd text-warn-500 border-warn-500/50">
                    {r}
                  </span>
                ))}
              </div>
            </div>
          )}

          {explain.data && (
            <p className="hint">
              {explain.data.provider} · {explain.data.model_name} ·{' '}
              {explain.data.latency_ms?.toFixed(0)}ms
            </p>
          )}
        </div>
      )}
    </section>
  );
}
