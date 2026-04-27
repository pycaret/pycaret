/**
 * Inline "Debug this failure" card for the RunDetail screen.
 *
 * Shown only on `run.status === 'failed'`. Opt-in: the LLM call fires on
 * button click, not on page load. Same UX shape as RunExplainerCard —
 * both route through different consultation types (`run_summary` vs.
 * `failure_debugging`) but present identical envelopes.
 */

import { useMutation } from '@tanstack/react-query';
import { llmApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';

export interface FailureDebuggerCardProps {
  runId: string;
}

export function FailureDebuggerCard({ runId }: FailureDebuggerCardProps) {
  const debug = useMutation({
    mutationFn: () => llmApi.debugRun({ run_id: runId }),
  });

  const advice = debug.data?.response_json;

  return (
    <section className="card space-y-4 border-danger-500/30">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-sm font-medium text-ink-900">✨ AI diagnosis</h2>
          <p className="hint mt-1">
            Reads the error + event tail and suggests a minimal fix. Advisory
            — check the reasoning before retrying.
          </p>
        </div>
        <button
          className="btn-secondary shrink-0"
          onClick={() => debug.mutate()}
          disabled={debug.isPending}
        >
          {debug.isPending ? 'Diagnosing…' : advice ? 'Re-diagnose' : 'Diagnose'}
        </button>
      </header>

      {debug.error && <p className="error">{errorMessage(debug.error)}</p>}

      {advice && (
        <div className="space-y-4 border-t border-ink-200 pt-4">
          {advice.suggested_action && (
            <div>
              <h3 className="text-xs uppercase tracking-wider text-ink-500 mb-1">
                Suggested fix
              </h3>
              <p className="text-ink-900">{advice.suggested_action}</p>
            </div>
          )}

          {advice.reasoning_summary && (
            <div>
              <h3 className="text-xs uppercase tracking-wider text-ink-500 mb-1">
                Diagnosis
              </h3>
              <p className="text-sm text-ink-600 whitespace-pre-wrap">
                {advice.reasoning_summary}
              </p>
            </div>
          )}

          {advice.risk_flags.length > 0 && (
            <div>
              <h3 className="text-xs uppercase tracking-wider text-ink-500 mb-1">
                Caveats
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

          {Object.keys(advice.suggested_config_json).length > 0 && (
            <div>
              <h3 className="text-xs uppercase tracking-wider text-ink-500 mb-1">
                Suggested config
              </h3>
              <pre className="bg-white border border-ink-200 rounded p-3 font-mono text-xs text-ink-900 overflow-x-auto">
                {JSON.stringify(advice.suggested_config_json, null, 2)}
              </pre>
            </div>
          )}

          {debug.data && (
            <p className="hint">
              {debug.data.provider} · {debug.data.model_name} ·{' '}
              {debug.data.latency_ms?.toFixed(0)}ms
            </p>
          )}
        </div>
      )}
    </section>
  );
}
