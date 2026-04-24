/**
 * Modal that runs the deployment-risk-review LLM consultation on a
 * specific Pipeline before the user deploys it.
 *
 * Contract: fires once on open (auto-mutation), renders the standard
 * `LLMAdvice` envelope with the verdict prominently displayed. The UI
 * does NOT block the Deploy button; the reviewer is advisory per SPEC
 * § 12.3.
 */

import { useEffect } from 'react';
import { useMutation } from '@tanstack/react-query';
import { llmApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';

export interface DeploymentReviewModalProps {
  pipelineId: string;
  pipelineName: string;
  open: boolean;
  onClose: () => void;
}

/** Classify the verdict string the LLM returned, for tone-coded rendering. */
function verdictTone(suggestedAction: string): string {
  const v = suggestedAction.toUpperCase();
  if (v.startsWith('DO NOT DEPLOY')) return 'text-danger-500';
  if (v.startsWith('APPROVE WITH CAVEATS')) return 'text-warn-500';
  if (v.startsWith('APPROVE')) return 'text-success-500';
  return 'text-ink-100';
}

export function DeploymentReviewModal({
  pipelineId,
  pipelineName,
  open,
  onClose,
}: DeploymentReviewModalProps) {
  const review = useMutation({
    mutationFn: () => llmApi.reviewDeployment({ pipeline_id: pipelineId }),
  });

  useEffect(() => {
    if (open && !review.isPending && !review.data && !review.error) {
      review.mutate();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  useEffect(() => {
    if (!open) review.reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  if (!open) return null;

  const advice = review.data?.response_json;
  const tone = advice?.suggested_action ? verdictTone(advice.suggested_action) : '';

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink-950/80 px-4"
      role="dialog"
      aria-modal="true"
      onClick={onClose}
    >
      <div
        className="relative card max-w-2xl w-full max-h-[85vh] overflow-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-start justify-between gap-4 mb-4">
          <div>
            <h2 className="text-sm font-medium text-ink-100">
              ✨ Pre-deploy risk review
            </h2>
            <p className="hint mt-1">
              Reviewing <code className="font-mono text-ink-100">{pipelineName}</code>
            </p>
          </div>
          <button onClick={onClose} className="btn-ghost text-xs" aria-label="Close">
            ✕
          </button>
        </header>

        {review.isPending && (
          <p className="hint">Analysing pipeline + training leaderboard…</p>
        )}
        {review.error && <p className="error">{errorMessage(review.error)}</p>}

        {advice && (
          <div className="space-y-5">
            {advice.suggested_action && (
              <section>
                <h3 className="text-xs uppercase tracking-wider text-ink-200/60 mb-1">
                  Verdict
                </h3>
                <p className={`text-lg font-medium ${tone}`}>{advice.suggested_action}</p>
              </section>
            )}

            {advice.reasoning_summary && (
              <section>
                <h3 className="text-xs uppercase tracking-wider text-ink-200/60 mb-1">
                  Review
                </h3>
                <p className="text-sm text-ink-200/80 whitespace-pre-wrap">
                  {advice.reasoning_summary}
                </p>
              </section>
            )}

            {advice.risk_flags.length > 0 && (
              <section>
                <h3 className="text-xs uppercase tracking-wider text-ink-200/60 mb-1">
                  Risks flagged
                </h3>
                <div className="flex flex-wrap gap-1">
                  {advice.risk_flags.map((r) => (
                    <span key={r} className="kbd text-warn-500 border-warn-500/50">
                      {r}
                    </span>
                  ))}
                </div>
              </section>
            )}

            {Object.keys(advice.suggested_config_json).length > 0 && (
              <section>
                <h3 className="text-xs uppercase tracking-wider text-ink-200/60 mb-1">
                  Deployment hints
                </h3>
                <pre className="bg-ink-950 border border-ink-800 rounded p-3 font-mono text-xs text-ink-100 overflow-x-auto">
                  {JSON.stringify(advice.suggested_config_json, null, 2)}
                </pre>
              </section>
            )}

            {review.data && (
              <p className="hint pt-3 border-t border-ink-800">
                {review.data.provider} · {review.data.model_name} ·{' '}
                {review.data.latency_ms?.toFixed(0)}ms
              </p>
            )}
          </div>
        )}

        <footer className="mt-6 flex justify-end">
          <button onClick={onClose} className="btn-secondary">
            Close
          </button>
        </footer>
      </div>
    </div>
  );
}
