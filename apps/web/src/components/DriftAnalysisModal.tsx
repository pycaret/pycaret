/**
 * Modal that runs the drift-analysis LLM consultation on a specific
 * DriftReport. Same structural pattern as `<DeploymentReviewModal>` —
 * auto-fires on open, renders the canonical `LLMAdvice` envelope,
 * tone-codes the verdict.
 */

import { useEffect } from 'react';
import { useMutation } from '@tanstack/react-query';
import { llmApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import type { DriftReportRead } from '@/api/types';

export interface DriftAnalysisModalProps {
  report: DriftReportRead;
  open: boolean;
  onClose: () => void;
}

/** Map the 4 verdict prefixes the analyst prompt promises. */
function verdictTone(suggestedAction: string): string {
  const v = suggestedAction.toUpperCase();
  if (v.startsWith('RETRAIN NOW')) return 'text-danger-500';
  if (v.startsWith('INVESTIGATE')) return 'text-warn-500';
  if (v.startsWith('MONITOR')) return 'text-ink-700';
  if (v.startsWith('NO ACTION')) return 'text-success-500';
  return 'text-ink-900';
}

export function DriftAnalysisModal({ report, open, onClose }: DriftAnalysisModalProps) {
  const analyze = useMutation({
    mutationFn: () => llmApi.analyzeDrift({ drift_report_id: report.id }),
  });

  useEffect(() => {
    if (open && !analyze.isPending && !analyze.data && !analyze.error) {
      analyze.mutate();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  useEffect(() => {
    if (!open) analyze.reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  if (!open) return null;

  const advice = analyze.data?.response_json;
  const tone = advice?.suggested_action ? verdictTone(advice.suggested_action) : '';

  // Sort features by drift score descending — helps the user spot the
  // dominant drivers without reading every row.
  const sortedFeatures = Object.entries(report.feature_drift_json ?? {}).sort(
    ([, a], [, b]) => (b?.score ?? 0) - (a?.score ?? 0),
  );

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-white/80 px-4"
      role="dialog"
      aria-modal="true"
      onClick={onClose}
    >
      <div
        className="relative card max-w-3xl w-full max-h-[85vh] overflow-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-start justify-between gap-4 mb-4">
          <div>
            <h2 className="text-sm font-medium text-ink-900">✨ Drift analysis</h2>
            <p className="hint mt-1">
              Report from{' '}
              <span className="font-mono text-ink-900">
                {new Date(report.window_start).toLocaleDateString()}
              </span>{' '}
              to{' '}
              <span className="font-mono text-ink-900">
                {new Date(report.window_end).toLocaleDateString()}
              </span>{' '}
              · overall score {report.drift_score.toFixed(3)} ({report.drift_status})
            </p>
          </div>
          <button onClick={onClose} className="btn-ghost text-xs" aria-label="Close">
            ✕
          </button>
        </header>

        {/* ─────────────── the snapshot the analyst is reading from */}
        <section className="mb-5">
          <h3 className="text-xs uppercase tracking-wider text-ink-500 mb-2">
            Feature drift snapshot
          </h3>
          {sortedFeatures.length === 0 ? (
            <p className="hint">No per-feature breakdown recorded.</p>
          ) : (
            <table className="w-full text-xs">
              <thead className="text-ink-500">
                <tr>
                  <th className="text-left font-medium py-1">Feature</th>
                  <th className="text-left font-medium py-1">Kind</th>
                  <th className="text-right font-medium py-1">Score</th>
                </tr>
              </thead>
              <tbody>
                {sortedFeatures.map(([name, entry]) => (
                  <tr key={name} className="border-t border-ink-200">
                    <td className="py-1 font-mono text-ink-900">{name}</td>
                    <td className="py-1 font-mono text-ink-500">{entry.kind}</td>
                    <td className="py-1 text-right font-mono">
                      {entry.score.toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        {/* ─────────────── LLM output */}
        {analyze.isPending && <p className="hint">Asking the analyst…</p>}
        {analyze.error && <p className="error">{errorMessage(analyze.error)}</p>}

        {advice && (
          <div className="space-y-5">
            {advice.suggested_action && (
              <section>
                <h3 className="text-xs uppercase tracking-wider text-ink-500 mb-1">
                  Verdict
                </h3>
                <p className={`text-lg font-medium ${tone}`}>{advice.suggested_action}</p>
              </section>
            )}

            {advice.reasoning_summary && (
              <section>
                <h3 className="text-xs uppercase tracking-wider text-ink-500 mb-1">
                  Reasoning
                </h3>
                <p className="text-sm text-ink-600 whitespace-pre-wrap">
                  {advice.reasoning_summary}
                </p>
              </section>
            )}

            {advice.risk_flags.length > 0 && (
              <section>
                <h3 className="text-xs uppercase tracking-wider text-ink-500 mb-1">
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
                <h3 className="text-xs uppercase tracking-wider text-ink-500 mb-1">
                  Hints
                </h3>
                <pre className="bg-white border border-ink-200 rounded p-3 font-mono text-xs text-ink-900 overflow-x-auto">
                  {JSON.stringify(advice.suggested_config_json, null, 2)}
                </pre>
              </section>
            )}

            {analyze.data && (
              <p className="hint pt-3 border-t border-ink-200">
                {analyze.data.provider} · {analyze.data.model_name} ·{' '}
                {analyze.data.latency_ms?.toFixed(0)}ms
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
