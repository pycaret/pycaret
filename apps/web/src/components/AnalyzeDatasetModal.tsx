/**
 * Modal dialog that drives the dataset-consultant LLM call.
 *
 * Opens with the given `dataSourceId` + `workspaceId`, fires
 * `llmApi.analyzeDataset`, renders the returned `LLMAdvice` envelope:
 *   - suggested_action as a headline
 *   - reasoning_summary below
 *   - risk_flags as chips
 *   - suggested_config_json as a pretty-printed JSON block
 *
 * The LLM is advisory. The modal deliberately does *not* offer a
 * "Apply these settings" button — that's a V2 feature once the
 * RunConfig Pydantic model (ROADMAP MVP-1 exit) is live. For now
 * users copy + paste the suggested values into the setup wizard.
 */

import { useMutation } from '@tanstack/react-query';
import { llmApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import { useEffect } from 'react';

export interface AnalyzeDatasetModalProps {
  workspaceId: string;
  dataSourceId: string;
  dataSourceName: string;
  taskTypeHint?: string | null;
  open: boolean;
  onClose: () => void;
}

export function AnalyzeDatasetModal({
  workspaceId,
  dataSourceId,
  dataSourceName,
  taskTypeHint,
  open,
  onClose,
}: AnalyzeDatasetModalProps) {
  const analyze = useMutation({
    mutationFn: () =>
      llmApi.analyzeDataset({
        workspace_id: workspaceId,
        data_source_id: dataSourceId,
        task_type_hint: taskTypeHint ?? null,
      }),
  });

  useEffect(() => {
    if (open && !analyze.isPending && !analyze.data && !analyze.error) {
      analyze.mutate();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  useEffect(() => {
    // Reset when closed so re-opening re-fires.
    if (!open) analyze.reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  if (!open) return null;

  const advice = analyze.data?.response_json;

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
            <h2 className="text-sm font-medium text-ink-100">AI dataset consultant</h2>
            <p className="hint mt-1">
              Analyzing{' '}
              <code className="font-mono text-ink-100">{dataSourceName}</code>
            </p>
          </div>
          <button onClick={onClose} className="btn-ghost text-xs" aria-label="Close">
            ✕
          </button>
        </header>

        {analyze.isPending && (
          <p className="hint">Consulting the LLM… this usually takes a few seconds.</p>
        )}
        {analyze.error && <p className="error">{errorMessage(analyze.error)}</p>}

        {advice && (
          <div className="space-y-5">
            {advice.suggested_action && (
              <section>
                <h3 className="text-xs uppercase tracking-wider text-ink-200/60 mb-1">
                  Suggested action
                </h3>
                <p className="text-ink-100">{advice.suggested_action}</p>
              </section>
            )}

            {advice.reasoning_summary && (
              <section>
                <h3 className="text-xs uppercase tracking-wider text-ink-200/60 mb-1">
                  Reasoning
                </h3>
                <p className="text-sm text-ink-200/80 whitespace-pre-wrap">
                  {advice.reasoning_summary}
                </p>
              </section>
            )}

            {advice.risk_flags.length > 0 && (
              <section>
                <h3 className="text-xs uppercase tracking-wider text-ink-200/60 mb-1">
                  Risk flags
                </h3>
                <div className="flex flex-wrap gap-1">
                  {advice.risk_flags.map((r) => (
                    <span
                      key={r}
                      className="kbd text-warn-500 border-warn-500/50"
                    >
                      {r}
                    </span>
                  ))}
                </div>
              </section>
            )}

            <section>
              <h3 className="text-xs uppercase tracking-wider text-ink-200/60 mb-1">
                Suggested config
              </h3>
              <pre className="bg-ink-950 border border-ink-800 rounded p-3 font-mono text-xs text-ink-100 overflow-x-auto">
                {JSON.stringify(advice.suggested_config_json, null, 2)}
              </pre>
              <p className="hint mt-2">
                Advisory only. Copy values you like into the New Experiment wizard — a
                one-click apply button lands once `RunConfig` is a first-class shape
                (ROADMAP MVP-1 exit).
              </p>
            </section>

            {analyze.data && (
              <p className="hint pt-3 border-t border-ink-800">
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
