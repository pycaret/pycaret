/**
 * Modal that drives the experiment-designer LLM consultation.
 *
 * Inputs:
 *  - a dataset picker (workspace's CSV uploads — mandatory, since the LLM
 *    reads the column profile)
 *  - a free-text goal ("predict churn", "baseline regression on prices")
 *
 * Output: renders the standard `LLMAdvice` envelope + the suggested
 * `RunConfig`-shaped JSON. No one-click apply in v1 (waits on the canonical
 * `RunConfig` Pydantic model, MVP-1 exit criterion).
 */

import { useMemo, useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { dataSourcesApi, llmApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';

export interface ExperimentDesignerModalProps {
  workspaceId: string;
  open: boolean;
  onClose: () => void;
}

export function ExperimentDesignerModal({
  workspaceId,
  open,
  onClose,
}: ExperimentDesignerModalProps) {
  // Only CSV uploads can be analysed — the consultation needs a column profile.
  const sources = useQuery({
    queryKey: ['data-sources', workspaceId],
    queryFn: () => dataSourcesApi.list(workspaceId),
    enabled: open && !!workspaceId,
  });
  const csvs = useMemo(
    () => (sources.data ?? []).filter((d) => d.kind === 'csv_upload'),
    [sources.data],
  );

  const [dataSourceId, setDataSourceId] = useState('');
  const [goal, setGoal] = useState('');

  const design = useMutation({
    mutationFn: () =>
      llmApi.designExperiment({
        workspace_id: workspaceId,
        data_source_id: dataSourceId,
        goal: goal.trim(),
      }),
  });

  if (!open) return null;

  const advice = design.data?.response_json;
  const canSubmit =
    dataSourceId.length > 0 && goal.trim().length > 0 && !design.isPending;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-white/80 px-4"
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
            <h2 className="text-sm font-medium text-ink-900">AI experiment designer</h2>
            <p className="hint mt-1">
              Describe what you want to do + pick a dataset; the LLM proposes a
              full RunConfig.
            </p>
          </div>
          <button
            onClick={onClose}
            className="btn-ghost text-xs"
            aria-label="Close"
          >
            ✕
          </button>
        </header>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (canSubmit) design.mutate();
          }}
          className="space-y-4"
        >
          <div>
            <label className="field" htmlFor="ds">
              Dataset <span className="text-danger-500">*</span>
            </label>
            <select
              id="ds"
              className="input"
              value={dataSourceId}
              onChange={(e) => setDataSourceId(e.target.value)}
              required
            >
              <option value="">— pick an uploaded CSV —</option>
              {csvs.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name}
                </option>
              ))}
            </select>
            {csvs.length === 0 && sources.isFetched && (
              <p className="hint mt-1">
                No CSVs in this workspace yet. Upload one from the workspace
                home page, then come back.
              </p>
            )}
          </div>

          <div>
            <label className="field" htmlFor="goal">
              Goal <span className="text-danger-500">*</span>
            </label>
            <textarea
              id="goal"
              rows={3}
              className="input"
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              placeholder="e.g. Predict churn with a high-recall model; compare logistic + tree-based baselines."
              required
            />
          </div>

          {design.error && <p className="error">{errorMessage(design.error)}</p>}

          <button type="submit" className="btn-primary w-full" disabled={!canSubmit}>
            {design.isPending ? 'Designing…' : 'Design experiment'}
          </button>
        </form>

        {advice && (
          <section className="mt-6 space-y-5 border-t border-ink-200 pt-6">
            {advice.suggested_action && (
              <div>
                <h3 className="text-xs uppercase tracking-wider text-ink-500 mb-1">
                  Suggested action
                </h3>
                <p className="text-ink-900">{advice.suggested_action}</p>
              </div>
            )}

            {advice.reasoning_summary && (
              <div>
                <h3 className="text-xs uppercase tracking-wider text-ink-500 mb-1">
                  Reasoning
                </h3>
                <p className="text-sm text-ink-600 whitespace-pre-wrap">
                  {advice.reasoning_summary}
                </p>
              </div>
            )}

            {advice.risk_flags.length > 0 && (
              <div>
                <h3 className="text-xs uppercase tracking-wider text-ink-500 mb-1">
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
              </div>
            )}

            <div>
              <h3 className="text-xs uppercase tracking-wider text-ink-500 mb-1">
                Suggested RunConfig
              </h3>
              <pre className="bg-white border border-ink-200 rounded p-3 font-mono text-xs text-ink-900 overflow-x-auto">
                {JSON.stringify(advice.suggested_config_json, null, 2)}
              </pre>
              <p className="hint mt-2">
                Advisory — copy values into the wizard below. A one-click
                apply lands once <code className="font-mono">RunConfig</code>{' '}
                is a first-class Pydantic shape (ROADMAP MVP-1 exit).
              </p>
            </div>

            {design.data && (
              <p className="hint">
                {design.data.provider} · {design.data.model_name} ·{' '}
                {design.data.latency_ms?.toFixed(0)}ms
              </p>
            )}
          </section>
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
