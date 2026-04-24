/**
 * Surfaces drift reports for a given Deployment: lists existing snapshots,
 * lets an admin record a new one by pasting feature-drift + prediction-drift
 * JSON, and opens `<DriftAnalysisModal>` for any row.
 *
 * v1 design note: we don't compute drift server-side yet (no prediction log
 * + no job queue). The UI lets the caller paste JSON so CI tools + notebooks
 * can POST reports programmatically and still have a place to see them in
 * the app.
 */

import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { driftApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import type { DriftReportRead } from '@/api/types';
import { DriftAnalysisModal } from './DriftAnalysisModal';

const STATUS_TONE: Record<string, string> = {
  none: 'text-success-500',
  mild: 'text-ink-200',
  moderate: 'text-warn-500',
  severe: 'text-danger-500',
};

export interface DriftReportsCardProps {
  deploymentId: string;
}

/** Default textarea content — guides the user to the JSON shape the server expects. */
const FEATURE_DRIFT_PLACEHOLDER = JSON.stringify(
  { amount: { score: 0.31, kind: 'psi' }, age: { score: 0.05, kind: 'ks' } },
  null,
  2,
);
const PREDICTION_DRIFT_PLACEHOLDER = JSON.stringify(
  { kind: 'js', score: 0.02 },
  null,
  2,
);

export function DriftReportsCard({ deploymentId }: DriftReportsCardProps) {
  const qc = useQueryClient();
  const reports = useQuery({
    queryKey: ['drift-reports', deploymentId],
    queryFn: () => driftApi.list(deploymentId),
    enabled: !!deploymentId,
  });

  // ─────────────── form state
  const [score, setScore] = useState('0.2');
  const [sampleSize, setSampleSize] = useState('');
  const [featureJson, setFeatureJson] = useState(FEATURE_DRIFT_PLACEHOLDER);
  const [predictionJson, setPredictionJson] = useState(PREDICTION_DRIFT_PLACEHOLDER);
  const [formError, setFormError] = useState<string | null>(null);
  const [formOpen, setFormOpen] = useState(false);

  const create = useMutation({
    mutationFn: () => {
      setFormError(null);
      const numericScore = Number(score);
      if (!Number.isFinite(numericScore) || numericScore < 0 || numericScore > 1) {
        throw new Error('drift_score must be a number in [0, 1]');
      }
      let featureParsed: Record<string, unknown>;
      try {
        featureParsed = JSON.parse(featureJson) as Record<string, unknown>;
      } catch {
        throw new Error('feature_drift_json is not valid JSON');
      }
      let predictionParsed: Record<string, unknown> | null = null;
      if (predictionJson.trim()) {
        try {
          predictionParsed = JSON.parse(predictionJson) as Record<string, unknown>;
        } catch {
          throw new Error('prediction_drift_json is not valid JSON');
        }
      }
      const now = new Date();
      const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
      return driftApi.create(deploymentId, {
        window_start: weekAgo.toISOString(),
        window_end: now.toISOString(),
        drift_score: numericScore,
        // The Pydantic validator on the server enforces shape; we only
        // sanity-check the parse here.
        feature_drift_json: featureParsed as Record<
          string,
          { score: number; kind: 'psi' | 'ks' | 'chi2' | 'missing_rate' }
        >,
        prediction_drift_json: predictionParsed as
          | { kind: 'js' | 'ks'; score: number }
          | null,
        sample_size: sampleSize.trim() ? Number(sampleSize) : null,
      });
    },
    onSuccess: () => {
      setFormOpen(false);
      qc.invalidateQueries({ queryKey: ['drift-reports', deploymentId] });
    },
    onError: (err: Error) => setFormError(err.message),
  });

  // ─────────────── modal state
  const [analyzing, setAnalyzing] = useState<DriftReportRead | null>(null);

  return (
    <section>
      <header className="mb-3 flex items-baseline justify-between">
        <h2 className="text-sm font-medium text-ink-100">Drift reports</h2>
        <div className="flex items-center gap-3">
          <span className="hint">{reports.data?.length ?? 0} total</span>
          <button
            className="btn-secondary text-xs"
            onClick={() => setFormOpen((v) => !v)}
          >
            {formOpen ? 'Cancel' : 'Record snapshot'}
          </button>
        </div>
      </header>

      {formOpen && (
        <div className="card mb-4 space-y-3">
          <p className="hint">
            v1 drift reports are submitted explicitly. Paste the feature-level
            drift breakdown (PSI / KS / chi² / missing-rate) + the optional
            prediction-distribution shift. The server buckets{' '}
            <code>drift_status</code> from <code>drift_score</code>.
          </p>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              create.mutate();
            }}
            className="grid gap-3 md:grid-cols-2"
          >
            <div>
              <label className="field" htmlFor="drift-score">
                drift_score (0–1)
              </label>
              <input
                id="drift-score"
                className="input"
                value={score}
                onChange={(e) => setScore(e.target.value)}
                inputMode="decimal"
                placeholder="0.2"
              />
            </div>
            <div>
              <label className="field" htmlFor="drift-sample">
                sample_size
              </label>
              <input
                id="drift-sample"
                className="input"
                value={sampleSize}
                onChange={(e) => setSampleSize(e.target.value)}
                inputMode="numeric"
                placeholder="e.g. 400"
              />
            </div>
            <div className="md:col-span-2">
              <label className="field" htmlFor="drift-feature-json">
                feature_drift_json
              </label>
              <textarea
                id="drift-feature-json"
                rows={5}
                className="input font-mono text-xs resize-y"
                value={featureJson}
                onChange={(e) => setFeatureJson(e.target.value)}
              />
            </div>
            <div className="md:col-span-2">
              <label className="field" htmlFor="drift-prediction-json">
                prediction_drift_json (optional)
              </label>
              <textarea
                id="drift-prediction-json"
                rows={3}
                className="input font-mono text-xs resize-y"
                value={predictionJson}
                onChange={(e) => setPredictionJson(e.target.value)}
              />
            </div>
            {formError && (
              <p className="error md:col-span-2">{formError}</p>
            )}
            {create.error && !formError && (
              <p className="error md:col-span-2">{errorMessage(create.error)}</p>
            )}
            <div className="md:col-span-2 flex justify-end gap-2">
              <button
                type="button"
                className="btn-secondary"
                onClick={() => setFormOpen(false)}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="btn-primary"
                disabled={create.isPending}
              >
                {create.isPending ? 'Recording…' : 'Record'}
              </button>
            </div>
          </form>
        </div>
      )}

      {reports.isLoading && <p className="hint">Loading…</p>}
      {reports.error && <p className="error">{errorMessage(reports.error)}</p>}

      {reports.data && reports.data.length === 0 && (
        <div className="card text-sm text-ink-200/70">
          No drift reports yet. Record one manually above, or POST to{' '}
          <code className="font-mono text-xs">
            /api/v1/deployments/{deploymentId}/drift-reports
          </code>{' '}
          from a CI job.
        </div>
      )}

      {reports.data && reports.data.length > 0 && (
        <div className="card p-0 overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-ink-800 text-ink-200/70">
              <tr>
                <th className="px-4 py-2 text-left font-medium">Window</th>
                <th className="px-4 py-2 text-left font-medium">Score</th>
                <th className="px-4 py-2 text-left font-medium">Status</th>
                <th className="px-4 py-2 text-left font-medium">Sample</th>
                <th className="px-4 py-2 text-right font-medium">Action</th>
              </tr>
            </thead>
            <tbody>
              {reports.data.map((r) => (
                <tr
                  key={r.id}
                  className="border-t border-ink-800 hover:bg-ink-800/50"
                >
                  <td className="px-4 py-2 text-xs text-ink-200/70 font-mono">
                    {new Date(r.window_start).toLocaleDateString()} →{' '}
                    {new Date(r.window_end).toLocaleDateString()}
                  </td>
                  <td className="px-4 py-2 font-mono tabular-nums">
                    {r.drift_score.toFixed(3)}
                  </td>
                  <td className="px-4 py-2">
                    <span
                      className={`font-mono text-xs ${STATUS_TONE[r.drift_status] ?? ''}`}
                    >
                      {r.drift_status}
                    </span>
                  </td>
                  <td className="px-4 py-2 font-mono text-xs tabular-nums">
                    {r.sample_size ?? '—'}
                  </td>
                  <td className="px-4 py-2 text-right">
                    <button
                      className="btn-ghost text-xs"
                      onClick={() => setAnalyzing(r)}
                    >
                      ✨ Analyze
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {analyzing && (
        <DriftAnalysisModal
          report={analyzing}
          open={!!analyzing}
          onClose={() => setAnalyzing(null)}
        />
      )}
    </section>
  );
}
