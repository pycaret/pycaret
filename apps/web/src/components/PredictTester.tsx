/**
 * Inline /predict tester for a single deployment.
 *
 * Contract (services/api/pycaret_server/api/deployments.py § predict):
 *   POST /api/v1/deployments/{slug}/predict
 *   body: {"rows": [{col1: v1, col2: v2, ...}, ...]}
 *   response: {predictions: [{index, prediction}], latency_ms, ...}
 *
 * We keep the input as a raw JSON-array textarea rather than a dynamic form:
 *   - Schema discovery for arbitrary uploaded CSVs isn't something the
 *     deployment row currently tracks.
 *   - Advanced users want to paste bulk predictions.
 *   - A future session adds a column-aware form on top (leveraging the
 *     Pipeline's schema artifact).
 */

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { deploymentsApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';

const DEFAULT_INPUT = `[
  {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2},
  {"sepal length (cm)": 6.2, "sepal width (cm)": 3.4, "petal length (cm)": 5.4, "petal width (cm)": 2.3}
]`;

export interface PredictTesterProps {
  endpointSlug: string;
}

export function PredictTester({ endpointSlug }: PredictTesterProps) {
  const [text, setText] = useState(DEFAULT_INPUT);
  const [parseError, setParseError] = useState<string | null>(null);

  const predict = useMutation({
    mutationFn: () => {
      let rows: Record<string, unknown>[];
      try {
        const parsed = JSON.parse(text);
        if (!Array.isArray(parsed)) {
          throw new Error('Input must be a JSON array of row objects.');
        }
        rows = parsed as Record<string, unknown>[];
      } catch (e) {
        throw new Error((e as Error).message || 'Invalid JSON.');
      }
      return deploymentsApi.predict(endpointSlug, { rows });
    },
  });

  const handleChange = (next: string) => {
    setText(next);
    if (!next.trim()) {
      setParseError(null);
      return;
    }
    // Live-validate only as a hint; the actual parse happens in mutate so
    // errors show up in the same error slot as API errors.
    try {
      const parsed = JSON.parse(next);
      if (!Array.isArray(parsed)) throw new Error('must be a JSON array');
      setParseError(null);
    } catch (e) {
      setParseError((e as Error).message);
    }
  };

  return (
    <div className="card space-y-4">
      <h2 className="text-sm font-medium text-ink-100">Test a prediction</h2>

      <div>
        <label className="field" htmlFor="rows">
          Rows (JSON array)
        </label>
        <textarea
          id="rows"
          rows={10}
          className="input font-mono text-xs"
          value={text}
          onChange={(e) => handleChange(e.target.value)}
          spellCheck={false}
        />
        {parseError && <p className="hint text-danger-500 mt-1">JSON: {parseError}</p>}
        <p className="hint mt-1">
          Each row is an object with column names matching the model's training
          features. Default payload is the iris schema.
        </p>
      </div>

      {predict.error && <p className="error">{errorMessage(predict.error)}</p>}

      <button
        className="btn-primary w-full"
        disabled={predict.isPending || !!parseError}
        onClick={() => predict.mutate()}
      >
        {predict.isPending ? 'Predicting…' : 'Send request'}
      </button>

      {predict.data && (
        <section className="space-y-3">
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium text-ink-100">Response</span>
            <span className="font-mono text-xs text-ink-200/60 tabular-nums">
              {predict.data.latency_ms.toFixed(1)}ms · request{' '}
              <span title={predict.data.request_id}>
                {predict.data.request_id.slice(0, 8)}…
              </span>
            </span>
          </div>
          <div className="card p-0 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-ink-800 text-ink-200/70">
                <tr>
                  <th className="px-3 py-2 text-left font-medium">Index</th>
                  <th className="px-3 py-2 text-left font-medium">Prediction</th>
                </tr>
              </thead>
              <tbody>
                {predict.data.predictions.map((p) => (
                  <tr
                    key={p.index}
                    className="border-t border-ink-800 hover:bg-ink-800/50"
                  >
                    <td className="px-3 py-2 font-mono text-xs tabular-nums">
                      {p.index}
                    </td>
                    <td className="px-3 py-2 font-mono text-sm">
                      {typeof p.prediction === 'object'
                        ? JSON.stringify(p.prediction)
                        : String(p.prediction)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
