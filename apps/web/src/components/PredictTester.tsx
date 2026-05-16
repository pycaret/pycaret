/**
 * Inline /predict tester for a single deployment.
 *
 * Contract (services/api/pycaret_server/api/deployments.py § predict):
 *   POST /api/v1/deployments/{slug}/predict
 *   body: {"rows": [{col1: v1, col2: v2, ...}, ...]}
 *   response: {predictions: [{index, prediction}], latency_ms, ...}
 *
 * The textarea is pre-seeded with a real holdout row from the dataset
 * the underlying pipeline was trained on (via
 * GET /pipelines/:id/input-schema). Users can run a prediction without
 * knowing anything about the model's column schema. Iris fallback is
 * only used when no pipeline_id is wired (legacy callers / tests).
 */

import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { deploymentsApi, pipelinesApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';

const IRIS_FALLBACK = `[
  {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2},
  {"sepal length (cm)": 6.2, "sepal width (cm)": 3.4, "petal length (cm)": 5.4, "petal width (cm)": 2.3}
]`;

export interface PredictTesterProps {
  endpointSlug: string;
  /** Linked pipeline id; when present we fetch its input-schema and seed
   *  the textarea with a real holdout row instead of the iris fallback. */
  pipelineId?: string | null;
}

export function PredictTester({ endpointSlug, pipelineId }: PredictTesterProps) {
  const schema = useQuery({
    queryKey: ['pipelines', pipelineId, 'input-schema'],
    queryFn: () => pipelinesApi.inputSchema(pipelineId!),
    enabled: !!pipelineId,
    staleTime: 5 * 60 * 1000,
  });

  const seedJson = useMemo(() => {
    if (schema.data?.sample_row && Object.keys(schema.data.sample_row).length > 0) {
      return JSON.stringify([schema.data.sample_row], null, 2);
    }
    if (schema.data?.columns && schema.data.columns.length > 0) {
      const row: Record<string, unknown> = {};
      for (const c of schema.data.columns) row[c.name] = '';
      return JSON.stringify([row], null, 2);
    }
    return IRIS_FALLBACK;
  }, [schema.data]);

  const [text, setText] = useState(seedJson);
  const [parseError, setParseError] = useState<string | null>(null);

  // Re-seed when the schema arrives after first paint (or the user
  // navigates between deployments backed by different pipelines).
  useEffect(() => {
    setText(seedJson);
  }, [seedJson]);

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
      <h2 className="text-sm font-medium text-ink-900">Test a prediction</h2>

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
          {schema.data?.columns?.length
            ? `${schema.data.columns.length} feature${schema.data.columns.length === 1 ? '' : 's'} expected${schema.data.target ? ` · target ${schema.data.target} excluded` : ''}.`
            : "Each row is an object with column names matching the model's training features."}
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
            <span className="font-medium text-ink-900">Response</span>
            <span className="font-mono text-xs text-ink-500 tabular-nums">
              {predict.data.latency_ms.toFixed(1)}ms · request{' '}
              <span title={predict.data.request_id}>
                {predict.data.request_id.slice(0, 8)}…
              </span>
            </span>
          </div>
          <PredictionResults predictions={predict.data.predictions} />
        </section>
      )}
    </div>
  );
}

/**
 * Render the predictions table, with a per-class probability bar set
 * inline on each row for classification deployments. Single source of
 * truth so trial-detail and deployment-detail render the same way.
 */
function PredictionResults({
  predictions,
}: {
  predictions: Array<{
    index: number;
    prediction: unknown;
    probabilities?: Record<string, number>;
  }>;
}) {
  // Stable column order for class labels — pulled from the first row that
  // exposes probabilities. All rows share the same classes (sklearn's
  // contract) so we don't need to merge keys across rows.
  const classes = predictions.find((p) => p.probabilities)?.probabilities
    ? Object.keys(predictions.find((p) => p.probabilities)!.probabilities!)
    : null;

  return (
    <div className="card p-0 overflow-hidden">
      <table className="w-full text-sm">
        <thead className="bg-white text-ink-500 dark:bg-ink-900 border-b border-ink-200 dark:border-ink-800">
          <tr>
            <th className="px-3 py-2 text-left font-medium w-12">#</th>
            <th className="px-3 py-2 text-left font-medium">Prediction</th>
            {classes &&
              classes.map((c) => (
                <th
                  key={c}
                  className="px-3 py-2 text-right font-medium tabular-nums"
                  title={`P(class = ${c})`}
                >
                  P({c})
                </th>
              ))}
          </tr>
        </thead>
        <tbody>
          {predictions.map((p) => {
            const proba = p.probabilities;
            const top =
              proba && classes
                ? classes.reduce(
                    (acc, c) => (proba[c] > acc.v ? { c, v: proba[c] } : acc),
                    { c: classes[0], v: proba[classes[0]] ?? 0 },
                  )
                : null;
            return (
              <tr
                key={p.index}
                className="border-t border-ink-200 dark:border-ink-800 hover:bg-ink-50 dark:hover:bg-ink-950/40"
              >
                <td className="px-3 py-2 font-mono text-xs tabular-nums text-ink-500">
                  {p.index}
                </td>
                <td className="px-3 py-2 font-mono text-sm text-ink-900 dark:text-ink-50">
                  {typeof p.prediction === 'object'
                    ? JSON.stringify(p.prediction)
                    : String(p.prediction)}
                </td>
                {classes &&
                  classes.map((c) => {
                    const v = proba?.[c];
                    const isTop = top && top.c === c;
                    return (
                      <td
                        key={c}
                        className="px-3 py-2 text-right font-mono tabular-nums"
                      >
                        {typeof v === 'number' ? (
                          <span className="inline-flex items-center gap-2 justify-end">
                            <span
                              className="h-1 w-12 rounded-full bg-ink-100 dark:bg-ink-800 overflow-hidden"
                              aria-hidden
                            >
                              <span
                                className={`block h-full ${isTop ? 'bg-success-500' : 'bg-accent-500'}`}
                                style={{ width: `${Math.max(2, v * 100)}%` }}
                              />
                            </span>
                            <span
                              className={
                                isTop
                                  ? 'text-success-700 dark:text-success-400 font-semibold'
                                  : 'text-ink-700 dark:text-ink-300'
                              }
                            >
                              {v.toFixed(3)}
                            </span>
                          </span>
                        ) : (
                          <span className="text-ink-400">—</span>
                        )}
                      </td>
                    );
                  })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
