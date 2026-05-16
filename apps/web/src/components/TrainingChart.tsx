/**
 * Live training chart — subscribes to the same WebSocket event stream
 * as ``EventStream`` and renders the per-Trial primary-metric points
 * on a scatter plot in real time.
 *
 * Each ``model.created`` event carries
 * ``payload = {model_id, metrics: {Accuracy, AUC, ...}, fold_metrics: [...]}``
 * after the Phase 6 widening. We pick the most-prevalent supervised
 * metric (Accuracy → AUC → F1 → R2 → MAE → RMSE) the first time we see
 * one and plot every subsequent point against the same axis.
 *
 * If the run is unsupervised or the metric set is empty, the chart
 * renders an "(no scoreable metric yet)" placeholder rather than
 * burning console errors. The chart is purely additive — failure
 * never breaks the parent page.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import Plot from 'react-plotly.js';
import { useAuthStore } from '@/state/auth';
import type { WsEvent } from '@/api/types';

const AUTH_CLOSE_CODES = new Set([4401, 4403]);
const TERMINAL_SENTINEL = 'run.closed';

// In priority order. Higher = better for the first three; the chart
// inverts the y-axis when one of the error metrics is picked.
const PREFERRED_METRICS = [
  'Accuracy',
  'AUC',
  'F1',
  'R2',
  'MAE',
  'RMSE',
] as const;
const ERROR_METRICS = new Set(['MAE', 'RMSE', 'MSE', 'MAPE']);

interface TrialPoint {
  model_id: string;
  value: number;
  ts: number;
  duration_ms: number | null;
}

export interface TrainingChartProps {
  runId: string;
  className?: string;
}

export function TrainingChart({ runId, className }: TrainingChartProps) {
  const accessToken = useAuthStore((s) => s.accessToken);
  const [points, setPoints] = useState<TrialPoint[]>([]);
  const [chosenMetric, setChosenMetric] = useState<string | null>(null);
  const [status, setStatus] = useState<
    'connecting' | 'open' | 'closed' | 'error'
  >('connecting');
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(640);

  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    const obs = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setWidth(Math.max(280, Math.floor(entry.contentRect.width)));
      }
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    if (!runId || !accessToken) return;
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${window.location.host}/api/v1/runs/${runId}/events/ws?token=${encodeURIComponent(accessToken)}`;
    let retried = false;
    let cancelled = false;
    let ws: WebSocket | null = null;
    setPoints([]);
    setChosenMetric(null);

    const connect = () => {
      if (cancelled) return;
      setStatus('connecting');
      ws = new WebSocket(url);
      ws.onopen = () => setStatus('open');
      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data) as WsEvent;
          if (msg.kind === TERMINAL_SENTINEL) {
            retried = true;
            return;
          }
          // Only listen to model.compared (the CV-mean summary that ends a
          // compare-trial). model.created fires per-fold during training
          // with single-fold metrics, which made the chart and the
          // leaderboard table show different numbers for the same trial.
          // Sticking to model.compared keeps the chart and the table in
          // exact agreement.
          if (msg.kind !== 'model.compared') return;
          const payload = (msg.payload ?? {}) as Record<string, unknown>;
          const metrics = (payload.metrics ?? {}) as Record<string, unknown>;
          const modelId =
            (payload.model_id as string | undefined) ??
            (payload.estimator as string | undefined) ??
            'unknown';
          const status_ = payload.status as string | undefined;
          if (status_ && status_ !== 'succeeded') return; // failed trial, skip
          // Pick a metric to plot. Once chosen, stick with it.
          setChosenMetric((cur) => {
            if (cur != null) return cur;
            for (const m of PREFERRED_METRICS) {
              if (typeof metrics[m] === 'number') return m;
            }
            return cur;
          });
          setPoints((prev) => {
            const metricName = (() => {
              for (const m of PREFERRED_METRICS) {
                if (typeof metrics[m] === 'number') return m;
              }
              return null;
            })();
            if (metricName == null) return prev;
            const value = metrics[metricName] as number;
            // Dedupe by model_id (StrictMode + reconnect can fire the same
            // model.compared twice). Keep the last value seen — should be
            // identical anyway since model.compared is deterministic.
            const dedup = prev.filter((p) => p.model_id !== modelId);
            return dedup.concat({
              model_id: modelId,
              value,
              ts: msg.timestamp,
              duration_ms: msg.duration_ms ?? null,
            });
          });
        } catch {
          /* swallow malformed frames */
        }
      };
      ws.onerror = () => setStatus('error');
      ws.onclose = (e) => {
        setStatus('closed');
        if (AUTH_CLOSE_CODES.has(e.code)) return;
        if (!retried && !cancelled) {
          retried = true;
          setTimeout(connect, 500);
        }
      };
    };
    connect();
    return () => {
      cancelled = true;
      ws?.close();
    };
  }, [runId, accessToken]);

  const figure = useMemo(() => {
    if (points.length === 0 || chosenMetric == null) return null;
    const lower = ERROR_METRICS.has(chosenMetric);
    const xs = points.map((_, i) => i + 1);
    const ys = points.map((p) => p.value);
    return {
      data: [
        {
          type: 'scatter' as const,
          mode: 'lines+markers' as const,
          x: xs,
          y: ys,
          text: points.map(
            (p) => `${p.model_id} · ${chosenMetric}=${p.value.toFixed(4)}`,
          ),
          hovertemplate: '%{text}<extra></extra>',
          line: { color: '#5b8def', width: 2 },
          marker: {
            size: 8,
            color: ys.map((_, i) =>
              i ===
              (lower
                ? ys.indexOf(Math.min(...ys))
                : ys.indexOf(Math.max(...ys)))
                ? '#10b981'
                : '#5b8def',
            ),
          },
        },
      ],
      layout: {
        margin: { l: 56, r: 16, t: 24, b: 40 },
        xaxis: {
          title: { text: 'Trial' },
          tickmode: 'array' as const,
          tickvals: xs,
          ticktext: points.map((p) => p.model_id),
          tickangle: -30,
        },
        yaxis: {
          title: { text: chosenMetric },
          autorange: (lower ? 'reversed' : true) as boolean | 'reversed',
        },
        showlegend: false,
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
      },
    };
  }, [points, chosenMetric]);

  const statusLabel =
    status === 'open' ? 'live' : status === 'connecting' ? 'connecting…' : status;

  return (
    <section ref={containerRef} className={['card', className ?? ''].join(' ')}>
      <header className="mb-3 flex items-end justify-between gap-3">
        <div>
          <h3 className="h-section">Training progress</h3>
          <p className="text-xs text-ink-500 mt-0.5">
            Per-trial primary metric, streamed live as workers complete.
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs shrink-0">
          <span
            className={
              status === 'open'
                ? 'pill-accent'
                : status === 'closed'
                  ? 'pill-neutral'
                  : status === 'error'
                    ? 'pill-danger'
                    : 'pill-neutral'
            }
          >
            {statusLabel}
          </span>
          <span className="text-ink-500 tabular-nums">
            {points.length} {points.length === 1 ? 'trial' : 'trials'}
          </span>
          {chosenMetric && (
            <span className="pill-neutral font-mono">{chosenMetric}</span>
          )}
        </div>
      </header>

      {figure == null ? (
        <div className="text-sm text-ink-500 py-12 text-center">
          {status === 'open'
            ? 'Waiting for trial results…'
            : status === 'connecting'
              ? 'Connecting…'
              : 'No scoreable metric in this run.'}
        </div>
      ) : (
        <Plot
          data={figure.data}
          layout={
            { ...figure.layout, width, height: 280 } as Partial<Plotly.Layout>
          }
          config={{
            displayModeBar: false,
            displaylogo: false,
            responsive: true,
          }}
          useResizeHandler
          style={{ width: '100%', height: 280 }}
        />
      )}
    </section>
  );
}
