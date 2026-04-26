/**
 * Generic Plotly chart wrapper.
 *
 * The API returns Plotly figures as JSON; this component renders one.
 * Features:
 * - Skeleton loader while the upstream query is in flight.
 * - Error card with a retry hook when the request fails.
 * - "Empty" placeholder when no data is yet available.
 * - Responsive sizing — fills the parent's width and uses the figure's
 *   native height (or a fallback).
 *
 * Plot data shape comes from the API's PlotEnvelope (figure.data +
 * figure.layout). Pass `loading` / `error` from a TanStack Query
 * useQuery call.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import Plot from 'react-plotly.js';

import type { PlotlyFigure } from '../api/types';

export interface PlotlyFigureProps {
  /** The Plotly figure JSON returned by the API. */
  figure?: PlotlyFigure | null;
  /** True while the upstream query is in flight. */
  loading?: boolean;
  /** Set when the upstream query failed. */
  error?: Error | null | undefined;
  /** Retry callback wired to the upstream query's `refetch`. */
  onRetry?: () => void;
  /** Optional title shown above the chart (overrides figure.layout.title). */
  title?: string;
  /** Description / caption shown beneath the title. */
  caption?: string;
  /** Pinned height in px; auto when omitted. */
  height?: number;
  /** Optional className passed to the outer card. */
  className?: string;
  /** Hide the toolbar / mode-bar. */
  hideToolbar?: boolean;
}

const skeletonStyle: React.CSSProperties = {
  background:
    'linear-gradient(110deg, rgba(91,141,239,0.06), rgba(91,141,239,0.12), rgba(91,141,239,0.06))',
  backgroundSize: '200% 100%',
  animation: 'pcShimmer 1.4s linear infinite',
  borderRadius: 12,
};

export function PlotlyFigure({
  figure,
  loading = false,
  error,
  onRetry,
  title,
  caption,
  height,
  className,
  hideToolbar = false,
}: PlotlyFigureProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState<number>(640);

  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setWidth(Math.max(280, Math.floor(entry.contentRect.width)));
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const layout = useMemo(() => {
    if (!figure) return undefined;
    const layoutCopy = { ...(figure.layout || {}) };
    if (title) {
      layoutCopy.title = { text: title };
    }
    layoutCopy.autosize = true;
    return layoutCopy;
  }, [figure, title]);

  const config: Partial<Plotly.Config> = useMemo(
    () => ({
      displayModeBar: !hideToolbar,
      displaylogo: false,
      modeBarButtonsToRemove: [
        'lasso2d',
        'select2d',
        'autoScale2d',
        'toggleSpikelines',
      ] as Plotly.ModeBarDefaultButtons[],
      responsive: true,
    }),
    [hideToolbar],
  );

  const inferredHeight =
    height ??
    (figure?.layout && typeof figure.layout.height === 'number'
      ? (figure.layout.height as number)
      : 380);

  return (
    <div ref={containerRef} className={['card', className ?? ''].join(' ')} data-testid="plotly-figure">
      {(title || caption) && (
        <div style={{ marginBottom: 8 }}>
          {title && (
            <div style={{ fontWeight: 600, fontSize: 14, color: '#0F172A' }}>
              {title}
            </div>
          )}
          {caption && (
            <div style={{ fontSize: 12, color: '#64748B', marginTop: 2 }}>{caption}</div>
          )}
        </div>
      )}

      {loading && (
        <div
          aria-busy
          aria-live="polite"
          style={{ ...skeletonStyle, width: '100%', height: inferredHeight }}
        />
      )}

      {!loading && error && (
        <div
          role="alert"
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 12,
            padding: 16,
            border: '1px solid rgba(239, 68, 68, 0.25)',
            background: 'rgba(239, 68, 68, 0.08)',
            borderRadius: 12,
            color: '#7f1d1d',
            fontSize: 13,
          }}
        >
          <div>
            <div style={{ fontWeight: 600 }}>Couldn't render this chart.</div>
            <div style={{ marginTop: 2 }}>{error.message ?? 'Unknown error.'}</div>
          </div>
          {onRetry && (
            <button className="btn-secondary" onClick={onRetry}>
              Retry
            </button>
          )}
        </div>
      )}

      {!loading && !error && figure && layout && (
        <Plot
          data={figure.data as Plotly.Data[]}
          layout={{ ...layout, width, height: inferredHeight } as Partial<Plotly.Layout>}
          config={config}
          useResizeHandler
          style={{ width: '100%', height: inferredHeight }}
        />
      )}

      {!loading && !error && !figure && (
        <div
          style={{
            width: '100%',
            height: inferredHeight,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#94A3B8',
            fontSize: 13,
            border: '1px dashed rgba(148, 163, 184, 0.4)',
            borderRadius: 12,
          }}
        >
          No data yet.
        </div>
      )}

      <style>{`
        @keyframes pcShimmer {
          0% { background-position: 200% 0; }
          100% { background-position: -200% 0; }
        }
      `}</style>
    </div>
  );
}
