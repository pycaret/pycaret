/**
 * Live WebSocket event stream for a single run.
 *
 * Contract with the backend (services/api/pycaret_server/api/runs.py § ws_events):
 *  - Connection URL: `/api/v1/runs/:run_id/events/ws?token=<access_token>`.
 *  - On connect the server replays stored events for the run, then either
 *    live-streams new events or sends `{kind: "run.closed"}` + closes
 *    (if the run is already in a terminal state).
 *  - Each message is a JSON object matching `WsEvent`.
 *
 * We reconnect once automatically on an unexpected close (not on 4401/4403
 * which are auth failures; those are user-facing and shouldn't silently retry).
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { useAuthStore } from '@/state/auth';
import type { WsEvent } from '@/api/types';

export interface EventStreamProps {
  runId: string;
  /** When true, an empty stream renders a short placeholder. */
  showEmptyHint?: boolean;
  /** Optional cap on rendered events; older events are dropped. Default 500. */
  max?: number;
  className?: string;
}

const AUTH_CLOSE_CODES = new Set([4401, 4403]);
const TERMINAL_SENTINEL = 'run.closed';

/**
 * Pretty-print an event-kind string. `model.compare.started` →
 * `MODEL COMPARE STARTED`, styled monospace.
 */
function fmtKind(kind: string): string {
  return kind.toUpperCase().replaceAll('.', ' ');
}

function fmtTime(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString(undefined, {
    hour12: false,
  });
}

function kindTone(kind: string): string {
  if (kind === TERMINAL_SENTINEL) return 'text-ink-200/60';
  if (kind === 'error' || kind.endsWith('.failed')) return 'text-danger-500';
  if (kind === 'warning') return 'text-warn-500';
  if (kind.endsWith('.started')) return 'text-accent-400';
  if (kind.endsWith('.finished') || kind.endsWith('.created') || kind.endsWith('.fitted'))
    return 'text-success-500';
  return 'text-ink-100';
}

export function EventStream({
  runId,
  showEmptyHint = true,
  max = 500,
  className,
}: EventStreamProps) {
  const accessToken = useAuthStore((s) => s.accessToken);
  const [events, setEvents] = useState<WsEvent[]>([]);
  const [status, setStatus] = useState<'connecting' | 'open' | 'closed' | 'error'>(
    'connecting',
  );
  const [lastError, setLastError] = useState<string | null>(null);

  // Keep the event handlers in refs so the WS effect only re-runs when runId
  // or the token changes — not on every new event.
  const eventsRef = useRef(events);
  eventsRef.current = events;

  useEffect(() => {
    if (!runId || !accessToken) return;

    // Absolute WS URL. Vite's dev server proxies ws:// /api and /ws; in
    // production the nginx config forwards /api/v1/runs/*/events/ws to the API.
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${window.location.host}/api/v1/runs/${runId}/events/ws?token=${encodeURIComponent(accessToken)}`;

    let retried = false;
    let ws: WebSocket | null = null;
    let cancelled = false;

    const connect = () => {
      if (cancelled) return;
      setStatus('connecting');
      ws = new WebSocket(url);

      ws.onopen = () => setStatus('open');

      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data) as WsEvent;
          setEvents((prev) => {
            const next = prev.concat(msg);
            return next.length > max ? next.slice(next.length - max) : next;
          });
          if (msg.kind === TERMINAL_SENTINEL) {
            // Server will close right after — we don't need to reconnect.
            retried = true;
          }
        } catch {
          setLastError('malformed event');
        }
      };

      ws.onerror = () => {
        setStatus('error');
      };

      ws.onclose = (e) => {
        setStatus('closed');
        if (AUTH_CLOSE_CODES.has(e.code)) {
          setLastError(`auth failure (${e.code})`);
          return;
        }
        // Unexpected close → retry once.
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
  }, [runId, accessToken, max]);

  // Reset on run change so switching between runs doesn't mix streams.
  useEffect(() => {
    setEvents([]);
    setLastError(null);
  }, [runId]);

  const hasEvents = events.length > 0;
  const statusLabel = useMemo(() => {
    if (status === 'open') return 'live';
    if (status === 'connecting') return 'connecting…';
    return status;
  }, [status]);

  return (
    <div className={className}>
      <header className="mb-3 flex items-center justify-between text-xs">
        <h3 className="text-sm font-medium text-ink-100">Event stream</h3>
        <div className="flex items-center gap-2">
          <span
            className={
              status === 'open'
                ? 'text-accent-400'
                : status === 'closed'
                  ? 'text-ink-200/60'
                  : status === 'error'
                    ? 'text-danger-500'
                    : 'text-ink-200/70'
            }
          >
            ● {statusLabel}
          </span>
          <span className="text-ink-200/50">{events.length} events</span>
        </div>
      </header>

      {lastError && <p className="error mb-2">{lastError}</p>}

      {!hasEvents && showEmptyHint && (
        <p className="hint">
          Waiting for events… the stream shows every structured `Event` emitted by the
          engine as the run progresses.
        </p>
      )}

      {hasEvents && (
        <ol className="card p-0 divide-y divide-ink-800 text-sm max-h-[32rem] overflow-auto">
          {events.map((e, i) => (
            <li key={i} className="grid grid-cols-[auto_1fr_auto] gap-3 px-4 py-2">
              <span className="font-mono text-xs text-ink-200/50 tabular-nums">
                {fmtTime(e.timestamp)}
              </span>
              <div className="min-w-0">
                <span className={`font-mono text-xs ${kindTone(e.kind)}`}>
                  {fmtKind(e.kind)}
                </span>
                {e.message && (
                  <span className="ml-2 text-ink-200/80">{e.message}</span>
                )}
              </div>
              {e.duration_ms != null && (
                <span className="font-mono text-xs text-ink-200/50 tabular-nums">
                  {e.duration_ms < 1000
                    ? `${e.duration_ms.toFixed(0)}ms`
                    : `${(e.duration_ms / 1000).toFixed(1)}s`}
                </span>
              )}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
