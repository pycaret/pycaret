/**
 * Right-side drawer for the live engine event log.
 *
 * Replaces the inline EventStream block on RunDetail with a slide-out
 * panel modeled after how DataRobot / H2O surface activity logs:
 *
 *   - Anchors to the right edge.
 *   - Animated slide-in / slide-out, body scroll locked while open.
 *   - Backdrop click + Escape close.
 *   - Header chip with live/closed status and event count.
 *   - Each event row carries an icon + tone matching its kind so the
 *     log scans visually instead of as a wall of monospace text.
 *
 * Auto-opens once when a run kicks off (status=running) so the user
 * sees activity without having to click anything.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useAuthStore } from '@/state/auth';
import type { WsEvent } from '@/api/types';

const AUTH_CLOSE_CODES = new Set([4401, 4403]);
const TERMINAL_SENTINEL = 'run.closed';

// ─── Visual mapping per event kind ─────────────────────────────────────

type Tone = 'accent' | 'success' | 'danger' | 'warn' | 'neutral';

interface KindMeta {
  label: string;
  tone: Tone;
  icon: 'play' | 'check' | 'x' | 'flag' | 'split' | 'gear' | 'bolt' | 'dot' | 'flask';
}

const KIND_META: Record<string, KindMeta> = {
  'experiment.started': { label: 'Setup started', tone: 'accent', icon: 'flask' },
  'experiment.fitted': { label: 'Setup complete', tone: 'success', icon: 'check' },
  'model.compare.started': { label: 'Compare started', tone: 'accent', icon: 'split' },
  'model.compare.finished': { label: 'Compare finished', tone: 'success', icon: 'flag' },
  'model.create.started': { label: 'Training', tone: 'accent', icon: 'play' },
  'model.created': { label: 'Trained', tone: 'success', icon: 'check' },
  'model.compared': { label: 'Model done', tone: 'success', icon: 'check' },
  'model.tune.started': { label: 'Tuning', tone: 'accent', icon: 'gear' },
  'model.tuned': { label: 'Tuned', tone: 'success', icon: 'check' },
  'model.predicted': { label: 'Predicted', tone: 'accent', icon: 'bolt' },
  warning: { label: 'Warning', tone: 'warn', icon: 'flag' },
  error: { label: 'Error', tone: 'danger', icon: 'x' },
};

function metaFor(kind: string, payload: Record<string, unknown> | undefined): KindMeta {
  // Per-model failure events emit kind=model.compared with status=failed —
  // upgrade their tone so they jump out in red.
  if (
    kind === 'model.compared' &&
    payload &&
    (payload as { status?: string }).status === 'failed'
  ) {
    return { label: 'Model failed', tone: 'danger', icon: 'x' };
  }
  return (
    KIND_META[kind] ?? {
      label: kind.replaceAll('.', ' '),
      tone: 'neutral',
      icon: 'dot',
    }
  );
}

const TONE_STYLE: Record<Tone, { dot: string; chip: string; ring: string }> = {
  accent: {
    dot: 'bg-accent-500',
    chip: 'bg-accent-50 text-accent-700 dark:bg-accent-500/15 dark:text-accent-300',
    ring: 'ring-accent-200 dark:ring-accent-500/30',
  },
  success: {
    dot: 'bg-success-500',
    chip: 'bg-success-50 text-success-700 dark:bg-success-500/15 dark:text-success-400',
    ring: 'ring-success-200 dark:ring-success-500/30',
  },
  danger: {
    dot: 'bg-danger-500',
    chip: 'bg-danger-50 text-danger-700 dark:bg-danger-500/15 dark:text-danger-400',
    ring: 'ring-danger-200 dark:ring-danger-500/30',
  },
  warn: {
    dot: 'bg-warn-500',
    chip: 'bg-warn-50 text-warn-700 dark:bg-warn-500/15 dark:text-warn-400',
    ring: 'ring-warn-200 dark:ring-warn-500/30',
  },
  neutral: {
    dot: 'bg-ink-400',
    chip: 'bg-ink-100 text-ink-700 dark:bg-ink-800 dark:text-ink-300',
    ring: 'ring-ink-200 dark:ring-ink-700',
  },
};

function EventIcon({ icon }: { icon: KindMeta['icon'] }) {
  const common = {
    width: 14,
    height: 14,
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 2,
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
  };
  switch (icon) {
    case 'play':
      return (
        <svg {...common} aria-hidden>
          <polygon points="5 3 19 12 5 21 5 3" />
        </svg>
      );
    case 'check':
      return (
        <svg {...common} aria-hidden>
          <polyline points="20 6 9 17 4 12" />
        </svg>
      );
    case 'x':
      return (
        <svg {...common} aria-hidden>
          <line x1="18" y1="6" x2="6" y2="18" />
          <line x1="6" y1="6" x2="18" y2="18" />
        </svg>
      );
    case 'flag':
      return (
        <svg {...common} aria-hidden>
          <path d="M4 22V4a4 4 0 0 1 4-4h12l-3 5 3 5H4" />
        </svg>
      );
    case 'split':
      return (
        <svg {...common} aria-hidden>
          <path d="M6 3v18M18 3v18M3 12h18" />
        </svg>
      );
    case 'gear':
      return (
        <svg {...common} aria-hidden>
          <circle cx="12" cy="12" r="3" />
          <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 0 1-4 0v-.1a1.7 1.7 0 0 0-1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 0 1 0-4h.1a1.7 1.7 0 0 0 1.5-1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 0 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 0 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z" />
        </svg>
      );
    case 'bolt':
      return (
        <svg {...common} aria-hidden>
          <polygon points="13 2 4 14 11 14 11 22 20 10 13 10 13 2" />
        </svg>
      );
    case 'flask':
      return (
        <svg {...common} aria-hidden>
          <path d="M9 2h6v6l5 12a2 2 0 0 1-2 3H6a2 2 0 0 1-2-3l5-12V2z" />
        </svg>
      );
    case 'dot':
      return (
        <svg {...common} aria-hidden>
          <circle cx="12" cy="12" r="2" fill="currentColor" />
        </svg>
      );
  }
}

// ─── Drawer ─────────────────────────────────────────────────────────────

export interface EventLogDrawerProps {
  runId: string;
  open: boolean;
  onClose: () => void;
}

function fmtTime(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString(undefined, { hour12: false });
}

function fmtDuration(ms: number | null | undefined): string {
  if (ms == null) return '';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${Math.round(s - m * 60)}s`;
}

/**
 * Stable dedup key for an event. The backend has no per-event UUID, so we
 * compose one from the fields that uniquely identify it. Needed because:
 *   - React StrictMode (dev) mounts effects twice → two WS connections →
 *     two replays of stored events.
 *   - Our own onclose retry can also reconnect, triggering a fresh replay.
 *   - Both backends + clients may race during a reconnect window.
 */
function eventKey(e: WsEvent): string {
  return `${e.kind}|${e.timestamp}|${e.message ?? ''}|${e.duration_ms ?? ''}`;
}

export function EventLogDrawer({ runId, open, onClose }: EventLogDrawerProps) {
  const accessToken = useAuthStore((s) => s.accessToken);
  const [events, setEvents] = useState<WsEvent[]>([]);
  const [status, setStatus] = useState<'connecting' | 'open' | 'closed' | 'error'>(
    'connecting',
  );
  const [filter, setFilter] = useState<'all' | 'success' | 'failed'>('all');
  const tailRef = useRef<HTMLDivElement>(null);

  // Body-scroll lock + Escape to close.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.removeEventListener('keydown', onKey);
      document.body.style.overflow = prev;
    };
  }, [open, onClose]);

  // WebSocket: only kept open while the drawer is.
  useEffect(() => {
    if (!open || !runId || !accessToken) return;
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${window.location.host}/api/v1/runs/${runId}/events/ws?token=${encodeURIComponent(accessToken)}`;
    let retried = false;
    let cancelled = false;
    let ws: WebSocket | null = null;

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
          setEvents((prev) => {
            const key = eventKey(msg);
            if (prev.some((p) => eventKey(p) === key)) return prev;
            const next = prev.concat(msg);
            return next.length > 1000 ? next.slice(next.length - 1000) : next;
          });
        } catch {
          /* ignore malformed event */
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
  }, [open, runId, accessToken]);

  // Reset on run change so switching runs doesn't mix streams.
  useEffect(() => {
    setEvents([]);
  }, [runId]);

  // Auto-scroll to bottom whenever new events land — so live activity
  // is always in view without the user fighting the scroll.
  useEffect(() => {
    if (tailRef.current) {
      tailRef.current.scrollIntoView({ block: 'end' });
    }
  }, [events]);

  const filtered = useMemo(() => {
    if (filter === 'all') return events;
    return events.filter((e) => {
      const meta = metaFor(
        e.kind,
        (e.payload ?? {}) as Record<string, unknown>,
      );
      if (filter === 'failed') return meta.tone === 'danger';
      if (filter === 'success') return meta.tone === 'success';
      return true;
    });
  }, [events, filter]);

  if (!open) return null;

  return createPortal(
    <div
      className="fixed inset-0 z-[60]"
      role="dialog"
      aria-modal="true"
      aria-labelledby="event-drawer-title"
    >
      <div
        className="absolute inset-0 bg-ink-950/30 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden
      />
      <aside className="absolute top-0 right-0 h-full w-full sm:w-[480px] bg-white dark:bg-ink-950 border-l border-ink-200 dark:border-ink-800 shadow-2xl flex flex-col animate-[slideIn_180ms_ease-out]">
        <header className="flex items-start justify-between gap-3 px-5 py-4 border-b border-ink-200 dark:border-ink-800">
          <div>
            <h2
              id="event-drawer-title"
              className="text-sm font-semibold text-ink-900 dark:text-ink-50"
            >
              Event log
            </h2>
            <p className="text-xs text-ink-500 mt-0.5">
              Every structured event the engine emits during this run.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-ink-500 hover:text-ink-900 dark:hover:text-ink-50 -mt-1 -mr-1 p-1 rounded hover:bg-ink-100 dark:hover:bg-ink-900"
            aria-label="Close event log"
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden
            >
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </header>

        <div className="flex items-center justify-between gap-2 px-5 py-2 border-b border-ink-200 dark:border-ink-800 text-xs">
          <div className="inline-flex rounded-md border border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 p-0.5">
            {(['all', 'success', 'failed'] as const).map((k) => (
              <button
                key={k}
                onClick={() => setFilter(k)}
                className={`px-2.5 py-1 rounded font-medium capitalize transition-colors ${
                  filter === k
                    ? 'bg-ink-900 text-white dark:bg-white dark:text-ink-900'
                    : 'text-ink-600 hover:text-ink-900 dark:text-ink-400 dark:hover:text-ink-50'
                }`}
              >
                {k}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-2">
            <span
              className={
                status === 'open'
                  ? 'pill-accent'
                  : status === 'error'
                    ? 'pill-danger'
                    : 'pill-neutral'
              }
            >
              {status === 'open' ? 'live' : status}
            </span>
            <span className="text-ink-500 tabular-nums">
              {filtered.length} {filtered.length === 1 ? 'event' : 'events'}
            </span>
          </div>
        </div>

        <ol
          className="flex-1 overflow-y-auto px-3 py-2 space-y-1.5"
          aria-live="polite"
        >
          {filtered.length === 0 && (
            <li className="text-sm text-ink-500 px-2 py-12 text-center">
              {status === 'connecting'
                ? 'Connecting to the engine…'
                : 'No events yet — kick off a run to see live activity.'}
            </li>
          )}
          {filtered.map((e, i) => {
            const meta = metaFor(
              e.kind,
              (e.payload ?? {}) as Record<string, unknown>,
            );
            const tone = TONE_STYLE[meta.tone];
            return (
              <li
                key={`${e.kind}-${e.timestamp}-${i}`}
                className="flex items-start gap-3 px-2 py-2 rounded-md hover:bg-ink-50 dark:hover:bg-ink-900/40"
              >
                <span
                  className={`flex-shrink-0 mt-0.5 inline-flex h-6 w-6 items-center justify-center rounded-full ${tone.chip} ring-2 ${tone.ring}`}
                >
                  <EventIcon icon={meta.icon} />
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-baseline gap-2 flex-wrap">
                    <span className="text-xs font-semibold text-ink-900 dark:text-ink-50">
                      {meta.label}
                    </span>
                    <span className="font-mono text-[10px] text-ink-400">
                      {fmtTime(e.timestamp)}
                    </span>
                    {e.duration_ms != null && (
                      <span className="font-mono text-[10px] text-ink-400">
                        {fmtDuration(e.duration_ms)}
                      </span>
                    )}
                  </div>
                  {e.message && (
                    <p className="mt-0.5 text-xs text-ink-600 dark:text-ink-300 break-words">
                      {e.message}
                    </p>
                  )}
                </div>
              </li>
            );
          })}
          <div ref={tailRef} />
        </ol>
      </aside>

      <style>{`
        @keyframes slideIn {
          from { transform: translateX(100%); }
          to { transform: translateX(0); }
        }
      `}</style>
    </div>,
    document.body,
  );
}
