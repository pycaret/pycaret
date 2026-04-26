/**
 * Cmd-K command palette.
 *
 * Modal launcher mounted at the layout level. Opens with ⌘/Ctrl+K and
 * shows a fuzzy-filtered list of navigation actions across the app.
 * Arrow-key navigation, Enter to invoke.
 *
 * Future: integrate with workspace data so users can jump to recent
 * runs / pipelines / deployments by ID. The current version is purely
 * navigation; that's still a major UX upgrade.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';

interface Command {
  id: string;
  label: string;
  hint?: string;
  to?: string;
  action?: () => void;
  keywords?: string;
}

function fuzzyMatch(query: string, target: string): number {
  // Simple subsequence-match score: lower = better; -1 = no match.
  if (!query) return 0;
  const q = query.toLowerCase();
  const t = target.toLowerCase();
  let score = 0;
  let qi = 0;
  for (let i = 0; i < t.length && qi < q.length; i++) {
    if (t[i] === q[qi]) {
      score += i;
      qi += 1;
    }
  }
  return qi === q.length ? score : -1;
}

export function CommandPalette({ wsId }: { wsId?: string }) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  // Toggle on ⌘/Ctrl-K.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setOpen((v) => !v);
        setQuery('');
        setCursor(0);
      } else if (e.key === 'Escape' && open) {
        setOpen(false);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open]);

  useEffect(() => {
    if (open && inputRef.current) inputRef.current.focus();
  }, [open]);

  const commands = useMemo<Command[]>(() => {
    const list: Command[] = [
      { id: 'home', label: 'Workspaces', hint: 'Top-level list', to: '/' },
      {
        id: 'apikeys',
        label: 'API keys',
        hint: 'Personal access tokens',
        to: '/account/api-keys',
        keywords: 'token credential',
      },
      { id: 'audit', label: 'Audit log', hint: 'Admin', to: '/admin/audit' },
    ];
    if (wsId) {
      list.unshift(
        {
          id: 'ws-home',
          label: 'Workspace dashboard',
          hint: 'KPIs + recent runs',
          to: `/workspaces/${wsId}/home`,
          keywords: 'overview cockpit',
        },
        {
          id: 'ws-detail',
          label: 'Datasets & projects',
          to: `/workspaces/${wsId}`,
        },
        {
          id: 'compare',
          label: 'Model comparison (A/B)',
          hint: 'Diff two pipelines',
          to: `/workspaces/${wsId}/compare`,
          keywords: 'a/b ab diff side-by-side',
        },
        {
          id: 'drift',
          label: 'Drift dashboard',
          hint: 'Distribution shifts',
          to: `/workspaces/${wsId}/drift`,
          keywords: 'monitoring data quality',
        },
        {
          id: 'predict',
          label: 'Prediction explorer',
          hint: 'Live test deployments',
          to: `/workspaces/${wsId}/predictions`,
          keywords: 'inference test deployment',
        },
        {
          id: 'pipelines',
          label: 'Pipelines registry',
          to: `/workspaces/${wsId}/pipelines`,
        },
        {
          id: 'deployments',
          label: 'Deployments',
          to: `/workspaces/${wsId}/deployments`,
        },
        {
          id: 'llm',
          label: 'LLM settings',
          hint: 'Configure provider keys',
          to: `/workspaces/${wsId}/llm`,
        },
        {
          id: 'members',
          label: 'Workspace members',
          to: `/workspaces/${wsId}/members`,
        },
      );
    }
    return list;
  }, [wsId]);

  const filtered = useMemo(() => {
    if (!query) return commands;
    return commands
      .map((c) => {
        const text = [c.label, c.hint, c.keywords].filter(Boolean).join(' ');
        const score = fuzzyMatch(query, text);
        return { c, score };
      })
      .filter((x) => x.score >= 0)
      .sort((a, b) => a.score - b.score)
      .map((x) => x.c);
  }, [commands, query]);

  useEffect(() => {
    setCursor((c) => Math.min(c, Math.max(0, filtered.length - 1)));
  }, [filtered]);

  if (!open) return null;

  const invoke = (c: Command) => {
    setOpen(false);
    if (c.to) navigate(c.to);
    if (c.action) c.action();
  };

  return (
    <div
      role="dialog"
      aria-modal
      onClick={() => setOpen(false)}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(15, 23, 42, 0.55)',
        backdropFilter: 'blur(2px)',
        zIndex: 100,
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'center',
        paddingTop: '12vh',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 'min(560px, 90vw)',
          background: '#FFFFFF',
          borderRadius: 14,
          boxShadow: '0 24px 60px rgba(15, 23, 42, 0.35)',
          overflow: 'hidden',
        }}
      >
        <input
          ref={inputRef}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Jump to…  (try “drift” or “compare”)"
          onKeyDown={(e) => {
            if (e.key === 'ArrowDown') {
              e.preventDefault();
              setCursor((c) => Math.min(c + 1, filtered.length - 1));
            } else if (e.key === 'ArrowUp') {
              e.preventDefault();
              setCursor((c) => Math.max(c - 1, 0));
            } else if (e.key === 'Enter') {
              e.preventDefault();
              if (filtered[cursor]) invoke(filtered[cursor]);
            }
          }}
          style={{
            width: '100%',
            padding: '16px 20px',
            border: 'none',
            outline: 'none',
            fontSize: 15,
            borderBottom: '1px solid rgba(148,163,184,0.2)',
          }}
        />
        <div style={{ maxHeight: 420, overflowY: 'auto' }}>
          {filtered.length === 0 ? (
            <div
              style={{
                padding: 24,
                textAlign: 'center',
                color: '#94A3B8',
                fontSize: 13,
              }}
            >
              No matches.
            </div>
          ) : (
            filtered.map((c, i) => (
              <button
                key={c.id}
                onClick={() => invoke(c)}
                onMouseEnter={() => setCursor(i)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  width: '100%',
                  padding: '10px 20px',
                  border: 'none',
                  textAlign: 'left',
                  background: i === cursor ? 'rgba(91,141,239,0.10)' : 'transparent',
                  cursor: 'pointer',
                  fontSize: 14,
                }}
              >
                <span style={{ color: '#0F172A', fontWeight: 500 }}>{c.label}</span>
                {c.hint && (
                  <span style={{ color: '#94A3B8', fontSize: 12 }}>{c.hint}</span>
                )}
              </button>
            ))
          )}
        </div>
        <div
          style={{
            padding: '8px 20px',
            borderTop: '1px solid rgba(148,163,184,0.15)',
            color: '#64748B',
            fontSize: 11,
            display: 'flex',
            justifyContent: 'space-between',
          }}
        >
          <span>↑↓ to navigate · ↵ to select · esc to close</span>
          <span>⌘K</span>
        </div>
      </div>
    </div>
  );
}
