/**
 * Subtle "back" button that sits above page breadcrumbs across the app.
 *
 * Uses the router's history when there is one (so the button takes you to
 * the previous page you actually came from) and falls back to a sensible
 * route — `to` — when navigated to directly (refresh / deep link).
 */

import { useNavigate, type To } from 'react-router-dom';

export interface BackButtonProps {
  /** Fallback destination if there's no history to pop back to. */
  to?: To;
  /** Override the default "Back" label. */
  label?: string;
  className?: string;
}

export function BackButton({ to, label = 'Back', className }: BackButtonProps) {
  const navigate = useNavigate();
  const onClick = () => {
    // window.history.length > 1 isn't perfectly reliable across SPAs,
    // but it's the cheapest signal that we have a real entry to pop.
    if (window.history.length > 1) {
      navigate(-1);
    } else if (to) {
      navigate(to);
    } else {
      navigate('/');
    }
  };
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        'inline-flex items-center gap-1.5 text-xs text-ink-500 hover:text-ink-900 dark:hover:text-ink-50 mb-2 -ml-1 px-1.5 py-1 rounded hover:bg-ink-100/60 dark:hover:bg-ink-800/60 transition-colors',
        className ?? '',
      ].join(' ')}
      aria-label={label}
    >
      <svg
        width="14"
        height="14"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden
      >
        <polyline points="15 18 9 12 15 6" />
      </svg>
      <span>{label}</span>
    </button>
  );
}
