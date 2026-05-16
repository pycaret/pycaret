import { useEffect, useRef, type ReactNode } from 'react';
import { createPortal } from 'react-dom';

/**
 * Modal dialog primitive. Used for "create X" and "edit Y" flows
 * across the app — keeps the page layout clean (no always-visible
 * side panels eating viewport space).
 *
 * Conventions:
 *   - Centered, max-w-lg by default; pass `size="md"` / `"lg"` to widen
 *   - Backdrop click + Escape close
 *   - Trap focus within the dialog while open (basic — first focusable on open)
 *   - Portal'd to <body> so the layout's overflow rules don't clip it
 *   - Panel is hard-capped to the viewport (max-h: calc(100vh - 3rem)) and
 *     is a flex column, so the body can `flex-1 overflow-hidden` for
 *     internally-scrolling EDA / table content without spilling off-screen.
 */
export interface DialogProps {
  open: boolean;
  onClose: () => void;
  title: string;
  description?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  /**
   * Optional footer (typically buttons). If omitted, the dialog has no
   * footer — useful when the body itself contains a form with its own
   * submit button.
   */
  footer?: ReactNode;
  /**
   * Pass `true` for content that owns its own padding + layout (eg. a
   * split-pane EDA viewer). Default `false` keeps the px-6/py-5 wrapper
   * that "create X" forms want.
   */
  noBodyPadding?: boolean;
  children: ReactNode;
}

const sizeClass = {
  sm: 'max-w-md',
  md: 'max-w-lg',
  lg: 'max-w-2xl',
  xl: 'max-w-5xl',
  full: 'max-w-7xl',
};

export function Dialog({
  open,
  onClose,
  title,
  description,
  size = 'md',
  footer,
  noBodyPadding = false,
  children,
}: DialogProps) {
  const ref = useRef<HTMLDivElement>(null);

  // Escape closes; body scroll lock while open.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    // Focus the first focusable child of the dialog body.
    setTimeout(() => {
      const focusable = ref.current?.querySelector<HTMLElement>(
        'input, textarea, select, button:not([data-dialog-close])',
      );
      focusable?.focus();
    }, 50);
    return () => {
      document.removeEventListener('keydown', onKey);
      document.body.style.overflow = prev;
    };
  }, [open, onClose]);

  if (!open) return null;

  return createPortal(
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="dialog-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6"
    >
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-ink-950/30 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden
      />

      {/* Panel — hard-bounded to viewport, flex column so body can fill+scroll */}
      <div
        ref={ref}
        className={`relative w-full ${sizeClass[size]} rounded-xl bg-white dark:bg-ink-900 shadow-soft-3 border border-ink-200 flex flex-col`}
        style={{ maxHeight: 'calc(100vh - 3rem)' }}
      >
        {/* Header */}
        <div className="flex items-start justify-between gap-4 px-6 pt-5 pb-3 border-b border-ink-100 dark:border-ink-800 shrink-0">
          <div className="min-w-0 flex-1">
            <h2 id="dialog-title" className="text-base font-semibold text-ink-900 dark:text-ink-50">
              {title}
            </h2>
            {description && (
              <p className="mt-1 text-sm text-ink-500">{description}</p>
            )}
          </div>
          <button
            type="button"
            data-dialog-close
            onClick={onClose}
            aria-label="Close dialog"
            className="text-ink-400 dark:text-ink-500 hover:text-ink-700 dark:text-ink-300 transition-colors -mt-1 -mr-1 p-1 rounded-md hover:bg-ink-100 dark:bg-ink-800"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <path d="M18 6L6 18 M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Body — flex column with min-h-0 so a child with `flex-1 min-h-0`
            inherits a real height and can install its own overflow:auto.
            When noBodyPadding=true, we also become a flex container so the
            child doesn't have to rely on `h-full` (which resolves to 0 if the
            parent's height comes from flex sizing — a real Chrome quirk). */}
        <div
          className={`flex-1 min-h-0 ${
            noBodyPadding ? 'flex flex-col' : 'px-6 py-5 overflow-y-auto'
          }`}
        >
          {children}
        </div>

        {/* Footer (optional) */}
        {footer && (
          <div className="px-6 py-4 border-t border-ink-100 dark:border-ink-800 bg-ink-50 dark:bg-ink-950/40 rounded-b-xl flex items-center justify-end gap-2 shrink-0">
            {footer}
          </div>
        )}
      </div>
    </div>,
    document.body,
  );
}
