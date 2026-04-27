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
 */
export interface DialogProps {
  open: boolean;
  onClose: () => void;
  title: string;
  description?: string;
  size?: 'sm' | 'md' | 'lg';
  /**
   * Optional footer (typically buttons). If omitted, the dialog has no
   * footer — useful when the body itself contains a form with its own
   * submit button.
   */
  footer?: ReactNode;
  children: ReactNode;
}

const sizeClass = {
  sm: 'max-w-md',
  md: 'max-w-lg',
  lg: 'max-w-2xl',
};

export function Dialog({
  open,
  onClose,
  title,
  description,
  size = 'md',
  footer,
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
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto py-12 px-4"
    >
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-ink-950/30 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden
      />

      {/* Panel */}
      <div
        ref={ref}
        className={`relative w-full ${sizeClass[size]} rounded-xl bg-white shadow-soft-3 border border-ink-200`}
      >
        {/* Header */}
        <div className="flex items-start justify-between gap-4 px-6 pt-6 pb-3 border-b border-ink-100">
          <div className="min-w-0 flex-1">
            <h2 id="dialog-title" className="text-base font-semibold text-ink-900">
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
            className="text-ink-400 hover:text-ink-700 transition-colors -mt-1 -mr-1 p-1 rounded-md hover:bg-ink-100"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <path d="M18 6L6 18 M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Body */}
        <div className="px-6 py-5">{children}</div>

        {/* Footer (optional) */}
        {footer && (
          <div className="px-6 py-4 border-t border-ink-100 bg-ink-50/40 rounded-b-xl flex items-center justify-end gap-2">
            {footer}
          </div>
        )}
      </div>
    </div>,
    document.body,
  );
}
