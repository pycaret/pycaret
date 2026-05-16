/**
 * Inline help icon + tooltip — used everywhere a label needs a "what is
 * this?" affordance without consuming row real estate.
 *
 * The tooltip is CSS-driven (no portal), positioned above the icon by
 * default and centered. Triggers on hover and keyboard focus, dismisses
 * on Escape. Long help text wraps; the bubble caps at 280px wide.
 *
 * Visual style is intentionally restrained — small grey circle with a
 * lowercase "i", same shape across the app so users learn the pattern
 * fast.
 */

import { useId, useState, type ReactNode } from 'react';

export interface InfoIconProps {
  /** The tooltip content. Strings are wrapped, ReactNodes pass through. */
  children: ReactNode;
  /** Optional accessible label override (default: "More info"). */
  label?: string;
  /** Tooltip placement relative to the icon. */
  side?: 'top' | 'bottom';
  className?: string;
}

const SIDE_STYLE: Record<'top' | 'bottom', string> = {
  top: 'bottom-full mb-2',
  bottom: 'top-full mt-2',
};

const ARROW_STYLE: Record<'top' | 'bottom', string> = {
  top: 'top-full -mt-1 border-t-ink-900 dark:border-t-white',
  bottom: 'bottom-full -mb-1 rotate-180 border-t-ink-900 dark:border-t-white',
};

export function InfoIcon({
  children,
  label = 'More info',
  side = 'top',
  className,
}: InfoIconProps) {
  const id = useId();
  const [open, setOpen] = useState(false);

  return (
    <span
      className={['relative inline-flex items-center', className ?? ''].join(' ')}
    >
      <button
        type="button"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onKeyDown={(e) => {
          if (e.key === 'Escape') setOpen(false);
        }}
        aria-label={label}
        aria-describedby={open ? id : undefined}
        className="inline-flex h-3.5 w-3.5 items-center justify-center rounded-full text-[10px] leading-none text-ink-500 hover:text-ink-700 dark:hover:text-ink-300 border border-ink-300 dark:border-ink-700 hover:border-ink-500 dark:hover:border-ink-500 transition-colors cursor-help bg-transparent"
      >
        <span aria-hidden>i</span>
      </button>
      {open && (
        <span
          id={id}
          role="tooltip"
          className={`absolute left-1/2 -translate-x-1/2 z-50 ${SIDE_STYLE[side]} pointer-events-none`}
        >
          <span className="block w-max max-w-[280px] rounded-md bg-ink-900 dark:bg-white text-white dark:text-ink-900 text-xs leading-snug px-2.5 py-1.5 shadow-lg whitespace-normal">
            {children}
          </span>
          <span
            className={`absolute left-1/2 -translate-x-1/2 ${ARROW_STYLE[side]} h-0 w-0 border-x-4 border-x-transparent border-t-4`}
            aria-hidden
          />
        </span>
      )}
    </span>
  );
}
