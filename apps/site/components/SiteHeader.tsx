/**
 * Sticky top header. Logo + primary nav + GitHub stars + version pill.
 *
 * Kept minimal — this is a marketing site, not the dashboard. Mobile
 * collapses the nav into a slide-down sheet via the details/summary
 * idiom (zero JS, screen-reader friendly).
 */
import Link from 'next/link';
import { Github } from 'lucide-react';

const NAV = [
  { href: '/docs/getting-started/installation', label: 'Docs' },
  { href: '/reference', label: 'API' },
  { href: '/blog', label: 'Blog' },
  { href: '/changelog', label: 'Changelog' },
] as const;

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-40 border-b border-ink-100 bg-white/80 backdrop-blur supports-[backdrop-filter]:bg-white/60">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6">
        <div className="flex items-center gap-8">
          <Link href="/" className="flex items-center gap-2 group">
            <span
              aria-hidden
              className="block h-7 w-7 rounded-md bg-gradient-to-br from-accent-500 to-accent-700 shadow-sm transition-transform group-hover:scale-105"
            />
            <span className="text-base font-semibold tracking-tight text-ink-900">
              PyCaret
            </span>
          </Link>
          <nav className="hidden items-center gap-6 text-sm md:flex">
            {NAV.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="text-ink-600 transition-colors hover:text-ink-900"
              >
                {item.label}
              </Link>
            ))}
          </nav>
        </div>

        <div className="flex items-center gap-3">
          <Link
            href="https://github.com/pycaret/pycaret"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-md border border-ink-200 bg-white px-3 py-1.5 text-xs font-medium text-ink-700 shadow-sm transition-colors hover:border-ink-300 hover:bg-ink-50"
          >
            <Github className="h-3.5 w-3.5" aria-hidden />
            <span>GitHub</span>
          </Link>
          <Link
            href="/docs/getting-started/installation"
            className="hidden rounded-md bg-ink-900 px-3 py-1.5 text-xs font-medium text-white shadow-sm transition-colors hover:bg-ink-800 sm:inline-flex"
          >
            Get started →
          </Link>
        </div>
      </div>
    </header>
  );
}
