/**
 * Sticky sidebar nav for the docs / reference layouts.
 *
 * Renders a list of grouped links. The active item is computed
 * client-side via usePathname so we can highlight without a full
 * re-render — but the list itself is server-generated so it's still
 * crawlable + fast on first paint.
 */
'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

import type { SidebarSection } from '@/lib/content';

interface DocsSidebarProps {
  sections: SidebarSection[];
  /** Title shown above the nav (e.g. "Documentation" / "API reference"). */
  title?: string;
}

export function DocsSidebar({ sections, title = 'Documentation' }: DocsSidebarProps) {
  const pathname = usePathname();
  return (
    <aside className="sticky top-20 max-h-[calc(100vh-5rem)] w-64 shrink-0 overflow-y-auto pr-4">
      <div className="text-xs font-semibold uppercase tracking-wider text-ink-500">
        {title}
      </div>
      <nav className="mt-3 space-y-6">
        {sections.map((section) => (
          <div key={section.title}>
            <div className="text-xs font-semibold uppercase tracking-wider text-ink-400">
              {section.title}
            </div>
            <ul className="mt-2 space-y-1">
              {section.items.map((item) => {
                const active =
                  pathname === item.href ||
                  pathname === `${item.href}/` ||
                  pathname?.startsWith(`${item.href}/`);
                return (
                  <li key={item.href}>
                    <Link
                      href={item.href}
                      className={
                        active
                          ? 'block rounded-md bg-accent-50 px-3 py-1.5 text-sm font-medium text-accent-700'
                          : 'block rounded-md px-3 py-1.5 text-sm text-ink-600 transition-colors hover:bg-ink-50 hover:text-ink-900'
                      }
                    >
                      {item.title}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </nav>
    </aside>
  );
}
