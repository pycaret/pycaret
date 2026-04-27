/**
 * Footer with link columns + license note. Same minimal aesthetic as
 * the rest of the site.
 */
import Image from 'next/image';
import Link from 'next/link';

const COLUMNS = [
  {
    title: 'Product',
    links: [
      { href: '/docs/getting-started/quickstart', label: 'Quickstart' },
      { href: '/docs/concepts/overview', label: 'Concepts' },
      { href: '/reference', label: 'API reference' },
      { href: '/changelog', label: 'Changelog' },
    ],
  },
  {
    title: 'Resources',
    links: [
      { href: '/blog', label: 'Blog' },
      { href: '/docs/guides/migrating-from-3.x', label: 'Migrate from 3.x' },
      {
        href: 'https://github.com/pycaret/pycaret',
        label: 'GitHub',
        external: true,
      },
      {
        href: 'https://pypi.org/project/pycaret/',
        label: 'PyPI',
        external: true,
      },
    ],
  },
  {
    title: 'Community',
    links: [
      {
        href: 'https://github.com/pycaret/pycaret/discussions',
        label: 'Discussions',
        external: true,
      },
      {
        href: 'https://github.com/pycaret/pycaret/issues',
        label: 'Issues',
        external: true,
      },
      {
        href: 'https://github.com/pycaret/pycaret/blob/main/CONTRIBUTING.md',
        label: 'Contribute',
        external: true,
      },
    ],
  },
] as const;

export function SiteFooter() {
  return (
    <footer className="mt-24 border-t border-ink-100 bg-ink-50/50">
      <div className="mx-auto max-w-6xl px-6 py-12">
        <div className="grid grid-cols-2 gap-8 md:grid-cols-4">
          <div className="col-span-2 md:col-span-1">
            <Image
              src="/logo.png"
              alt="PyCaret"
              width={168}
              height={24}
              className="h-6 w-auto"
            />

            <p className="mt-3 text-sm text-ink-500">
              Low-code machine learning for Python. Open-source under MIT.
            </p>
          </div>
          {COLUMNS.map((col) => (
            <div key={col.title}>
              <div className="text-xs font-semibold uppercase tracking-wider text-ink-500">
                {col.title}
              </div>
              <ul className="mt-3 space-y-2 text-sm">
                {col.links.map((link) => (
                  <li key={link.href}>
                    <Link
                      href={link.href}
                      target={('external' in link && link.external) ? '_blank' : undefined}
                      rel={('external' in link && link.external) ? 'noopener noreferrer' : undefined}
                      className="text-ink-700 transition-colors hover:text-accent-600"
                    >
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
        <div className="mt-10 flex flex-col items-start justify-between gap-2 border-t border-ink-100 pt-6 md:flex-row md:items-center">
          <p className="text-xs text-ink-500">
            © {new Date().getFullYear()} PyCaret contributors. Released under the MIT License.
          </p>
          <p className="text-xs text-ink-500">
            Built with Next.js. Source on{' '}
            <Link
              href="https://github.com/pycaret/pycaret/tree/main/apps/site"
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent-600 hover:text-accent-700"
            >
              GitHub
            </Link>
            .
          </p>
        </div>
      </div>
    </footer>
  );
}
