/**
 * `/reference` index — API reference landing page.
 */
import Link from 'next/link';

import { listApiModules, moduleSlug } from '@/lib/api-tree';

export const metadata = {
  title: 'API reference',
  description: 'Auto-generated reference for every public PyCaret module.',
};

export default function ReferenceIndex() {
  const modules = listApiModules();
  return (
    <div>
      <h1 className="text-3xl font-semibold tracking-tight text-ink-900">
        API reference
      </h1>
      <p className="mt-3 text-lg leading-relaxed text-ink-600">
        Auto-generated from the source via{' '}
        <a
          className="text-accent-600 hover:text-accent-700"
          href="https://mkdocstrings.github.io/griffe/"
          target="_blank"
          rel="noopener noreferrer"
        >
          griffe
        </a>
        . Pinned to the version of <code className="font-mono">pycaret</code>{' '}
        that built this site.
      </p>
      {modules.length === 0 ? (
        <div className="mt-10 rounded-xl border border-ink-200 bg-ink-50 p-6 text-sm text-ink-600">
          The API tree hasn&rsquo;t been generated yet. From the repo root run:
          <pre className="mt-3 rounded bg-white px-3 py-2 font-mono text-xs">
            cd apps/site &amp;&amp; npm run gen:api
          </pre>
        </div>
      ) : (
        <ul className="mt-10 grid gap-3 md:grid-cols-2">
          {modules.map((qualname) => (
            <li key={qualname}>
              <Link
                href={`/reference/${moduleSlug(qualname).join('/')}`}
                className="block rounded-lg border border-ink-100 bg-white px-4 py-3 transition-colors hover:border-accent-200 hover:bg-accent-50/30"
              >
                <div className="font-mono text-sm font-medium text-ink-900">
                  {qualname}
                </div>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
