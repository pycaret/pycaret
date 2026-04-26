/**
 * Dynamic catch-all docs route.
 *
 * Resolves `/docs/<a>/<b>/...` against the MDX entries under
 * `content/docs/`. Static-export friendly: `generateStaticParams`
 * enumerates every entry at build time.
 */
import { notFound } from 'next/navigation';

import { MdxRenderer } from '@/components/MdxRenderer';
import { getContent, listContent } from '@/lib/content';

interface PageProps {
  params: Promise<{ slug?: string[] }>;
}

export async function generateStaticParams() {
  const list = listContent('docs');
  return list.map((entry) => ({ slug: entry.slug.length === 0 ? [] : entry.slug }));
}

export async function generateMetadata({ params }: PageProps) {
  const resolved = await params;
  const entry = getContent('docs', resolved.slug ?? []);
  if (!entry) return {};
  return {
    title: entry.frontmatter.title,
    description: entry.frontmatter.description,
  };
}

export default async function DocsPage({ params }: PageProps) {
  const resolved = await params;
  const entry = getContent('docs', resolved.slug ?? []);
  if (!entry) notFound();
  return (
    <div>
      {entry.frontmatter.section && (
        <div className="mb-2 text-xs font-medium uppercase tracking-wider text-accent-600">
          {entry.frontmatter.section}
        </div>
      )}
      <h1 className="text-3xl font-semibold tracking-tight text-ink-900">
        {entry.frontmatter.title}
      </h1>
      {entry.frontmatter.description && (
        <p className="mt-3 text-lg leading-relaxed text-ink-600">
          {entry.frontmatter.description}
        </p>
      )}
      <div className="mt-10">
        <MdxRenderer source={entry.body} />
      </div>
    </div>
  );
}
