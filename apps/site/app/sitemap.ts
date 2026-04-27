/**
 * Auto-generated `/sitemap.xml` for the static export.
 *
 * Enumerates: landing, /changelog, every doc MDX, every blog post,
 * /reference root + each module in the API tree. We don't try to track
 * lastModified per-page — the build timestamp is good enough for
 * a search-engine hint, and adding per-file mtime would mean an
 * fs.stat per entry on every CI build for negligible SEO benefit.
 */
import type { MetadataRoute } from 'next';

import { listContent } from '@/lib/content';
import { listApiModules, moduleSlug } from '@/lib/api-tree';

// Static export requires explicit force-static on metadata routes.
export const dynamic = 'force-static';

const SITE = 'https://pycaret.org';

export default function sitemap(): MetadataRoute.Sitemap {
  const now = new Date();
  const entries: MetadataRoute.Sitemap = [
    { url: `${SITE}/`, lastModified: now, changeFrequency: 'weekly', priority: 1 },
    { url: `${SITE}/changelog/`, lastModified: now, changeFrequency: 'weekly', priority: 0.8 },
    { url: `${SITE}/blog/`, lastModified: now, changeFrequency: 'weekly', priority: 0.6 },
    { url: `${SITE}/reference/`, lastModified: now, changeFrequency: 'weekly', priority: 0.8 },
  ];

  // Docs pages.
  for (const entry of listContent('docs')) {
    entries.push({
      url: `${SITE}${entry.href}/`,
      lastModified: now,
      changeFrequency: 'weekly',
      priority: 0.8,
    });
  }

  // Blog posts.
  for (const entry of listContent('blog')) {
    entries.push({
      url: `${SITE}${entry.href}/`,
      lastModified: now,
      changeFrequency: 'monthly',
      priority: 0.5,
    });
  }

  // API reference modules.
  for (const qualname of listApiModules()) {
    entries.push({
      url: `${SITE}/reference/${moduleSlug(qualname).join('/')}/`,
      lastModified: now,
      changeFrequency: 'monthly',
      priority: 0.6,
    });
  }

  return entries;
}
