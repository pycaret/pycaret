/**
 * Auto-generated `/robots.txt` for the static export.
 *
 * Allows everything; the only restriction we'd want is keeping crawlers
 * out of the API tree's internal anchors, but the JSON itself isn't
 * served as HTML so this is moot.
 */
import type { MetadataRoute } from 'next';

// Static export requires explicit force-static on metadata routes.
export const dynamic = 'force-static';

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      {
        userAgent: '*',
        allow: '/',
      },
    ],
    sitemap: 'https://pycaret.org/sitemap.xml',
    host: 'https://pycaret.org',
  };
}
