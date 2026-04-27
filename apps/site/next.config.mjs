/**
 * Next.js config for the PyCaret public site.
 *
 * - Static export so deployment to any CDN (Vercel, Cloudflare Pages,
 *   GitHub Pages) is trivial.
 * - MDX is consumed via the next-mdx-remote-client at request time, not
 *   compiled at build time, so we don't need the @next/mdx plugin here.
 */
/** @type {import('next').NextConfig} */
const config = {
  reactStrictMode: true,
  output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
  // Next.js 16 promoted typedRoutes from experimental to top-level. We
  // keep it off because the site uses string hrefs (constructed from
  // MDX front-matter / API tree) that the type-checker can't statically
  // verify.
  typedRoutes: false,
};

export default config;
