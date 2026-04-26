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
  experimental: {
    typedRoutes: false,
  },
};

export default config;
