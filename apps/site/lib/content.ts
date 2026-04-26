/**
 * Content loader for MDX-backed pages.
 *
 * Reads files from `content/<section>/<slug>.mdx`, parses front-matter
 * via gray-matter, and exposes a typed API for page routes to consume.
 *
 * Front-matter shape::
 *
 *     ---
 *     title: "Quickstart"
 *     description: "Set up an experiment in 60 seconds."
 *     order: 1                   # within its section
 *     section: "Getting started" # for the sidebar grouping
 *     ---
 *
 * The site has three content roots: `docs`, `blog`. The API reference
 * (`reference/`) is generated separately from `content/api-tree.json`.
 */
import fs from 'node:fs';
import path from 'node:path';

import matter from 'gray-matter';

const ROOT = path.join(process.cwd(), 'content');

export interface ContentFrontmatter {
  title: string;
  description?: string;
  order?: number;
  section?: string;
  date?: string; // for blog
  authors?: string[];
  tags?: string[];
}

export interface ContentEntry {
  slug: string[];
  href: string;
  frontmatter: ContentFrontmatter;
  body: string;
}

/** Walk every .mdx file under `content/<root>/`. */
export function listContent(rootSegment: 'docs' | 'blog'): ContentEntry[] {
  const root = path.join(ROOT, rootSegment);
  if (!fs.existsSync(root)) return [];
  const entries: ContentEntry[] = [];
  const walk = (dir: string, slug: string[]) => {
    for (const name of fs.readdirSync(dir)) {
      const full = path.join(dir, name);
      const stat = fs.statSync(full);
      if (stat.isDirectory()) {
        walk(full, [...slug, name]);
      } else if (name.endsWith('.mdx') || name.endsWith('.md')) {
        const raw = fs.readFileSync(full, 'utf8');
        const parsed = matter(raw);
        const fm = parsed.data as ContentFrontmatter;
        const baseName = name.replace(/\.mdx?$/, '');
        const fileSlug = [...slug, baseName === 'index' ? '' : baseName].filter(Boolean);
        entries.push({
          slug: fileSlug,
          href: `/${rootSegment}${fileSlug.length ? `/${fileSlug.join('/')}` : ''}`,
          frontmatter: {
            title: fm.title ?? baseName,
            description: fm.description,
            order: fm.order ?? 99,
            section: fm.section,
            date: fm.date,
            authors: fm.authors,
            tags: fm.tags,
          },
          body: parsed.content,
        });
      }
    }
  };
  walk(root, []);
  // Sort by section + order for stable rendering.
  entries.sort((a, b) => {
    const ax = a.frontmatter;
    const bx = b.frontmatter;
    if ((ax.section ?? '') !== (bx.section ?? '')) {
      return (ax.section ?? '').localeCompare(bx.section ?? '');
    }
    return (ax.order ?? 99) - (bx.order ?? 99);
  });
  return entries;
}

/** Look up a single entry by slug parts. */
export function getContent(
  rootSegment: 'docs' | 'blog',
  slug: string[],
): ContentEntry | null {
  const list = listContent(rootSegment);
  const target = slug.join('/');
  return (
    list.find(
      (e) => e.slug.join('/') === target || (target === '' && e.slug.length === 0),
    ) ?? null
  );
}

/** Group docs entries into the sidebar tree. */
export interface SidebarSection {
  title: string;
  items: Array<{ title: string; href: string; description?: string }>;
}

export function buildDocsSidebar(): SidebarSection[] {
  const list = listContent('docs');
  const groups = new Map<string, SidebarSection>();
  for (const entry of list) {
    const sectionTitle = entry.frontmatter.section ?? 'Other';
    if (!groups.has(sectionTitle)) {
      groups.set(sectionTitle, { title: sectionTitle, items: [] });
    }
    groups.get(sectionTitle)!.items.push({
      title: entry.frontmatter.title,
      href: entry.href,
      description: entry.frontmatter.description,
    });
  }
  // Stable section ordering for the docs sidebar.
  const order = [
    'Getting started',
    'Concepts',
    'Preprocessing',
    'Functions',
    'Guides',
    'Resources',
  ];
  return Array.from(groups.values()).sort((a, b) => {
    const ai = order.indexOf(a.title);
    const bi = order.indexOf(b.title);
    if (ai === -1 && bi === -1) return a.title.localeCompare(b.title);
    if (ai === -1) return 1;
    if (bi === -1) return -1;
    return ai - bi;
  });
}
