/**
 * Individual blog post — `/blog/<slug>`.
 *
 * Renders the MDX body of the matching `content/blog/*.mdx` file via
 * the shared MdxRenderer (which routes code fences through Shiki).
 */
import Link from 'next/link';
import { notFound } from 'next/navigation';

import { MdxRenderer } from '@/components/MdxRenderer';
import { getContent, listContent } from '@/lib/content';

interface PageProps {
  params: Promise<{ slug: string }>;
}

export async function generateStaticParams() {
  return listContent('blog').map((post) => ({ slug: post.slug.join('/') }));
}

export async function generateMetadata({ params }: PageProps) {
  const { slug } = await params;
  const post = getContent('blog', [slug]);
  if (!post) return {};
  return {
    title: post.frontmatter.title,
    description: post.frontmatter.description,
  };
}

export default async function BlogPost({ params }: PageProps) {
  const { slug } = await params;
  const post = getContent('blog', [slug]);
  if (!post) notFound();
  return (
    <article className="mx-auto max-w-3xl px-6 py-16">
      <Link
        href="/blog"
        className="text-xs text-ink-500 hover:text-ink-700"
      >
        ← Back to blog
      </Link>
      <header className="mt-6">
        <div className="font-mono text-xs text-ink-500">
          {post.frontmatter.date}
        </div>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight text-ink-900">
          {post.frontmatter.title}
        </h1>
        {post.frontmatter.description && (
          <p className="mt-3 text-lg leading-relaxed text-ink-600">
            {post.frontmatter.description}
          </p>
        )}
      </header>
      <div className="mt-12">
        <MdxRenderer source={post.body} />
      </div>
    </article>
  );
}
