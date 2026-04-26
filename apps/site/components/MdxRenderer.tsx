/**
 * Server-rendered MDX with Shiki-highlighted code fences.
 *
 * Uses next-mdx-remote-client so we can consume MDX from any source
 * (filesystem, CMS) without committing to the build-time @next/mdx
 * pipeline. Code fences flow through Shiki at render time.
 */
import { MDXRemote } from 'next-mdx-remote-client/rsc';
import remarkGfm from 'remark-gfm';
import rehypeSlug from 'rehype-slug';
import rehypeAutolinkHeadings from 'rehype-autolink-headings';

import { CodeBlock } from './CodeBlock';

interface MdxRendererProps {
  source: string;
}

const components = {
  pre: ({ children }: { children?: React.ReactNode }) => {
    // The raw <pre> wrapping a <code> from MDX. We fish out the code
    // string + language from props and re-render via Shiki.
    if (
      children &&
      typeof children === 'object' &&
      'props' in children &&
      typeof (children as { props?: { children?: unknown } }).props?.children === 'string'
    ) {
      const inner = children as {
        props: { children: string; className?: string };
      };
      const lang = (inner.props.className ?? '').replace('language-', '') || 'text';
      return <CodeBlock code={inner.props.children} language={lang} />;
    }
    return <pre>{children}</pre>;
  },
};

export async function MdxRenderer({ source }: MdxRendererProps) {
  return (
    <article className="prose-pycaret">
      <MDXRemote
        source={source}
        components={components}
        options={{
          mdxOptions: {
            // 'md' parses input as plain Markdown (no JSX expressions),
            // which is what we want for both hand-written guides and the
            // auto-imported release notes — the latter contain `<` / `{`
            // characters as part of code samples / type signatures and
            // would otherwise trip the MDX/JSX expression parser.
            format: 'md',
            remarkPlugins: [remarkGfm],
            rehypePlugins: [
              rehypeSlug,
              [
                rehypeAutolinkHeadings,
                {
                  behavior: 'append',
                  properties: {
                    className: ['heading-anchor'],
                    'aria-label': 'Anchor link',
                  },
                  content: { type: 'text', value: '#' },
                },
              ],
            ],
          },
        }}
      />
    </article>
  );
}
