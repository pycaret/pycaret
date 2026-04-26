/**
 * Server-rendered code block via Shiki.
 *
 * Pre-renders to HTML on the server (no JS shipped to the client), so
 * code samples are fast and SEO-friendly. Uses the `github-light`
 * theme to match our typography palette; can swap to a dark theme
 * later via a CSS variable on the body.
 */
import { codeToHtml } from 'shiki';

interface CodeBlockProps {
  code: string;
  language?: string;
  filename?: string;
  className?: string;
}

export async function CodeBlock({
  code,
  language = 'python',
  filename,
  className = '',
}: CodeBlockProps) {
  const html = await codeToHtml(code.trim(), {
    lang: language,
    theme: 'github-light',
  });

  return (
    <div
      className={`overflow-hidden rounded-xl border border-ink-200 bg-white shadow-sm ${className}`}
    >
      {filename && (
        <div className="flex items-center justify-between border-b border-ink-100 bg-ink-50/50 px-4 py-2 font-mono text-xs text-ink-500">
          <span>{filename}</span>
          <span className="uppercase tracking-wider text-ink-400">{language}</span>
        </div>
      )}
      <div
        className="text-sm [&>pre]:!bg-transparent [&>pre]:!p-5"
        dangerouslySetInnerHTML={{ __html: html }}
      />
    </div>
  );
}
