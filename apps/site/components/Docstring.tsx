/**
 * Renders a Python docstring as styled prose with code blocks.
 *
 * The griffe-extracted docstrings are RST-flavored (Sphinx/Napoleon
 * conventions). To get them rendered with proper `<pre><code>` blocks,
 * Shiki highlighting, etc., we convert RST quirks to Markdown and
 * route the result through the same MdxRenderer the docs pages use.
 *
 * Conversions:
 *   - 4-space-indented blocks → fenced ```python``` blocks
 *     (PyCaret docstrings always use Python in indented examples)
 *   - RST inline ``x`` → Markdown `x`
 *   - Sphinx :param:/:returns:/:raises: lines → bold-prefixed lines
 *
 * Anything else falls through unchanged — Markdown parses RST prose
 * close enough that `# headings`, `*emphasis*`, links, and lists work
 * without further translation.
 */
import { MdxRenderer } from './MdxRenderer';

function rstToMarkdown(raw: string): string {
  // 1. RST double-backtick → Markdown single-backtick.
  let s = raw.replace(/``([^`\n]+)``/g, '`$1`');

  // 2. Detect 4-space-indented blocks and wrap in ```python fences.
  //    griffe leaves docstrings as-is, so the indentation pattern
  //    is the standard ">>> from pep-257" 4-space convention.
  const lines = s.split('\n');
  const out: string[] = [];
  let inBlock = false;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const indented = /^ {4,}\S/.test(line);
    const blank = /^\s*$/.test(line);
    if (!inBlock && indented) {
      // Opening fence.
      out.push('```python');
      out.push(line.replace(/^ {4}/, ''));
      inBlock = true;
    } else if (inBlock && (indented || blank)) {
      out.push(blank ? '' : line.replace(/^ {4}/, ''));
    } else if (inBlock && !indented) {
      // Closing fence — drop trailing blanks first.
      while (out.length && out[out.length - 1] === '') out.pop();
      out.push('```');
      out.push('');
      out.push(line);
      inBlock = false;
    } else {
      out.push(line);
    }
  }
  if (inBlock) {
    while (out.length && out[out.length - 1] === '') out.pop();
    out.push('```');
  }
  s = out.join('\n');

  // 3. Sphinx field lists → bold-prefixed.
  s = s.replace(
    /^:(param|parameter|returns|return|raises|raise|type|rtype|note|warning|see also|example|examples)\s*([^:\n]*)?:/gim,
    (_m, kind: string, arg: string) => {
      const cap = kind.charAt(0).toUpperCase() + kind.slice(1);
      const rest = arg && arg.trim() ? ` \`${arg.trim()}\`` : '';
      return `**${cap}${rest}:**`;
    },
  );

  return s;
}

interface DocstringProps {
  text: string | null | undefined;
  /** "compact" tightens the prose typography for nested cards. */
  variant?: 'default' | 'compact';
}

export async function Docstring({ text, variant = 'default' }: DocstringProps) {
  if (!text) return null;
  const md = rstToMarkdown(text);
  return (
    <div
      className={
        variant === 'compact'
          ? 'docstring docstring-compact mt-2 text-sm'
          : 'docstring mt-4 text-base'
      }
    >
      <MdxRenderer source={md} />
    </div>
  );
}
