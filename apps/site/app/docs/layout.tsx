/**
 * Docs layout — sidebar on the left, content on the right.
 *
 * The sidebar is generated from the MDX front-matter at request time
 * (server-rendered, then hydrated only enough to highlight the active
 * link).
 */
import { DocsSidebar } from '@/components/DocsSidebar';
import { buildDocsSidebar } from '@/lib/content';

export default function DocsLayout({ children }: { children: React.ReactNode }) {
  const sections = buildDocsSidebar();
  return (
    <div className="mx-auto w-full max-w-7xl px-6 py-12">
      <div className="flex gap-12">
        <DocsSidebar sections={sections} title="Documentation" />
        <div className="min-w-0 flex-1">{children}</div>
      </div>
    </div>
  );
}
