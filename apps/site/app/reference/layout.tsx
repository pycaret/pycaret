/**
 * `/reference` layout — sidebar (auto-generated from the API tree) +
 * content area.
 */
import { DocsSidebar } from '@/components/DocsSidebar';
import { buildApiSidebar } from '@/lib/api-tree';

export default function ReferenceLayout({ children }: { children: React.ReactNode }) {
  const sections = buildApiSidebar();
  return (
    <div className="mx-auto w-full max-w-7xl px-6 py-12">
      <div className="flex gap-12">
        <DocsSidebar sections={sections} title="API reference" />
        <div className="min-w-0 flex-1">{children}</div>
      </div>
    </div>
  );
}
