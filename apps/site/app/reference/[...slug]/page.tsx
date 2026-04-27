/**
 * Dynamic API reference page — `/reference/<task>/<module>/...`.
 *
 * Slug parts are joined back into a `pycaret.<...>` qualname and looked
 * up in the auto-generated API tree.
 */
import Link from 'next/link';
import { notFound } from 'next/navigation';

import { listApiModules, getApiModule, moduleSlug } from '@/lib/api-tree';
import type { ApiClass, ApiFunction } from '@/lib/api-tree';
import { Docstring } from '@/components/Docstring';

interface PageProps {
  params: Promise<{ slug: string[] }>;
}

export async function generateStaticParams() {
  return listApiModules().map((qualname) => ({ slug: moduleSlug(qualname) }));
}

export async function generateMetadata({ params }: PageProps) {
  const { slug } = await params;
  const qualname = `pycaret.${slug.join('.')}`;
  return {
    title: qualname,
    description: `API reference for ${qualname}.`,
  };
}

export default async function ApiModulePage({ params }: PageProps) {
  const { slug } = await params;
  const qualname = `pycaret.${slug.join('.')}`;
  const mod = getApiModule(qualname);
  if (!mod) notFound();

  return (
    <div>
      <div className="mb-2 text-xs font-medium uppercase tracking-wider text-accent-600">
        Module
      </div>
      <h1 className="font-mono text-3xl font-semibold tracking-tight text-ink-900">
        {qualname}
      </h1>
      <Docstring text={mod.docstring} />


      <SectionHeader title="Classes" count={mod.classes.length} />
      {mod.classes.map((cls) => (
        <ClassCard key={cls.qualname} cls={cls} />
      ))}

      <SectionHeader title="Functions" count={mod.functions.length} />
      {mod.functions.map((fn) => (
        <FunctionCard key={fn.qualname} fn={fn} />
      ))}

      {mod.attributes.length > 0 && (
        <>
          <SectionHeader title="Attributes" count={mod.attributes.length} />
          <ul className="mt-4 space-y-2 text-sm text-ink-700">
            {mod.attributes.map((a) => (
              <li key={a.name} className="rounded bg-ink-50 px-3 py-2 font-mono">
                <span className="font-medium text-ink-900">{a.name}</span>
                {a.annotation && <span className="ml-1 text-ink-500">: {a.annotation}</span>}
                {a.value && <span className="ml-1 text-ink-500">= {a.value}</span>}
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

function SectionHeader({ title, count }: { title: string; count: number }) {
  if (count === 0) return null;
  return (
    <h2 className="mt-12 border-b border-ink-100 pb-2 text-xs font-semibold uppercase tracking-wider text-ink-500">
      {title} <span className="ml-2 text-ink-400">{count}</span>
    </h2>
  );
}

function ClassCard({ cls }: { cls: ApiClass }) {
  return (
    <section className="mt-6 rounded-xl border border-ink-100 bg-white p-6 shadow-sm">
      <div className="flex items-baseline gap-3">
        <h3 className="font-mono text-xl font-semibold text-ink-900">{cls.name}</h3>
        {cls.bases.length > 0 && (
          <span className="text-xs text-ink-500">
            extends {cls.bases.filter(Boolean).join(', ')}
          </span>
        )}
      </div>
      <Docstring text={cls.docstring} variant="compact" />

      {cls.attributes.length > 0 && (
        <div className="mt-4">
          <div className="text-xs font-semibold uppercase tracking-wider text-ink-400">
            Attributes
          </div>
          <ul className="mt-2 space-y-1.5 text-sm text-ink-700">
            {cls.attributes.map((a) => (
              <li key={a.name} className="font-mono">
                <span className="font-medium text-ink-900">{a.name}</span>
                {a.annotation && <span className="ml-1 text-ink-500">: {a.annotation}</span>}
              </li>
            ))}
          </ul>
        </div>
      )}
      {cls.methods.length > 0 && (
        <div className="mt-4">
          <div className="text-xs font-semibold uppercase tracking-wider text-ink-400">
            Methods
          </div>
          <div className="mt-2 space-y-3">
            {cls.methods.map((m) => (
              <FunctionCard key={m.qualname} fn={m} compact />
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function FunctionCard({ fn, compact = false }: { fn: ApiFunction; compact?: boolean }) {
  return (
    <section
      className={
        compact
          ? 'rounded-lg border border-ink-100 bg-ink-50/40 px-4 py-3'
          : 'mt-6 rounded-xl border border-ink-100 bg-white p-6 shadow-sm'
      }
    >
      <div className="font-mono text-sm">
        <span className="font-medium text-ink-900">{fn.name}</span>
        <span className="text-ink-600">
          {fn.signature.replace(`${fn.name}`, '')}
        </span>
      </div>
      <Docstring text={fn.docstring} variant="compact" />

    </section>
  );
}
