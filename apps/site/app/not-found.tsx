import Link from 'next/link';

export default function NotFound() {
  return (
    <div className="mx-auto max-w-xl px-6 py-32 text-center">
      <div className="font-mono text-xs uppercase tracking-wider text-accent-600">
        404
      </div>
      <h1 className="mt-3 text-4xl font-semibold tracking-tight text-ink-900">
        Page not found.
      </h1>
      <p className="mt-4 text-ink-600">
        The page you&rsquo;re looking for doesn&rsquo;t exist or has moved.
      </p>
      <div className="mt-8 flex items-center justify-center gap-3">
        <Link
          href="/"
          className="rounded-md bg-ink-900 px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-ink-800"
        >
          Back home
        </Link>
        <Link
          href="/docs/getting-started/installation"
          className="rounded-md border border-ink-200 bg-white px-5 py-2.5 text-sm font-medium text-ink-700 transition-colors hover:bg-ink-50"
        >
          Read the docs
        </Link>
      </div>
    </div>
  );
}
