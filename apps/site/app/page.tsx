/**
 * Landing page — pycaret.org/.
 *
 * Modern minimalist hero + feature grid + code sample + CTA. Heavy on
 * whitespace, light on chrome. The structure is a series of sections
 * stacked vertically; each one is a small Server Component imported
 * from `app/(home)/sections/`.
 */
import Link from 'next/link';
import {
  ArrowRight,
  BarChart3,
  Brain,
  Cpu,
  GitBranch,
  Layers,
  Rocket,
  Sparkles,
  Zap,
} from 'lucide-react';

import { CodeBlock } from '@/components/CodeBlock';

export default function HomePage() {
  return (
    <>
      <Hero />
      <SocialProof />
      <CodeSample />
      <FeatureGrid />
      <DashboardPreview />
      <Cta />
    </>
  );
}

// ───────────────────────────────────────────────────────────── hero

function Hero() {
  return (
    <section className="relative overflow-hidden">
      {/* Subtle background gradient. */}
      <div
        aria-hidden
        className="absolute inset-0 bg-gradient-to-b from-accent-50/40 via-white to-white"
      />
      <div className="relative mx-auto max-w-6xl px-6 pb-20 pt-24 md:pt-32">
        <div className="mx-auto max-w-3xl text-center">
          <span className="inline-flex items-center gap-2 rounded-full border border-accent-200 bg-accent-50 px-3 py-1 text-xs font-medium text-accent-700">
            <Sparkles className="h-3.5 w-3.5" aria-hidden />
            PyCaret 4.0 · Native sklearn engine + React dashboard
          </span>
          <h1 className="mt-6 text-balance text-5xl font-semibold tracking-tight text-ink-900 md:text-6xl">
            Low-code machine learning for{' '}
            <span className="bg-gradient-to-r from-accent-600 to-accent-800 bg-clip-text text-transparent">
              Python
            </span>
            .
          </h1>
          <p className="mt-6 text-pretty text-lg text-ink-600 md:text-xl">
            Set up an experiment, compare a dozen models, tune the winner, deploy a
            pipeline — in under twenty lines. Open-source. No vendor lock-in. Runs on
            your laptop or your cluster.
          </p>
          <div className="mt-10 flex flex-col items-center justify-center gap-3 sm:flex-row">
            <Link
              href="/docs/getting-started/installation"
              className="inline-flex items-center gap-2 rounded-md bg-ink-900 px-5 py-3 text-sm font-medium text-white shadow-sm transition-all hover:bg-ink-800 hover:shadow-md"
            >
              Get started
              <ArrowRight className="h-4 w-4" aria-hidden />
            </Link>
            <Link
              href="/docs/concepts/overview"
              className="inline-flex items-center gap-2 rounded-md border border-ink-200 bg-white px-5 py-3 text-sm font-medium text-ink-800 transition-all hover:border-ink-300 hover:bg-ink-50"
            >
              Read the concepts
            </Link>
          </div>
          <div className="mt-6 flex items-center justify-center gap-1 font-mono text-sm text-ink-500">
            <span className="text-ink-400">$</span>
            <code className="select-all">pip install pycaret</code>
          </div>
        </div>
      </div>
    </section>
  );
}

// ─────────────────────────────────────────────────────── social proof

function SocialProof() {
  // Logos / mentions are just textual placeholders for now —
  // the real thing comes from a curated list once partners exist.
  return (
    <section className="border-y border-ink-100 bg-white py-10">
      <div className="mx-auto max-w-6xl px-6">
        <p className="text-center text-xs font-medium uppercase tracking-wider text-ink-500">
          Used by data teams shipping production models · 8M+ PyPI downloads
        </p>
        <div className="mt-6 grid grid-cols-2 items-center gap-6 opacity-60 md:grid-cols-5 md:gap-8">
          {[
            'scikit-learn',
            'sktime',
            'optuna',
            'plotly',
            'fastapi',
          ].map((name) => (
            <div
              key={name}
              className="flex items-center justify-center font-mono text-sm text-ink-500"
            >
              {name}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// ─────────────────────────────────────────────────────── code sample

function CodeSample() {
  const code = `from pycaret.classification import ClassificationExperiment
from pycaret.datasets import get_data

# Sample dataset.
data = get_data("juice")

# Set up an experiment + compare 12 models.
exp = ClassificationExperiment(target="Purchase").fit(data)
top = exp.compare_models(n_select=3)

# Tune the winner. Pipeline-in / Pipeline-out.
tuned = exp.tune_model(top.best, n_iter=20)

# Deploy.
pipeline = exp.finalize_model(tuned.pipeline).pipeline
exp.save_model(pipeline, "production-juice-model")`;

  return (
    <section className="py-20 md:py-28">
      <div className="mx-auto max-w-6xl px-6">
        <div className="grid items-center gap-12 md:grid-cols-2 md:gap-16">
          <div>
            <div className="text-xs font-semibold uppercase tracking-wider text-accent-600">
              Twenty lines
            </div>
            <h2 className="mt-2 text-3xl font-semibold tracking-tight text-ink-900 md:text-4xl">
              The whole AutoML loop in your notebook.
            </h2>
            <p className="mt-4 text-ink-600">
              Setup, comparison, tuning, deployment — every step exposes a real
              sklearn-compatible <code className="font-mono">Pipeline</code>. No
              wrappers, no magic, no &quot;experiment.predict_model_2()&quot;.
            </p>
            <ul className="mt-6 space-y-3 text-sm text-ink-700">
              {[
                'Native sklearn 1.7+ pipelines (no internal wrappers).',
                'Same OOP API for classification, regression, clustering, anomaly, and time-series.',
                'Optional dashboard for teams who want a UI on top.',
              ].map((line) => (
                <li key={line} className="flex items-start gap-2">
                  <Zap
                    className="mt-0.5 h-4 w-4 shrink-0 text-accent-600"
                    aria-hidden
                  />
                  <span>{line}</span>
                </li>
              ))}
            </ul>
          </div>
          <CodeBlock language="python" code={code} />
        </div>
      </div>
    </section>
  );
}

// ───────────────────────────────────────────────────────── features

function FeatureGrid() {
  const items: Array<{ icon: React.ReactNode; title: string; body: string }> = [
    {
      icon: <Layers className="h-5 w-5" />,
      title: 'Five tasks, one API',
      body: 'Classification, regression, clustering, anomaly, time-series — same Experiment object, same verb names, same pipeline contract.',
    },
    {
      icon: <Brain className="h-5 w-5" />,
      title: 'sklearn-native',
      body: 'Every fitted output is a real sklearn or sktime Pipeline. joblib.dump it. Mount it behind a FastAPI route. Zero wrapping.',
    },
    {
      icon: <BarChart3 className="h-5 w-5" />,
      title: 'Plotly-native diagnostics',
      body: 'Confusion matrix, ROC, residuals, SHAP, decomposition, drift — all interactive, all framework-agnostic figures.',
    },
    {
      icon: <Rocket className="h-5 w-5" />,
      title: 'Production-ready',
      body: 'A built-in registry + serving layer turns any fitted pipeline into an HTTP endpoint with rolling p50/p95 stats.',
    },
    {
      icon: <GitBranch className="h-5 w-5" />,
      title: 'Workspace-aware',
      body: 'Multi-tenant projects, runs, deployments, and audit logs — wired into a FastAPI backend you can self-host.',
    },
    {
      icon: <Cpu className="h-5 w-5" />,
      title: 'Lean dependencies',
      body: 'PyCaret 4.0 dropped 50+ legacy dependencies. The core engine installs in seconds.',
    },
  ];

  return (
    <section className="border-y border-ink-100 bg-ink-50/40 py-20 md:py-28">
      <div className="mx-auto max-w-6xl px-6">
        <div className="mx-auto max-w-2xl text-center">
          <div className="text-xs font-semibold uppercase tracking-wider text-accent-600">
            Why PyCaret 4.0
          </div>
          <h2 className="mt-2 text-3xl font-semibold tracking-tight text-ink-900 md:text-4xl">
            Modern internals, the same simple API.
          </h2>
          <p className="mt-4 text-ink-600">
            We rewrote the engine from the ground up against sklearn 1.7 and Python
            3.11+. The notebook code you wrote in 3.x still works — it&rsquo;s just
            faster and lighter underneath.
          </p>
        </div>
        <div className="mt-12 grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {items.map((it) => (
            <div
              key={it.title}
              className="rounded-xl border border-ink-100 bg-white p-6 shadow-sm transition-shadow hover:shadow-md"
            >
              <div className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-accent-50 text-accent-600">
                {it.icon}
              </div>
              <h3 className="mt-4 text-base font-semibold text-ink-900">{it.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-ink-600">{it.body}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// ─────────────────────────────────────────────────── dashboard preview

function DashboardPreview() {
  return (
    <section className="py-20 md:py-28">
      <div className="mx-auto max-w-6xl px-6">
        <div className="mx-auto max-w-2xl text-center">
          <div className="text-xs font-semibold uppercase tracking-wider text-accent-600">
            Optional dashboard
          </div>
          <h2 className="mt-2 text-3xl font-semibold tracking-tight text-ink-900 md:text-4xl">
            A cockpit for the rest of your team.
          </h2>
          <p className="mt-4 text-ink-600">
            Give analysts and product owners a real UI: experiment design, run
            comparison, model cards with diagnostic plots, drift monitoring,
            prediction explorer. Self-hosted. MIT-licensed.
          </p>
        </div>

        <div className="mt-12 overflow-hidden rounded-2xl border border-ink-200 bg-gradient-to-br from-ink-50 to-white shadow-xl">
          <div className="flex items-center gap-2 border-b border-ink-200 bg-white px-4 py-3">
            <span className="block h-3 w-3 rounded-full bg-red-300" aria-hidden />
            <span className="block h-3 w-3 rounded-full bg-yellow-300" aria-hidden />
            <span className="block h-3 w-3 rounded-full bg-green-300" aria-hidden />
            <span className="ml-3 font-mono text-xs text-ink-500">
              pycaret.org/dashboard
            </span>
          </div>
          <div className="p-8 md:p-12">
            <div className="grid gap-4 md:grid-cols-3">
              {[
                ['Active experiments', '4'],
                ['Runs (7d)', '37'],
                ['Healthy deployments', '3'],
              ].map(([label, value]) => (
                <div
                  key={label}
                  className="rounded-lg border border-ink-100 bg-white p-4"
                >
                  <div className="text-xs uppercase tracking-wider text-ink-500">
                    {label}
                  </div>
                  <div className="mt-1 text-2xl font-semibold text-ink-900">
                    {value}
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-6 grid gap-4 md:grid-cols-2">
              <div className="rounded-lg border border-ink-100 bg-white p-4">
                <div className="text-sm font-semibold text-ink-900">
                  Confusion matrix
                </div>
                <div className="mt-3 grid grid-cols-2 gap-px overflow-hidden rounded">
                  {[
                    ['true_neg', 412, 'bg-accent-200'],
                    ['false_pos', 31, 'bg-accent-50'],
                    ['false_neg', 47, 'bg-accent-50'],
                    ['true_pos', 286, 'bg-accent-300'],
                  ].map(([key, count, bg]) => (
                    <div
                      key={key as string}
                      className={`${bg as string} flex items-center justify-center py-6 text-sm font-mono text-ink-900`}
                    >
                      {count as number}
                    </div>
                  ))}
                </div>
              </div>
              <div className="rounded-lg border border-ink-100 bg-white p-4">
                <div className="text-sm font-semibold text-ink-900">Leaderboard</div>
                <ul className="mt-3 space-y-2 text-sm">
                  {[
                    ['XGBoost', '0.943'],
                    ['LightGBM', '0.939'],
                    ['Random Forest', '0.928'],
                    ['Logistic Regression', '0.881'],
                  ].map(([model, score]) => (
                    <li
                      key={model}
                      className="flex items-center justify-between rounded px-3 py-1.5 odd:bg-ink-50"
                    >
                      <span>{model}</span>
                      <span className="font-mono text-ink-700">{score}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

// ────────────────────────────────────────────────────────────── cta

function Cta() {
  return (
    <section className="py-20 md:py-28">
      <div className="mx-auto max-w-3xl px-6 text-center">
        <h2 className="text-3xl font-semibold tracking-tight text-ink-900 md:text-4xl">
          Ready when you are.
        </h2>
        <p className="mt-4 text-ink-600">
          Install in seconds. Migrate from 3.x in minutes. Read the docs, browse the
          API, or jump straight into a notebook.
        </p>
        <div className="mt-8 flex flex-col items-center justify-center gap-3 sm:flex-row">
          <Link
            href="/docs/getting-started/quickstart"
            className="inline-flex items-center gap-2 rounded-md bg-ink-900 px-5 py-3 text-sm font-medium text-white shadow-sm transition-all hover:bg-ink-800"
          >
            Run the quickstart
            <ArrowRight className="h-4 w-4" aria-hidden />
          </Link>
          <Link
            href="https://github.com/pycaret/pycaret"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-md border border-ink-200 bg-white px-5 py-3 text-sm font-medium text-ink-800 transition-all hover:border-ink-300 hover:bg-ink-50"
          >
            Star on GitHub
          </Link>
        </div>
      </div>
    </section>
  );
}
