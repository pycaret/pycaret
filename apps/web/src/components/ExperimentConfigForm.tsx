/**
 * Experiment configuration form — schema-aware but opinionated.
 *
 * Replaces the generic DynamicForm + "Run plan" duo on /new-experiment with
 * a curated, visual layout. Maintainer (Moez, 2026-05-15) feedback:
 *
 *   - "Run plan" was redundant: experiments always compare models. The
 *     submit always sends plan='compare'.
 *   - Redundant noise dropped entirely (`preprocess`, `log_experiment`,
 *     `verbose`, `n_jobs`) — UI app doesn't need them.
 *   - `train_size` is a modern slider, not a number input.
 *   - Preprocessing toggles get visual cards with mini SVG aids so a user
 *     can decide without re-reading documentation.
 *   - CV / reproducibility / compute live in a quieter bottom row.
 *
 * Schema source of truth is still `describe/setup-params(task)` — we read
 * defaults + descriptions from there, but render dedicated widgets for the
 * params we care about instead of the generic input-per-kind fallback.
 */

import type { SetupParam, SetupParamSchema } from '@/api/types';
import type { ParamValues } from './DynamicForm.helpers';

export interface ExperimentConfigFormProps {
  schema: SetupParamSchema;
  values: ParamValues;
  onChange: (next: ParamValues) => void;
  disabled?: boolean;
  /** Optional row count from the picked DataSource — used to show the
   *  absolute train / test row split next to the slider. */
  rowCount?: number;
}

const HIDDEN_PARAM_NAMES = new Set([
  'target',
  'train_size',     // rendered as the slider below
  'preprocess',     // always true
  'log_experiment', // UI doesn't expose mlflow toggling
  'verbose',        // UI doesn't expose stdout logging
  'n_jobs',         // always -1 (use all cores)
]);

// We hand-render these. Everything else falls through to a small "Other
// options" group at the bottom so future engine params don't silently
// disappear from the UI.
const HAND_RENDERED = new Set([
  ...HIDDEN_PARAM_NAMES,
  'normalize',
  'transformation',
  'remove_outliers',
  'feature_selection',
  'fold',
  'fold_strategy',
  'session_id',
  'use_gpu',
]);

export function ExperimentConfigForm({
  schema,
  values,
  onChange,
  disabled,
  rowCount,
}: ExperimentConfigFormProps) {
  const byName = new Map(schema.parameters.map((p) => [p.name, p]));

  function setOne(name: string, v: unknown) {
    onChange({ ...values, [name]: v });
  }

  function valueOr<T>(name: string, fallback: T): T {
    const v = values[name];
    if (v === undefined || v === null) return fallback;
    return v as T;
  }

  const trainSize = Math.round(valueOr('train_size', 0.7) * 100);
  const others = schema.parameters.filter(
    (p) => !HAND_RENDERED.has(p.name),
  );

  return (
    <div className="space-y-6">
      {/* ── Train/test split ─────────────────────────────────────── */}
      <SubSection
        title="Train / test split"
        hint={byName.get('train_size')?.description}
      >
        <TrainSizeSlider
          value={trainSize}
          onChange={(pct) => setOne('train_size', pct / 100)}
          disabled={disabled}
          rowCount={rowCount}
        />
      </SubSection>

      {/* ── Preprocessing cards ──────────────────────────────────── */}
      <SubSection
        title="Preprocessing"
        hint="Toggle the transformations applied to numeric features before training."
      >
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
          <PreprocCard
            label="Normalize"
            description="Z-score scale numeric features. Helps KNN, SVM, neural nets — anything distance-based."
            checked={valueOr('normalize', false)}
            onToggle={(b) => setOne('normalize', b)}
            disabled={disabled}
            visual={<NormalizeVisual on={valueOr('normalize', false)} />}
          />
          <PreprocCard
            label="Transformation"
            description="Yeo-Johnson power transform — pushes skewed numerics toward a Gaussian."
            checked={valueOr('transformation', false)}
            onToggle={(b) => setOne('transformation', b)}
            disabled={disabled}
            visual={<TransformationVisual on={valueOr('transformation', false)} />}
          />
          <PreprocCard
            label="Remove outliers"
            description="Drop rows flagged by Isolation Forest before training. Off by default — outliers are sometimes signal."
            checked={valueOr('remove_outliers', false)}
            onToggle={(b) => setOne('remove_outliers', b)}
            disabled={disabled}
            visual={<OutlierVisual on={valueOr('remove_outliers', false)} />}
          />
          <PreprocCard
            label="Feature selection"
            description="Quick importance pass; drop low-signal features. Speeds up wide datasets."
            checked={valueOr('feature_selection', false)}
            onToggle={(b) => setOne('feature_selection', b)}
            disabled={disabled}
            visual={<FeatureSelVisual on={valueOr('feature_selection', false)} />}
          />
        </div>
      </SubSection>

      {/* ── CV / reproducibility / compute ──────────────────────── */}
      <SubSection
        title="Cross-validation & runtime"
        hint="How the engine validates each model and what to pin for reproducibility."
      >
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <Field label="Folds" hint={byName.get('fold')?.description}>
            <FoldStepper
              value={valueOr('fold', 10)}
              onChange={(n) => setOne('fold', n)}
              min={byName.get('fold')?.minimum ?? 2}
              max={Math.min(byName.get('fold')?.maximum ?? 20, 20)}
              disabled={disabled}
            />
          </Field>

          <Field label="Fold strategy" hint={byName.get('fold_strategy')?.description}>
            <Segmented
              options={(byName.get('fold_strategy')?.choices ?? []).map((c) => ({
                value: c,
                label: prettyFoldStrategy(c),
              }))}
              value={valueOr('fold_strategy', 'stratifiedkfold')}
              onChange={(v) => setOne('fold_strategy', v)}
              disabled={disabled}
            />
          </Field>

          <Field label="Random seed (optional)" hint={byName.get('session_id')?.description}>
            <SeedInput
              value={(values.session_id as number | null | undefined) ?? null}
              onChange={(v) => setOne('session_id', v)}
              disabled={disabled}
            />
          </Field>

          <Field label="Use GPU" hint={byName.get('use_gpu')?.description}>
            <ToggleRow
              checked={valueOr('use_gpu', false)}
              onChange={(b) => setOne('use_gpu', b)}
              disabled={disabled}
              labelOn="GPU enabled"
              labelOff="CPU only"
            />
          </Field>
        </div>
      </SubSection>

      {/* ── Future-proofing: any engine params we don't hand-render ── */}
      {others.length > 0 && (
        <SubSection
          title="Other options"
          hint="Schema-driven fallback — these came from the engine but don't have a dedicated widget yet."
        >
          <div className="space-y-2">
            {others.map((p) => (
              <GenericFallback
                key={p.name}
                param={p}
                value={values[p.name] ?? p.default}
                onChange={(v) => setOne(p.name, v)}
                disabled={disabled}
              />
            ))}
          </div>
        </SubSection>
      )}
    </div>
  );
}

// ────────────────────────────────────────────────── primitives

function SubSection({
  title,
  hint,
  children,
}: {
  title: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <section>
      <header className="mb-2">
        <h3 className="text-sm font-semibold text-ink-900 dark:text-ink-50">
          {title}
        </h3>
        {hint && <p className="text-xs text-ink-500 mt-0.5">{hint}</p>}
      </header>
      {children}
    </section>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 p-3">
      <label className="text-xs font-medium text-ink-700 dark:text-ink-300">
        {label}
      </label>
      <div className="mt-2">{children}</div>
      {hint && <p className="text-[11px] text-ink-500 mt-2">{hint}</p>}
    </div>
  );
}

// ────────────────────────────────────────────────── Train-size slider

function TrainSizeSlider({
  value,
  onChange,
  disabled,
  rowCount,
}: {
  value: number;
  onChange: (pct: number) => void;
  disabled?: boolean;
  rowCount?: number;
}) {
  const trainRows = rowCount ? Math.round((rowCount * value) / 100) : null;
  const testRows = rowCount ? rowCount - (trainRows ?? 0) : null;
  const presets = [60, 70, 75, 80, 85];

  return (
    <div className="rounded-lg border border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 p-4 space-y-3">
      {/* Big visual bar */}
      <div className="relative h-10 rounded-md overflow-hidden border border-ink-200 dark:border-ink-800">
        <div
          className="absolute inset-y-0 left-0 bg-accent-500/85 transition-[width] duration-150 ease-out"
          style={{ width: `${value}%` }}
        />
        <div
          className="absolute inset-y-0 right-0 bg-ink-300 dark:bg-ink-700 transition-[width] duration-150 ease-out"
          style={{ width: `${100 - value}%` }}
        />
        <div className="absolute inset-0 flex items-center justify-between px-3 text-xs font-semibold text-white">
          <span style={{ opacity: value > 18 ? 1 : 0 }}>
            Train {value}%
            {trainRows != null && (
              <span className="ml-1 opacity-80 font-normal">
                · {trainRows.toLocaleString()} rows
              </span>
            )}
          </span>
          <span
            className="text-ink-900 dark:text-ink-50"
            style={{ opacity: 100 - value > 12 ? 1 : 0 }}
          >
            {testRows != null && (
              <span className="mr-1 opacity-70 font-normal">
                {testRows.toLocaleString()} rows ·
              </span>
            )}
            Test {100 - value}%
          </span>
        </div>
      </div>

      {/* Native slider — 1% steps, 50-95 range */}
      <input
        type="range"
        min={50}
        max={95}
        step={1}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        disabled={disabled}
        className="w-full accent-accent-500"
      />

      {/* Preset chips */}
      <div className="flex items-center gap-1.5">
        {presets.map((p) => (
          <button
            key={p}
            type="button"
            onClick={() => onChange(p)}
            disabled={disabled}
            className={`text-xs px-2.5 py-1 rounded-full transition-colors ${
              value === p
                ? 'bg-ink-900 dark:bg-ink-50 text-white dark:text-ink-900'
                : 'bg-ink-100 dark:bg-ink-800 text-ink-700 dark:text-ink-300 hover:bg-ink-200 dark:hover:bg-ink-700'
            }`}
          >
            {p}/{100 - p}
          </button>
        ))}
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────── Preprocessing card

function PreprocCard({
  label,
  description,
  checked,
  onToggle,
  disabled,
  visual,
}: {
  label: string;
  description: string;
  checked: boolean;
  onToggle: (b: boolean) => void;
  disabled?: boolean;
  visual: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={() => !disabled && onToggle(!checked)}
      disabled={disabled}
      className={`text-left rounded-lg border p-3 transition-all ${
        checked
          ? 'border-accent-500 bg-accent-500/5 dark:bg-accent-500/10 shadow-soft-1'
          : 'border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 hover:border-ink-300 dark:hover:border-ink-700'
      } disabled:opacity-50 disabled:cursor-not-allowed`}
    >
      <div className="flex items-start justify-between gap-2 mb-2">
        <span className="text-sm font-semibold text-ink-900 dark:text-ink-50">
          {label}
        </span>
        <Switch checked={checked} onChange={() => {}} disabled={disabled} />
      </div>
      <div className="h-16 rounded bg-ink-50 dark:bg-ink-950/40 px-2 py-1 flex items-center justify-center">
        {visual}
      </div>
      <p className="text-[11px] text-ink-500 mt-2 leading-snug">{description}</p>
    </button>
  );
}

function Switch({
  checked,
  onChange,
  disabled,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <span
      role="switch"
      aria-checked={checked}
      onClick={(e) => {
        e.stopPropagation();
        if (!disabled) onChange(!checked);
      }}
      className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors shrink-0 cursor-pointer ${
        checked ? 'bg-accent-500' : 'bg-ink-300 dark:bg-ink-700'
      } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
    >
      <span
        className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${
          checked ? 'translate-x-4' : 'translate-x-0.5'
        }`}
      />
    </span>
  );
}

// ────────────────────────────────────────────────── Preprocessing visuals

function NormalizeVisual({ on }: { on: boolean }) {
  // Off: skewed; On: centered bell. Pure SVG so it scales crisp.
  const offBars = [4, 9, 16, 26, 38, 30, 22, 16, 10, 6];
  const onBars = [3, 8, 18, 32, 44, 44, 32, 18, 8, 3];
  const bars = on ? onBars : offBars;
  return (
    <svg width="100%" height="100%" viewBox="0 0 100 40" preserveAspectRatio="none">
      {bars.map((h, i) => (
        <rect
          key={i}
          x={i * 10 + 1}
          y={40 - h}
          width={8}
          height={h}
          rx={1}
          className={on ? 'fill-accent-500' : 'fill-ink-400 dark:fill-ink-600'}
        />
      ))}
    </svg>
  );
}

function TransformationVisual({ on }: { on: boolean }) {
  // Off: heavy right-skewed curve; On: symmetric bell.
  const pts = on
    ? 'M0,38 C20,38 30,8 50,8 C70,8 80,38 100,38'
    : 'M0,38 C5,38 8,8 18,8 C40,8 60,38 100,38';
  return (
    <svg width="100%" height="100%" viewBox="0 0 100 40" preserveAspectRatio="none">
      <path
        d={pts}
        fill="none"
        strokeWidth={2}
        strokeLinecap="round"
        className={on ? 'stroke-accent-500' : 'stroke-ink-400 dark:stroke-ink-600'}
      />
    </svg>
  );
}

function OutlierVisual({ on }: { on: boolean }) {
  // Cluster + a few stragglers; when on, stragglers fade.
  const cluster = [
    [30, 22],
    [38, 28],
    [42, 20],
    [48, 26],
    [55, 22],
    [60, 28],
    [50, 30],
    [44, 18],
  ];
  const outliers = [
    [12, 8],
    [88, 34],
    [90, 6],
  ];
  return (
    <svg width="100%" height="100%" viewBox="0 0 100 40">
      {cluster.map(([x, y], i) => (
        <circle
          key={i}
          cx={x}
          cy={y}
          r={2.2}
          className="fill-accent-500"
        />
      ))}
      {outliers.map(([x, y], i) => (
        <circle
          key={`o${i}`}
          cx={x}
          cy={y}
          r={2.2}
          className={
            on
              ? 'fill-rose-500/20 stroke-rose-500/40 stroke-1'
              : 'fill-rose-500'
          }
          strokeDasharray={on ? '1.5,1.5' : undefined}
        />
      ))}
    </svg>
  );
}

function FeatureSelVisual({ on }: { on: boolean }) {
  // Sorted feature importance bars; when on, low bars fade.
  const heights = [34, 28, 22, 18, 14, 10, 7, 5, 4, 3];
  return (
    <svg width="100%" height="100%" viewBox="0 0 100 40" preserveAspectRatio="none">
      {heights.map((h, i) => {
        const dim = on && i >= 6;
        return (
          <rect
            key={i}
            x={i * 10 + 2}
            y={40 - h}
            width={7}
            height={h}
            rx={1}
            className={
              dim
                ? 'fill-ink-300 dark:fill-ink-700'
                : on
                  ? 'fill-accent-500'
                  : 'fill-ink-400 dark:fill-ink-600'
            }
          />
        );
      })}
    </svg>
  );
}

// ────────────────────────────────────────────────── CV widgets

function FoldStepper({
  value,
  onChange,
  min,
  max,
  disabled,
}: {
  value: number;
  onChange: (n: number) => void;
  min: number;
  max: number;
  disabled?: boolean;
}) {
  const dec = () => onChange(Math.max(min, value - 1));
  const inc = () => onChange(Math.min(max, value + 1));
  return (
    <div className="inline-flex items-center rounded-md border border-ink-200 dark:border-ink-800 overflow-hidden">
      <button
        type="button"
        onClick={dec}
        disabled={disabled || value <= min}
        className="px-3 py-1.5 text-sm hover:bg-ink-100 dark:hover:bg-ink-800 disabled:opacity-30"
      >
        −
      </button>
      <span className="px-3 py-1.5 text-sm font-semibold tabular-nums min-w-[2.5rem] text-center border-x border-ink-200 dark:border-ink-800">
        {value}
      </span>
      <button
        type="button"
        onClick={inc}
        disabled={disabled || value >= max}
        className="px-3 py-1.5 text-sm hover:bg-ink-100 dark:hover:bg-ink-800 disabled:opacity-30"
      >
        +
      </button>
    </div>
  );
}

function Segmented<T extends string>({
  options,
  value,
  onChange,
  disabled,
}: {
  options: Array<{ value: T; label: string }>;
  value: T;
  onChange: (v: T) => void;
  disabled?: boolean;
}) {
  return (
    <div className="inline-flex rounded-md bg-ink-100 dark:bg-ink-800 p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          onClick={() => onChange(o.value)}
          disabled={disabled}
          className={`text-xs px-2.5 py-1 rounded transition-colors ${
            value === o.value
              ? 'bg-white dark:bg-ink-900 text-ink-900 dark:text-ink-50 shadow-soft-1'
              : 'text-ink-600 dark:text-ink-400 hover:text-ink-900 dark:hover:text-ink-50'
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

function prettyFoldStrategy(s: string): string {
  switch (s) {
    case 'kfold':
      return 'K-fold';
    case 'stratifiedkfold':
      return 'Stratified';
    case 'groupkfold':
      return 'Group';
    default:
      return s;
  }
}

function SeedInput({
  value,
  onChange,
  disabled,
}: {
  value: number | null;
  onChange: (v: number | null) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex items-center gap-2">
      <input
        type="number"
        value={value ?? ''}
        onChange={(e) => {
          const v = e.target.value.trim();
          onChange(v === '' ? null : Number(v));
        }}
        placeholder="random"
        disabled={disabled}
        className="input text-sm w-32"
      />
      {value != null && (
        <button
          type="button"
          onClick={() => onChange(null)}
          disabled={disabled}
          className="text-xs text-ink-500 hover:text-ink-900 dark:hover:text-ink-50"
        >
          clear
        </button>
      )}
    </div>
  );
}

function ToggleRow({
  checked,
  onChange,
  disabled,
  labelOn,
  labelOff,
}: {
  checked: boolean;
  onChange: (b: boolean) => void;
  disabled?: boolean;
  labelOn: string;
  labelOff: string;
}) {
  return (
    <div className="flex items-center gap-2">
      <Switch checked={checked} onChange={onChange} disabled={disabled} />
      <span className="text-xs text-ink-700 dark:text-ink-300">
        {checked ? labelOn : labelOff}
      </span>
    </div>
  );
}

// ────────────────────────────────────────────────── Generic fallback

function GenericFallback({
  param,
  value,
  onChange,
  disabled,
}: {
  param: SetupParam;
  value: unknown;
  onChange: (v: unknown) => void;
  disabled?: boolean;
}) {
  // Minimal renderer — only types we expect to encounter for future engine
  // additions. For richer rendering reach for DynamicForm.
  if (param.kind === 'bool') {
    return (
      <div className="flex items-center justify-between gap-3 px-3 py-2 rounded border border-ink-200 dark:border-ink-800">
        <div className="min-w-0">
          <p className="text-xs font-medium text-ink-900 dark:text-ink-50">
            {param.name}
          </p>
          {param.description && (
            <p className="text-[11px] text-ink-500 truncate">{param.description}</p>
          )}
        </div>
        <Switch
          checked={Boolean(value)}
          onChange={(b) => onChange(b)}
          disabled={disabled}
        />
      </div>
    );
  }
  return (
    <div className="flex items-center justify-between gap-3 px-3 py-2 rounded border border-ink-200 dark:border-ink-800">
      <div className="min-w-0">
        <p className="text-xs font-medium text-ink-900 dark:text-ink-50">
          {param.name}
        </p>
        {param.description && (
          <p className="text-[11px] text-ink-500 truncate">{param.description}</p>
        )}
      </div>
      <input
        value={String(value ?? '')}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className="input text-xs w-32"
      />
    </div>
  );
}
