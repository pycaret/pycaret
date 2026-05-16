/**
 * Dynamic setup-parameters form, 100% driven by the engine's
 * `describe_setup_params(task)` schema.
 *
 * This file is the load-bearing contract between the engine and the UI:
 * it must **never** hard-code a parameter name. The engine can add, remove,
 * or rename any parameter, and this form just works. The only kinds we
 * know about are structural (bool / int / float / enum / column / string);
 * everything else lives in the schema.
 *
 * Call sites:
 *   - Create-experiment wizard (preview config before submit).
 *   - Future: experiment-edit screen, LLM-generated config preview, etc.
 */

import { useId, useMemo } from 'react';
import type { SetupParam, SetupParamSchema } from '@/api/types';
import type { ParamValues } from './DynamicForm.helpers';

// ────────────────────────────────────────────────────────────────── types

export interface DynamicFormProps {
  schema: SetupParamSchema;
  /** Columns available in the user's dataset (drives `kind: 'column'` dropdowns). */
  columns?: string[];
  values: ParamValues;
  onChange: (next: ParamValues) => void;
  /**
   * Parameter names the caller wants to hide entirely. Useful when some
   * params (e.g. `target`) are already collected elsewhere in the wizard
   * and shouldn't be re-asked here.
   */
  hide?: string[];
  disabled?: boolean;
}

// ───────────────────────────────────────────────────────── ParamInput

interface ParamInputProps {
  param: SetupParam;
  value: unknown;
  onChange: (value: unknown) => void;
  columns?: string[];
  disabled?: boolean;
}

/**
 * Renders the correct HTML input for a single `SetupParam`. The switch on
 * `kind` is the extension point: adding a new kind means adding a case
 * here, nothing else.
 */
export function ParamInput({
  param,
  value,
  onChange,
  columns,
  disabled,
}: ParamInputProps) {
  const id = useId();

  switch (param.kind) {
    case 'bool': {
      const checked = Boolean(value);
      return (
        <button
          id={id}
          type="button"
          role="switch"
          aria-checked={checked}
          onClick={() => onChange(!checked)}
          disabled={disabled}
          className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors flex-shrink-0 ${
            checked
              ? 'bg-accent-500'
              : 'bg-ink-200 dark:bg-ink-700'
          } disabled:opacity-50`}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${
              checked ? 'translate-x-4' : 'translate-x-0.5'
            }`}
          />
        </button>
      );
    }

    case 'int':
    case 'float': {
      const num = value === '' || value === null || value === undefined ? '' : String(value);
      const step = param.kind === 'int' ? 1 : 0.01;
      return (
        <input
          id={id}
          type="number"
          className="input"
          value={num}
          min={param.minimum ?? undefined}
          max={param.maximum ?? undefined}
          step={step}
          onChange={(e) => {
            const raw = e.target.value;
            if (raw === '') {
              onChange(null);
              return;
            }
            const n = param.kind === 'int' ? parseInt(raw, 10) : parseFloat(raw);
            onChange(Number.isNaN(n) ? raw : n);
          }}
          disabled={disabled}
          required={param.required}
        />
      );
    }

    case 'enum': {
      const sel = value == null ? '' : String(value);
      return (
        <select
          id={id}
          className="input"
          value={sel}
          onChange={(e) => onChange(e.target.value || null)}
          disabled={disabled}
          required={param.required}
        >
          {!param.required && <option value="">— none —</option>}
          {(param.choices ?? []).map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      );
    }

    case 'column': {
      // When we don't know the columns yet (no dataset selected), fall back
      // to a free-text input. The API will reject unknown columns at run time.
      const sel = value == null ? '' : String(value);
      if (!columns || columns.length === 0) {
        return (
          <input
            id={id}
            type="text"
            className="input"
            placeholder="column name"
            value={sel}
            onChange={(e) => onChange(e.target.value || null)}
            disabled={disabled}
            required={param.required}
          />
        );
      }
      return (
        <select
          id={id}
          className="input"
          value={sel}
          onChange={(e) => onChange(e.target.value || null)}
          disabled={disabled}
          required={param.required}
        >
          {!param.required && <option value="">— none —</option>}
          {columns.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      );
    }

    case 'string':
    default: {
      const sel = value == null ? '' : String(value);
      return (
        <input
          id={id}
          type="text"
          className="input"
          value={sel}
          onChange={(e) => onChange(e.target.value || null)}
          disabled={disabled}
          required={param.required}
        />
      );
    }
  }
}

// ───────────────────────────────────────────────────────── DynamicForm

/**
 * Renders an entire setup-params schema grouped by the `group` field.
 * Groups appear in the order declared by `schema.groups` (server-driven).
 */
export function DynamicForm({
  schema,
  columns,
  values,
  onChange,
  hide,
  disabled,
}: DynamicFormProps) {
  const hidden = useMemo(() => new Set(hide ?? []), [hide]);

  // Group params, preserving schema.groups order. Params in unlisted groups
  // collapse into a final "Other" bucket.
  const grouped = useMemo(() => {
    const byGroup = new Map<string, SetupParam[]>();
    for (const g of schema.groups) byGroup.set(g, []);
    for (const p of schema.parameters) {
      if (hidden.has(p.name)) continue;
      const bucket = byGroup.get(p.group) ?? (byGroup.set(p.group, []).get(p.group) as SetupParam[]);
      bucket.push(p);
    }
    return [...byGroup.entries()].filter(([, ps]) => ps.length > 0);
  }, [schema, hidden]);

  const setOne = (name: string, v: unknown) => {
    onChange({ ...values, [name]: v });
  };

  return (
    <div className="space-y-4">
      {grouped.map(([group, params]) => {
        // Split bools (rendered as toggle rows) from typed inputs (laid
        // out in a 2-column grid). Booleans are visually heavier when
        // mixed with text inputs, so giving them their own block reads
        // much cleaner.
        const bools = params.filter((p) => p.kind === 'bool');
        const others = params.filter((p) => p.kind !== 'bool');
        return (
          <fieldset
            key={group}
            className="rounded-xl border border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 overflow-hidden"
          >
            <legend className="px-4 pt-4 text-xs uppercase tracking-wider text-ink-500 font-semibold">
              {group}
            </legend>
            {others.length > 0 && (
              <div className="px-4 pt-3 pb-4 grid gap-4 sm:grid-cols-2">
                {others.map((p) => (
                  <div key={p.name} className="min-w-0">
                    <label
                      className="flex items-baseline gap-1.5 text-xs font-medium text-ink-700 dark:text-ink-300 mb-1"
                      htmlFor={p.name}
                    >
                      <span>{p.name}</span>
                      {p.required && (
                        <span className="text-danger-500" aria-label="required">
                          *
                        </span>
                      )}
                    </label>
                    <ParamInput
                      param={p}
                      value={values[p.name] ?? p.default ?? null}
                      onChange={(v) => setOne(p.name, v)}
                      columns={columns}
                      disabled={disabled}
                    />
                    {p.description && (
                      <p className="mt-1 text-[11px] text-ink-500">
                        {p.description}
                      </p>
                    )}
                    {(p.minimum !== null || p.maximum !== null) &&
                      (p.kind === 'int' || p.kind === 'float') && (
                        <p className="mt-0.5 text-[11px] text-ink-400 font-mono">
                          {p.minimum ?? '−∞'} … {p.maximum ?? '∞'}
                        </p>
                      )}
                  </div>
                ))}
              </div>
            )}
            {bools.length > 0 && (
              <ul className="border-t border-ink-200 dark:border-ink-800 divide-y divide-ink-100 dark:divide-ink-800/60">
                {bools.map((p) => (
                  <li
                    key={p.name}
                    className="px-4 py-3 flex items-start gap-4"
                  >
                    <div className="flex-1 min-w-0">
                      <label
                        htmlFor={p.name}
                        className="text-sm font-medium text-ink-900 dark:text-ink-50 cursor-pointer"
                      >
                        {p.name}
                      </label>
                      {p.description && (
                        <p className="mt-0.5 text-[11px] text-ink-500">
                          {p.description}
                        </p>
                      )}
                    </div>
                    <ParamInput
                      param={p}
                      value={values[p.name] ?? p.default ?? null}
                      onChange={(v) => setOne(p.name, v)}
                      columns={columns}
                      disabled={disabled}
                    />
                  </li>
                ))}
              </ul>
            )}
          </fieldset>
        );
      })}
    </div>
  );
}
