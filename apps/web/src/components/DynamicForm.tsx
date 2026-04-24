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
        <label htmlFor={id} className="flex items-center gap-2 cursor-pointer select-none">
          <input
            id={id}
            type="checkbox"
            className="h-4 w-4 rounded border-ink-700 bg-ink-900 text-accent-500 focus:ring-accent-400"
            checked={checked}
            onChange={(e) => onChange(e.target.checked)}
            disabled={disabled}
          />
          <span className="text-sm text-ink-100">{param.name}</span>
        </label>
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
    <div className="space-y-8">
      {grouped.map(([group, params]) => (
        <fieldset key={group} className="space-y-4">
          <legend className="text-xs uppercase tracking-wider text-ink-200/60 mb-2">
            {group}
          </legend>
          {params.map((p) => (
            <div key={p.name}>
              {p.kind !== 'bool' && (
                <label className="field" htmlFor={p.name}>
                  {p.name}
                  {p.required && <span className="ml-1 text-danger-500">*</span>}
                </label>
              )}
              <ParamInput
                param={p}
                value={values[p.name] ?? p.default ?? null}
                onChange={(v) => setOne(p.name, v)}
                columns={columns}
                disabled={disabled}
              />
              {p.description && <p className="hint mt-1">{p.description}</p>}
              {(p.minimum !== null || p.maximum !== null) &&
                (p.kind === 'int' || p.kind === 'float') && (
                  <p className="hint mt-0.5 opacity-60">
                    Range: {p.minimum ?? '−∞'} … {p.maximum ?? '∞'}
                  </p>
                )}
            </div>
          ))}
        </fieldset>
      ))}
    </div>
  );
}
