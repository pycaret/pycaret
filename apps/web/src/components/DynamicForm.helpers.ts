/**
 * Pure helpers for DynamicForm. Kept in a separate file from the React
 * components so ESLint's react-refresh plugin is happy (fast-refresh only
 * works when a module exports components *only*).
 */

import type { SetupParamSchema } from '@/api/types';

/** Values keyed by parameter name. Each value type depends on that param's `kind`. */
export type ParamValues = Record<string, unknown>;

/** Merge schema defaults into a values object, preserving user overrides. */
export function applyDefaults(
  schema: SetupParamSchema,
  current: ParamValues,
): ParamValues {
  const next: ParamValues = { ...current };
  for (const p of schema.parameters) {
    if (!(p.name in next) && p.default !== null && p.default !== undefined) {
      next[p.name] = p.default;
    }
  }
  return next;
}

/**
 * Strip values equal to their schema default (or empty). The API payload
 * should capture *user intent* only — defaults are the engine's concern.
 */
export function stripDefaults(
  schema: SetupParamSchema,
  values: ParamValues,
): ParamValues {
  const out: ParamValues = {};
  for (const p of schema.parameters) {
    const v = values[p.name];
    if (v === undefined || v === null || v === '') continue;
    if (v === p.default) continue;
    out[p.name] = v;
  }
  return out;
}
