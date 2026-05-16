/**
 * Trial-action dialogs: Tune / Ensemble / Blend / Stack.
 *
 * One file, four dialogs — they share the same plumbing (open/close,
 * mutation, success toast, query invalidation), and only the body
 * (which engine knobs to expose) varies.
 *
 * On submit:
 *   - POST to the matching endpoint
 *   - Backend returns 202 immediately; worker emits events into the run
 *     event stream and writes a new Trial row when finished.
 *   - We invalidate the trials list query so the UI catches up via the
 *     run's existing polling/refresh.
 *   - We also fire onSubmitted so the parent can open the event drawer.
 */

import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Dialog } from './Dialog';
import { runsApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';

// ─── Tune ────────────────────────────────────────────────────────

const TUNE_OPTIMIZE_CHOICES: Record<string, string[]> = {
  classification: ['Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa', 'MCC'],
  regression: ['R2', 'MAE', 'MSE', 'RMSE', 'RMSLE', 'MAPE'],
};

export function TuneDialog({
  open,
  onClose,
  onSubmitted,
  runId,
  trialId,
  modelLabel,
  task,
}: {
  open: boolean;
  onClose: () => void;
  onSubmitted?: () => void;
  runId: string;
  trialId: string;
  modelLabel: string;
  task: string | null | undefined;
}) {
  const qc = useQueryClient();
  const [nIter, setNIter] = useState(10);
  const [optimize, setOptimize] = useState<string>('');

  // Suggest a sensible default optimize metric on open.
  useEffect(() => {
    if (!open) return;
    if (task === 'classification') setOptimize('Accuracy');
    else if (task === 'regression') setOptimize('R2');
  }, [open, task]);

  const mut = useMutation({
    mutationFn: () =>
      runsApi.trialTune(runId, trialId, {
        n_iter: nIter,
        ...(optimize ? { optimize } : {}),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['runs', runId, 'trials'] });
      onClose();
      onSubmitted?.();
    },
  });

  const metricChoices =
    task && TUNE_OPTIMIZE_CHOICES[task] ? TUNE_OPTIMIZE_CHOICES[task] : [];

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title={`Tune ${modelLabel}`}
      description="Random-search the estimator's hyperparameter space, keeping the run's CV settings. The tuned candidate lands as a new row in this run's trials table."
    >
      <form
        className="space-y-4"
        onSubmit={(e) => {
          e.preventDefault();
          mut.mutate();
        }}
      >
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs text-ink-500 mb-1">
              Search iterations
            </label>
            <input
              type="number"
              min={1}
              max={500}
              className="input"
              value={nIter}
              onChange={(e) => setNIter(Math.max(1, Number(e.target.value) || 1))}
            />
            <p className="text-[11px] text-ink-500 mt-1">
              More iterations = better hyperparams, slower run.
            </p>
          </div>
          <div>
            <label className="block text-xs text-ink-500 mb-1">
              Optimize metric
            </label>
            <select
              className="input"
              value={optimize}
              onChange={(e) => setOptimize(e.target.value)}
            >
              <option value="">— engine default —</option>
              {metricChoices.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
            <p className="text-[11px] text-ink-500 mt-1">
              Drives the CV scoring during search.
            </p>
          </div>
        </div>
        {mut.error && (
          <p className="text-xs text-danger-600">{errorMessage(mut.error)}</p>
        )}
        <DialogFooter
          onClose={onClose}
          submitting={mut.isPending}
          submitLabel="Start tuning"
        />
      </form>
    </Dialog>
  );
}

// ─── Ensemble (Bagging / Boosting on a single trial) ─────────────

export function EnsembleDialog({
  open,
  onClose,
  onSubmitted,
  runId,
  trialId,
  modelLabel,
}: {
  open: boolean;
  onClose: () => void;
  onSubmitted?: () => void;
  runId: string;
  trialId: string;
  modelLabel: string;
}) {
  const qc = useQueryClient();
  const [method, setMethod] = useState<'Bagging' | 'Boosting'>('Bagging');
  const [n, setN] = useState(10);
  const mut = useMutation({
    mutationFn: () =>
      runsApi.trialEnsemble(runId, trialId, { method, n_estimators: n }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['runs', runId, 'trials'] });
      onClose();
      onSubmitted?.();
    },
  });

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title={`Ensemble ${modelLabel}`}
      description="Wrap the estimator in Bagging or Boosting. Result lands as a new trial in this run."
    >
      <form
        className="space-y-4"
        onSubmit={(e) => {
          e.preventDefault();
          mut.mutate();
        }}
      >
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs text-ink-500 mb-1">Method</label>
            <div className="inline-flex w-full rounded-md border border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 p-0.5">
              {(['Bagging', 'Boosting'] as const).map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setMethod(m)}
                  className={`flex-1 px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                    method === m
                      ? 'bg-ink-900 text-white dark:bg-white dark:text-ink-900'
                      : 'text-ink-600 hover:text-ink-900 dark:text-ink-400 dark:hover:text-ink-50'
                  }`}
                >
                  {m}
                </button>
              ))}
            </div>
            <p className="text-[11px] text-ink-500 mt-1">
              Bagging reduces variance, Boosting reduces bias.
            </p>
          </div>
          <div>
            <label className="block text-xs text-ink-500 mb-1">
              n_estimators
            </label>
            <input
              type="number"
              min={2}
              max={200}
              className="input"
              value={n}
              onChange={(e) => setN(Math.max(2, Number(e.target.value) || 2))}
            />
            <p className="text-[11px] text-ink-500 mt-1">
              Number of base estimators in the ensemble.
            </p>
          </div>
        </div>
        {mut.error && (
          <p className="text-xs text-danger-600">{errorMessage(mut.error)}</p>
        )}
        <DialogFooter
          onClose={onClose}
          submitting={mut.isPending}
          submitLabel={`Start ${method.toLowerCase()}`}
        />
      </form>
    </Dialog>
  );
}

// ─── Shared source picker ───────────────────────────────────────
//
// Blend / Stack now own their source selection inside the dialog. This
// keeps the UX discoverable (explicit buttons on the run page) and lets
// callers either start from a pre-checked selection (row checkboxes) or
// a blank slate.

interface PickableTrial {
  id: string;
  model_id: string;
  metrics: Record<string, number | string>;
  is_best: boolean;
  has_artifact?: boolean;
}

function SourcePicker({
  trials,
  selected,
  onToggle,
  primaryMetric,
  nameOf,
}: {
  trials: PickableTrial[];
  selected: Set<string>;
  onToggle: (id: string) => void;
  primaryMetric: string | null;
  nameOf: (id: string) => string;
}) {
  if (trials.length === 0) {
    return (
      <p className="text-sm text-ink-500 px-2 py-6 text-center">
        No trials with stored pipelines yet — can't blend / stack without
        the base models on disk.
      </p>
    );
  }
  return (
    <div className="rounded-md border border-ink-200 dark:border-ink-800 overflow-hidden">
      <div className="px-3 py-2 text-[11px] uppercase tracking-wide text-ink-500 border-b border-ink-200 dark:border-ink-800 bg-ink-50/60 dark:bg-ink-950/40 flex items-center justify-between">
        <span>Pick trials to combine (at least 2)</span>
        <span className="tabular-nums">{selected.size} selected</span>
      </div>
      <ul className="max-h-72 overflow-y-auto divide-y divide-ink-200 dark:divide-ink-800">
        {trials.map((t) => {
          const disabled = t.has_artifact === false;
          const checked = selected.has(t.id);
          const primary =
            primaryMetric && typeof t.metrics[primaryMetric] === 'number'
              ? (t.metrics[primaryMetric] as number).toFixed(4)
              : null;
          return (
            <li key={t.id}>
              <label
                className={`flex items-center gap-3 px-3 py-2 text-sm cursor-pointer ${
                  checked
                    ? 'bg-accent-50/60 dark:bg-accent-500/10'
                    : 'hover:bg-ink-50 dark:hover:bg-ink-950/40'
                } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
                title={
                  disabled
                    ? 'No stored pipeline — cannot use as a source'
                    : undefined
                }
              >
                <input
                  type="checkbox"
                  className="accent-accent-500"
                  checked={checked}
                  disabled={disabled}
                  onChange={() => onToggle(t.id)}
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1.5">
                    <span className="font-medium text-ink-900 dark:text-ink-50 truncate">
                      {nameOf(t.model_id)}
                    </span>
                    {t.is_best && (
                      <span className="text-warn-500" title="Best">
                        ★
                      </span>
                    )}
                  </div>
                  <div className="text-[11px] text-ink-400 font-mono truncate">
                    {t.model_id}
                  </div>
                </div>
                {primary && (
                  <span className="font-mono text-xs tabular-nums text-ink-700 dark:text-ink-300">
                    {primaryMetric} {primary}
                  </span>
                )}
              </label>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function useToggleSet(initial: string[] | undefined, open: boolean) {
  const [set, setSet] = useState<Set<string>>(new Set(initial ?? []));
  // Reset selection each time the dialog opens so it doesn't carry stale
  // state from a prior invocation.
  useEffect(() => {
    if (open) setSet(new Set(initial ?? []));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);
  const toggle = (id: string) =>
    setSet((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  return [set, toggle] as const;
}

// ─── Blend (voting across N trials) ─────────────────────────────

export function BlendDialog({
  open,
  onClose,
  onSubmitted,
  runId,
  availableTrials,
  initialSelectedIds,
  nameOf,
  primaryMetric,
}: {
  open: boolean;
  onClose: () => void;
  onSubmitted?: () => void;
  runId: string;
  availableTrials: PickableTrial[];
  initialSelectedIds?: string[];
  nameOf: (id: string) => string;
  primaryMetric: string | null;
}) {
  const qc = useQueryClient();
  const [selected, toggleSelected] = useToggleSet(initialSelectedIds, open);
  const [method, setMethod] = useState<'auto' | 'hard' | 'soft'>('auto');
  const sourceIds = useMemo(() => Array.from(selected), [selected]);

  const mut = useMutation({
    mutationFn: () =>
      runsApi.runBlend(runId, {
        source_trial_ids: sourceIds,
        method,
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['runs', runId, 'trials'] });
      onClose();
      onSubmitted?.();
    },
  });

  const canSubmit = sourceIds.length >= 2 && !mut.isPending;

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="Blend models"
      description="Voting ensemble across selected base models. Hard votes pick the majority class; soft averages predicted probabilities (classifier must support predict_proba)."
      size="lg"
    >
      <form
        className="space-y-4"
        onSubmit={(e) => {
          e.preventDefault();
          if (canSubmit) mut.mutate();
        }}
      >
        <SourcePicker
          trials={availableTrials}
          selected={selected}
          onToggle={toggleSelected}
          primaryMetric={primaryMetric}
          nameOf={nameOf}
        />
        <div>
          <label className="block text-xs text-ink-500 mb-1">Voting method</label>
          <div className="inline-flex w-full rounded-md border border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 p-0.5">
            {(['auto', 'hard', 'soft'] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMethod(m)}
                className={`flex-1 px-3 py-1.5 rounded text-xs font-medium capitalize transition-colors ${
                  method === m
                    ? 'bg-ink-900 text-white dark:bg-white dark:text-ink-900'
                    : 'text-ink-600 hover:text-ink-900 dark:text-ink-400 dark:hover:text-ink-50'
                }`}
              >
                {m}
              </button>
            ))}
          </div>
          <p className="text-[11px] text-ink-500 mt-1">
            `auto` picks soft if every base supports probabilities, else hard.
          </p>
        </div>
        {mut.error && (
          <p className="text-xs text-danger-600">{errorMessage(mut.error)}</p>
        )}
        <DialogFooter
          onClose={onClose}
          submitting={mut.isPending}
          submitLabel="Start blending"
          disabled={!canSubmit}
        />
      </form>
    </Dialog>
  );
}

// ─── Stack (stacking with meta-learner) ─────────────────────────

export function StackDialog({
  open,
  onClose,
  onSubmitted,
  runId,
  availableTrials,
  initialSelectedIds,
  nameOf,
  primaryMetric,
  task,
}: {
  open: boolean;
  onClose: () => void;
  onSubmitted?: () => void;
  runId: string;
  availableTrials: PickableTrial[];
  initialSelectedIds?: string[];
  nameOf: (id: string) => string;
  primaryMetric: string | null;
  task: string | null | undefined;
}) {
  const qc = useQueryClient();
  const [selected, toggleSelected] = useToggleSet(initialSelectedIds, open);
  const [meta, setMeta] = useState<string>('');
  const sourceIds = useMemo(() => Array.from(selected), [selected]);

  const mut = useMutation({
    mutationFn: () =>
      runsApi.runStack(runId, {
        source_trial_ids: sourceIds,
        ...(meta ? { meta_model: meta } : {}),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['runs', runId, 'trials'] });
      onClose();
      onSubmitted?.();
    },
  });

  const metaChoices =
    task === 'classification'
      ? ['lr', 'rf', 'lightgbm', 'xgboost']
      : task === 'regression'
        ? ['lr', 'rf', 'lightgbm', 'xgboost']
        : [];

  const canSubmit = sourceIds.length >= 2 && !mut.isPending;

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="Stack models"
      description="Train base models on K-1 folds, feed their predictions to a meta-learner that produces the final output. Often beats blending when bases disagree."
      size="lg"
    >
      <form
        className="space-y-4"
        onSubmit={(e) => {
          e.preventDefault();
          if (canSubmit) mut.mutate();
        }}
      >
        <SourcePicker
          trials={availableTrials}
          selected={selected}
          onToggle={toggleSelected}
          primaryMetric={primaryMetric}
          nameOf={nameOf}
        />
        <div>
          <label className="block text-xs text-ink-500 mb-1">Meta-learner</label>
          <select
            className="input"
            value={meta}
            onChange={(e) => setMeta(e.target.value)}
          >
            <option value="">— engine default (logistic / linear) —</option>
            {metaChoices.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
          <p className="text-[11px] text-ink-500 mt-1">
            Linear models usually meta-learn best; tree-based metas can
            overfit on small leaderboards.
          </p>
        </div>
        {mut.error && (
          <p className="text-xs text-danger-600">{errorMessage(mut.error)}</p>
        )}
        <DialogFooter
          onClose={onClose}
          submitting={mut.isPending}
          submitLabel="Start stacking"
          disabled={!canSubmit}
        />
      </form>
    </Dialog>
  );
}

// ─── Shared bits ────────────────────────────────────────────────

function DialogFooter({
  onClose,
  submitting,
  submitLabel,
  disabled,
}: {
  onClose: () => void;
  submitting: boolean;
  submitLabel: string;
  disabled?: boolean;
}) {
  return (
    <div className="flex items-center justify-end gap-2 pt-2">
      <button
        type="button"
        className="btn-secondary"
        onClick={onClose}
        data-dialog-close
      >
        Cancel
      </button>
      <button
        type="submit"
        className="btn-primary"
        disabled={submitting || disabled}
      >
        {submitting ? 'Submitting…' : submitLabel}
      </button>
    </div>
  );
}
