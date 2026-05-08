/**
 * /workspaces/:wsId/templates — saved experiment configs.
 *
 * Each row is a (task, setup_params, plan_params) bundle that the
 * NewExperiment screen can pre-populate from. Admin-only writes; viewers
 * can read.
 */

import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useParams } from 'react-router-dom';
import { templatesApi, type ExperimentTemplate } from '@/api/endpoints';
import { errorMessage } from '@/api/client';

const TASKS = ['classification', 'regression', 'clustering', 'anomaly', 'time_series'];

export function ExperimentTemplates() {
  const { wsId } = useParams<{ wsId: string }>();
  const workspaceId = wsId ?? '';
  const qc = useQueryClient();

  const list = useQuery({
    queryKey: ['workspaces', workspaceId, 'templates'],
    queryFn: () => templatesApi.list(workspaceId),
    enabled: !!workspaceId,
  });

  const remove = useMutation({
    mutationFn: (id: string) => templatesApi.remove(id),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ['workspaces', workspaceId, 'templates'] }),
  });

  return (
    <div className="space-y-8">
      <header>
        <h1 className="h-page">Experiment templates</h1>
        <p className="mt-2 text-sm text-ink-500">
          Saved (task, setup, plan) bundles. Pickable on the New Experiment
          screen as a starting point.
        </p>
      </header>

      <NewTemplateForm workspaceId={workspaceId} />

      <section>
        <h2 className="h-section mb-4">Saved templates</h2>
        {list.isPending ? (
          <div className="card text-sm text-ink-500">Loading…</div>
        ) : !list.data || list.data.items.length === 0 ? (
          <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 p-8 text-center text-sm text-ink-500">
            No templates yet — create one above to share a known-good
            <code className="font-mono mx-1">setup_params</code> bundle with
            your workspace.
          </div>
        ) : (
          <div className="card overflow-hidden p-0">
            <table className="w-full text-sm">
              <thead className="bg-white text-ink-500 dark:bg-ink-900">
                <tr>
                  <th className="px-4 py-2 text-left font-medium">Name</th>
                  <th className="px-4 py-2 text-left font-medium">Task</th>
                  <th className="px-4 py-2 text-left font-medium">Description</th>
                  <th className="px-4 py-2 text-left font-medium">Setup keys</th>
                  <th className="px-4 py-2 text-right font-medium">Action</th>
                </tr>
              </thead>
              <tbody>
                {list.data.items.map((t: ExperimentTemplate) => (
                  <tr
                    key={t.id}
                    className="border-t border-ink-200 dark:border-ink-800 hover:bg-ink-50 dark:hover:bg-ink-950/40"
                  >
                    <td className="px-4 py-2 font-medium text-ink-900 dark:text-ink-50">
                      {t.name}
                    </td>
                    <td className="px-4 py-2">
                      <span className="pill-neutral">{t.task}</span>
                    </td>
                    <td className="px-4 py-2 text-ink-700 dark:text-ink-300 max-w-xs truncate">
                      {t.description ?? <span className="text-ink-400">—</span>}
                    </td>
                    <td className="px-4 py-2 text-xs font-mono text-ink-500">
                      {Object.keys(t.setup_params).join(', ') || '(empty)'}
                    </td>
                    <td className="px-4 py-2 text-right text-xs">
                      <button
                        className="text-danger-600 hover:underline"
                        onClick={() => {
                          if (confirm(`Delete template "${t.name}"?`)) {
                            remove.mutate(t.id);
                          }
                        }}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

function NewTemplateForm({ workspaceId }: { workspaceId: string }) {
  const qc = useQueryClient();
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [task, setTask] = useState(TASKS[0]);
  const [setupParamsText, setSetupParamsText] = useState(
    '{\n  "session_id": 42,\n  "fold": 5,\n  "verbose": false\n}',
  );

  const create = useMutation({
    mutationFn: () =>
      templatesApi.create(workspaceId, {
        name,
        description,
        task,
        setup_params: JSON.parse(setupParamsText),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['workspaces', workspaceId, 'templates'] });
      setName('');
      setDescription('');
    },
  });

  let parseError: string | null = null;
  try {
    JSON.parse(setupParamsText);
  } catch (e) {
    parseError = (e as Error).message;
  }

  return (
    <section className="card">
      <h2 className="text-sm font-medium text-ink-900 mb-3">New template</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div>
          <label className="field" htmlFor="t-name">Name</label>
          <input
            id="t-name"
            className="input"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="quick-classification"
          />
        </div>
        <div>
          <label className="field" htmlFor="t-task">Task</label>
          <select
            id="t-task"
            className="input"
            value={task}
            onChange={(e) => setTask(e.target.value)}
          >
            {TASKS.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>
        <div className="md:col-span-2">
          <label className="field" htmlFor="t-desc">Description (optional)</label>
          <input
            id="t-desc"
            className="input"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
        </div>
        <div className="md:col-span-2">
          <label className="field" htmlFor="t-setup">setup_params (JSON)</label>
          <textarea
            id="t-setup"
            className="input font-mono text-xs"
            rows={6}
            value={setupParamsText}
            onChange={(e) => setSetupParamsText(e.target.value)}
          />
          {parseError && (
            <p className="error mt-1">JSON: {parseError}</p>
          )}
        </div>
      </div>
      <div className="mt-3 flex items-center gap-3">
        <button
          className="btn-primary"
          disabled={!name.trim() || !!parseError || create.isPending}
          onClick={() => create.mutate()}
        >
          {create.isPending ? 'Saving…' : 'Save template'}
        </button>
        {create.error && (
          <p className="error">{errorMessage(create.error)}</p>
        )}
      </div>
    </section>
  );
}
