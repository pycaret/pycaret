/**
 * /workspaces/:wsId/schedules — manage drift-monitor + retrain schedules.
 *
 * v1 surface:
 *   - List existing schedules with last-run state.
 *   - "New schedule" form: pick kind (drift_monitor / retrain), pick target
 *     (deployment / experiment), pick interval seconds.
 *   - Per-row toggle / delete / run-now.
 */

import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useParams } from 'react-router-dom';
import {
  deploymentsApi,
  experimentsApi,
  projectsApi,
  schedulesApi,
  type Schedule,
} from '@/api/endpoints';
import { errorMessage } from '@/api/client';

export function Schedules() {
  const { wsId } = useParams<{ wsId: string }>();
  const workspaceId = wsId ?? '';
  const qc = useQueryClient();

  const list = useQuery({
    queryKey: ['workspaces', workspaceId, 'schedules'],
    queryFn: () => schedulesApi.list(workspaceId),
    enabled: !!workspaceId,
  });

  const patch = useMutation({
    mutationFn: (vars: { id: string; enabled: boolean }) =>
      schedulesApi.patch(vars.id, { enabled: vars.enabled }),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ['workspaces', workspaceId, 'schedules'] }),
  });

  const remove = useMutation({
    mutationFn: (id: string) => schedulesApi.remove(id),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ['workspaces', workspaceId, 'schedules'] }),
  });

  const runNow = useMutation({
    mutationFn: (id: string) => schedulesApi.runNow(id),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ['workspaces', workspaceId, 'schedules'] }),
  });

  return (
    <div className="space-y-8">
      <header>
        <h1 className="h-page">Schedules</h1>
        <p className="mt-2 text-sm text-ink-500">
          Drift monitors and retraining cron jobs. Run in-process via APScheduler.
        </p>
      </header>

      <NewScheduleForm workspaceId={workspaceId} />

      <section>
        <h2 className="h-section mb-3">Active schedules</h2>
        {list.isPending ? (
          <div className="card text-sm text-ink-500">Loading…</div>
        ) : list.error ? (
          <div className="card text-sm text-danger-600">{errorMessage(list.error)}</div>
        ) : !list.data || list.data.items.length === 0 ? (
          <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 p-8 text-center text-sm text-ink-500">
            No schedules yet — create one above to start monitoring drift or
            retraining a model on a cadence.
          </div>
        ) : (
          <div className="card overflow-hidden p-0">
            <table className="w-full text-sm">
              <thead className="bg-white text-ink-500 dark:bg-ink-900">
                <tr>
                  <th className="px-4 py-2 text-left font-medium">Kind</th>
                  <th className="px-4 py-2 text-left font-medium">Target</th>
                  <th className="px-4 py-2 text-left font-medium">Schedule</th>
                  <th className="px-4 py-2 text-left font-medium">Last run</th>
                  <th className="px-4 py-2 text-center font-medium">Enabled</th>
                  <th className="px-4 py-2 text-right font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {list.data.items.map((s: Schedule) => (
                  <tr
                    key={s.id}
                    className="border-t border-ink-200 dark:border-ink-800 hover:bg-ink-50 dark:hover:bg-ink-950/40"
                  >
                    <td className="px-4 py-2">
                      <span className="pill-neutral font-mono">{s.kind}</span>
                    </td>
                    <td className="px-4 py-2 font-mono text-xs text-ink-500">
                      {s.target_id ? `${s.target_id.slice(0, 8)}…` : '—'}
                    </td>
                    <td className="px-4 py-2 text-ink-700 dark:text-ink-300">
                      {s.schedule.interval_seconds
                        ? `every ${s.schedule.interval_seconds}s`
                        : s.schedule.cron
                          ? `cron: ${s.schedule.cron}`
                          : '—'}
                    </td>
                    <td className="px-4 py-2 text-xs">
                      {s.last_run_at ? (
                        <div className="flex items-center gap-2">
                          {s.last_status === 'ok' ? (
                            <span className="pill-success">ok</span>
                          ) : s.last_status === 'error' ? (
                            <span className="pill-danger">error</span>
                          ) : (
                            <span className="pill-neutral">{s.last_status ?? '?'}</span>
                          )}
                          <span className="text-ink-500">
                            {new Date(s.last_run_at).toLocaleString()}
                          </span>
                        </div>
                      ) : (
                        <span className="text-ink-400">never</span>
                      )}
                    </td>
                    <td className="px-4 py-2 text-center">
                      <input
                        type="checkbox"
                        checked={s.enabled}
                        disabled={patch.isPending}
                        onChange={(e) =>
                          patch.mutate({ id: s.id, enabled: e.target.checked })
                        }
                      />
                    </td>
                    <td className="px-4 py-2 text-right text-xs">
                      <button
                        className="text-accent-600 hover:underline"
                        disabled={runNow.isPending}
                        onClick={() => runNow.mutate(s.id)}
                      >
                        Run now
                      </button>
                      <span className="text-ink-300 mx-1">·</span>
                      <button
                        className="text-danger-600 hover:underline"
                        onClick={() => {
                          if (confirm(`Delete schedule ${s.id}?`)) {
                            remove.mutate(s.id);
                          }
                        }}
                        disabled={remove.isPending}
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

function NewScheduleForm({ workspaceId }: { workspaceId: string }) {
  const qc = useQueryClient();
  const [kind, setKind] = useState<'drift_monitor' | 'retrain'>('drift_monitor');
  const [targetId, setTargetId] = useState<string>('');
  const [interval, setInterval] = useState<number>(3600);

  const deployments = useQuery({
    queryKey: ['workspaces', workspaceId, 'deployments'],
    queryFn: () => deploymentsApi.list(workspaceId),
    enabled: !!workspaceId && kind === 'drift_monitor',
  });

  const projects = useQuery({
    queryKey: ['workspaces', workspaceId, 'projects'],
    queryFn: () => projectsApi.list(workspaceId),
    enabled: !!workspaceId && kind === 'retrain',
  });

  const allExperiments = useQuery({
    queryKey: ['workspaces', workspaceId, 'experiments-flat'],
    queryFn: async () => {
      const ps = await projectsApi.list(workspaceId);
      const result: Array<{
        id: string;
        name: string;
        project_name: string;
      }> = [];
      for (const p of ps) {
        const exps = await experimentsApi.list(p.id);
        for (const e of exps) {
          result.push({ id: e.id, name: e.name, project_name: p.name });
        }
      }
      return result;
    },
    enabled: !!workspaceId && kind === 'retrain' && !!projects.data,
  });

  const create = useMutation({
    mutationFn: () =>
      schedulesApi.create(workspaceId, {
        kind,
        target_id: targetId,
        schedule: { interval_seconds: interval },
        spec:
          kind === 'retrain'
            ? { plan: 'compare', sklearn_dataset: 'iris' }
            : null,
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['workspaces', workspaceId, 'schedules'] });
      setTargetId('');
    },
  });

  const targetOptions =
    kind === 'drift_monitor'
      ? (deployments.data ?? []).map((d) => ({
          id: d.id,
          label: `${d.endpoint_slug} (${d.status})`,
        }))
      : (allExperiments.data ?? []).map((e) => ({
          id: e.id,
          label: `${e.project_name} / ${e.name}`,
        }));

  return (
    <section className="card">
      <h2 className="text-sm font-medium text-ink-900 mb-3">New schedule</h2>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 items-end">
        <div>
          <label className="field" htmlFor="kind">Kind</label>
          <select
            id="kind"
            className="input"
            value={kind}
            onChange={(e) => {
              setKind(e.target.value as 'drift_monitor' | 'retrain');
              setTargetId('');
            }}
          >
            <option value="drift_monitor">Drift monitor</option>
            <option value="retrain">Retrain</option>
          </select>
        </div>
        <div className="md:col-span-2">
          <label className="field" htmlFor="target">
            {kind === 'drift_monitor' ? 'Deployment' : 'Experiment'}
          </label>
          <select
            id="target"
            className="input"
            value={targetId}
            onChange={(e) => setTargetId(e.target.value)}
          >
            <option value="">—</option>
            {targetOptions.map((o) => (
              <option key={o.id} value={o.id}>
                {o.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="field" htmlFor="interval">Interval (s)</label>
          <input
            id="interval"
            className="input"
            type="number"
            min={30}
            value={interval}
            onChange={(e) => setInterval(Number(e.target.value) || 30)}
          />
        </div>
      </div>
      <div className="mt-3 flex items-center gap-3">
        <button
          className="btn-primary"
          disabled={!targetId || interval < 30 || create.isPending}
          onClick={() => create.mutate()}
        >
          {create.isPending ? 'Creating…' : 'Create schedule'}
        </button>
        {create.error && (
          <p className="error">{errorMessage(create.error)}</p>
        )}
      </div>
    </section>
  );
}
