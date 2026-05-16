/**
 * /workspaces/:wsId/monitoring — Phase 10 alert rules + metrics.
 *
 * Rules CRUD on the left, per-deployment metric time-series on the
 * right. The metric panel auto-refreshes every 10s and the rule list
 * shows which rules have fired in their last evaluation window.
 */

import { useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useParams } from 'react-router-dom';
import { BackButton } from '@/components/BackButton';
import { Dialog } from '@/components/Dialog';
import { PlotlyFigure } from '@/components/PlotlyFigure';
import { deploymentsApi, monitoringApi } from '@/api/endpoints';
import type {
  AlertComparator,
  AlertDestination,
  AlertRule,
} from '@/api/types';

const COMPARATORS: AlertComparator[] = ['gt', 'gte', 'lt', 'lte', 'eq'];
const DESTINATIONS: AlertDestination[] = ['slack', 'email', 'webhook'];

export function Monitoring() {
  const { wsId = '' } = useParams<{ wsId: string }>();
  const qc = useQueryClient();
  const [createOpen, setCreateOpen] = useState(false);
  const [selectedDeployment, setSelectedDeployment] = useState<string | null>(
    null,
  );

  const deployments = useQuery({
    queryKey: ['deployments', wsId],
    queryFn: () => deploymentsApi.list(wsId),
    enabled: !!wsId,
  });

  const rules = useQuery({
    queryKey: ['alert-rules', wsId],
    queryFn: () => monitoringApi.listRules(wsId),
    enabled: !!wsId,
    refetchInterval: 15_000,
  });

  return (
    <div className="space-y-6">
      <header className="flex items-baseline justify-between">
        <div>
          <BackButton />
          <h1 className="h-page mt-2">Monitoring</h1>
          <p className="muted small">
            Alert rules + deployment metric time-series. Phase 10.
          </p>
        </div>
        <button
          type="button"
          className="btn-primary"
          onClick={() => setCreateOpen(true)}
        >
          New rule
        </button>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <section>
          <h2 className="h-section mb-3">
            Alert rules ({rules.data?.length ?? 0})
          </h2>
          {rules.isLoading && (
            <div className="card text-sm text-ink-500">Loading…</div>
          )}
          {rules.data && rules.data.length === 0 && (
            <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 p-6 text-center">
              <p className="text-sm text-ink-500">
                No rules configured. Hit "New rule" to wire up a Slack /
                email / webhook destination for a metric breach.
              </p>
            </div>
          )}
          {rules.data && rules.data.length > 0 && (
            <ul className="space-y-2">
              {rules.data.map((r) => (
                <RuleRow
                  key={r.id}
                  rule={r}
                  onToggle={() => {
                    monitoringApi
                      .patchRule(r.id, { enabled: !r.enabled })
                      .finally(() =>
                        qc.invalidateQueries({
                          queryKey: ['alert-rules', wsId],
                        }),
                      );
                  }}
                  onDelete={() => {
                    monitoringApi.deleteRule(r.id).finally(() =>
                      qc.invalidateQueries({
                        queryKey: ['alert-rules', wsId],
                      }),
                    );
                  }}
                />
              ))}
            </ul>
          )}
        </section>

        <section>
          <h2 className="h-section mb-3">Deployment metrics</h2>
          <select
            className="input w-full"
            value={selectedDeployment ?? ''}
            onChange={(e) => setSelectedDeployment(e.target.value || null)}
          >
            <option value="">— pick a deployment —</option>
            {(deployments.data ?? []).map((d) => (
              <option key={d.id} value={d.id}>
                {d.endpoint_slug}
              </option>
            ))}
          </select>
          {selectedDeployment ? (
            <MetricChart deploymentId={selectedDeployment} />
          ) : (
            <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 p-6 text-center mt-4">
              <p className="text-sm text-ink-500">
                Pick a deployment to render its metric time-series.
              </p>
            </div>
          )}
        </section>
      </div>

      <NewRuleDialog
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        wsId={wsId}
        deployments={deployments.data ?? []}
        onSaved={() => {
          setCreateOpen(false);
          qc.invalidateQueries({ queryKey: ['alert-rules', wsId] });
        }}
      />
    </div>
  );
}

function RuleRow({
  rule,
  onToggle,
  onDelete,
}: {
  rule: AlertRule;
  onToggle: () => void;
  onDelete: () => void;
}) {
  return (
    <li
      className={`card flex items-start gap-3 ${
        rule.enabled ? '' : 'opacity-60'
      }`}
    >
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-medium">{rule.name}</span>
          {rule.last_status === 'fired' && (
            <span className="pill bg-rose-50 text-rose-700 dark:bg-rose-500/15 dark:text-rose-300">
              fired
            </span>
          )}
          {!rule.enabled && <span className="pill-neutral">disabled</span>}
        </div>
        <p className="text-xs text-ink-500 mt-1">
          <code>{rule.metric}</code> {rule.comparator}{' '}
          <strong>{rule.threshold}</strong> over {rule.window_seconds}s →{' '}
          {rule.destination_kind}
        </p>
        {rule.last_fired_at && (
          <p className="text-xs text-ink-500 mt-1">
            last fired {new Date(rule.last_fired_at).toLocaleString()}
          </p>
        )}
      </div>
      <div className="flex flex-col gap-1 shrink-0">
        <button type="button" className="btn-secondary text-xs" onClick={onToggle}>
          {rule.enabled ? 'Disable' : 'Enable'}
        </button>
        <button
          type="button"
          className="text-xs text-danger-600 hover:underline"
          onClick={onDelete}
        >
          Delete
        </button>
      </div>
    </li>
  );
}

function MetricChart({ deploymentId }: { deploymentId: string }) {
  const [metric, setMetric] = useState('p95_latency_ms');
  const series = useQuery({
    queryKey: ['metrics', deploymentId, metric],
    queryFn: () =>
      monitoringApi.metrics(deploymentId, {
        metric,
        since_seconds: 3600,
        limit: 500,
      }),
    refetchInterval: 10_000,
  });

  const figure = useMemo(() => {
    const points = series.data?.points ?? [];
    if (points.length === 0) return null;
    return {
      data: [
        {
          type: 'scatter' as const,
          mode: 'lines+markers' as const,
          x: points.map((p) => p.ts),
          y: points.map((p) => p.value),
          line: { color: '#3b82f6' },
          name: metric,
        },
      ],
      layout: {
        margin: { l: 50, r: 20, t: 20, b: 50 },
        yaxis: { title: metric },
        xaxis: { title: 'time' },
      },
    };
  }, [series.data, metric]);

  return (
    <div className="space-y-3 mt-4">
      <div className="flex gap-2 flex-wrap">
        {[
          'p95_latency_ms',
          'p50_latency_ms',
          'requests_per_minute',
          'error_rate',
          'drift_score',
        ].map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setMetric(m)}
            className={`pill ${
              metric === m
                ? 'bg-accent-500 text-white'
                : 'bg-ink-100 text-ink-700 dark:bg-ink-800 dark:text-ink-300'
            }`}
          >
            {m}
          </button>
        ))}
      </div>
      {series.isLoading && (
        <div className="card text-sm text-ink-500">Loading…</div>
      )}
      {figure ? (
        <div className="card p-2">
          <PlotlyFigure figure={figure} />
        </div>
      ) : (
        <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 p-6 text-center">
          <p className="text-sm text-ink-500">
            No data points for <code>{metric}</code> in the last hour.
          </p>
        </div>
      )}
    </div>
  );
}

function NewRuleDialog({
  open,
  onClose,
  wsId,
  deployments,
  onSaved,
}: {
  open: boolean;
  onClose: () => void;
  wsId: string;
  deployments: { id: string; endpoint_slug: string }[];
  onSaved: () => void;
}) {
  const [name, setName] = useState('');
  const [metric, setMetric] = useState('p95_latency_ms');
  const [comparator, setComparator] = useState<AlertComparator>('gt');
  const [threshold, setThreshold] = useState('500');
  const [windowSec, setWindowSec] = useState('300');
  const [deploymentId, setDeploymentId] = useState('');
  const [destinationKind, setDestinationKind] =
    useState<AlertDestination>('slack');
  const [destinationConfig, setDestinationConfig] = useState(
    '{"webhook_url":"https://hooks.slack.com/services/…"}',
  );

  const save = useMutation({
    mutationFn: () =>
      monitoringApi.createRule(wsId, {
        name,
        metric,
        comparator,
        threshold: Number(threshold),
        window_seconds: Number(windowSec),
        deployment_id: deploymentId || null,
        destination_kind: destinationKind,
        destination_config: JSON.parse(destinationConfig || '{}'),
      }),
    onSuccess: onSaved,
  });

  return (
    <Dialog open={open} onClose={onClose} title="New alert rule" size="md">
      <div className="space-y-3">
        <label className="field">
          <span className="muted small">Name</span>
          <input
            className="input w-full"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="p95 latency over 500ms"
          />
        </label>
        <div className="grid grid-cols-3 gap-2">
          <label className="field">
            <span className="muted small">Metric</span>
            <input
              className="input w-full"
              value={metric}
              onChange={(e) => setMetric(e.target.value)}
            />
          </label>
          <label className="field">
            <span className="muted small">Comparator</span>
            <select
              className="input w-full"
              value={comparator}
              onChange={(e) =>
                setComparator(e.target.value as AlertComparator)
              }
            >
              {COMPARATORS.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span className="muted small">Threshold</span>
            <input
              className="input w-full"
              value={threshold}
              onChange={(e) => setThreshold(e.target.value)}
              type="number"
            />
          </label>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <label className="field">
            <span className="muted small">Window (sec)</span>
            <input
              className="input w-full"
              value={windowSec}
              onChange={(e) => setWindowSec(e.target.value)}
              type="number"
            />
          </label>
          <label className="field">
            <span className="muted small">Deployment (optional)</span>
            <select
              className="input w-full"
              value={deploymentId}
              onChange={(e) => setDeploymentId(e.target.value)}
            >
              <option value="">all deployments</option>
              {deployments.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.endpoint_slug}
                </option>
              ))}
            </select>
          </label>
        </div>
        <label className="field">
          <span className="muted small">Destination</span>
          <select
            className="input w-full"
            value={destinationKind}
            onChange={(e) =>
              setDestinationKind(e.target.value as AlertDestination)
            }
          >
            {DESTINATIONS.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span className="muted small">Destination config (JSON)</span>
          <textarea
            className="input w-full font-mono text-xs"
            rows={3}
            value={destinationConfig}
            onChange={(e) => setDestinationConfig(e.target.value)}
          />
        </label>
        <div className="flex justify-end gap-2">
          <button type="button" className="btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="btn-primary"
            disabled={!name.trim() || save.isPending}
            onClick={() => save.mutate()}
          >
            {save.isPending ? 'Saving…' : 'Save'}
          </button>
        </div>
      </div>
    </Dialog>
  );
}
