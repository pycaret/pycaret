/**
 * Drift Dashboard — `/workspaces/:wsId/drift`.
 *
 * Workspace-scoped drift monitoring. Lists existing drift reports
 * (from `driftApi.list`) with filters by deployment / time. Each row
 * expands to show feature-level PSI / KL / KS scores.
 *
 * The DriftReport schema already exists; this screen stitches the
 * existing card components into a real dashboard with searchable
 * filters and trend mini-charts.
 */

import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';

import { deploymentsApi, driftApi } from '../api/endpoints';
import type { DriftReportRead } from '../api/types';

function severity(report: DriftReportRead): 'low' | 'medium' | 'high' {
  // Heuristic: count features with score > 0.25 (moderate drift threshold).
  const fd = report.feature_drift_json ?? {};
  const scores = Object.values(fd).map((v) => v.score ?? 0);
  const moderate = scores.filter((s) => s > 0.25).length;
  if (moderate >= 3) return 'high';
  if (moderate >= 1) return 'medium';
  return 'low';
}

function _features(report: DriftReportRead): Array<{ psi: number }> {
  const fd = report.feature_drift_json ?? {};
  return Object.values(fd).map((v) => ({ psi: v.score ?? 0 }));
}

function SeverityBadge({ s }: { s: 'low' | 'medium' | 'high' }) {
  const palette: Record<string, [string, string]> = {
    low: ['#22C55E', 'rgba(34,197,94,0.12)'],
    medium: ['#F59E0B', 'rgba(245,158,11,0.12)'],
    high: ['#EF4444', 'rgba(239,68,68,0.12)'],
  };
  const [fg, bg] = palette[s];
  return (
    <span
      style={{
        fontSize: 11,
        fontWeight: 600,
        textTransform: 'uppercase',
        padding: '2px 8px',
        borderRadius: 999,
        color: fg,
        background: bg,
      }}
    >
      {s}
    </span>
  );
}

export function DriftDashboard() {
  const { wsId = '' } = useParams<{ wsId: string }>();
  const [q, setQ] = useState('');
  const [filterSev, setFilterSev] = useState<'all' | 'low' | 'medium' | 'high'>('all');

  const deployments = useQuery({
    queryKey: ['deployments', wsId],
    queryFn: () => deploymentsApi.list(wsId),
    enabled: !!wsId,
  });

  const reports = useQuery({
    queryKey: ['drift-reports', wsId],
    queryFn: () => driftApi.list(wsId),
    enabled: !!wsId,
  });

  const deploymentName = useMemo(() => {
    const map: Record<string, string> = {};
    for (const d of deployments.data ?? []) map[d.id] = d.endpoint_slug ?? d.id.slice(0, 8);
    return map;
  }, [deployments.data]);

  const filtered = useMemo(() => {
    const all = reports.data ?? [];
    return all
      .filter((r) => (filterSev === 'all' ? true : severity(r) === filterSev))
      .filter((r) => {
        if (!q) return true;
        const slug = deploymentName[r.deployment_id ?? ''] ?? '';
        return (slug + r.id).toLowerCase().includes(q.toLowerCase());
      });
  }, [reports.data, filterSev, q, deploymentName]);

  const counts = useMemo(() => {
    const all = reports.data ?? [];
    return {
      total: all.length,
      high: all.filter((r) => severity(r) === 'high').length,
      medium: all.filter((r) => severity(r) === 'medium').length,
      low: all.filter((r) => severity(r) === 'low').length,
    };
  }, [reports.data]);

  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <nav style={{ fontSize: 12, color: '#94A3B8' }}>
          <Link to={`/workspaces/${wsId}/home`} style={{ color: 'inherit' }}>
            Workspace
          </Link>{' '}
          / Drift dashboard
        </nav>
        <h1 style={{ fontSize: 24, fontWeight: 700, color: '#0F172A', margin: 0 }}>
          Drift monitoring
        </h1>
        <p style={{ color: '#64748B', fontSize: 13, margin: 0 }}>
          Distribution shifts across deployed pipelines. Sorted by recency.
        </p>
      </header>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(4, minmax(140px, 1fr))',
          gap: 12,
        }}
      >
        {[
          { label: 'Total reports', value: counts.total, color: '#0F172A' },
          { label: 'High severity', value: counts.high, color: '#EF4444' },
          { label: 'Medium', value: counts.medium, color: '#F59E0B' },
          { label: 'Low / clean', value: counts.low, color: '#22C55E' },
        ].map((t) => (
          <div key={t.label} className="card" style={{ padding: 14 }}>
            <div style={{ fontSize: 11, textTransform: 'uppercase', color: '#64748B' }}>
              {t.label}
            </div>
            <div style={{ fontSize: 26, fontWeight: 700, color: t.color, marginTop: 4 }}>
              {t.value}
            </div>
          </div>
        ))}
      </div>

      <div className="card">
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 12, flexWrap: 'wrap' }}>
          <input
            placeholder="Filter by deployment / id…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            className="input"
            style={{ flex: 1, minWidth: 220 }}
          />
          <select
            value={filterSev}
            onChange={(e) => setFilterSev(e.target.value as typeof filterSev)}
            className="input"
          >
            <option value="all">All severities</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
        </div>

        {reports.isLoading ? (
          <div style={{ color: '#94A3B8', fontSize: 13 }}>Loading…</div>
        ) : filtered.length === 0 ? (
          <div style={{ color: '#94A3B8', fontSize: 13, padding: 24, textAlign: 'center' }}>
            No drift reports match the current filter.
          </div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ textAlign: 'left', color: '#64748B', fontSize: 11, textTransform: 'uppercase' }}>
                <th style={{ padding: '8px 12px' }}>Severity</th>
                <th style={{ padding: '8px 12px' }}>Deployment</th>
                <th style={{ padding: '8px 12px' }}>Date</th>
                <th style={{ padding: '8px 12px' }}>Drifted features</th>
                <th style={{ padding: '8px 12px' }}>Avg PSI</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((r) => {
                const features = _features(r);
                const avgPsi = features.length
                  ? features.reduce((a, b) => a + b.psi, 0) / features.length
                  : 0;
                const drifted = features.filter((f) => f.psi > 0.1).length;
                return (
                  <tr key={r.id} style={{ borderTop: '1px solid rgba(148,163,184,0.15)' }}>
                    <td style={{ padding: '10px 12px' }}>
                      <SeverityBadge s={severity(r)} />
                    </td>
                    <td style={{ padding: '10px 12px' }}>
                      {deploymentName[r.deployment_id ?? ''] ?? '—'}
                    </td>
                    <td style={{ padding: '10px 12px', color: '#64748B' }}>
                      {new Date(r.created_at).toLocaleString()}
                    </td>
                    <td style={{ padding: '10px 12px' }}>
                      {drifted} / {features.length}
                    </td>
                    <td style={{ padding: '10px 12px', fontVariantNumeric: 'tabular-nums' }}>
                      {avgPsi.toFixed(3)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
