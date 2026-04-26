/**
 * Data Profile / EDA screen — `/workspaces/:wsId/datasets/:dataSourceId/profile`.
 *
 * Renders an interactive data profile for a CSV / DataSource. Loads
 * Plotly figures from `/api/v1/datasets/:id/plots/eda/:kind`. Layout:
 *  - Header: dataset name, row/col count, top tags.
 *  - profile_summary table (full-width).
 *  - missingness_map + correlation_heatmap side-by-side.
 *  - column_distribution: column picker + chart.
 *  - target_vs_feature: feature + target picker + chart.
 */

import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';

import { dataSourcesApi, plotsApi } from '../api/endpoints';
import { PlotlyFigure } from '../components/PlotlyFigure';
import type { PlotEnvelope } from '../api/types';

function useEdaPlot(dataSourceId: string, kind: string, params?: { column?: string; feature?: string; target?: string }) {
  return useQuery<PlotEnvelope, Error>({
    queryKey: ['datasets', dataSourceId, 'plots', kind, params ?? {}],
    queryFn: () => plotsApi.forDataset(dataSourceId, kind, params),
    enabled:
      !!dataSourceId &&
      // Don't auto-fire param-required charts until the user picks values.
      (kind !== 'column_distribution' || !!params?.column) &&
      (kind !== 'target_vs_feature' || (!!params?.feature && !!params?.target)),
    staleTime: 60_000,
  });
}

export function DataProfile() {
  const { dataSourceId = '' } = useParams<{ dataSourceId: string }>();

  const ds = useQuery({
    queryKey: ['datasets', dataSourceId],
    queryFn: () => dataSourcesApi.get(dataSourceId),
    enabled: !!dataSourceId,
  });

  const cfg = (ds.data?.config ?? {}) as Record<string, unknown>;
  const columns = useMemo<string[]>(
    () => (Array.isArray(cfg.columns) ? (cfg.columns as string[]) : []),
    [cfg.columns],
  );

  const [pickedColumn, setPickedColumn] = useState<string>('');
  const [pickedFeature, setPickedFeature] = useState<string>('');
  const [pickedTarget, setPickedTarget] = useState<string>('');

  // Default the column/feature pickers to the first available column.
  const defaultColumn = pickedColumn || columns[0] || '';
  const defaultFeature = pickedFeature || columns[0] || '';
  const defaultTarget = pickedTarget || columns[1] || '';

  const summary = useEdaPlot(dataSourceId, 'profile_summary');
  const missingness = useEdaPlot(dataSourceId, 'missingness_map');
  const correlation = useEdaPlot(dataSourceId, 'correlation_heatmap');
  const colDist = useEdaPlot(dataSourceId, 'column_distribution', { column: defaultColumn });
  const targetVsFeat = useEdaPlot(dataSourceId, 'target_vs_feature', {
    feature: defaultFeature,
    target: defaultTarget,
  });

  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <nav style={{ fontSize: 12, color: '#94A3B8' }}>
          <Link to="/" style={{ color: 'inherit' }}>Workspaces</Link> / Data profile
        </nav>
        <h1 style={{ fontSize: 24, fontWeight: 700, color: '#0F172A', margin: 0 }}>
          {ds.data?.name ?? 'Data profile'}
        </h1>
        <p style={{ color: '#64748B', fontSize: 13, margin: 0 }}>
          {typeof cfg.rows === 'number' ? `${(cfg.rows as number).toLocaleString()} rows` : null}
          {typeof cfg.rows === 'number' && columns.length > 0 ? ' • ' : null}
          {columns.length > 0 ? `${columns.length} columns` : null}
        </p>
      </header>

      <PlotlyFigure
        figure={summary.data?.figure}
        loading={summary.isLoading}
        error={summary.error ?? undefined}
        onRetry={() => summary.refetch()}
        title="Column profile"
        height={Math.max(280, 30 * (columns.length + 2) + 40)}
      />

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))',
          gap: 16,
        }}
      >
        <PlotlyFigure
          figure={missingness.data?.figure}
          loading={missingness.isLoading}
          error={missingness.error ?? undefined}
          onRetry={() => missingness.refetch()}
          title="Missingness"
          caption="Per-column missing-rate; flat bars mean clean data."
        />
        <PlotlyFigure
          figure={correlation.data?.figure}
          loading={correlation.isLoading}
          error={correlation.error ?? undefined}
          onRetry={() => correlation.refetch()}
          title="Correlation matrix"
          caption="Pearson correlation between numeric columns."
        />
      </div>

      <div className="card space-y-3">
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
          <label style={{ fontSize: 13, fontWeight: 600, color: '#0F172A' }}>
            Column distribution
          </label>
          <select
            value={defaultColumn}
            onChange={(e) => setPickedColumn(e.target.value)}
            className="input"
            style={{ minWidth: 220 }}
          >
            {columns.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>
        <PlotlyFigure
          figure={colDist.data?.figure}
          loading={colDist.isLoading}
          error={colDist.error ?? undefined}
          onRetry={() => colDist.refetch()}
        />
      </div>

      <div className="card space-y-3">
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
          <label style={{ fontSize: 13, fontWeight: 600, color: '#0F172A' }}>
            Feature × target
          </label>
          <select
            value={defaultFeature}
            onChange={(e) => setPickedFeature(e.target.value)}
            className="input"
            style={{ minWidth: 200 }}
          >
            {columns.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          <span style={{ color: '#94A3B8' }}>vs</span>
          <select
            value={defaultTarget}
            onChange={(e) => setPickedTarget(e.target.value)}
            className="input"
            style={{ minWidth: 200 }}
          >
            {columns.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>
        <PlotlyFigure
          figure={targetVsFeat.data?.figure}
          loading={targetVsFeat.isLoading}
          error={targetVsFeat.error ?? undefined}
          onRetry={() => targetVsFeat.refetch()}
        />
      </div>
    </div>
  );
}
